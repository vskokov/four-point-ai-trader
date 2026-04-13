"""
TradingEngine — top-level orchestration loop.

Wires together every module built in Phases 1–5 and drives the real-time
trading lifecycle:

  stream_bars WebSocket  →  bar_handler  →  HMM / OU / LLM / MWU  →  Executor
  APScheduler            →  sentiment_job_early (every 25 min, 07:00–10:29 ET, mon-fri)
                         →  sentiment_job_late  (every 35 min, 10:30–16:30 ET, mon-fri)
                         →  market_open_job     (09:31 ET, mon-fri)
                         →  eod_job             (16:05 ET, mon-fri)
"""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.executors.pool import ThreadPoolExecutor as APThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import text

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import trading_engine.config.settings as settings
from trading_engine.data.alphavantage_client import AlphaVantageNewsClient
from trading_engine.data.alpaca_client import AlpacaMarketData, AlpacaNewsClient
from trading_engine.data.fundamentals_client import FundamentalsClient
from trading_engine.data.storage import Storage
from trading_engine.execution.executor import OrderExecutor, RiskManager
from trading_engine.meta_agent.mwu_agent import MWUMetaAgent
from trading_engine.portfolio.portfolio_optimizer import PortfolioOptimizer
from trading_engine.orchestrator.state_manager import StateManager
from trading_engine.signals.hmm_regime import HMMRegimeDetector
from trading_engine.signals.llm_sentiment import LLMSentimentSignal
from trading_engine.signals.mean_reversion import OUSpreadSignal
from trading_engine.utils.logging import get_logger, regime_banner

logger = get_logger(__name__)

_ET = ZoneInfo("America/New_York")

# AV NEWS_SENTIMENT free tier accepts at most 50 tickers per request.
# We reserve the top N by market cap for AV (richer per-ticker sentiment);
# the remainder fall back to the Alpaca News API.
_AV_MAX_TICKERS = 30

# Regime smoothing: number of consecutive identical HMM labels required before
# the engine accepts a regime change.  Prevents single-bar HMM chatter.
_REGIME_SMOOTH_WINDOW = 3

# Minimum holding period: suppress direction reversals within this many minutes
# of the last signal change.  Matches the Alpaca free-tier 15-minute data delay
# — decisions cannot be more precise than the data latency.
_MIN_HOLD_MINUTES = 15


def _to_float(value: Any) -> float | None:
    """Cast numpy scalars (or None) to plain Python float for psycopg2 binding."""
    if value is None:
        return None
    return float(value)


# Default path for discovered pairs written by pair_scanner.py
_DISCOVERED_PAIRS_PATH = Path(__file__).parent.parent / "config" / "discovered_pairs.json"


def _load_discovered_pairs(path: Path | None = None) -> list[tuple[str, str]]:
    """
    Load cointegrated pairs from a JSON file written by ``pair_scanner.py``.

    Parameters
    ----------
    path:
        Path to ``discovered_pairs.json``.  Defaults to
        ``trading_engine/config/discovered_pairs.json``.

    Returns
    -------
    list of ``(ticker1, ticker2)`` tuples, or ``[]`` if the file is missing
    or malformed.

    Warnings
    --------
    - If the file does not exist, logs a warning and returns ``[]``.
    - If ``scanned_at`` is more than 14 days old, logs a staleness warning
      but still returns the pairs.
    """
    if path is None:
        path = _DISCOVERED_PAIRS_PATH

    if not path.exists():
        logger.warning("engine.pairs_file_not_found", path=str(path))
        return []

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "engine.pairs_file_malformed", path=str(path), error=str(exc)
        )
        return []

    # Staleness check
    scanned_at = data.get("scanned_at")
    if scanned_at:
        try:
            scanned_dt = datetime.fromisoformat(
                scanned_at.replace("Z", "+00:00")
            )
            age_days = (datetime.now(tz=timezone.utc) - scanned_dt).days
            if age_days > 14:
                logger.warning(
                    "engine.pairs_file_stale",
                    path=str(path),
                    age_days=age_days,
                )
        except (ValueError, TypeError):
            pass

    raw_pairs = data.get("pairs", [])
    result: list[tuple[str, str]] = []
    for entry in raw_pairs:
        try:
            result.append((entry["ticker1"], entry["ticker2"]))
        except (KeyError, TypeError):
            continue

    logger.info(
        "engine.pairs_loaded", path=str(path), n_pairs=len(result)
    )
    return result

# Regime label → direction signal (+1 bull, 0 neutral, -1 bear)
_REGIME_TO_SIGNAL: dict[str, int] = {"bull": 1, "neutral": 0, "bear": -1}

# Conservative signal-stats defaults used until real P&L history accumulates
_DEFAULT_SIGNAL_STATS: dict[str, float] = {
    "win_rate": 0.52,
    "avg_win":  0.015,
    "avg_loss": 0.010,
}

# Kelly stats computed from Alpaca fill P&L
_KELLY_LOOKBACK_DAYS  = 90   # days of fill history to consider
_KELLY_MIN_ROUNDTRIPS = 5    # minimum completed BUY→SELL pairs before trusting stats


class TradingEngine:
    """
    Autonomous trading engine orchestrator.

    Parameters
    ----------
    tickers:
        Universe of equity symbols, e.g. ``["AAPL", "MSFT", "JPM", "BAC"]``.
        Pair tickers discovered in *pairs_file* are automatically merged in.
    paper:
        If *True* (default) connects to Alpaca paper-trading.
    models_dir:
        Root directory for persisted model artefacts.  Defaults to
        ``trading_engine/models/``.  Pass ``tmp_path`` in tests.
    pairs_file:
        Path to ``discovered_pairs.json`` written by ``pair_scanner.py``.
        Defaults to ``trading_engine/config/discovered_pairs.json``.
        Pass ``None`` to use the default location.
    """

    def __init__(
        self,
        tickers: list[str],
        paper: bool = True,
        models_dir: Path | str | None = None,
        pairs_file: Path | str | None = None,
    ) -> None:
        self._tickers = list(tickers)
        self._pairs = _load_discovered_pairs(
            Path(pairs_file) if pairs_file is not None else None
        )

        # Auto-merge pair tickers into self._tickers so that both legs of every
        # pair receive WebSocket subscriptions, HMM detectors, MWU agents, and
        # portfolio weights.
        pair_tickers = {t for p in self._pairs for t in p}
        new_tickers = sorted(pair_tickers - set(self._tickers))
        if new_tickers:
            logger.info("engine.pair_tickers_added", tickers=new_tickers)
            self._tickers.extend(new_tickers)

        self._paper = paper
        self._models_dir = (
            Path(models_dir) if models_dir is not None
            else Path(__file__).parent.parent / "models"
        )

        logger.info(
            "engine.init",
            tickers=self._tickers,
            pairs=self._pairs,
            paper=self._paper,
        )

        # ------------------------------------------------------------------
        # Core infrastructure
        # ------------------------------------------------------------------
        self._storage = Storage(settings.DB_URL)
        self._alpaca = AlpacaMarketData(self._storage)
        self._av_client = AlphaVantageNewsClient()
        self._alpaca_news = AlpacaNewsClient()
        self._fundamentals = FundamentalsClient()

        # ------------------------------------------------------------------
        # Signal modules  (one HMM + one MWU per ticker)
        # ------------------------------------------------------------------
        self._hmm: dict[str, HMMRegimeDetector] = {
            t: HMMRegimeDetector(models_dir=self._models_dir)
            for t in self._tickers
        }
        self._ou_signals: dict[tuple, OUSpreadSignal] = {
            p: OUSpreadSignal(p[0], p[1], models_dir=self._models_dir)
            for p in self._pairs
        }
        self._llm = LLMSentimentSignal()
        self._mwu: dict[str, MWUMetaAgent] = {
            t: MWUMetaAgent(ticker=t, models_dir=self._models_dir)
            for t in self._tickers
        }

        # ------------------------------------------------------------------
        # Portfolio optimizer  (uses merged self._tickers)
        # ------------------------------------------------------------------
        self.portfolio_optimizer = PortfolioOptimizer(
            tickers=self._tickers,
            max_weight=0.10,
            min_weight=0.0,
        )

        # ------------------------------------------------------------------
        # Execution layer
        # ------------------------------------------------------------------
        self._risk = RiskManager()
        self._executor = OrderExecutor(
            self._alpaca,
            self._risk,
            paper=paper,
            portfolio_optimizer=self.portfolio_optimizer,
        )

        # ------------------------------------------------------------------
        # Engine state
        # ------------------------------------------------------------------
        self._state_manager = StateManager(state_dir=self._models_dir)
        # Per-ticker Kelly stats, updated by eod_job
        self._signal_stats: dict[str, dict[str, float]] = {
            t: dict(_DEFAULT_SIGNAL_STATS) for t in self._tickers
        }
        self._shutdown_event = threading.Event()
        self._emergency_close = False
        self._scheduler: BackgroundScheduler | None = None

        # Regime smoothing: one deque per ticker accumulates recent raw HMM labels.
        # The stable label only changes when _REGIME_SMOOTH_WINDOW bars all agree.
        self._regime_history: dict[str, deque] = {
            t: deque(maxlen=_REGIME_SMOOTH_WINDOW) for t in self._tickers
        }
        self._stable_regime_label: dict[str, str] = {
            t: "neutral" for t in self._tickers
        }
        # Smoothed regime *index* — must stay in sync with _stable_regime_label
        # so the regime row used by MWU matches the displayed/smoothed label.
        self._stable_regime: dict[str, int] = {
            t: 1 for t in self._tickers  # 1 = neutral default
        }

        # Holding period: track the last non-zero signal and when it changed
        # direction so reversals within _MIN_HOLD_MINUTES can be suppressed.
        self._last_active_signal: dict[str, int] = {t: 0 for t in self._tickers}
        self._last_signal_change_time: dict[str, datetime | None] = {
            t: None for t in self._tickers
        }

        # Load any persisted metadata
        self._load_state()

        # Load persisted model artefacts (HMM, Kalman inside OU, MWU weights)
        self._load_models()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Restore engine metadata from the last saved snapshot (if any)."""
        state = self._state_manager.load()

        if state is None:
            return

        self._signal_stats.update(state.get("signal_stats", {}))

        # Sanity-reset any poisoned win_rate values (e.g. from MWU mis-wiring)
        for ticker, stats in self._signal_stats.items():
            if stats.get("win_rate", 0.0) < 0.30:
                logger.warning(
                    "engine.state_restore.kelly_reset",
                    ticker=ticker,
                    poisoned_win_rate=stats.get("win_rate"),
                    hint="win_rate < 0.30 is implausibly low; reverting to defaults",
                )
                self._signal_stats[ticker] = dict(_DEFAULT_SIGNAL_STATS)

        logger.info("engine.state_restored", keys=list(state.keys()))

    def _save_state(self) -> None:
        """Persist current engine metadata."""
        self._state_manager.save(
            {
                "tickers":      self._tickers,
                "pairs":        [list(p) for p in self._pairs],
                "paper":        self._paper,
                "signal_stats": self._signal_stats,
                "hmm_fitted":   {t: self._hmm[t].is_fitted for t in self._tickers},
            }
        )

    def _compute_kelly_stats_from_fills(self) -> dict[str, dict[str, float]]:
        """
        Compute Kelly sizing stats from Alpaca confirmed fill P&L.

        Fetches all CLOSED orders from the last _KELLY_LOOKBACK_DAYS days, pairs
        BUY and SELL fills per ticker using FIFO, and computes win_rate, avg_win,
        avg_loss from the resulting round-trips.

        Returns
        -------
        dict
            ``{ticker: {"win_rate": float, "avg_win": float, "avg_loss": float}}``
            for tickers with at least _KELLY_MIN_ROUNDTRIPS completed round-trips.
            Tickers below the threshold are omitted; the caller should fall back
            to ``_DEFAULT_SIGNAL_STATS``.
        """
        from datetime import timedelta
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        cutoff = datetime.now(timezone.utc) - timedelta(days=_KELLY_LOOKBACK_DAYS)
        try:
            orders = self._executor._trading.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    after=cutoff,
                    limit=500,
                )
            )
        except Exception as exc:
            logger.warning("engine.kelly_stats.fetch_failed", error=str(exc))
            return {}

        # Collect filled orders per ticker
        fills: dict[str, list[dict[str, Any]]] = {}
        for o in orders:
            if o.filled_qty is None or float(o.filled_qty) == 0:
                continue
            if o.filled_avg_price is None:
                continue
            fills.setdefault(o.symbol, []).append(
                {
                    "side":  o.side,
                    "qty":   float(o.filled_qty),
                    "price": float(o.filled_avg_price),
                    "time":  o.filled_at,
                }
            )

        result: dict[str, dict[str, float]] = {}
        for ticker, ticker_fills in fills.items():
            if ticker not in self._tickers:
                continue

            ticker_fills.sort(key=lambda x: x["time"])

            # FIFO pairing: each SELL consumes the oldest unmatched BUY
            buy_queue: list[dict[str, Any]] = []
            wins:   list[float] = []
            losses: list[float] = []

            for fill in ticker_fills:
                if fill["side"] == OrderSide.BUY:
                    buy_queue.append(fill)
                elif fill["side"] == OrderSide.SELL and buy_queue:
                    buy = buy_queue.pop(0)
                    pct = (fill["price"] - buy["price"]) / buy["price"]
                    if pct > 0:
                        wins.append(pct)
                    else:
                        losses.append(abs(pct))

            n = len(wins) + len(losses)
            if n < _KELLY_MIN_ROUNDTRIPS:
                continue  # not enough data; caller uses defaults

            win_rate = len(wins) / n
            avg_win  = float(sum(wins)   / len(wins))   if wins   else 0.005
            avg_loss = float(sum(losses) / len(losses)) if losses else 0.005

            result[ticker] = {
                "win_rate": win_rate,
                "avg_win":  max(avg_win,  0.001),
                "avg_loss": max(avg_loss, 0.001),
            }

        return result

    def _update_kelly_stats(self) -> None:
        """
        Recompute Kelly stats from Alpaca fill P&L and update ``_signal_stats``
        in-place.  Tickers with insufficient round-trips keep their current stats
        (or ``_DEFAULT_SIGNAL_STATS`` if never updated).
        """
        computed = self._compute_kelly_stats_from_fills()
        for ticker in self._tickers:
            if ticker in computed:
                self._signal_stats[ticker] = computed[ticker]
                logger.info(
                    "engine.kelly_stats.updated",
                    ticker=ticker,
                    **computed[ticker],
                )
            else:
                logger.debug(
                    "engine.kelly_stats.insufficient_data",
                    ticker=ticker,
                    hint=f"need {_KELLY_MIN_ROUNDTRIPS} round-trips, keeping current stats",
                )

    def _load_models(self) -> None:
        """Attempt to load persisted HMM models for each ticker."""
        for ticker in self._tickers:
            try:
                self._hmm[ticker].load(ticker)
                logger.info("engine.hmm_loaded", ticker=ticker)
            except FileNotFoundError:
                logger.warning(
                    "engine.hmm_not_found",
                    ticker=ticker,
                    hint="run startup_checks() to fit",
                )

    # ------------------------------------------------------------------
    # Startup checks
    # ------------------------------------------------------------------

    def startup_checks(self) -> None:
        """
        Verify all external services are accessible and models are fitted.

        1. Alpaca paper account reachable.
        2. Ollama / Gemma running.
        3. TimescaleDB reachable.
        4. HMM models — if any unfitted, attempt to fit on last 252 days.

        Raises
        ------
        RuntimeError
            If any critical dependency is unavailable.
        """
        logger.info("engine.startup_checks.start")

        # 1. Alpaca
        try:
            acct = self._alpaca.get_account_info()
            logger.info("engine.startup_checks.alpaca_ok", equity=acct["equity"])
        except Exception as exc:
            raise RuntimeError(f"Alpaca unreachable: {exc}") from exc

        # 2. Ollama
        try:
            import ollama
            client = ollama.Client(host=settings.OLLAMA_HOST, timeout=10)
            models = client.list()
            tags = [m.model for m in models.models]
            if not any(settings.OLLAMA_MODEL in tag for tag in tags):
                raise RuntimeError(
                    f"Ollama model {settings.OLLAMA_MODEL!r} not found. "
                    f"Available: {tags}"
                )
            logger.info("engine.startup_checks.ollama_ok", model=settings.OLLAMA_MODEL)
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc

        # 3. TimescaleDB
        try:
            with self._storage._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("engine.startup_checks.db_ok")
        except Exception as exc:
            raise RuntimeError(f"TimescaleDB unreachable: {exc}") from exc

        # 4. HMM models
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=int(252 * 1.4))   # generous fetch window

        # Minimum raw rows needed: 20-bar burn-in + 30 HMM minimum + 10 buffer
        _MIN_OHLCV_ROWS = 60

        for ticker in self._tickers:
            if not self._hmm[ticker].is_fitted:
                logger.info("engine.startup_checks.hmm_fit", ticker=ticker)

                # Seed DB with Alpaca history if not enough rows exist
                try:
                    existing = self._storage.query_ohlcv(ticker, start, end)
                    if len(existing) < _MIN_OHLCV_ROWS:
                        logger.info(
                            "engine.startup_checks.seeding_history",
                            ticker=ticker,
                            rows_in_db=len(existing),
                            needed=_MIN_OHLCV_ROWS,
                        )
                        seeded = self._alpaca.fetch_historical_ohlcv(
                            [ticker], start, end, timeframe="1Day"
                        )
                        if seeded.empty:
                            logger.warning(
                                "engine.startup_checks.seed_no_data",
                                ticker=ticker,
                                hint="ticker may not be available on IEX feed",
                            )
                        else:
                            logger.info(
                                "engine.startup_checks.seeded",
                                ticker=ticker,
                                n_bars=len(seeded),
                            )
                except Exception as exc:
                    logger.warning(
                        "engine.startup_checks.seed_failed",
                        ticker=ticker,
                        error=str(exc),
                    )

                try:
                    self._hmm[ticker].fit(ticker, start, end, self._storage)
                    logger.info("engine.startup_checks.hmm_fitted", ticker=ticker)
                except Exception as exc:
                    logger.warning(
                        "engine.startup_checks.hmm_fit_failed",
                        ticker=ticker,
                        error=str(exc),
                    )

        # Seed portfolio weights synchronously so get_target_weight() never
        # returns 0 + warning before market_open_job fires for the first time.
        # min-variance requires only OHLCV history (already seeded above) and
        # no return views, so it is safe to call here.
        logger.info("engine.startup_checks.portfolio_init")
        try:
            self.portfolio_optimizer.compute_min_variance()
            logger.info("engine.startup_checks.portfolio_weights_seeded")
        except Exception as exc:
            logger.warning(
                "engine.startup_checks.portfolio_init_failed",
                error=str(exc),
                hint="weights will be initialized on first market_open_job run",
            )

        logger.info("engine.startup_checks.done")

    # ------------------------------------------------------------------
    # Internal signal helpers
    # ------------------------------------------------------------------

    def _get_latest_llm_signal(self, ticker: str) -> dict[str, Any]:
        """
        Query signal_log for the most recent ``llm_sentiment`` entry for *ticker*
        written within the last 12 hours.  Returns neutral if nothing found.
        """
        try:
            with self._storage._engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT value, metadata
                        FROM   signal_log
                        WHERE  ticker      = :ticker
                          AND  signal_name = 'llm_sentiment'
                          AND  time       >= NOW() - INTERVAL '12 hours'
                        ORDER  BY time DESC
                        LIMIT  1
                        """
                    ),
                    {"ticker": ticker},
                ).fetchone()
        except Exception as exc:
            logger.warning("engine.llm_signal_query_failed", ticker=ticker, error=str(exc))
            return {"signal": 0, "confidence": 0.0}

        if row is None:
            return {"signal": 0, "confidence": 0.0, "contributing_headlines": []}

        value = float(row[0])
        _raw = row[1]
        meta: dict[str, Any] = _raw if isinstance(_raw, dict) else (json.loads(_raw) if _raw else {})
        direction = int(meta.get("direction", 0))
        confidence = float(meta.get("confidence", abs(value)))
        headlines: list[dict[str, Any]] = meta.get("contributing_headlines", [])
        return {
            "signal":                  direction,
            "confidence":              min(confidence, 1.0),
            "contributing_headlines":  headlines,
        }

    def _is_earnings_guard_triggered(self, ticker: str) -> bool:
        """
        Return ``True`` if today or tomorrow is an earnings date for *ticker*.

        Calls ``FundamentalsClient.get_earnings_dates`` which caches results for
        24 hours — the first call per day may hit yfinance; subsequent calls are
        instant dict lookups.  On any exception the guard fails **open** (returns
        ``False``) so a data fetch failure never blocks trading.
        """
        try:
            dates = self._fundamentals.get_earnings_dates([ticker])
            earnings_dt = dates.get(ticker)
            if earnings_dt is None:
                return False
            today = datetime.now(tz=timezone.utc).date()
            tomorrow = today + timedelta(days=1)
            earnings_date = (
                earnings_dt.date()
                if hasattr(earnings_dt, "date")
                else earnings_dt
            )
            return earnings_date in (today, tomorrow)
        except Exception as exc:
            logger.warning(
                "engine.earnings_guard.check_failed",
                ticker=ticker,
                error=str(exc),
            )
            return False  # fail open — don't block trades on guard failure

    def _get_analyst_signal(self, ticker: str) -> dict[str, Any]:
        """
        Return the consensus analyst recommendation as a MWU signal dict.

        Maps ``recommendationKey`` via ``FundamentalsClient.get_analyst_recommendations``
        to ``{"signal": -1/0/+1, "confidence": float}``.

        Fails open — returns neutral on any exception so a yfinance outage
        never blocks order submission.
        """
        try:
            recs = self._fundamentals.get_analyst_recommendations([ticker])
            direction = recs.get(ticker, 0) or 0
            confidence = 0.7 if direction != 0 else 0.0
            return {"signal": direction, "confidence": confidence}
        except Exception as exc:
            logger.warning(
                "engine.analyst_signal.failed",
                ticker=ticker,
                error=str(exc)[:120],
            )
            return {"signal": 0, "confidence": 0.0}

    def _get_ou_signal_for_ticker(self, ticker: str) -> dict[str, Any]:
        """
        Return the OU spread signal for the first pair that contains *ticker*.
        Returns neutral if no pair exists or not enough data.
        """
        for pair, ou in self._ou_signals.items():
            if ticker not in pair:
                continue
            try:
                end = datetime.now(tz=timezone.utc)
                start = end - timedelta(days=3)   # ~390 min-bars
                df1 = self._storage.query_ohlcv(pair[0], start, end)
                df2 = self._storage.query_ohlcv(pair[1], start, end)

                # Align on shared timestamps.  Alpaca may deliver a different
                # number of bars per ticker due to gaps or feed latency; passing
                # mismatched lengths to KalmanHedgeRatio raises a ValueError.
                if len(df1) != len(df2):
                    shared_times = set(df1["time"]).intersection(df2["time"])
                    n_before = (len(df1), len(df2))
                    df1 = df1[df1["time"].isin(shared_times)].reset_index(drop=True)
                    df2 = df2[df2["time"].isin(shared_times)].reset_index(drop=True)
                    logger.debug(
                        "engine.ou_series_aligned",
                        pair=pair,
                        before=n_before,
                        after=len(df1),
                    )

                min_rows = max(ou.lookback, 10)
                if len(df1) < min_rows or len(df2) < min_rows:
                    return {"signal": 0, "confidence": 0.0, "z_score": None,
                            "spread_value": None, "pair": f"{pair[0]}/{pair[1]}"}
                result = ou.compute_signal(df1, df2, storage=self._storage)
                z = result["z_score"]
                return {
                    "signal":       result["signal"],
                    "confidence":   min(abs(z) / 3.0, 1.0),
                    "z_score":      z,
                    "spread_value": result.get("spread"),
                    "pair":         f"{pair[0]}/{pair[1]}",
                }
            except Exception as exc:
                logger.warning(
                    "engine.ou_signal_failed",
                    ticker=ticker,
                    pair=pair,
                    error=str(exc),
                )
                return {"signal": 0, "confidence": 0.0, "z_score": None,
                        "spread_value": None, "pair": f"{pair[0]}/{pair[1]}"}
        return {"signal": 0, "confidence": 0.0, "z_score": None,
                "spread_value": None, "pair": None}

    # ------------------------------------------------------------------
    # Bar handler
    # ------------------------------------------------------------------

    def bar_handler(self, bar: dict[str, Any]) -> None:
        """
        Process one new price bar.  Called by the Alpaca WebSocket stream on
        each 1-minute (or 5-minute) bar.  Runs in the stream's background thread.

        Steps
        -----
        1. Insert bar to storage.
        2. HMM online update; predict regime.
        3. OU spread signal for any pair containing this ticker.
        4. Retrieve latest LLM sentiment signal from signal_log.
        5. MWU scheduled_update() → ensemble decision.
        6. Submit order if final_signal != 0.
        7. Check circuit breaker; trigger emergency shutdown if breached.
        """
        ticker = bar["ticker"]

        if ticker not in self._tickers:
            return  # bar for a symbol outside our universe

        # 1. Persist
        self._storage.insert_ohlcv([bar])

        # 2. HMM regime
        hmm = self._hmm[ticker]
        hmm.partial_fit_online(bar)

        hmm_signal: dict[str, Any] = {"signal": 0, "confidence": 0.0}
        regime = 1   # default neutral
        regime_label = "neutral"
        regime_probs: dict[str, float] = {}

        if hmm.is_fitted:
            try:
                regime_result = hmm.predict_regime(ticker, storage=self._storage)
                regime = regime_result["regime"]
                label = regime_result["label"]
                direction = _REGIME_TO_SIGNAL.get(label, 0)
                probs = regime_result["probs"]
                confidence = float(max(probs))
                hmm_signal = {"signal": direction, "confidence": confidence}
                regime_label = label
                # Map prob list to named dict using state_labels from the HMM
                state_labels = hmm.state_labels  # {state_idx: label_str}
                regime_probs = {
                    state_labels.get(i, str(i)): float(p)
                    for i, p in enumerate(probs)
                }
                logger.info(regime_banner(label, ticker))
            except Exception as exc:
                logger.warning(
                    "engine.hmm_predict_failed", ticker=ticker, error=str(exc)
                )

        # Regime smoothing: only adopt a new regime after _REGIME_SMOOTH_WINDOW
        # consecutive bars agree.  A single outlier bar cannot flip the regime.
        self._regime_history[ticker].append(regime_label)
        _hist = self._regime_history[ticker]
        if len(_hist) == _REGIME_SMOOTH_WINDOW and len(set(_hist)) == 1:
            self._stable_regime_label[ticker] = _hist[0]
            # Keep the stable regime *index* in sync so that MWU reads from the
            # same row that the smoothed label represents.
            self._stable_regime[ticker] = regime
        smoothed_label = self._stable_regime_label[ticker]
        smoothed_regime = self._stable_regime[ticker]
        if smoothed_label != regime_label:
            logger.debug(
                "engine.regime_smoothed",
                ticker=ticker,
                raw_label=regime_label,
                raw_regime=regime,
                smoothed_label=smoothed_label,
                smoothed_regime=smoothed_regime,
            )
            regime_label = smoothed_label
            regime = smoothed_regime
            direction = _REGIME_TO_SIGNAL.get(smoothed_label, 0)
            hmm_signal = {"signal": direction, "confidence": hmm_signal["confidence"]}

        # 3. OU spread signal
        ou_signal = self._get_ou_signal_for_ticker(ticker)

        # 4. LLM sentiment
        llm_signal = self._get_latest_llm_signal(ticker)

        # 5. Analyst recommendations (cached 24 h via FundamentalsClient)
        analyst_signal = self._get_analyst_signal(ticker)

        # 6. Assemble signals dict and run MWU
        signals: dict[str, dict[str, Any]] = {
            "hmm_regime":    hmm_signal,
            "ou_spread":     ou_signal,
            "llm_sentiment": llm_signal,
            "analyst_recs":  analyst_signal,
        }

        decision = self._mwu[ticker].scheduled_update(
            ticker=ticker,
            signals=signals,
            regime=regime,
            storage=self._storage,
        )

        final_signal: int = decision["final_signal"]

        # Minimum holding period: suppress direction reversals within _MIN_HOLD_MINUTES.
        # Rationale: Alpaca free-tier data has a ~15-minute delay, so no decision can
        # be more precise than that time scale.  Prevents rapid BUY→SELL→BUY churn
        # driven by single-bar HMM noise that slips through regime smoothing.
        _now = datetime.now(tz=timezone.utc)
        _prev_sig = self._last_active_signal[ticker]
        _last_t = self._last_signal_change_time[ticker]
        if (
            final_signal != 0
            and _prev_sig != 0
            and final_signal != _prev_sig
            and _last_t is not None
            and (_now - _last_t) < timedelta(minutes=_MIN_HOLD_MINUTES)
        ):
            _elapsed_min = (_now - _last_t).total_seconds() / 60
            logger.info(
                "engine.holding_period.suppressed",
                ticker=ticker,
                proposed_signal=final_signal,
                held_signal=_prev_sig,
                elapsed_min=round(_elapsed_min, 1),
            )
            final_signal = 0
        if final_signal != 0 and final_signal != _prev_sig:
            self._last_active_signal[ticker] = final_signal
            self._last_signal_change_time[ticker] = _now

        # 7. Order submission + trade log
        if final_signal != 0:
            # Always persist the decision so the dashboard shows it even when
            # the market is closed or the order is rejected.
            try:
                self._storage.insert_trade_log({
                    "time":                   datetime.now(tz=timezone.utc),
                    "ticker":                 ticker,
                    "final_signal":           final_signal,
                    "score":                  float(decision["score"]),
                    "regime":                 regime,
                    "regime_label":           regime_label,
                    "regime_probs":           regime_probs,
                    "hmm_signal":             int(hmm_signal["signal"]),
                    "hmm_confidence":         float(hmm_signal["confidence"]),
                    "ou_signal":              int(ou_signal.get("signal", 0)),
                    "ou_confidence":          float(ou_signal.get("confidence", 0.0)),
                    "ou_zscore":              _to_float(ou_signal.get("z_score")),
                    "ou_spread_value":        _to_float(ou_signal.get("spread_value")),
                    "ou_pair":                ou_signal.get("pair"),
                    "llm_signal":             int(llm_signal.get("signal", 0)),
                    "llm_confidence":         float(llm_signal.get("confidence", 0.0)),
                    "analyst_signal":         int(analyst_signal.get("signal", 0)),
                    "analyst_confidence":     float(analyst_signal.get("confidence", 0.0)),
                    "mwu_weights":            decision.get("weights"),
                    "contributing_headlines": llm_signal.get("contributing_headlines", []),
                })
            except Exception as exc:
                logger.warning("engine.trade_log_failed", ticker=ticker, error=str(exc))

            # Earnings guard: skip order submission on earnings day or the day before.
            # The decision is still logged to trade_log (above) for audit purposes.
            if self._is_earnings_guard_triggered(ticker):
                earnings_dates = self._fundamentals.get_earnings_dates([ticker])
                earnings_dt = earnings_dates.get(ticker)
                logger.info(
                    "engine.earnings_guard.triggered",
                    ticker=ticker,
                    earnings_date=str(earnings_dt.date()) if earnings_dt else None,
                )
            elif not self._alpaca.is_market_open():
                logger.debug(
                    "engine.bar_handler.order_skipped_market_closed", ticker=ticker
                )
            else:
                try:
                    account_info = self._alpaca.get_account_info()
                    self._executor.submit_order(
                        ticker=ticker,
                        signal=final_signal,
                        confidence=abs(decision["score"]),
                        account_info=account_info,
                        signal_stats=self._signal_stats[ticker],
                    )
                except Exception as exc:
                    logger.error(
                        "engine.order_failed", ticker=ticker, error=str(exc)
                    )

        # 8. Circuit breaker check (after any order attempt)
        try:
            account_info = self._alpaca.get_account_info()
            if self._risk.circuit_breaker(account_info):
                logger.critical(
                    "engine.circuit_breaker_triggered",
                    ticker=ticker,
                    equity=account_info["equity"],
                )
                self._emergency_close = True
                self._shutdown_event.set()
        except Exception as exc:
            logger.warning("engine.account_check_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Scheduled jobs
    # ------------------------------------------------------------------

    def sentiment_job(self) -> None:
        """
        Fetch news and run LLM sentiment pipeline.

        Invoked by two APScheduler cron jobs:
          - ``sentiment_job_early``: every 25 min, 07:00–10:29 ET, Mon–Fri
          - ``sentiment_job_late``:  every 35 min, 10:30–16:30 ET, Mon–Fri

        All tickers are fetched via Alpaca News (no Alpha Vantage calls).
        AV's free tier rejects multi-ticker queries with "Invalid inputs" and
        has a 20-call/day hard limit that is exhausted within a few hours.
        Alpaca News has no rate limits and returns equivalent article coverage.
        """
        logger.info(
            "engine.sentiment_job.start",
            n_tickers=len(self._tickers),
        )
        try:
            results = self._llm.run_pipeline(
                self._tickers,
                self._av_client,
                self._storage,
                av_tickers=[],           # empty → skip AV, all tickers via Alpaca
                alpaca_client=self._alpaca_news,
            )
            logger.info(
                "engine.sentiment_job.done",
                n_tickers=len(results),
            )
        except Exception as exc:
            logger.error("engine.sentiment_job.failed", error=str(exc))

    def market_open_job(self) -> None:
        """
        Compute daily portfolio target weights and execute rebalance at 09:31 ET (Mon–Fri).

        1. Collect the most recent MWU decision for each ticker.
        2. Run Black-Litterman optimisation; fall back to min-variance on failure.
        3. If optimisation succeeded, call _execute_rebalance_orders().
        """
        optimized = False
        try:
            mwu_scores: dict[str, dict[str, Any]] = {}
            for ticker in self._tickers:
                agent = self._mwu.get(ticker)
                if agent is None:
                    continue
                last = getattr(agent, "last_decision", None)
                if last:
                    weights_values = list(last.get("weights", {}).values())
                    confidence = max(weights_values) if weights_values else 0.33
                    mwu_scores[ticker] = {
                        "score": last.get("score", 0.0),
                        "confidence": confidence,
                        "final_signal": last.get("final_signal", 0),
                    }

            result = self.portfolio_optimizer.compute_black_litterman(mwu_scores)
            logger.info(
                "engine.portfolio_optimized",
                method=result["method"],
                weights=result["weights"],
                n_views=result["n_views"],
            )
            optimized = True
        except Exception as exc:
            logger.error("engine.portfolio_optimization_failed", error=str(exc))
            try:
                self.portfolio_optimizer.compute_min_variance()
                optimized = True
            except Exception as exc2:
                logger.error("engine.min_variance_fallback_failed", error=str(exc2))

        if not optimized:
            logger.warning("engine.rebalance.skipped_no_weights")
            return

        self._execute_rebalance_orders()

    def _execute_rebalance_orders(self) -> None:
        """
        Fetch current positions, compute rebalance orders, and execute them.

        Execution order
        ---------------
        Sells are processed before buys so that capital freed by reducing
        positions is available for the subsequent purchases.

        Safety
        ------
        - Circuit-breaker is checked before any order is submitted.
        - Each order is wrapped in its own try/except; a failure on one ticker
          never prevents the remaining tickers from executing.
        - Sell quantities are capped at the currently held share count.
        """
        # Fetch account state and run circuit-breaker gate
        try:
            account_info = self._alpaca.get_account_info()
            equity = float(account_info["equity"])
        except Exception as exc:
            logger.error("engine.rebalance.account_fetch_failed", error=str(exc))
            return

        if self._risk.circuit_breaker(account_info):
            logger.warning(
                "engine.rebalance.halted_circuit_breaker",
                equity=equity,
            )
            return

        if not self._alpaca.is_market_open():
            logger.warning("engine.rebalance.skipped_market_closed")
            return

        # Fetch current open positions
        try:
            positions = self._executor.get_positions()
        except Exception as exc:
            logger.error("engine.rebalance.positions_fetch_failed", error=str(exc))
            return

        orders = self.portfolio_optimizer.get_rebalance_orders(positions, equity)

        sells = [o for o in orders if o["action"] == "sell"]
        buys  = [o for o in orders if o["action"] == "buy"]
        n_executed = n_skipped = n_errors = 0

        # Pre-fetch today's filled buys to avoid Pattern Day Trader (PDT) rejections.
        # Selling a position opened the same day triggers Alpaca error 40310100 on
        # accounts with < $25 K equity.  We skip such sells proactively rather than
        # submitting orders we know will be rejected.
        same_day_buys: set[str] = self._executor.get_todays_filled_buy_symbols()
        if same_day_buys:
            logger.info(
                "engine.rebalance.same_day_buys_detected",
                symbols=sorted(same_day_buys),
            )

        # ---- Phase 1: execute all sells first to free up capital ----
        for order in sells:
            ticker = order["ticker"]
            dollar_amount = float(order["dollar_amount"])

            try:
                # PDT pre-check: skip same-day sells to avoid Alpaca error 40310100.
                if ticker in same_day_buys:
                    logger.warning(
                        "engine.rebalance.pdt_skip",
                        ticker=ticker,
                        reason="position_opened_today",
                    )
                    n_skipped += 1
                    continue

                quote = self._alpaca.get_latest_quote(ticker)
                price = float(quote["mid"])
                if price <= 0:
                    logger.warning("engine.rebalance.zero_price", ticker=ticker)
                    n_skipped += 1
                    continue

                n_shares = int(dollar_amount / price)
                if n_shares <= 0:
                    logger.debug(
                        "engine.rebalance.order_too_small",
                        ticker=ticker,
                        dollar_amount=dollar_amount,
                    )
                    n_skipped += 1
                    continue

                # Cap at currently held quantity to avoid over-selling
                held = 0
                if not positions.empty and "ticker" in positions.columns:
                    pos_row = positions[positions["ticker"] == ticker]
                    if not pos_row.empty:
                        held = int(float(pos_row.iloc[0]["qty"]))
                if held <= 0:
                    logger.debug("engine.rebalance.sell_no_position", ticker=ticker)
                    n_skipped += 1
                    continue
                n_shares = min(n_shares, held)

                order_req = MarketOrderRequest(
                    symbol=ticker,
                    qty=n_shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                submitted = self._executor._trading.submit_order(order_req)
                n_executed += 1
                logger.info(
                    "engine.rebalance.order_submitted",
                    ticker=ticker,
                    action="sell",
                    qty=n_shares,
                    dollar_amount=round(dollar_amount, 2),
                    target_weight=order["target_weight"],
                    order_id=str(submitted.id),
                )

            except Exception as exc:
                exc_str = str(exc)
                # Reactive fallback: PDT rejection from Alpaca (error code 40310100).
                # Downgrade to warning + skip so a PDT block doesn't inflate n_errors.
                if "40310100" in exc_str:
                    n_skipped += 1
                    logger.warning(
                        "engine.rebalance.pdt_skip",
                        ticker=ticker,
                        reason="pattern_day_trader_protection",
                    )
                else:
                    n_errors += 1
                    logger.error(
                        "engine.rebalance.order_failed",
                        ticker=ticker,
                        action="sell",
                        error=exc_str,
                    )

        # ---- Phase 2: re-fetch cash after sells, then execute buys ----
        try:
            account_info = self._alpaca.get_account_info()
            available_cash = float(account_info["cash"])
        except Exception as exc:
            logger.error("engine.rebalance.cash_refetch_failed", error=str(exc))
            n_skipped += len(buys)
            logger.info(
                "engine.rebalance.summary",
                n_executed=n_executed,
                n_skipped=n_skipped,
                n_errors=n_errors,
                n_sells=len(sells),
                n_buys=len(buys),
            )
            return

        logger.info(
            "engine.rebalance.cash_after_sells",
            available_cash=round(available_cash, 2),
        )

        for order in buys:
            ticker = order["ticker"]
            dollar_amount = float(order["dollar_amount"])

            # Cap buy at remaining available cash
            if dollar_amount > available_cash:
                if available_cash <= 0:
                    logger.info("engine.rebalance.no_cash_remaining", ticker=ticker)
                    n_skipped += 1
                    continue
                logger.info(
                    "engine.rebalance.buy_capped_by_cash",
                    ticker=ticker,
                    requested=round(dollar_amount, 2),
                    available=round(available_cash, 2),
                )
                dollar_amount = available_cash

            try:
                quote = self._alpaca.get_latest_quote(ticker)
                price = float(quote["mid"])
                if price <= 0:
                    logger.warning("engine.rebalance.zero_price", ticker=ticker)
                    n_skipped += 1
                    continue

                n_shares = int(dollar_amount / price)
                if n_shares <= 0:
                    logger.debug(
                        "engine.rebalance.order_too_small",
                        ticker=ticker,
                        dollar_amount=dollar_amount,
                    )
                    n_skipped += 1
                    continue

                order_req = MarketOrderRequest(
                    symbol=ticker,
                    qty=n_shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                submitted = self._executor._trading.submit_order(order_req)
                available_cash -= n_shares * price   # track cash spend
                n_executed += 1
                logger.info(
                    "engine.rebalance.order_submitted",
                    ticker=ticker,
                    action="buy",
                    qty=n_shares,
                    dollar_amount=round(dollar_amount, 2),
                    target_weight=order["target_weight"],
                    order_id=str(submitted.id),
                )

            except Exception as exc:
                n_errors += 1
                logger.error(
                    "engine.rebalance.order_failed",
                    ticker=ticker,
                    action="buy",
                    error=str(exc),
                )

        logger.info(
            "engine.rebalance.summary",
            n_executed=n_executed,
            n_skipped=n_skipped,
            n_errors=n_errors,
            n_sells=len(sells),
            n_buys=len(buys),
        )

    def eod_job(self) -> None:
        """
        End-of-day housekeeping (runs at 16:05 ET, Mon–Fri via APScheduler).

        1. Log daily P&L summary.
        2. MWU performance report per ticker.
        3. Update signal stats for Kelly sizing.
        4. Persist all model state.
        """
        logger.info("engine.eod_job.start")

        # Daily P&L
        try:
            account_info = self._alpaca.get_account_info()
            positions = self._executor.get_positions()
            logger.info(
                "engine.eod_job.pnl_summary",
                equity=account_info["equity"],
                cash=account_info["cash"],
                open_positions=len(positions),
            )
        except Exception as exc:
            logger.error("engine.eod_job.pnl_failed", error=str(exc))

        # MWU performance + Kelly stat update
        for ticker in self._tickers:
            try:
                report = self._mwu[ticker].performance_report()
                logger.info(
                    "engine.eod_job.mwu_report",
                    ticker=ticker,
                    n_updates=report.get("n_updates"),
                    weights=report.get("current_weights"),
                )

            except Exception as exc:
                logger.error(
                    "engine.eod_job.mwu_report_failed", ticker=ticker, error=str(exc)
                )

        # Update Kelly stats from confirmed Alpaca fill P&L
        try:
            self._update_kelly_stats()
        except Exception as exc:
            logger.error("engine.eod_job.kelly_stats_failed", error=str(exc))

        # Persist state
        try:
            self._save_state()
        except Exception as exc:
            logger.error("engine.eod_job.state_save_failed", error=str(exc))

        logger.info("engine.eod_job.done")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the trading engine.

        1. Configure APScheduler with two sentiment cron windows and EOD (16:05 ET)
           jobs — 4 jobs total.
        2. Start the Alpaca WebSocket bar stream (background thread).
        3. Block until ``KeyboardInterrupt`` or the circuit-breaker fires.
        4. On shutdown: persist state; close all positions if emergency flag set.
        """
        logger.info("engine.run.start", tickers=self._tickers)

        # APScheduler
        self._scheduler = BackgroundScheduler(
            executors={"default": APThreadPoolExecutor(max_workers=4)},
            job_defaults={"misfire_grace_time": 300},
        )

        # Sentiment job — two cron windows to cover pre-market through close.
        # Budget math: ~8 calls (early) + ~10 calls (late) ≈ 18 calls/day.
        self._scheduler.add_job(
            self.sentiment_job,
            "cron",
            hour="7-10",
            minute="*/25",
            day_of_week="mon-fri",
            timezone=_ET,
            id="sentiment_job_early",
        )
        self._scheduler.add_job(
            self.sentiment_job,
            "cron",
            hour="10-16",
            minute="*/35",
            day_of_week="mon-fri",
            timezone=_ET,
            id="sentiment_job_late",
        )
        self._scheduler.add_job(
            self.market_open_job,
            "cron",
            day_of_week="mon-fri",
            hour=9,
            minute=31,
            timezone=_ET,
            id="market_open_job",
            # Fire immediately at engine startup so weights are refreshed with
            # the latest MWU decisions on every restart, not just at 09:31 ET.
            # _execute_rebalance_orders() is guarded by is_market_open(), so
            # no orders are placed outside trading hours.
            next_run_time=datetime.now(tz=timezone.utc),
        )
        self._scheduler.add_job(
            self.eod_job,
            "cron",
            day_of_week="mon-fri",
            hour=16,
            minute=5,
            timezone=_ET,
            id="eod_job",
        )

        self._scheduler.start()
        logger.info("engine.run.scheduler_started")

        # Live bar stream
        self._alpaca.stream_bars(self._tickers, self.bar_handler)
        logger.info("engine.run.stream_started", tickers=self._tickers)

        # Block until shutdown
        try:
            self._shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("engine.run.keyboard_interrupt")

        self._shutdown()

    def _shutdown(self) -> None:
        """Graceful shutdown sequence."""
        logger.info(
            "engine.shutdown.start",
            emergency=self._emergency_close,
        )

        # Stop the bar stream
        try:
            self._alpaca.stop_stream()
        except Exception as exc:
            logger.warning("engine.shutdown.stream_stop_failed", error=str(exc))

        # Stop the scheduler
        if self._scheduler and self._scheduler.running:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception as exc:
                logger.warning("engine.shutdown.scheduler_stop_failed", error=str(exc))

        # Emergency liquidation
        if self._emergency_close:
            logger.critical("engine.shutdown.emergency_liquidation")
            try:
                self._executor.close_all_positions()
            except Exception as exc:
                logger.critical(
                    "engine.shutdown.liquidation_failed", error=str(exc)
                )

        # Final state save
        try:
            self._save_state()
        except Exception as exc:
            logger.error("engine.shutdown.state_save_failed", error=str(exc))

        logger.info("engine.shutdown.done")
