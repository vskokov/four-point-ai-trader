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
from trading_engine.data.alpaca_client import AlpacaMarketData
from trading_engine.data.storage import Storage
from trading_engine.execution.executor import OrderExecutor, RiskManager
from trading_engine.meta_agent.mwu_agent import MWUMetaAgent
from trading_engine.portfolio.portfolio_optimizer import PortfolioOptimizer
from trading_engine.orchestrator.state_manager import StateManager
from trading_engine.signals.hmm_regime import HMMRegimeDetector
from trading_engine.signals.llm_sentiment import LLMSentimentSignal
from trading_engine.signals.mean_reversion import OUSpreadSignal
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

_ET = ZoneInfo("America/New_York")

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
            t: MWUMetaAgent(models_dir=self._models_dir)
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

        # Load any persisted metadata
        self._load_state()

        # Load persisted model artefacts (HMM, Kalman inside OU, MWU weights)
        self._load_models()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Restore engine metadata from the last saved snapshot (if any)."""
        try:
            state = self._state_manager.load()
        except ValueError as exc:
            logger.error("engine.state_load_failed", error=str(exc))
            state = None

        if state is None:
            return

        self._signal_stats.update(state.get("signal_stats", {}))
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
            return {"signal": 0, "confidence": 0.0}

        value = float(row[0])
        meta: dict[str, Any] = json.loads(row[1]) if row[1] else {}
        direction = int(meta.get("direction", 0))
        confidence = float(meta.get("confidence", abs(value)))
        return {"signal": direction, "confidence": min(confidence, 1.0)}

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
                min_rows = max(ou.lookback, 10)
                if len(df1) < min_rows or len(df2) < min_rows:
                    return {"signal": 0, "confidence": 0.0}
                result = ou.compute_signal(df1, df2, storage=self._storage)
                z = result["z_score"]
                return {
                    "signal": result["signal"],
                    "confidence": min(abs(z) / 3.0, 1.0),
                }
            except Exception as exc:
                logger.warning(
                    "engine.ou_signal_failed",
                    ticker=ticker,
                    pair=pair,
                    error=str(exc),
                )
                return {"signal": 0, "confidence": 0.0}
        return {"signal": 0, "confidence": 0.0}

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

        if hmm.is_fitted:
            try:
                regime_result = hmm.predict_regime(ticker, storage=self._storage)
                regime = regime_result["regime"]
                label = regime_result["label"]
                direction = _REGIME_TO_SIGNAL.get(label, 0)
                confidence = float(max(regime_result["probs"]))
                hmm_signal = {"signal": direction, "confidence": confidence}
            except Exception as exc:
                logger.warning(
                    "engine.hmm_predict_failed", ticker=ticker, error=str(exc)
                )

        # 3. OU spread signal
        ou_signal = self._get_ou_signal_for_ticker(ticker)

        # 4. LLM sentiment
        llm_signal = self._get_latest_llm_signal(ticker)

        # 5. Assemble signals dict and run MWU
        signals: dict[str, dict[str, Any]] = {
            "hmm_regime":    hmm_signal,
            "ou_spread":     ou_signal,
            "llm_sentiment": llm_signal,
        }

        decision = self._mwu[ticker].scheduled_update(
            ticker=ticker,
            signals=signals,
            regime=regime,
            storage=self._storage,
        )

        final_signal: int = decision["final_signal"]

        # 6. Order submission
        if final_signal != 0:
            if not self._alpaca.is_market_open():
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

        # 7. Circuit breaker check (after any order attempt)
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

        Includes a soft budget guard: if the AV daily call count has already
        reached 20 (the hard limit in ``_check_and_increment``), the run is
        skipped to avoid raising ``RateLimitExceeded``.  The hard limit in
        ``AlphaVantageNewsClient`` remains the authoritative cap.
        """
        count = self._av_client.get_daily_call_count()
        if count >= 20:
            logger.warning(
                "engine.sentiment_job.skipped",
                reason="daily_budget_reached",
                calls_today=count,
            )
            return

        logger.info("engine.sentiment_job.start", calls_today=count)
        try:
            results = self._llm.run_pipeline(
                self._tickers, self._av_client, self._storage
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

        # ---- Phase 1: execute all sells first to free up capital ----
        for order in sells:
            ticker = order["ticker"]
            dollar_amount = float(order["dollar_amount"])

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
                n_errors += 1
                logger.error(
                    "engine.rebalance.order_failed",
                    ticker=ticker,
                    action="sell",
                    error=str(exc),
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

                # Derive ensemble win rate from per-signal rates
                rates = [
                    v for v in report.get("per_signal_win_rate", {}).values()
                    if v is not None
                ]
                if rates:
                    self._signal_stats[ticker]["win_rate"] = sum(rates) / len(rates)
            except Exception as exc:
                logger.error(
                    "engine.eod_job.mwu_report_failed", ticker=ticker, error=str(exc)
                )

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
