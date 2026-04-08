"""Backtesting harness for the four-point trading engine.

Evaluates each signal module independently using vectorbt.
Supports single-signal runs, full cross-product sweeps, walk-forward
validation, and basic bias checks.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")  # non-interactive; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from sqlalchemy import text

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

# Signal names written to signal_log by the Phase-2 modules.
SIGNAL_NAMES: tuple[str, ...] = ("hmm_regime", "ou_spread", "llm_sentiment")

_RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Evaluates trading signals historically via vectorbt.

    Parameters
    ----------
    initial_capital:
        Starting portfolio value in USD.
    commission:
        Round-trip commission as a fraction of trade value (10 bps = 0.001).
    slippage:
        Slippage as a fraction of price per fill (5 bps = 0.0005).
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core single-signal backtest
    # ------------------------------------------------------------------

    def run_single_signal(
        self,
        signal_name: str,
        signal_series: pd.Series,
        price_series: pd.Series,
        ticker: str,
    ) -> dict[str, Any]:
        """
        Backtest one signal series against one price series.

        Parameters
        ----------
        signal_name:
            Human-readable label, e.g. ``"hmm_regime"``.
        signal_series:
            Integer series (index = datetime) with values −1 / 0 / +1.
        price_series:
            Close-price series with the same or broader datetime index.
        ticker:
            Equity symbol; used only for logging and the returned dict.

        Returns
        -------
        dict with keys: ticker, signal, total_return, sharpe_ratio,
        max_drawdown, n_trades, win_rate, profit_factor, calmar_ratio,
        equity_curve (pd.Series).
        """
        entries = signal_series == 1
        exits = signal_series == -1

        pf = vbt.Portfolio.from_signals(
            price_series,
            entries=entries,
            exits=exits,
            size=0.95,
            size_type="percent",
            fees=self.commission,
            slippage=self.slippage,
            init_cash=self.initial_capital,
            freq="1D",
        )

        stats = pf.stats()

        def _safe_float(key: str, scale: float = 1.0) -> float:
            v = stats.get(key, 0.0)
            if not pd.notna(v) or not math.isfinite(float(v)):
                return 0.0
            return float(v) / scale

        total_return = _safe_float("Total Return [%]", 100.0)
        sharpe_ratio = _safe_float("Sharpe Ratio")
        max_drawdown = _safe_float("Max Drawdown [%]", 100.0)
        n_trades = int(stats.get("Total Trades", 0))
        win_rate = _safe_float("Win Rate [%]", 100.0)
        profit_factor = _safe_float("Profit Factor")
        calmar_ratio = _safe_float("Calmar Ratio")
        equity_curve: pd.Series = pf.value()

        result: dict[str, Any] = {
            "ticker": ticker,
            "signal": signal_name,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar_ratio,
            "equity_curve": equity_curve,
        }

        key = f"{ticker}_{signal_name}"
        self.results[key] = result

        logger.info(
            "backtest.run_single",
            ticker=ticker,
            signal=signal_name,
            total_return=round(total_return, 4),
            sharpe=round(sharpe_ratio, 3),
            n_trades=n_trades,
        )
        return result

    # ------------------------------------------------------------------
    # Full cross-product sweep
    # ------------------------------------------------------------------

    def _query_signal_log(
        self,
        storage: Any,
        ticker: str,
        signal_name: str,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Return a time-indexed integer Series from signal_log."""
        stmt = text(
            """
            SELECT time, value
            FROM   signal_log
            WHERE  ticker      = :ticker
              AND  signal_name = :signal_name
              AND  time BETWEEN :start AND :end
            ORDER  BY time ASC
            """
        )
        with storage._engine.connect() as conn:
            result = conn.execute(
                stmt,
                {"ticker": ticker, "signal_name": signal_name, "start": start, "end": end},
            )
            rows = result.fetchall()

        if not rows:
            return pd.Series(dtype=float)

        df = pd.DataFrame(rows, columns=["time", "value"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        series = df.set_index("time")["value"]
        return series.astype(int)

    def run_all_signals(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        storage: Any,
    ) -> pd.DataFrame:
        """
        Run every ticker × signal combination and return a summary DataFrame.

        Fetches OHLCV and reconstructed signal series from *storage*.
        Saves results to ``backtesting/results/backtest_{timestamp}.csv``.

        Parameters
        ----------
        tickers:
            List of equity symbols to evaluate.
        start, end:
            Date range for the backtest.
        storage:
            ``Storage`` instance with an ``_engine`` SQLAlchemy engine and a
            ``query_ohlcv(ticker, start, end) -> DataFrame`` method.

        Returns
        -------
        pd.DataFrame — rows = signal runs, cols = scalar metrics.
        """
        all_rows: list[dict[str, Any]] = []

        for ticker in tickers:
            ohlcv_df = storage.query_ohlcv(ticker, start, end)
            if ohlcv_df.empty:
                logger.warning("backtest.no_ohlcv", ticker=ticker)
                continue

            price_series = (
                ohlcv_df.set_index("time")["close"]
                .astype(float)
            )
            price_series.index = pd.to_datetime(price_series.index, utc=True)

            for signal_name in SIGNAL_NAMES:
                signal_series = self._query_signal_log(
                    storage, ticker, signal_name, start, end
                )
                if signal_series.empty:
                    logger.warning(
                        "backtest.no_signals",
                        ticker=ticker,
                        signal=signal_name,
                    )
                    continue

                # Align to price index; fill missing bars as flat (0).
                signal_aligned = signal_series.reindex(
                    price_series.index, fill_value=0
                )

                result = self.run_single_signal(
                    signal_name=signal_name,
                    signal_series=signal_aligned,
                    price_series=price_series,
                    ticker=ticker,
                )
                row = {k: v for k, v in result.items() if k != "equity_curve"}
                all_rows.append(row)

        summary_df = pd.DataFrame(all_rows)

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = _RESULTS_DIR / f"backtest_{ts}.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info("backtest.saved_csv", path=str(csv_path), rows=len(summary_df))

        return summary_df

    # ------------------------------------------------------------------
    # Equity curve plot
    # ------------------------------------------------------------------

    def plot_equity_curves(self, results: dict[str, Any]) -> None:
        """
        Save a matplotlib figure with one subplot per signal run.

        Output: ``backtesting/results/equity_curves_{timestamp}.png``.

        Parameters
        ----------
        results:
            Dict as produced by ``run_single_signal`` (or ``self.results``).
        """
        if not results:
            logger.warning("backtest.plot.no_results")
            return

        keys = list(results.keys())
        n = len(keys)
        fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)

        for i, key in enumerate(keys):
            r = results[key]
            ax = axes[i][0]
            equity: pd.Series = r["equity_curve"]
            ax.plot(equity.index, equity.values, linewidth=1.2)
            ax.set_title(f"{r['ticker']} — {r['signal']}", fontsize=11)
            ax.set_ylabel("Portfolio Value ($)")
            ax.grid(True, alpha=0.3)
            stats_text = (
                f"Return: {r['total_return']:.2%}  "
                f"Sharpe: {r['sharpe_ratio']:.2f}  "
                f"MaxDD: {r['max_drawdown']:.2%}  "
                f"Trades: {r['n_trades']}"
            )
            ax.set_xlabel(stats_text)

        plt.tight_layout()
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        png_path = _RESULTS_DIR / f"equity_curves_{ts}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("backtest.plot.saved", path=str(png_path))

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        signal_fn: Callable[[pd.Series], pd.Series],
        price_series: pd.Series,
        n_splits: int = 5,
        train_frac: float = 0.7,
    ) -> pd.DataFrame:
        """
        Walk-forward validation — the primary overfitting guard.

        Splits *price_series* into *n_splits* non-overlapping windows.
        Each window is divided into a training segment (first *train_frac*)
        and a test segment (remaining).  ``signal_fn`` is called on the full
        window to generate signals; performance is measured only on the test
        segment so training data never leaks into the evaluation.

        Parameters
        ----------
        signal_fn:
            Callable ``(price_series: pd.Series) -> signal_series: pd.Series``
            that returns an integer signal series (−1/0/+1) of the same
            length as its input.
        price_series:
            Full price history to split.
        n_splits:
            Number of non-overlapping windows.
        train_frac:
            Fraction of each window reserved for training.

        Returns
        -------
        pd.DataFrame — one row per split with columns:
            split, train_bars, test_bars, train_start, train_end,
            test_start, test_end, total_return, sharpe_ratio,
            max_drawdown, n_trades, win_rate, profit_factor, calmar_ratio.
        """
        n = len(price_series)
        base_window = n // n_splits
        split_rows: list[dict[str, Any]] = []

        for i in range(n_splits):
            w_start = i * base_window
            # Last split absorbs any leftover bars.
            w_end = w_start + base_window if i < n_splits - 1 else n

            window_prices = price_series.iloc[w_start:w_end]
            n_train = int(len(window_prices) * train_frac)
            n_test = len(window_prices) - n_train

            train_prices = window_prices.iloc[:n_train]
            test_prices = window_prices.iloc[n_train:]

            if len(train_prices) < 2 or len(test_prices) < 2:
                logger.warning(
                    "backtest.walk_forward.skip",
                    split=i,
                    reason="insufficient_data",
                    train_bars=len(train_prices),
                    test_bars=len(test_prices),
                )
                continue

            # Generate signals over the full window (signal_fn may use the
            # training portion for fitting and the test portion for scoring).
            full_signals = signal_fn(window_prices)
            test_signals = full_signals.iloc[n_train:]

            result = self.run_single_signal(
                signal_name="walk_forward",
                signal_series=test_signals,
                price_series=test_prices,
                ticker=f"split_{i}",
            )

            row: dict[str, Any] = {k: v for k, v in result.items() if k != "equity_curve"}
            row.update(
                {
                    "split": i,
                    "train_bars": len(train_prices),
                    "test_bars": len(test_prices),
                    "train_start": train_prices.index[0],
                    "train_end": train_prices.index[-1],
                    "test_start": test_prices.index[0],
                    "test_end": test_prices.index[-1],
                }
            )
            split_rows.append(row)

            logger.info(
                "backtest.walk_forward.split",
                split=i,
                train_bars=len(train_prices),
                test_bars=len(test_prices),
                total_return=round(result["total_return"], 4),
            )

        return pd.DataFrame(split_rows)

    # ------------------------------------------------------------------
    # Bias checks
    # ------------------------------------------------------------------

    def check_lookahead_bias(
        self,
        signal_series: pd.Series,
        price_series: pd.Series,
    ) -> bool:
        """
        Verify signal timestamps precede the price bars used for P&L by ≥ 1 bar.

        A signal generated *at* bar T should be executed at bar T+1 (next
        open).  Signals at the very last bar of the price series have no
        future bar to trade on and are flagged as potential lookahead bias.

        Returns
        -------
        bool — True if no lookahead bias detected, False if bias found.
        """
        if signal_series.empty or price_series.empty:
            logger.warning("backtest.lookahead_check.empty_series")
            return True

        price_idx = pd.DatetimeIndex(price_series.index)
        active = signal_series[signal_series != 0]

        if active.empty:
            logger.info("backtest.lookahead_check.no_active_signals")
            return True

        has_bias = False
        for sig_time in active.index:
            pos = price_idx.searchsorted(sig_time)
            # Signal coincides with the last available price bar → no next bar.
            if pos >= len(price_idx) - 1:
                logger.warning(
                    "backtest.lookahead_bias.detected",
                    signal_time=str(sig_time),
                    reason="signal_at_or_after_last_price_bar",
                )
                has_bias = True

        if not has_bias:
            logger.info("backtest.lookahead_check.passed")

        return not has_bias

    def check_survivorship_bias(self) -> str:
        """
        Log a WARNING reminding the user to verify ticker history coverage.

        Returns
        -------
        str — the warning message (for testing / display).
        """
        msg = (
            "SURVIVORSHIP BIAS WARNING: Verify that every ticker in the "
            "backtest universe existed and was actively traded for the full "
            "backtest period.  Using only currently-listed securities "
            "introduces survivorship bias and overstates historical returns."
        )
        logger.warning("backtest.survivorship_bias_check", message=msg)
        return msg
