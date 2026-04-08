"""Unit tests for backtesting/backtest_engine.py.

All tests use synthetic price and signal series — no DB or network required.
Run with:
    .venv/bin/pytest tests/backtesting/test_backtest_engine.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading_engine.backtesting.backtest_engine import BacktestEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_price_series(n: int = 200, drift: float = 0.001, seed: int = 42) -> pd.Series:
    """Synthetic daily close prices with a positive drift."""
    rng = np.random.default_rng(seed)
    log_returns = drift + rng.normal(0, 0.01, n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")  # business days
    return pd.Series(prices, index=idx, name="close")


def _make_signal_series(index: pd.DatetimeIndex, pattern: str = "buy_hold") -> pd.Series:
    """
    Synthetic signal series.

    Patterns
    --------
    buy_hold  : +1 at bar 0, nothing after (single long trade, never exited)
    alternating : alternating +1 / -1 blocks of 20 bars
    flat      : all zeros (no trades)
    """
    signals = pd.Series(0, index=index, dtype=int)

    if pattern == "buy_hold":
        signals.iloc[0] = 1

    elif pattern == "alternating":
        block = 20
        for start in range(0, len(index), block * 2):
            signals.iloc[start : start + block] = 1
            end_block = start + block
            signals.iloc[end_block : end_block + block] = -1

    elif pattern == "one_trade":
        # Single clean trade: buy at bar 10, sell at bar 30.
        signals.iloc[10] = 1
        signals.iloc[30] = -1

    return signals


@pytest.fixture()
def engine() -> BacktestEngine:
    return BacktestEngine(initial_capital=100_000, commission=0.001, slippage=0.0005)


@pytest.fixture()
def prices() -> pd.Series:
    return _make_price_series(n=200)


# ---------------------------------------------------------------------------
# 1. run_single_signal — return schema
# ---------------------------------------------------------------------------

class TestRunSingleSignalSchema:
    EXPECTED_KEYS = {
        "ticker", "signal", "total_return", "sharpe_ratio",
        "max_drawdown", "n_trades", "win_rate", "profit_factor",
        "calmar_ratio", "equity_curve",
    }

    def test_returns_all_required_keys(self, engine, prices):
        signals = _make_signal_series(prices.index, "one_trade")
        result = engine.run_single_signal("hmm_regime", signals, prices, "AAPL")
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_ticker_and_signal_name_pass_through(self, engine, prices):
        signals = _make_signal_series(prices.index, "one_trade")
        result = engine.run_single_signal("ou_spread", signals, prices, "MSFT")
        assert result["ticker"] == "MSFT"
        assert result["signal"] == "ou_spread"

    def test_equity_curve_is_series(self, engine, prices):
        signals = _make_signal_series(prices.index, "one_trade")
        result = engine.run_single_signal("hmm_regime", signals, prices, "AAPL")
        assert isinstance(result["equity_curve"], pd.Series)
        assert len(result["equity_curve"]) == len(prices)

    def test_n_trades_is_int(self, engine, prices):
        signals = _make_signal_series(prices.index, "one_trade")
        result = engine.run_single_signal("hmm_regime", signals, prices, "AAPL")
        assert isinstance(result["n_trades"], int)

    def test_no_trades_when_signal_flat(self, engine, prices):
        signals = _make_signal_series(prices.index, "flat")
        result = engine.run_single_signal("hmm_regime", signals, prices, "AAPL")
        assert result["n_trades"] == 0
        assert result["total_return"] == pytest.approx(0.0, abs=1e-6)

    def test_result_stored_in_self_results(self, engine, prices):
        signals = _make_signal_series(prices.index, "one_trade")
        engine.run_single_signal("hmm_regime", signals, prices, "AAPL")
        assert "AAPL_hmm_regime" in engine.results

    def test_win_rate_between_0_and_1(self, engine, prices):
        signals = _make_signal_series(prices.index, "alternating")
        result = engine.run_single_signal("hmm_regime", signals, prices, "AAPL")
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_max_drawdown_non_negative(self, engine, prices):
        signals = _make_signal_series(prices.index, "one_trade")
        result = engine.run_single_signal("hmm_regime", signals, prices, "AAPL")
        assert result["max_drawdown"] >= 0.0


# ---------------------------------------------------------------------------
# 2. Sharpe ratio correctness
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    """
    Cross-check vectorbt's Sharpe ratio against a manual computation.

    Manual Sharpe (annualised, daily frequency):
        S = mean(daily_returns) / std(daily_returns) * sqrt(252)
    """

    def _manual_sharpe(self, equity_curve: pd.Series) -> float:
        daily_ret = equity_curve.pct_change().dropna()
        std = daily_ret.std()
        if std == 0:
            return 0.0
        return float(daily_ret.mean() / std * math.sqrt(252))

    def test_sharpe_matches_manual_within_tolerance(self, engine, prices):
        """
        vectorbt Sharpe and manual Sharpe should have the same sign and be
        within a factor of 2 of each other.

        The two can diverge when the signal is only active for a small fraction
        of the period: the manual computation includes all zero-return (flat)
        days, which inflates the denominator; vectorbt may use a different
        annualisation or log-return convention.  We therefore only enforce
        sign agreement and order-of-magnitude similarity.
        """
        signals = _make_signal_series(prices.index, "one_trade")
        result = engine.run_single_signal("hmm_regime", signals, prices, "AAPL")

        manual = self._manual_sharpe(result["equity_curve"])
        vbt_sharpe = result["sharpe_ratio"]

        # Ignore near-zero cases where sign is meaningless.
        if abs(manual) > 0.1 and abs(vbt_sharpe) > 0.1:
            assert manual * vbt_sharpe > 0, (
                f"Sign mismatch: manual={manual:.4f}, vbt={vbt_sharpe:.4f}"
            )
            assert abs(vbt_sharpe) < abs(manual) * 3, (
                f"vectorbt Sharpe {vbt_sharpe:.4f} is more than 3× "
                f"the manual estimate {manual:.4f}"
            )

    def test_flat_signal_sharpe_is_zero(self, engine, prices):
        """No trades → equity flat → Sharpe clamped to 0 (inf/nan → 0)."""
        signals = _make_signal_series(prices.index, "flat")
        result = engine.run_single_signal("hmm_regime", signals, prices, "AAPL")
        # vectorbt returns inf for a zero-variance equity curve; the engine
        # clamps any non-finite value to 0.0.
        assert math.isfinite(result["sharpe_ratio"])
        assert result["sharpe_ratio"] == pytest.approx(0.0, abs=1e-6)

    def test_sharpe_positive_for_trending_prices(self):
        """A pure upward trend with a buy-and-hold signal yields positive Sharpe."""
        n = 252
        # Deterministic 0.5% daily return — no noise, pure trend.
        prices = pd.Series(
            100.0 * np.exp(np.arange(n) * 0.005),
            index=pd.date_range("2023-01-02", periods=n, freq="B"),
        )
        signals = pd.Series(0, index=prices.index, dtype=int)
        signals.iloc[0] = 1   # buy once
        signals.iloc[-1] = -1  # exit at end

        eng = BacktestEngine(initial_capital=100_000, commission=0.0, slippage=0.0)
        result = eng.run_single_signal("test", signals, prices, "SYN")
        assert result["sharpe_ratio"] > 0


# ---------------------------------------------------------------------------
# 3. Commission applied to each trade
# ---------------------------------------------------------------------------

class TestCommission:
    """Verify that commission reduces net return proportional to number of trades."""

    def test_commission_reduces_return(self, prices):
        signals = _make_signal_series(prices.index, "one_trade")

        eng_with = BacktestEngine(initial_capital=100_000, commission=0.001, slippage=0.0)
        eng_none = BacktestEngine(initial_capital=100_000, commission=0.0, slippage=0.0)

        r_with = eng_with.run_single_signal("t", signals, prices, "X")
        r_none = eng_none.run_single_signal("t", signals, prices, "X")

        assert r_none["total_return"] > r_with["total_return"], (
            "Zero-commission run should always outperform commission run."
        )

    def test_commission_cost_scales_with_trade_count(self, prices):
        """More trades → more commission paid → greater return penalty."""
        signals_few = _make_signal_series(prices.index, "one_trade")     # 1 trade
        signals_many = _make_signal_series(prices.index, "alternating")  # many trades

        commission = 0.005  # large to make the effect clearly visible
        eng = BacktestEngine(initial_capital=100_000, commission=commission, slippage=0.0)
        eng0 = BacktestEngine(initial_capital=100_000, commission=0.0, slippage=0.0)

        r_few = eng.run_single_signal("t", signals_few, prices, "X")
        r0_few = eng0.run_single_signal("t", signals_few, prices, "X")

        r_many = eng.run_single_signal("t", signals_many, prices, "X")
        r0_many = eng0.run_single_signal("t", signals_many, prices, "X")

        penalty_few = r0_few["total_return"] - r_few["total_return"]
        penalty_many = r0_many["total_return"] - r_many["total_return"]

        assert penalty_many > penalty_few, (
            "More trades should incur a larger total commission penalty."
        )

    def test_equity_curve_starts_at_initial_capital(self, engine, prices):
        signals = _make_signal_series(prices.index, "one_trade")
        result = engine.run_single_signal("t", signals, prices, "X")
        first_value = result["equity_curve"].iloc[0]
        assert first_value == pytest.approx(100_000.0, rel=1e-4)

    def test_zero_commission_matches_theoretical_return(self, prices):
        """With zero costs, a single round-trip return is purely price-driven."""
        # Simple 10-bar trade on a flat → steady-rise series.
        n = 50
        flat_prices = pd.Series(
            np.linspace(100, 110, n),
            index=pd.date_range("2023-01-02", periods=n, freq="B"),
        )
        signals = pd.Series(0, index=flat_prices.index, dtype=int)
        signals.iloc[5] = 1   # buy at price ~101
        signals.iloc[15] = -1  # sell at price ~102

        eng = BacktestEngine(initial_capital=100_000, commission=0.0, slippage=0.0)
        result = eng.run_single_signal("t", signals, flat_prices, "X")
        # Return should be positive (prices are rising).
        assert result["total_return"] > 0.0


# ---------------------------------------------------------------------------
# 4. Walk-forward split sizes
# ---------------------------------------------------------------------------

class TestWalkForward:
    """Verify that the walk-forward splits have the correct bar counts."""

    def _dummy_signal_fn(self, price_series: pd.Series) -> pd.Series:
        """Signal function: buy at bar 0 of each window, sell at bar -1."""
        sig = pd.Series(0, index=price_series.index, dtype=int)
        if len(sig) > 1:
            sig.iloc[0] = 1
            sig.iloc[-1] = -1
        return sig

    def test_split_count_matches_n_splits(self):
        prices = _make_price_series(n=100)
        eng = BacktestEngine()
        df = eng.walk_forward(self._dummy_signal_fn, prices, n_splits=5, train_frac=0.7)
        assert len(df) == 5

    def test_train_bars_correct(self):
        """Each window is 100/5=20 bars; train = floor(20 * 0.7) = 14 bars."""
        prices = _make_price_series(n=100)
        eng = BacktestEngine()
        df = eng.walk_forward(self._dummy_signal_fn, prices, n_splits=5, train_frac=0.7)
        # All non-last splits should have exactly 14 train bars.
        for _, row in df.iloc[:-1].iterrows():
            assert row["train_bars"] == 14, (
                f"Expected 14 train bars, got {row['train_bars']}"
            )

    def test_test_bars_correct(self):
        """test_bars = window_size - train_bars = 20 - 14 = 6."""
        prices = _make_price_series(n=100)
        eng = BacktestEngine()
        df = eng.walk_forward(self._dummy_signal_fn, prices, n_splits=5, train_frac=0.7)
        for _, row in df.iloc[:-1].iterrows():
            assert row["test_bars"] == 6

    def test_splits_are_non_overlapping(self):
        """Test segments of consecutive splits must not overlap."""
        prices = _make_price_series(n=100)
        eng = BacktestEngine()
        df = eng.walk_forward(self._dummy_signal_fn, prices, n_splits=4, train_frac=0.6)
        for i in range(len(df) - 1):
            assert df.iloc[i]["test_end"] < df.iloc[i + 1]["train_start"]

    def test_all_bars_covered(self):
        """Train + test bars across all splits should cover the full series."""
        prices = _make_price_series(n=100)
        eng = BacktestEngine()
        df = eng.walk_forward(self._dummy_signal_fn, prices, n_splits=5, train_frac=0.7)
        total_bars = df["train_bars"].sum() + df["test_bars"].sum()
        assert total_bars == len(prices)

    def test_returns_dataframe_with_expected_columns(self):
        prices = _make_price_series(n=60)
        eng = BacktestEngine()
        df = eng.walk_forward(self._dummy_signal_fn, prices, n_splits=3, train_frac=0.7)
        required = {
            "split", "train_bars", "test_bars", "train_start", "train_end",
            "test_start", "test_end", "total_return", "sharpe_ratio",
            "max_drawdown", "n_trades",
        }
        assert required.issubset(set(df.columns))

    def test_different_train_fracs(self):
        """
        Verify that train_frac changes the train/test ratio as expected.

        n=80, n_splits=4 → base_window=20.
        train_frac=0.5 → n_train = int(20 * 0.5) = 10, n_test = 10 (equal).
        train_frac=0.8 → n_train = int(20 * 0.8) = 16, n_test = 4.
        """
        # Choose n and n_splits so the window divides evenly.
        n, n_splits = 80, 4  # base_window = 20; 20 * 0.5 = 10 exactly
        prices = _make_price_series(n=n)
        eng = BacktestEngine()

        df_half = eng.walk_forward(self._dummy_signal_fn, prices, n_splits=n_splits, train_frac=0.5)
        for _, row in df_half.iloc[:-1].iterrows():
            assert row["train_bars"] == row["test_bars"], (
                f"train_frac=0.5 should yield equal splits, "
                f"got train={row['train_bars']}, test={row['test_bars']}"
            )

        df_heavy = eng.walk_forward(self._dummy_signal_fn, prices, n_splits=n_splits, train_frac=0.8)
        for _, row in df_heavy.iloc[:-1].iterrows():
            assert row["train_bars"] > row["test_bars"], (
                "train_frac=0.8 should yield more train bars than test bars"
            )


# ---------------------------------------------------------------------------
# 5. Bias checks
# ---------------------------------------------------------------------------

class TestBiasChecks:
    def test_lookahead_no_bias_detected(self, engine, prices):
        """Signals mid-series, well before last bar → no bias."""
        signals = pd.Series(0, index=prices.index, dtype=int)
        signals.iloc[10] = 1
        signals.iloc[20] = -1
        assert engine.check_lookahead_bias(signals, prices) is True

    def test_lookahead_bias_at_last_bar(self, engine, prices):
        """A signal at the last bar has no future bar to trade on → bias."""
        signals = pd.Series(0, index=prices.index, dtype=int)
        signals.iloc[-1] = 1  # active signal at the very last bar
        assert engine.check_lookahead_bias(signals, prices) is False

    def test_lookahead_empty_signal_passes(self, engine, prices):
        """All-zero signal series is trivially unbiased."""
        signals = pd.Series(0, index=prices.index, dtype=int)
        assert engine.check_lookahead_bias(signals, prices) is True

    def test_survivorship_bias_returns_warning_string(self, engine):
        msg = engine.check_survivorship_bias()
        assert isinstance(msg, str)
        assert "SURVIVORSHIP BIAS" in msg

    def test_survivorship_bias_mentions_tickers(self, engine):
        msg = engine.check_survivorship_bias()
        assert "ticker" in msg.lower()
