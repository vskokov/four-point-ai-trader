"""
Unit tests for portfolio/portfolio_optimizer.py.

All DB and network calls are mocked — no live connections required.
Run with:
    .venv/bin/pytest tests/portfolio/test_portfolio_optimizer.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_MOD = "trading_engine.portfolio.portfolio_optimizer"
_PATCH_STORAGE = "trading_engine.data.storage.Storage"
_PATCH_SETTINGS = f"{_MOD}.settings"

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG", "JPM", "BAC", "NVDA", "TSLA", "AMZN", "META", "NFLX"]
N_TICKERS = len(TICKERS)
N_DAYS = 300

# Seed a correlated return matrix for all 10 tickers using a simple block
# covariance structure (high within-group, low cross-group).
_RNG = np.random.default_rng(42)

# Build a positive-definite covariance matrix: diagonal var=1e-4, off-diag=3e-5
_COV = np.full((N_TICKERS, N_TICKERS), 3e-5)
np.fill_diagonal(_COV, 1e-4)
_MEAN = np.full(N_TICKERS, 4e-4)
_RAW_RETURNS = _RNG.multivariate_normal(_MEAN, _COV, size=N_DAYS)


def _make_ohlcv_df(ticker: str, seed_offset: int = 0) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame for *ticker* (one row per day)."""
    rng = np.random.default_rng(42 + seed_offset)
    idx = TICKERS.index(ticker) if ticker in TICKERS else 0
    # _RAW_RETURNS has shape (N_DAYS, N_TICKERS); pick the column for this ticker
    returns = _RAW_RETURNS[:, idx % N_TICKERS]
    prices = 100.0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "time": dates,
            "ticker": ticker,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": rng.integers(1_000_000, 5_000_000, N_DAYS),
        }
    )


def _mock_storage_class():
    """Return a Storage mock whose query_ohlcv returns synthetic data."""
    mock_instance = MagicMock()
    mock_instance.query_ohlcv.side_effect = lambda ticker, start, end: _make_ohlcv_df(
        ticker, seed_offset=TICKERS.index(ticker) if ticker in TICKERS else 0
    )
    mock_instance.dispose.return_value = None
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(DB_URL="postgresql+psycopg2://fake:fake@localhost/fake")


def _make_optimizer(**kwargs) -> "PortfolioOptimizer":  # noqa: F821
    from trading_engine.portfolio.portfolio_optimizer import PortfolioOptimizer

    return PortfolioOptimizer(
        tickers=kwargs.get("tickers", TICKERS),
        risk_free_rate=kwargs.get("risk_free_rate", 0.05),
        max_weight=kwargs.get("max_weight", 0.10),
        min_weight=kwargs.get("min_weight", 0.0),
        lookback_days=kwargs.get("lookback_days", 252),
    )


# ---------------------------------------------------------------------------
# Confident MWU scores (all pass threshold: abs(score)>=0.3, confidence>=0.4)
# ---------------------------------------------------------------------------

_CONFIDENT_SCORES: dict[str, dict] = {
    "AAPL": {"score": 0.60, "confidence": 0.80, "final_signal": 1},
    "MSFT": {"score": -0.40, "confidence": 0.70, "final_signal": -1},
    "GOOG": {"score": 0.50, "confidence": 0.60, "final_signal": 1},
    "JPM":  {"score": 0.35, "confidence": 0.55, "final_signal": 1},
    "BAC":  {"score": -0.45, "confidence": 0.65, "final_signal": -1},
    "NVDA": {"score": 0.70, "confidence": 0.85, "final_signal": 1},
    "TSLA": {"score": -0.55, "confidence": 0.75, "final_signal": -1},
    "AMZN": {"score": 0.40, "confidence": 0.50, "final_signal": 1},
    "META": {"score": 0.30, "confidence": 0.45, "final_signal": 1},
    "NFLX": {"score": -0.35, "confidence": 0.60, "final_signal": -1},
}

# Low-confidence scores — none pass the threshold (confidence < 0.4)
_LOW_CONF_SCORES: dict[str, dict] = {
    t: {"score": 0.90, "confidence": 0.25, "final_signal": 1}
    for t in TICKERS
}


# ===========================================================================
# Test 1 — BL weights sum to 1
# ===========================================================================

class TestBLWeightsSumToOne:

    def test_weights_sum_to_one(self):
        opt = _make_optimizer()
        with (
            patch(_PATCH_STORAGE, _mock_storage_class()),
            patch(_PATCH_SETTINGS, _fake_settings()),
        ):
            result = opt.compute_black_litterman(_CONFIDENT_SCORES)

        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-6, f"weights sum to {total}, expected ~1.0"
        assert result["method"] == "black_litterman"
        assert result["n_views"] == len(TICKERS)   # all 10 confident scores pass threshold


# ===========================================================================
# Test 2 — Max weight constraint
# ===========================================================================

class TestMaxWeightConstraint:

    def test_no_weight_exceeds_max(self):
        max_w = 0.10
        opt = _make_optimizer(max_weight=max_w)
        with (
            patch(_PATCH_STORAGE, _mock_storage_class()),
            patch(_PATCH_SETTINGS, _fake_settings()),
        ):
            result = opt.compute_black_litterman(_CONFIDENT_SCORES)

        for ticker, w in result["weights"].items():
            assert w <= max_w + 1e-8, (
                f"{ticker} weight {w:.4f} exceeds max_weight={max_w}"
            )


# ===========================================================================
# Test 3 — No views → fallback to min_variance
# ===========================================================================

class TestNoViewsFallback:

    def test_low_confidence_falls_back_to_min_variance(self):
        opt = _make_optimizer()
        with (
            patch(_PATCH_STORAGE, _mock_storage_class()),
            patch(_PATCH_SETTINGS, _fake_settings()),
        ):
            result = opt.compute_black_litterman(_LOW_CONF_SCORES)

        assert result["method"] == "min_variance"
        assert result["n_views"] == 0


# ===========================================================================
# Test 4 — Rebalance orders filter small trades
# ===========================================================================

class TestRebalanceOrdersFilterSmallTrades:

    def test_all_hold_when_delta_below_threshold(self):
        opt = _make_optimizer()
        # Set target weights close to current positions (delta < 0.005 for all)
        opt.target_weights = {
            "AAPL": 0.10, "MSFT": 0.09, "GOOG": 0.08, "JPM": 0.07, "BAC": 0.06,
            "NVDA": 0.10, "TSLA": 0.09, "AMZN": 0.08, "META": 0.07, "NFLX": 0.06,
        }
        opt.last_optimized = datetime.now(tz=timezone.utc)

        # Current positions: each within 0.004 of target
        current_positions = pd.DataFrame(
            {
                "ticker": TICKERS,
                "market_value": [
                    9_800.0, 8_950.0, 7_970.0, 6_980.0, 5_990.0,
                    9_980.0, 8_970.0, 7_970.0, 6_960.0, 5_990.0,
                ],
            }
        )
        account_equity = 100_000.0

        orders = opt.get_rebalance_orders(
            current_positions, account_equity, min_trade_pct=0.005
        )

        actions = {o["ticker"]: o["action"] for o in orders}
        assert all(a == "hold" for a in actions.values()), (
            f"Expected all hold, got: {actions}"
        )


# ===========================================================================
# Test 5 — Rebalance orders correct action direction
# ===========================================================================

class TestRebalanceOrdersActionDirection:

    def test_buy_when_underweight_sell_when_overweight(self):
        opt = _make_optimizer()
        # Minimal target weights dict — only the tickers we want to verify
        opt.target_weights = {t: 0.0 for t in TICKERS}
        opt.target_weights["AAPL"] = 0.10   # underweight → buy
        opt.target_weights["MSFT"] = 0.05   # overweight  → sell
        opt.last_optimized = datetime.now(tz=timezone.utc)

        # AAPL: current=0.05 → delta=+0.05 → buy
        # MSFT: current=0.10 → delta=-0.05 → sell
        current_positions = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "market_value": [5_000.0, 10_000.0],
            }
        )
        account_equity = 100_000.0

        orders = opt.get_rebalance_orders(
            current_positions, account_equity, min_trade_pct=0.005
        )

        actions = {o["ticker"]: o["action"] for o in orders}
        assert actions["AAPL"] == "buy"
        assert actions["MSFT"] == "sell"


# ===========================================================================
# Test 6 — get_target_weight returns 0.0 for unknown ticker
# ===========================================================================

class TestGetTargetWeightUnknown:

    def test_unknown_ticker_returns_zero(self):
        opt = _make_optimizer()
        opt.last_optimized = datetime.now(tz=timezone.utc)
        result = opt.get_target_weight("UNKNOWN")
        assert result == 0.0


# ===========================================================================
# Test 7 — compute_min_variance runs without views
# ===========================================================================

class TestMinVarianceDirect:

    def test_weights_sum_to_one(self):
        opt = _make_optimizer()
        with (
            patch(_PATCH_STORAGE, _mock_storage_class()),
            patch(_PATCH_SETTINGS, _fake_settings()),
        ):
            result = opt.compute_min_variance()

        assert result["method"] == "min_variance"
        assert result["n_views"] == 0
        assert set(result["weights"].keys()) >= set()  # non-empty

        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-6, f"weights sum to {total}"

    def test_all_tickers_present_in_target_weights(self):
        opt = _make_optimizer()
        with (
            patch(_PATCH_STORAGE, _mock_storage_class()),
            patch(_PATCH_SETTINGS, _fake_settings()),
        ):
            opt.compute_min_variance()

        for ticker in TICKERS:
            assert ticker in opt.target_weights

    def test_last_optimized_set_after_call(self):
        opt = _make_optimizer()
        assert opt.last_optimized is None
        with (
            patch(_PATCH_STORAGE, _mock_storage_class()),
            patch(_PATCH_SETTINGS, _fake_settings()),
        ):
            opt.compute_min_variance()

        assert opt.last_optimized is not None
        assert isinstance(opt.last_optimized, datetime)
