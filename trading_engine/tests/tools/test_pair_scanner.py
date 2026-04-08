"""
Unit tests for trading_engine/tools/pair_scanner.py.

All Alpaca calls are mocked.  Synthetic price data is generated inline:
- Two genuinely cointegrated pairs (p2 = beta * p1 + OU_noise).
- The remaining tickers are independent random walks.

Run with:
    .venv/bin/pytest tests/tools/test_pair_scanner.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=42)
_N = 504
_DATES = pd.date_range("2023-01-01", periods=_N, freq="B", tz="UTC")


def _random_walk(seed_offset: int = 0, start: float = 100.0) -> np.ndarray:
    """Independent random walk starting at *start*."""
    rng = np.random.default_rng(seed=42 + seed_offset)
    return np.cumsum(rng.standard_normal(_N) * 0.5) + start


def _ou_noise(kappa: float = 0.15, scale: float = 0.05, seed_offset: int = 0) -> np.ndarray:
    """
    Stationary OU noise series with mean-reversion speed *kappa*.

    *scale* controls the per-step innovation magnitude relative to the base
    price series.  Use a small value (e.g. 0.05) so that the noise is tiny
    compared to the underlying price moves and does not degrade log-return
    correlation between the cointegrated pair.
    """
    rng = np.random.default_rng(seed=99 + seed_offset)
    noise = np.zeros(_N)
    for i in range(1, _N):
        noise[i] = (1 - kappa) * noise[i - 1] + rng.standard_normal() * scale
    return noise


def _make_close_df(
    n_random: int = 6,
    n_coint_pairs: int = 2,
    beta: float = 1.5,
) -> pd.DataFrame:
    """
    Build a wide close-price DataFrame.

    Columns: RW0..RW{n_random-1}, C0A, C0B [, C1A, C1B, ...]
    - RW* are independent random walks.
    - C*A and C*B are genuinely cointegrated: C*B = beta * C*A + OU_noise.
    """
    data: dict[str, np.ndarray] = {}

    for i in range(n_random):
        data[f"RW{i}"] = _random_walk(seed_offset=i)

    for j in range(n_coint_pairs):
        base = _random_walk(seed_offset=100 + j, start=200.0)
        noise = _ou_noise(kappa=0.20, seed_offset=j)
        data[f"C{j}A"] = base
        data[f"C{j}B"] = beta * base + noise + 10.0

    return pd.DataFrame(data, index=_DATES)


@pytest.fixture
def close_df() -> pd.DataFrame:
    return _make_close_df()


# ---------------------------------------------------------------------------
# Module imports (after fixture definitions to allow patching)
# ---------------------------------------------------------------------------

from trading_engine.tools.pair_scanner import (  # noqa: E402
    correlation_filter,
    cointegration_filter,
    filter_tickers,
    ou_filter,
    run_scan,
)


# ---------------------------------------------------------------------------
# filter_tickers
# ---------------------------------------------------------------------------

class TestFilterTickers:

    def test_keeps_complete_tickers(self, close_df):
        result = filter_tickers(close_df.copy())
        assert set(result.columns) == set(close_df.columns)

    def test_drops_ticker_with_too_many_nans(self, close_df):
        df = close_df.copy()
        # Inject >10 % NaN into one column
        n_nan = int(_N * 0.15)
        df.iloc[:n_nan, 0] = np.nan
        bad_col = df.columns[0]
        result = filter_tickers(df)
        assert bad_col not in result.columns

    def test_retains_tickers_with_few_nans(self, close_df):
        df = close_df.copy()
        # Only 5 % NaN — should survive
        n_nan = int(_N * 0.05)
        df.iloc[:n_nan, 0] = np.nan
        result = filter_tickers(df)
        assert df.columns[0] in result.columns


# ---------------------------------------------------------------------------
# correlation_filter
# ---------------------------------------------------------------------------

class TestCorrelationFilter:

    def test_cointegrated_pair_survives(self, close_df):
        """C0A/C0B and C1A/C1B are strongly correlated on log-returns."""
        surviving = correlation_filter(close_df, min_correlation=0.70)
        surviving_pairs = {(t1, t2) for t1, t2, _ in surviving}
        assert ("C0A", "C0B") in surviving_pairs
        assert ("C1A", "C1B") in surviving_pairs

    def test_uncorrelated_rw_pair_filtered(self, close_df):
        """RW0 and RW1 should have low log-return correlation and be pruned."""
        # Use a high threshold so only the cointegrated pairs survive
        surviving = correlation_filter(close_df, min_correlation=0.95)
        surviving_pairs = {(t1, t2) for t1, t2, _ in surviving}
        # Random walks with independent noise should not reach 0.95 correlation
        assert ("RW0", "RW1") not in surviving_pairs

    def test_count_matches_structure(self, close_df):
        surviving = correlation_filter(close_df, min_correlation=0.0)
        tickers = list(close_df.columns)
        n_total = len(tickers) * (len(tickers) - 1) // 2
        assert len(surviving) <= n_total

    def test_correlation_values_are_floats(self, close_df):
        surviving = correlation_filter(close_df, min_correlation=0.50)
        for _, _, corr in surviving:
            assert isinstance(corr, float)
            assert -1.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# cointegration_filter
# ---------------------------------------------------------------------------

class TestCointegrationFilter:

    def test_cointegrated_pairs_pass(self, close_df):
        """The two genuinely cointegrated pairs should survive EG test."""
        candidate_pairs = [
            ("C0A", "C0B", 0.99),
            ("C1A", "C1B", 0.99),
        ]
        results = cointegration_filter(close_df, candidate_pairs, max_pvalue=0.05)
        passing = {(r["ticker1"], r["ticker2"]) for r in results}
        assert ("C0A", "C0B") in passing
        assert ("C1A", "C1B") in passing

    def test_random_walk_pair_fails(self, close_df):
        """Two independent random walks should NOT be cointegrated."""
        candidate_pairs = [("RW0", "RW1", 0.5)]
        results = cointegration_filter(close_df, candidate_pairs, max_pvalue=0.05)
        passing = {(r["ticker1"], r["ticker2"]) for r in results}
        assert ("RW0", "RW1") not in passing

    def test_result_contains_required_keys(self, close_df):
        candidate_pairs = [("C0A", "C0B", 0.99)]
        results = cointegration_filter(close_df, candidate_pairs, max_pvalue=0.05)
        assert results  # at least one result
        required_keys = {
            "ticker1", "ticker2", "correlation",
            "eg_pvalue", "johansen_trace_stat", "beta_ols",
        }
        assert required_keys.issubset(results[0].keys())

    def test_empty_input_returns_empty(self, close_df):
        results = cointegration_filter(close_df, [], max_pvalue=0.05)
        assert results == []


# ---------------------------------------------------------------------------
# ou_filter (half-life filter)
# ---------------------------------------------------------------------------

class TestOuFilter:

    def _coint_result(self, t1: str, t2: str, close_df: pd.DataFrame) -> dict:
        """Helper: get cointegration result for a pair."""
        from trading_engine.signals.mean_reversion import CointegrationTest
        coint = CointegrationTest()
        r = coint.test(close_df[t1].astype(float), close_df[t2].astype(float))
        return {
            "ticker1": t1,
            "ticker2": t2,
            "correlation": 0.99,
            "eg_pvalue": round(r["eg_pvalue"], 6),
            "johansen_trace_stat": round(r["johansen_trace_stat"], 4),
            "beta_ols": round(r["beta_ols"], 6),
        }

    def test_cointegrated_pair_passes_default_window(self, close_df):
        """C0A/C0B has ~5 bar half-life due to kappa=0.20 — should pass 5–60 window."""
        pair_result = self._coint_result("C0A", "C0B", close_df)
        results = ou_filter(close_df, [pair_result], min_half_life=2.0, max_half_life=120.0)
        assert results, "expected C0A/C0B to pass ou_filter"

    def test_too_short_half_life_filtered_out(self):
        """
        A pair whose spread reverts extremely fast (half-life < min_half_life)
        must be excluded by the lower bound of the half-life filter.

        Construction: spread = AR(1) with coefficient b = 0.05, giving
        kappa = 1 - 0.05 = 0.95 and half_life = ln(2) / 0.95 ≈ 0.73 bars.
        OLS on 504 points reliably estimates b close to 0.05, so the
        computed half_life is well below the min_half_life=5 threshold.
        """
        rng = np.random.default_rng(77)
        n = 504
        dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")

        # Build a cointegrated pair where the spread has AR(1) coefficient ≈ 0.05
        p1 = np.cumsum(rng.standard_normal(n) * 0.5) + 200
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = 0.05 * spread[i - 1] + rng.standard_normal() * 0.5
        # p2 is p1 minus the fast-reverting spread (beta=1)
        p2 = p1 - spread

        df = pd.DataFrame({"FAST_A": p1, "FAST_B": p2}, index=dates)

        from trading_engine.signals.mean_reversion import CointegrationTest
        ct = CointegrationTest()
        cint = ct.test(df["FAST_A"].astype(float), df["FAST_B"].astype(float))
        pair_entry = {
            "ticker1": "FAST_A", "ticker2": "FAST_B",
            "correlation": 0.99,
            "eg_pvalue": round(cint["eg_pvalue"], 6),
            "johansen_trace_stat": round(cint["johansen_trace_stat"], 4),
            "beta_ols": round(cint["beta_ols"], 6),
        }

        # min_half_life=5 should filter out this pair (expected half_life < 2)
        results = ou_filter(df, [pair_entry], min_half_life=5.0, max_half_life=60.0)
        assert len(results) == 0, (
            f"expected fast-reverting pair (hl ≈ 0.7) to be filtered by "
            f"min_half_life=5.0; got: "
            f"{[r['half_life_bars'] for r in results]}"
        )

    def test_ou_result_contains_required_keys(self, close_df):
        pair_result = self._coint_result("C0A", "C0B", close_df)
        results = ou_filter(close_df, [pair_result], min_half_life=2.0, max_half_life=120.0)
        if results:
            required = {"ticker1", "ticker2", "half_life_bars", "kappa", "mu", "sigma"}
            assert required.issubset(results[0].keys())


# ---------------------------------------------------------------------------
# run_scan — integration (Alpaca mocked)
# ---------------------------------------------------------------------------

_SCANNER_MOD = "trading_engine.tools.pair_scanner"


class TestRunScan:

    def _make_mock_alpaca(self, close_df: pd.DataFrame) -> MagicMock:
        mock = MagicMock()
        mock.get_historical_bars.return_value = close_df
        return mock

    def test_max_pairs_cap(self, close_df, tmp_path):
        """With max_pairs=1 only the top-ranked pair is output."""
        output = tmp_path / "pairs.json"
        mock_alpaca = self._make_mock_alpaca(close_df)

        result = run_scan(
            tickers=list(close_df.columns),
            output=output,
            min_correlation=0.90,
            max_pvalue=0.05,
            min_half_life=2.0,
            max_half_life=120.0,
            max_pairs=1,
            _alpaca=mock_alpaca,
        )

        assert len(result["pairs"]) <= 1

    def test_json_output_structure(self, close_df, tmp_path):
        """Output file must be valid JSON with all required top-level keys."""
        output = tmp_path / "pairs.json"
        mock_alpaca = self._make_mock_alpaca(close_df)

        run_scan(
            tickers=list(close_df.columns),
            output=output,
            min_correlation=0.90,
            max_pvalue=0.05,
            min_half_life=2.0,
            max_half_life=120.0,
            max_pairs=10,
            _alpaca=mock_alpaca,
        )

        assert output.exists()
        data = json.loads(output.read_text())

        required_top = {
            "scanned_at", "lookback_days", "n_tickers_scanned",
            "n_candidate_pairs", "n_correlated", "n_cointegrated",
            "n_selected", "pairs",
        }
        assert required_top.issubset(data.keys())
        assert isinstance(data["pairs"], list)

    def test_json_pair_entry_keys(self, close_df, tmp_path):
        """Each pair entry must contain all required fields."""
        output = tmp_path / "pairs.json"
        mock_alpaca = self._make_mock_alpaca(close_df)

        run_scan(
            tickers=list(close_df.columns),
            output=output,
            min_correlation=0.90,
            max_pvalue=0.05,
            min_half_life=2.0,
            max_half_life=120.0,
            max_pairs=10,
            _alpaca=mock_alpaca,
        )

        data = json.loads(output.read_text())
        required_pair_keys = {
            "ticker1", "ticker2", "correlation", "eg_pvalue",
            "johansen_trace_stat", "beta_ols",
            "half_life_bars", "kappa", "mu", "sigma",
        }
        for pair in data["pairs"]:
            assert required_pair_keys.issubset(pair.keys()), (
                f"Missing keys in pair {pair}: "
                f"{required_pair_keys - pair.keys()}"
            )

    def test_scanned_at_is_valid_iso_timestamp(self, close_df, tmp_path):
        """scanned_at must be parseable as an ISO 8601 UTC timestamp."""
        output = tmp_path / "pairs.json"
        mock_alpaca = self._make_mock_alpaca(close_df)

        run_scan(
            tickers=list(close_df.columns),
            output=output,
            min_correlation=0.90,
            max_pvalue=0.05,
            min_half_life=2.0,
            max_half_life=120.0,
            _alpaca=mock_alpaca,
        )

        data = json.loads(output.read_text())
        scanned_at = data["scanned_at"]
        # Should parse without raising
        dt = datetime.fromisoformat(scanned_at.replace("Z", "+00:00"))
        assert dt.tzinfo is not None

    def test_empty_result_when_no_pairs_pass(self, tmp_path):
        """When no pairs pass the filters, pairs must be [] and no crash."""
        # Generate data where nothing is correlated at 0.99 threshold
        rng = np.random.default_rng(7)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame(
            {
                "A": np.cumsum(rng.standard_normal(n)) + 100,
                "B": np.cumsum(rng.standard_normal(n)) + 100,
            },
            index=dates,
        )

        output = tmp_path / "pairs.json"
        mock_alpaca = MagicMock()
        mock_alpaca.get_historical_bars.return_value = df

        result = run_scan(
            tickers=["A", "B"],
            output=output,
            min_correlation=0.99,  # impossibly high for independent RWs
            _alpaca=mock_alpaca,
        )

        assert result["pairs"] == []
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["pairs"] == []

    def test_ranked_by_eg_pvalue(self, close_df, tmp_path):
        """Pairs must be sorted by eg_pvalue ascending (best cointegration first)."""
        output = tmp_path / "pairs.json"
        mock_alpaca = self._make_mock_alpaca(close_df)

        result = run_scan(
            tickers=list(close_df.columns),
            output=output,
            min_correlation=0.90,
            max_pvalue=0.05,
            min_half_life=2.0,
            max_half_life=120.0,
            _alpaca=mock_alpaca,
        )

        pvalues = [p["eg_pvalue"] for p in result["pairs"]]
        assert pvalues == sorted(pvalues)

    def test_n_selected_matches_pairs_length(self, close_df, tmp_path):
        """n_selected in the result must match len(pairs)."""
        output = tmp_path / "pairs.json"
        mock_alpaca = self._make_mock_alpaca(close_df)

        result = run_scan(
            tickers=list(close_df.columns),
            output=output,
            min_correlation=0.90,
            max_pvalue=0.05,
            min_half_life=2.0,
            max_half_life=120.0,
            _alpaca=mock_alpaca,
        )

        assert result["n_selected"] == len(result["pairs"])
