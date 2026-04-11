"""
Unit tests for analysis/parameter_sweep.py.

All tests use synthetic DataFrames — no DB required.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading_engine.analysis.parameter_sweep import (
    sweep_entry_z,
    sweep_eta,
    sweep_hours_back,
    sweep_min_confidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a labeled decisions DataFrame with sensible defaults."""
    defaults = {
        "time":              pd.Timestamp("2026-01-02 10:00:00", tz="UTC"),
        "ticker":            "AAPL",
        "final_signal":      1,
        "score":             0.5,
        "regime":            1,
        "hmm_signal":        0,
        "ou_signal":         0,
        "ou_zscore":         0.0,
        "llm_signal":        0,
        "llm_confidence":    0.0,
        "contributing_headlines": [],
        "fwd_ret_1m":        0.0,
        "fwd_ret_15m":       0.01,
        "fwd_ret_1h":        0.01,
        "fwd_ret_4h":        0.01,
        "correct_1m":        True,
        "correct_15m":       True,
        "correct_1h":        True,
        "correct_4h":        True,
    }
    records = []
    for i, row in enumerate(rows):
        r = dict(defaults)
        r["time"] = pd.Timestamp("2026-01-02 10:00:00", tz="UTC") + pd.Timedelta(minutes=i)
        r.update(row)
        records.append(r)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# sweep_hours_back
# ---------------------------------------------------------------------------


class TestSweepHoursBack:
    def _headline(self, age_hours: float, decision_time: pd.Timestamp) -> dict:
        pub = decision_time - pd.Timedelta(hours=age_hours)
        return {"title": "test", "published_at": pub.isoformat()}

    def test_returns_one_row_per_cutoff(self) -> None:
        t = pd.Timestamp("2026-01-02 10:00:00", tz="UTC")
        df = _make_df([{
            "llm_signal": 1,
            "contributing_headlines": [self._headline(0.5, t)],
        }] * 5)
        result = sweep_hours_back(df, cutoffs_h=[1.0, 2.0])
        assert len(result) == 2

    def test_fresh_headlines_classified_correctly(self) -> None:
        # Headlines 30 min old → "fresh" for cutoff=1h, 2h, 4h, 8h
        t = pd.Timestamp("2026-01-02 10:00:00", tz="UTC")
        df = _make_df([{
            "llm_signal": 1, "fwd_ret_15m": 0.01,
            "contributing_headlines": [self._headline(0.5, t)],
        }] * 5)
        result = sweep_hours_back(df, cutoffs_h=[1.0, 2.0])
        # All 5 trades should be "fresh" for both cutoffs
        assert result.iloc[0]["n_fresh"] == 5
        assert result.iloc[1]["n_fresh"] == 5

    def test_old_headlines_classified_as_mixed(self) -> None:
        # Headlines 3 hours old → "mixed" for cutoff=2h, "fresh" for cutoff=4h
        t = pd.Timestamp("2026-01-02 10:00:00", tz="UTC")
        df = _make_df([{
            "llm_signal": 1, "fwd_ret_15m": 0.01,
            "contributing_headlines": [self._headline(3.0, t)],
        }] * 5)
        result = sweep_hours_back(df, cutoffs_h=[2.0, 4.0])
        # cutoff=2h: oldest=3h > 2h → mixed
        assert result.iloc[0]["n_mixed"] == 5
        assert result.iloc[0]["n_fresh"] == 0
        # cutoff=4h: oldest=3h <= 4h → fresh
        assert result.iloc[1]["n_fresh"] == 5
        assert result.iloc[1]["n_mixed"] == 0

    def test_empty_df_returns_empty(self) -> None:
        result = sweep_hours_back(pd.DataFrame(), cutoffs_h=[1.0, 2.0])
        assert result.empty

    def test_no_llm_signals_returns_empty(self) -> None:
        df = _make_df([{"llm_signal": 0}] * 5)
        result = sweep_hours_back(df, cutoffs_h=[1.0, 2.0])
        assert result.empty

    def test_missing_headlines_treated_as_nan(self) -> None:
        # Rows with empty contributing_headlines get NaN age → mixed for all cutoffs
        df = _make_df([{"llm_signal": 1, "contributing_headlines": []}] * 5)
        result = sweep_hours_back(df, cutoffs_h=[1.0, 2.0])
        # n_mixed should equal n rows (NaN age > any cutoff)
        assert result.iloc[0]["n_mixed"] == 5


# ---------------------------------------------------------------------------
# sweep_entry_z
# ---------------------------------------------------------------------------


class TestSweepEntryZ:
    def test_returns_one_row_per_threshold(self) -> None:
        df = _make_df([{"ou_zscore": 2.5, "fwd_ret_15m": -0.01}] * 5)
        result = sweep_entry_z(df, z_thresholds=[1.5, 2.0, 2.5, 3.0])
        assert len(result) == 4

    def test_fewer_trades_at_higher_threshold(self) -> None:
        rows = (
            [{"ou_zscore": 1.6, "fwd_ret_15m": -0.01}] * 4 +
            [{"ou_zscore": 2.5, "fwd_ret_15m": -0.01}] * 3
        )
        df = _make_df(rows)
        result = sweep_entry_z(df, z_thresholds=[1.5, 2.5])
        n_1_5 = result.loc[result["entry_z"] == 1.5, "n_trades"].iloc[0]
        n_2_5 = result.loc[result["entry_z"] == 2.5, "n_trades"].iloc[0]
        assert n_1_5 > n_2_5

    def test_mean_reversion_direction(self) -> None:
        # Positive z → short signal (-1); if price falls, correct
        df = _make_df([{"ou_zscore": 2.5, "fwd_ret_15m": -0.01}] * 5)
        result = sweep_entry_z(df, z_thresholds=[2.0])
        assert abs(result.iloc[0]["win_rate_15m"] - 1.0) < 1e-6

    def test_negative_z_long_signal(self) -> None:
        # Negative z → long signal (+1); if price rises, correct
        df = _make_df([{"ou_zscore": -2.5, "fwd_ret_15m": 0.01}] * 5)
        result = sweep_entry_z(df, z_thresholds=[2.0])
        assert abs(result.iloc[0]["win_rate_15m"] - 1.0) < 1e-6

    def test_below_threshold_not_fired(self) -> None:
        df = _make_df([{"ou_zscore": 1.0, "fwd_ret_15m": 0.01}] * 5)
        result = sweep_entry_z(df, z_thresholds=[2.0])
        assert result.iloc[0]["n_trades"] == 0

    def test_empty_df_returns_empty(self) -> None:
        result = sweep_entry_z(pd.DataFrame())
        assert result.empty

    def test_all_horizons_in_result(self) -> None:
        df = _make_df([{"ou_zscore": 2.5, "fwd_ret_15m": -0.01}] * 5)
        result = sweep_entry_z(df, z_thresholds=[2.0])
        for h in ("1m", "15m", "1h", "4h"):
            assert f"win_rate_{h}" in result.columns


# ---------------------------------------------------------------------------
# sweep_min_confidence
# ---------------------------------------------------------------------------


class TestSweepMinConfidence:
    def test_returns_one_row_per_threshold(self) -> None:
        df = _make_df([{"score": 0.5}] * 5)
        result = sweep_min_confidence(df, thresholds=[0.3, 0.4, 0.5])
        assert len(result) == 3

    def test_higher_threshold_suppresses_more_trades(self) -> None:
        rows = [{"score": 0.35}] * 3 + [{"score": 0.55}] * 4
        df = _make_df(rows)
        result = sweep_min_confidence(df, thresholds=[0.30, 0.50])
        n_30 = result.loc[result["threshold"] == 0.30, "n_active"].iloc[0]
        n_50 = result.loc[result["threshold"] == 0.50, "n_active"].iloc[0]
        assert n_30 > n_50

    def test_pct_suppressed_increases_with_threshold(self) -> None:
        rows = [{"score": 0.35}] * 4 + [{"score": 0.65}] * 4
        df = _make_df(rows)
        result = sweep_min_confidence(df, thresholds=[0.30, 0.60])
        pct_30 = result.loc[result["threshold"] == 0.30, "pct_suppressed"].iloc[0]
        pct_60 = result.loc[result["threshold"] == 0.60, "pct_suppressed"].iloc[0]
        assert pct_60 > pct_30

    def test_accuracy_computed_correctly(self) -> None:
        # All decisions above threshold and all correct
        df = _make_df([{"score": 0.6, "correct_15m": True, "fwd_ret_15m": 0.01}] * 5)
        result = sweep_min_confidence(df, thresholds=[0.5])
        assert abs(result.iloc[0]["win_rate_15m"] - 1.0) < 1e-6

    def test_empty_df_returns_empty(self) -> None:
        result = sweep_min_confidence(pd.DataFrame())
        assert result.empty

    def test_zero_score_rows_excluded_from_total(self) -> None:
        # score=0 rows should not count as potentially tradeable
        rows = [{"score": 0.0}] * 10 + [{"score": 0.5}] * 5
        df   = _make_df(rows)
        result = sweep_min_confidence(df, thresholds=[0.3])
        # total denominator should be 5 (only scored decisions)
        assert result.iloc[0]["n_active"] == 5


# ---------------------------------------------------------------------------
# sweep_eta
# ---------------------------------------------------------------------------


class TestSweepEta:
    def _make_seq_df(self, n: int = 20) -> pd.DataFrame:
        """Make a sequence of alternating correct decisions with known regimes."""
        rows = []
        for i in range(n):
            rows.append({
                "time":        pd.Timestamp("2026-01-02 10:00:00", tz="UTC") + pd.Timedelta(minutes=i),
                "ticker":      "AAPL",
                "regime":      1,
                "hmm_signal":  1,
                "ou_signal":   1,
                "llm_signal":  1,
                "score":       0.5,
                "fwd_ret_15m": 0.01,   # all positive → all correct
            })
        return pd.DataFrame(rows)

    def test_returns_one_row_per_eta(self) -> None:
        df     = self._make_seq_df()
        result = sweep_eta(df, eta_values=[0.01, 0.1, 0.5])
        assert len(result) == 3

    def test_n_updates_matches_sequence_length(self) -> None:
        df     = self._make_seq_df(20)
        result = sweep_eta(df, eta_values=[0.1])
        # Every row has fwd_ret_15m != 0 → all are updates
        assert result.iloc[0]["n_updates"] == 20

    def test_final_accuracy_between_0_and_1(self) -> None:
        df     = self._make_seq_df(20)
        result = sweep_eta(df, eta_values=[0.01, 0.1, 0.5])
        for _, row in result.iterrows():
            acc = row["final_accuracy_15m"]
            if acc == acc:  # not NaN
                assert 0.0 <= acc <= 1.0

    def test_insufficient_data_returns_empty(self) -> None:
        df     = self._make_seq_df(5)  # < 10 rows
        result = sweep_eta(df, eta_values=[0.1])
        assert result.empty

    def test_weight_range_positive(self) -> None:
        df     = self._make_seq_df(20)
        result = sweep_eta(df, eta_values=[0.1])
        assert result.iloc[0]["weight_range"] >= 0.0

    def test_high_eta_larger_weight_range(self) -> None:
        # High eta should cause more weight divergence than low eta
        df     = self._make_seq_df(30)
        result = sweep_eta(df, eta_values=[0.01, 0.5])
        wr_low  = result.loc[result["eta"] == 0.01, "weight_range"].iloc[0]
        wr_high = result.loc[result["eta"] == 0.50, "weight_range"].iloc[0]
        assert wr_high >= wr_low
