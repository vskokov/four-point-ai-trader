"""
Unit tests for analysis/signal_quality.py.

All tests use synthetic DataFrames — no DB required.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading_engine.analysis.signal_quality import (
    compute_ensemble_accuracy,
    compute_signal_accuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labeled_df(rows: list[dict]) -> pd.DataFrame:
    """Build a labeled DataFrame as output_labeler would produce."""
    defaults = {
        "time":           pd.Timestamp("2026-01-02 10:00:00", tz="UTC"),
        "ticker":         "AAPL",
        "regime":         1,  # neutral
        "regime_label":   "neutral",
        "score":          0.5,
        "hmm_signal":     0,
        "hmm_confidence": 0.0,
        "ou_signal":      0,
        "ou_confidence":  0.0,
        "ou_zscore":      0.0,
        "llm_signal":     0,
        "llm_confidence": 0.0,
        "final_signal":   0,
        "fwd_ret_1m":     0.0,
        "fwd_ret_15m":    0.0,
        "fwd_ret_1h":     0.0,
        "fwd_ret_4h":     0.0,
        "correct_1m":     None,
        "correct_15m":    None,
        "correct_1h":     None,
        "correct_4h":     None,
    }
    records = []
    for i, row in enumerate(rows):
        r = dict(defaults)
        r["time"] = pd.Timestamp("2026-01-02 10:00:00", tz="UTC") + pd.Timedelta(minutes=i)
        r.update(row)
        records.append(r)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# compute_signal_accuracy
# ---------------------------------------------------------------------------


class TestComputeSignalAccuracy:
    def test_perfect_accuracy_hmm(self) -> None:
        # hmm_signal always correct
        df = _make_labeled_df(
            [
                {"hmm_signal": 1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": 0.01, "correct_15m": True},
                {"hmm_signal": 1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": 0.02, "correct_15m": True},
                {"hmm_signal": -1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": -0.01, "correct_15m": True},
                {"hmm_signal": -1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": -0.02, "correct_15m": True},
                {"hmm_signal": 1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": 0.01, "correct_15m": True},
            ]
        )
        result = compute_signal_accuracy(df)
        assert not result.empty
        # win_rate_15m should be 1.0 (signal matches return direction every time)
        hmm_all = result.loc[("hmm", "all")]
        assert abs(hmm_all["win_rate_15m"] - 1.0) < 1e-6

    def test_zero_accuracy_hmm(self) -> None:
        # hmm_signal always wrong (signal +1 but negative return)
        df = _make_labeled_df(
            [
                {"hmm_signal": 1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": -0.01, "correct_15m": False},
                {"hmm_signal": 1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": -0.01, "correct_15m": False},
                {"hmm_signal": 1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": -0.01, "correct_15m": False},
                {"hmm_signal": 1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": -0.01, "correct_15m": False},
                {"hmm_signal": 1, "hmm_confidence": 0.8,
                 "fwd_ret_15m": -0.01, "correct_15m": False},
            ]
        )
        result = compute_signal_accuracy(df)
        # Win rate based on signal's own vote vs return should be 0.0
        hmm_all = result.loc[("hmm", "all")]
        assert abs(hmm_all["win_rate_15m"] - 0.0) < 1e-6

    def test_neutral_signal_rows_excluded_from_win_rate(self) -> None:
        # Mix of hmm_signal=0 (neutral) and hmm_signal=1 (directional, correct)
        df = _make_labeled_df(
            [
                {"hmm_signal": 0, "hmm_confidence": 0.0, "fwd_ret_15m": 0.01},
                {"hmm_signal": 0, "hmm_confidence": 0.0, "fwd_ret_15m": -0.01},
                {"hmm_signal": 1, "hmm_confidence": 0.8, "fwd_ret_15m": 0.02},
                {"hmm_signal": 1, "hmm_confidence": 0.8, "fwd_ret_15m": 0.02},
                {"hmm_signal": 1, "hmm_confidence": 0.8, "fwd_ret_15m": 0.02},
            ]
        )
        result = compute_signal_accuracy(df)
        hmm_all = result.loc[("hmm", "all")]
        # Only 3 directional rows, all correct
        assert abs(hmm_all["win_rate_15m"] - 1.0) < 1e-6
        assert hmm_all["n"] == 3

    def test_all_three_signals_in_result(self) -> None:
        df = _make_labeled_df(
            [
                {
                    "hmm_signal": 1, "hmm_confidence": 0.8,
                    "ou_signal":  1, "ou_confidence":  0.7,
                    "llm_signal": 1, "llm_confidence": 0.9,
                    "fwd_ret_15m": 0.01,
                }
            ] * 6  # 6 rows to pass the n>=5 threshold for regime segments
        )
        result = compute_signal_accuracy(df)
        signals_found = {sig for sig, _ in result.index}
        assert "hmm" in signals_found
        assert "ou"  in signals_found
        assert "llm" in signals_found

    def test_regime_segmentation(self) -> None:
        rows = []
        for i in range(6):
            rows.append({"hmm_signal": 1, "hmm_confidence": 0.8, "regime": 2,
                         "fwd_ret_15m": 0.01})
        for i in range(6):
            rows.append({"hmm_signal": 1, "hmm_confidence": 0.8, "regime": 0,
                         "fwd_ret_15m": -0.01})
        df = _make_labeled_df(rows)
        result = compute_signal_accuracy(df)
        assert ("hmm", "regime:bull") in result.index
        assert ("hmm", "regime:bear") in result.index

    def test_empty_df_returns_empty(self) -> None:
        df = pd.DataFrame(columns=["hmm_signal", "hmm_confidence", "fwd_ret_15m"])
        result = compute_signal_accuracy(df)
        assert result.empty

    def test_result_has_expected_columns(self) -> None:
        df = _make_labeled_df(
            [{"hmm_signal": 1, "hmm_confidence": 0.8, "fwd_ret_15m": 0.01}] * 5
        )
        result = compute_signal_accuracy(df)
        for h in ("1m", "15m", "1h", "4h"):
            assert f"win_rate_{h}" in result.columns
            assert f"ic_{h}" in result.columns


# ---------------------------------------------------------------------------
# compute_ensemble_accuracy
# ---------------------------------------------------------------------------


class TestComputeEnsembleAccuracy:
    def test_all_correct_returns_1(self) -> None:
        df = _make_labeled_df(
            [
                {"final_signal": 1, "score": 0.6,
                 "correct_15m": True, "fwd_ret_15m": 0.01},
            ] * 5
        )
        result = compute_ensemble_accuracy(df)
        assert abs(result.loc["all"]["win_rate_15m"] - 1.0) < 1e-6

    def test_all_incorrect_returns_0(self) -> None:
        df = _make_labeled_df(
            [
                {"final_signal": 1, "score": 0.6,
                 "correct_15m": False, "fwd_ret_15m": -0.01},
            ] * 5
        )
        result = compute_ensemble_accuracy(df)
        assert abs(result.loc["all"]["win_rate_15m"] - 0.0) < 1e-6

    def test_neutral_decisions_excluded(self) -> None:
        df = _make_labeled_df(
            [
                {"final_signal": 0, "score": 0.0,
                 "correct_15m": None, "fwd_ret_15m": 0.01},
                {"final_signal": 1, "score": 0.6,
                 "correct_15m": True, "fwd_ret_15m": 0.01},
                {"final_signal": 1, "score": 0.6,
                 "correct_15m": True, "fwd_ret_15m": 0.01},
            ] * 3
        )
        result = compute_ensemble_accuracy(df)
        # Should count only directional decisions
        assert result.loc["all"]["n"] == 6

    def test_score_bucket_segmentation(self) -> None:
        rows = [
            {"final_signal": 1, "score": 0.35, "correct_15m": True, "fwd_ret_15m": 0.01},
        ] * 5 + [
            {"final_signal": 1, "score": 0.65, "correct_15m": False, "fwd_ret_15m": -0.01},
        ] * 5
        df = _make_labeled_df(rows)
        result = compute_ensemble_accuracy(df)
        # Both buckets should appear
        idx = result.index.tolist()
        low_bucket  = [s for s in idx if "0.30" in s]
        high_bucket = [s for s in idx if "0.60" in s]
        assert len(low_bucket)  > 0
        assert len(high_bucket) > 0

    def test_all_horizons_in_result(self) -> None:
        df = _make_labeled_df(
            [{"final_signal": 1, "score": 0.5, "correct_15m": True,
              "fwd_ret_15m": 0.01}] * 5
        )
        result = compute_ensemble_accuracy(df)
        for h in ("1m", "15m", "1h", "4h"):
            assert f"win_rate_{h}" in result.columns
