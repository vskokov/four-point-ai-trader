"""
Unit tests for analysis/outcome_labeler.py.

All tests use synthetic DataFrames — no DB required.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading_engine.analysis.outcome_labeler import compute_outcome_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    final_signal: int,
    close_at: float,
    close_1m: float | None,
    close_15m: float | None = None,
    close_1h: float | None  = None,
    close_4h: float | None  = None,
) -> pd.DataFrame:
    """Build a minimal single-row DataFrame for compute_outcome_labels."""
    return pd.DataFrame(
        [
            {
                "time":         pd.Timestamp("2026-01-02 10:00:00", tz="UTC"),
                "ticker":       "AAPL",
                "final_signal": final_signal,
                "score":        0.5,
                "close_at":     close_at,
                "close_1m":     close_1m,
                "close_15m":    close_15m if close_15m is not None else close_1m,
                "close_1h":     close_1h  if close_1h  is not None else close_1m,
                "close_4h":     close_4h  if close_4h  is not None else close_1m,
            }
        ]
    )


# ---------------------------------------------------------------------------
# forward return computation
# ---------------------------------------------------------------------------


class TestForwardReturns:
    def test_positive_move_positive_return(self) -> None:
        df = _make_df(final_signal=1, close_at=100.0, close_1m=101.0)
        out = compute_outcome_labels(df)
        assert abs(out.iloc[0]["fwd_ret_1m"] - 0.01) < 1e-9

    def test_negative_move_negative_return(self) -> None:
        df = _make_df(final_signal=1, close_at=100.0, close_1m=99.0)
        out = compute_outcome_labels(df)
        assert abs(out.iloc[0]["fwd_ret_1m"] - (-0.01)) < 1e-9

    def test_all_four_horizons_computed(self) -> None:
        df = _make_df(
            final_signal=1,
            close_at=100.0,
            close_1m=101.0,
            close_15m=102.0,
            close_1h=103.0,
            close_4h=104.0,
        )
        out = compute_outcome_labels(df)
        assert abs(out.iloc[0]["fwd_ret_1m"]  - 0.01) < 1e-9
        assert abs(out.iloc[0]["fwd_ret_15m"] - 0.02) < 1e-9
        assert abs(out.iloc[0]["fwd_ret_1h"]  - 0.03) < 1e-9
        assert abs(out.iloc[0]["fwd_ret_4h"]  - 0.04) < 1e-9

    def test_missing_forward_close_gives_nan(self) -> None:
        df = _make_df(final_signal=1, close_at=100.0, close_1m=None)
        df["close_1m"] = float("nan")
        out = compute_outcome_labels(df)
        assert math.isnan(out.iloc[0]["fwd_ret_1m"])

    def test_missing_close_at_gives_nan(self) -> None:
        df = _make_df(final_signal=1, close_at=100.0, close_1m=101.0)
        df["close_at"] = float("nan")
        out = compute_outcome_labels(df)
        assert math.isnan(out.iloc[0]["fwd_ret_1m"])


# ---------------------------------------------------------------------------
# correctness labels
# ---------------------------------------------------------------------------


class TestCorrectnessLabels:
    def test_long_rising_is_correct(self) -> None:
        df = _make_df(final_signal=1, close_at=100.0, close_1m=101.0)
        out = compute_outcome_labels(df)
        assert out.iloc[0]["correct_1m"] is True or out.iloc[0]["correct_1m"] == True  # noqa: E712

    def test_long_falling_is_incorrect(self) -> None:
        df = _make_df(final_signal=1, close_at=100.0, close_1m=99.0)
        out = compute_outcome_labels(df)
        assert out.iloc[0]["correct_1m"] is False or out.iloc[0]["correct_1m"] == False  # noqa: E712

    def test_short_falling_is_correct(self) -> None:
        df = _make_df(final_signal=-1, close_at=100.0, close_1m=99.0)
        out = compute_outcome_labels(df)
        assert out.iloc[0]["correct_1m"] == True  # noqa: E712

    def test_short_rising_is_incorrect(self) -> None:
        df = _make_df(final_signal=-1, close_at=100.0, close_1m=101.0)
        out = compute_outcome_labels(df)
        assert out.iloc[0]["correct_1m"] == False  # noqa: E712

    def test_neutral_signal_correct_is_nan(self) -> None:
        df = _make_df(final_signal=0, close_at=100.0, close_1m=101.0)
        out = compute_outcome_labels(df)
        assert pd.isna(out.iloc[0]["correct_1m"])

    def test_all_correct_columns_present(self) -> None:
        df = _make_df(final_signal=1, close_at=100.0, close_1m=101.0)
        out = compute_outcome_labels(df)
        for h in ("1m", "15m", "1h", "4h"):
            assert f"correct_{h}" in out.columns

    def test_missing_forward_close_correct_is_nan(self) -> None:
        df = _make_df(final_signal=1, close_at=100.0, close_1m=None)
        df["close_1m"] = float("nan")
        out = compute_outcome_labels(df)
        assert pd.isna(out.iloc[0]["correct_1m"])

    def test_flat_return_is_incorrect(self) -> None:
        # price unchanged → fwd_ret = 0 → signal * 0 = 0 → not > 0 → False
        df = _make_df(final_signal=1, close_at=100.0, close_1m=100.0)
        out = compute_outcome_labels(df)
        assert out.iloc[0]["correct_1m"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# Multiple rows
# ---------------------------------------------------------------------------


class TestMultipleRows:
    def test_multiple_rows_processed_independently(self) -> None:
        rows = [
            {
                "time": pd.Timestamp("2026-01-02 10:00:00", tz="UTC"),
                "ticker": "AAPL",
                "final_signal": 1,
                "score": 0.5,
                "close_at": 100.0,
                "close_1m": 101.0,
                "close_15m": 101.0,
                "close_1h": 101.0,
                "close_4h": 101.0,
            },
            {
                "time": pd.Timestamp("2026-01-02 10:01:00", tz="UTC"),
                "ticker": "AAPL",
                "final_signal": -1,
                "score": -0.5,
                "close_at": 100.0,
                "close_1m": 99.0,
                "close_15m": 99.0,
                "close_1h": 99.0,
                "close_4h": 99.0,
            },
        ]
        df  = pd.DataFrame(rows)
        out = compute_outcome_labels(df)
        assert out.iloc[0]["correct_1m"] == True   # noqa: E712
        assert out.iloc[1]["correct_1m"] == True   # noqa: E712

    def test_input_df_not_mutated(self) -> None:
        df  = _make_df(final_signal=1, close_at=100.0, close_1m=101.0)
        original_cols = list(df.columns)
        compute_outcome_labels(df)
        assert list(df.columns) == original_cols
