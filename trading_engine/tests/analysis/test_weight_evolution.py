"""
Unit tests for analysis/weight_evolution.py.

All tests use synthetic DataFrames — no DB required.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading_engine.analysis.weight_evolution import (
    extract_weight_history,
    summarise_weight_evolution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal trade_log-style DataFrame."""
    defaults = {
        "time":         pd.Timestamp("2026-01-02 10:00:00", tz="UTC"),
        "ticker":       "AAPL",
        "regime":       1,
        "regime_label": "neutral",
        "final_signal": 1,
        "mwu_weights":  {
            "hmm_regime":    2 / 7,
            "ou_spread":     2 / 7,
            "llm_sentiment": 2 / 7,
            "analyst_recs":  1 / 7,
        },
    }
    records = []
    for i, row in enumerate(rows):
        r = dict(defaults)
        r["time"] = pd.Timestamp("2026-01-02 10:00:00", tz="UTC") + pd.Timedelta(minutes=i)
        r.update(row)
        records.append(r)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# extract_weight_history
# ---------------------------------------------------------------------------


class TestExtractWeightHistory:
    def test_basic_extraction(self) -> None:
        df = _make_df([{}])
        wdf = extract_weight_history(df)
        assert len(wdf) == 1
        assert abs(wdf.iloc[0]["hmm_regime"] - 2 / 7) < 1e-9

    def test_signal_columns_present(self) -> None:
        df = _make_df([{}])
        wdf = extract_weight_history(df)
        for sig in ("hmm_regime", "ou_spread", "llm_sentiment", "analyst_recs"):
            assert sig in wdf.columns

    def test_null_mwu_weights_skipped(self) -> None:
        df = _make_df([{"mwu_weights": None}, {"mwu_weights": None}])
        wdf = extract_weight_history(df)
        assert len(wdf) == 0

    def test_non_dict_weights_skipped(self) -> None:
        df = _make_df([{"mwu_weights": "bad"}, {}])
        wdf = extract_weight_history(df)
        assert len(wdf) == 1

    def test_output_sorted_by_time(self) -> None:
        rows = [
            {"time": pd.Timestamp("2026-01-02 11:00:00", tz="UTC")},
            {"time": pd.Timestamp("2026-01-02 09:00:00", tz="UTC")},
            {"time": pd.Timestamp("2026-01-02 10:00:00", tz="UTC")},
        ]
        df  = _make_df(rows)
        wdf = extract_weight_history(df)
        times = wdf["time"].tolist()
        assert times == sorted(times)

    def test_regime_label_derived(self) -> None:
        df  = _make_df([{"regime": 0}, {"regime": 2}])
        wdf = extract_weight_history(df)
        labels = set(wdf["regime_label"])
        assert "bear" in labels
        assert "bull" in labels

    def test_multiple_tickers(self) -> None:
        rows = [
            {"ticker": "AAPL"},
            {"ticker": "MSFT"},
        ]
        df  = _make_df(rows)
        wdf = extract_weight_history(df)
        assert set(wdf["ticker"]) == {"AAPL", "MSFT"}

    def test_empty_df_returns_empty(self) -> None:
        wdf = extract_weight_history(pd.DataFrame())
        assert wdf.empty

    def test_string_mwu_weights_skipped(self) -> None:
        # The DB returns JSON strings in some edge cases; these should not crash
        df = _make_df([{"mwu_weights": '{"hmm_regime": 0.3}'}])
        # A plain string is not a dict, so it should be skipped
        wdf = extract_weight_history(df)
        assert len(wdf) == 0


# ---------------------------------------------------------------------------
# summarise_weight_evolution
# ---------------------------------------------------------------------------


class TestSummariseWeightEvolution:
    def _make_wdf(self, final_weights: dict, ticker: str = "AAPL", regime: int = 1) -> pd.DataFrame:
        rows = [
            {
                "time":         pd.Timestamp("2026-01-02 10:00:00", tz="UTC"),
                "ticker":       ticker,
                "regime":       regime,
                "regime_label": {0: "bear", 1: "neutral", 2: "bull"}.get(regime, "neutral"),
                **final_weights,
            }
        ]
        return pd.DataFrame(rows)

    def test_per_ticker_regime_populated(self) -> None:
        wdf = self._make_wdf(
            {"hmm_regime": 0.3, "ou_spread": 0.3,
             "llm_sentiment": 0.3, "analyst_recs": 0.1}
        )
        summary = summarise_weight_evolution(wdf)
        assert ("AAPL", "neutral") in summary["per_ticker_regime"]

    def test_no_drift_below_threshold(self) -> None:
        # All weights at initial values → no drift flagged
        w = {"hmm_regime": 2/7, "ou_spread": 2/7, "llm_sentiment": 2/7, "analyst_recs": 1/7}
        wdf = self._make_wdf(w)
        summary = summarise_weight_evolution(wdf)
        assert len(summary["drifted_signals"]) == 0

    def test_large_drift_flagged(self) -> None:
        # hmm_regime drifted from 2/7≈0.286 to 0.5 — well above 20 %
        w = {"hmm_regime": 0.5, "ou_spread": 0.2, "llm_sentiment": 0.2, "analyst_recs": 0.1}
        wdf = self._make_wdf(w)
        summary = summarise_weight_evolution(wdf)
        drifted_sigs = {d["signal"] for d in summary["drifted_signals"]}
        assert "hmm_regime" in drifted_sigs

    def test_collapsed_signal_flagged(self) -> None:
        # ou_spread collapsed to 0.02 (< 0.05 threshold)
        w = {"hmm_regime": 0.4, "ou_spread": 0.02,
             "llm_sentiment": 0.4, "analyst_recs": 0.18}
        wdf = self._make_wdf(w)
        summary = summarise_weight_evolution(wdf)
        collapsed_sigs = {c["signal"] for c in summary["collapsed_signals"]}
        assert "ou_spread" in collapsed_sigs

    def test_normal_weights_not_collapsed(self) -> None:
        w = {"hmm_regime": 0.3, "ou_spread": 0.3,
             "llm_sentiment": 0.3, "analyst_recs": 0.1}
        wdf = self._make_wdf(w)
        summary = summarise_weight_evolution(wdf)
        assert len(summary["collapsed_signals"]) == 0

    def test_empty_wdf_returns_empty_structure(self) -> None:
        summary = summarise_weight_evolution(pd.DataFrame())
        assert summary["per_ticker_regime"] == {}
        assert summary["drifted_signals"]   == []
        assert summary["collapsed_signals"] == []

    def test_final_row_used_not_first(self) -> None:
        # Two rows for the same ticker/regime; the LAST one has the collapsed signal
        rows = [
            {
                "time": pd.Timestamp("2026-01-02 09:00:00", tz="UTC"),
                "ticker": "AAPL", "regime": 1, "regime_label": "neutral",
                "hmm_regime": 0.3, "ou_spread": 0.3,
                "llm_sentiment": 0.3, "analyst_recs": 0.1,
            },
            {
                "time": pd.Timestamp("2026-01-02 11:00:00", tz="UTC"),
                "ticker": "AAPL", "regime": 1, "regime_label": "neutral",
                "hmm_regime": 0.5, "ou_spread": 0.01,  # collapsed
                "llm_sentiment": 0.4, "analyst_recs": 0.09,
            },
        ]
        wdf = pd.DataFrame(rows)
        wdf["time"] = pd.to_datetime(wdf["time"])
        summary = summarise_weight_evolution(wdf)
        collapsed_sigs = {c["signal"] for c in summary["collapsed_signals"]}
        assert "ou_spread" in collapsed_sigs
