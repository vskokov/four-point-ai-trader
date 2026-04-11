"""
Unit tests for meta_agent/mwu_agent.py.

All storage and filesystem I/O is controlled via tmp_path or mocks.
Run with:
    .venv/bin/pytest tests/meta_agent/test_mwu_agent.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trading_engine.meta_agent.mwu_agent import MWUMetaAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(tmp_path: Path, ticker: str = "AAPL", **kwargs: Any) -> MWUMetaAgent:
    """Return a fresh MWUMetaAgent using tmp_path as the models directory."""
    return MWUMetaAgent(ticker=ticker, models_dir=tmp_path, **kwargs)


def _bull_signals(conf: float = 1.0) -> dict[str, dict[str, Any]]:
    return {
        "hmm_regime":    {"signal": 1,  "confidence": conf},
        "ou_spread":     {"signal": 1,  "confidence": conf},
        "llm_sentiment": {"signal": 1,  "confidence": conf},
        "analyst_recs":  {"signal": 1,  "confidence": conf},
    }


def _bear_signals(conf: float = 1.0) -> dict[str, dict[str, Any]]:
    return {
        "hmm_regime":    {"signal": -1, "confidence": conf},
        "ou_spread":     {"signal": -1, "confidence": conf},
        "llm_sentiment": {"signal": -1, "confidence": conf},
        "analyst_recs":  {"signal": -1, "confidence": conf},
    }


def _mixed_signals() -> dict[str, dict[str, Any]]:
    """hmm_regime=+1, ou_spread=−1, llm_sentiment=0, analyst_recs=0."""
    return {
        "hmm_regime":    {"signal": 1,  "confidence": 0.8},
        "ou_spread":     {"signal": -1, "confidence": 0.6},
        "llm_sentiment": {"signal": 0,  "confidence": 0.5},
        "analyst_recs":  {"signal": 0,  "confidence": 0.0},
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_weights_shape_at_start(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        assert agent.weights.shape == (3, 4)

    def test_weights_sum_to_one_per_regime(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        for r in range(3):
            assert abs(agent.weights[r].sum() - 1.0) < 1e-12

    def test_signal_names_set_correctly(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        assert agent.signal_names == [
            "hmm_regime", "ou_spread", "llm_sentiment", "analyst_recs"
        ]

    def test_weight_history_empty_at_start(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        assert agent.weight_history == []

    def test_custom_eta(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, eta=0.5)
        assert agent.eta == 0.5

    def test_loads_persisted_weights(self, tmp_path: Path) -> None:
        """If mwu_weights_AAPL.npy exists with correct shape, it is loaded."""
        saved = np.array([
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.3, 0.2],
            [0.3, 0.3, 0.2, 0.2],
        ])
        np.save(str(tmp_path / "mwu_weights_AAPL.npy"), saved)
        agent = _make_agent(tmp_path)
        np.testing.assert_allclose(agent.weights, saved)

    def test_ignores_wrong_shape_file(self, tmp_path: Path) -> None:
        """A persisted file with wrong shape is ignored; half-weight init is used."""
        bad = np.ones((2, 5))
        np.save(str(tmp_path / "mwu_weights_AAPL.npy"), bad)
        agent = _make_agent(tmp_path)
        # Falls back to _default_weights() — half-weight for analyst_recs
        expected_row = np.array([2 / 7, 2 / 7, 2 / 7, 1 / 7])
        for r in range(3):
            np.testing.assert_allclose(agent.weights[r], expected_row, rtol=1e-10)


# ---------------------------------------------------------------------------
# Half-weight initialisation for analyst_recs
# ---------------------------------------------------------------------------

class TestHalfWeightInit:
    def test_analyst_recs_starts_at_half_weight(self, tmp_path: Path) -> None:
        """analyst_recs initial weight = 1/7; the other three are each 2/7."""
        agent = _make_agent(tmp_path)
        w = agent.weights[0]
        # Indices: hmm_regime=0, ou_spread=1, llm_sentiment=2, analyst_recs=3
        assert abs(w[0] - 2 / 7) < 1e-10
        assert abs(w[1] - 2 / 7) < 1e-10
        assert abs(w[2] - 2 / 7) < 1e-10
        assert abs(w[3] - 1 / 7) < 1e-10

    def test_half_weight_consistent_across_regimes(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        expected_row = np.array([2 / 7, 2 / 7, 2 / 7, 1 / 7])
        for r in range(3):
            np.testing.assert_allclose(agent.weights[r], expected_row, rtol=1e-10)

    def test_fallback_reset_uses_half_weight(self, tmp_path: Path) -> None:
        """When all weights collapse to exactly 0, reset uses _default_weights()."""
        agent = _make_agent(tmp_path, eta=0.1)
        # Force weights to exactly zero so row_sum == 0 triggers the fallback
        agent.weights[1] = np.array([0.0, 0.0, 0.0, 0.0])
        agent.update_weights("AAPL", _bull_signals(), regime_t=1, actual_direction=1)
        expected_row = np.array([2 / 7, 2 / 7, 2 / 7, 1 / 7])
        np.testing.assert_allclose(agent.weights[1], expected_row, rtol=1e-10)

    def test_shape_mismatch_on_load_resets_to_half_weight(self, tmp_path: Path) -> None:
        """Old (3, 3) weight file triggers shape-mismatch warning, resets to 4-signal init."""
        old_weights = np.full((3, 3), 1 / 3)
        np.save(str(tmp_path / "mwu_weights_AAPL.npy"), old_weights)
        agent = _make_agent(tmp_path)
        # Shape mismatch → falls back to default init
        assert agent.weights.shape == (3, 4)
        expected_row = np.array([2 / 7, 2 / 7, 2 / 7, 1 / 7])
        for r in range(3):
            np.testing.assert_allclose(agent.weights[r], expected_row, rtol=1e-10)


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------

class TestDecide:
    def test_returns_required_keys(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.decide("AAPL", _bull_signals(), regime=2)
        assert set(result.keys()) == {
            "ticker", "final_signal", "score", "regime", "weights", "timestamp"
        }

    def test_bull_signals_produce_positive_score(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.decide("AAPL", _bull_signals(conf=1.0), regime=2)
        assert result["score"] > 0

    def test_bear_signals_produce_negative_score(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.decide("AAPL", _bear_signals(conf=1.0), regime=0)
        assert result["score"] < 0

    def test_bull_signals_high_confidence_give_final_signal_plus_one(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path, min_confidence=0.1)
        result = agent.decide("AAPL", _bull_signals(conf=1.0), regime=1)
        assert result["final_signal"] == 1

    def test_bear_signals_high_confidence_give_final_signal_minus_one(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path, min_confidence=0.1)
        result = agent.decide("AAPL", _bear_signals(conf=1.0), regime=1)
        assert result["final_signal"] == -1

    def test_low_score_below_min_confidence_gives_neutral(
        self, tmp_path: Path
    ) -> None:
        # All signals neutral → score = 0 → final_signal must be 0
        agent = _make_agent(tmp_path, min_confidence=0.3)
        neutral = {
            "hmm_regime":    {"signal": 0, "confidence": 0.0},
            "ou_spread":     {"signal": 0, "confidence": 0.0},
            "llm_sentiment": {"signal": 0, "confidence": 0.0},
        }
        result = agent.decide("AAPL", neutral, regime=1)
        assert result["final_signal"] == 0

    def test_weights_dict_keys_match_signal_names(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.decide("TSLA", _bull_signals(), regime=0)
        assert set(result["weights"].keys()) == set(agent.signal_names)

    def test_weights_dict_values_sum_to_one(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.decide("TSLA", _bull_signals(), regime=0)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-12

    def test_ticker_and_regime_preserved_in_result(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.decide("MSFT", _bull_signals(), regime=2)
        assert result["ticker"] == "MSFT"
        assert result["regime"] == 2

    def test_timestamp_is_utc_datetime(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.decide("AAPL", _bull_signals(), regime=1)
        assert isinstance(result["timestamp"], datetime)
        assert result["timestamp"].tzinfo is not None


# ---------------------------------------------------------------------------
# Missing signals
# ---------------------------------------------------------------------------

class TestMissingSignals:
    def test_missing_signal_treated_as_neutral(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        # Only one signal provided
        partial = {"hmm_regime": {"signal": 1, "confidence": 1.0}}
        result = agent.decide("AAPL", partial, regime=1)
        # ou_spread and llm_sentiment are missing → treated as 0*0=0
        # score = w[0]*1*1 + w[1]*0*0 + w[2]*0*0 = 1/3
        expected_score = agent.weights[1, 0] * 1.0 * 1.0
        assert abs(result["score"] - expected_score) < 1e-12

    def test_empty_signals_dict_gives_zero_score(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.decide("AAPL", {}, regime=1)
        assert result["score"] == 0.0

    def test_none_signal_value_treated_as_neutral(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        signals = {
            "hmm_regime":    None,
            "ou_spread":     {"signal": 1, "confidence": 0.9},
            "llm_sentiment": None,
        }
        result = agent.decide("AAPL", signals, regime=1)
        # Only ou_spread contributes
        expected = agent.weights[1, 1] * 1.0 * 0.9
        assert abs(result["score"] - expected) < 1e-12

    def test_update_weights_with_missing_signals_no_error(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path)
        # Should not raise
        agent.update_weights(
            ticker="AAPL",
            signals_t={},
            regime_t=1,
            actual_direction=1,
        )
        # Weights must still sum to 1
        assert abs(agent.weights[1].sum() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# update_weights() — MWU correctness
# ---------------------------------------------------------------------------

class TestUpdateWeights:
    def test_weights_sum_to_one_after_update(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        for regime in range(3):
            agent.update_weights("AAPL", _bull_signals(), regime, actual_direction=1)
        for r in range(3):
            assert abs(agent.weights[r].sum() - 1.0) < 1e-12

    def test_correct_signal_gains_relative_weight_over_time(
        self, tmp_path: Path
    ) -> None:
        """
        Signal 'hmm_regime' always correct, 'ou_spread', 'llm_sentiment', and
        'analyst_recs' always wrong.  After many rounds hmm_regime's weight
        should dominate.
        """
        agent = _make_agent(tmp_path, eta=0.3)
        signals = {
            "hmm_regime":    {"signal": 1,  "confidence": 1.0},
            "ou_spread":     {"signal": -1, "confidence": 1.0},
            "llm_sentiment": {"signal": -1, "confidence": 1.0},
            "analyst_recs":  {"signal": -1, "confidence": 1.0},
        }
        for _ in range(30):
            agent.update_weights(
                ticker="AAPL",
                signals_t=signals,
                regime_t=1,
                actual_direction=1,
            )
        # hmm_regime (index 0) should have the highest weight in regime 1
        assert agent.weights[1, 0] > agent.weights[1, 1]
        assert agent.weights[1, 0] > agent.weights[1, 2]
        assert agent.weights[1, 0] > agent.weights[1, 3]

    def test_consistently_wrong_signal_loses_weight(
        self, tmp_path: Path
    ) -> None:
        """ou_spread always wrong; its weight should fall significantly."""
        agent = _make_agent(tmp_path, eta=0.3)
        initial_ou_weight = agent.weights[1, 1]
        signals = {
            "hmm_regime":    {"signal": 1,  "confidence": 1.0},
            "ou_spread":     {"signal": -1, "confidence": 1.0},  # always wrong
            "llm_sentiment": {"signal": 1,  "confidence": 1.0},
            "analyst_recs":  {"signal": 1,  "confidence": 1.0},
        }
        for _ in range(20):
            agent.update_weights("AAPL", signals, regime_t=1, actual_direction=1)
        assert agent.weights[1, 1] < initial_ou_weight

    def test_regime_isolation_other_regimes_unchanged(
        self, tmp_path: Path
    ) -> None:
        """An update for regime 1 must not touch weights for regimes 0 and 2."""
        agent = _make_agent(tmp_path)
        weights_r0_before = agent.weights[0].copy()
        weights_r2_before = agent.weights[2].copy()

        agent.update_weights("AAPL", _bull_signals(), regime_t=1, actual_direction=1)

        np.testing.assert_array_equal(agent.weights[0], weights_r0_before)
        np.testing.assert_array_equal(agent.weights[2], weights_r2_before)

    def test_weight_history_appended(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        assert len(agent.weight_history) == 0
        agent.update_weights("AAPL", _bull_signals(), regime_t=0, actual_direction=1)
        assert len(agent.weight_history) == 1
        agent.update_weights("AAPL", _bull_signals(), regime_t=1, actual_direction=-1)
        assert len(agent.weight_history) == 2

    def test_weight_history_entry_structure(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.update_weights("TSLA", _bull_signals(), regime_t=2, actual_direction=1)
        entry = agent.weight_history[0]
        assert "timestamp" in entry
        assert "regime" in entry
        assert "weights" in entry
        assert entry["regime"] == 2
        assert isinstance(entry["weights"], np.ndarray)

    def test_neutral_signal_gets_half_loss(self, tmp_path: Path) -> None:
        """
        A neutral signal (value=0) should receive loss=0.5, which is less than
        a wrong signal (loss=1) but more than a correct one (loss=0).  After
        one update the weight of the neutral signal should lie between the
        weight of the correct and the wrong signal.
        """
        agent = _make_agent(tmp_path, eta=0.5)
        signals = {
            "hmm_regime":    {"signal": 1,  "confidence": 1.0},  # correct
            "ou_spread":     {"signal": 0,  "confidence": 0.0},  # neutral → loss=0.5
            "llm_sentiment": {"signal": -1, "confidence": 1.0},  # wrong
            "analyst_recs":  {"signal": 0,  "confidence": 0.0},  # neutral → loss=0.5
        }
        agent.update_weights("AAPL", signals, regime_t=1, actual_direction=1)
        w = agent.weights[1]
        # correct (index 0) > neutral (index 1) > wrong (index 2)
        assert w[0] > w[1] > w[2]

    def test_multiple_updates_weights_always_sum_to_one(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path, eta=0.2)
        directions = [1, -1, 1, 1, -1, -1, 1, -1]
        for actual in directions:
            agent.update_weights(
                "AAPL", _mixed_signals(), regime_t=0, actual_direction=actual
            )
        for r in range(3):
            assert abs(agent.weights[r].sum() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Weight persistence (save / load cycle)
# ---------------------------------------------------------------------------

class TestWeightPersistence:
    def test_weights_persisted_after_update(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.update_weights("AAPL", _bull_signals(), regime_t=0, actual_direction=1)
        weights_file = tmp_path / "mwu_weights_AAPL.npy"
        assert weights_file.exists()

    def test_saved_weights_loadable_by_new_instance(self, tmp_path: Path) -> None:
        agent1 = _make_agent(tmp_path, eta=0.4)
        # Run several updates to change weights non-trivially
        for _ in range(10):
            agent1.update_weights(
                "AAPL",
                {
                    "hmm_regime":    {"signal": 1,  "confidence": 1.0},
                    "ou_spread":     {"signal": -1, "confidence": 1.0},
                    "llm_sentiment": {"signal": 1,  "confidence": 1.0},
                    "analyst_recs":  {"signal": 1,  "confidence": 1.0},
                },
                regime_t=1,
                actual_direction=1,
            )
        saved_weights = agent1.weights.copy()

        # A brand new instance pointed at the same tmp_path loads the saved file
        agent2 = _make_agent(tmp_path)
        np.testing.assert_allclose(agent2.weights, saved_weights)

    def test_load_cycle_preserves_per_regime_normalisation(
        self, tmp_path: Path
    ) -> None:
        agent1 = _make_agent(tmp_path, eta=0.3)
        for r in range(3):
            agent1.update_weights(
                "AAPL", _bull_signals(), regime_t=r, actual_direction=1
            )

        agent2 = _make_agent(tmp_path)
        for r in range(3):
            assert abs(agent2.weights[r].sum() - 1.0) < 1e-12

    def test_no_file_at_start_no_error(self, tmp_path: Path) -> None:
        """No models file → no crash; half-weight init used."""
        agent = _make_agent(tmp_path)
        assert agent.weights.shape == (3, 4)
        expected_row = np.array([2 / 7, 2 / 7, 2 / 7, 1 / 7])
        for r in range(3):
            np.testing.assert_allclose(agent.weights[r], expected_row, rtol=1e-10)


# ---------------------------------------------------------------------------
# get_actual_direction()
# ---------------------------------------------------------------------------

class TestGetActualDirection:
    def _make_ohlcv_pair(
        self,
        decision_time: datetime,
        price_before: float,
        price_after: float,
    ) -> pd.DataFrame:
        """Return a two-bar DataFrame: one bar strictly before and one strictly after."""
        times = [
            decision_time - timedelta(minutes=1),
            decision_time + timedelta(minutes=1),
        ]
        prices = [price_before, price_after]
        return pd.DataFrame(
            {
                "time":   pd.to_datetime(times, utc=True),
                "ticker": "AAPL",
                "open":   prices,
                "high":   prices,
                "low":    prices,
                "close":  prices,
                "volume": [1000, 1000],
            }
        )

    def test_price_up_returns_plus_one(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        storage = MagicMock()
        df = self._make_ohlcv_pair(base, price_before=100.0, price_after=100.2)
        storage.query_ohlcv.return_value = df
        result = agent.get_actual_direction("AAPL", base, horizon_bars=1, storage=storage)
        assert result == 1

    def test_price_down_returns_minus_one(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        storage = MagicMock()
        df = self._make_ohlcv_pair(base, price_before=100.0, price_after=99.7)
        storage.query_ohlcv.return_value = df
        result = agent.get_actual_direction("AAPL", base, horizon_bars=1, storage=storage)
        assert result == -1

    def test_tiny_change_returns_zero(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        storage = MagicMock()
        # 0.001% change — below 0.05% threshold
        df = self._make_ohlcv_pair(base, price_before=100.0, price_after=100.001)
        storage.query_ohlcv.return_value = df
        result = agent.get_actual_direction("AAPL", base, horizon_bars=1, storage=storage)
        assert result == 0

    def test_no_storage_returns_zero(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        result = agent.get_actual_direction("AAPL", base, storage=None)
        assert result == 0

    def test_empty_dataframe_returns_zero(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        storage = MagicMock()
        storage.query_ohlcv.return_value = pd.DataFrame()
        result = agent.get_actual_direction("AAPL", base, storage=storage)
        assert result == 0

    def test_no_bars_after_decision_returns_zero(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        storage = MagicMock()
        # Only one bar strictly before decision_time — nothing after
        df = pd.DataFrame(
            {
                "time":   pd.to_datetime([base - timedelta(minutes=2)], utc=True),
                "ticker": ["AAPL"],
                "open":   [100.0],
                "high":   [100.0],
                "low":    [100.0],
                "close":  [100.0],
                "volume": [1000],
            }
        )
        storage.query_ohlcv.return_value = df
        result = agent.get_actual_direction("AAPL", base, storage=storage)
        assert result == 0


# ---------------------------------------------------------------------------
# performance_report()
# ---------------------------------------------------------------------------

class TestPerformanceReport:
    def test_empty_report_when_no_updates(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        report = agent.performance_report()
        assert report["n_updates"] == 0
        for name in agent.signal_names:
            assert report["per_signal_win_rate"][name] is None

    def test_report_n_updates_matches_call_count(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        for _ in range(5):
            agent.update_weights("AAPL", _bull_signals(), regime_t=1, actual_direction=1)
        report = agent.performance_report()
        assert report["n_updates"] == 5

    def test_perfect_signal_win_rate_is_one(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        signals = {
            "hmm_regime":    {"signal": 1, "confidence": 1.0},
            "ou_spread":     {"signal": 1, "confidence": 1.0},
            "llm_sentiment": {"signal": 1, "confidence": 1.0},
            "analyst_recs":  {"signal": 1, "confidence": 1.0},
        }
        for _ in range(10):
            agent.update_weights("AAPL", signals, regime_t=1, actual_direction=1)
        report = agent.performance_report()
        for name in agent.signal_names:
            assert report["per_signal_win_rate"][name] == 1.0

    def test_always_wrong_signal_win_rate_is_zero(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        signals = {
            "hmm_regime":    {"signal": -1, "confidence": 1.0},
            "ou_spread":     {"signal": -1, "confidence": 1.0},
            "llm_sentiment": {"signal": -1, "confidence": 1.0},
            "analyst_recs":  {"signal": -1, "confidence": 1.0},
        }
        for _ in range(10):
            agent.update_weights("AAPL", signals, regime_t=0, actual_direction=1)
        report = agent.performance_report()
        for name in agent.signal_names:
            assert report["per_signal_win_rate"][name] == 0.0

    def test_current_weights_in_report_sum_to_one(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.update_weights("AAPL", _bull_signals(), regime_t=1, actual_direction=1)
        report = agent.performance_report()
        for regime_label, w_dict in report["current_weights"].items():
            assert abs(sum(w_dict.values()) - 1.0) < 1e-12

    def test_report_has_all_required_keys(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.update_weights("AAPL", _bull_signals(), regime_t=0, actual_direction=-1)
        report = agent.performance_report()
        assert "n_updates" in report
        assert "per_signal_win_rate" in report
        assert "per_regime_win_rate" in report
        assert "current_weights" in report
        assert "weight_history_length" in report


# ---------------------------------------------------------------------------
# scheduled_update()
# ---------------------------------------------------------------------------

class TestScheduledUpdate:
    def test_returns_decision_dict(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        result = agent.scheduled_update("AAPL", _bull_signals(), regime=1)
        assert "final_signal" in result
        assert "score" in result

    def test_pending_decision_stored(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent.scheduled_update("AAPL", _bull_signals(), regime=1)
        assert len(agent._pending) == 1

    def test_pending_decision_evaluated_after_horizon(
        self, tmp_path: Path
    ) -> None:
        """
        A decision made in the past (> horizon minutes ago) should be
        evaluated and cleared from pending when scheduled_update is called again.
        """
        agent = _make_agent(tmp_path)

        # Inject a "past" decision directly so it appears old enough
        past_time = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
        old_decision = {
            "ticker": "AAPL",
            "final_signal": 1,
            "score": 0.5,
            "regime": 1,
            "weights": {},
            "timestamp": past_time,
        }
        old_key = ("AAPL", past_time.isoformat())
        agent._pending[old_key] = {
            "decision": old_decision,
            "signals_t": _bull_signals(),
            "regime_t": 1,
            "horizon_bars": 1,
        }

        storage = MagicMock()
        storage.query_ohlcv.return_value = pd.DataFrame()  # returns 0 direction

        # This call should process the old pending decision
        agent.scheduled_update(
            "AAPL", _bull_signals(), regime=1, storage=storage, horizon_bars=5
        )
        # The old entry should have been consumed
        assert old_key not in agent._pending

    def test_weights_still_normalised_after_scheduled_update(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path)
        agent.scheduled_update("AAPL", _bull_signals(), regime=0)
        for r in range(3):
            assert abs(agent.weights[r].sum() - 1.0) < 1e-12
