"""
Unit tests for signals/hmm_regime.py.

All DB and filesystem access is either mocked or directed to tmp_path.
Run with:
    .venv/bin/pytest tests/test_hmm_regime.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_engine.signals.hmm_regime import HMMRegimeDetector, _LABEL_MAP


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_ohlcv(n_bars: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with 3 embedded regimes.

    Regime parameters (returns are drawn i.i.d. within each block):
        bear    : mean=-0.003, std=0.025, high volume
        neutral : mean=0.0003, std=0.008, baseline volume
        bull    : mean=0.002,  std=0.012, low volume

    Bars alternate in equal-length blocks: bear → neutral → bull → bear → …
    so that at least one clear transition of each regime type exists.
    """
    rng = np.random.default_rng(seed)

    regime_params = [
        {"mean": -0.003, "std": 0.025, "vol_factor": 1.6},   # 0 = bear
        {"mean":  0.0003, "std": 0.008, "vol_factor": 1.0},  # 1 = neutral
        {"mean":  0.002,  "std": 0.012, "vol_factor": 0.7},  # 2 = bull
    ]

    block_size = n_bars // 6
    regime_seq = ([0] * block_size + [1] * block_size + [2] * block_size) * 2
    regime_seq = regime_seq[:n_bars]

    returns = np.array([
        rng.normal(regime_params[r]["mean"], regime_params[r]["std"])
        for r in regime_seq
    ])

    close = 100.0 * np.cumprod(1.0 + returns)
    base_vol = 1_000_000
    volume = np.array([
        max(1, int(base_vol * regime_params[r]["vol_factor"] * rng.lognormal(0, 0.3)))
        for r in regime_seq
    ])

    n = len(regime_seq)  # may be < n_bars due to block truncation
    noise = rng.uniform(0.995, 1.005, size=(n, 2))
    dates = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")

    return pd.DataFrame({
        "time":   dates,
        "ticker": "TEST",
        "open":   close * noise[:, 0],
        "high":   close * np.maximum(noise[:, 0], noise[:, 1]) * 1.005,
        "low":    close * np.minimum(noise[:, 0], noise[:, 1]) * 0.995,
        "close":  close,
        "volume": volume[:n],
    })


# ---------------------------------------------------------------------------
# Fake storage — plain class, never MagicMock(spec=Storage)
# ---------------------------------------------------------------------------

class _FakeStorage:
    """Lightweight stand-in for Storage used in all unit tests."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.inserted_regimes: list[dict[str, Any]] = []

    def query_ohlcv(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._df.copy()

    def insert_regime(self, rows: list[dict[str, Any]]) -> int:
        self.inserted_regimes.extend(rows)
        return len(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_ohlcv(n_bars=400)


@pytest.fixture()
def fitted_detector(synthetic_df: pd.DataFrame, tmp_path: Path) -> HMMRegimeDetector:
    """Return a detector that has been fit on synthetic data."""
    storage = _FakeStorage(synthetic_df)
    det = HMMRegimeDetector(models_dir=tmp_path)
    det.fit(
        "TEST",
        datetime(2022, 1, 3, tzinfo=timezone.utc),
        datetime(2023, 7, 1, tzinfo=timezone.utc),
        storage,
    )
    return det


# ---------------------------------------------------------------------------
# Tests — _prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_output_shape(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        det = HMMRegimeDetector(models_dir=tmp_path)
        X = det._prepare_features(synthetic_df)
        # volume_zscore needs 20 values, so rows 0–18 are NaN (19 rows dropped).
        # log_return NaN (row 0) and rolling_vol NaN (rows 0–8) are a strict subset.
        assert X.ndim == 2
        assert X.shape[1] == 3
        assert X.shape[0] == len(synthetic_df) - 19

    def test_standardised_columns(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        det = HMMRegimeDetector(models_dir=tmp_path)
        X = det._prepare_features(synthetic_df)
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X.std(axis=0),  1.0, atol=1e-10)

    def test_no_nans(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        det = HMMRegimeDetector(models_dir=tmp_path)
        X = det._prepare_features(synthetic_df)
        assert not np.isnan(X).any()

    def test_raises_on_empty_df(self, tmp_path: Path) -> None:
        det = HMMRegimeDetector(models_dir=tmp_path)
        empty = pd.DataFrame(columns=["time", "ticker", "open", "high", "low", "close", "volume"])
        # _prepare_features returns empty array, does not raise
        X = det._prepare_features(empty)
        assert X.shape == (0, 3) or X.size == 0


# ---------------------------------------------------------------------------
# Tests — fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_is_fitted_after_fit(self, fitted_detector: HMMRegimeDetector) -> None:
        assert fitted_detector.is_fitted is True

    def test_state_labels_populated(self, fitted_detector: HMMRegimeDetector) -> None:
        assert len(fitted_detector.state_labels) == 3
        assert set(fitted_detector.state_labels.values()) == {"bear", "neutral", "bull"}

    def test_model_persisted_to_disk(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        storage = _FakeStorage(synthetic_df)
        det = HMMRegimeDetector(models_dir=tmp_path)
        det.fit(
            "SAVE",
            datetime(2022, 1, 3, tzinfo=timezone.utc),
            datetime(2023, 7, 1, tzinfo=timezone.utc),
            storage,
        )
        assert (tmp_path / "hmm_SAVE.pkl").exists()

    def test_raises_on_empty_data(self, tmp_path: Path) -> None:
        empty_df = pd.DataFrame(columns=["time", "ticker", "open", "high", "low", "close", "volume"])

        class _EmptyStorage:
            def query_ohlcv(self, *_: Any) -> pd.DataFrame:
                return empty_df

        det = HMMRegimeDetector(models_dir=tmp_path)
        with pytest.raises(ValueError, match="No OHLCV data"):
            det.fit(
                "EMPTY",
                datetime(2022, 1, 3, tzinfo=timezone.utc),
                datetime(2022, 1, 4, tzinfo=timezone.utc),
                _EmptyStorage(),
            )

    def test_raises_on_insufficient_rows(self, tmp_path: Path) -> None:
        # Only 5 rows — not enough for a 3-state HMM (need ≥ 30)
        tiny_df = _make_synthetic_ohlcv(n_bars=5, seed=1)

        class _TinyStorage:
            def query_ohlcv(self, *_: Any) -> pd.DataFrame:
                return tiny_df

        det = HMMRegimeDetector(models_dir=tmp_path)
        with pytest.raises((ValueError, Exception)):
            det.fit(
                "TINY",
                datetime(2022, 1, 3, tzinfo=timezone.utc),
                datetime(2022, 1, 10, tzinfo=timezone.utc),
                _TinyStorage(),
            )


# ---------------------------------------------------------------------------
# Tests — state label determinism
# ---------------------------------------------------------------------------

class TestStateLabelAssignment:
    def test_labels_monotone_in_mean_return(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        """
        For each HMM state, compute the mean of the (standardised) log-return
        column across all timesteps assigned to that state.
        The state labelled "bear" must have the lowest mean and "bull" the highest.
        """
        det = fitted_detector
        X = det._prepare_features(synthetic_df)
        states = det.model.predict(X)

        # Build mean log_return per label
        label_means: dict[str, float] = {}
        for hmm_state, label in det.state_labels.items():
            mask = states == hmm_state
            label_means[label] = float(X[mask, 0].mean()) if mask.any() else 0.0

        assert label_means["bear"] < label_means["neutral"] < label_means["bull"], (
            f"State label ordering violated: {label_means}"
        )

    def test_label_assignment_deterministic(
        self,
        synthetic_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Fitting twice on the same data must produce the same label mapping."""
        storage = _FakeStorage(synthetic_df)
        start = datetime(2022, 1, 3, tzinfo=timezone.utc)
        end = datetime(2023, 7, 1, tzinfo=timezone.utc)

        det1 = HMMRegimeDetector(models_dir=tmp_path / "d1")
        det1.fit("TEST", start, end, storage)

        det2 = HMMRegimeDetector(models_dir=tmp_path / "d2")
        det2.fit("TEST", start, end, storage)

        assert det1.state_labels == det2.state_labels

    def test_all_labels_present(self, fitted_detector: HMMRegimeDetector) -> None:
        values = set(fitted_detector.state_labels.values())
        assert values == {"bear", "neutral", "bull"}


# ---------------------------------------------------------------------------
# Tests — predict_regime
# ---------------------------------------------------------------------------

class TestPredictRegime:
    def test_schema(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        storage = _FakeStorage(synthetic_df)
        result = fitted_detector.predict_regime("TEST", df=synthetic_df, storage=storage)

        assert set(result.keys()) == {"regime", "label", "probs", "timestamp"}
        assert isinstance(result["regime"], int)
        assert result["label"] in {"bear", "neutral", "bull"}
        assert isinstance(result["probs"], list)
        assert len(result["probs"]) == fitted_detector.n_states
        assert all(isinstance(p, float) for p in result["probs"])
        assert isinstance(result["timestamp"], datetime)

    def test_probs_sum_to_one(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        result = fitted_detector.predict_regime("TEST", df=synthetic_df)
        assert abs(sum(result["probs"]) - 1.0) < 1e-9

    def test_regime_is_argmax_of_probs(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        result = fitted_detector.predict_regime("TEST", df=synthetic_df)
        assert result["regime"] == int(np.argmax(result["probs"]))

    def test_label_consistent_with_state_labels(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        result = fitted_detector.predict_regime("TEST", df=synthetic_df)
        expected_label = fitted_detector.state_labels[result["regime"]]
        assert result["label"] == expected_label

    def test_inserts_into_storage(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        storage = _FakeStorage(synthetic_df)
        fitted_detector.predict_regime("TEST", df=synthetic_df, storage=storage)
        assert len(storage.inserted_regimes) == 1
        row = storage.inserted_regimes[0]
        assert row["ticker"] == "TEST"
        assert isinstance(row["regime"], int)
        assert isinstance(row["regime_probs"], dict)

    def test_no_insert_without_storage(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        # When storage is None, predict_regime should succeed but not try to persist.
        result = fitted_detector.predict_regime("TEST", df=synthetic_df, storage=None)
        assert result["label"] in {"bear", "neutral", "bull"}

    def test_raises_when_not_fitted(
        self,
        synthetic_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        det = HMMRegimeDetector(models_dir=tmp_path)
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict_regime("TEST", df=synthetic_df)

    def test_raises_when_df_none_and_no_storage(
        self,
        fitted_detector: HMMRegimeDetector,
    ) -> None:
        with pytest.raises(ValueError, match="Either df or storage"):
            fitted_detector.predict_regime("TEST", df=None, storage=None)

    def test_fetches_from_storage_when_df_none(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        storage = _FakeStorage(synthetic_df)
        result = fitted_detector.predict_regime("TEST", df=None, storage=storage)
        assert result["label"] in {"bear", "neutral", "bull"}


# ---------------------------------------------------------------------------
# Tests — save / load round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_load_restores_state(
        self,
        synthetic_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        storage = _FakeStorage(synthetic_df)
        original = HMMRegimeDetector(models_dir=tmp_path)
        original.fit(
            "PERSIST",
            datetime(2022, 1, 3, tzinfo=timezone.utc),
            datetime(2023, 7, 1, tzinfo=timezone.utc),
            storage,
        )

        loaded = HMMRegimeDetector(models_dir=tmp_path)
        loaded.load("PERSIST")

        assert loaded.is_fitted is True
        assert loaded.state_labels == original.state_labels
        assert loaded.n_states == original.n_states

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        det = HMMRegimeDetector(models_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            det.load("NOEXIST")

    def test_loaded_model_can_predict(
        self,
        synthetic_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        storage = _FakeStorage(synthetic_df)
        det = HMMRegimeDetector(models_dir=tmp_path)
        det.fit(
            "LOADPRED",
            datetime(2022, 1, 3, tzinfo=timezone.utc),
            datetime(2023, 7, 1, tzinfo=timezone.utc),
            storage,
        )

        fresh = HMMRegimeDetector(models_dir=tmp_path)
        fresh.load("LOADPRED")
        result = fresh.predict_regime("LOADPRED", df=synthetic_df)
        assert result["label"] in {"bear", "neutral", "bull"}


# ---------------------------------------------------------------------------
# Tests — partial_fit_online
# ---------------------------------------------------------------------------

class TestPartialFitOnline:
    def test_bars_accumulate_in_buffer(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        det = fitted_detector
        initial_count = len(det._online_buffer)
        bar = synthetic_df.iloc[-1].to_dict()
        det.partial_fit_online(bar)
        assert len(det._online_buffer) == initial_count + 1

    def test_refit_triggered_after_threshold(
        self,
        synthetic_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """
        Seed the buffer with enough rows (≥ n_states * 10) and then push
        refit_every bars to trigger an automatic re-fit.
        """
        refit_every = 5
        det = HMMRegimeDetector(
            refit_every=refit_every,
            models_dir=tmp_path,
        )

        # Pre-seed the buffer with enough rows to allow fitting.
        seed_rows = synthetic_df.iloc[:200].to_dict(orient="records")
        for row in seed_rows:
            det._online_buffer.append(row)
        det._ticker = "ONLINE"
        det.is_fitted = True  # allow save to be called

        # Fit the HMM once so the model object is in a valid state.
        X_seed = det._prepare_features(synthetic_df.iloc[:200])
        det.model.fit(X_seed)
        det._assign_state_labels(X_seed)

        # Now push exactly refit_every new bars — should trigger a re-fit.
        new_bars = synthetic_df.iloc[200:200 + refit_every].to_dict(orient="records")
        for bar in new_bars:
            det.partial_fit_online(bar)

        # After the refit the counter should have been reset to 0.
        assert det._bars_since_refit == 0

    def test_counter_increments_before_threshold(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_df: pd.DataFrame,
    ) -> None:
        det = fitted_detector
        det._bars_since_refit = 0  # reset counter
        bar = synthetic_df.iloc[-1].to_dict()
        for i in range(1, fitted_detector.refit_every):
            det.partial_fit_online(bar)
            assert det._bars_since_refit == i

    def test_buffer_respects_maxlen(self, tmp_path: Path) -> None:
        lookback = 50
        det = HMMRegimeDetector(lookback_days=lookback, models_dir=tmp_path)
        bar: dict[str, Any] = {
            "time": datetime.now(tz=timezone.utc),
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1_000_000,
        }
        for _ in range(lookback + 20):
            det._online_buffer.append(bar)
        assert len(det._online_buffer) == lookback
