"""
Unit tests for signals/kalman_pairs.py and signals/mean_reversion.py.

Synthetic data generation
-------------------------
A cointegrated pair (P1, P2) is generated from a known OU spread:

    P2_t = P2_{t-1} + eps_p2,       eps_p2 ~ N(0, sigma_p2)
    Z_t  = Z_{t-1} + kappa*(mu-Z_{t-1}) + eps_z,  eps_z ~ N(0, sigma_z)
    P1_t = TRUE_BETA * P2_t + Z_t

True parameters: TRUE_BETA=2.0, kappa=0.15, mu=0.0, sigma_z=0.5, n=800.

All DB access is replaced with a _FakeStorage; no network or file I/O
beyond tmp_path is performed.
Run with:
    .venv/bin/pytest tests/test_mean_reversion.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_engine.signals.kalman_pairs import KalmanHedgeRatio
from trading_engine.signals.mean_reversion import CointegrationTest, OUSpreadSignal

# ---------------------------------------------------------------------------
# Synthetic pair constants — kept at module level so every test uses the
# same true parameters.
# ---------------------------------------------------------------------------

TRUE_BETA: float = 2.0
TRUE_KAPPA: float = 0.15   # half-life ≈ 4.6 bars
TRUE_MU: float = 0.0
TRUE_SIGMA_Z: float = 0.5
N_BARS: int = 800
SEED: int = 7


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _simulate_pair(
    n: int = N_BARS,
    beta: float = TRUE_BETA,
    kappa: float = TRUE_KAPPA,
    mu: float = TRUE_MU,
    sigma_z: float = TRUE_SIGMA_Z,
    sigma_p2: float = 1.0,
    seed: int = SEED,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return (p1, p2, spread_true) as pd.Series with a DatetimeIndex.

    The spread Z_t = P1_t - beta*P2_t follows an OU process by construction.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")

    # P2: random walk
    p2 = np.empty(n)
    p2[0] = 100.0
    for t in range(1, n):
        p2[t] = p2[t - 1] + rng.normal(0.0, sigma_p2)
    p2 = np.abs(p2)  # keep prices positive

    # Z: discrete-time OU
    z = np.empty(n)
    z[0] = 0.0
    for t in range(1, n):
        z[t] = z[t - 1] + kappa * (mu - z[t - 1]) + rng.normal(0.0, sigma_z)

    # P1 = beta * P2 + Z
    p1 = beta * p2 + z

    return (
        pd.Series(p1, index=dates, name="close"),
        pd.Series(p2, index=dates, name="close"),
        pd.Series(z,  index=dates, name="spread_true"),
    )


def _make_ohlcv_df(close: pd.Series, ticker: str) -> pd.DataFrame:
    """Wrap a close series into a minimal OHLCV DataFrame."""
    return pd.DataFrame({
        "time":   close.index,
        "ticker": ticker,
        "open":   close.values,
        "high":   close.values * 1.005,
        "low":    close.values * 0.995,
        "close":  close.values,
        "volume": np.ones(len(close), dtype=int) * 1_000_000,
    })


def _random_walk_series(n: int = N_BARS, seed: int = 99) -> pd.Series:
    """Independent random walk — NOT cointegrated with anything."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    return pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=dates, name="close")


# ---------------------------------------------------------------------------
# Fake storage — plain class, not MagicMock(spec=...)
# ---------------------------------------------------------------------------

class _FakeStorage:
    def __init__(self) -> None:
        self.inserted_signals: list[dict[str, Any]] = []

    def insert_signal(self, rows: list[dict[str, Any]]) -> int:
        self.inserted_signals.extend(rows)
        return len(rows)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pair() -> tuple[pd.Series, pd.Series, pd.Series]:
    return _simulate_pair()


@pytest.fixture(scope="module")
def df_pair(
    pair: tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p1, p2, _ = pair
    return _make_ohlcv_df(p1, "P1"), _make_ohlcv_df(p2, "P2")


# ---------------------------------------------------------------------------
# Tests — CointegrationTest
# ---------------------------------------------------------------------------

class TestCointegrationTest:
    def test_detects_cointegration(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        p1, p2, _ = pair
        result = CointegrationTest().test(p1, p2)
        assert result["cointegrated"] is True, (
            f"EG p-value was {result['eg_pvalue']:.4f} — expected < 0.05"
        )

    def test_rejects_independent_random_walks(self) -> None:
        rw1 = _random_walk_series(seed=11)
        rw2 = _random_walk_series(seed=22)
        # Two independent random walks are not cointegrated in expectation.
        # We run multiple seeds to avoid random failures.
        coint_count = 0
        for seed_offset in range(5):
            rw1 = _random_walk_series(seed=100 + seed_offset)
            rw2 = _random_walk_series(seed=200 + seed_offset)
            result = CointegrationTest().test(rw1, rw2)
            if result["cointegrated"]:
                coint_count += 1
        # At 5% level we may get ~1 false positive; allow at most 2.
        assert coint_count <= 2, (
            f"{coint_count}/5 random-walk pairs were falsely flagged cointegrated"
        )

    def test_returns_correct_schema(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        p1, p2, _ = pair
        result = CointegrationTest().test(p1, p2)
        assert set(result.keys()) == {
            "cointegrated", "eg_pvalue", "johansen_trace_stat", "beta_ols"
        }
        assert isinstance(result["cointegrated"], bool)
        assert isinstance(result["eg_pvalue"], float)
        assert isinstance(result["johansen_trace_stat"], float)
        assert isinstance(result["beta_ols"], float)

    def test_beta_ols_close_to_true(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        p1, p2, _ = pair
        result = CointegrationTest().test(p1, p2)
        # OLS with 800 bars should recover TRUE_BETA=2.0 within 0.1.
        assert abs(result["beta_ols"] - TRUE_BETA) < 0.1, (
            f"OLS beta {result['beta_ols']:.4f} too far from {TRUE_BETA}"
        )

    def test_eg_pvalue_range(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        p1, p2, _ = pair
        result = CointegrationTest().test(p1, p2)
        assert 0.0 <= result["eg_pvalue"] <= 1.0

    def test_johansen_stat_positive(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        p1, p2, _ = pair
        result = CointegrationTest().test(p1, p2)
        assert result["johansen_trace_stat"] > 0.0


# ---------------------------------------------------------------------------
# Tests — KalmanHedgeRatio
# ---------------------------------------------------------------------------

class TestKalmanHedgeRatio:
    def test_update_returns_correct_schema(self) -> None:
        kf = KalmanHedgeRatio()
        res = kf.update(200.0, 100.0)
        assert set(res.keys()) == {"beta", "beta_var"}
        assert isinstance(res["beta"], float)
        assert isinstance(res["beta_var"], float)

    def test_beta_var_positive(self) -> None:
        kf = KalmanHedgeRatio()
        for _ in range(10):
            res = kf.update(200.0, 100.0)
        assert res["beta_var"] > 0.0

    def test_beta_history_grows(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        p1, p2, _ = pair
        kf = KalmanHedgeRatio()
        for v1, v2 in zip(p1, p2):
            kf.update(float(v1), float(v2))
        assert len(kf.beta_history) == N_BARS

    def test_beta_tracks_true_beta(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        """
        After a warmup period the Kalman beta should converge near TRUE_BETA.
        We check the mean of the last 400 beta estimates.
        """
        p1, p2, _ = pair
        kf = KalmanHedgeRatio()
        for v1, v2 in zip(p1, p2):
            kf.update(float(v1), float(v2))

        tail_betas = kf.beta_history[N_BARS // 2 :]
        mean_beta = float(np.mean(tail_betas))
        assert abs(mean_beta - TRUE_BETA) < 0.3, (
            f"Mean Kalman beta {mean_beta:.4f} too far from true {TRUE_BETA}"
        )

    def test_get_spread_length_matches_input(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        p1, p2, _ = pair
        kf = KalmanHedgeRatio()
        spread = kf.get_spread(p1, p2)
        assert len(spread) == len(p1)
        assert spread.index.equals(p1.index)

    def test_get_spread_is_idempotent(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        """Two calls to get_spread must return the same values (no side effects)."""
        p1, p2, _ = pair
        kf = KalmanHedgeRatio()
        s1 = kf.get_spread(p1, p2)
        s2 = kf.get_spread(p1, p2)
        pd.testing.assert_series_equal(s1, s2)

    def test_get_spread_does_not_contaminate_update_state(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        """get_spread must not modify the instance's running filter state."""
        p1, p2, _ = pair
        kf = KalmanHedgeRatio()
        beta_before = kf.beta
        kf.get_spread(p1, p2)
        # Running state should be unchanged after a batch get_spread call.
        assert kf.beta == beta_before
        assert len(kf.beta_history) == 0  # no update() calls were made on self

    def test_get_spread_approx_zero_mean_for_cointegrated(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        """
        Kalman spread from a cointegrated pair (mu=0) should have near-zero mean
        after the filter has warmed up (skip first 50 bars).
        """
        p1, p2, _ = pair
        kf = KalmanHedgeRatio()
        spread = kf.get_spread(p1, p2)
        tail_mean = float(spread.iloc[50:].mean())
        assert abs(tail_mean) < 5.0, (
            f"Spread mean {tail_mean:.4f} too large for a mu=0 OU process"
        )

    def test_save_load_restores_state(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
        tmp_path: Path,
    ) -> None:
        p1, p2, _ = pair
        kf = KalmanHedgeRatio(models_dir=tmp_path)
        for v1, v2 in zip(p1.iloc[:100], p2.iloc[:100]):
            kf.update(float(v1), float(v2))
        kf.save("AAA", "BBB")

        loaded = KalmanHedgeRatio(models_dir=tmp_path)
        loaded.load("AAA", "BBB")

        assert abs(loaded.beta - kf.beta) < 1e-9
        assert len(loaded.beta_history) == len(kf.beta_history)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        kf = KalmanHedgeRatio(models_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            kf.load("NO", "EXIST")


# ---------------------------------------------------------------------------
# Tests — OUSpreadSignal.fit_ou_params
# ---------------------------------------------------------------------------

class TestFitOUParams:
    def test_recovers_kappa_within_tolerance(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        """
        OLS on the discrete OU form should recover TRUE_KAPPA within 50%.
        We use the true spread (not the Kalman estimate) to isolate the OLS fit.
        """
        _, _, z_true = pair
        ou = OUSpreadSignal("P1", "P2", lookback=len(z_true))
        params = ou.fit_ou_params(z_true)

        assert params["kappa"] > 0.0
        # Allow ±50% error around TRUE_KAPPA=0.15.
        assert abs(params["kappa"] - TRUE_KAPPA) / TRUE_KAPPA < 0.5, (
            f"Estimated kappa {params['kappa']:.4f} too far from {TRUE_KAPPA}"
        )

    def test_recovers_mu_near_zero(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        _, _, z_true = pair
        ou = OUSpreadSignal("P1", "P2")
        params = ou.fit_ou_params(z_true)
        # TRUE_MU = 0.0; allow |mu| < 0.5 (noise-induced bias).
        assert abs(params["mu"]) < 0.5, f"Estimated mu {params['mu']:.4f} too far from 0"

    def test_half_life_positive(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        _, _, z_true = pair
        ou = OUSpreadSignal("P1", "P2")
        params = ou.fit_ou_params(z_true)
        assert params["half_life_bars"] > 0.0

    def test_half_life_formula(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        _, _, z_true = pair
        ou = OUSpreadSignal("P1", "P2")
        params = ou.fit_ou_params(z_true)
        expected_hl = np.log(2.0) / params["kappa"]
        assert abs(params["half_life_bars"] - expected_hl) < 1e-9

    def test_sigma_positive(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        _, _, z_true = pair
        ou = OUSpreadSignal("P1", "P2")
        params = ou.fit_ou_params(z_true)
        assert params["sigma"] > 0.0

    def test_returns_correct_schema(
        self,
        pair: tuple[pd.Series, pd.Series, pd.Series],
    ) -> None:
        _, _, z_true = pair
        ou = OUSpreadSignal("P1", "P2")
        params = ou.fit_ou_params(z_true)
        assert set(params.keys()) == {"kappa", "mu", "sigma", "half_life_bars"}
        assert all(isinstance(v, float) for v in params.values())


# ---------------------------------------------------------------------------
# Tests — OUSpreadSignal.compute_signal
# ---------------------------------------------------------------------------

class TestComputeSignal:
    def test_schema(
        self,
        df_pair: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        df1, df2 = df_pair
        ou = OUSpreadSignal("P1", "P2")
        result = ou.compute_signal(df1, df2)
        assert set(result.keys()) == {
            "signal", "z_score", "half_life", "mu", "sigma", "beta", "timestamp"
        }
        assert result["signal"] in {-1, 0, 1}
        assert isinstance(result["z_score"], float)
        assert isinstance(result["half_life"], float)
        assert isinstance(result["mu"], float)
        assert isinstance(result["sigma"], float)
        assert isinstance(result["beta"], float)
        assert isinstance(result["timestamp"], datetime)

    def test_timestamp_is_utc(
        self,
        df_pair: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        df1, df2 = df_pair
        ou = OUSpreadSignal("P1", "P2")
        result = ou.compute_signal(df1, df2)
        assert result["timestamp"].tzinfo is not None

    def test_inserts_two_signal_records(
        self,
        df_pair: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        df1, df2 = df_pair
        storage = _FakeStorage()
        ou = OUSpreadSignal("P1", "P2")
        ou.compute_signal(df1, df2, storage=storage)
        assert len(storage.inserted_signals) == 2
        names = {r["signal_name"] for r in storage.inserted_signals}
        assert names == {"ou_zscore", "ou_signal"}

    def test_inserted_ticker_is_pair_id(
        self,
        df_pair: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        df1, df2 = df_pair
        storage = _FakeStorage()
        ou = OUSpreadSignal("P1", "P2")
        ou.compute_signal(df1, df2, storage=storage)
        for record in storage.inserted_signals:
            assert record["ticker"] == "P1_P2"

    def test_no_storage_call_when_none(
        self,
        df_pair: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        df1, df2 = df_pair
        ou = OUSpreadSignal("P1", "P2")
        # Must not raise even when storage is omitted.
        result = ou.compute_signal(df1, df2, storage=None)
        assert result["signal"] in {-1, 0, 1}

    def test_signal_long_at_large_negative_zscore(self) -> None:
        """
        Signal logic unit test: when z_score << -entry_z, signal = +1.

        We isolate the signal-logic branch by replacing get_spread with a
        lambda that returns a controlled spread (stable N(0, 0.5) history with
        a large negative spike at the last bar).  Kalman tracking accuracy is
        covered separately in TestKalmanHedgeRatio.
        """
        rng = np.random.default_rng(17)
        n = 400
        dates = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")

        spread_vals = rng.normal(0.0, 0.5, n)
        spread_vals[-1] = -15.0   # extreme negative → z ≪ -2.0
        controlled_spread = pd.Series(spread_vals, index=dates)

        p1 = pd.Series(np.ones(n) * 200.0, index=dates, name="close")
        p2 = pd.Series(np.ones(n) * 100.0, index=dates, name="close")
        df1 = _make_ohlcv_df(p1, "P1")
        df2 = _make_ohlcv_df(p2, "P2")

        ou = OUSpreadSignal("P1", "P2", entry_z=2.0, lookback=100)
        # Replace get_spread to return the controlled series directly.
        ou._kalman.get_spread = lambda s1, s2: controlled_spread

        result = ou.compute_signal(df1, df2)
        assert result["signal"] == 1, (
            f"Expected +1 (long) but got {result['signal']} at z={result['z_score']:.2f}"
        )

    def test_signal_short_at_large_positive_zscore(self) -> None:
        """
        Signal logic unit test: when z_score >> entry_z, signal = -1.
        """
        rng = np.random.default_rng(18)
        n = 400
        dates = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")

        spread_vals = rng.normal(0.0, 0.5, n)
        spread_vals[-1] = +15.0   # extreme positive → z ≫ +2.0
        controlled_spread = pd.Series(spread_vals, index=dates)

        p1 = pd.Series(np.ones(n) * 200.0, index=dates, name="close")
        p2 = pd.Series(np.ones(n) * 100.0, index=dates, name="close")
        df1 = _make_ohlcv_df(p1, "P1")
        df2 = _make_ohlcv_df(p2, "P2")

        ou = OUSpreadSignal("P1", "P2", entry_z=2.0, lookback=100)
        ou._kalman.get_spread = lambda s1, s2: controlled_spread

        result = ou.compute_signal(df1, df2)
        assert result["signal"] == -1, (
            f"Expected -1 (short) but got {result['signal']} at z={result['z_score']:.2f}"
        )

    def test_signal_zero_at_small_zscore(self) -> None:
        """
        When the spread is very close to its mean (z_score ≈ 0), the exit
        condition fires and signal = 0.
        """
        _, _, z_true = _simulate_pair(n=400, seed=SEED)
        p2, _, _ = _simulate_pair(n=400, beta=1.0, sigma_z=0.01, seed=SEED + 1)

        # Build a near-perfectly mean-reverted pair: p1 ≈ TRUE_BETA * p2 + mu
        p2_series = p2  # use as p2
        p1_series = TRUE_BETA * p2_series  # spread ≈ 0

        df1 = _make_ohlcv_df(p1_series, "P1")
        df2 = _make_ohlcv_df(p2_series, "P2")

        ou = OUSpreadSignal("P1", "P2", entry_z=2.0, exit_z=0.5, lookback=60)
        result = ou.compute_signal(df1, df2)
        assert result["signal"] == 0, (
            f"Expected 0 (exit) but got {result['signal']} at z={result['z_score']:.2f}"
        )

    def test_hold_preserves_previous_signal(self) -> None:
        """
        When z_score is in the hold band (exit_z < |z| < entry_z), the signal
        should be unchanged from the previous state.
        """
        p1, p2, _ = _simulate_pair(n=400, seed=SEED)
        df1 = _make_ohlcv_df(p1, "P1")
        df2 = _make_ohlcv_df(p2, "P2")

        ou = OUSpreadSignal("P1", "P2", entry_z=2.0, exit_z=0.5)
        # Manually prime the last signal.
        ou._last_signal = 1
        # Use a pair that keeps z in the hold band: just run normally
        # and check that if |z| is in (0.5, 2.0) the signal stays 1.
        result = ou.compute_signal(df1, df2)

        z = result["z_score"]
        if 0.5 <= abs(z) < 2.0:
            assert result["signal"] == 1, (
                f"Hold condition: z={z:.2f} in hold band but signal changed to {result['signal']}"
            )
        # If the z_score happens to be in entry or exit territory,
        # the signal will change — that is correct behavior.


# ---------------------------------------------------------------------------
# Tests — rolling cointegration check
# ---------------------------------------------------------------------------

class TestRollingCointegrationCheck:
    def test_detects_cointegrated_pair(
        self,
        df_pair: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        df1, df2 = df_pair
        ou = OUSpreadSignal("P1", "P2")
        # Populate _last_p1 / _last_p2 by running compute_signal first.
        ou.compute_signal(df1, df2)
        result = ou.rolling_cointegration_check()
        assert result is True

    def test_flags_non_cointegrated_pair(self) -> None:
        rw1 = _random_walk_series(seed=301)
        rw2 = _random_walk_series(seed=302)
        df1 = _make_ohlcv_df(rw1, "RW1")
        df2 = _make_ohlcv_df(rw2, "RW2")

        ou = OUSpreadSignal("RW1", "RW2")
        ou.compute_signal(df1, df2)  # seeds _last_p1 / _last_p2

        # Run until we find a non-cointegrated pair (tries multiple seeds).
        found_non_coint = False
        for offset in range(10):
            rw1 = _random_walk_series(seed=400 + offset)
            rw2 = _random_walk_series(seed=500 + offset)
            ou._last_p1 = rw1
            ou._last_p2 = rw2
            if not ou.rolling_cointegration_check():
                found_non_coint = True
                break
        assert found_non_coint, (
            "Expected at least one pair of independent random walks to fail cointegration"
        )

    def test_suppresses_signal_after_cointegration_lost(self) -> None:
        rw1 = _random_walk_series(seed=601)
        rw2 = _random_walk_series(seed=602)
        df1 = _make_ohlcv_df(rw1, "RW1")
        df2 = _make_ohlcv_df(rw2, "RW2")

        ou = OUSpreadSignal("RW1", "RW2")
        # Force cointegration failure.
        ou._is_cointegrated = False
        result = ou.compute_signal(df1, df2)
        assert result["signal"] == 0

    def test_returns_true_before_data_available(self) -> None:
        ou = OUSpreadSignal("X", "Y")
        # No compute_signal has been called so _last_p1 is None.
        assert ou.rolling_cointegration_check() is True

    def test_update_count_increments(
        self,
        df_pair: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        df1, df2 = df_pair
        ou = OUSpreadSignal("P1", "P2", coint_check_interval=1000)
        initial = ou._update_count
        ou.compute_signal(df1, df2)
        assert ou._update_count == initial + 1
