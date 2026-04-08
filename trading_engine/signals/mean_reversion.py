"""Ornstein-Uhlenbeck mean-reversion signal for cointegrated pairs.

Two classes:

CointegrationTest — wrapper around Engle-Granger and Johansen tests.

OUSpreadSignal — end-to-end signal generator:
  1. Adaptive hedge ratio via Kalman filter (kalman_pairs.KalmanHedgeRatio)
  2. OU parameter estimation via OLS on the discrete-time form
  3. Z-score thresholding to produce ±1 / 0 trading signals
  4. Periodic cointegration health checks
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from trading_engine.signals.kalman_pairs import KalmanHedgeRatio
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# Cointegration test
# ---------------------------------------------------------------------------

class CointegrationTest:
    """
    Wrapper around Engle-Granger and Johansen cointegration tests.

    The primary decision rule is the Engle-Granger p-value threshold.
    The Johansen trace statistic is returned for informational purposes.
    """

    def test(self, p1: pd.Series, p2: pd.Series) -> dict[str, Any]:
        """
        Run both cointegration tests on two price series.

        Parameters
        ----------
        p1, p2:
            Price series (typically daily close).  Must share the same index
            and contain no NaNs.

        Returns
        -------
        dict with keys:
            cointegrated          : bool   — True if EG p-value < 0.05
            eg_pvalue             : float
            johansen_trace_stat   : float  — trace statistic for r=0
            beta_ols              : float  — OLS regression slope (p1 ~ p2)
        """
        v1 = np.asarray(p1, dtype=float)
        v2 = np.asarray(p2, dtype=float)

        # Engle-Granger: regresses p1 on p2, tests residuals for unit root.
        _, eg_pvalue, _ = coint(v1, v2)

        # Johansen trace test.
        data = np.column_stack([v1, v2])
        johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)
        johansen_trace_stat = float(johansen_result.lr1[0])

        # OLS beta: p1 = alpha + beta * p2 + eps
        X_ols = sm.add_constant(v2)
        ols_result = sm.OLS(v1, X_ols).fit()
        beta_ols = float(ols_result.params[1])

        cointegrated = bool(eg_pvalue < 0.05)

        logger.info(
            "coint_test",
            cointegrated=cointegrated,
            eg_pvalue=float(eg_pvalue),
            johansen_trace=johansen_trace_stat,
            beta_ols=beta_ols,
        )

        return {
            "cointegrated": cointegrated,
            "eg_pvalue": float(eg_pvalue),
            "johansen_trace_stat": johansen_trace_stat,
            "beta_ols": beta_ols,
        }


# ---------------------------------------------------------------------------
# OU spread signal
# ---------------------------------------------------------------------------

class OUSpreadSignal:
    """
    Mean-reversion trading signal for a cointegrated pair.

    Architecture
    ------------
    - Kalman filter estimates the time-varying hedge ratio beta online.
    - OU parameters (kappa, mu, sigma) are estimated by OLS on the discrete
      autoregressive form of the spread over the last *lookback* bars.
    - Z-score thresholds produce long (+1), short (-1), or flat (0) signals.
    - Cointegration is re-verified every ``coint_check_interval`` calls to
      ``compute_signal``; if it fails the signal is forced to zero.

    Parameters
    ----------
    ticker1, ticker2:
        Equity symbols forming the pair.  Used as identifiers in signal_log
        and for Kalman model persistence.
    entry_z:
        Z-score magnitude that triggers a new position (default 2.0).
    exit_z:
        Z-score magnitude below which an open position is closed (default 0.5).
    lookback:
        Number of bars used for rolling OU parameter estimation.
    coint_check_interval:
        Number of ``compute_signal`` calls between cointegration re-checks.
    models_dir:
        Directory for Kalman model persistence.  Injectable for tests.
    """

    def __init__(
        self,
        ticker1: str,
        ticker2: str,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 60,
        coint_check_interval: int = 1200,
        models_dir: Path | str | None = None,
    ) -> None:
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self._coint_check_interval = coint_check_interval
        self._models_dir = Path(models_dir) if models_dir is not None else _MODELS_DIR

        self._kalman = KalmanHedgeRatio(models_dir=self._models_dir)
        self._coint_test = CointegrationTest()

        # Running state
        self._last_signal: int = 0
        self._is_cointegrated: bool = True
        self._update_count: int = 0
        self._last_p1: pd.Series | None = None
        self._last_p2: pd.Series | None = None

    # ------------------------------------------------------------------
    # OU parameter estimation
    # ------------------------------------------------------------------

    def fit_ou_params(self, spread: pd.Series) -> dict[str, float]:
        """
        Fit OU parameters from the spread via OLS on the discrete form.

        Discrete-time OU:
            Z_t = a + b * Z_{t-1} + eps

        Maps to continuous parameters with dt = 1 bar:
            kappa = 1 - b       (mean-reversion speed)
            mu    = a / kappa   (long-run mean)
            sigma = std(eps)    (noise per bar)

        Parameters
        ----------
        spread:
            Spread series Z_t = P1 - beta * P2.

        Returns
        -------
        dict with keys: kappa, mu, sigma, half_life_bars.
        """
        y = spread.iloc[1:].values.astype(float)
        x = spread.iloc[:-1].values.astype(float)
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()

        a = float(ols.params[0])   # intercept
        b = float(ols.params[1])   # AR(1) coefficient

        kappa = max(1.0 - b, 1e-8)      # ensure positive mean-reversion
        mu = a / kappa
        sigma = float(ols.resid.std())
        half_life = float(np.log(2.0) / kappa)

        logger.info(
            "ou.params",
            ticker1=self.ticker1,
            ticker2=self.ticker2,
            kappa=kappa,
            mu=mu,
            sigma=sigma,
            half_life=half_life,
        )

        return {
            "kappa": kappa,
            "mu": mu,
            "sigma": sigma,
            "half_life_bars": half_life,
        }

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def compute_signal(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        storage: Any = None,
    ) -> dict[str, Any]:
        """
        Compute the current mean-reversion signal for the pair.

        Parameters
        ----------
        df1, df2:
            OHLCV DataFrames for ticker1 and ticker2 (must have a ``close``
            column and matching length).
        storage:
            Optional ``Storage`` instance.  If provided, the z-score and
            signal are written to ``signal_log``.

        Returns
        -------
        dict with keys:
            signal    : int   — +1 long spread, -1 short spread, 0 flat
            z_score   : float
            half_life : float — half-life in bars
            mu        : float — OU long-run mean
            sigma     : float — OU noise per bar
            beta      : float — current Kalman beta
            timestamp : datetime
        """
        p1 = df1["close"].astype(float)
        p2 = df2["close"].astype(float)

        # Store for periodic cointegration check.
        self._last_p1 = p1
        self._last_p2 = p2
        self._update_count += 1

        # Periodic cointegration check.
        if self._update_count % self._coint_check_interval == 0:
            self.rolling_cointegration_check()

        timestamp = datetime.now(tz=timezone.utc)

        if not self._is_cointegrated:
            logger.warning(
                "ou.signal_suppressed",
                reason="cointegration_lost",
                ticker1=self.ticker1,
                ticker2=self.ticker2,
            )
            self._last_signal = 0
            return {
                "signal": 0,
                "z_score": 0.0,
                "half_life": 0.0,
                "mu": 0.0,
                "sigma": 0.0,
                "beta": float(self._kalman.beta),
                "timestamp": timestamp,
            }

        # Compute Kalman spread over full history.
        # get_spread() stores the final beta in _batch_beta as a side effect.
        spread = self._kalman.get_spread(p1, p2)
        beta = getattr(self._kalman, "_batch_beta", 0.0)

        # Fit OU params on the rolling lookback window.
        window = spread.iloc[-self.lookback :] if len(spread) >= self.lookback else spread
        ou_params = self.fit_ou_params(window)
        mu = ou_params["mu"]
        sigma = ou_params["sigma"]
        half_life = ou_params["half_life_bars"]

        # Z-score at the latest bar.
        if sigma < 1e-10:
            z_score = 0.0
        else:
            z_score = float((spread.iloc[-1] - mu) / sigma)

        # Signal logic.
        if z_score < -self.entry_z:
            signal = 1          # spread too low → long
        elif z_score > self.entry_z:
            signal = -1         # spread too high → short
        elif abs(z_score) < self.exit_z:
            signal = 0          # mean-reversion complete → exit
        else:
            signal = self._last_signal   # within band → hold
        self._last_signal = signal

        logger.info(
            "ou.signal",
            ticker1=self.ticker1,
            ticker2=self.ticker2,
            signal=signal,
            z_score=z_score,
            half_life=half_life,
            beta=beta,
        )

        # Persist to signal_log if storage provided.
        if storage is not None:
            pair_id = f"{self.ticker1}_{self.ticker2}"
            storage.insert_signal([
                {
                    "time": timestamp,
                    "ticker": pair_id,
                    "signal_name": "ou_zscore",
                    "value": z_score,
                    "metadata": {
                        "mu": mu,
                        "sigma": sigma,
                        "beta": beta,
                        "half_life": half_life,
                    },
                },
                {
                    "time": timestamp,
                    "ticker": pair_id,
                    "signal_name": "ou_signal",
                    "value": float(signal),
                    "metadata": {"z_score": z_score},
                },
            ])

        return {
            "signal": signal,
            "z_score": z_score,
            "half_life": half_life,
            "mu": mu,
            "sigma": sigma,
            "beta": beta,
            "timestamp": timestamp,
        }

    # ------------------------------------------------------------------
    # Periodic cointegration check
    # ------------------------------------------------------------------

    def rolling_cointegration_check(self, interval_bars: int = 1200) -> bool:
        """
        Re-run the cointegration test on the most recently seen price series.

        Designed to be called periodically from ``compute_signal`` (every
        ``coint_check_interval`` bars) or directly in tests with
        ``interval_bars=1`` to force an immediate check.

        If cointegration is lost the signal is forced to zero on the next
        ``compute_signal`` call.

        Parameters
        ----------
        interval_bars:
            Minimum bars since last check before a re-test is performed.
            Pass 1 to force an immediate re-test regardless of count.

        Returns
        -------
        bool — True if the pair is currently cointegrated.
        """
        if self._last_p1 is None or self._last_p2 is None:
            return self._is_cointegrated

        result = self._coint_test.test(self._last_p1, self._last_p2)
        self._is_cointegrated = result["cointegrated"]

        if not self._is_cointegrated:
            logger.warning(
                "ou.cointegration_lost",
                ticker1=self.ticker1,
                ticker2=self.ticker2,
                eg_pvalue=result["eg_pvalue"],
            )

        return self._is_cointegrated
