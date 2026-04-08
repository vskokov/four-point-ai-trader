"""Kalman-filter adaptive hedge ratio for pairs trading.

Tracks a time-varying beta in the spread equation

    P1_t = beta_t * P2_t + epsilon_t,  epsilon ~ N(0, V)
    beta_t = beta_{t-1} + omega_t,     omega   ~ N(0, W)

using a Dynamic Linear Model implemented with filterpy.

The measurement matrix H = [[p2_t]] is updated every bar so the filter
correctly accounts for the changing regressor without lookahead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "models"


class KalmanHedgeRatio:
    """
    Online Kalman filter for adaptive beta (hedge ratio) estimation.

    Parameters
    ----------
    delta:
        Controls state (process) noise: W = delta / (1 - delta).
        Smaller delta → slower beta adaptation, less noise sensitivity.
    observation_noise:
        Measurement noise variance V = observation_noise.
    models_dir:
        Directory for save / load.  Defaults to ``trading_engine/models/``.
        Pass a ``tmp_path`` in tests to avoid touching the project tree.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        observation_noise: float = 1e-3,
        models_dir: Path | str | None = None,
    ) -> None:
        self.delta = delta
        self.observation_noise = observation_noise
        self._models_dir = Path(models_dir) if models_dir is not None else _MODELS_DIR

        # Scalar state: beta (dim_x=1), scalar observation: P1 (dim_z=1).
        self.model = KalmanFilter(dim_x=1, dim_z=1)
        self.model.x = np.array([[0.0]])            # initial beta estimate
        self.model.F = np.array([[1.0]])            # state transition: beta_t = beta_{t-1}
        self.model.H = np.array([[1.0]])            # placeholder; overwritten each bar
        self.model.P = np.eye(1) * 1e6             # large initial uncertainty
        self.model.R = np.array([[observation_noise]])
        self.model.Q = np.array([[delta / (1.0 - delta)]])  # W

        # Public state mirrors (updated after every update() call).
        self.beta: float = 0.0
        self.P: float = 1e6

        self.beta_history: list[float] = []

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, p1: float, p2: float) -> dict[str, float]:
        """
        Ingest one bar and return the posterior beta estimate.

        The sequence is:
          1. predict() — propagate state forward (prior)
          2. set H = [[p2]] — observation depends on current p2
          3. update(p1) — condition on new measurement

        Parameters
        ----------
        p1, p2:
            Close prices of the two legs at this bar.

        Returns
        -------
        dict with keys: beta (float), beta_var (float).
        """
        self.model.predict()
        self.model.H = np.array([[p2]])
        self.model.update(np.array([[p1]]))

        self.beta = float(self.model.x[0, 0])
        self.P = float(self.model.P[0, 0])
        self.beta_history.append(self.beta)

        return {"beta": self.beta, "beta_var": self.P}

    # ------------------------------------------------------------------
    # Batch spread computation (no side effects on self)
    # ------------------------------------------------------------------

    def get_spread(
        self,
        p1_series: pd.Series,
        p2_series: pd.Series,
    ) -> pd.Series:
        """
        Compute the online Kalman spread Z_t = P1_t - beta_t * P2_t
        over the full series without lookahead.

        A fresh filter (same hyper-parameters as this instance) is used
        so that repeated calls to ``get_spread`` are idempotent and do not
        corrupt the running state maintained by ``update()``.

        Parameters
        ----------
        p1_series, p2_series:
            Close-price series for the two legs, aligned index.

        Returns
        -------
        pd.Series indexed like *p1_series*.
        """
        fresh = KalmanHedgeRatio(
            delta=self.delta,
            observation_noise=self.observation_noise,
            models_dir=self._models_dir,
        )
        betas: list[float] = []
        for p1, p2 in zip(p1_series, p2_series):
            res = fresh.update(float(p1), float(p2))
            betas.append(res["beta"])

        # Store the final batch beta so callers (e.g. OUSpreadSignal) can
        # read it without having to run the filter again.  Does NOT touch
        # self.beta or self.beta_history so update()-based online usage
        # remains unaffected.
        self._batch_beta: float = betas[-1] if betas else 0.0
        self._batch_beta_var: float = fresh.P

        beta_series = pd.Series(betas, index=p1_series.index)
        spread = p1_series - beta_series * p2_series
        spread.name = "spread"
        return spread

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, ticker1: str, ticker2: str) -> None:
        """Persist to ``{models_dir}/kalman_{ticker1}_{ticker2}.pkl``."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        path = self._models_dir / f"kalman_{ticker1}_{ticker2}.pkl"
        joblib.dump(self, path)
        logger.info("kalman.save", ticker1=ticker1, ticker2=ticker2, path=str(path))

    def load(self, ticker1: str, ticker2: str) -> None:
        """
        Load state in-place from ``{models_dir}/kalman_{ticker1}_{ticker2}.pkl``.

        Raises
        ------
        FileNotFoundError
        """
        path = self._models_dir / f"kalman_{ticker1}_{ticker2}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No saved Kalman model for ({ticker1}, {ticker2}) at {path}"
            )
        loaded: KalmanHedgeRatio = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        logger.info("kalman.load", ticker1=ticker1, ticker2=ticker2, path=str(path))
