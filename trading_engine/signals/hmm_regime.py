"""HMM-based market regime detector.

Detects market regimes (bear / neutral / bull) from OHLCV data using a
Gaussian Hidden Markov Model.  State labels are assigned deterministically
after fitting by ranking hidden states on mean log-return, so the mapping
is stable regardless of the arbitrary internal state ordering that EM
produces.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

# Default directory for persisted models — relative to the package root.
_MODELS_DIR = Path(__file__).parent.parent / "models"

# Rank-to-label: rank 0 = lowest mean return = bear, 2 = highest = bull.
_LABEL_MAP: dict[int, str] = {0: "bear", 1: "neutral", 2: "bull"}


class HMMRegimeDetector:
    """
    Gaussian HMM regime detector with deterministic state labelling.

    Parameters
    ----------
    n_states:
        Number of hidden states.  Default 3 (bear / neutral / bull).
    n_iter:
        Maximum EM iterations per fit call.
    covariance_type:
        HMM covariance structure passed to ``GaussianHMM``.
    lookback_days:
        Training window length and online-buffer capacity in trading days.
    refit_every:
        Number of new bars ingested before an automatic re-fit in online mode.
    models_dir:
        Directory for ``save`` / ``load``.  Defaults to
        ``trading_engine/models/``.  Pass a ``tmp_path`` in tests.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 1000,
        covariance_type: str = "full",
        lookback_days: int = 252,
        refit_every: int = 20,
        models_dir: Path | str | None = None,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.lookback_days = lookback_days
        self.refit_every = refit_every
        self._models_dir = Path(models_dir) if models_dir is not None else _MODELS_DIR

        self.model = GaussianHMM(
            n_components=n_states,
            n_iter=n_iter,
            covariance_type=covariance_type,
            random_state=42,
        )
        self.is_fitted: bool = False
        # Maps HMM internal state int -> semantic label ("bear"/"neutral"/"bull").
        self.state_labels: dict[int, str] = {}
        self._ticker: str | None = None

        # Rolling bar buffer for online updates.
        self._online_buffer: deque[dict[str, Any]] = deque(maxlen=lookback_days)
        self._bars_since_refit: int = 0

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build a standardised (T, 3) feature matrix from an OHLCV DataFrame.

        Features
        --------
        log_return    : log(close_t / close_{t-1})
        rolling_vol   : 10-day rolling standard deviation of log_return
        volume_zscore : (volume − 20-day mean) / 20-day std

        All three columns are standardised to zero mean and unit variance.
        Rows containing NaN are dropped before standardisation.

        Parameters
        ----------
        df:
            DataFrame with at least ``close`` and ``volume`` columns.

        Returns
        -------
        np.ndarray of shape (T, 3).
        """
        work = df.copy()
        work["log_return"] = np.log(work["close"] / work["close"].shift(1))
        work["rolling_vol"] = work["log_return"].rolling(10).std()
        vol_mean = work["volume"].rolling(20).mean()
        vol_std = work["volume"].rolling(20).std()
        work["volume_zscore"] = (work["volume"] - vol_mean) / vol_std

        feat = work[["log_return", "rolling_vol", "volume_zscore"]].dropna()
        X = feat.to_numpy(dtype=np.float64)

        # Standardise column-wise.
        col_means = X.mean(axis=0)
        col_stds = X.std(axis=0)
        col_stds[col_stds == 0.0] = 1.0  # guard against constant columns
        X = (X - col_means) / col_stds
        return X

    # ------------------------------------------------------------------
    # State-label assignment (deterministic, post-hoc)
    # ------------------------------------------------------------------

    def _assign_state_labels(self, X: np.ndarray) -> None:
        """
        Rank HMM states by mean log-return (column 0) and assign semantic
        labels.  Called after every fit so labels are always consistent.
        """
        states = self.model.predict(X)
        state_means = np.array([
            X[states == s, 0].mean() if (states == s).any() else 0.0
            for s in range(self.n_states)
        ])
        # rank_order[i] = HMM state index with the i-th smallest mean return.
        rank_order = np.argsort(state_means)
        self.state_labels = {
            int(hmm_state): _LABEL_MAP.get(rank, f"state_{rank}")
            for rank, hmm_state in enumerate(rank_order)
        }

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        storage: Any,
    ) -> None:
        """
        Fetch OHLCV, fit the HMM, assign state labels, and persist to disk.

        Parameters
        ----------
        ticker:
            Equity symbol.
        start, end:
            Inclusive date range for the training window.
        storage:
            Object with a ``query_ohlcv(ticker, start, end) -> DataFrame``
            method (i.e. a ``Storage`` instance).
        """
        df = storage.query_ohlcv(ticker, start, end)
        if df.empty:
            raise ValueError(f"No OHLCV data for {ticker} in [{start}, {end}]")

        X = self._prepare_features(df)
        min_rows = self.n_states * 10
        if len(X) < min_rows:
            raise ValueError(
                f"Insufficient data: {len(X)} rows after NaN removal, "
                f"need ≥ {min_rows} for a {self.n_states}-state HMM."
            )

        logger.info(
            "hmm.fit.start",
            ticker=ticker,
            rows=len(X),
            n_states=self.n_states,
        )

        self.model.fit(X)
        self.is_fitted = True
        self._ticker = ticker

        self._assign_state_labels(X)

        logger.info(
            "hmm.fit.done",
            ticker=ticker,
            state_labels=self.state_labels,
            transmat=self.model.transmat_.tolist(),
            state_means=self.model.means_.tolist(),
        )

        self.save(ticker)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_regime(
        self,
        ticker: str,
        df: pd.DataFrame | None = None,
        storage: Any = None,
    ) -> dict[str, Any]:
        """
        Return the current market regime for *ticker*.

        Uses the forward algorithm (``predict_proba``) to compute posterior
        state probabilities P(s_t | r_{1:t}) and reports the distribution at
        the last timestep.

        Parameters
        ----------
        ticker:
            Equity symbol.
        df:
            Pre-fetched OHLCV DataFrame.  If *None*, the last
            ``lookback_days`` are fetched from *storage*.
        storage:
            ``Storage`` instance.  Required when *df* is *None*.
            Also used to persist the result via ``insert_regime``.

        Returns
        -------
        dict with keys:
            regime    : int   — HMM state index of the most likely state
            label     : str   — "bear" / "neutral" / "bull"
            probs     : list[float] — posterior prob per HMM state
            timestamp : datetime
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.  Call fit() or load() first.")

        if df is None:
            if storage is None:
                raise ValueError("Either df or storage must be provided.")
            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(days=int(self.lookback_days * 1.4))
            df = storage.query_ohlcv(ticker, start, end)

        X = self._prepare_features(df)
        if len(X) == 0:
            raise ValueError("No valid feature rows after NaN removal.")

        # predict_proba runs the forward algorithm: shape (T, n_states).
        probs_seq: np.ndarray = self.model.predict_proba(X)
        last_probs = probs_seq[-1]

        hmm_state = int(np.argmax(last_probs))
        label = self.state_labels.get(hmm_state, f"state_{hmm_state}")
        timestamp = datetime.now(tz=timezone.utc)

        result: dict[str, Any] = {
            "regime": hmm_state,
            "label": label,
            "probs": last_probs.tolist(),
            "timestamp": timestamp,
        }

        logger.info(
            "hmm.predict",
            ticker=ticker,
            regime=hmm_state,
            label=label,
            probs=result["probs"],
        )

        if storage is not None:
            storage.insert_regime([{
                "time": timestamp,
                "ticker": ticker,
                "regime": hmm_state,
                "regime_probs": {str(s): float(p) for s, p in enumerate(last_probs)},
            }])

        return result

    # ------------------------------------------------------------------
    # Online partial-fit
    # ------------------------------------------------------------------

    def partial_fit_online(self, new_bar: dict[str, Any]) -> None:
        """
        Ingest one new OHLCV bar and re-fit every ``refit_every`` bars.

        The bar is added to a rolling deque of size ``lookback_days``.
        When the refit threshold is reached the model is re-trained on the
        full buffer and the state labels are re-assigned.

        Parameters
        ----------
        new_bar:
            Dict containing at least: time, open, high, low, close, volume.
        """
        self._online_buffer.append(new_bar)
        self._bars_since_refit += 1

        min_rows = self.n_states * 10
        if (
            self._bars_since_refit >= self.refit_every
            and len(self._online_buffer) >= min_rows
        ):
            df = pd.DataFrame(list(self._online_buffer))
            X = self._prepare_features(df)
            if len(X) >= min_rows:
                logger.info(
                    "hmm.online_refit",
                    ticker=self._ticker,
                    buffer_size=len(self._online_buffer),
                )
                self.model.fit(X)
                self.is_fitted = True
                self._assign_state_labels(X)
                self._bars_since_refit = 0
                if self._ticker:
                    self.save(self._ticker)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, ticker: str) -> None:
        """Persist the detector to ``{models_dir}/hmm_{ticker}.pkl``."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        path = self._models_dir / f"hmm_{ticker}.pkl"
        joblib.dump(self, path)
        logger.info("hmm.save", ticker=ticker, path=str(path))

    def load(self, ticker: str) -> None:
        """
        Load model state in-place from ``{models_dir}/hmm_{ticker}.pkl``.

        Raises
        ------
        FileNotFoundError
            If no saved model exists for *ticker*.
        """
        path = self._models_dir / f"hmm_{ticker}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No saved HMM model for '{ticker}' at {path}"
            )
        loaded: HMMRegimeDetector = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        logger.info("hmm.load", ticker=ticker, path=str(path))


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_current_regime(ticker: str, storage: Any | None = None) -> dict[str, Any]:
    """
    Return the current regime for *ticker*, loading or fitting as needed.

    If a persisted model exists it is loaded; otherwise the model is trained
    on the last ~1.4 × 252 calendar days of OHLCV data.

    Parameters
    ----------
    ticker:
        Equity symbol.
    storage:
        ``Storage`` instance.  If *None*, one is constructed lazily from
        ``DB_URL`` in the environment so that importing this module never
        touches credentials.

    Returns
    -------
    dict with keys: regime, label, probs, timestamp.
    """
    if storage is None:
        # Lazy import to avoid crashing at import time when env vars are absent.
        from trading_engine.data.storage import Storage  # noqa: PLC0415
        import trading_engine.config.settings as _settings  # noqa: PLC0415
        storage = Storage(_settings.DB_URL)

    detector = HMMRegimeDetector()
    model_path = _MODELS_DIR / f"hmm_{ticker}.pkl"

    if model_path.exists():
        detector.load(ticker)
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=int(252 * 1.4))
        detector.fit(ticker, start, end, storage)

    return detector.predict_regime(ticker, storage=storage)
