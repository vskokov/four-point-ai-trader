"""Multiplicative Weights Update (MWU) meta-agent.

Combines signals from hmm_regime, ou_spread, and llm_sentiment using the
Multiplicative Weights Update algorithm, conditioned on the current HMM
market regime.

For each regime r ∈ {0,1,2} a weight vector w^r ∈ R^3 is maintained.
After each decision round the per-signal losses are used to exponentially
down-weight signals that disagreed with the realised price direction, then
weights are renormalised.  The ensemble signal is the sign of the
confidence-weighted dot product of the current regime's weight vector with
the incoming signal values.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

# Default models directory — resolved relative to this source file so it works
# regardless of where the process is launched from.
_MODELS_DIR = Path(__file__).parent.parent / "models"
_WEIGHTS_FILENAME = "mwu_weights.npy"

# Canonical signal names — order defines column indices in the weight matrix.
_SIGNAL_NAMES: list[str] = ["hmm_regime", "ou_spread", "llm_sentiment"]

# Regime label lookup
_REGIME_LABELS: dict[int, str] = {0: "bear", 1: "neutral", 2: "bull"}


class MWUMetaAgent:
    """
    Multiplicative Weights Update meta-agent for combining trading signals.

    Parameters
    ----------
    eta:
        Learning rate (ε) for the MWU exponential update.  Higher values
        react faster to signal quality changes; lower values are more stable.
    n_signals:
        Number of input signals.  Matches ``len(signal_names)``.
    n_regimes:
        Number of HMM regimes.  Must match the HMMRegimeDetector setup.
    min_confidence:
        Minimum absolute score required to issue a directional signal.
        Scores below this threshold produce a neutral (0) decision.
    models_dir:
        Directory for ``mwu_weights.npy`` persistence.  Defaults to
        ``trading_engine/models/``.  Pass ``tmp_path`` in tests.
    """

    def __init__(
        self,
        eta: float = 0.1,
        n_signals: int = 3,
        n_regimes: int = 3,
        min_confidence: float = 0.3,
        models_dir: Path | str | None = None,
    ) -> None:
        self.eta = eta
        self.n_signals = n_signals
        self.n_regimes = n_regimes
        self.min_confidence = min_confidence
        self.signal_names: list[str] = _SIGNAL_NAMES[:n_signals]
        self._models_dir = Path(models_dir) if models_dir is not None else _MODELS_DIR

        # Uniform initialisation: shape (n_regimes, n_signals)
        self.weights: np.ndarray = np.full(
            (n_regimes, n_signals), 1.0 / n_signals, dtype=float
        )

        # History entries: {"timestamp": ..., "regime": ..., "weights": ndarray}
        self.weight_history: list[dict[str, Any]] = []

        # Pending decisions waiting for outcome evaluation.
        # Maps (ticker, decision_time_isoformat) -> decision dict
        self._pending: dict[tuple[str, str], dict[str, Any]] = {}

        # Accumulated update records for performance reporting.
        # Each entry: {"ticker", "regime", "signals", "actual_direction",
        #              "losses", "timestamp"}
        self._update_log: list[dict[str, Any]] = []

        self._load_weights()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _weights_path(self) -> Path:
        return self._models_dir / _WEIGHTS_FILENAME

    def _load_weights(self) -> None:
        """Load persisted weight matrix if it exists."""
        path = self._weights_path()
        if path.exists():
            try:
                loaded = np.load(str(path))
                if loaded.shape == (self.n_regimes, self.n_signals):
                    self.weights = loaded.astype(float)
                    logger.info(
                        "mwu.weights.loaded",
                        path=str(path),
                        shape=list(self.weights.shape),
                    )
                else:
                    logger.warning(
                        "mwu.weights.shape_mismatch",
                        expected=(self.n_regimes, self.n_signals),
                        found=list(loaded.shape),
                    )
            except Exception as exc:
                logger.warning(
                    "mwu.weights.load_error",
                    path=str(path),
                    exc=str(exc),
                )

    def _save_weights(self) -> None:
        """Persist weight matrix to disk, creating the models directory if needed."""
        path = self._weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.weights)
        logger.debug("mwu.weights.saved", path=str(path))

    # ------------------------------------------------------------------
    # Core signal handling
    # ------------------------------------------------------------------

    def _resolve_signals(
        self, signals: dict[str, dict[str, Any]]
    ) -> list[dict[str, float]]:
        """
        Return a list (one entry per signal name) with guaranteed ``signal``
        and ``confidence`` keys.  Missing signals are filled with neutral
        defaults (signal=0, confidence=0).
        """
        resolved: list[dict[str, float]] = []
        for name in self.signal_names:
            if name in signals and signals[name] is not None:
                entry = signals[name]
                resolved.append(
                    {
                        "signal": float(entry.get("signal", 0)),
                        "confidence": float(entry.get("confidence", 0.0)),
                    }
                )
            else:
                resolved.append({"signal": 0.0, "confidence": 0.0})
        return resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        ticker: str,
        signals: dict[str, dict[str, Any]],
        regime: int,
    ) -> dict[str, Any]:
        """
        Produce an ensemble trading decision.

        Parameters
        ----------
        ticker:
            Equity symbol being traded.
        signals:
            Mapping of signal name → dict with at least ``"signal"`` (int,
            −1/0/+1) and ``"confidence"`` (float, 0–1).  Missing keys are
            treated as neutral.
        regime:
            Current HMM regime index (0=bear, 1=neutral, 2=bull).

        Returns
        -------
        dict with keys: ticker, final_signal, score, regime, weights,
        timestamp.
        """
        resolved = self._resolve_signals(signals)
        w = self.weights[regime]

        score: float = sum(
            w[k] * resolved[k]["signal"] * resolved[k]["confidence"]
            for k in range(self.n_signals)
        )

        if abs(score) < self.min_confidence:
            final_signal = 0
        elif score > 0:
            final_signal = 1
        else:
            final_signal = -1

        weights_dict = dict(zip(self.signal_names, w.tolist()))
        timestamp = datetime.now(tz=timezone.utc)

        logger.info(
            "mwu.decide",
            ticker=ticker,
            regime=regime,
            regime_label=_REGIME_LABELS.get(regime, "unknown"),
            signals={
                name: {
                    "signal": resolved[k]["signal"],
                    "confidence": resolved[k]["confidence"],
                }
                for k, name in enumerate(self.signal_names)
            },
            weights=weights_dict,
            score=round(score, 6),
            final_signal=final_signal,
        )

        return {
            "ticker": ticker,
            "final_signal": final_signal,
            "score": score,
            "regime": regime,
            "weights": weights_dict,
            "timestamp": timestamp,
        }

    def update_weights(
        self,
        ticker: str,
        signals_t: dict[str, dict[str, Any]],
        regime_t: int,
        actual_direction: int,
    ) -> None:
        """
        Apply one MWU update step for the given regime.

        Parameters
        ----------
        ticker:
            Equity symbol (logged for traceability).
        signals_t:
            Signal dict at decision time (same format as ``decide``).
        regime_t:
            Regime index active at decision time.
        actual_direction:
            Realised price direction: +1 (up), −1 (down), 0 (flat).
        """
        resolved = self._resolve_signals(signals_t)

        losses: list[float] = []
        for entry in resolved:
            sig_val = entry["signal"]
            if sig_val == 0:
                loss = 0.5
            elif sig_val == actual_direction:
                loss = 0.0
            else:
                loss = 1.0
            losses.append(loss)

        # MWU multiplicative update for regime_t only
        for k, loss in enumerate(losses):
            self.weights[regime_t, k] *= np.exp(-self.eta * loss)

        # Renormalise
        row_sum = self.weights[regime_t].sum()
        if row_sum > 0:
            self.weights[regime_t] /= row_sum
        else:
            # Fallback: uniform reset if all weights collapsed to zero
            self.weights[regime_t] = np.full(self.n_signals, 1.0 / self.n_signals)

        timestamp = datetime.now(tz=timezone.utc)

        # History snapshot
        self.weight_history.append(
            {
                "timestamp": timestamp,
                "ticker": ticker,
                "regime": regime_t,
                "weights": self.weights.copy(),
            }
        )

        # Update log for performance reporting
        self._update_log.append(
            {
                "timestamp": timestamp,
                "ticker": ticker,
                "regime": regime_t,
                "signals": {
                    name: resolved[k]["signal"]
                    for k, name in enumerate(self.signal_names)
                },
                "actual_direction": actual_direction,
                "losses": dict(zip(self.signal_names, losses)),
            }
        )

        self._save_weights()

        logger.info(
            "mwu.weights.updated",
            ticker=ticker,
            regime=regime_t,
            regime_label=_REGIME_LABELS.get(regime_t, "unknown"),
            actual_direction=actual_direction,
            losses=dict(zip(self.signal_names, losses)),
            new_weights={
                r_label: dict(
                    zip(self.signal_names, self.weights[r].tolist())
                )
                for r, r_label in _REGIME_LABELS.items()
                if r < self.n_regimes
            },
        )

    def get_actual_direction(
        self,
        ticker: str,
        decision_time: datetime,
        horizon_bars: int = 1,
        storage: Any = None,
    ) -> int:
        """
        Determine the realised price direction after *horizon_bars* bars.

        Parameters
        ----------
        ticker:
            Equity symbol.
        decision_time:
            Timestamp of the original decision.
        horizon_bars:
            Number of 1-minute bars to look ahead.  The default of 1
            corresponds to the next bar's close.
        storage:
            ``Storage`` instance.  Must be provided — this method does not
            hold a module-level reference to storage.

        Returns
        -------
        int
            +1 if price rose, −1 if price fell, 0 if change < 0.05 %.
        """
        if storage is None:
            logger.warning(
                "mwu.get_actual_direction.no_storage",
                ticker=ticker,
                decision_time=str(decision_time),
            )
            return 0

        # Fetch a generous window around the decision time
        start = decision_time - timedelta(minutes=5)
        end = decision_time + timedelta(minutes=horizon_bars + 10)

        df: pd.DataFrame = storage.query_ohlcv(ticker, start, end)

        if df.empty:
            logger.warning(
                "mwu.get_actual_direction.no_data",
                ticker=ticker,
                decision_time=str(decision_time),
            )
            return 0

        # Ensure timezone-aware comparison
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(timezone.utc)

        decision_ts = decision_time if decision_time.tzinfo else decision_time.replace(
            tzinfo=timezone.utc
        )

        # Find the bar at or just before decision_time
        before = df[df["time"] <= decision_ts]
        after = df[df["time"] > decision_ts]

        if before.empty or after.empty:
            logger.warning(
                "mwu.get_actual_direction.insufficient_bars",
                ticker=ticker,
                n_before=len(before),
                n_after=len(after),
            )
            return 0

        price_at = float(before.iloc[-1]["close"])
        # Take the close horizon_bars ahead
        horizon_idx = min(horizon_bars - 1, len(after) - 1)
        price_after = float(after.iloc[horizon_idx]["close"])

        if price_at == 0:
            return 0

        pct_change = (price_after - price_at) / price_at

        if pct_change > 0.0005:
            direction = 1
        elif pct_change < -0.0005:
            direction = -1
        else:
            direction = 0

        logger.info(
            "mwu.get_actual_direction",
            ticker=ticker,
            decision_time=str(decision_time),
            price_at=price_at,
            price_after=price_after,
            pct_change=round(pct_change, 6),
            direction=direction,
        )
        return direction

    def performance_report(self) -> dict[str, Any]:
        """
        Compute per-signal win rates and per-regime weight evolution.

        Returns
        -------
        dict with keys:
            n_updates, per_signal_win_rate, per_regime_win_rate,
            current_weights, weight_history_summary.
        """
        if not self._update_log:
            report: dict[str, Any] = {
                "n_updates": 0,
                "per_signal_win_rate": {name: None for name in self.signal_names},
                "per_regime_win_rate": {
                    _REGIME_LABELS[r]: None for r in range(self.n_regimes)
                },
                "current_weights": {
                    _REGIME_LABELS[r]: dict(
                        zip(self.signal_names, self.weights[r].tolist())
                    )
                    for r in range(self.n_regimes)
                },
                "weight_history_length": 0,
            }
            logger.info("mwu.performance_report", **report)
            return report

        # Per-signal win rate: fraction of rounds where signal matched actual
        signal_correct: dict[str, int] = {name: 0 for name in self.signal_names}
        signal_total: dict[str, int] = {name: 0 for name in self.signal_names}
        regime_correct: dict[int, int] = {r: 0 for r in range(self.n_regimes)}
        regime_total: dict[int, int] = {r: 0 for r in range(self.n_regimes)}

        for entry in self._update_log:
            actual = entry["actual_direction"]
            regime = entry["regime"]
            regime_total[regime] += 1

            any_correct = False
            for name in self.signal_names:
                sig_val = entry["signals"].get(name, 0)
                signal_total[name] += 1
                if sig_val != 0 and sig_val == actual:
                    signal_correct[name] += 1
                    any_correct = True
            if any_correct:
                regime_correct[regime] += 1

        per_signal_win_rate = {
            name: (
                signal_correct[name] / signal_total[name]
                if signal_total[name] > 0
                else None
            )
            for name in self.signal_names
        }

        per_regime_win_rate = {
            _REGIME_LABELS[r]: (
                regime_correct[r] / regime_total[r]
                if regime_total[r] > 0
                else None
            )
            for r in range(self.n_regimes)
        }

        current_weights = {
            _REGIME_LABELS[r]: dict(
                zip(self.signal_names, self.weights[r].tolist())
            )
            for r in range(self.n_regimes)
        }

        report = {
            "n_updates": len(self._update_log),
            "per_signal_win_rate": per_signal_win_rate,
            "per_regime_win_rate": per_regime_win_rate,
            "current_weights": current_weights,
            "weight_history_length": len(self.weight_history),
        }

        logger.info(
            "mwu.performance_report",
            report=json.dumps(report, default=str),
        )
        return report

    def scheduled_update(
        self,
        ticker: str,
        signals: dict[str, dict[str, Any]],
        regime: int,
        storage: Any = None,
        horizon_bars: int = 1,
    ) -> dict[str, Any]:
        """
        Called after each trading bar to run one full online-learning cycle.

        Produces a decision, stores it as a pending record, and — if there are
        pending decisions that have passed their horizon — evaluates outcomes
        and calls ``update_weights``.

        Parameters
        ----------
        ticker:
            Equity symbol.
        signals:
            Current signal readings.
        regime:
            Current HMM regime index.
        storage:
            ``Storage`` instance used by ``get_actual_direction``.
        horizon_bars:
            Look-ahead bars used when evaluating past decisions.

        Returns
        -------
        The decision dict from ``decide``.
        """
        decision = self.decide(ticker, signals, regime)
        decision_key = (ticker, decision["timestamp"].isoformat())
        self._pending[decision_key] = {
            "decision": decision,
            "signals_t": signals,
            "regime_t": regime,
            "horizon_bars": horizon_bars,
        }

        # Evaluate any pending decisions whose horizon has elapsed
        now = datetime.now(tz=timezone.utc)
        keys_to_remove: list[tuple[str, str]] = []

        for key, pending in self._pending.items():
            dec_time = pending["decision"]["timestamp"]
            # horizon_bars is treated as minutes here for scheduling purposes
            if (now - dec_time) >= timedelta(minutes=pending["horizon_bars"]):
                actual = self.get_actual_direction(
                    ticker=key[0],
                    decision_time=dec_time,
                    horizon_bars=pending["horizon_bars"],
                    storage=storage,
                )
                self.update_weights(
                    ticker=key[0],
                    signals_t=pending["signals_t"],
                    regime_t=pending["regime_t"],
                    actual_direction=actual,
                )
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._pending[key]

        return decision
