"""
MWU weight trajectory analysis.

``trade_log.mwu_weights`` stores a snapshot of the regime-specific weight
dict at the moment of every decision, giving a free time-series of how MWU
adapted to signal quality.

Public API
----------
extract_weight_history(df)      -> pd.DataFrame
    Unpack mwu_weights JSON column into a tidy time-series.

summarise_weight_evolution(wdf) -> dict
    Identify significant drifts and collapsed signals.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_SIGNAL_NAMES = ("hmm_regime", "ou_spread", "llm_sentiment", "analyst_recs")
_REGIME_LABELS = {0: "bear", 1: "neutral", 2: "bull"}

# Initial priors for drift computation
_INITIAL_WEIGHTS: dict[str, float] = {
    "hmm_regime":    2 / 7,
    "ou_spread":     2 / 7,
    "llm_sentiment": 2 / 7,
    "analyst_recs":  1 / 7,
}

_DRIFT_THRESHOLD_PCT = 20.0   # flag drifts larger than ±20 %
_COLLAPSE_THRESHOLD  = 0.05   # flag weights below 5 % (effectively zero)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def extract_weight_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpack the ``mwu_weights`` JSON column into a tidy time-series.

    Each row in ``df`` that has a non-null ``mwu_weights`` dict contributes
    one row to the output.  Weights are keyed by signal name and represent
    the regime-specific normalised weight at decision time.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`outcome_labeler.load_labeled_decisions` (or any
        DataFrame with ``time``, ``ticker``, ``regime``, ``mwu_weights``).

    Returns
    -------
    pd.DataFrame
        Columns: ``time``, ``ticker``, ``regime``, ``regime_label``,
        ``hmm_regime``, ``ou_spread``, ``llm_sentiment``, ``analyst_recs``.
        Sorted ascending by ``time``.  Empty if no valid rows found.
    """
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        weights = row.get("mwu_weights")
        if not isinstance(weights, dict):
            continue

        regime_raw = row.get("regime")
        regime = int(regime_raw) if regime_raw is not None else -1

        rows.append(
            {
                "time":         row["time"],
                "ticker":       row["ticker"],
                "regime":       regime,
                "regime_label": _REGIME_LABELS.get(regime, "unknown"),
                **{sig: float(weights.get(sig, float("nan"))) for sig in _SIGNAL_NAMES},
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["time", "ticker", "regime", "regime_label"] + list(_SIGNAL_NAMES)
        )

    wdf = pd.DataFrame(rows)
    wdf["time"] = pd.to_datetime(wdf["time"])
    return wdf.sort_values("time").reset_index(drop=True)


def summarise_weight_evolution(wdf: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise how MWU weights drifted from their initial priors.

    Parameters
    ----------
    wdf : pd.DataFrame
        Output of :func:`extract_weight_history`.

    Returns
    -------
    dict with keys:

    ``per_ticker_regime`` : dict[(ticker, regime_label) -> {signal: weight}]
        Final weight dict for each (ticker, regime) combination.

    ``drifted_signals`` : list[dict]
        Signals that drifted more than ``_DRIFT_THRESHOLD_PCT`` % from their
        prior.  Each entry has keys: ticker, regime, signal, initial, final,
        drift_pct.

    ``collapsed_signals`` : list[dict]
        Signals whose final weight fell below ``_COLLAPSE_THRESHOLD``
        (effectively ignored by MWU).  Keys: ticker, regime, signal,
        final_weight.
    """
    if wdf.empty:
        return {
            "per_ticker_regime": {},
            "drifted_signals":   [],
            "collapsed_signals": [],
        }

    per_ticker_regime: dict[tuple[str, str], dict[str, float]] = {}
    drifted:   list[dict[str, Any]] = []
    collapsed: list[dict[str, Any]] = []

    for (ticker, regime_label), group in wdf.groupby(["ticker", "regime_label"]):
        last = group.sort_values("time").iloc[-1]
        final_weights = {
            sig: float(last[sig])
            for sig in _SIGNAL_NAMES
            if sig in last and last[sig] == last[sig]  # not NaN
        }
        per_ticker_regime[(str(ticker), str(regime_label))] = final_weights

        for sig, final_w in final_weights.items():
            init_w    = _INITIAL_WEIGHTS.get(sig, 0.25)
            drift_pct = (final_w - init_w) / init_w * 100.0

            if abs(drift_pct) > _DRIFT_THRESHOLD_PCT:
                drifted.append(
                    {
                        "ticker":     str(ticker),
                        "regime":     str(regime_label),
                        "signal":     sig,
                        "initial":    round(init_w, 4),
                        "final":      round(final_w, 4),
                        "drift_pct":  round(drift_pct, 1),
                    }
                )

            if final_w < _COLLAPSE_THRESHOLD:
                collapsed.append(
                    {
                        "ticker":       str(ticker),
                        "regime":       str(regime_label),
                        "signal":       sig,
                        "final_weight": round(final_w, 4),
                    }
                )

    return {
        "per_ticker_regime": per_ticker_regime,
        "drifted_signals":   drifted,
        "collapsed_signals": collapsed,
    }
