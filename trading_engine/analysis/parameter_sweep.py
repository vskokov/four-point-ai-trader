"""
Parameter sensitivity sweeps.

All sweeps operate on the labeled DataFrame produced by outcome_labeler —
no DB queries, no API calls.  Each sweep re-evaluates the effect of changing
one parameter using data that was already logged.

Public API
----------
sweep_hours_back(df, cutoffs_h)       -> pd.DataFrame
    News-window sensitivity via contributing_headlines ages.

sweep_entry_z(df, z_thresholds)       -> pd.DataFrame
    OU entry threshold sensitivity via logged ou_zscore.

sweep_min_confidence(df, thresholds)  -> pd.DataFrame
    MWU score-gate sensitivity via logged score.

sweep_eta(df, eta_values)             -> pd.DataFrame
    MWU learning-rate sensitivity via weight-update replay.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_HORIZONS = ("1m", "15m", "1h", "4h")

# Initial MWU weight vector (hmm, ou, llm, analyst_recs)
_INIT_WEIGHTS = np.array([2 / 7, 2 / 7, 2 / 7, 1 / 7], dtype=float)
_INIT_WEIGHTS /= _INIT_WEIGHTS.sum()


# ---------------------------------------------------------------------------
# 4a. hours_back sweep
# ---------------------------------------------------------------------------


def sweep_hours_back(
    df: pd.DataFrame,
    cutoffs_h: list[float] | None = None,
) -> pd.DataFrame:
    """
    Estimate how LLM accuracy varies with the news-window length.

    Uses ``contributing_headlines[*].published_at`` already stored in
    ``trade_log`` to categorise each LLM-based decision as either *fresh*
    (all its contributing headlines are within the cutoff) or *mixed* (at
    least one headline is older than the cutoff).

    Parameters
    ----------
    df : pd.DataFrame
        Labeled decisions; must contain ``contributing_headlines`` (list of
        dicts), ``llm_signal``, and ``correct_*`` columns.
    cutoffs_h : list[float] or None
        Hours-back thresholds to evaluate.  Default: ``[1.0, 2.0, 4.0, 8.0]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``hours_back``, ``n_fresh``, ``n_mixed``,
        ``win_rate_1m_fresh``, ``win_rate_1m_mixed``,
        ``win_rate_15m_fresh``, ``win_rate_15m_mixed``,
        ``win_rate_1h_fresh``, ``win_rate_1h_mixed``.
    """
    if cutoffs_h is None:
        cutoffs_h = [1.0, 2.0, 4.0, 8.0]

    if df.empty or "llm_signal" not in df.columns:
        return pd.DataFrame()
    llm = df[df["llm_signal"] != 0].copy()
    if llm.empty:
        return pd.DataFrame()

    # Pre-compute oldest-headline age (hours) per row
    llm["_oldest_h"] = llm.apply(_oldest_headline_age_hours, axis=1)

    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs_h:
        # NaN age (missing/empty headlines) → unknown age → treat as mixed
        fresh = llm[llm["_oldest_h"].notna() & (llm["_oldest_h"] <= cutoff)]
        mixed = llm[llm["_oldest_h"].isna()  | (llm["_oldest_h"] >  cutoff)]

        row: dict[str, Any] = {
            "hours_back": cutoff,
            "n_fresh":    len(fresh),
            "n_mixed":    len(mixed),
        }
        for h in ("1m", "15m", "1h"):
            row[f"win_rate_{h}_fresh"] = _win_rate(fresh, f"fwd_ret_{h}", "llm_signal")
            row[f"win_rate_{h}_mixed"] = _win_rate(mixed, f"fwd_ret_{h}", "llm_signal")
        rows.append(row)

    return pd.DataFrame(rows)


def _oldest_headline_age_hours(row: pd.Series) -> float:
    """Return the age (hours) of the oldest contributing headline, or ``nan``."""
    headlines = row.get("contributing_headlines")
    if not isinstance(headlines, list) or not headlines:
        return float("nan")

    decision_time = _ensure_utc(pd.to_datetime(row["time"]))
    max_age_h = 0.0

    for h in headlines:
        pub = h.get("published_at", "")
        if not pub:
            continue
        try:
            pub_dt  = pd.to_datetime(pub, utc=True)
            age_h   = (decision_time - pub_dt).total_seconds() / 3600.0
            max_age_h = max(max_age_h, age_h)
        except Exception:
            continue

    return max_age_h if max_age_h > 0 else float("nan")


# ---------------------------------------------------------------------------
# 4b. entry_z sweep
# ---------------------------------------------------------------------------


def sweep_entry_z(
    df: pd.DataFrame,
    z_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Simulate OU accuracy with different entry Z-score thresholds.

    For mean reversion a high positive z-score → short signal (−1) and a
    large negative z-score → long signal (+1), i.e. ``sim_signal = -sign(z)``.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled decisions; must contain ``ou_zscore`` and ``fwd_ret_*``.
    z_thresholds : list[float] or None
        Entry-Z values to evaluate.  Default: ``[1.5, 2.0, 2.5, 3.0]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``entry_z``, ``n_trades``,
        ``win_rate_1m``, ``win_rate_15m``, ``win_rate_1h``, ``win_rate_4h``.
    """
    if z_thresholds is None:
        z_thresholds = [1.5, 2.0, 2.5, 3.0]

    if df.empty or "ou_zscore" not in df.columns:
        return pd.DataFrame()
    ou = df[df["ou_zscore"].notna()].copy()
    if ou.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for z_thresh in z_thresholds:
        fired = ou[ou["ou_zscore"].abs() >= z_thresh].copy()
        # mean-reversion: trade against the spread direction
        fired["_sim_sig"] = -(np.sign(fired["ou_zscore"]).astype(int))

        row: dict[str, Any] = {"entry_z": z_thresh, "n_trades": len(fired)}
        for h in _HORIZONS:
            ret_col = f"fwd_ret_{h}"
            if ret_col not in fired.columns:
                row[f"win_rate_{h}"] = float("nan")
                continue
            valid   = fired.dropna(subset=[ret_col])
            correct = valid["_sim_sig"] * valid[ret_col] > 0
            row[f"win_rate_{h}"] = float(correct.mean()) if len(valid) > 0 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4c. min_confidence sweep
# ---------------------------------------------------------------------------


def sweep_min_confidence(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate ensemble accuracy at different |score| gate thresholds.

    Shows: if we only traded when ``abs(score) >= threshold``, what would
    accuracy and trade-frequency be?

    Parameters
    ----------
    df : pd.DataFrame
        Labeled decisions; must contain ``score``, ``correct_*`` columns.
        Rows with ``score == 0`` or NaN are excluded from the count of
        potentially tradeable decisions.
    thresholds : list[float] or None
        Score thresholds to evaluate.
        Default: 0.30, 0.35, … 0.70 (steps of 0.05).

    Returns
    -------
    pd.DataFrame
        Columns: ``threshold``, ``n_active``, ``pct_suppressed``,
        ``win_rate_1m``, ``win_rate_15m``, ``win_rate_1h``, ``win_rate_4h``.
    """
    if thresholds is None:
        thresholds = [round(v, 2) for v in np.arange(0.30, 0.71, 0.05)]

    if df.empty or "score" not in df.columns:
        return pd.DataFrame()
    scored = df[df["score"].notna() & (df["score"] != 0)]
    total  = len(scored)
    if total == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for thresh in thresholds:
        active = scored[scored["score"].abs() >= thresh]
        pct_suppressed = (1 - len(active) / total) * 100.0

        row: dict[str, Any] = {
            "threshold":      thresh,
            "n_active":       len(active),
            "pct_suppressed": round(pct_suppressed, 1),
        }
        for h in _HORIZONS:
            corr_col = f"correct_{h}"
            if corr_col not in active.columns or len(active) == 0:
                row[f"win_rate_{h}"] = float("nan")
                continue
            valid = active.dropna(subset=[corr_col])
            row[f"win_rate_{h}"] = float(valid[corr_col].mean()) if len(valid) > 0 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4d. eta sweep
# ---------------------------------------------------------------------------


def sweep_eta(
    df: pd.DataFrame,
    eta_values: list[float] | None = None,
    n_regimes: int = 3,
    n_signals: int = 4,
) -> pd.DataFrame:
    """
    Replay MWU weight updates with different learning rates.

    Reconstructs the (signal, actual_direction) sequence from the labeled
    DataFrame and re-runs the MWU algorithm for each eta value.  Uses
    ``fwd_ret_15m > 0`` as the realised direction (same horizon as MWU's
    ``horizon_bars=1`` in practice, but 15 m gives more stable labels).

    Note: ``analyst_recs`` is not logged in trade_log columns, so its
    simulated weight update uses ``signal = 0`` (neutral — loss = 0.5).
    The replay is an approximation for the three logged signals.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled decisions with ``hmm_signal``, ``ou_signal``, ``llm_signal``,
        ``regime``, ``fwd_ret_15m``.
    eta_values : list[float] or None
        Learning rates.  Default: ``[0.01, 0.05, 0.1, 0.2, 0.5]``.
    n_regimes, n_signals : int
        MWU matrix dimensions.

    Returns
    -------
    pd.DataFrame
        Columns: ``eta``, ``final_accuracy_15m``, ``weight_range``
        (max-min across all weight matrix entries, indicating convergence),
        ``n_updates``.
        Empty if fewer than 10 decisions with ``fwd_ret_15m`` data.
    """
    if eta_values is None:
        eta_values = [0.01, 0.05, 0.1, 0.2, 0.5]

    seq = df.dropna(subset=["fwd_ret_15m"]).sort_values("time")
    if len(seq) < 10:
        return pd.DataFrame()

    actual_dir = seq["fwd_ret_15m"].apply(
        lambda r: 1 if r > 0 else (-1 if r < 0 else 0)
    )

    rows: list[dict[str, Any]] = []
    for eta in eta_values:
        weights     = np.tile(_INIT_WEIGHTS.copy(), (n_regimes, 1))
        n_updates   = 0
        correct_15m: list[bool] = []

        for (_, row), actual in zip(seq.iterrows(), actual_dir):
            regime = int(row.get("regime", 1) or 1)
            if regime < 0 or regime >= n_regimes:
                regime = 1

            w = weights[regime]

            # Signal values: hmm, ou, llm, analyst_recs (=0, not logged)
            sigs = np.array(
                [
                    float(row.get("hmm_signal", 0) or 0),
                    float(row.get("ou_signal",  0) or 0),
                    float(row.get("llm_signal", 0) or 0),
                    0.0,
                ]
            )
            # Proxy confidence: 0.5 for non-zero signals
            conf  = np.where(sigs != 0, 0.5, 0.0)
            score = float(np.dot(w, sigs * conf))
            pred  = 1 if score > 0.3 else (-1 if score < -0.3 else 0)

            if pred != 0 and actual != 0:
                correct_15m.append(pred == actual)

            # MWU update
            if actual != 0:
                for k in range(n_signals):
                    s_k  = sigs[k]
                    loss = 0.5 if s_k == 0 else (0.0 if s_k * actual > 0 else 1.0)
                    weights[regime, k] *= np.exp(-eta * loss)
                row_sum = weights[regime].sum()
                if row_sum > 1e-10:
                    weights[regime] /= row_sum
                n_updates += 1

        rows.append(
            {
                "eta":               eta,
                "final_accuracy_15m": (
                    float(np.mean(correct_15m)) if correct_15m else float("nan")
                ),
                "weight_range":      float(weights.max() - weights.min()),
                "n_updates":         n_updates,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _win_rate(df: pd.DataFrame, ret_col: str, sig_col: str) -> float:
    """Fraction of rows where ``sig_col * ret_col > 0``."""
    if df.empty or ret_col not in df.columns:
        return float("nan")
    valid   = df.dropna(subset=[ret_col])
    correct = valid[sig_col] * valid[ret_col] > 0
    return float(correct.mean()) if len(valid) > 0 else float("nan")


def _ensure_utc(ts: Any) -> Any:
    """Make a pandas Timestamp timezone-aware (UTC)."""
    if hasattr(ts, "tz") and ts.tz is None:
        return ts.tz_localize("UTC")
    return ts
