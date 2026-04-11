"""
Per-signal accuracy analysis.

For each of the three signals with dedicated trade_log columns
(hmm, ou, llm) compute win rates and information coefficients across
multiple horizons, segmented by regime, confidence band, and time-of-day.

All four signals (hmm, ou, llm, analyst) now have dedicated columns in
trade_log and are included in the per-signal accuracy analysis.

Public API
----------
compute_signal_accuracy(df)  -> pd.DataFrame
    Per-signal accuracy, segmented breakdown.

compute_ensemble_accuracy(df) -> pd.DataFrame
    Ensemble (final_signal) accuracy, segmented by regime and score band.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_HORIZONS = ("1m", "15m", "1h", "4h")

_REGIME_LABELS = {0: "bear", 1: "neutral", 2: "bull"}

# Map analysis label → (signal column, confidence column)
# analyst_recs is included now that trade_log has analyst_signal/analyst_confidence.
_SIGNAL_MAP: dict[str, tuple[str, str]] = {
    "hmm":      ("hmm_signal",      "hmm_confidence"),
    "ou":       ("ou_signal",       "ou_confidence"),
    "llm":      ("llm_signal",      "llm_confidence"),
    "analyst":  ("analyst_signal",  "analyst_confidence"),
}

_CONFIDENCE_BANDS = [(0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def compute_signal_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-signal directional accuracy across four forward horizons.

    For each signal that has its own column in the labeled DataFrame, compute:

    * **win_rate_Xm** — fraction of non-zero signal bars where
      ``signal * fwd_ret_Xm > 0``
    * **ic_Xm** — Pearson IC of ``signal × confidence`` vs ``fwd_ret_Xm``

    Segments:
    * ``all`` — overall
    * ``regime:{label}`` — filtered to bear / neutral / bull bars
    * ``conf:{lo}-{hi}`` — filtered to confidence band
    * ``tod:{open|midday|close}`` — filtered to ET time-of-day window

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`outcome_labeler.compute_outcome_labels`.

    Returns
    -------
    pd.DataFrame
        MultiIndex (signal, segment) with columns ``n``,
        ``win_rate_1m``, ``ic_1m``, ``win_rate_15m``, ``ic_15m``,
        ``win_rate_1h``, ``ic_1h``, ``win_rate_4h``, ``ic_4h``.
    """
    rows: list[dict[str, Any]] = []

    for sig_name, (sig_col, conf_col) in _SIGNAL_MAP.items():
        if sig_col not in df.columns:
            continue

        # ---- overall ----
        _append_row(rows, df, sig_name, "all", sig_col, conf_col)

        # ---- by regime ----
        if "regime" in df.columns:
            for regime_val, regime_label in _REGIME_LABELS.items():
                sub = df[df["regime"] == regime_val]
                if len(sub) >= 5:
                    _append_row(rows, sub, sig_name, f"regime:{regime_label}", sig_col, conf_col)

        # ---- by confidence band ----
        if conf_col in df.columns:
            for lo, hi in _CONFIDENCE_BANDS:
                sub = df[(df[conf_col] >= lo) & (df[conf_col] < hi)]
                if len(sub) >= 5:
                    _append_row(rows, sub, sig_name, f"conf:{lo:.1f}-{hi:.1f}", sig_col, conf_col)

        # ---- by time of day (ET) ----
        if "time" in df.columns:
            _append_tod_rows(rows, df, sig_name, sig_col, conf_col)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.set_index(["signal", "segment"])
    return result


def compute_ensemble_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ensemble (``final_signal``) accuracy at each horizon.

    Segments: overall, per regime, per |score| bucket.

    Returns
    -------
    pd.DataFrame
        Index ``segment``, columns ``n``,
        ``win_rate_1m``, ``win_rate_15m``, ``win_rate_1h``, ``win_rate_4h``.
    """
    active = df[df["final_signal"] != 0]
    rows: list[dict[str, Any]] = []

    _append_ensemble_row(rows, active, "all")

    if "regime" in df.columns:
        for regime_val, regime_label in _REGIME_LABELS.items():
            sub = active[active["regime"] == regime_val]
            if len(sub) >= 5:
                _append_ensemble_row(rows, sub, f"regime:{regime_label}")

    if "score" in df.columns:
        for lo, hi in [(0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.01)]:
            sub = active[active["score"].abs().between(lo, hi)]
            if len(sub) >= 5:
                _append_ensemble_row(rows, sub, f"score:{lo:.2f}-{hi:.2f}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("segment")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _append_row(
    rows: list[dict],
    df: pd.DataFrame,
    sig_name: str,
    segment: str,
    sig_col: str,
    conf_col: str,
) -> None:
    """Compute win rates and ICs for one (signal, segment) pair."""
    active = df[df[sig_col] != 0]
    if len(active) == 0:
        return

    row: dict[str, Any] = {
        "signal":  sig_name,
        "segment": segment,
        "n":       len(active),
    }

    for h in _HORIZONS:
        ret_col   = f"fwd_ret_{h}"
        corr_col  = f"correct_{h}"

        valid = active.dropna(subset=[ret_col])

        # win rate uses the signal's own vote vs forward return direction
        win_col_present = corr_col in active.columns
        if win_col_present:
            # recompute using this signal's vote, not ensemble final_signal
            sig_correct = (active[sig_col] * valid[ret_col] > 0)
            valid_wr    = active.loc[valid.index].dropna(subset=[ret_col])
            sig_correct = (valid_wr[sig_col] * valid_wr[ret_col] > 0)
            row[f"win_rate_{h}"] = float(sig_correct.mean()) if len(sig_correct) > 0 else float("nan")
        else:
            row[f"win_rate_{h}"] = float("nan")

        # IC: correlation of signed-confidence with forward return
        if len(valid) > 2 and conf_col in valid.columns:
            signed_conf = valid[sig_col] * valid[conf_col]
            ic = signed_conf.corr(valid[ret_col])
            row[f"ic_{h}"] = float(ic) if ic == ic else float("nan")
        else:
            row[f"ic_{h}"] = float("nan")

    rows.append(row)


def _append_ensemble_row(
    rows: list[dict],
    df: pd.DataFrame,
    segment: str,
) -> None:
    row: dict[str, Any] = {"segment": segment, "n": len(df)}
    for h in _HORIZONS:
        corr_col = f"correct_{h}"
        if corr_col not in df.columns:
            row[f"win_rate_{h}"] = float("nan")
            continue
        valid = df.dropna(subset=[corr_col])
        row[f"win_rate_{h}"] = float(valid[corr_col].mean()) if len(valid) > 0 else float("nan")
    rows.append(row)


def _append_tod_rows(
    rows: list[dict],
    df: pd.DataFrame,
    sig_name: str,
    sig_col: str,
    conf_col: str,
) -> None:
    """Add time-of-day segments (ET open / midday / close)."""
    try:
        times = pd.to_datetime(df["time"])
        if times.dt.tz is None:
            times = times.dt.tz_localize("UTC")
        hours_et = times.dt.tz_convert("America/New_York").dt.hour

        tod_map = {
            "tod:open":    (hours_et >= 9)  & (hours_et <= 10),
            "tod:midday":  (hours_et >= 11) & (hours_et <= 13),
            "tod:close":   (hours_et >= 14) & (hours_et <= 15),
        }
        for label, mask in tod_map.items():
            sub = df[mask]
            if len(sub) >= 5:
                _append_row(rows, sub, sig_name, label, sig_col, conf_col)
    except Exception:
        pass
