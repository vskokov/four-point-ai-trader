"""
Markdown report generator.

Aggregates outputs from all analysis modules into a single human-readable
Markdown document with an executive summary, per-signal accuracy tables,
MWU weight evolution, parameter sensitivity curves, and specific
parameter recommendations.

Public API
----------
generate_report(labeled_df, signal_accuracy, ensemble_accuracy,
                weight_summary, sweep_hours_back, sweep_entry_z,
                sweep_min_confidence, sweep_eta, output_path) -> str
    Write the report to ``output_path`` and return the content string.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_HORIZONS = ("1m", "15m", "1h", "4h")

_INITIAL_WEIGHTS: dict[str, float] = {
    "hmm_regime":    2 / 7,
    "ou_spread":     2 / 7,
    "llm_sentiment": 2 / 7,
    "analyst_recs":  1 / 7,
}


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def generate_report(
    labeled_df: pd.DataFrame,
    signal_accuracy: pd.DataFrame,
    ensemble_accuracy: pd.DataFrame,
    weight_summary: dict[str, Any],
    sweep_hours_back: pd.DataFrame,
    sweep_entry_z: pd.DataFrame,
    sweep_min_confidence: pd.DataFrame,
    sweep_eta: pd.DataFrame,
    output_path: Path,
) -> str:
    """
    Generate and write a Markdown decision-quality report.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Full labeled decisions from outcome_labeler.
    signal_accuracy : pd.DataFrame
        Output of signal_quality.compute_signal_accuracy.
    ensemble_accuracy : pd.DataFrame
        Output of signal_quality.compute_ensemble_accuracy.
    weight_summary : dict
        Output of weight_evolution.summarise_weight_evolution.
    sweep_* : pd.DataFrame
        Outputs of the corresponding parameter_sweep functions.
    output_path : Path
        Destination file.  Parent directories are created if needed.

    Returns
    -------
    str
        The full Markdown content written to disk.
    """
    sections: list[str] = []

    # ---- Header ----
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(f"# Decision Quality Report\n\nGenerated: {now}\n")

    # ---- Executive summary ----
    sections += _executive_summary(labeled_df, ensemble_accuracy)

    # ---- Ensemble accuracy ----
    sections.append("## Ensemble Decision Accuracy\n")
    if not ensemble_accuracy.empty:
        sections.append(_df_table(ensemble_accuracy.reset_index()))
    else:
        sections.append("*No data.*")
    sections.append("")

    # ---- Per-signal accuracy ----
    sections.append("## Per-Signal Accuracy\n")
    sections.append(
        "> Win rates use each signal's *own* vote vs forward return, "
        "not the ensemble `final_signal`.\n"
    )
    if not signal_accuracy.empty:
        sections.append(_df_table(signal_accuracy.reset_index()))
    else:
        sections.append("*No data.*")
    sections.append("")

    # ---- Weight evolution ----
    sections += _weight_section(weight_summary)

    # ---- Parameter sensitivity ----
    sections.append("## Parameter Sensitivity\n")
    sections += _hours_back_section(sweep_hours_back)
    sections += _entry_z_section(sweep_entry_z)
    sections += _min_confidence_section(sweep_min_confidence)
    sections += _eta_section(sweep_eta)

    # ---- Recommendations ----
    sections.append("## Recommendations\n")
    recs = _derive_recommendations(
        sweep_min_confidence, sweep_entry_z, sweep_hours_back, weight_summary
    )
    if recs:
        sections.extend(recs)
    else:
        sections.append(
            "*Insufficient data for specific recommendations. "
            "Run again after 2+ weeks of engine operation.*"
        )
    sections.append("")

    # ---- Schema gap ----
    sections.append("## Schema Notes\n")
    sections.append(
        "All four MWU signals (hmm, ou, llm, analyst_recs) now have dedicated columns "
        "in `trade_log` (`analyst_signal`, `analyst_confidence`).  "
        "Per-signal accuracy analysis covers all four signals."
    )

    content = "\n".join(sections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _executive_summary(
    df: pd.DataFrame,
    ens_acc: pd.DataFrame,
) -> list[str]:
    lines = ["## Executive Summary\n"]

    if df.empty:
        lines.append("*No decisions found.*\n")
        return lines

    tickers    = sorted(df["ticker"].unique().tolist())
    n_total    = len(df)
    n_active   = int((df["final_signal"] != 0).sum())
    t_min      = df["time"].min()
    t_max      = df["time"].max()
    date_range = f"{t_min}  →  {t_max}"

    lines += [
        f"- **Tickers**: {', '.join(tickers)}",
        f"- **Date range**: {date_range}",
        f"- **Total decisions logged**: {n_total}",
        f"- **Directional decisions** (final_signal ≠ 0): {n_active}",
    ]

    if not ens_acc.empty:
        try:
            all_row = (
                ens_acc.loc["all"]
                if "all" in ens_acc.index
                else ens_acc.iloc[0]
            )
            wr_15m = all_row.get("win_rate_15m", float("nan"))
            wr_1h  = all_row.get("win_rate_1h",  float("nan"))
            lines.append(f"- **Overall ensemble win rate (15 m)**: {_pct(wr_15m)}")
            lines.append(f"- **Overall ensemble win rate (1 h)**:  {_pct(wr_1h)}")
        except Exception:
            pass

    lines.append("")
    return lines


def _weight_section(weight_summary: dict[str, Any]) -> list[str]:
    lines = ["## MWU Weight Evolution\n"]

    if not weight_summary:
        lines += ["*No weight data available.*\n", ""]
        return lines

    lines.append("### Final weights vs initial priors (2/7, 2/7, 2/7, 1/7)\n")

    per_tr = weight_summary.get("per_ticker_regime", {})
    for (ticker, regime), weights in sorted(per_tr.items()):
        lines.append(f"**{ticker} — {regime} regime**\n")
        for sig, w in weights.items():
            init  = _INITIAL_WEIGHTS.get(sig, 0.25)
            delta = w - init
            arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "→")
            lines.append(f"- {sig}: {w:.3f}  (prior {init:.3f}  {arrow} {delta:+.3f})")
        lines.append("")

    drifted = weight_summary.get("drifted_signals", [])
    if drifted:
        lines.append("### Significant drifts (>20 % from prior)\n")
        for d in drifted:
            lines.append(
                f"- **{d['ticker']} / {d['regime']} / {d['signal']}**: "
                f"{d['initial']:.3f} → {d['final']:.3f}  "
                f"({d['drift_pct']:+.1f} %)"
            )
        lines.append("")

    collapsed = weight_summary.get("collapsed_signals", [])
    if collapsed:
        lines.append("### Collapsed signals (weight < 0.05 — effectively ignored by MWU)\n")
        for c in collapsed:
            lines.append(
                f"- **{c['ticker']} / {c['regime']} / {c['signal']}**: "
                f"{c['final_weight']:.4f}"
            )
        lines.append("")

    return lines


def _hours_back_section(sw: pd.DataFrame) -> list[str]:
    lines = ["### `hours_back` — News Window Sensitivity\n"]
    lines.append(
        "_Fresh = all contributing headlines within the cutoff window; "
        "Mixed = at least one headline is older._\n"
    )
    if not sw.empty:
        lines.append(_df_table(sw))
    else:
        lines.append("*No LLM signal data with contributing_headlines timestamps.*")
    lines.append("")
    return lines


def _entry_z_section(sw: pd.DataFrame) -> list[str]:
    lines = ["### `entry_z` — OU Spread Entry Threshold Sensitivity\n"]
    lines.append(
        "_Simulated: `sim_signal = -sign(ou_zscore)` when `|ou_zscore| >= entry_z`._\n"
    )
    if not sw.empty:
        lines.append(_df_table(sw))
    else:
        lines.append("*No OU z-score data available.*")
    lines.append("")
    return lines


def _min_confidence_section(sw: pd.DataFrame) -> list[str]:
    lines = ["### `min_confidence` — MWU Score Gate Sensitivity\n"]
    lines.append(
        "_Higher threshold = fewer trades, potentially higher accuracy. "
        "`pct_suppressed` shows what fraction of trades would become neutral._\n"
    )
    if not sw.empty:
        lines.append(_df_table(sw))
    else:
        lines.append("*No score data available.*")
    lines.append("")
    return lines


def _eta_section(sw: pd.DataFrame) -> list[str]:
    lines = ["### `eta` — MWU Learning Rate Sensitivity\n"]
    lines.append(
        "_Replay of weight updates using logged signal values. "
        "`analyst_recs` omitted (not logged). "
        "`weight_range` = max − min across all weight entries (higher = more converged)._\n"
    )
    if not sw.empty:
        lines.append(_df_table(sw))
    else:
        lines.append(
            "*Insufficient data for eta sweep (need ≥ 10 decisions with 15 m outcomes).*"
        )
    lines.append("")
    return lines


def _derive_recommendations(
    sw_mc:   pd.DataFrame,
    sw_ez:   pd.DataFrame,
    sw_hb:   pd.DataFrame,
    w_sum:   dict[str, Any],
) -> list[str]:
    """Return a list of Markdown recommendation blocks."""
    recs: list[str] = []

    # min_confidence
    if not sw_mc.empty and "win_rate_15m" in sw_mc.columns:
        valid = sw_mc.dropna(subset=["win_rate_15m"])
        if len(valid) >= 3:
            best    = valid.loc[valid["win_rate_15m"].idxmax()]
            b_thr   = float(best["threshold"])
            b_wr    = float(best["win_rate_15m"])
            cur_row = valid[valid["threshold"].between(0.29, 0.31)]
            c_wr    = float(cur_row["win_rate_15m"].iloc[0]) if len(cur_row) > 0 else float("nan")
            if b_thr > 0.31 and b_wr > c_wr + 0.02:
                recs.append(
                    f"### `min_confidence`: 0.3 → {b_thr:.2f}\n\n"
                    f"Evidence: threshold {b_thr:.2f} gives {_pct(b_wr)} win_rate_15m "
                    f"vs {_pct(c_wr)} at the current 0.3.  "
                    f"Suppresses {float(best['pct_suppressed']):.1f} % of trades.\n"
                )

    # entry_z
    if not sw_ez.empty and "win_rate_15m" in sw_ez.columns:
        valid = sw_ez.dropna(subset=["win_rate_15m"])
        if len(valid) >= 2:
            best    = valid.loc[valid["win_rate_15m"].idxmax()]
            b_z     = float(best["entry_z"])
            b_wr    = float(best["win_rate_15m"])
            cur_row = valid[valid["entry_z"] == 2.0]
            c_wr    = float(cur_row["win_rate_15m"].iloc[0]) if len(cur_row) > 0 else float("nan")
            if b_z != 2.0 and b_wr > c_wr + 0.02:
                recs.append(
                    f"### `entry_z`: 2.0 → {b_z:.1f}\n\n"
                    f"Evidence: z={b_z:.1f} gives {_pct(b_wr)} OU win_rate_15m "
                    f"vs {_pct(c_wr)} at z=2.0.  "
                    f"n_trades at new threshold: {int(best['n_trades'])}.\n"
                )

    # hours_back
    if not sw_hb.empty and "win_rate_15m_fresh" in sw_hb.columns:
        valid = sw_hb.dropna(subset=["win_rate_15m_fresh"])
        if len(valid) >= 2:
            best    = valid.loc[valid["win_rate_15m_fresh"].idxmax()]
            b_hb    = float(best["hours_back"])
            b_wr    = float(best["win_rate_15m_fresh"])
            cur_row = valid[valid["hours_back"] == 2.0]
            c_wr    = float(cur_row["win_rate_15m_fresh"].iloc[0]) if len(cur_row) > 0 else float("nan")
            if b_hb != 2.0 and b_wr > c_wr + 0.02:
                recs.append(
                    f"### `hours_back`: 2 → {b_hb:.0f}\n\n"
                    f"Evidence: {b_hb:.0f} h window gives {_pct(b_wr)} LLM win_rate_15m "
                    f"(fresh-only bucket) vs {_pct(c_wr)} for the current 2 h.  "
                    f"n_fresh={int(best['n_fresh'])}.\n"
                )

    # Collapsed weights
    if w_sum:
        for c in w_sum.get("collapsed_signals", []):
            recs.append(
                f"### Route `{c['signal']}` confidence to 0 in {c['ticker']} "
                f"{c['regime']} regime\n\n"
                f"Evidence: MWU down-weighted `{c['signal']}` to "
                f"{c['final_weight']:.4f} in the {c['regime']} regime "
                f"— it contributes almost nothing to the ensemble score.  "
                f"Consider setting its confidence to 0 via signal routing for "
                f"that regime.\n"
            )

    return recs


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pct(v: float) -> str:
    if v != v:  # NaN check without math import
        return "N/A"
    return f"{v * 100:.1f} %"


def _df_table(df: pd.DataFrame) -> str:
    """Format a DataFrame as a Markdown table (no tabulate dependency)."""
    if df.empty:
        return "*empty*"

    cols = list(df.columns)

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            if v != v:  # NaN
                return "—"
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    body   = [
        "| " + " | ".join(_fmt(row[c]) for c in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep] + body)
