"""
Four-Point AI Trader — Trade Decision Dashboard
================================================
Standalone Streamlit app. Reads trade_log from TimescaleDB and displays
the reasoning behind each trade: HMM regime, OU spread, LLM sentiment,
MWU weights, and contributing news headlines.

Run from the trading_engine/ directory:
    .venv/bin/streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / ".env")

DB_URL = os.environ.get("DB_URL", "")

_SIGNAL_LABEL = {1: "BUY", -1: "SELL"}
_SIGNAL_COLOR = {1: "🟢", -1: "🔴"}
_REGIME_COLOR = {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_engine() -> sqlalchemy.Engine:
    if not DB_URL:
        st.error("DB_URL not set. Check your .env file.")
        st.stop()
    return sqlalchemy.create_engine(DB_URL, pool_pre_ping=True)


def _load_trades(limit: int) -> pd.DataFrame:
    sql = sqlalchemy.text(
        """
        SELECT id, time, ticker, final_signal, score,
               regime, regime_label, regime_probs,
               hmm_signal, hmm_confidence,
               ou_signal, ou_confidence, ou_zscore, ou_spread_value, ou_pair,
               llm_signal, llm_confidence,
               mwu_weights, contributing_headlines
        FROM   trade_log
        ORDER  BY time DESC
        LIMIT  :limit
        """
    )
    with _get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"limit": limit})


def _parse_json(val: Any) -> Any:
    """Parse a JSONB value that may already be a dict/list or a JSON string."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _regime_badge(label: str | None) -> str:
    if not label:
        return "—"
    emoji = _REGIME_COLOR.get(label.lower(), "⚪")
    return f"{emoji} **{label.upper()}**"


def _render_regime_probs(probs: dict | None) -> None:
    if not probs:
        st.caption("No regime probability data.")
        return
    for label, p in sorted(probs.items(), key=lambda x: -x[1]):
        emoji = _REGIME_COLOR.get(label.lower(), "⚪")
        st.progress(float(p), text=f"{emoji} {label.capitalize()}: {p:.1%}")


def _render_trade(row: pd.Series) -> None:
    """Render one trade as a labelled expander."""
    ts = row["time"]
    if hasattr(ts, "tz_localize") and ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    ts_str = str(ts)[:19].replace("T", " ")

    sig = int(row["final_signal"])
    emoji = _SIGNAL_COLOR.get(sig, "⚪")
    label = _SIGNAL_LABEL.get(sig, str(sig))
    score = f"{float(row['score']):.4f}" if row["score"] is not None else "—"

    header = f"{emoji} {ts_str}  |  **{row['ticker']}**  |  {label}  |  score {score}"

    with st.expander(header, expanded=False):
        col_hmm, col_ou, col_llm = st.columns(3)

        # ---- HMM ----
        with col_hmm:
            st.markdown("#### HMM Regime")
            regime_label = row.get("regime_label") or "—"
            st.markdown(_regime_badge(regime_label))
            probs = _parse_json(row.get("regime_probs"))
            _render_regime_probs(probs)
            hmm_sig = row.get("hmm_signal")
            hmm_conf = row.get("hmm_confidence")
            st.caption(
                f"Signal: **{hmm_sig}** &nbsp;|&nbsp; "
                f"Confidence: **{float(hmm_conf):.2f}**"
                if hmm_sig is not None and hmm_conf is not None
                else "No HMM data"
            )

        # ---- OU Spread ----
        with col_ou:
            st.markdown("#### OU Spread")
            pair = row.get("ou_pair")
            if pair:
                st.markdown(f"**Pair:** {pair}")
                ou_z = row.get("ou_zscore")
                ou_spread = row.get("ou_spread_value")
                ou_sig = row.get("ou_signal")
                ou_conf = row.get("ou_confidence")
                st.metric("Z-score", f"{float(ou_z):.3f}" if ou_z is not None else "—")
                st.metric("Spread", f"{float(ou_spread):.5f}" if ou_spread is not None else "—")
                st.caption(
                    f"Signal: **{ou_sig}** &nbsp;|&nbsp; "
                    f"Confidence: **{float(ou_conf):.2f}**"
                    if ou_sig is not None and ou_conf is not None
                    else ""
                )
            else:
                st.caption("No active pair for this ticker.")

        # ---- LLM Sentiment ----
        with col_llm:
            st.markdown("#### LLM Sentiment")
            llm_sig = row.get("llm_signal")
            llm_conf = row.get("llm_confidence")
            llm_label = _SIGNAL_LABEL.get(int(llm_sig), "NEUTRAL") if llm_sig is not None else "—"
            st.markdown(f"Direction: **{llm_label}**")
            st.caption(
                f"Signal: **{llm_sig}** &nbsp;|&nbsp; "
                f"Confidence: **{float(llm_conf):.2f}**"
                if llm_sig is not None and llm_conf is not None
                else "No LLM data"
            )

        st.divider()

        # ---- MWU Weights ----
        weights = _parse_json(row.get("mwu_weights"))
        if weights:
            st.markdown("#### MWU Ensemble Weights")
            w_col1, w_col2, w_col3 = st.columns(3)
            for col, (name, val) in zip(
                [w_col1, w_col2, w_col3],
                weights.items(),
            ):
                col.metric(name.replace("_", " ").title(), f"{float(val):.3f}")

        # ---- Contributing Headlines ----
        headlines = _parse_json(row.get("contributing_headlines"))
        if headlines:
            st.markdown(f"#### News Headlines ({len(headlines)})")
            for h in headlines:
                title = h.get("title", "—")
                source = h.get("source") or "unknown source"
                pub = str(h.get("published_at", ""))[:16].replace("T", " ")
                av_label = h.get("av_sentiment_label", "")
                av_score = h.get("av_sentiment_score")
                av_str = (
                    f"AV: {av_label} ({float(av_score):.3f})"
                    if av_score is not None
                    else f"AV: {av_label}"
                )
                st.markdown(
                    f"- {title}  \n"
                    f"  <small>{source} · {pub} · {av_str}</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No headlines recorded for this signal.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Four-Point AI Trader — Decision Log",
        page_icon="📈",
        layout="wide",
    )

    # Auto-refresh every 60 seconds
    st_autorefresh(interval=60_000, key="trade_refresh")

    st.title("📈 Four-Point AI Trader — Decision Log")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        limit = st.selectbox("Trades to display", [25, 50, 100], index=0)
        st.caption("Auto-refreshes every 60 s.")

    # Load data
    try:
        df = _load_trades(limit)
    except Exception as exc:
        st.error(f"Failed to load trades: {exc}")
        return

    if df.empty:
        st.info("No trades recorded yet. The engine will populate this table when it submits orders.")
        return

    st.caption(f"Showing {len(df)} most recent trades. Last refreshed: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    for _, row in df.iterrows():
        _render_trade(row)


if __name__ == "__main__":
    main()
