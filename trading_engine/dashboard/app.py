"""
Four-Point AI Trader — Trade Decision Dashboard
================================================
Standalone Streamlit app. Reads trade_log from TimescaleDB and live account
data from Alpaca, and displays:

  • Key metrics: equity, daily / weekly change, available cash
  • Equity chart with BUY / SELL markers overlaid
  • Trade decision log with Alpaca fill-status badges

Run from the trading_engine/ directory:
    .venv/bin/streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import sqlalchemy
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / ".env")

DB_URL             = os.environ.get("DB_URL", "")
ALPACA_API_KEY     = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY  = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL    = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
_PAPER             = "paper" in ALPACA_BASE_URL

_SIGNAL_LABEL = {1: "BUY", -1: "SELL"}
_SIGNAL_COLOR = {1: "🟢", -1: "🔴"}
_REGIME_COLOR = {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}

_ORDER_STATUS_BADGE: dict[str, tuple[str, str]] = {
    "filled":           ("✅", "Filled"),
    "partially_filled": ("🟡", "Partial"),
    "cancelled":        ("❌", "Cancelled"),
    "rejected":         ("🚫", "Rejected"),
    "pending_new":      ("⏳", "Pending"),
    "new":              ("⏳", "Submitted"),
    "accepted":         ("⏳", "Accepted"),
    "held":             ("⏳", "Held"),
}

_CHART_PERIOD_MAP = {
    "Today (1D)": ("1D", "5Min"),
    "1 Week":     ("1W", "1H"),
    "1 Month":    ("1M", "1D"),
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_db_engine() -> sqlalchemy.Engine:
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
    with _get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"limit": limit})


def _load_all_trades_for_chart() -> pd.DataFrame:
    """Load the last 7 days of trade decisions for chart marker overlay."""
    sql = sqlalchemy.text(
        """
        SELECT time, ticker, final_signal
        FROM   trade_log
        WHERE  time >= NOW() - INTERVAL '7 days'
        ORDER  BY time ASC
        """
    )
    with _get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn)


def _parse_json(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Alpaca helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_alpaca_client():
    from alpaca.trading.client import TradingClient
    if not ALPACA_API_KEY:
        return None
    return TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=_PAPER,
    )


@st.cache_data(ttl=30)
def _load_account_info() -> dict | None:
    """Load current account snapshot from Alpaca (cached 30 s)."""
    client = _get_alpaca_client()
    if client is None:
        return None
    try:
        acct = client.get_account()
        return {
            "equity":          float(acct.equity),
            "cash":            float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "buying_power":    float(acct.buying_power),
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def _load_portfolio_history(period: str, timeframe_str: str) -> pd.DataFrame:
    """
    Load portfolio equity history from Alpaca.

    Returns a DataFrame with columns: time (UTC), equity, profit_loss,
    profit_loss_pct.  Empty DataFrame on failure.
    """
    from alpaca.trading.requests import GetPortfolioHistoryRequest

    client = _get_alpaca_client()
    if client is None:
        return pd.DataFrame()
    try:
        history = client.get_portfolio_history(
            GetPortfolioHistoryRequest(
                period=period,
                timeframe=timeframe_str,
                extended_hours=False,
            )
        )
        if not history or not history.timestamp:
            return pd.DataFrame()

        df = pd.DataFrame({
            "time":             [pd.Timestamp(ts, unit="s", tz="UTC") for ts in history.timestamp],
            "equity":           history.equity,
            "profit_loss":      history.profit_loss,
            "profit_loss_pct":  history.profit_loss_pct,
        })
        return df.dropna(subset=["equity"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def _load_orders(days_back: int = 1) -> pd.DataFrame:
    """
    Load orders from the last *days_back* days from Alpaca.

    Returns a DataFrame with columns: symbol, side, status, qty,
    filled_qty, filled_avg_price, submitted_at, filled_at.
    Empty DataFrame on failure.
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    from datetime import timedelta

    client = _get_alpaca_client()
    if client is None:
        return pd.DataFrame()

    after = datetime.now(timezone.utc) - timedelta(days=days_back)
    try:
        orders = client.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                after=after,
                limit=500,
            )
        )
        if not orders:
            return pd.DataFrame()

        rows = []
        for o in orders:
            side_val = o.side.value if hasattr(o.side, "value") else str(o.side)
            status_val = o.status.value if hasattr(o.status, "value") else str(o.status)
            rows.append({
                "order_id":         str(o.id),
                "symbol":           o.symbol,
                "side":             side_val,
                "status":           status_val,
                "qty":              float(o.qty or 0),
                "filled_qty":       float(o.filled_qty or 0),
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                "submitted_at":     pd.Timestamp(o.submitted_at) if o.submitted_at else None,
                "filled_at":        pd.Timestamp(o.filled_at) if o.filled_at else None,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _find_order(row: pd.Series, orders_df: pd.DataFrame) -> dict | None:
    """
    Find the Alpaca order that matches a trade decision.

    Matches on ticker + side + submitted_at within ±5 minutes of the
    decision time.  Returns the closest match or None.
    """
    if orders_df.empty or orders_df.get("submitted_at") is None:
        return None

    ticker = str(row["ticker"])
    signal = int(row["final_signal"])
    side   = "buy" if signal == 1 else "sell"

    ts = row["time"]
    if hasattr(ts, "tz_localize") and ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = pd.Timestamp(ts)

    window = pd.Timedelta(minutes=5)
    submitted = orders_df["submitted_at"]

    mask = (
        (orders_df["symbol"] == ticker)
        & (orders_df["side"] == side)
        & submitted.notna()
        & (submitted >= ts - window)
        & (submitted <= ts + window)
    )
    matches = orders_df[mask]
    if matches.empty:
        return None

    # Pick the temporally closest match
    closest_idx = (matches["submitted_at"] - ts).abs().idxmin()
    return matches.loc[closest_idx].to_dict()


def _fill_badge(order: dict | None) -> str:
    """Return a short HTML badge string for the order fill status."""
    if order is None:
        return "<span style='color:grey'>⚪ No order</span>"
    status = order.get("status", "")
    emoji, label = _ORDER_STATUS_BADGE.get(status, ("❓", status.title()))
    filled_qty = order.get("filled_qty", 0)
    avg_price  = order.get("filled_avg_price")

    detail = ""
    if filled_qty and avg_price:
        detail = f" &nbsp;{filled_qty:.0f} sh @ ${avg_price:.2f}"
    elif filled_qty:
        detail = f" &nbsp;{filled_qty:.0f} sh"

    color = {
        "filled":           "#22c55e",
        "partially_filled": "#eab308",
        "cancelled":        "#ef4444",
        "rejected":         "#ef4444",
    }.get(status, "#94a3b8")

    return (
        f"<span style='color:{color}'>{emoji} {label}{detail}</span>"
    )


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _render_equity_chart(
    history_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    period_label: str,
    orders_df: pd.DataFrame,
) -> None:
    if history_df.empty:
        st.info("Portfolio history unavailable. Check Alpaca credentials.")
        return

    # Convert all timestamps to ET so the x-axis shows New York market time.
    history_df = history_df.copy()
    history_df["time"] = history_df["time"].dt.tz_convert("America/New_York")

    fig = go.Figure()

    # ---- Equity line ----
    fig.add_trace(go.Scatter(
        x=history_df["time"],
        y=history_df["equity"],
        mode="lines",
        name="Equity",
        line=dict(color="#38bdf8", width=2),
        hovertemplate="%{x|%b %d %H:%M}<br>$%{y:,.2f}<extra></extra>",
    ))

    # ---- P&L fill under equity line ----
    if not history_df["profit_loss"].isna().all():
        base = float(history_df["equity"].iloc[0])
        fig.add_trace(go.Scatter(
            x=history_df["time"],
            y=history_df["equity"],
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.07)",
            showlegend=False,
            hoverinfo="skip",
        ))

    # ---- BUY / SELL trade markers ----
    if not trades_df.empty:
        time_min = history_df["time"].min()
        time_max = history_df["time"].max()

        t_col = pd.to_datetime(trades_df["time"], utc=True).dt.tz_convert("America/New_York")
        mask = (t_col >= time_min) & (t_col <= time_max)
        in_range = trades_df[mask].copy()
        # Keep timezone by using a list of Timestamps rather than .values
        # (.values strips tz → tz-naive datetime64, causing subtract errors)
        in_range["_t"] = t_col[mask].tolist()

        # Only show markers for decisions that produced an actual Alpaca order.
        if not orders_df.empty:
            submitted_mask = in_range.apply(
                lambda r: _find_order(r, orders_df) is not None, axis=1
            )
            in_range = in_range[submitted_mask]

        def _nearest_equity(t: pd.Timestamp) -> float:
            ts = pd.Timestamp(t)
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            idx = (history_df["time"] - ts).abs().idxmin()
            return float(history_df.loc[idx, "equity"])

        buys  = in_range[in_range["final_signal"] == 1]
        sells = in_range[in_range["final_signal"] == -1]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["_t"],
                y=[_nearest_equity(t) for t in buys["_t"]],
                mode="markers+text",
                name="BUY",
                marker=dict(symbol="triangle-up", size=11, color="#22c55e",
                            line=dict(width=1, color="#15803d")),
                text=buys["ticker"],
                textposition="top center",
                textfont=dict(size=8, color="#22c55e"),
                hovertemplate="%{text}<br>BUY @ %{x|%H:%M}<extra></extra>",
            ))

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["_t"],
                y=[_nearest_equity(t) for t in sells["_t"]],
                mode="markers+text",
                name="SELL",
                marker=dict(symbol="triangle-down", size=11, color="#ef4444",
                            line=dict(width=1, color="#b91c1c")),
                text=sells["ticker"],
                textposition="bottom center",
                textfont=dict(size=8, color="#ef4444"),
                hovertemplate="%{text}<br>SELL @ %{x|%H:%M}<extra></extra>",
            ))

    # Y-axis range: pad 5 % above/below the actual data extent so the chart
    # doesn't stretch all the way to zero (caused by fill="tozeroy").
    eq_vals = history_df["equity"].dropna()
    eq_min, eq_max = float(eq_vals.min()), float(eq_vals.max())
    y_pad = max((eq_max - eq_min) * 0.10, eq_max * 0.002)

    fig.update_layout(
        title=dict(text=f"Portfolio Equity — {period_label} (ET)", font=dict(size=14)),
        xaxis_title=None,
        yaxis_title="Equity ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        height=360,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)", showgrid=True),
        yaxis=dict(
            range=[eq_min - y_pad, eq_max + y_pad],
            gridcolor="rgba(128,128,128,0.15)",
            showgrid=True,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _render_metrics(acct: dict | None, history_df: pd.DataFrame) -> None:
    """Render the four top-level KPI tiles."""
    col_eq, col_day, col_week, col_cash = st.columns(4)

    equity = acct["equity"] if acct else None
    cash   = acct["cash"]   if acct else None

    # Compute daily / weekly deltas from portfolio history
    daily_delta = daily_pct = weekly_delta = weekly_pct = None

    if not history_df.empty and equity is not None:
        now_utc = pd.Timestamp.now(tz="UTC")

        # Daily: first equity value of today
        today_start = now_utc.normalize()
        today_rows = history_df[history_df["time"] >= today_start]
        if not today_rows.empty:
            day_base    = float(today_rows["equity"].iloc[0])
            daily_delta = equity - day_base
            daily_pct   = daily_delta / day_base if day_base else None

        # Weekly: first equity value of this Mon
        week_start = (now_utc - pd.Timedelta(days=now_utc.dayofweek)).normalize()
        week_rows = history_df[history_df["time"] >= week_start]
        if not week_rows.empty:
            week_base    = float(week_rows["equity"].iloc[0])
            weekly_delta = equity - week_base
            weekly_pct   = weekly_delta / week_base if week_base else None

    def _fmt_delta(delta: float | None, pct: float | None) -> str | None:
        if delta is None:
            return None
        sign = "+" if delta >= 0 else ""
        pct_str = f" ({sign}{pct:.2%})" if pct is not None else ""
        return f"{sign}${delta:,.2f}{pct_str}"

    def _delta_color(delta: float | None) -> str:
        if delta is None:
            return "off"
        return "normal" if delta >= 0 else "inverse"

    with col_eq:
        st.metric(
            "💰 Equity",
            f"${equity:,.2f}" if equity is not None else "—",
        )
    with col_day:
        st.metric(
            "📈 Daily Change",
            _fmt_delta(daily_delta, daily_pct) or "—",
            delta=f"{daily_pct:.2%}" if daily_pct is not None else None,
            delta_color=_delta_color(daily_delta),
        )
    with col_week:
        st.metric(
            "📅 Weekly Change",
            _fmt_delta(weekly_delta, weekly_pct) or "—",
            delta=f"{weekly_pct:.2%}" if weekly_pct is not None else None,
            delta_color=_delta_color(weekly_delta),
        )
    with col_cash:
        st.metric(
            "💵 Cash",
            f"${cash:,.2f}" if cash is not None else "—",
        )


# ---------------------------------------------------------------------------
# Trade renderer
# ---------------------------------------------------------------------------

def _render_trade(row: pd.Series, orders_df: pd.DataFrame) -> None:
    """Render one trade as a labelled expander."""
    ts = row["time"]
    if hasattr(ts, "tz_localize") and ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    ts_str = str(ts)[:19].replace("T", " ")

    sig   = int(row["final_signal"])
    emoji = _SIGNAL_COLOR.get(sig, "⚪")
    label = _SIGNAL_LABEL.get(sig, str(sig))
    score = f"{float(row['score']):.4f}" if row["score"] is not None else "—"

    # Fill-status badge — compact version for header, full version inside
    order = _find_order(row, orders_df)
    fill  = _fill_badge(order)

    # Compact fill indicator for the header (emoji + label only, no price detail)
    if order is None:
        fill_short = "⚪"
    else:
        status = order.get("status", "")
        fill_short = _ORDER_STATUS_BADGE.get(status, ("❓", ""))[0]

    header = (
        f"{emoji} {ts_str}  |  **{row['ticker']}**  |  {label}  |  "
        f"score {score}  |  {fill_short}"
    )

    with st.expander(header, expanded=False):
        # Full fill status with share count and price
        st.markdown(
            f"**Alpaca fill status:** {fill}",
            unsafe_allow_html=True,
        )
        st.divider()

        col_hmm, col_ou, col_llm = st.columns(3)

        # ---- HMM ----
        with col_hmm:
            st.markdown("#### HMM Regime")
            regime_label = row.get("regime_label") or "—"
            st.markdown(_regime_badge(regime_label))
            probs = _parse_json(row.get("regime_probs"))
            _render_regime_probs(probs)
            hmm_sig  = row.get("hmm_signal")
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
                ou_z      = row.get("ou_zscore")
                ou_spread = row.get("ou_spread_value")
                ou_sig    = row.get("ou_signal")
                ou_conf   = row.get("ou_confidence")
                st.metric("Z-score", f"{float(ou_z):.3f}" if ou_z is not None else "—")
                st.metric("Spread",  f"{float(ou_spread):.5f}" if ou_spread is not None else "—")
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
            llm_sig  = row.get("llm_signal")
            llm_conf = row.get("llm_confidence")
            llm_label = (
                _SIGNAL_LABEL.get(int(llm_sig), "NEUTRAL")
                if llm_sig is not None else "—"
            )
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
            w_cols = st.columns(3)
            for col, (name, val) in zip(w_cols, weights.items()):
                col.metric(name.replace("_", " ").title(), f"{float(val):.3f}")

        # ---- Contributing Headlines ----
        headlines = _parse_json(row.get("contributing_headlines"))
        if headlines:
            st.markdown(f"#### News Headlines ({len(headlines)})")
            for h in headlines:
                title    = h.get("title", "—")
                source   = h.get("source") or "unknown source"
                pub      = str(h.get("published_at", ""))[:16].replace("T", " ")
                av_label = h.get("av_sentiment_label", "")
                av_score = h.get("av_sentiment_score")
                av_str   = (
                    f"AV: {av_label} ({float(av_score):.3f})"
                    if av_score is not None else f"AV: {av_label}"
                )
                st.markdown(
                    f"- {title}  \n"
                    f"  <small>{source} · {pub} · {av_str}</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No headlines recorded for this signal.")


# ---------------------------------------------------------------------------
# UI helpers (unchanged from original)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Four-Point AI Trader",
        page_icon="📈",
        layout="wide",
    )

    # Auto-refresh every 60 seconds
    st_autorefresh(interval=60_000, key="trade_refresh")

    st.title("📈 Four-Point AI Trader")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Settings")
        limit = st.selectbox("Trades to display", [25, 50, 100], index=0)
        period_label = st.selectbox(
            "Chart period",
            list(_CHART_PERIOD_MAP.keys()),
            index=0,
        )
        st.caption("Auto-refreshes every 60 s.")

    period, timeframe_str = _CHART_PERIOD_MAP[period_label]

    # Extend the orders lookback to cover the selected chart period so that
    # markers for older bars can be matched to actual submitted orders.
    _PERIOD_DAYS = {"Today (1D)": 2, "1 Week": 8, "1 Month": 32}
    orders_days_back = _PERIOD_DAYS.get(period_label, 2)

    # ---- Load all data ----
    acct       = _load_account_info()
    history_df = _load_portfolio_history(period, timeframe_str)
    orders_df  = _load_orders(orders_days_back)

    try:
        trades_df = _load_trades(limit)
        chart_trades_df = _load_all_trades_for_chart()
    except Exception as exc:
        st.error(f"Failed to load trades from DB: {exc}")
        return

    # ---- Metrics row ----
    _render_metrics(acct, history_df)

    st.divider()

    # ---- Equity chart ----
    _render_equity_chart(history_df, chart_trades_df, period_label, orders_df)

    st.divider()

    # ---- Trade log ----
    st.subheader("Trade Decision Log")
    if trades_df.empty:
        st.info("No trades recorded yet.")
        return

    st.caption(
        f"Showing {len(trades_df)} most recent trades. "
        f"Last refreshed: {pd.Timestamp.now().strftime('%H:%M:%S')}"
    )

    for _, row in trades_df.iterrows():
        _render_trade(row, orders_df)


if __name__ == "__main__":
    main()
