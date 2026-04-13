"""
Four-Point AI Trader — Dashboard v2
=====================================
Four-tab Streamlit app:
  📊 Overview     — portfolio metrics, equity curve, open positions, trade log
  🔍 Ticker Detail — candlestick + regime bands + OU z-score + MWU score subplots
  📉 Signals       — MWU weight evolution, signal win rates, signal agreement matrix
  📰 News          — recent headlines with sentiment scores

Run from trading_engine/:
    .venv/bin/streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlalchemy
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / ".env")

DB_URL            = os.environ.get("DB_URL", "")
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
_PAPER            = "paper" in ALPACA_BASE_URL

_SIGNAL_LABEL = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}
_SIGNAL_COLOR = {1: "🟢", -1: "🔴", 0: "⚪"}
_REGIME_EMOJI = {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}
_REGIME_FILL  = {
    "bull":    "rgba(34,197,94,0.12)",
    "neutral": "rgba(234,179,8,0.08)",
    "bear":    "rgba(239,68,68,0.12)",
}
_REGIME_INT_LABEL = {0: "bear", 1: "neutral", 2: "bull"}

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
_PERIOD_TO_DAYS = {"Today (1D)": 1, "1 Week": 7, "1 Month": 30}

_BG   = "rgba(0,0,0,0)"
_GRID = "rgba(128,128,128,0.15)"

_SIGNALS = ["hmm_regime", "ou_spread", "llm_sentiment", "analyst_recs"]
_SIGNAL_COL_MAP = {
    "hmm_regime":    "hmm_signal",
    "ou_spread":     "ou_signal",
    "llm_sentiment": "llm_signal",
    "analyst_recs":  "analyst_signal",
}
_WEIGHT_COLORS = {
    "hmm_regime":    "#38bdf8",
    "ou_spread":     "#f59e0b",
    "llm_sentiment": "#a78bfa",
    "analyst_recs":  "#34d399",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_json(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (TypeError, ValueError):
        return None


def _to_et(series: pd.Series) -> pd.Series:
    """Convert a datetime series to America/New_York (handles tz-naive → UTC first)."""
    if series.empty:
        return series
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    return series.dt.tz_convert("America/New_York")


def _ts_to_et(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("America/New_York")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_db_engine() -> sqlalchemy.Engine:
    if not DB_URL:
        st.error("DB_URL not set. Check your .env file.")
        st.stop()
    return sqlalchemy.create_engine(DB_URL, pool_pre_ping=True)


@st.cache_data(ttl=30)
def _load_trades(limit: int) -> pd.DataFrame:
    sql = sqlalchemy.text("""
        SELECT id, time, ticker, final_signal, score,
               regime, regime_label, regime_probs,
               hmm_signal, hmm_confidence,
               ou_signal, ou_confidence, ou_zscore, ou_spread_value, ou_pair,
               llm_signal, llm_confidence,
               analyst_signal, analyst_confidence,
               mwu_weights, contributing_headlines
        FROM   trade_log
        ORDER  BY time DESC
        LIMIT  :limit
    """)
    with _get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"limit": limit})


@st.cache_data(ttl=30)
def _load_trades_for_ticker(ticker: str, limit: int) -> pd.DataFrame:
    sql = sqlalchemy.text("""
        SELECT id, time, ticker, final_signal, score,
               regime, regime_label, regime_probs,
               hmm_signal, hmm_confidence,
               ou_signal, ou_confidence, ou_zscore, ou_spread_value, ou_pair,
               llm_signal, llm_confidence,
               analyst_signal, analyst_confidence,
               mwu_weights, contributing_headlines
        FROM   trade_log
        WHERE  ticker = :ticker
        ORDER  BY time DESC
        LIMIT  :limit
    """)
    with _get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"ticker": ticker, "limit": limit})


@st.cache_data(ttl=60)
def _load_all_tickers() -> list[str]:
    sql = sqlalchemy.text("SELECT DISTINCT ticker FROM trade_log ORDER BY ticker")
    with _get_db_engine().connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=60)
def _load_ohlcv(ticker: str, days_back: int) -> pd.DataFrame:
    sql = sqlalchemy.text("""
        SELECT time, open, high, low, close, volume
        FROM   ohlcv
        WHERE  ticker = :ticker
          AND  time  >= NOW() - (:days * INTERVAL '1 day')
        ORDER  BY time
    """)
    with _get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"ticker": ticker, "days": days_back})


@st.cache_data(ttl=60)
def _load_regime_history(ticker: str, days_back: int) -> pd.DataFrame:
    sql = sqlalchemy.text("""
        SELECT time, regime
        FROM   regime_log
        WHERE  ticker = :ticker
          AND  time  >= NOW() - (:days * INTERVAL '1 day')
        ORDER  BY time
    """)
    with _get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"ticker": ticker, "days": days_back})


@st.cache_data(ttl=120)
def _load_weight_evolution() -> pd.DataFrame:
    sql = sqlalchemy.text("""
        SELECT time, ticker, regime_label, mwu_weights
        FROM   trade_log
        WHERE  time        >= NOW() - INTERVAL '30 days'
          AND  mwu_weights IS NOT NULL
        ORDER  BY time
    """)
    with _get_db_engine().connect() as conn:
        df = pd.read_sql(sql, conn)
    if df.empty:
        return df
    weights_parsed = df["mwu_weights"].apply(_parse_json)
    for sig in _SIGNALS:
        df[sig] = weights_parsed.apply(
            lambda w, s=sig: float(w[s]) if isinstance(w, dict) and s in w else None
        )
    return df


@st.cache_data(ttl=300)
def _compute_win_rates() -> pd.DataFrame:
    """LATERAL JOIN trade_log × ohlcv for per-signal forward-return correctness."""
    sql = sqlalchemy.text("""
        SELECT
            t.id, t.time, t.ticker, t.final_signal,
            t.hmm_signal, t.ou_signal, t.llm_signal, t.analyst_signal,
            base_bar.close   AS close_at,
            bar_1m.close     AS close_1m,
            bar_15m.close    AS close_15m,
            bar_1h.close     AS close_1h
        FROM trade_log t
        LEFT JOIN LATERAL (
            SELECT close FROM ohlcv
            WHERE ticker = t.ticker AND time <= t.time
            ORDER BY time DESC LIMIT 1
        ) base_bar ON TRUE
        LEFT JOIN LATERAL (
            SELECT close FROM ohlcv
            WHERE ticker = t.ticker AND time > t.time
            ORDER BY time ASC LIMIT 1
        ) bar_1m ON TRUE
        LEFT JOIN LATERAL (
            SELECT close FROM ohlcv
            WHERE ticker = t.ticker AND time >= t.time + INTERVAL '14 minutes'
            ORDER BY time ASC LIMIT 1
        ) bar_15m ON TRUE
        LEFT JOIN LATERAL (
            SELECT close FROM ohlcv
            WHERE ticker = t.ticker AND time >= t.time + INTERVAL '59 minutes'
            ORDER BY time ASC LIMIT 1
        ) bar_1h ON TRUE
        WHERE t.time >= NOW() - INTERVAL '30 days'
          AND t.final_signal != 0
    """)
    with _get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn)


@st.cache_data(ttl=120)
def _load_news(ticker: str | None, limit: int) -> pd.DataFrame:
    if ticker and ticker != "All":
        sql = sqlalchemy.text("""
            SELECT ticker, title, source, sentiment_score, sentiment_confidence,
                   llm_direction, fetched_at
            FROM   news
            WHERE  ticker = :ticker
            ORDER  BY fetched_at DESC LIMIT :limit
        """)
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
    else:
        sql = sqlalchemy.text("""
            SELECT ticker, title, source, sentiment_score, sentiment_confidence,
                   llm_direction, fetched_at
            FROM   news
            ORDER  BY fetched_at DESC LIMIT :limit
        """)
        params = {"limit": limit}
    with _get_db_engine().connect() as conn:
        return pd.read_sql(sql, conn, params=params)


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
            "time":            [pd.Timestamp(ts, unit="s", tz="UTC") for ts in history.timestamp],
            "equity":          history.equity,
            "profit_loss":     history.profit_loss,
            "profit_loss_pct": history.profit_loss_pct,
        })
        return df.dropna(subset=["equity"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def _load_orders(days_back: int = 1) -> pd.DataFrame:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
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
            side_val   = o.side.value   if hasattr(o.side,   "value") else str(o.side)
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
                "filled_at":        pd.Timestamp(o.filled_at)    if o.filled_at    else None,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def _load_open_positions() -> pd.DataFrame:
    client = _get_alpaca_client()
    if client is None:
        return pd.DataFrame()
    try:
        positions = client.get_all_positions()
        if not positions:
            return pd.DataFrame()
        rows = []
        for p in positions:
            rows.append({
                "Symbol":          p.symbol,
                "Side":            p.side.value if hasattr(p.side, "value") else str(p.side),
                "Qty":             float(p.qty),
                "Avg Entry ($)":   float(p.avg_entry_price),
                "Price ($)":       float(p.current_price),
                "Mkt Value ($)":   float(p.market_value),
                "Unreal P&L ($)":  float(p.unrealized_pl),
                "P&L %":           round(float(p.unrealized_plpc) * 100, 2),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _find_order(row: pd.Series, orders_df: pd.DataFrame) -> dict | None:
    if orders_df.empty:
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
        & (orders_df["side"]   == side)
        & submitted.notna()
        & (submitted >= ts - window)
        & (submitted <= ts + window)
    )
    matches = orders_df[mask]
    if matches.empty:
        return None
    return matches.loc[(matches["submitted_at"] - ts).abs().idxmin()].to_dict()


def _fill_badge(order: dict | None) -> str:
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
    color = {"filled": "#22c55e", "partially_filled": "#eab308",
             "cancelled": "#ef4444", "rejected": "#ef4444"}.get(status, "#94a3b8")
    return f"<span style='color:{color}'>{emoji} {label}{detail}</span>"


# ---------------------------------------------------------------------------
# Chart: equity curve (Overview)
# ---------------------------------------------------------------------------

def _render_equity_chart(
    history_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    period_label: str,
    filled_orders_df: pd.DataFrame,
) -> None:
    if history_df.empty:
        st.info("Portfolio history unavailable — check Alpaca credentials.")
        return

    history_df = history_df.copy()
    history_df["time"] = _to_et(history_df["time"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["time"], y=history_df["equity"],
        mode="lines", name="Equity",
        line=dict(color="#38bdf8", width=2),
        hovertemplate="%{x|%b %d %H:%M}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=history_df["time"], y=history_df["equity"],
        mode="none", fill="tozeroy",
        fillcolor="rgba(56,189,248,0.07)",
        showlegend=False, hoverinfo="skip",
    ))

    if not trades_df.empty and not filled_orders_df.empty:
        time_min = history_df["time"].min()
        time_max = history_df["time"].max()
        t_col = _to_et(pd.to_datetime(trades_df["time"], utc=True))
        mask = (t_col >= time_min) & (t_col <= time_max)
        in_range = trades_df[mask].copy()
        in_range["_t"] = t_col[mask].tolist()

        fill_mask = in_range.apply(
            lambda r: _find_order(r, filled_orders_df) is not None, axis=1
        )
        in_range = in_range[fill_mask]

        def _nearest_equity(t: pd.Timestamp) -> float:
            ts = pd.Timestamp(t)
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            return float(history_df.loc[(history_df["time"] - ts).abs().idxmin(), "equity"])

        for signal_val, sym, color, border, pos in [
            (1,  "triangle-up",   "#22c55e", "#15803d", "top center"),
            (-1, "triangle-down", "#ef4444", "#b91c1c", "bottom center"),
        ]:
            subset = in_range[in_range["final_signal"] == signal_val]
            if not subset.empty:
                label = "BUY (filled)" if signal_val == 1 else "SELL (filled)"
                fig.add_trace(go.Scatter(
                    x=subset["_t"],
                    y=[_nearest_equity(t) for t in subset["_t"]],
                    mode="markers+text", name=label,
                    marker=dict(symbol=sym, size=11, color=color,
                                line=dict(width=1, color=border)),
                    text=subset["ticker"], textposition=pos,
                    textfont=dict(size=8, color=color),
                    hovertemplate=f"%{{text}}<br>{label} @ %{{x|%H:%M}}<extra></extra>",
                ))

    eq_vals = history_df["equity"].dropna()
    eq_min, eq_max = float(eq_vals.min()), float(eq_vals.max())
    y_pad = max((eq_max - eq_min) * 0.10, eq_max * 0.002)

    fig.update_layout(
        title=dict(text=f"Portfolio Equity — {period_label} (ET)", font=dict(size=14)),
        xaxis_title=None, yaxis_title="Equity ($)", yaxis_tickformat="$,.0f",
        hovermode="x unified", height=360,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(range=[eq_min - y_pad, eq_max + y_pad], gridcolor=_GRID),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Chart: ticker detail (candlestick + regime bands + subplots)
# ---------------------------------------------------------------------------

def _regime_spans(regime_df: pd.DataFrame) -> list[dict[str, Any]]:
    if regime_df.empty:
        return []
    spans: list[dict[str, Any]] = []
    cur_label = _REGIME_INT_LABEL.get(int(regime_df.iloc[0]["regime"]), "neutral")
    cur_start = regime_df.iloc[0]["time"]
    for _, row in regime_df.iloc[1:].iterrows():
        label = _REGIME_INT_LABEL.get(int(row["regime"]), "neutral")
        if label != cur_label:
            spans.append({"start": cur_start, "end": row["time"], "label": cur_label})
            cur_label, cur_start = label, row["time"]
    spans.append({"start": cur_start, "end": pd.Timestamp.now(tz="UTC"), "label": cur_label})
    return spans


def _render_ticker_chart(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    score_df: pd.DataFrame,
    filled_orders: pd.DataFrame,
) -> None:
    if ohlcv_df.empty:
        st.info(f"No OHLCV data found for {ticker} in the selected period.")
        return

    ohlcv_df = ohlcv_df.copy()
    ohlcv_df["time"] = _to_et(pd.to_datetime(ohlcv_df["time"], utc=True))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.175, 0.175],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} Price (ET)", "OU Z-Score", "MWU Ensemble Score"),
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlcv_df["time"],
        open=ohlcv_df["open"], high=ohlcv_df["high"],
        low=ohlcv_df["low"],   close=ohlcv_df["close"],
        name="OHLCV",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    # Regime background bands
    for span in _regime_spans(regime_df):
        fig.add_vrect(
            x0=_ts_to_et(span["start"]),
            x1=_ts_to_et(span["end"]),
            fillcolor=_REGIME_FILL.get(span["label"], _REGIME_FILL["neutral"]),
            opacity=1.0, layer="below", line_width=0,
        )

    # Confirmed-fill markers on price row
    if not filled_orders.empty:
        ticker_fills = filled_orders[filled_orders["symbol"] == ticker]
        if not ticker_fills.empty:
            for side_val, sym, color, label in [
                ("buy",  "triangle-up",   "#22c55e", "BUY fill"),
                ("sell", "triangle-down", "#ef4444", "SELL fill"),
            ]:
                side_fills = ticker_fills[ticker_fills["side"] == side_val].copy()
                if side_fills.empty:
                    continue
                fill_times = side_fills["filled_at"].dropna().apply(_ts_to_et)
                if fill_times.empty:
                    continue
                # Snap each fill time to nearest OHLCV bar price
                prices = fill_times.apply(
                    lambda t: float(
                        ohlcv_df.loc[(ohlcv_df["time"] - t).abs().idxmin(), "close"]
                    )
                )
                avg_prices = side_fills.loc[fill_times.index, "filled_avg_price"]
                hover = [
                    f"{label}<br>Fill: ${ap:.2f}<br>@ {t.strftime('%H:%M')}"
                    for t, ap in zip(fill_times, avg_prices)
                ]
                fig.add_trace(go.Scatter(
                    x=list(fill_times), y=list(prices),
                    mode="markers", name=label,
                    marker=dict(symbol=sym, size=12, color=color,
                                line=dict(width=1.5, color="white")),
                    hovertext=hover, hoverinfo="text",
                ), row=1, col=1)

    # Row 2: OU z-score from trade_log.ou_zscore
    if not score_df.empty and "ou_zscore" in score_df.columns:
        ou = score_df.dropna(subset=["ou_zscore"]).copy()
        if not ou.empty:
            ou_times = _to_et(pd.to_datetime(ou["time"], utc=True))
            fig.add_trace(go.Scatter(
                x=list(ou_times), y=list(ou["ou_zscore"].astype(float)),
                mode="lines+markers", name="OU Z-score",
                line=dict(color="#f59e0b", width=1.5), marker=dict(size=4),
                hovertemplate="%{x|%H:%M}<br>z=%{y:.3f}<extra></extra>",
            ), row=2, col=1)
            for lvl in (2.0, -2.0):
                fig.add_hline(y=lvl, line_dash="dash",
                              line_color="rgba(245,158,11,0.4)",
                              line_width=1, row=2, col=1)

    # Row 3: MWU ensemble score from trade_log.score
    if not score_df.empty and "score" in score_df.columns:
        sc = score_df.dropna(subset=["score"]).copy()
        if not sc.empty:
            sc_times = _to_et(pd.to_datetime(sc["time"], utc=True))
            fig.add_trace(go.Scatter(
                x=list(sc_times), y=list(sc["score"].astype(float)),
                mode="lines+markers", name="MWU Score",
                line=dict(color="#a78bfa", width=1.5), marker=dict(size=4),
                hovertemplate="%{x|%H:%M}<br>score=%{y:.4f}<extra></extra>",
            ), row=3, col=1)
            for lvl in (0.2, -0.2):
                fig.add_hline(y=lvl, line_dash="dash",
                              line_color="rgba(167,139,250,0.4)",
                              line_width=1, row=3, col=1)
            fig.add_hline(y=0, line_dash="dot",
                          line_color="rgba(255,255,255,0.2)",
                          line_width=1, row=3, col=1)

    fig.update_layout(
        height=650, showlegend=True, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=_GRID, row=i, col=1)
        fig.update_yaxes(gridcolor=_GRID, row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

def _render_metrics(acct: dict | None, history_df: pd.DataFrame) -> None:
    col_eq, col_day, col_week, col_cash = st.columns(4)
    equity = acct["equity"] if acct else None
    cash   = acct["cash"]   if acct else None
    daily_delta = daily_pct = weekly_delta = weekly_pct = None

    if not history_df.empty and equity is not None:
        now_utc     = pd.Timestamp.now(tz="UTC")
        today_rows  = history_df[history_df["time"] >= now_utc.normalize()]
        if not today_rows.empty:
            day_base    = float(today_rows["equity"].iloc[0])
            daily_delta = equity - day_base
            daily_pct   = daily_delta / day_base if day_base else None

        week_start = (now_utc - pd.Timedelta(days=now_utc.dayofweek)).normalize()
        week_rows  = history_df[history_df["time"] >= week_start]
        if not week_rows.empty:
            week_base    = float(week_rows["equity"].iloc[0])
            weekly_delta = equity - week_base
            weekly_pct   = weekly_delta / week_base if week_base else None

    def _fmt(d: float | None, p: float | None) -> str | None:
        if d is None:
            return None
        sign = "+" if d >= 0 else ""
        return f"{sign}${d:,.2f}" + (f" ({sign}{p:.2%})" if p is not None else "")

    def _dc(d: float | None) -> str:
        return "normal" if (d or 0) >= 0 else "inverse"

    with col_eq:
        st.metric("💰 Equity", f"${equity:,.2f}" if equity else "—")
    with col_day:
        st.metric("📈 Daily P&L", _fmt(daily_delta, daily_pct) or "—",
                  delta=f"{daily_pct:.2%}" if daily_pct else None,
                  delta_color=_dc(daily_delta))
    with col_week:
        st.metric("📅 Weekly P&L", _fmt(weekly_delta, weekly_pct) or "—",
                  delta=f"{weekly_pct:.2%}" if weekly_pct else None,
                  delta_color=_dc(weekly_delta))
    with col_cash:
        st.metric("💵 Cash", f"${cash:,.2f}" if cash else "—")


# ---------------------------------------------------------------------------
# Trade expander
# ---------------------------------------------------------------------------

def _regime_badge(label: str | None) -> str:
    if not label:
        return "—"
    return f"{_REGIME_EMOJI.get(label.lower(), '⚪')} **{label.upper()}**"


def _render_regime_probs(probs: dict | None) -> None:
    if not probs:
        st.caption("No regime probability data.")
        return
    for label, p in sorted(probs.items(), key=lambda x: -x[1]):
        st.progress(float(p),
                    text=f"{_REGIME_EMOJI.get(label.lower(), '⚪')} {label.capitalize()}: {p:.1%}")


def _render_trade(row: pd.Series, orders_df: pd.DataFrame) -> None:
    ts = row["time"]
    if hasattr(ts, "tz_localize") and ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    ts_str   = str(ts)[:19].replace("T", " ")
    sig      = int(row["final_signal"])
    score    = f"{float(row['score']):.4f}" if row.get("score") is not None else "—"
    order    = _find_order(row, orders_df)
    status   = order.get("status", "") if order else ""
    fill_ico = _ORDER_STATUS_BADGE.get(status, ("⚪", ""))[0] if order else "⚪"

    header = (
        f"{_SIGNAL_COLOR.get(sig, '⚪')} {ts_str}  |  **{row['ticker']}**  |  "
        f"{_SIGNAL_LABEL.get(sig, str(sig))}  |  score {score}  |  {fill_ico}"
    )
    with st.expander(header, expanded=False):
        st.markdown(f"**Alpaca fill:** {_fill_badge(order)}", unsafe_allow_html=True)
        st.divider()

        col_hmm, col_ou, col_llm, col_analyst = st.columns(4)
        with col_hmm:
            st.markdown("#### HMM Regime")
            st.markdown(_regime_badge(row.get("regime_label")))
            _render_regime_probs(_parse_json(row.get("regime_probs")))
            hmm_sig, hmm_conf = row.get("hmm_signal"), row.get("hmm_confidence")
            st.caption(
                f"Signal: **{int(hmm_sig)}** | Conf: **{float(hmm_conf):.2f}**"
                if pd.notna(hmm_sig) and pd.notna(hmm_conf) else "No HMM data"
            )
        with col_ou:
            st.markdown("#### OU Spread")
            pair = row.get("ou_pair")
            if pair and pd.notna(pair):
                st.markdown(f"**Pair:** `{pair}`")
                ou_z = row.get("ou_zscore")
                ou_s = row.get("ou_spread_value")
                ou_sig, ou_conf = row.get("ou_signal"), row.get("ou_confidence")
                st.metric("Z-score", f"{float(ou_z):.3f}" if pd.notna(ou_z) else "—")
                st.metric("Spread",  f"{float(ou_s):.5f}" if pd.notna(ou_s) else "—")
                st.caption(
                    f"Signal: **{int(ou_sig)}** | Conf: **{float(ou_conf):.2f}**"
                    if pd.notna(ou_sig) and pd.notna(ou_conf) else ""
                )
            else:
                st.caption("No active pair for this ticker.")
        with col_llm:
            st.markdown("#### LLM Sentiment")
            llm_sig, llm_conf = row.get("llm_signal"), row.get("llm_confidence")
            llm_lbl = _SIGNAL_LABEL.get(int(llm_sig), "NEUTRAL") if pd.notna(llm_sig) else "—"
            st.markdown(f"Direction: **{llm_lbl}**")
            st.caption(
                f"Signal: **{int(llm_sig)}** | Conf: **{float(llm_conf):.2f}**"
                if pd.notna(llm_sig) and pd.notna(llm_conf) else "No LLM data"
            )
        with col_analyst:
            st.markdown("#### Analyst Recs")
            a_sig, a_conf = row.get("analyst_signal"), row.get("analyst_confidence")
            a_lbl = _SIGNAL_LABEL.get(int(a_sig), "NEUTRAL") if pd.notna(a_sig) else "—"
            st.markdown(f"Direction: **{a_lbl}**")
            st.caption(
                f"Signal: **{int(a_sig)}** | Conf: **{float(a_conf):.2f}**"
                if pd.notna(a_sig) and pd.notna(a_conf) else "No analyst data"
            )

        st.divider()
        weights = _parse_json(row.get("mwu_weights"))
        if weights:
            st.markdown("#### MWU Ensemble Weights")
            w_cols = st.columns(len(weights))
            for col, (name, val) in zip(w_cols, weights.items()):
                col.metric(name.replace("_", " ").title(), f"{float(val):.3f}")

        headlines = _parse_json(row.get("contributing_headlines"))
        if headlines:
            st.markdown(f"#### News Headlines ({len(headlines)})")
            for h in headlines:
                title    = h.get("title", "—")
                source   = h.get("source") or "unknown"
                pub      = str(h.get("published_at", ""))[:16].replace("T", " ")
                av_label = h.get("av_sentiment_label", "")
                av_score = h.get("av_sentiment_score")
                av_str   = (f"AV: {av_label} ({float(av_score):.3f})"
                            if av_score is not None else f"AV: {av_label}")
                st.markdown(
                    f"- {title}  \n  <small>{source} · {pub} · {av_str}</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No headlines recorded.")


# ---------------------------------------------------------------------------
# Signals tab helpers
# ---------------------------------------------------------------------------

def _render_weight_evolution(weight_df: pd.DataFrame, regime_filter: str) -> None:
    if weight_df.empty:
        st.info("No weight evolution data yet (requires trades in the last 30 days).")
        return
    df = weight_df.copy()
    if regime_filter != "All":
        df = df[df["regime_label"] == regime_filter.lower()]
    if df.empty:
        st.info(f"No decisions in regime '{regime_filter}' yet.")
        return
    df["time"] = _to_et(pd.to_datetime(df["time"], utc=True))

    fig = go.Figure()
    for sig in _SIGNALS:
        if sig not in df.columns:
            continue
        valid = df[sig].dropna()
        if valid.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df.loc[valid.index, "time"], y=valid.astype(float),
            mode="lines+markers",
            name=sig.replace("_", " ").title(),
            line=dict(color=_WEIGHT_COLORS.get(sig, "#ffffff"), width=2),
            marker=dict(size=4),
            hovertemplate=f"{sig}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"MWU Weight Evolution — Regime: {regime_filter} (ET)",
        yaxis=dict(range=[0, 1], title="Weight", gridcolor=_GRID),
        xaxis=dict(gridcolor=_GRID),
        hovermode="x unified", height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_win_rates(wr_df: pd.DataFrame) -> None:
    if wr_df.empty:
        st.info("No trade data for win rate computation.")
        return

    results = []
    for sig_name, sig_col in _SIGNAL_COL_MAP.items():
        if sig_col not in wr_df.columns:
            continue
        for hz_label, close_col in [("1m", "close_1m"), ("15m", "close_15m"), ("1h", "close_1h")]:
            if close_col not in wr_df.columns or "close_at" not in wr_df.columns:
                continue
            sub = wr_df.dropna(subset=[sig_col, close_col, "close_at"]).copy()
            sub = sub[sub[sig_col] != 0]
            if sub.empty:
                continue
            fwd = sub[close_col].astype(float) - sub["close_at"].astype(float)
            direction = fwd.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            correct = (sub[sig_col].astype(int) * direction) > 0
            results.append({
                "Signal":   sig_name.replace("_", " ").title(),
                "Horizon":  hz_label,
                "Win Rate": float(correct.mean()) * 100,
                "N":        len(correct),
            })

    if not results:
        st.info("Not enough labeled data to compute win rates yet.")
        return

    res = pd.DataFrame(results)
    _color_map = {
        "Hmm Regime":    _WEIGHT_COLORS["hmm_regime"],
        "Ou Spread":     _WEIGHT_COLORS["ou_spread"],
        "Llm Sentiment": _WEIGHT_COLORS["llm_sentiment"],
        "Analyst Recs":  _WEIGHT_COLORS["analyst_recs"],
    }

    fig = go.Figure()
    for sig in res["Signal"].unique():
        sub = res[res["Signal"] == sig]
        base_color = _color_map.get(sig, "#94a3b8")
        fig.add_trace(go.Bar(
            name=sig, x=sub["Horizon"], y=sub["Win Rate"],
            marker_color=[base_color if w >= 50 else "#ef4444" for w in sub["Win Rate"]],
            text=[f"{w:.1f}%\n(n={n})" for w, n in zip(sub["Win Rate"], sub["N"])],
            textposition="outside",
            customdata=sub["N"],
            hovertemplate=f"{sig} %{{x}}: %{{y:.1f}}% (n=%{{customdata}})<extra></extra>",
        ))

    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        barmode="group",
        title="Signal Win Rates — forward return horizons (last 30 days)",
        yaxis=dict(title="Win Rate (%)", range=[0, 115], gridcolor=_GRID),
        xaxis=dict(title="Horizon"),
        height=380, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_signal_matrix(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        st.info("No trade decisions to display.")
        return

    display = trades_df.head(50).copy()
    display["time"] = _to_et(pd.to_datetime(display["time"], utc=True))
    display["Trade"] = (
        display["time"].dt.strftime("%m/%d %H:%M") + "  " + display["ticker"]
    )

    cols_to_show = {
        "HMM":     "hmm_signal",
        "OU":      "ou_signal",
        "LLM":     "llm_signal",
        "Analyst": "analyst_signal",
        "Final":   "final_signal",
    }
    matrix = pd.DataFrame({"Trade": display["Trade"].values})
    for col_label, src_col in cols_to_show.items():
        if src_col in display.columns:
            matrix[col_label] = display[src_col].apply(
                lambda v: _SIGNAL_LABEL.get(int(v), "?") if pd.notna(v) else "—"
            ).values

    def _cell_color(val: str) -> str:
        if val == "BUY":
            return "background-color: rgba(34,197,94,0.25); color: #22c55e"
        if val == "SELL":
            return "background-color: rgba(239,68,68,0.25); color: #ef4444"
        return "color: #94a3b8"

    styled = matrix.set_index("Trade").style.applymap(_cell_color)
    st.dataframe(styled, use_container_width=True, height=500)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Four-Point AI Trader",
        page_icon="📈",
        layout="wide",
    )
    st_autorefresh(interval=60_000, key="trade_refresh")
    st.title("📈 Four-Point AI Trader")

    with st.sidebar:
        st.header("Settings")
        limit        = st.selectbox("Trades to display", [25, 50, 100], index=0)
        period_label = st.selectbox("Chart period", list(_CHART_PERIOD_MAP.keys()), index=0)
        st.caption(f"Auto-refreshes every 60 s.  \nLast: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    period_alpaca, timeframe_str = _CHART_PERIOD_MAP[period_label]
    days_back   = _PERIOD_TO_DAYS[period_label]
    orders_days = max(days_back + 1, 2)

    # Shared data loaded once
    acct          = _load_account_info()
    history_df    = _load_portfolio_history(period_alpaca, timeframe_str)
    all_orders    = _load_orders(orders_days)
    filled_orders = (
        all_orders[all_orders["status"] == "filled"].copy()
        if not all_orders.empty else pd.DataFrame()
    )

    try:
        trades_df   = _load_trades(limit)
        all_tickers = _load_all_tickers()
    except Exception as exc:
        st.error(f"Failed to load data from DB: {exc}")
        return

    tab_overview, tab_ticker, tab_signals, tab_news = st.tabs([
        "📊 Overview", "🔍 Ticker Detail", "📉 Signals", "📰 News",
    ])

    # ── Overview ─────────────────────────────────────────────────────────────
    with tab_overview:
        _render_metrics(acct, history_df)
        st.divider()

        try:
            chart_trades = _load_trades(500)
        except Exception:
            chart_trades = pd.DataFrame()
        _render_equity_chart(history_df, chart_trades, period_label, filled_orders)

        st.subheader("Open Positions")
        pos_df = _load_open_positions()
        if pos_df.empty:
            st.info("No open positions.")
        else:
            def _pnl_color(val: float) -> str:
                return "color: #22c55e" if val >= 0 else "color: #ef4444"
            st.dataframe(
                pos_df.style.applymap(_pnl_color, subset=["Unreal P&L ($)", "P&L %"]),
                use_container_width=True, hide_index=True,
            )

        st.divider()
        st.subheader("Trade Decision Log")
        if trades_df.empty:
            st.info("No trades recorded yet.")
        else:
            st.caption(f"Showing {len(trades_df)} most recent decisions.")
            for _, row in trades_df.iterrows():
                _render_trade(row, all_orders)

    # ── Ticker Detail ─────────────────────────────────────────────────────────
    with tab_ticker:
        if not all_tickers:
            st.info("No tickers in trade_log yet.")
        else:
            col_t, col_p = st.columns([2, 1])
            with col_t:
                sel_ticker = st.selectbox("Ticker", all_tickers, key="ticker_sel")
            with col_p:
                sel_period = st.selectbox(
                    "Period", list(_CHART_PERIOD_MAP.keys()), index=0, key="ticker_period"
                )
            sel_days  = _PERIOD_TO_DAYS[sel_period]
            ohlcv_df  = _load_ohlcv(sel_ticker, sel_days)
            regime_df = _load_regime_history(sel_ticker, sel_days)
            score_df  = _load_trades_for_ticker(sel_ticker, 500)
            _render_ticker_chart(sel_ticker, ohlcv_df, regime_df, score_df, filled_orders)

            st.subheader(f"Recent Decisions — {sel_ticker}")
            ticker_trades = _load_trades_for_ticker(sel_ticker, limit)
            if ticker_trades.empty:
                st.info(f"No decisions recorded for {sel_ticker}.")
            else:
                for _, row in ticker_trades.iterrows():
                    _render_trade(row, all_orders)

    # ── Signals ───────────────────────────────────────────────────────────────
    with tab_signals:
        st.subheader("MWU Weight Evolution")
        regime_filter = st.selectbox(
            "Filter by regime", ["All", "Bull", "Neutral", "Bear"],
            key="weight_regime",
        )
        try:
            _render_weight_evolution(_load_weight_evolution(), regime_filter)
        except Exception as exc:
            st.warning(f"Weight evolution unavailable: {exc}")

        st.divider()
        st.subheader("Signal Win Rates (last 30 days)")
        try:
            _render_win_rates(_compute_win_rates())
        except Exception as exc:
            st.warning(f"Win rate computation unavailable: {exc}")

        st.divider()
        st.subheader("Signal Agreement Matrix — last 50 decisions")
        try:
            _render_signal_matrix(_load_trades(50))
        except Exception as exc:
            st.warning(f"Signal matrix unavailable: {exc}")

    # ── News ──────────────────────────────────────────────────────────────────
    with tab_news:
        col_nt, col_nl = st.columns([2, 1])
        with col_nt:
            news_ticker = st.selectbox(
                "Filter by ticker", ["All"] + all_tickers, key="news_ticker"
            )
        with col_nl:
            news_limit = st.selectbox("Show", [50, 100, 200], key="news_limit")

        try:
            news_df = _load_news(
                ticker=None if news_ticker == "All" else news_ticker,
                limit=news_limit,
            )
        except Exception as exc:
            st.error(f"Failed to load news: {exc}")
            news_df = pd.DataFrame()

        if news_df.empty:
            st.info("No news records found.")
        else:
            if "fetched_at" in news_df.columns:
                news_df["fetched_at"] = (
                    _to_et(pd.to_datetime(news_df["fetched_at"], utc=True))
                    .dt.strftime("%m/%d %H:%M ET")
                )
            if "llm_direction" in news_df.columns:
                news_df["Sentiment"] = news_df["llm_direction"].apply(
                    lambda v: _SIGNAL_LABEL.get(int(v), "—") if pd.notna(v) else "—"
                )
                news_df = news_df.drop(columns=["llm_direction"])

            col_renames = {
                "ticker":               "Ticker",
                "fetched_at":           "Fetched (ET)",
                "title":                "Headline",
                "source":               "Source",
                "sentiment_score":      "Score",
                "sentiment_confidence": "Conf",
            }
            display_cols = [c for c in col_renames if c in news_df.columns]
            if "Sentiment" in news_df.columns:
                display_cols.append("Sentiment")
            news_display = news_df[display_cols].rename(columns=col_renames)

            def _sent_color(val: str) -> str:
                if val == "BUY":   return "color: #22c55e"
                if val == "SELL":  return "color: #ef4444"
                return "color: #94a3b8"

            styled = news_display.style
            if "Sentiment" in news_display.columns:
                styled = styled.applymap(_sent_color, subset=["Sentiment"])

            st.dataframe(styled, use_container_width=True, hide_index=True, height=600)


if __name__ == "__main__":
    main()
