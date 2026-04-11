"""
Compute forward-return outcome labels for each trade_log decision.

Joins trade_log with ohlcv to find the closing price at four forward horizons
(+1 m, +15 m, +1 h, +4 h) after each decision, then derives directional
correctness labels.

No live engine or API calls — reads only from the DB.

Public API
----------
load_labeled_decisions(db_url, ticker=None, days=None) -> pd.DataFrame
    Full pipeline: DB query + outcome label computation.

compute_outcome_labels(df) -> pd.DataFrame
    Pure-DataFrame transform, useful for testing with synthetic data.
"""

from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# SQL — correlated subqueries are fine for offline analysis (not hot path)
# ---------------------------------------------------------------------------

_QUERY = """
SELECT
    t.id,
    t.time,
    t.ticker,
    t.final_signal,
    t.score,
    t.regime,
    t.regime_label,
    t.hmm_signal,
    t.hmm_confidence,
    t.ou_signal,
    t.ou_confidence,
    t.ou_zscore,
    t.ou_spread_value,
    t.ou_pair,
    t.llm_signal,
    t.llm_confidence,
    t.mwu_weights,
    t.contributing_headlines,
    (SELECT o.close FROM ohlcv o
       WHERE o.ticker = t.ticker
         AND o.time   >= t.time
       ORDER BY o.time ASC LIMIT 1)                              AS close_at,
    (SELECT o.close FROM ohlcv o
       WHERE o.ticker = t.ticker
         AND o.time   >= t.time + INTERVAL '1 minute'
       ORDER BY o.time ASC LIMIT 1)                              AS close_1m,
    (SELECT o.close FROM ohlcv o
       WHERE o.ticker = t.ticker
         AND o.time   >= t.time + INTERVAL '15 minutes'
       ORDER BY o.time ASC LIMIT 1)                              AS close_15m,
    (SELECT o.close FROM ohlcv o
       WHERE o.ticker = t.ticker
         AND o.time   >= t.time + INTERVAL '1 hour'
       ORDER BY o.time ASC LIMIT 1)                              AS close_1h,
    (SELECT o.close FROM ohlcv o
       WHERE o.ticker = t.ticker
         AND o.time   >= t.time + INTERVAL '4 hours'
       ORDER BY o.time ASC LIMIT 1)                              AS close_4h
FROM trade_log t
WHERE (:ticker_filter IS NULL OR t.ticker = :ticker_filter)
  AND (:days_filter   IS NULL OR t.time >= NOW() - (:days_filter * INTERVAL '1 day'))
ORDER BY t.time ASC
"""

_HORIZONS = ("1m", "15m", "1h", "4h")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_labeled_decisions(
    db_url: str,
    ticker: str | None = None,
    days: int | None = None,
) -> pd.DataFrame:
    """
    Load trade_log decisions from the DB and attach forward-return labels.

    Rows with ``final_signal == 0`` are included (they are useful for the
    min_confidence sweep).  Callers that want only directional decisions can
    filter with ``df[df['final_signal'] != 0]``.

    Parameters
    ----------
    db_url : str
        SQLAlchemy-compatible PostgreSQL URL.
    ticker : str or None
        Restrict to a single ticker, or ``None`` for all tickers.
    days : int or None
        Restrict to the most recent N calendar days, or ``None`` for all
        history.

    Returns
    -------
    pd.DataFrame
        See :func:`compute_outcome_labels` for the full column list.
    """
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(_QUERY),
                {"ticker_filter": ticker, "days_filter": days},
            )
            raw = pd.DataFrame(result.fetchall(), columns=result.keys())
    finally:
        engine.dispose()

    if raw.empty:
        return raw

    # Parse JSONB columns returned as strings by psycopg2
    for col in ("mwu_weights", "contributing_headlines", "regime_probs"):
        if col in raw.columns:
            raw[col] = raw[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v
            )

    return compute_outcome_labels(raw)


def compute_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive forward-return and correctness columns from a DataFrame that
    already contains ``close_at``, ``close_1m``, ``close_15m``, ``close_1h``,
    ``close_4h``, ``final_signal``.

    This is the pure-DataFrame transform, separated from the DB query so it
    can be exercised in unit tests with synthetic data.

    Added columns
    -------------
    fwd_ret_1m, fwd_ret_15m, fwd_ret_1h, fwd_ret_4h : float
        ``(close_Xm - close_at) / close_at``
    correct_1m, correct_15m, correct_1h, correct_4h : bool or NaN
        ``True`` when ``final_signal * fwd_ret > 0``.
        ``NaN`` when ``final_signal == 0`` (neutral decision — no bet placed)
        or when the forward close is missing (e.g. market closed).

    Returns
    -------
    pd.DataFrame
        Input df with the eight new columns appended.
    """
    df = df.copy()

    for h in _HORIZONS:
        close_col = f"close_{h}"
        ret_col   = f"fwd_ret_{h}"
        corr_col  = f"correct_{h}"

        if close_col in df.columns and "close_at" in df.columns:
            df[ret_col] = (df[close_col] - df["close_at"]) / df["close_at"]
        else:
            df[ret_col] = float("nan")

        # correct_* is meaningful only for directional decisions that have a
        # valid forward return.  NaN forward return → NaN correctness.
        directional  = df["final_signal"] != 0
        has_ret      = df[ret_col].notna()
        signed       = df["final_signal"] * df[ret_col]
        df[corr_col] = (signed > 0).where(directional & has_ret)

    return df
