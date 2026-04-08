"""TimescaleDB storage layer — connection pool, schema bootstrap, and CRUD helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ     NOT NULL,
    ticker      TEXT            NOT NULL,
    open        FLOAT           NOT NULL,
    high        FLOAT           NOT NULL,
    low         FLOAT           NOT NULL,
    close       FLOAT           NOT NULL,
    volume      BIGINT          NOT NULL
);
"""

_DDL_OHLCV_HYPERTABLE = """
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);
"""

_DDL_OHLCV_INDEX = """
CREATE INDEX IF NOT EXISTS ohlcv_ticker_time_idx ON ohlcv (ticker, time DESC);
"""

_DDL_NEWS = """
CREATE TABLE IF NOT EXISTS news (
    id                  SERIAL          PRIMARY KEY,
    fetched_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    ticker              TEXT            NOT NULL,
    title               TEXT            NOT NULL,
    summary             TEXT,
    source              TEXT,
    sentiment_score     FLOAT,
    sentiment_confidence FLOAT,
    llm_direction       INT,
    headline_hash       TEXT            UNIQUE NOT NULL
);
"""

_DDL_SIGNAL_LOG = """
CREATE TABLE IF NOT EXISTS signal_log (
    time        TIMESTAMPTZ     NOT NULL,
    ticker      TEXT            NOT NULL,
    signal_name TEXT            NOT NULL,
    value       FLOAT           NOT NULL,
    metadata    JSONB
);
"""

_DDL_SIGNAL_LOG_HYPERTABLE = """
SELECT create_hypertable('signal_log', 'time', if_not_exists => TRUE);
"""

_DDL_REGIME_LOG = """
CREATE TABLE IF NOT EXISTS regime_log (
    time            TIMESTAMPTZ     NOT NULL,
    ticker          TEXT            NOT NULL,
    regime          INT             NOT NULL,
    regime_probs    JSONB
);
"""

_DDL_REGIME_LOG_HYPERTABLE = """
SELECT create_hypertable('regime_log', 'time', if_not_exists => TRUE);
"""


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _build_engine(db_url: str) -> Engine:
    return create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Storage:
    """
    Manages the TimescaleDB connection pool and all persistence operations.

    Parameters
    ----------
    db_url:
        SQLAlchemy-compatible PostgreSQL URL, e.g.
        ``postgresql+psycopg2://user:pass@localhost:5432/trading``.
    """

    def __init__(self, db_url: str) -> None:
        self._engine = _build_engine(db_url)
        logger.info("storage.init", db_url=db_url.split("@")[-1])  # hide credentials
        self._bootstrap_schema()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_schema(self) -> None:
        logger.info("storage.bootstrap", status="start")
        with self._engine.begin() as conn:
            conn.execute(text(_DDL_OHLCV))
            conn.execute(text(_DDL_OHLCV_HYPERTABLE))
            conn.execute(text(_DDL_OHLCV_INDEX))
            conn.execute(text(_DDL_NEWS))
            conn.execute(text(_DDL_SIGNAL_LOG))
            conn.execute(text(_DDL_SIGNAL_LOG_HYPERTABLE))
            conn.execute(text(_DDL_REGIME_LOG))
            conn.execute(text(_DDL_REGIME_LOG_HYPERTABLE))
        logger.info("storage.bootstrap", status="done")

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def insert_ohlcv(self, rows: list[dict[str, Any]]) -> int:
        """
        Insert one or more OHLCV rows.

        Parameters
        ----------
        rows:
            Each dict must contain keys: time, ticker, open, high, low,
            close, volume.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not rows:
            return 0
        stmt = text(
            """
            INSERT INTO ohlcv (time, ticker, open, high, low, close, volume)
            VALUES (:time, :ticker, :open, :high, :low, :close, :volume)
            ON CONFLICT DO NOTHING
            """
        )
        with self._engine.begin() as conn:
            conn.execute(stmt, rows)
        logger.info("storage.insert_ohlcv", count=len(rows))
        return len(rows)

    def query_ohlcv(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV rows for *ticker* between *start* and *end* (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: time, ticker, open, high, low, close, volume.
            Sorted ascending by time.
        """
        stmt = text(
            """
            SELECT time, ticker, open, high, low, close, volume
            FROM   ohlcv
            WHERE  ticker = :ticker
              AND  time BETWEEN :start AND :end
            ORDER  BY time ASC
            """
        )
        with self._engine.connect() as conn:
            result = conn.execute(stmt, {"ticker": ticker, "start": start, "end": end})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        logger.info(
            "storage.query_ohlcv",
            ticker=ticker,
            start=str(start),
            end=str(end),
            rows=len(df),
        )
        return df

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def insert_news(self, rows: list[dict[str, Any]]) -> int:
        """
        Insert news items, deduplicating by headline hash.

        Parameters
        ----------
        rows:
            Each dict may contain: ticker, title, summary, source,
            sentiment_score, sentiment_confidence, llm_direction,
            fetched_at (optional — defaults to now).
            ``headline_hash`` is computed automatically from *title*.

        Returns
        -------
        int
            Number of rows actually inserted (skips duplicates).
        """
        if not rows:
            return 0
        # RETURNING id lets us count rows actually inserted vs. skipped by
        # ON CONFLICT DO NOTHING — executemany rowcount is unreliable for this.
        stmt = text(
            """
            INSERT INTO news
                (fetched_at, ticker, title, summary, source,
                 sentiment_score, sentiment_confidence, llm_direction,
                 headline_hash)
            VALUES
                (:fetched_at, :ticker, :title, :summary, :source,
                 :sentiment_score, :sentiment_confidence, :llm_direction,
                 :headline_hash)
            ON CONFLICT (headline_hash) DO NOTHING
            RETURNING id
            """
        )
        now = datetime.now(tz=timezone.utc)
        enriched: list[dict[str, Any]] = []
        for row in rows:
            r = dict(row)
            r.setdefault("fetched_at", now)
            r.setdefault("summary", None)
            r.setdefault("source", None)
            r.setdefault("sentiment_score", None)
            r.setdefault("sentiment_confidence", None)
            r.setdefault("llm_direction", None)
            r["headline_hash"] = hashlib.sha256(
                r["title"].encode("utf-8")
            ).hexdigest()
            enriched.append(r)

        inserted = 0
        with self._engine.begin() as conn:
            for row in enriched:
                result = conn.execute(stmt, row)
                inserted += len(result.fetchall())

        logger.info("storage.insert_news", submitted=len(rows), inserted=inserted)
        return inserted

    def query_news(self, ticker: str, hours_back: float = 24.0) -> pd.DataFrame:
        """
        Retrieve news for *ticker* fetched within the last *hours_back* hours.

        Returns
        -------
        pd.DataFrame
            Sorted descending by fetched_at.
        """
        stmt = text(
            """
            SELECT id, fetched_at, ticker, title, summary, source,
                   sentiment_score, sentiment_confidence, llm_direction,
                   headline_hash
            FROM   news
            WHERE  ticker     = :ticker
              AND  fetched_at >= NOW() - (:hours * INTERVAL '1 hour')
            ORDER  BY fetched_at DESC
            """
        )
        with self._engine.connect() as conn:
            result = conn.execute(stmt, {"ticker": ticker, "hours": hours_back})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        logger.info(
            "storage.query_news", ticker=ticker, hours_back=hours_back, rows=len(df)
        )
        return df

    # ------------------------------------------------------------------
    # Signal log
    # ------------------------------------------------------------------

    def insert_signal(self, rows: list[dict[str, Any]]) -> int:
        """
        Append signal observations.

        Parameters
        ----------
        rows:
            Each dict must contain: time, ticker, signal_name, value.
            ``metadata`` (dict) is optional.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not rows:
            return 0
        stmt = text(
            """
            INSERT INTO signal_log (time, ticker, signal_name, value, metadata)
            VALUES (:time, :ticker, :signal_name, :value, :metadata)
            """
        )
        prepared = []
        for row in rows:
            r = dict(row)
            meta = r.get("metadata")
            r["metadata"] = json.dumps(meta) if meta is not None else None
            prepared.append(r)

        with self._engine.begin() as conn:
            conn.execute(stmt, prepared)
        logger.info("storage.insert_signal", count=len(rows))
        return len(rows)

    # ------------------------------------------------------------------
    # Regime log
    # ------------------------------------------------------------------

    def insert_regime(self, rows: list[dict[str, Any]]) -> int:
        """
        Append regime observations.

        Parameters
        ----------
        rows:
            Each dict must contain: time, ticker, regime.
            ``regime_probs`` (dict) is optional.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not rows:
            return 0
        stmt = text(
            """
            INSERT INTO regime_log (time, ticker, regime, regime_probs)
            VALUES (:time, :ticker, :regime, :regime_probs)
            """
        )
        prepared = []
        for row in rows:
            r = dict(row)
            probs = r.get("regime_probs")
            r["regime_probs"] = json.dumps(probs) if probs is not None else None
            prepared.append(r)

        with self._engine.begin() as conn:
            conn.execute(stmt, prepared)
        logger.info("storage.insert_regime", count=len(rows))
        return len(rows)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Release all connections back to the pool and shut it down."""
        self._engine.dispose()
        logger.info("storage.dispose")
