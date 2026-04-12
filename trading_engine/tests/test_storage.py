"""
Integration tests for data/storage.py.

Requirements:
  - A running PostgreSQL/TimescaleDB instance reachable via TEST_DB_URL.
  - If TEST_DB_URL is not set the tests are skipped automatically.

Run with:
    TEST_DB_URL="postgresql+psycopg2://trader:traderpass@localhost:5432/trading_test" \
        pytest trading_engine/tests/test_storage.py -v
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DB_URL = os.getenv("TEST_DB_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DB_URL not set — skipping storage integration tests",
)


@pytest.fixture(scope="module")
def storage():
    """Bootstrap a Storage instance against the test database."""
    from trading_engine.data.storage import Storage
    from trading_engine.utils.logging import configure_logging

    configure_logging("DEBUG")
    s = Storage(TEST_DB_URL)  # type: ignore[arg-type]
    yield s
    # Teardown: drop test data so the suite is idempotent
    with s._engine.begin() as conn:
        conn.execute(__import__("sqlalchemy").text("DELETE FROM ohlcv WHERE ticker = 'TEST'"))
        conn.execute(__import__("sqlalchemy").text("DELETE FROM news WHERE ticker = 'TEST'"))
        conn.execute(__import__("sqlalchemy").text("DELETE FROM signal_log WHERE ticker = 'TEST'"))
        conn.execute(__import__("sqlalchemy").text("DELETE FROM regime_log WHERE ticker = 'TEST'"))
    s.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(tz=timezone.utc).replace(microsecond=0)


def _ohlcv_row(offset_minutes: int = 0) -> dict:
    t = _NOW - timedelta(minutes=offset_minutes)
    return {
        "time": t,
        "ticker": "TEST",
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 103.0,
        "volume": 1_000_000,
    }


# ---------------------------------------------------------------------------
# OHLCV tests
# ---------------------------------------------------------------------------

class TestOHLCV:
    def test_insert_returns_count(self, storage) -> None:
        rows = [_ohlcv_row(i) for i in range(3)]
        inserted = storage.insert_ohlcv(rows)
        assert inserted == 3

    def test_insert_empty_noop(self, storage) -> None:
        assert storage.insert_ohlcv([]) == 0

    def test_insert_duplicate_returns_reduced_count(self, storage) -> None:
        # Use offsets not used by any other test in this class to avoid
        # session-scoped DB state collisions
        rows = [_ohlcv_row(i) for i in range(30, 33)]
        first = storage.insert_ohlcv(rows)
        assert first == 3
        # Re-submitting the same rows must return 0 — all are duplicates
        second = storage.insert_ohlcv(rows)
        assert second == 0

    def test_query_returns_dataframe(self, storage) -> None:
        storage.insert_ohlcv([_ohlcv_row(10)])
        start = _NOW - timedelta(hours=1)
        end = _NOW + timedelta(hours=1)
        df = storage.query_ohlcv("TEST", start, end)
        assert not df.empty
        assert set(df.columns) >= {"time", "ticker", "open", "high", "low", "close", "volume"}

    def test_query_filters_ticker(self, storage) -> None:
        df = storage.query_ohlcv("NONEXISTENT", _NOW - timedelta(hours=1), _NOW + timedelta(hours=1))
        assert df.empty

    def test_query_time_range(self, storage) -> None:
        # Insert a row far in the past
        old_row = _ohlcv_row()
        old_row["time"] = _NOW - timedelta(days=10)
        storage.insert_ohlcv([old_row])

        # Query only recent window — should not include the old row
        df = storage.query_ohlcv("TEST", _NOW - timedelta(hours=1), _NOW + timedelta(hours=1))
        assert all(r >= _NOW - timedelta(hours=1) for r in df["time"])


# ---------------------------------------------------------------------------
# News tests
# ---------------------------------------------------------------------------

class TestNews:
    def test_insert_news(self, storage) -> None:
        rows = [
            {
                "ticker": "TEST",
                "title": "Test headline alpha",
                "summary": "Some summary",
                "source": "Reuters",
                "sentiment_score": 0.7,
                "sentiment_confidence": 0.9,
                "llm_direction": 1,
            },
            {
                "ticker": "TEST",
                "title": "Test headline beta",
            },
        ]
        inserted = storage.insert_news(rows)
        assert inserted == 2

    def test_insert_news_deduplication(self, storage) -> None:
        """Same (ticker, headline_hash) inserted twice → second insert is skipped."""
        row = {"ticker": "TEST", "title": "Duplicate headline gamma"}
        storage.insert_news([row])
        second = storage.insert_news([row])
        assert second == 0

    def test_insert_news_same_hash_different_tickers(self, storage) -> None:
        """Same headline_hash for two different tickers must both be accepted."""
        title = "Shared macro headline for dedup test"
        row_aapl = {"ticker": "TEST",  "title": title}
        row_msft = {"ticker": "TEST2", "title": title}
        n1 = storage.insert_news([row_aapl])
        n2 = storage.insert_news([row_msft])
        assert n1 == 1, "First ticker insert should succeed"
        assert n2 == 1, "Second ticker with same hash should also succeed"

    def test_insert_news_same_ticker_same_hash_is_deduped(self, storage) -> None:
        """True duplicate (same ticker + same hash) is rejected on second insert."""
        row = {"ticker": "TEST", "title": "Ticker-scoped dedup check"}
        assert storage.insert_news([row]) == 1
        assert storage.insert_news([row]) == 0

    def test_insert_news_empty(self, storage) -> None:
        assert storage.insert_news([]) == 0

    def test_query_news_returns_dataframe(self, storage) -> None:
        storage.insert_news([{"ticker": "TEST", "title": "Query test headline delta"}])
        df = storage.query_news("TEST", hours_back=1.0)
        assert not df.empty
        assert "title" in df.columns

    def test_query_news_no_results(self, storage) -> None:
        df = storage.query_news("NONEXISTENT", hours_back=1.0)
        assert df.empty


# ---------------------------------------------------------------------------
# Signal log tests
# ---------------------------------------------------------------------------

class TestSignalLog:
    def test_insert_signal(self, storage) -> None:
        rows = [
            {
                "time": _NOW,
                "ticker": "TEST",
                "signal_name": "rsi_14",
                "value": 65.3,
                "metadata": {"window": 14, "source": "ta-lib"},
            },
            {
                "time": _NOW,
                "ticker": "TEST",
                "signal_name": "macd_histogram",
                "value": 0.42,
            },
        ]
        assert storage.insert_signal(rows) == 2

    def test_insert_signal_empty(self, storage) -> None:
        assert storage.insert_signal([]) == 0


# ---------------------------------------------------------------------------
# Regime log tests
# ---------------------------------------------------------------------------

class TestRegimeLog:
    def test_insert_regime(self, storage) -> None:
        rows = [
            {
                "time": _NOW,
                "ticker": "TEST",
                "regime": 1,
                "regime_probs": {0: 0.1, 1: 0.8, 2: 0.1},
            },
            {
                "time": _NOW - timedelta(minutes=1),
                "ticker": "TEST",
                "regime": 0,
            },
        ]
        assert storage.insert_regime(rows) == 2

    def test_insert_regime_empty(self, storage) -> None:
        assert storage.insert_regime([]) == 0
