"""
Unit tests for data/alpaca_client.py.

All Alpaca SDK calls are mocked — no live API access.
Run with:
    .venv/bin/pytest tests/test_alpaca_client.py -v
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Test-only exception that mimics alpaca APIError's duck-typed interface
# ---------------------------------------------------------------------------

class _FakeAPIError(Exception):
    """Lightweight stand-in for alpaca.common.exceptions.APIError in tests."""
    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for Alpaca model objects
# ---------------------------------------------------------------------------

def _make_bar(
    symbol: str,
    ts: datetime,
    o: float = 100.0,
    h: float = 102.0,
    lo: float = 99.0,
    c: float = 101.0,
    vol: int = 500_000,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        timestamp=ts,
        open=o,
        high=h,
        low=lo,
        close=c,
        volume=vol,
    )


def _make_quote(
    symbol: str,
    bid: float = 150.0,
    ask: float = 150.1,
    ts: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        bid_price=bid,
        ask_price=ask,
        timestamp=ts or datetime.now(tz=timezone.utc),
    )


def _make_account(
    equity: str = "100000.00",
    cash: str = "80000.00",
    buying_power: str = "160000.00",
    portfolio_value: str = "100000.00",
) -> SimpleNamespace:
    return SimpleNamespace(
        equity=equity,
        cash=cash,
        buying_power=buying_power,
        portfolio_value=portfolio_value,
    )


def _make_news_article(
    headline: str,
    symbols: list[str],
    source: str = "Reuters",
    summary: str = "A summary.",
    created_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        headline=headline,
        symbols=symbols,
        source=source,
        summary=summary,
        created_at=created_at or datetime.now(tz=timezone.utc),
    )


_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
_START = datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
_END   = datetime(2025, 1, 15, 16, 0, 0, tzinfo=timezone.utc)

# Patch targets (module-level so they're easy to update if paths change)
_MOD = "trading_engine.data.alpaca_client"
_HIST_CLIENT  = f"{_MOD}.StockHistoricalDataClient"
_STREAM_CLASS = f"{_MOD}.StockDataStream"
_TRADING_CLIENT = f"{_MOD}.TradingClient"
_NEWS_CLIENT  = f"{_MOD}.NewsClient"
_SETTINGS     = f"{_MOD}.settings"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_storage() -> MagicMock:
    storage = MagicMock()
    storage.insert_ohlcv.return_value = 0
    return storage


@pytest.fixture()
def mock_settings(monkeypatch):
    """Inject fake credentials so settings import never fails."""
    fake = SimpleNamespace(
        ALPACA_API_KEY="fake-key",
        ALPACA_SECRET_KEY="fake-secret",
    )
    monkeypatch.setattr(_MOD + ".settings", fake)
    return fake


# ---------------------------------------------------------------------------
# AlpacaMarketData — fetch_historical_ohlcv
# ---------------------------------------------------------------------------

class TestFetchHistoricalOHLCV:

    def _make_bar_set(self, ticker: str, bars: list) -> MagicMock:
        bar_set = MagicMock()
        bar_set.data = {ticker: bars}
        return bar_set

    def test_returns_dataframe_with_correct_columns(self, mock_storage, mock_settings):
        bars = [_make_bar("AAPL", _START), _make_bar("AAPL", _END)]
        bar_set = self._make_bar_set("AAPL", bars)

        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            MockHist.return_value.get_stock_bars.return_value = bar_set
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            df = client.fetch_historical_ohlcv(["AAPL"], _START, _END)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"ticker", "open", "high", "low", "close", "volume"}
        assert df.index.name == "time"
        assert len(df) == 2

    def test_inserts_into_storage(self, mock_storage, mock_settings):
        bars = [_make_bar("AAPL", _START)]
        bar_set = self._make_bar_set("AAPL", bars)

        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            MockHist.return_value.get_stock_bars.return_value = bar_set
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            client.fetch_historical_ohlcv(["AAPL"], _START, _END)

        mock_storage.insert_ohlcv.assert_called_once()
        records = mock_storage.insert_ohlcv.call_args[0][0]
        assert len(records) == 1
        assert records[0]["ticker"] == "AAPL"
        assert records[0]["open"] == 100.0
        assert records[0]["volume"] == 500_000

    def test_multi_ticker(self, mock_storage, mock_settings):
        bar_set = MagicMock()
        bar_set.data = {
            "AAPL": [_make_bar("AAPL", _START)],
            "MSFT": [_make_bar("MSFT", _START, o=200.0)],
        }

        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            MockHist.return_value.get_stock_bars.return_value = bar_set
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            df = client.fetch_historical_ohlcv(["AAPL", "MSFT"], _START, _END)

        assert len(df) == 2
        tickers_returned = set(df["ticker"].tolist())
        assert tickers_returned == {"AAPL", "MSFT"}

    def test_empty_response_returns_empty_dataframe(self, mock_storage, mock_settings):
        bar_set = MagicMock()
        bar_set.data = {}

        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            MockHist.return_value.get_stock_bars.return_value = bar_set
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            df = client.fetch_historical_ohlcv(["AAPL"], _START, _END)

        assert df.empty
        mock_storage.insert_ohlcv.assert_not_called()

    def test_invalid_timeframe_raises(self, mock_storage, mock_settings):
        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            with pytest.raises(ValueError, match="Unsupported timeframe"):
                client.fetch_historical_ohlcv(["AAPL"], _START, _END, timeframe="2Day")

    def test_df_index_is_utc_datetime(self, mock_storage, mock_settings):
        bar_set = MagicMock()
        bar_set.data = {"AAPL": [_make_bar("AAPL", _START)]}

        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            MockHist.return_value.get_stock_bars.return_value = bar_set
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            df = client.fetch_historical_ohlcv(["AAPL"], _START, _END)

        assert str(df.index.tz) == "UTC"


# ---------------------------------------------------------------------------
# AlpacaMarketData — get_latest_quote
# ---------------------------------------------------------------------------

class TestGetLatestQuote:

    def test_returns_bid_ask_mid_timestamp(self, mock_storage, mock_settings):
        quote = _make_quote("AAPL", bid=149.90, ask=150.10)

        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            MockHist.return_value.get_stock_latest_quote.return_value = {"AAPL": quote}
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            result = client.get_latest_quote("AAPL")

        assert result["bid"] == 149.90
        assert result["ask"] == 150.10
        assert result["mid"] == pytest.approx(150.0, abs=1e-4)
        assert "timestamp" in result

    def test_mid_is_average(self, mock_storage, mock_settings):
        quote = _make_quote("TSLA", bid=200.0, ask=200.4)

        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            MockHist.return_value.get_stock_latest_quote.return_value = {"TSLA": quote}
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            result = client.get_latest_quote("TSLA")

        assert result["mid"] == pytest.approx(200.2, abs=1e-4)


# ---------------------------------------------------------------------------
# AlpacaMarketData — get_account_info
# ---------------------------------------------------------------------------

class TestGetAccountInfo:

    def test_returns_float_fields(self, mock_storage, mock_settings):
        account = _make_account(
            equity="105000.50",
            cash="82000.00",
            buying_power="164000.00",
            portfolio_value="105000.50",
        )

        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT) as MockTrading,
        ):
            MockTrading.return_value.get_account.return_value = account
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            info = client.get_account_info()

        assert info["equity"] == pytest.approx(105000.50)
        assert info["cash"] == pytest.approx(82000.00)
        assert info["buying_power"] == pytest.approx(164000.00)
        assert info["portfolio_value"] == pytest.approx(105000.50)
        assert all(isinstance(v, float) for v in info.values())


# ---------------------------------------------------------------------------
# AlpacaMarketData — stream_bars
# ---------------------------------------------------------------------------

class TestStreamBars:

    def test_subscribes_and_starts_thread(self, mock_storage, mock_settings):
        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS) as MockStream,
            patch(_TRADING_CLIENT),
            patch("threading.Thread") as MockThread,
        ):
            mock_thread_instance = MagicMock()
            MockThread.return_value = mock_thread_instance

            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            callback = MagicMock()
            client.stream_bars(["AAPL", "MSFT"], callback)

            MockStream.return_value.subscribe_bars.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    def test_stop_stream(self, mock_storage, mock_settings):
        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS) as MockStream,
            patch(_TRADING_CLIENT),
        ):
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            client.stop_stream()
            MockStream.return_value.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetry:

    def test_retries_on_rate_limit_then_succeeds(self, mock_storage, mock_settings):
        bar_set = MagicMock()
        bar_set.data = {"AAPL": [_make_bar("AAPL", _START)]}

        call_count = 0

        def _flaky(_):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise _FakeAPIError(429, "rate limit")
            return bar_set

        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
            patch(f"{_MOD}.time.sleep"),
        ):
            MockHist.return_value.get_stock_bars.side_effect = _flaky
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            df = client.fetch_historical_ohlcv(["AAPL"], _START, _END)

        assert call_count == 2
        assert len(df) == 1

    def test_raises_immediately_on_auth_error(self, mock_storage, mock_settings):
        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
        ):
            MockHist.return_value.get_stock_bars.side_effect = _FakeAPIError(403, "forbidden")
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            with pytest.raises(_FakeAPIError):
                client.fetch_historical_ohlcv(["AAPL"], _START, _END)

            # Called exactly once — no retries on auth errors
            assert MockHist.return_value.get_stock_bars.call_count == 1

    def test_exhausts_retries_and_raises(self, mock_storage, mock_settings):
        with (
            patch(_HIST_CLIENT) as MockHist,
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT),
            patch(f"{_MOD}.time.sleep"),
        ):
            MockHist.return_value.get_stock_bars.side_effect = _FakeAPIError(429, "rate limit")
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            with pytest.raises(_FakeAPIError):
                client.fetch_historical_ohlcv(["AAPL"], _START, _END)

            # 1 initial + 3 retries = 4 total calls
            assert MockHist.return_value.get_stock_bars.call_count == 4


# ---------------------------------------------------------------------------
# AlpacaMarketData — is_market_open / get_market_clock
# ---------------------------------------------------------------------------

class TestIsMarketOpen:

    def _make_clock(self, is_open: bool) -> SimpleNamespace:
        return SimpleNamespace(
            is_open=is_open,
            next_open=_NOW,
            next_close=_NOW,
            timestamp=_NOW,
        )

    def test_returns_true_when_market_open(self, mock_storage, mock_settings):
        clock = self._make_clock(is_open=True)

        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT) as MockTrading,
        ):
            MockTrading.return_value.get_clock.return_value = clock
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            result = client.is_market_open()

        assert result is True

    def test_returns_false_when_market_closed(self, mock_storage, mock_settings):
        clock = self._make_clock(is_open=False)

        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT) as MockTrading,
        ):
            MockTrading.return_value.get_clock.return_value = clock
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            result = client.is_market_open()

        assert result is False

    def test_cache_prevents_duplicate_api_calls(self, mock_storage, mock_settings):
        """Two calls within 60 seconds must only hit get_clock() once."""
        clock = self._make_clock(is_open=True)

        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT) as MockTrading,
            patch(f"{_MOD}.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 0.0
            MockTrading.return_value.get_clock.return_value = clock
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)

            client.is_market_open()
            assert MockTrading.return_value.get_clock.call_count == 1

            # Second call within the 60-second window
            mock_time.monotonic.return_value = 30.0
            client.is_market_open()
            assert MockTrading.return_value.get_clock.call_count == 1  # still 1

    def test_cache_expires_after_60_seconds(self, mock_storage, mock_settings):
        """A call more than 60 seconds after the cached result must fetch fresh data."""
        clock = self._make_clock(is_open=True)

        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT) as MockTrading,
            patch(f"{_MOD}.time") as mock_time,
        ):
            mock_time.monotonic.return_value = 0.0
            MockTrading.return_value.get_clock.return_value = clock
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)

            client.is_market_open()
            assert MockTrading.return_value.get_clock.call_count == 1

            # Advance past the 60-second expiry
            mock_time.monotonic.return_value = 61.0
            client.is_market_open()
            assert MockTrading.return_value.get_clock.call_count == 2


class TestGetMarketClock:

    def test_returns_all_required_keys(self, mock_storage, mock_settings):
        clock = SimpleNamespace(
            is_open=True,
            next_open=_NOW,
            next_close=_END,
            timestamp=_START,
        )

        with (
            patch(_HIST_CLIENT),
            patch(_STREAM_CLASS),
            patch(_TRADING_CLIENT) as MockTrading,
        ):
            MockTrading.return_value.get_clock.return_value = clock
            from trading_engine.data.alpaca_client import AlpacaMarketData
            client = AlpacaMarketData(mock_storage)
            result = client.get_market_clock()

        assert set(result.keys()) == {"is_open", "next_open", "next_close", "timestamp"}
        assert result["is_open"] is True
        assert result["next_open"] == _NOW
        assert result["next_close"] == _END
        assert result["timestamp"] == _START


# ---------------------------------------------------------------------------
# AlpacaNewsClient — fetch_news
# ---------------------------------------------------------------------------

class TestAlpacaNewsClient:

    def _make_news_set(self, articles: list) -> MagicMock:
        ns = MagicMock()
        ns.data = {"news": articles}
        return ns

    def test_returns_list_of_dicts(self, mock_settings):
        articles = [
            _make_news_article("Apple hits all-time high", ["AAPL"]),
            _make_news_article("Markets rally broadly",   ["AAPL", "MSFT"]),
        ]

        with patch(_NEWS_CLIENT) as MockNews:
            MockNews.return_value.get_news.return_value = self._make_news_set(articles)
            from trading_engine.data.alpaca_client import AlpacaNewsClient
            client = AlpacaNewsClient()
            rows = client.fetch_news(["AAPL", "MSFT"], hours_back=8)

        # "Markets rally broadly" matches both AAPL and MSFT → 2 rows
        assert len(rows) == 3
        keys = {"ticker", "title", "summary", "source", "published_at", "headline_hash"}
        for row in rows:
            assert keys.issubset(row.keys())

    def test_skips_articles_not_matching_tickers(self, mock_settings):
        articles = [
            _make_news_article("TSLA soars", ["TSLA"]),
        ]

        with patch(_NEWS_CLIENT) as MockNews:
            MockNews.return_value.get_news.return_value = self._make_news_set(articles)
            from trading_engine.data.alpaca_client import AlpacaNewsClient
            client = AlpacaNewsClient()
            rows = client.fetch_news(["AAPL"], hours_back=8)

        assert rows == []

    def test_deduplication_within_batch(self, mock_settings):
        # Same headline appears twice for the same ticker
        articles = [
            _make_news_article("Duplicate headline", ["AAPL"]),
            _make_news_article("Duplicate headline", ["AAPL"]),
        ]

        with patch(_NEWS_CLIENT) as MockNews:
            MockNews.return_value.get_news.return_value = self._make_news_set(articles)
            from trading_engine.data.alpaca_client import AlpacaNewsClient
            client = AlpacaNewsClient()
            rows = client.fetch_news(["AAPL"])

        assert len(rows) == 1

    def test_headline_hash_is_sha256_of_title(self, mock_settings):
        headline = "Apple earnings beat expectations"
        articles = [_make_news_article(headline, ["AAPL"])]

        with patch(_NEWS_CLIENT) as MockNews:
            MockNews.return_value.get_news.return_value = self._make_news_set(articles)
            from trading_engine.data.alpaca_client import AlpacaNewsClient
            client = AlpacaNewsClient()
            rows = client.fetch_news(["AAPL"])

        expected_hash = hashlib.sha256(headline.encode("utf-8")).hexdigest()
        assert rows[0]["headline_hash"] == expected_hash

    def test_does_not_call_storage(self, mock_settings):
        """News client must NOT write to DB — that is the sentiment module's job."""
        articles = [_make_news_article("Headline", ["AAPL"])]
        mock_storage = MagicMock()

        with patch(_NEWS_CLIENT) as MockNews:
            MockNews.return_value.get_news.return_value = self._make_news_set(articles)
            from trading_engine.data.alpaca_client import AlpacaNewsClient
            client = AlpacaNewsClient()
            client.fetch_news(["AAPL"])

        mock_storage.insert_news.assert_not_called()
        mock_storage.insert_ohlcv.assert_not_called()

    def test_none_summary_becomes_none(self, mock_settings):
        article = _make_news_article("Headline", ["AAPL"], summary="")

        with patch(_NEWS_CLIENT) as MockNews:
            MockNews.return_value.get_news.return_value = self._make_news_set([article])
            from trading_engine.data.alpaca_client import AlpacaNewsClient
            client = AlpacaNewsClient()
            rows = client.fetch_news(["AAPL"])

        # Empty string → None
        assert rows[0]["summary"] is None
