"""
Unit tests for data/fundamentals_client.py.

All yfinance network calls are mocked — no live connections required.
Run with:
    .venv/bin/pytest tests/test_fundamentals_client.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_MOD = "trading_engine.data.fundamentals_client"


def _make_client():
    from trading_engine.data.fundamentals_client import FundamentalsClient
    return FundamentalsClient()


def _mock_yf_ticker(market_cap: float | None):
    """Return a yf.Ticker mock whose .info['marketCap'] == market_cap."""
    m = MagicMock()
    m.info = {"marketCap": market_cap}
    return m


# ===========================================================================
# get_market_caps — basic fetch
# ===========================================================================

class TestGetMarketCaps:

    def test_returns_cap_for_single_ticker(self):
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker(3e12)):
            client = _make_client()
            caps = client.get_market_caps(["AAPL"])
        assert caps["AAPL"] == pytest.approx(3e12)

    def test_returns_multiple_tickers(self):
        def _side(ticker):
            return _mock_yf_ticker({"AAPL": 3e12, "MSFT": 2e12}[ticker])

        with patch(f"{_MOD}.yf.Ticker", side_effect=_side):
            client = _make_client()
            caps = client.get_market_caps(["AAPL", "MSFT"])
        assert caps["AAPL"] == pytest.approx(3e12)
        assert caps["MSFT"] == pytest.approx(2e12)

    def test_missing_market_cap_returns_zero(self):
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker(None)):
            client = _make_client()
            caps = client.get_market_caps(["UNKN"])
        assert caps["UNKN"] == 0.0

    def test_exception_returns_zero_and_continues(self):
        def _bad(_ticker):
            raise RuntimeError("network error")

        with patch(f"{_MOD}.yf.Ticker", side_effect=_bad):
            client = _make_client()
            caps = client.get_market_caps(["AAPL"])
        assert caps["AAPL"] == 0.0

    def test_empty_ticker_list_returns_empty_dict(self):
        with patch(f"{_MOD}.yf.Ticker") as mock_ticker:
            client = _make_client()
            caps = client.get_market_caps([])
        assert caps == {}
        mock_ticker.assert_not_called()


# ===========================================================================
# Caching — second call must not re-fetch
# ===========================================================================

class TestCaching:

    def test_second_call_uses_cache(self):
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker(1e12)) as mock_ticker:
            client = _make_client()
            client.get_market_caps(["AAPL"])
            client.get_market_caps(["AAPL"])   # should hit cache
        # yf.Ticker called only once despite two get_market_caps calls
        assert mock_ticker.call_count == 1

    def test_stale_cache_refetches(self):
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker(1e12)) as mock_ticker:
            client = _make_client()
            client.get_market_caps(["AAPL"])

            # Manually expire the cache entry
            stale_time = datetime.now(tz=timezone.utc) - timedelta(hours=25)
            client._cache["AAPL"]["fetched_at"] = stale_time

            client.get_market_caps(["AAPL"])   # should re-fetch

        assert mock_ticker.call_count == 2

    def test_partial_cache_only_fetches_missing(self):
        call_log: list[str] = []

        def _side(ticker):
            call_log.append(ticker)
            return _mock_yf_ticker(1e12)

        with patch(f"{_MOD}.yf.Ticker", side_effect=_side):
            client = _make_client()
            client.get_market_caps(["AAPL"])         # fetches AAPL
            client.get_market_caps(["AAPL", "MSFT"]) # only fetches MSFT

        assert call_log == ["AAPL", "MSFT"]

    def test_cache_returns_correct_value(self):
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker(5e11)):
            client = _make_client()
            first  = client.get_market_caps(["GILD"])
            second = client.get_market_caps(["GILD"])
        assert first["GILD"] == pytest.approx(5e11)
        assert second["GILD"] == pytest.approx(5e11)


# ===========================================================================
# get_earnings_dates
# ===========================================================================

def _mock_yf_ticker_earnings(earnings_date):
    """
    Return a yf.Ticker mock whose .calendar returns the given earnings date(s).
    earnings_date may be a pd.Timestamp, list of pd.Timestamps, or None.
    """
    m = MagicMock()
    if earnings_date is None:
        m.calendar = {}
    elif isinstance(earnings_date, list):
        m.calendar = {"Earnings Date": earnings_date}
    else:
        m.calendar = {"Earnings Date": [earnings_date]}
    return m


class TestGetEarningsDates:

    def test_returns_future_earnings_date(self):
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker_earnings(future)):
            client = _make_client()
            result = client.get_earnings_dates(["AAPL"])
        assert result["AAPL"] is not None
        assert result["AAPL"].date() == future.date()

    def test_returns_none_when_calendar_empty(self):
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker_earnings(None)):
            client = _make_client()
            result = client.get_earnings_dates(["AAPL"])
        assert result["AAPL"] is None

    def test_returns_none_for_past_date_only(self):
        past = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=5)
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker_earnings(past)):
            client = _make_client()
            result = client.get_earnings_dates(["AAPL"])
        assert result["AAPL"] is None

    def test_returns_soonest_when_multiple_future_dates(self):
        sooner = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=5)
        later  = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker_earnings([sooner, later])):
            client = _make_client()
            result = client.get_earnings_dates(["AAPL"])
        assert result["AAPL"].date() == sooner.date()

    def test_ignores_past_and_returns_earliest_future(self):
        past   = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=7)
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker_earnings([past, future])):
            client = _make_client()
            result = client.get_earnings_dates(["AAPL"])
        assert result["AAPL"].date() == future.date()

    def test_returns_none_on_exception(self):
        def _bad(_ticker):
            raise RuntimeError("network error")

        with patch(f"{_MOD}.yf.Ticker", side_effect=_bad):
            client = _make_client()
            result = client.get_earnings_dates(["AAPL"])
        assert result["AAPL"] is None

    def test_empty_ticker_list_returns_empty_dict(self):
        with patch(f"{_MOD}.yf.Ticker") as mock_ticker:
            client = _make_client()
            result = client.get_earnings_dates([])
        assert result == {}
        mock_ticker.assert_not_called()

    def test_second_call_uses_cache(self):
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker_earnings(future)) as mock_ticker:
            client = _make_client()
            client.get_earnings_dates(["AAPL"])
            client.get_earnings_dates(["AAPL"])   # should hit cache
        assert mock_ticker.call_count == 1

    def test_stale_earnings_cache_refetches(self):
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker_earnings(future)) as mock_ticker:
            client = _make_client()
            client.get_earnings_dates(["AAPL"])

            stale_time = datetime.now(tz=timezone.utc) - timedelta(hours=25)
            client._earnings_cache["AAPL"]["fetched_at"] = stale_time

            client.get_earnings_dates(["AAPL"])

        assert mock_ticker.call_count == 2

    def test_handles_tz_naive_timestamp(self):
        # yfinance sometimes returns tz-naive Timestamps
        future_naive = pd.Timestamp.now() + pd.Timedelta(days=5)  # no tz
        with patch(f"{_MOD}.yf.Ticker", return_value=_mock_yf_ticker_earnings(future_naive)):
            client = _make_client()
            result = client.get_earnings_dates(["MSFT"])
        # Should still return a valid datetime (tz-naive attached UTC)
        assert result["MSFT"] is not None
