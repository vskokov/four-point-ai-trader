"""
Unit tests for data/alphavantage_client.py.

All HTTP calls are mocked — no live Alpha Vantage access.
Run with:
    .venv/bin/pytest tests/test_alphavantage_client.py -v
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures — realistic Alpha Vantage response payloads
# ---------------------------------------------------------------------------

_AV_FEED_MULTI = {
    "items": "2",
    "sentiment_score_definition": "x <= -0.35: Bearish...",
    "relevance_score_definition": "0 < x <= 1...",
    "feed": [
        {
            "title": "Apple Reports Record Quarterly Revenue",
            "url": "https://example.com/apple-revenue",
            "time_published": "20240115T143000",
            "authors": ["Jane Doe"],
            "summary": "Apple Inc. reported record quarterly revenue beating estimates.",
            "source": "Reuters",
            "overall_sentiment_score": 0.5,
            "overall_sentiment_label": "Bullish",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.95",
                    "ticker_sentiment_score": "0.45",
                    "ticker_sentiment_label": "Bullish",
                }
            ],
        },
        {
            "title": "Tech Stocks Rally on Fed Pivot Hopes",
            "url": "https://example.com/tech-rally",
            "time_published": "20240115T120000",
            "authors": [],
            "summary": "Technology stocks rallied broadly.",
            "source": "Bloomberg",
            "overall_sentiment_score": 0.3,
            "overall_sentiment_label": "Somewhat-Bullish",
            "ticker_sentiment": [
                {
                    "ticker": "AAPL",
                    "relevance_score": "0.70",
                    "ticker_sentiment_score": "0.25",
                    "ticker_sentiment_label": "Somewhat-Bullish",
                },
                {
                    "ticker": "MSFT",
                    "relevance_score": "0.80",
                    "ticker_sentiment_score": "0.35",
                    "ticker_sentiment_label": "Somewhat-Bullish",
                },
            ],
        },
    ],
}

_AV_EMPTY_FEED = {"items": "0", "feed": []}

_AV_INFORMATION = {
    "Information": (
        "Thank you for using Alpha Vantage! Our standard API rate limit is "
        "25 requests per day."
    )
}

_AV_NOTE = {
    "Note": (
        "Thank you for using Alpha Vantage! Our standard API call frequency "
        "is 5 calls per minute."
    )
}

_MOD = "trading_engine.data.alphavantage_client"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(tmp_path: Path, monkeypatch) -> "AlphaVantageNewsClient":  # noqa: F821
    """Instantiate client with a temp rate-state file and fake API key."""
    monkeypatch.setattr(f"{_MOD}.settings", MagicMock(ALPHAVANTAGE_API_KEY="fake-av-key"))
    from trading_engine.data.alphavantage_client import AlphaVantageNewsClient
    return AlphaVantageNewsClient(rate_state_path=tmp_path / "av_rate_state.json")


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# fetch_news — happy path
# ---------------------------------------------------------------------------

class TestFetchNewsHappyPath:

    def test_returns_list_of_dicts(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL", "MSFT"])
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_expected_keys_present(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL"])
        expected = {
            "ticker", "title", "summary", "source", "published_at",
            "relevance_score", "av_sentiment_score", "av_sentiment_label",
            "headline_hash",
        }
        for row in rows:
            assert expected.issubset(row.keys())

    def test_flattens_multi_ticker_article(self, tmp_path, monkeypatch):
        # "Tech Stocks Rally..." has AAPL + MSFT in ticker_sentiment
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL", "MSFT"])
        titles = [r["title"] for r in rows]
        assert titles.count("Tech Stocks Rally on Fed Pivot Hopes") == 2

    def test_total_record_count(self, tmp_path, monkeypatch):
        # Article 1: 1 ticker (AAPL). Article 2: 2 tickers (AAPL, MSFT) → 3 total
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL", "MSFT"])
        assert len(rows) == 3

    def test_published_at_is_utc_datetime(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL"])
        for row in rows:
            assert isinstance(row["published_at"], datetime)
            assert row["published_at"].tzinfo is not None
            assert row["published_at"].tzinfo == timezone.utc

    def test_sentiment_scores_are_floats(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL"])
        for row in rows:
            assert isinstance(row["relevance_score"], float)
            assert isinstance(row["av_sentiment_score"], float)

    def test_sentiment_score_values(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL"])
        # First article: AAPL relevance 0.95, score 0.45
        apple_row = next(r for r in rows if r["title"] == "Apple Reports Record Quarterly Revenue")
        assert apple_row["relevance_score"] == pytest.approx(0.95)
        assert apple_row["av_sentiment_score"] == pytest.approx(0.45)
        assert apple_row["av_sentiment_label"] == "Bullish"

    def test_headline_hash_is_sha256_of_title(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL"])
        for row in rows:
            assert row["headline_hash"] == _sha256(row["title"])

    def test_empty_feed_returns_empty_list(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_EMPTY_FEED)):
            rows = client.fetch_news(["AAPL"])
        assert rows == []

    def test_request_uses_correct_params(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        mock_get = MagicMock(return_value=_mock_response(_AV_EMPTY_FEED))
        with patch(f"{_MOD}.requests.get", mock_get):
            client.fetch_news(["AAPL", "MSFT"], hours_back=4)
        _, kwargs = mock_get.call_args
        params = kwargs["params"]
        assert params["function"] == "NEWS_SENTIMENT"
        assert params["tickers"] == "AAPL,MSFT"
        assert params["limit"] == "50"
        assert params["apikey"] == "fake-av-key"
        # time_from should be a YYYYMMDDTHHMM string
        assert len(params["time_from"]) == 13
        assert "T" in params["time_from"]


# ---------------------------------------------------------------------------
# fetch_news — ticker filtering
# ---------------------------------------------------------------------------

class TestTickerFiltering:

    def test_only_requested_tickers_in_output(self, tmp_path, monkeypatch):
        # Request only AAPL — MSFT rows should be excluded
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["AAPL"])
        tickers_in_output = {r["ticker"] for r in rows}
        assert tickers_in_output == {"AAPL"}

    def test_article_with_no_matching_ticker_skipped(self, tmp_path, monkeypatch):
        # Feed has AAPL articles; requesting TSLA → no matches
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_FEED_MULTI)):
            rows = client.fetch_news(["TSLA"])
        assert rows == []


# ---------------------------------------------------------------------------
# fetch_news — deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_duplicate_title_same_ticker_deduplicated(self, tmp_path, monkeypatch):
        payload = {
            "feed": [
                {
                    "title": "Repeat Headline",
                    "time_published": "20240115T100000",
                    "summary": "First occurrence",
                    "source": "A",
                    "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.8",
                                          "ticker_sentiment_score": "0.2",
                                          "ticker_sentiment_label": "Neutral"}],
                },
                {
                    "title": "Repeat Headline",
                    "time_published": "20240115T110000",
                    "summary": "Second occurrence",
                    "source": "B",
                    "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.7",
                                          "ticker_sentiment_score": "0.1",
                                          "ticker_sentiment_label": "Neutral"}],
                },
            ]
        }
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(payload)):
            rows = client.fetch_news(["AAPL"])
        assert len(rows) == 1

    def test_same_title_different_ticker_not_deduplicated(self, tmp_path, monkeypatch):
        # Same article appearing for AAPL and MSFT is two distinct records
        payload = {
            "feed": [
                {
                    "title": "Cross-ticker Headline",
                    "time_published": "20240115T100000",
                    "summary": "Both mentioned",
                    "source": "C",
                    "ticker_sentiment": [
                        {"ticker": "AAPL", "relevance_score": "0.8",
                         "ticker_sentiment_score": "0.2", "ticker_sentiment_label": "Neutral"},
                        {"ticker": "MSFT", "relevance_score": "0.6",
                         "ticker_sentiment_score": "0.1", "ticker_sentiment_label": "Neutral"},
                    ],
                }
            ]
        }
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(payload)):
            rows = client.fetch_news(["AAPL", "MSFT"])
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# AV non-standard error detection
# ---------------------------------------------------------------------------

class TestAVBodyErrors:

    def test_information_key_raises(self, tmp_path, monkeypatch):
        from trading_engine.data.alphavantage_client import AlphaVantageError
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_INFORMATION)):
            with pytest.raises(AlphaVantageError, match="Information"):
                client.fetch_news(["AAPL"])

    def test_note_key_raises(self, tmp_path, monkeypatch):
        from trading_engine.data.alphavantage_client import AlphaVantageError
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_NOTE)):
            with pytest.raises(AlphaVantageError, match="Note"):
                client.fetch_news(["AAPL"])

    def test_http_error_propagates(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        resp = MagicMock()
        resp.raise_for_status.side_effect = Exception("429 Too Many Requests")
        with patch(f"{_MOD}.requests.get", return_value=resp):
            with pytest.raises(Exception, match="429"):
                client.fetch_news(["AAPL"])


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:

    def _write_state(self, path: Path, count: int) -> None:
        state_file = path / "av_rate_state.json"
        state_file.write_text(json.dumps({"date": date.today().isoformat(), "count": count}))

    def test_counter_increments_on_each_call(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        state_file = tmp_path / "av_rate_state.json"
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_EMPTY_FEED)):
            client.fetch_news(["AAPL"])
            client.fetch_news(["AAPL"])
        state = json.loads(state_file.read_text())
        assert state["count"] == 2

    def test_counter_resets_on_new_day(self, tmp_path, monkeypatch):
        # Seed yesterday's state with count=19
        state_file = tmp_path / "av_rate_state.json"
        state_file.write_text(json.dumps({"date": "2000-01-01", "count": 19}))
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_EMPTY_FEED)):
            client.fetch_news(["AAPL"])   # should not raise
        state = json.loads(state_file.read_text())
        assert state["count"] == 1       # reset to 0 then incremented

    def test_hard_limit_raises_before_request(self, tmp_path, monkeypatch):
        from trading_engine.data.alphavantage_client import RateLimitExceeded
        self._write_state(tmp_path, 20)  # at the hard limit
        client = _make_client(tmp_path, monkeypatch)
        mock_get = MagicMock()
        with patch(f"{_MOD}.requests.get", mock_get):
            with pytest.raises(RateLimitExceeded):
                client.fetch_news(["AAPL"])
        mock_get.assert_not_called()     # request must not be made

    def test_hard_limit_at_19_does_not_raise(self, tmp_path, monkeypatch):
        self._write_state(tmp_path, 19)  # one below the limit
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_EMPTY_FEED)):
            client.fetch_news(["AAPL"])  # should succeed

    def test_warning_logged_above_threshold(self, tmp_path, monkeypatch, caplog):
        import logging
        self._write_state(tmp_path, 15)  # at warn threshold
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_EMPTY_FEED)):
            with caplog.at_level(logging.WARNING):
                client.fetch_news(["AAPL"])
        # structlog doesn't write to caplog by default; check via mock instead
        # Re-test: patch logger.warning directly
        with patch(f"{_MOD}.logger") as mock_log:
            self._write_state(tmp_path, 15)
            client2 = _make_client(tmp_path, monkeypatch)
            with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_EMPTY_FEED)):
                client2.fetch_news(["AAPL"])
            mock_log.warning.assert_called_once()
            call_kwargs = mock_log.warning.call_args
            assert "rate_limit" in call_kwargs[0][0]

    def test_missing_state_file_starts_at_zero(self, tmp_path, monkeypatch):
        # No state file at all
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_EMPTY_FEED)):
            client.fetch_news(["AAPL"])
        state = json.loads((tmp_path / "av_rate_state.json").read_text())
        assert state["count"] == 1
        assert state["date"] == date.today().isoformat()


# ---------------------------------------------------------------------------
# is_market_hours
# ---------------------------------------------------------------------------

class TestIsMarketHours:

    def _mock_now_et(self, monkeypatch, weekday: int, hour: int, minute: int):
        """
        Patch datetime.now inside the module to return a controlled ET datetime.
        weekday: 0=Mon … 6=Sun
        """
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")

        # Build a real datetime object in ET at the given time
        # Use a known date for each weekday: Mon=2024-01-15, ..., Sun=2024-01-21
        base_monday = datetime(2024, 1, 15, tzinfo=_ET)  # a known Monday
        target = base_monday.replace(
            day=15 + weekday, hour=hour, minute=minute, second=0, microsecond=0
        )

        original_datetime = datetime

        class _FakeDatetime(original_datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return target.astimezone(tz)
                return target

        monkeypatch.setattr(f"{_MOD}.datetime", _FakeDatetime)

    def test_open_weekday_during_hours(self, tmp_path, monkeypatch):
        self._mock_now_et(monkeypatch, weekday=0, hour=10, minute=30)  # Mon 10:30 ET
        client = _make_client(tmp_path, monkeypatch)
        assert client.is_market_hours() is True

    def test_exactly_at_open(self, tmp_path, monkeypatch):
        self._mock_now_et(monkeypatch, weekday=1, hour=9, minute=30)   # Tue 09:30 ET
        client = _make_client(tmp_path, monkeypatch)
        assert client.is_market_hours() is True

    def test_one_minute_before_open(self, tmp_path, monkeypatch):
        self._mock_now_et(monkeypatch, weekday=2, hour=9, minute=29)   # Wed 09:29 ET
        client = _make_client(tmp_path, monkeypatch)
        assert client.is_market_hours() is False

    def test_exactly_at_close_is_closed(self, tmp_path, monkeypatch):
        self._mock_now_et(monkeypatch, weekday=3, hour=16, minute=0)   # Thu 16:00 ET
        client = _make_client(tmp_path, monkeypatch)
        assert client.is_market_hours() is False

    def test_one_minute_before_close(self, tmp_path, monkeypatch):
        self._mock_now_et(monkeypatch, weekday=4, hour=15, minute=59)  # Fri 15:59 ET
        client = _make_client(tmp_path, monkeypatch)
        assert client.is_market_hours() is True

    def test_saturday_is_closed(self, tmp_path, monkeypatch):
        self._mock_now_et(monkeypatch, weekday=5, hour=12, minute=0)   # Sat 12:00 ET
        client = _make_client(tmp_path, monkeypatch)
        assert client.is_market_hours() is False

    def test_sunday_is_closed(self, tmp_path, monkeypatch):
        self._mock_now_et(monkeypatch, weekday=6, hour=11, minute=0)   # Sun 11:00 ET
        client = _make_client(tmp_path, monkeypatch)
        assert client.is_market_hours() is False


# ---------------------------------------------------------------------------
# Per-minute rate guard
# ---------------------------------------------------------------------------

class TestPerMinuteRateGuard:
    """Tests for _enforce_per_minute_limit (uses mocked time.monotonic / time.sleep)."""

    def _make_mock_time(self, monotonic_value: float = 0.0) -> MagicMock:
        mock_time = MagicMock()
        mock_time.monotonic.return_value = monotonic_value
        mock_time.sleep = MagicMock()
        return mock_time

    def test_five_rapid_calls_no_sleep(self, tmp_path, monkeypatch):
        """First 5 calls within the same instant must not trigger sleep."""
        mock_time = self._make_mock_time(0.0)
        monkeypatch.setattr(f"{_MOD}.time", mock_time)
        client = _make_client(tmp_path, monkeypatch)

        for _ in range(5):
            client._enforce_per_minute_limit()

        mock_time.sleep.assert_not_called()

    def test_sixth_rapid_call_triggers_sleep(self, tmp_path, monkeypatch):
        """6th call within the same 60-second window must trigger time.sleep."""
        mock_time = self._make_mock_time(0.0)
        monkeypatch.setattr(f"{_MOD}.time", mock_time)
        client = _make_client(tmp_path, monkeypatch)

        for _ in range(5):
            client._enforce_per_minute_limit()

        # 6th call: 5 entries at t=0.0 → sleep for 60 seconds
        client._enforce_per_minute_limit()

        mock_time.sleep.assert_called_once()
        sleep_arg = mock_time.sleep.call_args[0][0]
        assert sleep_arg == pytest.approx(60.0)

    def test_call_after_60_seconds_no_sleep(self, tmp_path, monkeypatch):
        """After 60 seconds the old entries are pruned; 6th call must not sleep."""
        # First 5 calls at t=0; 6th call at t=61 (past the 60-second window)
        times = [0.0, 0.0, 0.0, 0.0, 0.0, 61.0, 61.0]
        idx = [0]

        mock_time = MagicMock()
        mock_time.sleep = MagicMock()

        def fake_monotonic():
            val = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return val

        mock_time.monotonic.side_effect = fake_monotonic
        monkeypatch.setattr(f"{_MOD}.time", mock_time)

        client = _make_client(tmp_path, monkeypatch)
        for _ in range(5):
            client._enforce_per_minute_limit()

        # 6th call at t=61: entries at t=0 are pruned (61-0=61 >= 60)
        client._enforce_per_minute_limit()

        mock_time.sleep.assert_not_called()

    def test_recent_call_times_pruned_after_window(self, tmp_path, monkeypatch):
        """After pruning, _recent_call_times must contain only in-window entries."""
        times = [0.0] * 3 + [65.0]
        idx = [0]

        mock_time = MagicMock()
        mock_time.sleep = MagicMock()

        def fake_monotonic():
            val = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return val

        mock_time.monotonic.side_effect = fake_monotonic
        monkeypatch.setattr(f"{_MOD}.time", mock_time)

        client = _make_client(tmp_path, monkeypatch)
        for _ in range(3):
            client._enforce_per_minute_limit()  # t=0 each

        # At t=65 the 3 entries at t=0 are outside the 60-second window
        client._enforce_per_minute_limit()

        # Only the entry for t=65 should remain
        assert len(client._recent_call_times) == 1


# ---------------------------------------------------------------------------
# get_daily_call_count
# ---------------------------------------------------------------------------

class TestGetDailyCallCount:

    def test_returns_zero_when_no_state_file(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        assert client.get_daily_call_count() == 0

    def test_returns_correct_count_after_calls(self, tmp_path, monkeypatch):
        client = _make_client(tmp_path, monkeypatch)
        with patch(f"{_MOD}.requests.get", return_value=_mock_response(_AV_EMPTY_FEED)):
            client.fetch_news(["AAPL"])
            client.fetch_news(["AAPL"])
        assert client.get_daily_call_count() == 2

    def test_returns_zero_for_stale_date(self, tmp_path, monkeypatch):
        """A state file from a past date must be treated as zero for today."""
        state_file = tmp_path / "av_rate_state.json"
        state_file.write_text(json.dumps({"date": "2000-01-01", "count": 15}))
        client = _make_client(tmp_path, monkeypatch)
        assert client.get_daily_call_count() == 0

    def test_returns_correct_count_from_preloaded_state(self, tmp_path, monkeypatch):
        """If today's state is pre-seeded, count is read correctly."""
        state_file = tmp_path / "av_rate_state.json"
        state_file.write_text(
            json.dumps({"date": date.today().isoformat(), "count": 7})
        )
        client = _make_client(tmp_path, monkeypatch)
        assert client.get_daily_call_count() == 7
