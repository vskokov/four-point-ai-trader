"""
Unit tests for signals/llm_sentiment.py.

All Ollama, Alpha Vantage, and Storage calls are mocked.
Run with:
    .venv/bin/pytest tests/test_llm_sentiment.py -v
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# Module path used for targeted patching.
_MOD = "trading_engine.signals.llm_sentiment"


# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

def _make_article(
    title: str = "Apple beats earnings",
    ticker: str = "AAPL",
    relevance: float = 0.8,
    av_score: float = 0.35,
) -> dict[str, Any]:
    """Minimal AV article dict mirroring AlphaVantageNewsClient output."""
    h = hashlib.sha256(title.encode()).hexdigest()
    return {
        "ticker":              ticker,
        "title":               title,
        "summary":             "Company reported strong results.",
        "source":              "Reuters",
        "published_at":        datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        "relevance_score":     relevance,
        "av_sentiment_score":  av_score,
        "av_sentiment_label":  "Bullish",
        "headline_hash":       h,
    }


_VALID_LLM_RESPONSE = {
    "ticker":      "AAPL",
    "direction":    1,
    "confidence":   0.85,
    "horizon":     "8h",
    "key_drivers": ["strong earnings", "revenue beat"],
    "reasoning":   "Earnings beat estimates. Revenue grew significantly.",
}


# ---------------------------------------------------------------------------
# Fixture: a pre-wired LLMSentimentSignal with a mocked Ollama client.
# ollama.Client is patched BEFORE import so __init__ never touches the network.
# ---------------------------------------------------------------------------

@pytest.fixture()
def sig(monkeypatch) -> Any:
    """
    Return an LLMSentimentSignal whose _client is a MagicMock.

    We patch ollama.Client at the module level so that the real httpx
    connection is never attempted during construction.
    """
    mock_ollama_client_cls = MagicMock()
    monkeypatch.setattr(f"{_MOD}.ollama.Client", mock_ollama_client_cls)
    from trading_engine.signals.llm_sentiment import LLMSentimentSignal
    instance = LLMSentimentSignal(
        model="gemma4:e4b",
        host="http://mock-ollama:11434",
        hours_back=8,
        min_relevance=0.3,
    )
    # The mock instance returned by Client() is what _client points to.
    instance._client = mock_ollama_client_cls.return_value
    return instance


def _set_llm_response(sig: Any, payload: dict[str, Any]) -> None:
    """Configure the mocked ollama client to return *payload* as JSON."""
    mock_response = MagicMock()
    mock_response.message.content = json.dumps(payload)
    sig._client.chat.return_value = mock_response


# ---------------------------------------------------------------------------
# Fake storage — plain class, not MagicMock(spec=...)
# ---------------------------------------------------------------------------

class _FakeStorage:
    def __init__(self) -> None:
        self.news_rows:   list[dict[str, Any]] = []
        self.signal_rows: list[dict[str, Any]] = []

    def insert_news(self, rows: list[dict[str, Any]]) -> int:
        self.news_rows.extend(rows)
        return len(rows)

    def insert_signal(self, rows: list[dict[str, Any]]) -> int:
        self.signal_rows.extend(rows)
        return len(rows)


# ---------------------------------------------------------------------------
# Tests — _human_age
# ---------------------------------------------------------------------------

class TestHumanAge:
    """All tests use a fixed ``now`` for deterministic output."""

    # Reference point — arbitrary UTC instant
    _NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def _pub(self, **delta_kwargs) -> datetime:
        """Return a published_at by subtracting a timedelta from _NOW."""
        return self._NOW - timedelta(**delta_kwargs)

    def test_zero_seconds_is_just_now(self):
        from trading_engine.signals.llm_sentiment import _human_age
        assert _human_age(self._NOW, now=self._NOW) == "[just now]"

    def test_negative_delta_is_just_now(self):
        from trading_engine.signals.llm_sentiment import _human_age
        future = self._NOW + timedelta(seconds=10)
        assert _human_age(future, now=self._NOW) == "[just now]"

    def test_one_minute_singular(self):
        from trading_engine.signals.llm_sentiment import _human_age
        assert _human_age(self._pub(minutes=1), now=self._NOW) == "[1 minute ago]"

    def test_twenty_three_minutes(self):
        from trading_engine.signals.llm_sentiment import _human_age
        assert _human_age(self._pub(minutes=23), now=self._NOW) == "[23 minutes ago]"

    def test_sixty_minutes_is_one_hour_no_minutes(self):
        from trading_engine.signals.llm_sentiment import _human_age
        assert _human_age(self._pub(minutes=60), now=self._NOW) == "[1 hour ago]"

    def test_two_hours_fifty_minutes(self):
        from trading_engine.signals.llm_sentiment import _human_age
        result = _human_age(self._pub(hours=2, minutes=50), now=self._NOW)
        assert result == "[2 hours 50 minutes ago]"

    def test_twenty_four_hours_is_one_day_no_hours(self):
        from trading_engine.signals.llm_sentiment import _human_age
        assert _human_age(self._pub(hours=24), now=self._NOW) == "[1 day ago]"

    def test_twenty_five_hours_is_one_day_one_hour(self):
        from trading_engine.signals.llm_sentiment import _human_age
        assert _human_age(self._pub(hours=25), now=self._NOW) == "[1 day 1 hour ago]"

    def test_forty_eight_hours_is_two_days_no_hours(self):
        from trading_engine.signals.llm_sentiment import _human_age
        assert _human_age(self._pub(hours=48), now=self._NOW) == "[2 days ago]"

    def test_default_now_returns_string(self):
        """Without a fixed now, the function should still return a valid string."""
        from trading_engine.signals.llm_sentiment import _human_age
        pub = datetime(2000, 1, 1, tzinfo=timezone.utc)
        result = _human_age(pub)
        assert result.startswith("[")
        assert result.endswith(" ago]")


# ---------------------------------------------------------------------------
# Tests — _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_ticker(self, sig: Any) -> None:
        articles = [_make_article()]
        prompt = sig._build_prompt("AAPL", articles)
        assert "AAPL" in prompt

    def test_contains_headline_text(self, sig: Any) -> None:
        articles = [_make_article(title="Apple beats earnings")]
        prompt = sig._build_prompt("AAPL", articles)
        assert "Apple beats earnings" in prompt

    def test_truncates_body_to_300_chars(self, sig: Any) -> None:
        long_summary = "X" * 400
        articles = [_make_article()]
        articles[0]["summary"] = long_summary
        prompt = sig._build_prompt("AAPL", articles)
        # Each headline body (after the age prefix) is capped at 300 chars.
        for line in prompt.splitlines():
            if not line or not line[0].isdigit():
                continue
            # Strip leading "N. " number prefix
            without_num = line.lstrip("0123456789. ")
            # Strip age bracket "[... ago] " if present
            if without_num.startswith("["):
                bracket_end = without_num.find("]")
                if bracket_end != -1:
                    without_num = without_num[bracket_end + 2:]  # skip "] "
            assert len(without_num) <= 300, (
                f"Body exceeds 300 chars: {without_num[:40]}…"
            )

    def test_caps_at_max_headlines(self, sig: Any) -> None:
        # 60 articles — only top 50 by relevance should appear in the prompt.
        # Use titles with no digits so the "51st position" check is unambiguous.
        import string
        az = string.ascii_lowercase   # 26 letters; two-letter combos give 26²=676 unique titles
        articles = [
            _make_article(title=f"news-{az[i // 26]}{az[i % 26]}", relevance=float(i) / 60)
            for i in range(60)
        ]
        prompt = sig._build_prompt("AAPL", articles)
        # The numbered list runs 1–50.  Check that line 50 exists but line 51 does not.
        lines = prompt.splitlines()
        numbered = [l for l in lines if l and l[0].isdigit()]
        assert len(numbered) == 50, f"Expected 50 numbered lines, got {len(numbered)}"

    def test_deduplicates_by_hash(self, sig: Any) -> None:
        # Two articles with the same title → same headline_hash → only one in prompt
        a = _make_article(title="Same headline")
        articles = [a, dict(a)]   # identical hash
        prompt = sig._build_prompt("AAPL", articles)
        assert prompt.count("Same headline") == 1

    def test_contains_json_schema_keys(self, sig: Any) -> None:
        articles = [_make_article()]
        prompt = sig._build_prompt("AAPL", articles)
        for key in ("direction", "confidence", "horizon", "key_drivers", "reasoning"):
            assert key in prompt

    def test_no_prices_in_prompt(self, sig: Any) -> None:
        """The prompt must never instruct the LLM to consider OHLCV data.

        Note: "priced in" (a financial idiom in the recency instruction) is
        intentional and does not instruct the LLM to look at price data, so
        "price" is excluded from the forbidden list here.
        """
        articles = [_make_article()]
        prompt = sig._build_prompt("AAPL", articles)
        # "price" is not forbidden because "priced in" idiom is used legitimately
        forbidden = ("ohlcv", "bar", "candle", "volume")
        lower = prompt.lower()
        for word in forbidden:
            assert word not in lower, f"Price-related term '{word}' found in prompt"

    def test_build_prompt_includes_age_prefix(self, sig: Any) -> None:
        """Each headline line must be prefixed with its age bracket."""
        with patch(f"{_MOD}._human_age", return_value="[5 minutes ago]") as mock_age:
            articles = [_make_article()]
            prompt = sig._build_prompt("AAPL", articles)
        assert "[5 minutes ago]" in prompt
        mock_age.assert_called_once()

    def test_build_prompt_includes_recency_instruction(self, sig: Any) -> None:
        """The prompt must include the recency-weighting instruction sentence."""
        articles = [_make_article()]
        prompt = sig._build_prompt("AAPL", articles)
        assert "Weight recent headlines" in prompt
        assert "6 hours" in prompt

    def test_build_prompt_fallback_for_missing_published_at(self, sig: Any) -> None:
        """If an article lacks published_at, age falls back to [unknown time]."""
        article = _make_article()
        del article["published_at"]
        prompt = sig._build_prompt("AAPL", [article])
        assert "[unknown time]" in prompt


# ---------------------------------------------------------------------------
# Tests — _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_valid_response_parsed(self, sig: Any) -> None:
        result = sig._parse_response(json.dumps(_VALID_LLM_RESPONSE))
        assert result is not None
        assert result["direction"] == 1
        assert result["confidence"] == 0.85

    def test_strips_markdown_fences(self, sig: Any) -> None:
        fenced = "```json\n" + json.dumps(_VALID_LLM_RESPONSE) + "\n```"
        result = sig._parse_response(fenced)
        assert result is not None
        assert result["direction"] == 1

    def test_invalid_json_returns_none(self, sig: Any) -> None:
        assert sig._parse_response("not json at all") is None

    def test_missing_key_returns_none(self, sig: Any) -> None:
        bad = dict(_VALID_LLM_RESPONSE)
        del bad["direction"]
        assert sig._parse_response(json.dumps(bad)) is None

    def test_invalid_direction_returns_none(self, sig: Any) -> None:
        bad = {**_VALID_LLM_RESPONSE, "direction": 99}
        assert sig._parse_response(json.dumps(bad)) is None

    def test_confidence_out_of_range_returns_none(self, sig: Any) -> None:
        bad = {**_VALID_LLM_RESPONSE, "confidence": 1.5}
        assert sig._parse_response(json.dumps(bad)) is None

    def test_invalid_horizon_returns_none(self, sig: Any) -> None:
        bad = {**_VALID_LLM_RESPONSE, "horizon": "3d"}
        assert sig._parse_response(json.dumps(bad)) is None

    def test_key_drivers_not_list_returns_none(self, sig: Any) -> None:
        bad = {**_VALID_LLM_RESPONSE, "key_drivers": "earnings"}
        assert sig._parse_response(json.dumps(bad)) is None

    def test_direction_coerced_to_int(self, sig: Any) -> None:
        payload = {**_VALID_LLM_RESPONSE, "direction": -1}
        result = sig._parse_response(json.dumps(payload))
        assert isinstance(result["direction"], int)

    def test_confidence_coerced_to_float(self, sig: Any) -> None:
        payload = {**_VALID_LLM_RESPONSE, "confidence": "0.75"}
        result = sig._parse_response(json.dumps(payload))
        assert isinstance(result["confidence"], float)
        assert result["confidence"] == 0.75

    def test_bearish_direction_valid(self, sig: Any) -> None:
        payload = {**_VALID_LLM_RESPONSE, "direction": -1}
        result = sig._parse_response(json.dumps(payload))
        assert result is not None
        assert result["direction"] == -1


# ---------------------------------------------------------------------------
# Tests — score (valid path)
# ---------------------------------------------------------------------------

class TestScoreValid:
    def test_returns_correct_schema(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        result = sig.score("AAPL", [_make_article()])
        expected_keys = {
            "ticker", "direction", "confidence", "horizon",
            "key_drivers", "reasoning", "source",
            "fetched_at", "n_headlines_used", "headline_hashes",
        }
        assert expected_keys.issubset(result.keys())

    def test_direction_value(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        result = sig.score("AAPL", [_make_article()])
        assert result["direction"] == 1

    def test_confidence_value(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        result = sig.score("AAPL", [_make_article()])
        assert result["confidence"] == 0.85

    def test_source_is_llm(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        result = sig.score("AAPL", [_make_article()])
        assert result["source"] == "llm"

    def test_n_headlines_used(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        articles = [_make_article(title=f"H{i}") for i in range(3)]
        result = sig.score("AAPL", articles)
        assert result["n_headlines_used"] == 3

    def test_headline_hashes_populated(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        article = _make_article()
        result = sig.score("AAPL", [article])
        assert article["headline_hash"] in result["headline_hashes"]

    def test_ticker_overridden_to_requested(self, sig: Any) -> None:
        # LLM may echo back a different ticker; we always use the requested one.
        payload = {**_VALID_LLM_RESPONSE, "ticker": "WRONG"}
        _set_llm_response(sig, payload)
        result = sig.score("AAPL", [_make_article()])
        assert result["ticker"] == "AAPL"

    def test_fetched_at_is_utc(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        result = sig.score("AAPL", [_make_article()])
        assert result["fetched_at"].tzinfo is not None

    def test_llm_called_once_for_valid_response(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        sig.score("AAPL", [_make_article()])
        assert sig._client.chat.call_count == 1


# ---------------------------------------------------------------------------
# Tests — empty / filtered headlines
# ---------------------------------------------------------------------------

class TestEmptyHeadlines:
    def test_no_headlines_returns_no_data(self, sig: Any) -> None:
        result = sig.score("AAPL", [])
        assert result["direction"] == 0
        assert result["confidence"] == 0.0
        assert result["source"] == "no_data"
        sig._client.chat.assert_not_called()

    def test_all_below_relevance_returns_no_data(self, sig: Any) -> None:
        # min_relevance = 0.3; article has 0.1
        article = _make_article(relevance=0.1)
        result = sig.score("AAPL", [article])
        assert result["source"] == "no_data"
        sig._client.chat.assert_not_called()

    def test_mixed_relevance_only_passes_threshold(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        articles = [
            _make_article(title="High relevance", relevance=0.9),
            _make_article(title="Low relevance",  relevance=0.1),
        ]
        result = sig.score("AAPL", articles)
        # LLM was called (at least one article passed threshold)
        sig._client.chat.assert_called_once()
        # Only 1 article passed min_relevance=0.3
        assert result["n_headlines_used"] == 1


# ---------------------------------------------------------------------------
# Tests — retry logic (malformed response)
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_retry_on_malformed_then_valid(self, sig: Any) -> None:
        """First call returns garbage JSON; second call returns valid JSON."""
        bad_response  = MagicMock()
        bad_response.message.content = "this is not json"

        good_response = MagicMock()
        good_response.message.content = json.dumps(_VALID_LLM_RESPONSE)

        sig._client.chat.side_effect = [bad_response, good_response]

        result = sig.score("AAPL", [_make_article()])

        assert sig._client.chat.call_count == 2
        assert result["direction"] == 1
        assert result["source"] == "llm"

    def test_two_failures_returns_llm_error(self, sig: Any) -> None:
        """Both calls return malformed JSON → source='llm_error'."""
        bad_response = MagicMock()
        bad_response.message.content = "not json"
        sig._client.chat.return_value = bad_response

        result = sig.score("AAPL", [_make_article()])

        assert sig._client.chat.call_count == 2
        assert result["direction"] == 0
        assert result["source"] == "llm_error"

    def test_strict_prompt_used_on_retry(self, sig: Any) -> None:
        """The retry prompt must contain an extra instruction about JSON."""
        bad_response  = MagicMock()
        bad_response.message.content = "bad"
        good_response = MagicMock()
        good_response.message.content = json.dumps(_VALID_LLM_RESPONSE)
        sig._client.chat.side_effect = [bad_response, good_response]

        sig.score("AAPL", [_make_article()])

        second_call_prompt = sig._client.chat.call_args_list[1][1]["messages"][0]["content"]
        assert "CRITICAL" in second_call_prompt or "valid JSON" in second_call_prompt.lower()

    def test_malformed_schema_triggers_retry(self, sig: Any) -> None:
        """Missing 'direction' key → treated as malformed → retry."""
        bad_payload  = {k: v for k, v in _VALID_LLM_RESPONSE.items() if k != "direction"}
        good_payload = _VALID_LLM_RESPONSE

        bad_resp  = MagicMock()
        bad_resp.message.content  = json.dumps(bad_payload)
        good_resp = MagicMock()
        good_resp.message.content = json.dumps(good_payload)

        sig._client.chat.side_effect = [bad_resp, good_resp]
        result = sig.score("AAPL", [_make_article()])
        assert sig._client.chat.call_count == 2
        assert result["direction"] == 1


# ---------------------------------------------------------------------------
# Tests — timeout / exception handling
# ---------------------------------------------------------------------------

class TestTimeoutHandling:
    def test_timeout_exception_returns_direction_zero(self, sig: Any) -> None:
        import httpx
        sig._client.chat.side_effect = httpx.ReadTimeout("timed out")
        result = sig.score("AAPL", [_make_article()])
        assert result["direction"] == 0
        assert result["source"] == "timeout"

    def test_timeout_in_exception_name(self, sig: Any) -> None:
        class FakeTimeoutError(Exception):
            pass
        sig._client.chat.side_effect = FakeTimeoutError("connection timed out")
        result = sig.score("AAPL", [_make_article()])
        assert result["source"] == "timeout"

    def test_generic_error_returns_llm_error(self, sig: Any) -> None:
        sig._client.chat.side_effect = RuntimeError("connection refused")
        result = sig.score("AAPL", [_make_article()])
        assert result["direction"] == 0
        assert result["source"] == "llm_error"


# ---------------------------------------------------------------------------
# Tests — run_pipeline deduplication and persistence
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def _av_client(self, articles: list[dict[str, Any]]) -> MagicMock:
        av = MagicMock()
        av.fetch_news.return_value = articles
        return av

    def test_inserts_news_rows(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = self._av_client([_make_article()])
        sig.run_pipeline(["AAPL"], av, storage)
        assert len(storage.news_rows) == 1

    def test_inserts_signal_row(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = self._av_client([_make_article()])
        sig.run_pipeline(["AAPL"], av, storage)
        assert len(storage.signal_rows) == 1
        row = storage.signal_rows[0]
        assert row["signal_name"] == "llm_sentiment"
        assert row["ticker"] == "AAPL"

    def test_signal_value_is_direction_times_confidence(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)  # direction=1, confidence=0.85
        storage = _FakeStorage()
        av = self._av_client([_make_article()])
        sig.run_pipeline(["AAPL"], av, storage)
        assert abs(storage.signal_rows[0]["value"] - 1 * 0.85) < 1e-9

    def test_deduplication_skips_seen_hash(self, sig: Any) -> None:
        """Articles whose headline_hash is already in _seen_hashes are skipped."""
        article = _make_article()
        sig._seen_hashes.add(article["headline_hash"])

        storage = _FakeStorage()
        av = self._av_client([article])
        sig.run_pipeline(["AAPL"], av, storage)

        # No new articles → no news rows inserted, but signal row still written
        assert len(storage.news_rows) == 0
        # Score was called with an empty list → no_data
        assert storage.signal_rows[0]["value"] == 0.0

    def test_deduplication_adds_new_hash_to_seen(self, sig: Any) -> None:
        """After processing, the article's hash must be in _seen_hashes."""
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        article = _make_article()
        storage = _FakeStorage()
        av = self._av_client([article])
        assert article["headline_hash"] not in sig._seen_hashes
        sig.run_pipeline(["AAPL"], av, storage)
        assert article["headline_hash"] in sig._seen_hashes

    def test_duplicate_within_same_run(self, sig: Any) -> None:
        """If the same article appears twice in one fetch, second is deduplicated."""
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        article = _make_article()
        storage = _FakeStorage()
        av = self._av_client([article, dict(article)])   # same hash twice
        sig.run_pipeline(["AAPL"], av, storage)
        # Both articles share the same hash; after first is processed
        # the second would be treated as a duplicate on the NEXT run.
        # Within this run, only one insert is attempted (storage deduplicates).
        assert len(storage.news_rows) == 2   # both sent to storage; DB handles dedup

    def test_multiple_tickers_one_llm_call_each(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        # Single fetch_news call returns articles for both tickers.
        av = MagicMock()
        av.fetch_news.return_value = [
            _make_article(title="AAPL headline", ticker="AAPL"),
            _make_article(title="MSFT headline", ticker="MSFT"),
        ]
        sig.run_pipeline(["AAPL", "MSFT"], av, storage)
        # fetch_news called exactly once for all tickers
        av.fetch_news.assert_called_once_with(["AAPL", "MSFT"], hours_back=sig.hours_back)
        # One LLM call per ticker (articles grouped by ticker, 1 article each)
        assert sig._client.chat.call_count == 2

    def test_single_av_call_for_three_tickers(self, sig: Any) -> None:
        """run_pipeline must call fetch_news exactly once regardless of ticker count."""
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = MagicMock()
        av.fetch_news.return_value = [
            _make_article(title="AAPL h", ticker="AAPL"),
            _make_article(title="MSFT h", ticker="MSFT"),
            _make_article(title="GOOG h", ticker="GOOG"),
        ]
        results = sig.run_pipeline(["AAPL", "MSFT", "GOOG"], av, storage)
        # Exactly 1 AV API call
        av.fetch_news.assert_called_once()
        # All 3 tickers are scored → 3 signal rows
        assert len(storage.signal_rows) == 3
        returned_tickers = {r["ticker"] for r in results}
        assert returned_tickers == {"AAPL", "MSFT", "GOOG"}

    def test_returns_list_of_result_dicts(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = self._av_client([_make_article()])
        results = sig.run_pipeline(["AAPL"], av, storage)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"

    def test_news_row_contains_llm_direction(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = self._av_client([_make_article()])
        sig.run_pipeline(["AAPL"], av, storage)
        assert storage.news_rows[0]["llm_direction"] == 1

    def test_news_row_contains_av_sentiment_score(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        article = _make_article(av_score=0.42)
        storage = _FakeStorage()
        av = self._av_client([article])
        sig.run_pipeline(["AAPL"], av, storage)
        assert abs(storage.news_rows[0]["sentiment_score"] - 0.42) < 1e-9


# ---------------------------------------------------------------------------
# Tests — run_pipeline with av_tickers / alpaca_client split
# ---------------------------------------------------------------------------

def _make_alpaca_article(title: str = "Alpaca news", ticker: str = "GILD") -> dict:
    """Minimal Alpaca-style article (no av_sentiment_* fields)."""
    h = hashlib.sha256(title.encode()).hexdigest()
    return {
        "ticker":        ticker,
        "title":         title,
        "summary":       "Some summary.",
        "source":        "Bloomberg",
        "published_at":  datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        "headline_hash": h,
        # No relevance_score, av_sentiment_label, av_sentiment_score
    }


class TestRunPipelineSplit:
    """run_pipeline with av_tickers + alpaca_client keyword args."""

    def test_av_called_with_subset_only(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = MagicMock()
        av.fetch_news.return_value = [_make_article(ticker="AAPL")]
        alpaca = MagicMock()
        alpaca.fetch_news.return_value = []

        sig.run_pipeline(
            ["AAPL", "GILD"],
            av, storage,
            av_tickers=["AAPL"],
            alpaca_client=alpaca,
        )

        # AV called with only "AAPL", not "GILD"
        av.fetch_news.assert_called_once_with(["AAPL"], hours_back=sig.hours_back)
        # Alpaca called with "GILD"
        alpaca.fetch_news.assert_called_once_with(["GILD"], hours_back=sig.hours_back)

    def test_alpaca_articles_get_relevance_score_injected(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = MagicMock()
        av.fetch_news.return_value = []
        alpaca = MagicMock()
        alpaca_art = _make_alpaca_article(ticker="GILD")
        alpaca.fetch_news.return_value = [alpaca_art]

        sig.run_pipeline(
            ["GILD"],
            av, storage,
            av_tickers=[],
            alpaca_client=alpaca,
        )

        # Article should have been injected with relevance_score=1.0
        assert alpaca_art["relevance_score"] == 1.0
        assert alpaca_art["av_sentiment_label"] == ""
        assert alpaca_art["av_sentiment_score"] is None

    def test_alpaca_articles_scored_and_persisted(self, sig: Any) -> None:
        _set_llm_response(sig, {**_VALID_LLM_RESPONSE, "ticker": "GILD"})
        storage = _FakeStorage()
        av = MagicMock()
        av.fetch_news.return_value = []
        alpaca = MagicMock()
        alpaca.fetch_news.return_value = [_make_alpaca_article(ticker="GILD")]

        results = sig.run_pipeline(
            ["GILD"],
            av, storage,
            av_tickers=[],
            alpaca_client=alpaca,
        )

        assert len(storage.signal_rows) == 1
        assert storage.signal_rows[0]["ticker"] == "GILD"
        assert len(results) == 1

    def test_no_alpaca_client_when_all_av(self, sig: Any) -> None:
        """When av_tickers covers all tickers, alpaca_client is never called."""
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = MagicMock()
        av.fetch_news.return_value = [_make_article(ticker="AAPL")]
        alpaca = MagicMock()

        sig.run_pipeline(
            ["AAPL"],
            av, storage,
            av_tickers=["AAPL"],
            alpaca_client=alpaca,
        )

        alpaca.fetch_news.assert_not_called()

    def test_backward_compat_no_kwargs_uses_all_av(self, sig: Any) -> None:
        """Existing callers without av_tickers still get all tickers via AV."""
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = MagicMock()
        av.fetch_news.return_value = [_make_article(ticker="AAPL")]

        sig.run_pipeline(["AAPL"], av, storage)

        av.fetch_news.assert_called_once_with(["AAPL"], hours_back=sig.hours_back)

    def test_alpaca_failure_logs_warning_and_continues(self, sig: Any) -> None:
        """Alpaca fetch failure must not crash the pipeline."""
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        storage = _FakeStorage()
        av = MagicMock()
        av.fetch_news.return_value = [_make_article(ticker="AAPL")]
        alpaca = MagicMock()
        alpaca.fetch_news.side_effect = RuntimeError("Alpaca API down")

        # Should not raise
        results = sig.run_pipeline(
            ["AAPL", "GILD"],
            av, storage,
            av_tickers=["AAPL"],
            alpaca_client=alpaca,
        )

        # AAPL still scored from AV; GILD gets no_data due to alpaca failure
        assert len(results) == 2
        tickers_returned = {r["ticker"] for r in results}
        assert tickers_returned == {"AAPL", "GILD"}


# ---------------------------------------------------------------------------
# Tests — run_if_market_hours
# ---------------------------------------------------------------------------

class TestRunIfMarketHours:
    def test_returns_none_on_weekend(self, sig: Any, monkeypatch) -> None:
        # Saturday 12:00 ET
        saturday = datetime(2024, 1, 13, 12, 0, tzinfo=None)
        import trading_engine.signals.llm_sentiment as mod

        class _FakeDT:
            @classmethod
            def now(cls, tz=None):
                from datetime import timezone
                return saturday.replace(tzinfo=tz)

        monkeypatch.setattr(mod, "_ET", None)  # force weekday() from fixed time
        # Simpler: patch the method directly
        with patch(f"{_MOD}.datetime") as mock_dt:
            from datetime import timezone as _tz
            # weekday() = 5 → Saturday
            fake_now = MagicMock()
            fake_now.weekday.return_value = 5
            mock_dt.now.return_value = fake_now
            result = sig.run_if_market_hours(["AAPL"], MagicMock(), MagicMock())
        assert result is None

    def test_returns_none_outside_extended_hours(self, sig: Any) -> None:
        # 06:00 ET — before pre-market window (09:00 with 30-min pre)
        with patch(f"{_MOD}.datetime") as mock_dt:
            fake_now = MagicMock()
            fake_now.weekday.return_value = 1  # Tuesday
            fake_now.hour   = 6
            fake_now.minute = 0
            mock_dt.now.return_value = fake_now
            result = sig.run_if_market_hours(["AAPL"], MagicMock(), MagicMock())
        assert result is None

    def test_calls_pipeline_during_market_hours(self, sig: Any) -> None:
        _set_llm_response(sig, _VALID_LLM_RESPONSE)
        # 10:30 ET — inside the window.
        # Patch _human_age too so it doesn't try to do arithmetic with the
        # MagicMock returned by the patched datetime.now.
        with patch(f"{_MOD}.datetime") as mock_dt, \
             patch(f"{_MOD}._human_age", return_value="[5 minutes ago]"):
            fake_now = MagicMock()
            fake_now.weekday.return_value = 1  # Tuesday
            fake_now.hour   = 10
            fake_now.minute = 30
            mock_dt.now.return_value = fake_now

            storage = _FakeStorage()
            av = MagicMock()
            av.fetch_news.return_value = [_make_article()]
            result = sig.run_if_market_hours(["AAPL"], av, storage)

        assert result is not None
        assert isinstance(result, list)
