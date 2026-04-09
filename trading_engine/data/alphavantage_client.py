"""
Alpha Vantage news ingestion client.

Primary news source for the pipeline — provides pre-computed per-ticker
sentiment scores alongside articles, which the Alpaca News API does not.

No alpaca-py dependencies; uses only requests + stdlib.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

import trading_engine.config.settings as settings
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.alphavantage.co/query"
_DEFAULT_RATE_STATE_PATH = (
    Path(__file__).parent.parent / "config" / "av_rate_state.json"
)
_DAILY_WARN_THRESHOLD = 15   # log warning above this many calls today
_DAILY_HARD_LIMIT = 20       # raise RateLimitExceeded at or above this
_ET = ZoneInfo("America/New_York")
_MARKET_OPEN  = (9,  30)     # (hour, minute) ET
_MARKET_CLOSE = (16,  0)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RateLimitExceeded(Exception):
    """Raised when the local daily call counter reaches the hard limit."""


class AlphaVantageError(Exception):
    """Raised for API-level errors returned inside a 200 response body."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AlphaVantageNewsClient:
    """
    Fetches news and per-ticker sentiment from Alpha Vantage NEWS_SENTIMENT.

    Parameters
    ----------
    rate_state_path:
        Override the JSON file used to track daily call counts.
        Defaults to ``config/av_rate_state.json`` inside the package root.
        Pass a ``tmp_path`` / ``Path`` in tests to avoid touching production state.
    """

    def __init__(self, rate_state_path: Path | None = None) -> None:
        self._api_key = settings.ALPHAVANTAGE_API_KEY
        self._rate_state_path = rate_state_path or _DEFAULT_RATE_STATE_PATH
        self._recent_call_times: list[float] = []
        logger.info("av_client.init", rate_state=str(self._rate_state_path))

    # ------------------------------------------------------------------
    # Rate-limit tracking
    # ------------------------------------------------------------------

    def _load_rate_state(self) -> dict[str, Any]:
        """Return persisted state or a blank slate if the file is absent."""
        try:
            return json.loads(self._rate_state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"date": "", "count": 0}

    def _save_rate_state(self, state: dict[str, Any]) -> None:
        self._rate_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._rate_state_path.write_text(json.dumps(state))

    def _check_and_increment(self) -> int:
        """
        Load today's call count, enforce limits, increment, persist.

        Returns
        -------
        int
            The count *before* this call (i.e. number of prior calls today).

        Raises
        ------
        RateLimitExceeded
            If the count is already at or above ``_DAILY_HARD_LIMIT``.
        """
        state = self._load_rate_state()
        today = date.today().isoformat()
        if state.get("date") != today:
            state = {"date": today, "count": 0}

        count = int(state["count"])

        if count >= _DAILY_HARD_LIMIT:
            raise RateLimitExceeded(
                f"Alpha Vantage daily hard limit reached "
                f"({count} calls today, limit {_DAILY_HARD_LIMIT}). "
                f"Limit resets at midnight."
            )

        if count >= _DAILY_WARN_THRESHOLD:
            logger.warning(
                "av_client.rate_limit.approaching",
                calls_today=count,
                hard_limit=_DAILY_HARD_LIMIT,
            )

        state["count"] = count + 1
        self._save_rate_state(state)
        logger.debug("av_client.rate_counter", calls_today=count + 1)
        return count

    # ------------------------------------------------------------------
    # Per-minute rate guard
    # ------------------------------------------------------------------

    def _enforce_per_minute_limit(self, max_per_minute: int = 5) -> None:
        """
        Block until making another call would not exceed *max_per_minute* within
        any rolling 60-second window.  Uses ``time.monotonic`` timestamps stored
        in ``_recent_call_times``.

        On a typical run the free-tier pattern is 1 call per pipeline run, so
        this guard should never trigger — it is a safety net only.
        """
        now = time.monotonic()
        # Prune entries older than 60 seconds
        self._recent_call_times = [t for t in self._recent_call_times if now - t < 60]

        if len(self._recent_call_times) >= max_per_minute:
            # Sleep until the oldest entry falls outside the 60-second window
            sleep_duration = 60.0 - (now - self._recent_call_times[0])
            if sleep_duration > 0:
                logger.debug(
                    "av_client.per_minute_limit.sleeping",
                    sleep_seconds=round(sleep_duration, 1),
                )
                time.sleep(sleep_duration)
                # Prune again after waking
                now = time.monotonic()
                self._recent_call_times = [
                    t for t in self._recent_call_times if now - t < 60
                ]

        self._recent_call_times.append(now)

    # ------------------------------------------------------------------
    # Daily call count (public, for external budget checks)
    # ------------------------------------------------------------------

    def get_daily_call_count(self) -> int:
        """
        Return the number of AV API calls made today (resets at midnight).

        Reads from the same rate-state file used by ``_check_and_increment``.
        Returns 0 if the file is missing or was last updated on a different day.
        """
        state = self._load_rate_state()
        if state.get("date") != date.today().isoformat():
            return 0
        return int(state.get("count", 0))

    # ------------------------------------------------------------------
    # News fetch
    # ------------------------------------------------------------------

    def fetch_news(
        self,
        tickers: list[str],
        hours_back: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Fetch and parse NEWS_SENTIMENT articles for *tickers*.

        Parameters
        ----------
        tickers:
            Symbols to query, e.g. ``["AAPL", "MSFT"]``.
        hours_back:
            Look-back window in hours (relative to now UTC).

        Returns
        -------
        list[dict]
            One dict per (article × ticker) pair.
            Keys:
              ticker, title, summary, source,
              published_at (datetime UTC),
              relevance_score (float),
              av_sentiment_score (float),
              av_sentiment_label (str),
              headline_hash (str — SHA-256 of title).

        Raises
        ------
        RateLimitExceeded
            Daily call counter at or above hard limit.
        AlphaVantageError
            API returned a 200 with an "Information" or "Note" key
            (AV's non-standard way of signalling rate or auth errors).
        requests.HTTPError
            Non-200 HTTP response.
        """
        ticker_set = set(tickers)
        time_from = (
            datetime.now(tz=timezone.utc) - timedelta(hours=hours_back)
        ).strftime("%Y%m%dT%H%M")

        logger.info(
            "av_client.fetch_news.start",
            tickers=tickers,
            hours_back=hours_back,
            time_from=time_from,
        )

        self._enforce_per_minute_limit()
        self._check_and_increment()

        params: dict[str, str] = {
            "function": "NEWS_SENTIMENT",
            "tickers":  ",".join(tickers),
            "time_from": time_from,
            "limit":    "50",
            "apikey":   self._api_key,
        }

        response = requests.get(_BASE_URL, params=params, timeout=15)
        response.raise_for_status()

        body: dict[str, Any] = response.json()
        self._check_av_body_errors(body)

        feed: list[dict[str, Any]] = body.get("feed", [])
        rows = self._parse_feed(feed, ticker_set)

        logger.info(
            "av_client.fetch_news.done",
            tickers=tickers,
            articles_in_feed=len(feed),
            records_returned=len(rows),
        )
        return rows

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_av_body_errors(body: dict[str, Any]) -> None:
        """
        Alpha Vantage returns HTTP 200 even for errors.
        Detect the two known error shapes and raise ``AlphaVantageError``.
        """
        for key in ("Information", "Note"):
            if key in body:
                raise AlphaVantageError(
                    f"Alpha Vantage API error ({key}): {body[key]}"
                )

    @staticmethod
    def _parse_time_published(raw: str) -> datetime:
        """Parse AV's ``YYYYMMDDTHHMMSS`` string to a UTC-aware datetime."""
        # AV response uses HHMMSS; request param uses HHMM — handle both
        fmt = "%Y%m%dT%H%M%S" if len(raw) >= 15 else "%Y%m%dT%H%M"
        return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)

    @staticmethod
    def _parse_feed(
        feed: list[dict[str, Any]],
        ticker_set: set[str],
    ) -> list[dict[str, Any]]:
        """
        Flatten feed → one record per (article × requested-ticker) pair.
        Deduplicates by (headline_hash, ticker) within the batch.
        """
        seen: set[tuple[str, str]] = set()
        rows: list[dict[str, Any]] = []

        for article in feed:
            title = article.get("title", "").strip()
            if not title:
                continue

            headline_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()
            published_at  = AlphaVantageNewsClient._parse_time_published(
                article.get("time_published", "19700101T000000")
            )
            summary = article.get("summary") or None
            source  = article.get("source") or None

            for ts in article.get("ticker_sentiment", []):
                ticker = ts.get("ticker", "")
                if ticker not in ticker_set:
                    continue

                key = (headline_hash, ticker)
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    {
                        "ticker":             ticker,
                        "title":              title,
                        "summary":            summary,
                        "source":             source,
                        "published_at":       published_at,
                        "relevance_score":    float(ts.get("relevance_score", 0)),
                        "av_sentiment_score": float(ts.get("ticker_sentiment_score", 0)),
                        "av_sentiment_label": ts.get("ticker_sentiment_label", ""),
                        "headline_hash":      headline_hash,
                    }
                )

        return rows

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def is_market_hours(self) -> bool:
        """
        Return ``True`` if the current time is within NYSE market hours
        (09:30–16:00 ET, Monday–Friday, no holiday adjustment).

        Uses ``America/New_York`` via ``zoneinfo`` so ET/EDT transitions are
        handled automatically.
        """
        now_et = datetime.now(tz=_ET)

        if now_et.weekday() >= 5:          # Saturday=5, Sunday=6
            return False

        open_minutes  = _MARKET_OPEN[0]  * 60 + _MARKET_OPEN[1]
        close_minutes = _MARKET_CLOSE[0] * 60 + _MARKET_CLOSE[1]
        now_minutes   = now_et.hour * 60 + now_et.minute

        return open_minutes <= now_minutes < close_minutes
