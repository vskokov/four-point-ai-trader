"""LLM-based news sentiment signal via local Ollama.

Scores recent news headlines for a ticker using a locally-hosted Gemma
model (gemma4:e4b by default), producing a directional signal (−1/0/+1)
with associated confidence for the trading pipeline.

Design constraints
------------------
* The LLM receives ONLY news text — never prices, OHLCV data, or trading
  history.
* Total prompt size is capped at _MAX_HEADLINES articles (highest relevance
  first) to stay within context limits.
* Each Ollama call has a 60-second timeout; on timeout the module returns a
  neutral (direction=0) result rather than crashing.
* Retry logic: if the LLM returns malformed JSON the prompt is tightened and
  the call is retried once; on a second failure a safe default is returned.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import ollama

import trading_engine.config.settings as settings
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_age(published_at: datetime, now: datetime | None = None) -> str:
    """
    Format the age of *published_at* as a human-readable bracket string.

    Parameters
    ----------
    published_at:
        UTC-aware publication datetime.
    now:
        Reference instant (UTC).  Defaults to ``datetime.now(tz=timezone.utc)``.
        Pass a fixed value in tests for deterministic output.

    Examples
    --------
    >>> _human_age(pub, now=pub + timedelta(minutes=23))
    '[23 minutes ago]'
    >>> _human_age(pub, now=pub + timedelta(hours=2, minutes=50))
    '[2 hours 50 minutes ago]'
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    delta = now - published_at
    total_seconds = delta.total_seconds()

    if total_seconds <= 0:
        return "[just now]"

    if total_seconds < 3600:                        # under 1 hour
        m = int(total_seconds // 60)
        unit = "minute" if m == 1 else "minutes"
        return f"[{m} {unit} ago]"

    if total_seconds < 86400:                       # 1 hour – 24 hours
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        h_str = "1 hour" if h == 1 else f"{h} hours"
        if m == 0:
            return f"[{h_str} ago]"
        m_str = "1 minute" if m == 1 else f"{m} minutes"
        return f"[{h_str} {m_str} ago]"

    # 24+ hours
    d = int(total_seconds // 86400)
    h = int((total_seconds % 86400) // 3600)
    d_str = "1 day" if d == 1 else f"{d} days"
    if h == 0:
        return f"[{d_str} ago]"
    h_str = "1 hour" if h == 1 else f"{h} hours"
    return f"[{d_str} {h_str} ago]"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_HEADLINES = 50
_REQUIRED_KEYS = frozenset(
    {"ticker", "direction", "confidence", "horizon", "key_drivers", "reasoning"}
)
_VALID_DIRECTIONS: frozenset[int] = frozenset({-1, 0, 1})
_VALID_HORIZONS: frozenset[str] = frozenset({"4h", "8h", "1d"})
_ET = ZoneInfo("America/New_York")

# NYSE core session boundaries (minutes since midnight ET)
_OPEN_MINUTES  = 9 * 60 + 30   # 09:30
_CLOSE_MINUTES = 16 * 60        # 16:00


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMSentimentSignal:
    """
    Generates a directional sentiment signal from news headlines via a local
    LLM (Ollama / gemma4:e4b).

    Parameters
    ----------
    model:
        Ollama model tag.  Defaults to ``settings.OLLAMA_MODEL``.
    host:
        Ollama server URL.  Defaults to ``settings.OLLAMA_HOST``.
    hours_back:
        Look-back window passed to ``AlphaVantageNewsClient.fetch_news``.
    min_relevance:
        Minimum Alpha Vantage ``relevance_score`` for a headline to be
        included in the LLM prompt.  Articles below this threshold are
        discarded before scoring.
    """

    def __init__(
        self,
        model: str = settings.OLLAMA_MODEL,
        host: str = settings.OLLAMA_HOST,
        hours_back: int = 2,
        min_relevance: float = 0.3,
    ) -> None:
        self.model = model
        self.hours_back = hours_back
        self.min_relevance = min_relevance
        # 60-second timeout is enforced at the httpx layer inside the client.
        self._client = ollama.Client(host=host, timeout=60)
        # In-process cache: (ticker, headline_hash) pairs already scored this
        # session.  Keyed per-ticker so a shared article can be scored once per
        # ticker rather than being suppressed after the first ticker processes it.
        self._seen_hashes: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, ticker: str, headlines: list[dict[str, Any]]) -> str:
        """
        Build a strict, JSON-only prompt from a de-duplicated headline list.

        Headlines are de-duplicated by ``headline_hash``, sorted by
        ``relevance_score`` descending, and capped at ``_MAX_HEADLINES``.
        Each article body is truncated to 300 characters.

        Parameters
        ----------
        ticker:
            Equity symbol being scored.
        headlines:
            Dicts from ``AlphaVantageNewsClient.fetch_news`` (must include
            ``title``, optionally ``summary`` and ``headline_hash``).
        """
        # De-duplicate by headline_hash (or title as fallback) and sort.
        seen_hashes: set[str] = set()
        unique: list[dict[str, Any]] = []
        for h in sorted(
            headlines,
            key=lambda x: x.get("relevance_score", 0.0),
            reverse=True,
        ):
            hsh = h.get("headline_hash") or h.get("title", "")
            if hsh not in seen_hashes:
                seen_hashes.add(hsh)
                unique.append(h)
            if len(unique) >= _MAX_HEADLINES:
                break

        numbered: list[str] = []
        for i, h in enumerate(unique, 1):
            title   = (h.get("title") or "").strip()
            summary = (h.get("summary") or "").strip()
            body    = f"{title}. {summary}" if summary else title
            pub     = h.get("published_at")
            age_str = _human_age(pub) if pub is not None else "[unknown time]"
            numbered.append(f"{i}. {age_str} {body[:300]}")

        headline_block = "\n".join(numbered)

        return (
            "You are a quantitative analyst. "
            "Respond ONLY with valid JSON. No preamble, no markdown, no explanation.\n\n"
            f"Ticker: {ticker}\n\n"
            "News headlines (analyze ONLY these — ignore any prior knowledge "
            "about this company, its products, management, or financials):\n"
            "Each headline is prefixed with its age (e.g. [2 hours 30 minutes ago]). "
            "Weight recent headlines more heavily — news older than 6 hours may already "
            "be priced in.\n"
            f"{headline_block}\n\n"
            "Return exactly this JSON object — no other text:\n"
            "{\n"
            f'  "ticker": "{ticker}",\n'
            '  "direction": <integer -1 (bearish) | 0 (neutral) | 1 (bullish)>,\n'
            '  "confidence": <float 0.0–1.0>,\n'
            '  "horizon": "<4h | 8h | 1d>",\n'
            '  "key_drivers": ["<driver1>", "<driver2>"],\n'
            '  "reasoning": "<1–2 sentences only>"\n'
            "}"
        )

    # ------------------------------------------------------------------
    # Response parsing & validation
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove leading/trailing markdown code fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop opening fence (``` or ```json) and closing fence
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        return text

    def _parse_response(self, content: str) -> dict[str, Any] | None:
        """
        Parse and validate an Ollama response string.

        Returns
        -------
        dict if valid, None if schema validation fails.
        """
        text = self._strip_fences(content)
        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("llm.parse.json_error", preview=text[:120])
            return None

        # Required keys
        if not _REQUIRED_KEYS.issubset(data.keys()):
            missing = _REQUIRED_KEYS - data.keys()
            logger.warning("llm.parse.missing_keys", missing=list(missing))
            return None

        # direction: must be in {-1, 0, 1}
        try:
            direction = int(data["direction"])
        except (TypeError, ValueError):
            logger.warning("llm.parse.bad_direction", value=data.get("direction"))
            return None
        if direction not in _VALID_DIRECTIONS:
            logger.warning("llm.parse.bad_direction", value=direction)
            return None

        # confidence: float in [0, 1]
        try:
            confidence = float(data["confidence"])
        except (TypeError, ValueError):
            logger.warning("llm.parse.bad_confidence", value=data.get("confidence"))
            return None
        if not (0.0 <= confidence <= 1.0):
            logger.warning("llm.parse.confidence_out_of_range", value=confidence)
            return None

        # horizon
        if data.get("horizon") not in _VALID_HORIZONS:
            logger.warning("llm.parse.bad_horizon", value=data.get("horizon"))
            return None

        # key_drivers: list
        if not isinstance(data.get("key_drivers"), list):
            logger.warning("llm.parse.bad_key_drivers", value=data.get("key_drivers"))
            return None

        data["direction"]   = direction
        data["confidence"]  = confidence
        return data

    # ------------------------------------------------------------------
    # LLM call with retry
    # ------------------------------------------------------------------

    def _error_result(
        self,
        ticker: str,
        source: str,
        headlines: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a safe neutral result when the LLM cannot be used."""
        hashes = [h["headline_hash"] for h in (headlines or []) if "headline_hash" in h]
        return {
            "ticker":           ticker,
            "direction":        0,
            "confidence":       0.0,
            "horizon":          "8h",
            "key_drivers":      [],
            "reasoning":        "",
            "source":           source,
            "fetched_at":       datetime.now(tz=timezone.utc),
            "n_headlines_used": len(headlines) if headlines else 0,
            "headline_hashes":  hashes,
        }

    def _invoke_llm(
        self,
        ticker: str,
        prompt: str,
        headlines: list[dict[str, Any]],
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """
        Call Ollama and parse the response.  Retries once with a stricter
        prompt if the first call returns invalid JSON.  On timeout or a
        second parse failure returns a neutral error result.
        """
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            raw = response.message.content
        except Exception as exc:
            exc_name = type(exc).__name__.lower()
            exc_msg  = str(exc).lower()
            if "timeout" in exc_name or "timeout" in exc_msg:
                logger.warning("llm.timeout", ticker=ticker, model=self.model)
                return self._error_result(ticker, "timeout", headlines)
            logger.error(
                "llm.error",
                ticker=ticker,
                exc_type=type(exc).__name__,
                exc=str(exc)[:200],
            )
            return self._error_result(ticker, "llm_error", headlines)

        parsed = self._parse_response(raw)

        if parsed is None:
            if not strict:
                logger.warning("llm.retry", ticker=ticker, reason="malformed_response")
                strict_prompt = (
                    prompt
                    + "\n\nCRITICAL: Your previous response was not valid JSON. "
                    "Output ONLY the raw JSON object — no markdown, no text before "
                    "or after the braces. Start your response with { and end with }."
                )
                return self._invoke_llm(
                    ticker, strict_prompt, headlines, strict=True
                )
            logger.error("llm.parse_failed_after_retry", ticker=ticker)
            return self._error_result(ticker, "llm_error", headlines)

        hashes = [h["headline_hash"] for h in headlines if "headline_hash" in h]
        parsed.update(
            {
                "ticker":           ticker,
                "fetched_at":       datetime.now(tz=timezone.utc),
                "n_headlines_used": len(headlines),
                "headline_hashes":  hashes,
                "source":           "llm",
            }
        )
        return parsed

    # ------------------------------------------------------------------
    # Public scoring entry point
    # ------------------------------------------------------------------

    def score(
        self,
        ticker: str,
        headlines: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Score a list of news headlines for *ticker*.

        Parameters
        ----------
        ticker:
            Equity symbol.
        headlines:
            Dicts from ``AlphaVantageNewsClient.fetch_news``.

        Returns
        -------
        dict with keys:
            ticker, direction (int), confidence (float), horizon (str),
            key_drivers (list), reasoning (str), source (str),
            fetched_at (datetime), n_headlines_used (int),
            headline_hashes (list[str]).
        """
        relevant = [
            h for h in headlines
            if float(h.get("relevance_score", 0.0)) >= self.min_relevance
        ]

        if not relevant:
            logger.info("llm.score.no_relevant_headlines", ticker=ticker)
            return self._error_result(ticker, "no_data")

        # Sort by relevance descending; cap at _MAX_HEADLINES
        relevant.sort(key=lambda h: h.get("relevance_score", 0.0), reverse=True)
        relevant = relevant[:_MAX_HEADLINES]

        prompt = self._build_prompt(ticker, relevant)

        logger.info(
            "llm.score.start",
            ticker=ticker,
            n_headlines=len(relevant),
            model=self.model,
        )

        result = self._invoke_llm(ticker, prompt, relevant)

        logger.info(
            "llm.score.done",
            ticker=ticker,
            direction=result["direction"],
            confidence=result["confidence"],
            source=result["source"],
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        tickers: list[str],
        av_client: Any,
        storage: Any,
        *,
        av_tickers: list[str] | None = None,
        alpaca_client: Any = None,
    ) -> list[dict[str, Any]]:
        """
        End-to-end pipeline: fetch → deduplicate → score → persist.

        Parameters
        ----------
        tickers:
            Equity symbols to process (all tickers, used for scoring loop).
        av_client:
            ``AlphaVantageNewsClient`` instance for news fetching.
        storage:
            ``Storage`` instance for persistence.
        av_tickers:
            Subset of *tickers* to fetch via AV.  When ``None`` (default),
            all tickers are fetched via AV (original behaviour).  When
            provided, remaining tickers are fetched via *alpaca_client*.
        alpaca_client:
            ``AlpacaNewsClient`` instance used as fallback for tickers not
            in *av_tickers*.  Ignored when *av_tickers* is ``None``.

        Returns
        -------
        list of per-ticker result dicts.
        """
        results: list[dict[str, Any]] = []

        # Determine which tickers use AV vs Alpaca fallback.
        _av_set = set(av_tickers) if av_tickers is not None else set(tickers)
        _av_fetch_list = [t for t in tickers if t in _av_set] if av_tickers is not None else tickers
        _alpaca_fetch_list = [t for t in tickers if t not in _av_set]

        # 1. Single AV call for av_fetch_list — 1 API call per pipeline run.
        all_articles: list[dict[str, Any]] = av_client.fetch_news(
            _av_fetch_list, hours_back=self.hours_back
        ) if _av_fetch_list else []

        # 1b. Alpaca fallback for remaining tickers.
        if alpaca_client and _alpaca_fetch_list:
            logger.info(
                "llm.pipeline.alpaca_fallback",
                n_tickers=len(_alpaca_fetch_list),
            )
            try:
                alpaca_articles = alpaca_client.fetch_news(
                    _alpaca_fetch_list, hours_back=self.hours_back
                )
                # Inject fields expected by score() and condensed_headlines.
                for a in alpaca_articles:
                    a.setdefault("relevance_score", 1.0)
                    a.setdefault("av_sentiment_label", "")
                    a.setdefault("av_sentiment_score", None)
                all_articles.extend(alpaca_articles)
            except Exception as exc:
                logger.warning(
                    "llm.pipeline.alpaca_fallback_failed",
                    tickers=_alpaca_fetch_list,
                    error=str(exc)[:120],
                )

        # Group articles by ticker (each article dict has a "ticker" field).
        articles_by_ticker: dict[str, list[dict[str, Any]]] = {t: [] for t in tickers}
        for _art in all_articles:
            t = _art.get("ticker", "")
            if t in articles_by_ticker:
                articles_by_ticker[t].append(_art)

        for ticker in tickers:
            articles = articles_by_ticker.get(ticker, [])
            n_fetched = len(articles)

            # 2. Deduplicate against in-process cache.
            # Key is (ticker, headline_hash) so a shared article can be scored
            # once per ticker; processing AAPL does not suppress it for MSFT.
            new_articles: list[dict[str, Any]] = []
            n_skipped = 0
            for article in articles:
                h = article.get("headline_hash", "")
                if h and (ticker, h) in self._seen_hashes:
                    n_skipped += 1
                else:
                    new_articles.append(article)

            n_new = len(new_articles)
            logger.info(
                "llm.pipeline.fetch",
                ticker=ticker,
                n_fetched=n_fetched,
                n_new=n_new,
                n_skipped_duplicate=n_skipped,
            )

            # 3. Score
            score_result = self.score(ticker, new_articles)
            direction  = score_result["direction"]
            confidence = score_result["confidence"]

            # 4. Persist news rows (storage deduplicates via UNIQUE(ticker, headline_hash))
            if new_articles:
                news_rows = [
                    {
                        "ticker":                a["ticker"],
                        "title":                 a["title"],
                        "summary":               a.get("summary"),
                        "source":                a.get("source"),
                        "sentiment_score":        a.get("av_sentiment_score"),
                        "sentiment_confidence":   confidence,
                        "llm_direction":          direction,
                        "fetched_at":             a.get("published_at"),
                    }
                    for a in new_articles
                ]
                storage.insert_news(news_rows)

            # Update in-process dedup cache with (ticker, headline_hash) pairs
            for a in new_articles:
                h = a.get("headline_hash", "")
                if h:
                    self._seen_hashes.add((ticker, h))

            # 5. Signal log: signed confidence = direction × confidence
            # Store a condensed snapshot of contributing headlines so the
            # dashboard can display them alongside each trade decision.
            condensed_headlines = [
                {
                    "title":              a.get("title", ""),
                    "source":             a.get("source"),
                    "published_at":       a["published_at"].isoformat()
                                          if hasattr(a.get("published_at"), "isoformat")
                                          else str(a.get("published_at", "")),
                    "av_sentiment_label": a.get("av_sentiment_label", ""),
                    "av_sentiment_score": a.get("av_sentiment_score"),
                    "relevance_score":    a.get("relevance_score"),
                }
                for a in new_articles
            ]
            timestamp = datetime.now(tz=timezone.utc)
            storage.insert_signal([
                {
                    "time":        timestamp,
                    "ticker":      ticker,
                    "signal_name": "llm_sentiment",
                    "value":       float(direction) * confidence,
                    "metadata": {
                        "direction":              direction,
                        "confidence":             confidence,
                        "horizon":                score_result.get("horizon", "8h"),
                        "n_headlines":            score_result.get("n_headlines_used", 0),
                        "source":                 score_result.get("source", "llm"),
                        "contributing_headlines": condensed_headlines,
                    },
                }
            ])

            logger.info(
                "llm.pipeline.result",
                ticker=ticker,
                direction=direction,
                confidence=confidence,
                n_new=n_new,
                source=score_result.get("source"),
            )

            results.append(score_result)

        return results

    # ------------------------------------------------------------------
    # Market-hours guard
    # ------------------------------------------------------------------

    def run_if_market_hours(
        self,
        tickers: list[str],
        av_client: Any,
        storage: Any,
        pre_post_minutes: int = 30,
    ) -> list[dict[str, Any]] | None:
        """
        Call ``run_pipeline`` only during NYSE hours ± *pre_post_minutes*.

        .. deprecated::
            Market-hours gating is now handled by APScheduler cron triggers in
            ``TradingEngine.run()``.  The two cron windows (07:00–10:29 ET every
            25 min, 10:30–16:30 ET every 35 min) replace the need for this guard.
            This method is retained for backward compatibility with existing tests.

        The extended window is:
            (09:30 − pre_post_minutes) ET  to  (16:00 + pre_post_minutes) ET
        on weekdays.  ``AlphaVantageNewsClient.is_market_hours()`` covers only
        the core session; this method provides the wider scheduling window.

        Returns
        -------
        list of result dicts, or *None* if outside the active window.
        """
        now_et = datetime.now(tz=_ET)

        if now_et.weekday() >= 5:       # Saturday = 5, Sunday = 6
            logger.debug("llm.run_if_market_hours.skip", reason="weekend")
            return None

        now_min   = now_et.hour * 60 + now_et.minute
        open_min  = _OPEN_MINUTES  - pre_post_minutes
        close_min = _CLOSE_MINUTES + pre_post_minutes

        if not (open_min <= now_min < close_min):
            logger.debug(
                "llm.run_if_market_hours.skip",
                reason="outside_extended_hours",
                now_et=now_et.strftime("%H:%M"),
            )
            return None

        return self.run_pipeline(tickers, av_client, storage)
