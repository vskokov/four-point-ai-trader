"""
Fundamentals data client using yfinance.

Provides market-cap and other company fundamentals for ticker screening,
and upcoming earnings dates for the earnings-date risk guard in bar_handler.

Planned additions (TODO):
  - analyst_recommendations()  — additional orthogonal sentiment signal
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

_CACHE_TTL_SECONDS = 86_400   # 24 hours
_FETCH_WORKERS = 10           # parallel yfinance requests
_VIX_CACHE_TTL_SECONDS = 300  # 5 minutes


class FundamentalsClient:
    """
    Fetches company fundamentals via yfinance.

    Results are cached in-process for 24 hours to avoid repeated network
    calls across sentiment pipeline runs (which fire every 25–35 minutes).
    """

    def __init__(self) -> None:
        # ticker → {"value": float, "fetched_at": datetime}
        self._cache: dict[str, dict[str, Any]] = {}
        # ticker → {"value": datetime | None, "fetched_at": datetime}
        self._earnings_cache: dict[str, dict[str, Any]] = {}
        # ticker → {"value": int (-1/0/1), "fetched_at": datetime}
        self._recs_cache: dict[str, dict[str, Any]] = {}
        logger.info("fundamentals_client.init")

    # ------------------------------------------------------------------
    # Market cap
    # ------------------------------------------------------------------

    def get_market_caps(self, tickers: list[str]) -> dict[str, float]:
        """
        Return market capitalisation (USD) for each ticker.

        Tickers with no data return 0.0 so they sort to the bottom of any
        cap-weighted ranking without causing errors.  Results are cached
        for 24 hours; only uncached or stale tickers trigger network calls.

        Parameters
        ----------
        tickers:
            Equity symbols, e.g. ``["AAPL", "MSFT", "GILD"]``.

        Returns
        -------
        dict[str, float]
            ``{ticker: market_cap_usd}``.
        """
        now = datetime.now(tz=timezone.utc)
        caps: dict[str, float] = {}
        to_fetch: list[str] = []

        for t in tickers:
            entry = self._cache.get(t)
            if entry and (now - entry["fetched_at"]).total_seconds() < _CACHE_TTL_SECONDS:
                caps[t] = entry["value"]
            else:
                to_fetch.append(t)

        if to_fetch:
            logger.info(
                "fundamentals_client.market_cap.fetch",
                n_tickers=len(to_fetch),
                tickers=to_fetch,
            )
            fetched = self._fetch_market_caps_parallel(to_fetch)
            for t, cap in fetched.items():
                caps[t] = cap
                self._cache[t] = {"value": cap, "fetched_at": now}

        logger.info(
            "fundamentals_client.market_cap.done",
            n_total=len(tickers),
            n_fetched=len(to_fetch),
            n_cached=len(tickers) - len(to_fetch),
        )
        return caps

    def _fetch_market_caps_parallel(
        self, tickers: list[str]
    ) -> dict[str, float]:
        """Fetch market caps for *tickers* in parallel using a thread pool."""
        result: dict[str, float] = {}

        def _fetch_one(ticker: str) -> tuple[str, float]:
            try:
                info = yf.Ticker(ticker).info
                cap = float(info.get("marketCap") or 0.0)
            except Exception as exc:
                logger.warning(
                    "fundamentals_client.market_cap.error",
                    ticker=ticker,
                    exc=str(exc)[:120],
                )
                cap = 0.0
            return ticker, cap

        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, cap = future.result()
                result[ticker] = cap

        return result

    # ------------------------------------------------------------------
    # Earnings dates
    # ------------------------------------------------------------------

    def get_earnings_dates(self, tickers: list[str]) -> dict[str, datetime | None]:
        """
        Return the next upcoming earnings date (UTC) for each ticker, or
        ``None`` if unknown.

        Results are cached in-process for 24 hours.  Only tickers with a
        stale or absent cache entry trigger network calls.

        Parameters
        ----------
        tickers:
            Equity symbols, e.g. ``["AAPL", "NVDA"]``.

        Returns
        -------
        dict[str, datetime | None]
            ``{ticker: next_earnings_utc}`` — ``None`` when no upcoming
            earnings date is available.
        """
        if not tickers:
            return {}

        now = datetime.now(tz=timezone.utc)
        result: dict[str, datetime | None] = {}
        to_fetch: list[str] = []

        for t in tickers:
            entry = self._earnings_cache.get(t)
            if entry and (now - entry["fetched_at"]).total_seconds() < _CACHE_TTL_SECONDS:
                result[t] = entry["value"]
            else:
                to_fetch.append(t)

        if to_fetch:
            logger.info(
                "fundamentals_client.earnings_date.fetch",
                n_tickers=len(to_fetch),
                tickers=to_fetch,
            )
            fetched = self._fetch_earnings_dates_parallel(to_fetch)
            for t, dt in fetched.items():
                result[t] = dt
                self._earnings_cache[t] = {"value": dt, "fetched_at": now}

        return result

    def _fetch_earnings_dates_parallel(
        self, tickers: list[str]
    ) -> dict[str, datetime | None]:
        """Fetch the next upcoming earnings date for each ticker in parallel."""
        result: dict[str, datetime | None] = {}

        def _fetch_one(ticker: str) -> tuple[str, datetime | None]:
            try:
                cal = yf.Ticker(ticker).calendar
                if not cal:
                    return ticker, None

                raw = cal.get("Earnings Date")
                if raw is None:
                    return ticker, None

                # Normalise to a flat list of objects with a .date() method.
                if not isinstance(raw, list):
                    raw = [raw]

                today = datetime.now(tz=timezone.utc).date()
                future_dates: list[datetime] = []
                for d in raw:
                    try:
                        # yfinance returns pd.Timestamp (tz-naive or tz-aware).
                        # Convert to UTC-aware Python datetime.
                        if hasattr(d, "tzinfo"):
                            # pd.Timestamp — attach UTC if tz-naive
                            if d.tzinfo is None:
                                dt = d.tz_localize("UTC").to_pydatetime()
                            else:
                                dt = d.tz_convert("UTC").to_pydatetime()
                        else:
                            # Bare date / string fallback
                            dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                        if dt.date() >= today:
                            future_dates.append(dt)
                    except Exception:
                        continue

                if not future_dates:
                    return ticker, None
                return ticker, min(future_dates)

            except Exception as exc:
                logger.warning(
                    "fundamentals_client.earnings_date.error",
                    ticker=ticker,
                    exc=str(exc)[:120],
                )
                return ticker, None

        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, dt = future.result()
                result[ticker] = dt

        return result

    # ------------------------------------------------------------------
    # Analyst recommendations
    # ------------------------------------------------------------------

    # Map yfinance recommendationKey values → directional signal.
    _REC_MAP: dict[str, int] = {
        "strongBuy":    1,
        "buy":          1,
        "outperform":   1,
        "overweight":   1,
        "hold":         0,
        "neutral":      0,
        "marketperform": 0,
        "sell":         -1,
        "strongSell":   -1,
        "underperform": -1,
        "underweight":  -1,
    }

    def get_analyst_recommendations(self, tickers: list[str]) -> dict[str, int]:
        """
        Return the consensus analyst recommendation as a directional signal
        for each ticker.

        Maps ``yf.Ticker(t).info["recommendationKey"]`` to:
        - ``+1`` — buy / strong_buy / outperform / overweight
        - ``0``  — hold / neutral / marketperform, or unknown
        - ``-1`` — sell / strong_sell / underperform / underweight

        Results are cached in-process for 24 hours (analyst ratings update
        at most weekly).

        Parameters
        ----------
        tickers:
            Equity symbols, e.g. ``["AAPL", "MSFT"]``.

        Returns
        -------
        dict[str, int]
            ``{ticker: signal}`` — defaults to ``0`` when data is unavailable.
        """
        if not tickers:
            return {}

        now = datetime.now(tz=timezone.utc)
        result: dict[str, int] = {}
        to_fetch: list[str] = []

        for t in tickers:
            entry = self._recs_cache.get(t)
            if entry and (now - entry["fetched_at"]).total_seconds() < _CACHE_TTL_SECONDS:
                result[t] = entry["value"]
            else:
                to_fetch.append(t)

        if to_fetch:
            logger.info(
                "fundamentals_client.analyst_recs.fetch",
                n_tickers=len(to_fetch),
                tickers=to_fetch,
            )
            fetched = self._fetch_analyst_recs_parallel(to_fetch)
            for t, signal in fetched.items():
                result[t] = signal
                self._recs_cache[t] = {"value": signal, "fetched_at": now}

        return result

    # ------------------------------------------------------------------
    # VIX
    # ------------------------------------------------------------------

    def get_vix(self) -> float | None:
        """
        Fetch the current VIX level from yfinance.  Returns ``None`` on failure
        (fail open — a data outage must not block order submission).

        Results are cached for 5 minutes (``_VIX_CACHE_TTL_SECONDS``).
        The cache is stored as ``self._vix_cache`` and initialised lazily on the
        first call so no ``__init__`` change is needed.
        """
        now = datetime.now(timezone.utc)
        cached = getattr(self, "_vix_cache", None)
        if cached and (now - cached["fetched_at"]).total_seconds() < _VIX_CACHE_TTL_SECONDS:
            return cached["value"]
        try:
            vix = yf.Ticker("^VIX").fast_info["last_price"]
            value: float | None = float(vix) if vix is not None else None
        except Exception as exc:
            logger.warning("fundamentals_client.vix.error", exc=str(exc)[:120])
            value = None
        self._vix_cache: dict[str, Any] = {"value": value, "fetched_at": now}
        return value

    def _fetch_analyst_recs_parallel(
        self, tickers: list[str]
    ) -> dict[str, int]:
        """Fetch analyst recommendation signals for *tickers* in parallel."""
        result: dict[str, int] = {}

        def _fetch_one(ticker: str) -> tuple[str, int]:
            try:
                info = yf.Ticker(ticker).info
                key = (info.get("recommendationKey") or "").strip().lower()
                # Normalise casing variants (e.g. "strongBuy" vs "strongbuy")
                for raw_key, signal in self._REC_MAP.items():
                    if key == raw_key.lower():
                        return ticker, signal
                return ticker, 0
            except Exception as exc:
                logger.warning(
                    "fundamentals_client.analyst_recs.error",
                    ticker=ticker,
                    exc=str(exc)[:120],
                )
                return ticker, 0

        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, signal = future.result()
                result[ticker] = signal

        return result
