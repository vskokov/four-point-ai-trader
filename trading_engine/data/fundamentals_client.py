"""
Fundamentals data client using yfinance.

Provides market-cap and other company fundamentals for ticker screening.
Used by the sentiment pipeline to rank tickers by size so the AV API quota
(top N by market cap via AV) and the Alpaca fallback (remaining tickers)
are allocated optimally.

Planned additions (TODO):
  - earnings_dates(tickers)    — pause/reduce positions around earnings events
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


class FundamentalsClient:
    """
    Fetches company fundamentals via yfinance.

    Results are cached in-process for 24 hours to avoid repeated network
    calls across sentiment pipeline runs (which fire every 25–35 minutes).
    """

    def __init__(self) -> None:
        # ticker → {"value": float, "fetched_at": datetime}
        self._cache: dict[str, dict[str, Any]] = {}
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
