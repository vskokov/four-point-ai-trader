"""
Alpaca data ingestion clients.

AlpacaMarketData  — historical OHLCV bars, latest quotes, live bar stream,
                    and paper-account info.  Used in the default pipeline.

AlpacaNewsClient  — news articles via Alpaca News API.
                    OPTIONAL: not called by default.  Alpha Vantage
                    (alphavantage_client.py) is the primary news source
                    because it ships pre-computed per-ticker sentiment scores.
                    Instantiate this only for a supplementary / fallback feed.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import (
    NewsRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient

import trading_engine.config.settings as settings
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timeframe map
# ---------------------------------------------------------------------------

_TIMEFRAME_MAP: dict[str, TimeFrame] = {
    "1Min":  TimeFrame.Minute,
    "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
    "1Hour": TimeFrame.Hour,
    "1Day":  TimeFrame.Day,
}

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_AUTH_STATUSES = {401, 403}


def _with_retry(fn: Callable[[], Any], label: str = "") -> Any:
    """
    Call *fn* up to ``_MAX_RETRIES + 1`` times.

    Uses duck-typing on ``status_code`` so it works with both real
    ``alpaca.common.exceptions.APIError`` instances and test fakes.

    - Raises immediately on 401/403 (auth error — retrying won't help).
    - Exponential back-off (1 s, 2 s, 4 s) on 429 (rate limit).
    - Re-raises immediately on any other exception code.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)

            if status in _AUTH_STATUSES:
                logger.error(
                    "alpaca.auth_error",
                    label=label,
                    status=status,
                    message=getattr(exc, "message", str(exc)),
                )
                raise

            if status == 429 and attempt < _MAX_RETRIES:
                wait = 2 ** attempt      # 1 s → 2 s → 4 s
                logger.warning(
                    "alpaca.rate_limit_retry",
                    label=label,
                    attempt=attempt + 1,
                    wait_s=wait,
                )
                time.sleep(wait)
                continue

            # Non-retryable error or retries exhausted
            if attempt == _MAX_RETRIES and status == 429:
                logger.error(
                    "alpaca.retry_exhausted", label=label, attempts=attempt + 1
                )
            raise


# ---------------------------------------------------------------------------
# AlpacaMarketData
# ---------------------------------------------------------------------------

class AlpacaMarketData:
    """
    Fetches market data from Alpaca and persists it via ``Storage``.

    Parameters
    ----------
    storage:
        An initialised ``Storage`` instance.  All OHLCV writes go through it.
    """

    def __init__(self, storage: Any) -> None:  # Any avoids circular import
        self._hist = StockHistoricalDataClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
        )
        self._stream = StockDataStream(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
        )
        self._trading = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True,
        )
        self._storage = storage
        self._clock_cache: dict = {}
        logger.info("alpaca_market.init")

    # ------------------------------------------------------------------
    # Historical OHLCV
    # ------------------------------------------------------------------

    def fetch_historical_ohlcv(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for one or more *tickers* and persist them.

        Parameters
        ----------
        tickers:
            List of symbols, e.g. ``["AAPL", "MSFT"]``.
        start, end:
            Timezone-aware datetimes (UTC recommended).
        timeframe:
            One of ``"1Min"``, ``"5Min"``, ``"1Hour"``, ``"1Day"``.

        Returns
        -------
        pd.DataFrame
            Columns: ticker, open, high, low, close, volume.
            Index: ``time`` (DatetimeTZDtype UTC), sorted ascending.
        """
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}. "
                f"Choose from {list(_TIMEFRAME_MAP)}"
            )

        logger.info(
            "alpaca_market.fetch_historical.start",
            tickers=tickers,
            start=str(start),
            end=str(end),
            timeframe=timeframe,
        )

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=_TIMEFRAME_MAP[timeframe],
            start=start,
            end=end,
        )
        bar_set = _with_retry(
            lambda: self._hist.get_stock_bars(request),
            label="get_stock_bars",
        )

        records: list[dict[str, Any]] = []
        for ticker in tickers:
            bars = bar_set.data.get(ticker, [])
            for bar in bars:
                records.append(
                    {
                        "time":   bar.timestamp,
                        "ticker": ticker,
                        "open":   float(bar.open),
                        "high":   float(bar.high),
                        "low":    float(bar.low),
                        "close":  float(bar.close),
                        "volume": int(bar.volume),
                    }
                )

        if records:
            self._storage.insert_ohlcv(records)

        df = pd.DataFrame(records)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
        else:
            df = pd.DataFrame(
                columns=["ticker", "open", "high", "low", "close", "volume"]
            )

        logger.info(
            "alpaca_market.fetch_historical.done",
            tickers=tickers,
            rows=len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Historical bars (wide form, no DB write)
    # ------------------------------------------------------------------

    def get_historical_bars(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Fetch close prices for *tickers* without writing to storage.

        Returns
        -------
        pd.DataFrame
            Wide-form: index = dates (DatetimeIndex, UTC), columns = tickers,
            values = daily close price.  Missing bars are ``NaN``.
        """
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}. "
                f"Choose from {list(_TIMEFRAME_MAP)}"
            )

        logger.info(
            "alpaca_market.get_historical_bars.start",
            tickers=tickers,
            start=str(start),
            end=str(end),
            timeframe=timeframe,
        )

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=_TIMEFRAME_MAP[timeframe],
            start=start,
            end=end,
        )
        bar_set = _with_retry(
            lambda: self._hist.get_stock_bars(request),
            label="get_historical_bars",
        )

        # Build wide-form dict: date → {ticker: close}
        rows: dict[Any, dict[str, float]] = {}
        for ticker in tickers:
            for bar in bar_set.data.get(ticker, []):
                date_key = pd.Timestamp(bar.timestamp).normalize()
                if date_key not in rows:
                    rows[date_key] = {}
                rows[date_key][ticker] = float(bar.close)

        if not rows:
            df = pd.DataFrame(columns=tickers, dtype=float)
            df.index = pd.DatetimeIndex([], name="time", tz="UTC")
            return df

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index = pd.DatetimeIndex(df.index, tz="UTC", name="time")
        df = df.sort_index().reindex(columns=tickers)

        logger.info(
            "alpaca_market.get_historical_bars.done",
            tickers=tickers,
            rows=len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Latest quote
    # ------------------------------------------------------------------

    def get_latest_quote(self, ticker: str) -> dict[str, Any]:
        """
        Return the latest NBBO quote for *ticker*.

        Returns
        -------
        dict
            Keys: ``bid``, ``ask``, ``mid``, ``timestamp``.
        """
        logger.info("alpaca_market.latest_quote.start", ticker=ticker)

        request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quotes = _with_retry(
            lambda: self._hist.get_stock_latest_quote(request),
            label="get_stock_latest_quote",
        )
        quote = quotes[ticker]

        result = {
            "bid":       float(quote.bid_price),
            "ask":       float(quote.ask_price),
            "mid":       round((float(quote.bid_price) + float(quote.ask_price)) / 2, 4),
            "timestamp": quote.timestamp,
        }

        logger.info(
            "alpaca_market.latest_quote.done",
            ticker=ticker,
            bid=result["bid"],
            ask=result["ask"],
        )
        return result

    # ------------------------------------------------------------------
    # Live bar stream
    # ------------------------------------------------------------------

    def stream_bars(
        self,
        tickers: list[str],
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """
        Start a WebSocket bar stream in a background daemon thread.

        On each incoming bar the record is:
          1. Persisted via ``insert_ohlcv()``.
          2. Forwarded to *callback* as a plain dict (same shape as
             ``fetch_historical_ohlcv`` records).

        The stream runs until the process exits (daemon thread) or
        ``stop_stream()`` is called.

        Parameters
        ----------
        tickers:
            Symbols to subscribe to.
        callback:
            Synchronous callable invoked for every bar.
        """
        async def _handler(bar: Any) -> None:
            row: dict[str, Any] = {
                "time":   bar.timestamp,
                "ticker": bar.symbol,
                "open":   float(bar.open),
                "high":   float(bar.high),
                "low":    float(bar.low),
                "close":  float(bar.close),
                "volume": int(bar.volume),
            }
            self._storage.insert_ohlcv([row])
            callback(row)

        self._stream.subscribe_bars(_handler, *tickers)

        thread = threading.Thread(
            target=self._stream.run,
            daemon=True,
            name="alpaca-stream",
        )
        thread.start()

        logger.info("alpaca_market.stream.started", tickers=tickers)

    def stop_stream(self) -> None:
        """Stop the live bar stream."""
        self._stream.stop()
        logger.info("alpaca_market.stream.stopped")

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict[str, Any]:
        """
        Return paper-account summary.

        Returns
        -------
        dict
            Keys: ``equity``, ``cash``, ``buying_power``, ``portfolio_value``.
            All values are ``float``.
        """
        logger.info("alpaca_market.account.start")

        account = _with_retry(
            self._trading.get_account,
            label="get_account",
        )

        result = {
            "equity":          float(account.equity),
            "cash":            float(account.cash),
            "buying_power":    float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }

        logger.info("alpaca_market.account.done", **result)
        return result

    # ------------------------------------------------------------------
    # Market clock
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """
        Return *True* if the exchange is currently open.

        Calls Alpaca's ``get_clock()`` API, which accounts for weekends,
        holidays, and early closes.  Result is cached for 60 seconds to
        avoid excessive API calls from high-frequency bar_handler invocations.
        """
        now_mono = time.monotonic()
        if self._clock_cache and (now_mono - self._clock_cache["cached_at"]) < 60.0:
            return self._clock_cache["is_open"]

        clock = _with_retry(self._trading.get_clock, label="get_clock")
        is_open: bool = bool(clock.is_open)

        if not is_open:
            logger.debug(
                "alpaca_market.clock.closed",
                next_open=str(clock.next_open),
                next_close=str(clock.next_close),
            )

        self._clock_cache = {"is_open": is_open, "cached_at": now_mono}
        return is_open

    def get_market_clock(self) -> dict[str, Any]:
        """
        Return detailed market clock information (always fetches fresh data).

        Returns
        -------
        dict
            Keys: ``is_open`` (bool), ``next_open`` (datetime),
            ``next_close`` (datetime), ``timestamp`` (datetime).
        """
        clock = _with_retry(self._trading.get_clock, label="get_clock")
        return {
            "is_open":    bool(clock.is_open),
            "next_open":  clock.next_open,
            "next_close": clock.next_close,
            "timestamp":  clock.timestamp,
        }


# ---------------------------------------------------------------------------
# AlpacaNewsClient  — OPTIONAL, not in the default pipeline
# ---------------------------------------------------------------------------

class AlpacaNewsClient:
    """
    Fetches news from Alpaca's News API.

    **OPTIONAL** — not invoked in the default pipeline.
    ``AlphaVantageNewsClient`` (alphavantage_client.py) is the primary source
    because it provides pre-computed per-ticker sentiment scores.

    Use this class only as a supplementary or fallback feed.
    Do **not** insert into the DB here — sentiment scoring and storage
    are the responsibility of the sentiment module.
    """

    def __init__(self) -> None:
        self._client = NewsClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
        )
        logger.info(
            "alpaca_news.init",
            note="optional client — not in default pipeline",
        )

    def fetch_news(
        self,
        tickers: list[str],
        hours_back: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent news articles for *tickers*.

        Parameters
        ----------
        tickers:
            Symbols to query, e.g. ``["AAPL", "MSFT"]``.
        hours_back:
            Look-back window in hours (relative to now, UTC).

        Returns
        -------
        list[dict]
            One dict per (article, matched-ticker) pair.
            Keys: ``ticker``, ``title``, ``summary``, ``source``,
                  ``published_at``, ``headline_hash``.
            ``sentiment_score``, ``sentiment_confidence``, and
            ``llm_direction`` are intentionally absent — set by the
            sentiment module before storage.
        """
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(hours=hours_back)
        ticker_set = set(tickers)

        logger.info(
            "alpaca_news.fetch.start",
            tickers=tickers,
            hours_back=hours_back,
            start=str(start),
        )

        # NewsRequest.symbols is Optional[str] — comma-separated tickers
        request = NewsRequest(
            symbols=",".join(tickers),
            start=start,
            end=end,
            sort="DESC",
            limit=50,
        )
        news_set = _with_retry(
            lambda: self._client.get_news(request),
            label="get_news",
        )

        seen_hashes: set[str] = set()
        rows: list[dict[str, Any]] = []

        for article in news_set.news:
            # Assign ticker: first symbol in the article that we requested.
            matched = [s for s in (article.symbols or []) if s in ticker_set]
            if not matched:
                continue                # article not relevant to our tickers

            headline_hash = hashlib.sha256(
                article.headline.encode("utf-8")
            ).hexdigest()

            for ticker in matched:
                key = (headline_hash, ticker)
                if key in seen_hashes:
                    continue
                seen_hashes.add(key)

                rows.append(
                    {
                        "ticker":        ticker,
                        "title":         article.headline,
                        "summary":       article.summary or None,
                        "source":        article.source or None,
                        "published_at":  article.created_at,
                        "headline_hash": headline_hash,
                    }
                )

        logger.info(
            "alpaca_news.fetch.done",
            tickers=tickers,
            articles=len(rows),
        )
        return rows
