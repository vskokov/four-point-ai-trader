"""
Alpaca connectivity and data-quality check.

Loads credentials from .env, then tests:
  1. Paper account info (equity, cash).
  2. Market clock (is_open, next open/close).
  3. Latest quote for AAPL.
  4. Historical OHLCV bars for AAPL (last 5 trading days).
  5. AlpacaNewsClient — news fetch for 3 tickers.

Uses the IEX free-tier feed throughout (no SIP subscription required).

Usage (from trading_engine/):
    .venv/bin/python scripts/check_alpaca.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SEP = "-" * 60


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗  {msg}")


def _info(msg: str) -> None:
    print(f"     {msg}")


# ---------------------------------------------------------------------------
# Minimal no-op storage so AlpacaMarketData can be constructed without a DB
# ---------------------------------------------------------------------------

class _NoopStorage:
    """Accepts insert_ohlcv calls and discards them."""
    def insert_ohlcv(self, rows):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(SEP)
    print("Alpaca connectivity check")
    print(SEP)

    failures = 0

    # ---- Import clients ----
    try:
        from trading_engine.data.alpaca_client import AlpacaMarketData, AlpacaNewsClient
    except Exception as exc:
        _fail(f"Import failed: {exc}")
        return 1

    # ---- 1. Account info ----
    print("\n[1] Paper account info")
    try:
        market = AlpacaMarketData(_NoopStorage())
        account = market.get_account_info()
        _ok(f"Connected — equity=${account['equity']:,.2f}  cash=${account['cash']:,.2f}")
        _info(f"buying_power=${account['buying_power']:,.2f}  portfolio_value=${account['portfolio_value']:,.2f}")
    except Exception as exc:
        _fail(f"get_account_info failed: {exc}")
        failures += 1

    # ---- 2. Market clock ----
    print("\n[2] Market clock")
    try:
        clock = market.get_market_clock()
        status = "OPEN" if clock["is_open"] else "CLOSED"
        _ok(f"Market is {status}")
        _info(f"next_open ={clock['next_open']}")
        _info(f"next_close={clock['next_close']}")
        _info(f"timestamp ={clock['timestamp']}")
        _info(f"is_market_open() (cached) → {market.is_market_open()}")
    except Exception as exc:
        _fail(f"get_market_clock failed: {exc}")
        failures += 1

    # ---- 3. Latest quote ----
    print("\n[3] Latest quote — AAPL")
    try:
        quote = market.get_latest_quote("AAPL")
        _ok(f"bid={quote['bid']:.4f}  ask={quote['ask']:.4f}  mid={quote['mid']:.4f}")
        _info(f"timestamp={quote.get('timestamp', 'n/a')}")
    except Exception as exc:
        _fail(f"get_latest_quote failed: {exc}")
        _info("This fails outside market hours on the IEX feed — not always an error.")
        failures += 1

    # ---- 4. Historical OHLCV ----
    print("\n[4] Historical OHLCV — AAPL, last 10 calendar days, 1Day bars")
    try:
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=10)
        df = market.fetch_historical_ohlcv(["AAPL"], start, end, timeframe="1Day")
        if df.empty:
            _fail("No bars returned — check IEX feed access and date range.")
            failures += 1
        else:
            _ok(f"{len(df)} bar(s) returned")
            print(df[["ticker", "open", "high", "low", "close", "volume"]].tail(3).to_string())
    except Exception as exc:
        _fail(f"fetch_historical_ohlcv failed: {exc}")
        failures += 1

    # ---- 5. Alpaca News ----
    print("\n[5] AlpacaNewsClient — news for AAPL, MSFT, NVDA (last 4 hours)")
    try:
        news_client = AlpacaNewsClient()
        articles = news_client.fetch_news(["AAPL", "MSFT", "NVDA"], hours_back=4)
        _ok(f"{len(articles)} article-ticker pair(s) returned")
        if articles:
            print(f"\n  Sample (up to 3):")
            for a in articles[:3]:
                title  = (a.get("title") or "")[:80]
                ticker = a.get("ticker", "")
                source = a.get("source", "")
                pub    = str(a.get("published_at", ""))[:19]
                print(f"\n  [{ticker}] {title}")
                print(f"       source={source}  pub={pub}")
        else:
            _info("No articles in window — try increasing hours_back.")
    except Exception as exc:
        _fail(f"AlpacaNewsClient.fetch_news failed: {exc}")
        failures += 1

    # ---- Summary ----
    print(f"\n{SEP}")
    if failures == 0:
        print("Alpaca check PASSED — all tests green.")
    else:
        print(f"Alpaca check FAILED — {failures} test(s) failed.")
    print(SEP)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
