"""
Alpha Vantage connectivity and data-quality check.

Loads credentials from .env, then:
  1. Shows current daily call count (free-tier budget).
  2. Fetches news for a small test set (3 tickers, 2-hour window).
  3. Prints article count, sample titles, and per-ticker sentiment.

Usage (from trading_engine/):
    .venv/bin/python scripts/check_alphavantage.py

WARNING: This script consumes 1 of your 20 daily AV API calls.
         Run only when the call count is below 15.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Load .env before importing any project module
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Make project importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_engine.data.alphavantage_client import (
    AlphaVantageNewsClient,
    AlphaVantageError,
    RateLimitExceeded,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Small set — enough to verify multi-ticker fetch without burning quota
TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]
HOURS_BACK   = 2

SEP = "-" * 60


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗  {msg}")


def _info(msg: str) -> None:
    print(f"     {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(SEP)
    print("Alpha Vantage connectivity check")
    print(SEP)

    client = AlphaVantageNewsClient()

    # ---- 1. Daily call count ----
    print("\n[1] Daily call budget")
    count = client.get_daily_call_count()
    if count >= 20:
        _fail(f"Hard limit reached: {count}/20 calls today — aborting fetch.")
        return 1
    elif count >= 15:
        _fail(f"Budget warning: {count}/20 calls today — proceeding but be careful.")
    else:
        _ok(f"Calls today: {count}/20")

    # ---- 2. Fetch news ----
    print(f"\n[2] Fetch news — tickers={TEST_TICKERS}, hours_back={HOURS_BACK}")
    try:
        articles = client.fetch_news(TEST_TICKERS, hours_back=HOURS_BACK)
        _ok(f"Fetch succeeded — {len(articles)} article-ticker pairs returned")
    except RateLimitExceeded as exc:
        _fail(f"Rate limit exceeded: {exc}")
        return 1
    except AlphaVantageError as exc:
        _fail(f"AV API error: {exc}")
        _info("Common causes: invalid API key, >50 tickers, bad time_from format.")
        return 1
    except Exception as exc:
        _fail(f"Unexpected error: {exc}")
        return 1

    # ---- 3. Per-ticker breakdown ----
    print("\n[3] Articles per ticker")
    by_ticker: dict[str, list] = {t: [] for t in TEST_TICKERS}
    for a in articles:
        t = a.get("ticker", "")
        if t in by_ticker:
            by_ticker[t].append(a)

    all_ok = True
    for ticker, arts in by_ticker.items():
        if arts:
            _ok(f"{ticker}: {len(arts)} article(s)")
        else:
            _info(f"{ticker}: 0 articles (no news in last {HOURS_BACK}h — not an error)")

    # ---- 4. Sample data ----
    if articles:
        print("\n[4] Sample articles (up to 5)")
        for a in articles[:5]:
            title   = a.get("title", "")[:80]
            ticker  = a.get("ticker", "")
            label   = a.get("av_sentiment_label", "")
            score   = a.get("av_sentiment_score", 0.0)
            rel     = a.get("relevance_score", 0.0)
            pub     = str(a.get("published_at", ""))[:19]
            source  = a.get("source", "")
            print(f"\n  [{ticker}] {title}")
            print(f"       source={source}  pub={pub}")
            print(f"       av_sentiment={label} ({score:+.3f})  relevance={rel:.3f}")
    else:
        print("\n[4] No articles returned — try a wider hours_back window.")

    # ---- 5. Updated call count ----
    new_count = client.get_daily_call_count()
    print(f"\n[5] Call count after fetch: {new_count}/20")

    print(f"\n{SEP}")
    print("Alpha Vantage check PASSED" if all_ok else "Alpha Vantage check PASSED (with warnings)")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
