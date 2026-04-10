"""
yfinance / FundamentalsClient connectivity and data-quality check.

No API key required — yfinance uses public Yahoo Finance endpoints.

Tests:
  1. Raw yfinance fetch for a single ticker (AAPL) — shows all useful fields.
  2. FundamentalsClient.get_market_caps() for a cross-sector sample.
  3. Cache behaviour — second call must be instant (< 0.1 s).
  4. Preview of other yfinance fields relevant to future features
     (beta, shortRatio, recommendationKey, nextEarningsDate).

Usage (from trading_engine/):
    .venv/bin/python scripts/check_yfinance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# .env not strictly needed for yfinance, but load it for consistency
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yfinance as yf

SEP = "-" * 60

# Representative cross-sector sample including some smaller/quantum names
# that may have limited yfinance coverage
SAMPLE_TICKERS = [
    "AAPL", "MSFT", "NVDA",          # mega-cap tech
    "LMT", "RTX", "KTOS",            # defense
    "JNJ", "LLY", "NVO",             # healthcare / pharma
    "IONQ", "QBTS", "RGTI", "QUBT",  # quantum (small-cap, may have gaps)
    "GILD",                           # the 51st ticker that triggered the AV bug
]


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗  {msg}")


def _info(msg: str) -> None:
    print(f"     {msg}")


def _fmt_cap(cap: float) -> str:
    if cap >= 1e12:
        return f"${cap/1e12:.2f}T"
    if cap >= 1e9:
        return f"${cap/1e9:.2f}B"
    if cap >= 1e6:
        return f"${cap/1e6:.2f}M"
    return f"${cap:.0f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(SEP)
    print("yfinance / FundamentalsClient connectivity check")
    print(SEP)

    failures = 0

    # ---- 1. Raw yfinance single-ticker deep inspection ----
    print("\n[1] Raw yfinance info — AAPL")
    fields_of_interest = [
        ("marketCap",         "Market cap"),
        ("beta",              "Beta (52-wk vs S&P)"),
        ("shortRatio",        "Short ratio (days to cover)"),
        ("shortPercentOfFloat", "Short % of float"),
        ("recommendationKey", "Analyst recommendation"),
        ("recommendationMean","Analyst score (1=Strong Buy, 5=Strong Sell)"),
        ("trailingPE",        "Trailing P/E"),
        ("forwardPE",         "Forward P/E"),
        ("dividendYield",     "Dividend yield"),
        ("fiftyTwoWeekHigh",  "52-week high"),
        ("fiftyTwoWeekLow",   "52-week low"),
        ("averageVolume",     "Average daily volume"),
    ]
    try:
        info = yf.Ticker("AAPL").info
        _ok("yf.Ticker('AAPL').info fetched successfully")
        for field, label in fields_of_interest:
            val = info.get(field)
            if val is None:
                _warn(f"{label:40s} → not available")
            else:
                _info(f"{label:40s} → {val}")
    except Exception as exc:
        _fail(f"yf.Ticker fetch failed: {exc}")
        failures += 1

    # ---- 2. Earnings / calendar ----
    print("\n[2] Earnings dates — AAPL")
    try:
        ticker_obj = yf.Ticker("AAPL")
        cal = ticker_obj.calendar
        if cal is not None and not cal.empty:
            _ok("Calendar fetched")
            print(cal.to_string())
        else:
            _warn("Calendar returned None or empty — may be unavailable for this ticker.")
    except Exception as exc:
        _warn(f"Calendar fetch raised: {exc}")

    # ---- 3. FundamentalsClient.get_market_caps() — full sample ----
    print(f"\n[3] FundamentalsClient.get_market_caps() — {len(SAMPLE_TICKERS)} tickers")
    try:
        from trading_engine.data.fundamentals_client import FundamentalsClient
        client = FundamentalsClient()

        t0 = time.monotonic()
        caps = client.get_market_caps(SAMPLE_TICKERS)
        elapsed = time.monotonic() - t0

        _ok(f"Fetched in {elapsed:.1f}s (parallel, {len(SAMPLE_TICKERS)} tickers)")

        # Sort by cap descending and display
        print(f"\n  {'Ticker':<8} {'Market Cap':>12}  {'AV tier'}")
        print(f"  {'------':<8} {'-----------':>12}  {'---------'}")
        sorted_tickers = sorted(SAMPLE_TICKERS, key=lambda t: caps.get(t, 0.0), reverse=True)
        for i, t in enumerate(sorted_tickers):
            cap = caps[t]
            tier = "→ AV (top 30)" if i < 30 else "→ Alpaca fallback"
            cap_str = _fmt_cap(cap) if cap > 0 else "N/A"
            marker = "  " if cap > 0 else "⚠ "
            print(f"  {marker}{t:<6} {cap_str:>12}  {tier}")

        zero_caps = [t for t in SAMPLE_TICKERS if caps.get(t, 0) == 0]
        if zero_caps:
            _warn(f"No market cap data for: {zero_caps}")
            _info("These will sort to the bottom (Alpaca fallback) — acceptable.")

    except Exception as exc:
        _fail(f"FundamentalsClient.get_market_caps failed: {exc}")
        failures += 1

    # ---- 4. Cache verification ----
    print("\n[4] Cache verification — second call must be instant")
    try:
        t1 = time.monotonic()
        caps2 = client.get_market_caps(SAMPLE_TICKERS)
        elapsed2 = time.monotonic() - t1

        if elapsed2 < 0.5:
            _ok(f"Cache hit — second call took {elapsed2*1000:.1f}ms (< 500ms)")
        else:
            _warn(f"Second call took {elapsed2:.2f}s — cache may not be working.")

        # Spot check a value is identical
        assert caps2["AAPL"] == caps["AAPL"], "Cached value differs!"
        _ok("Cached values are consistent with first fetch")
    except Exception as exc:
        _fail(f"Cache check failed: {exc}")
        failures += 1

    # ---- 5. AV split preview ----
    print(f"\n[5] AV / Alpaca split preview (top 30 vs rest for your full universe)")
    try:
        from trading_engine.data.fundamentals_client import FundamentalsClient as FC
        # Use the same client (cache already warm)
        sorted_by_cap = sorted(SAMPLE_TICKERS, key=lambda t: caps.get(t, 0.0), reverse=True)
        av_set     = sorted_by_cap[:30]
        alpaca_set = sorted_by_cap[30:]
        _info(f"AV tickers     ({len(av_set)}): {av_set}")
        _info(f"Alpaca tickers ({len(alpaca_set)}): {alpaca_set}")
    except Exception as exc:
        _fail(f"Split preview failed: {exc}")

    # ---- Summary ----
    print(f"\n{SEP}")
    if failures == 0:
        print("yfinance check PASSED.")
    else:
        print(f"yfinance check FAILED — {failures} critical test(s) failed.")
    print(SEP)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
