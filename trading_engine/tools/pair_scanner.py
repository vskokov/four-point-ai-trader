"""
Pair discovery scanner — standalone CLI tool.

Scans a curated ticker universe for cointegrated pairs and writes the results
to a JSON file.  The trading engine reads this file at startup.

Run weekly (or whenever the universe changes):

  python -m trading_engine.tools.pair_scanner \\
      --tickers LMT NOC RTX GD BA MSFT GOOG AAPL NVDA AMD \\
      --lookback-days 504 \\
      --output config/discovered_pairs.json \\
      --min-correlation 0.70 \\
      --max-pvalue 0.05 \\
      --min-half-life 5 \\
      --max-half-life 60 \\
      --max-pairs 10

Pipeline
--------
1. Fetch historical daily close prices from Alpaca.
2. Drop tickers with >10 % missing bars.
3. Correlation pre-filter on log-returns (|corr| >= min_correlation).
4. Engle-Granger + Johansen cointegration test (p-value < max_pvalue).
5. OU parameter estimation; filter by half-life window.
6. Rank by EG p-value ascending; keep top max_pairs.
7. Write JSON output.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from trading_engine.data.alpaca_client import AlpacaMarketData
from trading_engine.signals.mean_reversion import CointegrationTest, OUSpreadSignal
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_OUTPUT = Path(__file__).parent.parent / "config" / "discovered_pairs.json"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pair-scanner",
        description="Scan a ticker universe for cointegrated pairs.",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        metavar="TICKER",
        help="Ticker universe to scan.",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=504,
        metavar="N",
        help="Historical lookback in calendar days (default: 504 ≈ 2 trading years).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="PATH",
        help="Output JSON file path.",
    )
    p.add_argument(
        "--min-correlation",
        type=float,
        default=0.70,
        metavar="FLOAT",
        help="Minimum absolute Pearson correlation on log-returns (default: 0.70).",
    )
    p.add_argument(
        "--max-pvalue",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Maximum Engle-Granger p-value (default: 0.05).",
    )
    p.add_argument(
        "--min-half-life",
        type=float,
        default=5.0,
        metavar="FLOAT",
        help="Minimum OU half-life in bars (default: 5).",
    )
    p.add_argument(
        "--max-half-life",
        type=float,
        default=60.0,
        metavar="FLOAT",
        help="Maximum OU half-life in bars (default: 60).",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of output pairs (default: 10).",
    )
    return p


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def fetch_close_prices(
    alpaca: AlpacaMarketData,
    tickers: list[str],
    lookback_days: int,
) -> pd.DataFrame:
    """
    Fetch daily close prices for all *tickers*.

    A 50 % calendar-day buffer is added to account for weekends and holidays
    so that the returned DataFrame contains at least *lookback_days* trading bars.

    Returns
    -------
    pd.DataFrame
        Wide-form: index = dates, columns = tickers, values = close price.
    """
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=int(lookback_days * 1.5))

    logger.info(
        "pair_scanner.fetch.start",
        n_tickers=len(tickers),
        start=str(start.date()),
        end=str(end.date()),
    )

    df = alpaca.get_historical_bars(tickers, start, end, timeframe="1Day")

    logger.info(
        "pair_scanner.fetch.done",
        rows=len(df),
        n_tickers=len(df.columns),
    )
    return df


def filter_tickers(
    df: pd.DataFrame,
    max_missing_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Drop any ticker column with more than *max_missing_frac* missing rows.

    Drops rows that are entirely NaN after removing bad tickers, then forward-
    fills any remaining isolated NaNs (e.g. a single halt day) before returning.
    """
    n = len(df)
    keep: list[str] = []

    for col in df.columns:
        missing_frac = df[col].isna().sum() / n if n > 0 else 1.0
        if missing_frac > max_missing_frac:
            logger.warning(
                "pair_scanner.ticker_dropped",
                ticker=col,
                missing_pct=round(missing_frac * 100, 1),
            )
        else:
            keep.append(col)

    df = df[keep]
    # Drop rows where ALL remaining tickers are NaN (e.g. market-wide halts).
    df = df.dropna(how="all")
    # Forward-fill isolated NaNs (e.g. single-day data gaps).
    df = df.ffill().dropna()
    return df


def correlation_filter(
    df: pd.DataFrame,
    min_correlation: float,
) -> list[tuple[str, str, float]]:
    """
    Return pairs whose log-return Pearson |correlation| >= *min_correlation*.

    Log-return correlation is the correct pre-screen for cointegration; raw
    price correlation is spurious for any two trending series.

    Returns
    -------
    list of (ticker1, ticker2, correlation)
    """
    log_returns = np.log(df / df.shift(1)).dropna()
    corr_matrix = log_returns.corr()

    tickers = list(df.columns)
    n_candidate = len(tickers) * (len(tickers) - 1) // 2
    surviving: list[tuple[str, str, float]] = []

    for t1, t2 in combinations(tickers, 2):
        corr = float(corr_matrix.loc[t1, t2])
        if abs(corr) >= min_correlation:
            surviving.append((t1, t2, corr))

    logger.info(
        "pair_scanner.correlation_filter",
        n_candidate=n_candidate,
        n_surviving=len(surviving),
        min_correlation=min_correlation,
    )
    return surviving


def cointegration_filter(
    df: pd.DataFrame,
    pairs: list[tuple[str, str, float]],
    max_pvalue: float,
) -> list[dict[str, Any]]:
    """
    Run CointegrationTest on each candidate pair.

    Returns pairs with ``eg_pvalue < max_pvalue``.

    Returns
    -------
    list of dicts with keys:
        ticker1, ticker2, correlation, eg_pvalue,
        johansen_trace_stat, beta_ols
    """
    coint_test = CointegrationTest()
    results: list[dict[str, Any]] = []

    for t1, t2, corr in pairs:
        p1 = df[t1].astype(float)
        p2 = df[t2].astype(float)

        try:
            result = coint_test.test(p1, p2)
        except Exception as exc:
            logger.warning(
                "pair_scanner.coint_test_failed",
                ticker1=t1,
                ticker2=t2,
                error=str(exc),
            )
            continue

        if result["eg_pvalue"] < max_pvalue:
            results.append(
                {
                    "ticker1": t1,
                    "ticker2": t2,
                    "correlation": round(corr, 4),
                    "eg_pvalue": round(result["eg_pvalue"], 6),
                    "johansen_trace_stat": round(result["johansen_trace_stat"], 4),
                    "beta_ols": round(result["beta_ols"], 6),
                }
            )

    logger.info(
        "pair_scanner.cointegration_filter",
        n_input=len(pairs),
        n_cointegrated=len(results),
        max_pvalue=max_pvalue,
    )
    return results


def ou_filter(
    df: pd.DataFrame,
    pairs: list[dict[str, Any]],
    min_half_life: float,
    max_half_life: float,
) -> list[dict[str, Any]]:
    """
    Fit OU parameters for each pair and keep those within the half-life window.

    A temporary ``OUSpreadSignal`` instance is reused across pairs solely to
    call ``fit_ou_params(spread)``; it does not store any state between calls.

    Returns
    -------
    list of dicts adding:
        half_life_bars, kappa, mu, sigma
    """
    _helper = OUSpreadSignal("_scanner_", "_scanner_")
    results: list[dict[str, Any]] = []

    for pair in pairs:
        t1 = pair["ticker1"]
        t2 = pair["ticker2"]
        beta = pair["beta_ols"]

        spread = df[t1].astype(float) - beta * df[t2].astype(float)

        try:
            ou = _helper.fit_ou_params(spread)
        except Exception as exc:
            logger.warning(
                "pair_scanner.ou_fit_failed",
                ticker1=t1,
                ticker2=t2,
                error=str(exc),
            )
            continue

        half_life = ou["half_life_bars"]
        if not (min_half_life <= half_life <= max_half_life):
            logger.debug(
                "pair_scanner.half_life_filtered",
                ticker1=t1,
                ticker2=t2,
                half_life=round(half_life, 1),
                min=min_half_life,
                max=max_half_life,
            )
            continue

        results.append(
            {
                **pair,
                "half_life_bars": round(half_life, 2),
                "kappa": round(ou["kappa"], 6),
                "mu": round(ou["mu"], 6),
                "sigma": round(ou["sigma"], 6),
            }
        )

    logger.info(
        "pair_scanner.ou_filter",
        n_input=len(pairs),
        n_surviving=len(results),
        min_half_life=min_half_life,
        max_half_life=max_half_life,
    )
    return results


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _build_result(
    lookback_days: int,
    n_tickers_scanned: int,
    n_candidate_pairs: int,
    n_correlated: int,
    n_cointegrated: int,
    n_selected: int,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scanned_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookback_days": lookback_days,
        "n_tickers_scanned": n_tickers_scanned,
        "n_candidate_pairs": n_candidate_pairs,
        "n_correlated": n_correlated,
        "n_cointegrated": n_cointegrated,
        "n_selected": n_selected,
        "pairs": pairs,
    }


def _write_output(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write via .tmp → rename
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(path)
    logger.info("pair_scanner.output_written", path=str(path))


# ---------------------------------------------------------------------------
# Top-level scan function (callable from tests without CLI)
# ---------------------------------------------------------------------------

def run_scan(
    tickers: list[str],
    lookback_days: int = 504,
    output: Path = _DEFAULT_OUTPUT,
    min_correlation: float = 0.70,
    max_pvalue: float = 0.05,
    min_half_life: float = 5.0,
    max_half_life: float = 60.0,
    max_pairs: int = 10,
    _alpaca: AlpacaMarketData | None = None,
) -> dict[str, Any]:
    """
    Run the full pair discovery pipeline and write results to *output*.

    Parameters
    ----------
    _alpaca:
        Injectable ``AlpacaMarketData`` instance (used in tests to pass a mock).
        If *None*, a real instance is created using credentials from settings.

    Returns
    -------
    dict — the same structure written to the JSON output file.
    """
    if _alpaca is None:
        import trading_engine.config.settings  # noqa: F401 — triggers env validation
        _alpaca = AlpacaMarketData(storage=None)

    # 1. Fetch prices
    close_df = fetch_close_prices(_alpaca, tickers, lookback_days)
    n_tickers_input = len(close_df.columns)

    # 2. Drop tickers with too many gaps
    close_df = filter_tickers(close_df)
    n_tickers_scanned = len(close_df.columns)

    if n_tickers_scanned < 2:
        logger.warning("pair_scanner.too_few_tickers", n=n_tickers_scanned)
        result = _build_result(
            lookback_days=lookback_days,
            n_tickers_scanned=n_tickers_scanned,
            n_candidate_pairs=0,
            n_correlated=0,
            n_cointegrated=0,
            n_selected=0,
            pairs=[],
        )
        _write_output(output, result)
        return result

    n_candidate_pairs = n_tickers_scanned * (n_tickers_scanned - 1) // 2

    # 3. Correlation filter (on log-returns)
    correlated = correlation_filter(close_df, min_correlation)

    # 4. Cointegration filter
    cointegrated = cointegration_filter(close_df, correlated, max_pvalue)

    # 5–6. OU parameter estimation + half-life filter
    ou_pairs = ou_filter(close_df, cointegrated, min_half_life, max_half_life)

    # 7. Rank by EG p-value ascending (strongest cointegration first)
    ou_pairs.sort(key=lambda x: x["eg_pvalue"])
    selected = ou_pairs[:max_pairs]

    logger.info(
        "pair_scanner.scan_complete",
        n_tickers_input=n_tickers_input,
        n_tickers_scanned=n_tickers_scanned,
        n_candidate_pairs=n_candidate_pairs,
        n_correlated=len(correlated),
        n_cointegrated=len(cointegrated),
        n_selected=len(selected),
    )
    for pair in selected:
        logger.info(
            "pair_scanner.selected_pair",
            ticker1=pair["ticker1"],
            ticker2=pair["ticker2"],
            eg_pvalue=pair["eg_pvalue"],
            half_life_bars=pair["half_life_bars"],
        )

    # 8. Write output
    result = _build_result(
        lookback_days=lookback_days,
        n_tickers_scanned=n_tickers_scanned,
        n_candidate_pairs=n_candidate_pairs,
        n_correlated=len(correlated),
        n_cointegrated=len(cointegrated),
        n_selected=len(selected),
        pairs=selected,
    )
    _write_output(output, result)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = _build_parser().parse_args(argv)

    tickers = [t.upper() for t in args.tickers]
    run_scan(
        tickers=tickers,
        lookback_days=args.lookback_days,
        output=args.output,
        min_correlation=args.min_correlation,
        max_pvalue=args.max_pvalue,
        min_half_life=args.min_half_life,
        max_half_life=args.max_half_life,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
