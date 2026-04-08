"""
Four-Point AI Trader — entry point.

Usage
-----
  python -m trading_engine.main \\
      --tickers AAPL MSFT JPM BAC \\
      --pairs JPM,BAC \\
      --paper

  # Live trading (use with extreme caution):
  python -m trading_engine.main --tickers AAPL --live
"""

from __future__ import annotations

import argparse
import signal
import sys

from dotenv import load_dotenv

from trading_engine.orchestrator.engine import TradingEngine
from trading_engine.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_pair(s: str) -> tuple[str, str]:
    """Parse a ``"JPM,BAC"`` string into a ``("JPM", "BAC")`` tuple."""
    parts = [p.strip().upper() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Pair must be two comma-separated tickers, got: {s!r}"
        )
    return (parts[0], parts[1])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="four-point-trader",
        description="Autonomous stock trading engine.",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "JPM", "BAC"],
        metavar="TICKER",
        help="Equity symbols to trade (default: AAPL MSFT JPM BAC).",
    )
    p.add_argument(
        "--pairs",
        nargs="*",
        default=["JPM,BAC"],
        type=str,
        metavar="T1,T2",
        help="Cointegrated pairs for OU mean-reversion (e.g. JPM,BAC).",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Connect to Alpaca live trading.  Omit for paper (default).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    p.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Optional path for newline-delimited JSON log file.",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging(log_level=args.log_level, log_file=args.log_file)

    paper = not args.live
    tickers = [t.upper() for t in args.tickers]
    pairs: list[tuple[str, str]] = [_parse_pair(s) for s in (args.pairs or [])]

    logger.info(
        "main.start",
        tickers=tickers,
        pairs=pairs,
        paper=paper,
    )

    engine = TradingEngine(tickers=tickers, pairs=pairs, paper=paper)

    # Graceful SIGTERM / SIGINT handler — signals the shutdown event
    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("main.signal_received", signal=sig_name)
        engine._shutdown_event.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        engine.startup_checks()
    except RuntimeError as exc:
        logger.error("main.startup_failed", error=str(exc))
        sys.exit(1)

    engine.run()
    logger.info("main.exit")


if __name__ == "__main__":
    main()
