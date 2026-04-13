"""
Four-Point AI Trader — entry point.

Workflow
--------
  # 1. Discover pairs (run weekly or when the universe changes):
  python -m trading_engine.tools.pair_scanner --tickers LMT NOC RTX GD BA ...

  # 2. Run the engine (reads discovered_pairs.json automatically):
  python -m trading_engine.main --tickers AAPL MSFT --paper

  # Live trading (use with extreme caution):
  python -m trading_engine.main --tickers AAPL --live
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

from trading_engine.orchestrator.engine import TradingEngine
from trading_engine.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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
        "--pairs-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to discovered_pairs.json written by pair_scanner.py "
            "(default: trading_engine/config/discovered_pairs.json)."
        ),
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
    p.add_argument(
        "--update-kelly-stats",
        action="store_true",
        help=(
            "Recompute Kelly sizing stats from Alpaca confirmed fill P&L, "
            "write updated engine_state.json, and exit."
        ),
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

    logger.info(
        "main.start",
        tickers=tickers,
        pairs_file=str(args.pairs_file) if args.pairs_file else "default",
        paper=paper,
    )

    engine = TradingEngine(
        tickers=tickers,
        paper=paper,
        pairs_file=args.pairs_file,
    )

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

    if args.update_kelly_stats:
        engine._update_kelly_stats()
        engine._save_state()
        logger.info("main.kelly_stats_updated")
        return

    engine.run()
    logger.info("main.exit")


if __name__ == "__main__":
    main()
