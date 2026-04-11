"""
CLI entry point for the decision quality analysis.

Usage
-----
    cd trading_engine
    .venv/bin/python -m analysis.run_analysis \\
        --db-url "$DB_URL" \\
        --days 14 \\
        --output-dir analysis/reports/

Or via environment variable:
    export DB_URL="postgresql+psycopg2://trader:traderpass@localhost:5432/trading"
    .venv/bin/python -m analysis.run_analysis --days 14
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is on sys.path when this module is run directly
_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading_engine.analysis.outcome_labeler import load_labeled_decisions
from trading_engine.analysis.signal_quality import (
    compute_ensemble_accuracy,
    compute_signal_accuracy,
)
from trading_engine.analysis.weight_evolution import (
    extract_weight_history,
    summarise_weight_evolution,
)
from trading_engine.analysis.parameter_sweep import (
    sweep_entry_z,
    sweep_eta,
    sweep_hours_back,
    sweep_min_confidence,
)
from trading_engine.analysis.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse decision quality and generate a parameter recommendation "
            "report from trade_log + ohlcv data."
        )
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DB_URL"),
        help="SQLAlchemy PostgreSQL URL (or set DB_URL env var).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Analyse the last N calendar days (default: all history).",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Restrict analysis to a single ticker symbol.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/reports",
        help="Directory for the output report (default: analysis/reports/).",
    )
    args = parser.parse_args()

    if not args.db_url:
        print(
            "ERROR: --db-url is required (or set the DB_URL environment variable).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading decisions (days={args.days}, ticker={args.ticker}) …")
    df = load_labeled_decisions(args.db_url, ticker=args.ticker, days=args.days)

    if df.empty:
        print(
            "No trade_log data found.  "
            "Run the engine for at least a few days first.",
            file=sys.stderr,
        )
        sys.exit(0)

    n = len(df)
    n_active = int((df["final_signal"] != 0).sum())
    print(f"Loaded {n} decisions ({n_active} directional).  Running analysis …")

    sig_acc  = compute_signal_accuracy(df)
    ens_acc  = compute_ensemble_accuracy(df)
    wdf      = extract_weight_history(df)
    w_sum    = summarise_weight_evolution(wdf)
    sw_hb    = sweep_hours_back(df)
    sw_ez    = sweep_entry_z(df)
    sw_mc    = sweep_min_confidence(df)
    sw_et    = sweep_eta(df)

    date_str  = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(args.output_dir)
    out_path  = out_dir / f"report_{date_str}.md"

    print(f"Writing report → {out_path} …")
    generate_report(
        labeled_df=df,
        signal_accuracy=sig_acc,
        ensemble_accuracy=ens_acc,
        weight_summary=w_sum,
        sweep_hours_back=sw_hb,
        sweep_entry_z=sw_ez,
        sweep_min_confidence=sw_mc,
        sweep_eta=sw_et,
        output_path=out_path,
    )

    print(f"\nReport written: {out_path}")

    # Quick console summary
    print("\n── Quick summary ──────────────────────────────────────")
    if not ens_acc.empty and "win_rate_15m" in ens_acc.columns:
        try:
            row = (
                ens_acc.loc["all"]
                if "all" in ens_acc.index
                else ens_acc.iloc[0]
            )
            wr_15m = row["win_rate_15m"]
            wr_1h  = row.get("win_rate_1h", float("nan"))
            if wr_15m == wr_15m:
                print(f"  Ensemble win_rate_15m (all):  {wr_15m * 100:.1f} %")
            if wr_1h == wr_1h:
                print(f"  Ensemble win_rate_1h  (all):  {wr_1h  * 100:.1f} %")
        except Exception:
            pass

    if w_sum:
        collapsed = w_sum.get("collapsed_signals", [])
        if collapsed:
            print(f"\n  ⚠  {len(collapsed)} collapsed signal(s) detected:")
            for c in collapsed:
                print(
                    f"     {c['ticker']} / {c['regime']} / {c['signal']} "
                    f"— weight {c['final_weight']:.4f}"
                )

    print("──────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
