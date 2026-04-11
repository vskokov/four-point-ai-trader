"""Offline decision-quality analysis framework.

Run after 2+ weeks of engine operation to assess signal quality and
generate parameter recommendations.  No live engine calls required.

Usage
-----
    cd trading_engine
    .venv/bin/python -m analysis.run_analysis --db-url "$DB_URL" --days 14
"""
