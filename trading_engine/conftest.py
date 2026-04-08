"""
Pytest configuration — adds the repo root to sys.path so that
'from trading_engine.xxx import ...' works when pytest is invoked
from inside the trading_engine/ directory.
"""

import os
import sys

# Insert the directory that *contains* trading_engine/ into the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
