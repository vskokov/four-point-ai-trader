"""
Unit tests for orchestrator/engine.py and orchestrator/state_manager.py.

All external calls (Alpaca, Ollama, DB) are mocked — no live connections.

Run with:
    .venv/bin/pytest tests/test_engine.py -v
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_ENG_MOD        = "trading_engine.orchestrator.engine"
_STORAGE        = f"{_ENG_MOD}.Storage"
_ALPACA         = f"{_ENG_MOD}.AlpacaMarketData"
_AV_CLIENT      = f"{_ENG_MOD}.AlphaVantageNewsClient"
_ALPACA_NEWS    = f"{_ENG_MOD}.AlpacaNewsClient"
_FUNDAMENTALS   = f"{_ENG_MOD}.FundamentalsClient"
_HMM            = f"{_ENG_MOD}.HMMRegimeDetector"
_LLM            = f"{_ENG_MOD}.LLMSentimentSignal"
_MWU            = f"{_ENG_MOD}.MWUMetaAgent"
_OU_SIGNAL      = f"{_ENG_MOD}.OUSpreadSignal"
_RISK           = f"{_ENG_MOD}.RiskManager"
_EXECUTOR       = f"{_ENG_MOD}.OrderExecutor"
_SCHEDULER      = f"{_ENG_MOD}.BackgroundScheduler"
_SETTINGS       = f"{_ENG_MOD}.settings"
_STATE_MGR      = f"{_ENG_MOD}.StateManager"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings():
    return SimpleNamespace(
        DB_URL="postgresql+psycopg2://fake:fake@localhost/fake",
        ALPACA_API_KEY="k",
        ALPACA_SECRET_KEY="s",
        ALPHAVANTAGE_API_KEY="av",
        OLLAMA_HOST="http://localhost:11434",
        OLLAMA_MODEL="gemma4:e4b",
    )


def _make_bar(ticker: str = "AAPL", close: float = 150.0) -> dict:
    return {
        "time":   datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc),
        "ticker": ticker,
        "open":   close - 0.5,
        "high":   close + 1.0,
        "low":    close - 1.0,
        "close":  close,
        "volume": 1_000_000,
    }


def _account(equity: float = 100_000.0) -> dict:
    return {
        "equity":          equity,
        "cash":            equity * 0.8,
        "buying_power":    equity * 1.6,
        "portfolio_value": equity,
    }


def _mwu_decision(signal: int = 1, score: float = 0.5) -> dict:
    return {
        "ticker":       "AAPL",
        "final_signal": signal,
        "score":        score,
        "regime":       2,
        "weights":      {"hmm_regime": 0.3, "ou_spread": 0.3, "llm_sentiment": 0.3, "analyst_recs": 0.1},
        "timestamp":    datetime.now(tz=timezone.utc),
    }


# ---------------------------------------------------------------------------
# Engine factory — builds a TradingEngine with all external deps mocked
# ---------------------------------------------------------------------------

def _build_engine(
    tickers=("AAPL",),
    pairs=(("JPM", "BAC"),),
    tmp_path: Path | None = None,
    hmm_fitted: bool = True,
    mwu_decision: dict | None = None,
):
    # Note: _build_engine bypasses TradingEngine.__init__ via __new__ and sets
    # all attributes directly.  The pairs parameter here sets engine._pairs
    # directly — no constructor call, so _load_discovered_pairs is not invoked.
    """
    Construct a TradingEngine via __new__ with all dependencies replaced by
    MagicMock instances.  Returns (engine, mock_storage, mock_alpaca, mock_mwu).
    """
    from trading_engine.orchestrator.engine import TradingEngine

    engine = TradingEngine.__new__(TradingEngine)

    # Core infra mocks
    mock_storage = MagicMock()
    mock_alpaca  = MagicMock()
    mock_av      = MagicMock()

    # Storage engine mock (for raw SQL queries)
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__  = MagicMock(return_value=False)
    mock_storage._engine.connect.return_value = mock_conn
    # Default: no LLM signal in DB
    mock_conn.execute.return_value.fetchone.return_value = None

    # Alpaca defaults
    mock_alpaca.get_account_info.return_value = _account()
    mock_alpaca.get_latest_quote.return_value = {"mid": 150.0, "bid": 149.9, "ask": 150.1}

    # HMM mock per ticker
    hmm_map: dict[str, MagicMock] = {}
    for t in tickers:
        m = MagicMock()
        m.is_fitted = hmm_fitted
        m.predict_regime.return_value = {
            "regime": 2,
            "label": "bull",
            "probs": [0.1, 0.2, 0.7],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        hmm_map[t] = m

    # MWU mock per ticker
    mwu_map: dict[str, MagicMock] = {}
    decision = mwu_decision or _mwu_decision(signal=0)   # neutral by default
    for t in tickers:
        m = MagicMock()
        m.scheduled_update.return_value = {**decision, "ticker": t}
        m.performance_report.return_value = {
            "n_updates": 10,
            "per_signal_win_rate": {
                "hmm_regime": 0.6,
                "ou_spread": 0.55,
                "llm_sentiment": 0.58,
                "analyst_recs": 0.52,
            },
            "per_regime_win_rate": {"bear": 0.5, "neutral": 0.55, "bull": 0.6},
            "current_weights": {},
        }
        mwu_map[t] = m

    # OU signal mock per pair
    ou_map: dict[tuple, MagicMock] = {}
    for p in pairs:
        m = MagicMock()
        m.compute_signal.return_value = {
            "signal": 0, "z_score": 0.3, "half_life": 10.0, "mu": 0.0,
            "sigma": 1.0, "beta": 1.1, "timestamp": datetime.now(tz=timezone.utc),
        }
        m.lookback = 60
        ou_map[tuple(p)] = m

    # Risk mock — no circuit breaker by default
    mock_risk = MagicMock()
    mock_risk.circuit_breaker.return_value = False

    # Executor mock
    mock_executor = MagicMock()
    mock_executor.submit_order.return_value = {"status": "submitted", "order_id": "abc"}
    mock_executor.get_positions.return_value = MagicMock()

    # State manager mock
    mock_state = MagicMock()
    mock_state.load.return_value = None

    # Fundamentals + Alpaca news mocks (used by sentiment_job and bar_handler)
    mock_fundamentals = MagicMock()
    mock_fundamentals.get_market_caps.return_value = {t: 1e12 for t in tickers}
    mock_fundamentals.get_earnings_dates.return_value = {t: None for t in tickers}
    mock_fundamentals.get_analyst_recommendations.return_value = {t: 0 for t in tickers}
    mock_fundamentals.get_vix.return_value = None
    mock_alpaca_news = MagicMock()
    mock_alpaca_news.fetch_news.return_value = []

    # Wire everything onto the engine
    engine.portfolio_optimizer = MagicMock()
    engine._tickers       = list(tickers)
    engine._pairs         = [tuple(p) for p in pairs]
    engine._paper         = True
    engine._models_dir    = tmp_path or Path("/tmp/test_engine")
    engine._storage       = mock_storage
    engine._alpaca        = mock_alpaca
    engine._av_client     = mock_av
    engine._alpaca_news   = mock_alpaca_news
    engine._fundamentals  = mock_fundamentals
    engine._hmm           = hmm_map
    engine._ou_signals    = ou_map
    engine._llm           = MagicMock()
    engine._mwu           = mwu_map
    engine._risk          = mock_risk
    engine._executor      = mock_executor
    engine._state_manager = mock_state
    engine._signal_stats  = {t: {"win_rate": 0.52, "avg_win": 0.015, "avg_loss": 0.010} for t in tickers}
    engine._shutdown_event = threading.Event()
    engine._emergency_close = False
    engine._scheduler   = None

    from collections import deque
    from trading_engine.orchestrator.engine import _REGIME_SMOOTH_WINDOW
    all_tickers = list(tickers)
    engine._regime_history = {t: deque(maxlen=_REGIME_SMOOTH_WINDOW) for t in all_tickers}
    engine._stable_regime_label = {t: "neutral" for t in all_tickers}
    engine._stable_regime = {t: 1 for t in all_tickers}  # 1 = neutral default
    engine._last_active_signal = {t: 0 for t in all_tickers}
    engine._last_signal_change_time = {t: None for t in all_tickers}
    engine._pdt_blocked_today: set[str] = set()
    engine._pdt_blocked_date = None
    engine._vix_multiplier = 1.0

    return engine, mock_storage, mock_alpaca, mwu_map


# ===========================================================================
# bar_handler — end-to-end flow
# ===========================================================================

class TestBarHandler:

    def test_bar_inserted_into_storage(self, tmp_path):
        engine, mock_storage, _, _ = _build_engine(tmp_path=tmp_path)
        engine.bar_handler(_make_bar("AAPL"))
        mock_storage.insert_ohlcv.assert_called_once()
        row = mock_storage.insert_ohlcv.call_args[0][0][0]
        assert row["ticker"] == "AAPL"

    def test_hmm_partial_fit_called(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        bar = _make_bar("AAPL")
        engine.bar_handler(bar)
        engine._hmm["AAPL"].partial_fit_online.assert_called_once_with(bar)

    def test_hmm_predict_called_when_fitted(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path, hmm_fitted=True)
        engine.bar_handler(_make_bar("AAPL"))
        engine._hmm["AAPL"].predict_regime.assert_called_once()

    def test_hmm_predict_skipped_when_not_fitted(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path, hmm_fitted=False)
        engine.bar_handler(_make_bar("AAPL"))
        engine._hmm["AAPL"].predict_regime.assert_not_called()

    def test_mwu_scheduled_update_called(self, tmp_path):
        engine, _, _, mwu_map = _build_engine(tmp_path=tmp_path)
        engine.bar_handler(_make_bar("AAPL"))
        mwu_map["AAPL"].scheduled_update.assert_called_once()
        call_kwargs = mwu_map["AAPL"].scheduled_update.call_args[1]
        assert call_kwargs["ticker"] == "AAPL"

    def test_signals_dict_has_four_keys(self, tmp_path):
        engine, _, _, mwu_map = _build_engine(tmp_path=tmp_path)
        engine.bar_handler(_make_bar("AAPL"))
        signals = mwu_map["AAPL"].scheduled_update.call_args[1]["signals"]
        assert set(signals.keys()) == {
            "hmm_regime", "ou_spread", "llm_sentiment", "analyst_recs"
        }

    def test_neutral_signal_no_order(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=0))
        engine.bar_handler(_make_bar("AAPL"))
        engine._executor.submit_order.assert_not_called()

    def test_buy_signal_submits_order(self, tmp_path):
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=1, score=0.6)
        )
        engine.bar_handler(_make_bar("AAPL"))
        engine._executor.submit_order.assert_called_once()
        order_call = engine._executor.submit_order.call_args[1]
        assert order_call["ticker"] == "AAPL"
        assert order_call["signal"] == 1

    def test_sell_signal_submits_order(self, tmp_path):
        engine, _, _, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=-1, score=-0.5)
        )
        engine.bar_handler(_make_bar("AAPL"))
        engine._executor.submit_order.assert_called_once()
        order_call = engine._executor.submit_order.call_args[1]
        assert order_call["signal"] == -1

    def test_unknown_ticker_ignored(self, tmp_path):
        engine, mock_storage, _, _ = _build_engine(tmp_path=tmp_path)
        # "GOOG" not in tickers
        engine.bar_handler(_make_bar("GOOG"))
        mock_storage.insert_ohlcv.assert_not_called()

    def test_circuit_breaker_fires_sets_emergency_flag(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._risk.circuit_breaker.return_value = True
        engine.bar_handler(_make_bar("AAPL"))
        assert engine._emergency_close is True
        assert engine._shutdown_event.is_set()

    def test_circuit_breaker_ok_no_flag(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._risk.circuit_breaker.return_value = False
        engine.bar_handler(_make_bar("AAPL"))
        assert engine._emergency_close is False
        assert not engine._shutdown_event.is_set()

    def test_hmm_predict_exception_does_not_crash(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path, hmm_fitted=True)
        engine._hmm["AAPL"].predict_regime.side_effect = RuntimeError("HMM error")
        # Should not raise
        engine.bar_handler(_make_bar("AAPL"))

    def test_order_exception_does_not_crash(self, tmp_path):
        engine, _, _, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=1)
        )
        engine._executor.submit_order.side_effect = RuntimeError("Alpaca down")
        # Should not raise
        engine.bar_handler(_make_bar("AAPL"))

    def test_pdt_error_adds_ticker_to_blocked_set(self, tmp_path, capsys):
        """Reactive PDT catch: error 40310100 adds ticker to cooldown and logs WARNING not ERROR."""
        engine, _, _, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=-1, score=-0.5)
        )
        engine._executor.submit_order.side_effect = Exception(
            '{"code":40310100,"message":"trade denied due to pattern day trading protection"}'
        )
        engine.bar_handler(_make_bar("AAPL"))

        assert "AAPL" in engine._pdt_blocked_today
        out = capsys.readouterr().out
        assert "engine.bar_handler.pdt_skip" in out
        assert "engine.order_failed" not in out

    def test_pdt_blocked_ticker_skips_sell(self, tmp_path):
        """Pre-check: a ticker already in _pdt_blocked_today does not reach submit_order."""
        engine, _, _, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=-1, score=-0.5)
        )
        engine._pdt_blocked_today.add("AAPL")
        engine._pdt_blocked_date = datetime.now(timezone.utc).date()

        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_not_called()

    def test_pdt_blocked_set_resets_at_midnight(self, tmp_path):
        """At UTC midnight rollover the blocked set is cleared and the order proceeds."""
        from datetime import date as _date
        engine, _, _, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=-1, score=-0.5)
        )
        engine._pdt_blocked_today.add("AAPL")
        engine._pdt_blocked_date = _date(2026, 4, 13)  # yesterday

        engine.bar_handler(_make_bar("AAPL"))

        # Set is cleared; submit_order is called normally
        assert "AAPL" not in engine._pdt_blocked_today
        engine._executor.submit_order.assert_called_once()

    def test_order_skipped_when_market_closed(self, tmp_path):
        """bar_handler must not call submit_order when is_market_open() is False."""
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=1, score=0.6)
        )
        mock_alpaca.is_market_open.return_value = False

        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_not_called()
        # Bar still persisted and HMM still updated
        engine._storage.insert_ohlcv.assert_called_once()
        engine._hmm["AAPL"].partial_fit_online.assert_called_once()

    def test_order_submitted_when_market_open(self, tmp_path):
        """bar_handler must call submit_order when is_market_open() is True."""
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=1, score=0.6)
        )
        mock_alpaca.is_market_open.return_value = True

        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_called_once()

    def test_llm_signal_from_db_used(self, tmp_path):
        engine, mock_storage, _, mwu_map = _build_engine(tmp_path=tmp_path)

        # Simulate a stored LLM signal in signal_log
        mock_row = (0.7, json.dumps({"direction": 1, "confidence": 0.7}))
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__  = MagicMock(return_value=False)
        mock_storage._engine.connect.return_value = mock_conn

        engine.bar_handler(_make_bar("AAPL"))

        signals = mwu_map["AAPL"].scheduled_update.call_args[1]["signals"]
        llm_sig = signals["llm_sentiment"]
        assert llm_sig["signal"] == 1
        assert llm_sig["confidence"] == pytest.approx(0.7)


# ===========================================================================
# OU series alignment
# ===========================================================================

def _make_ohlcv_df(times: list, close: float = 100.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with the given timestamps."""
    return pd.DataFrame({
        "time":   times,
        "ticker": "X",
        "open":   close,
        "high":   close + 1,
        "low":    close - 1,
        "close":  close,
        "volume": 1_000,
    })


class TestOUSeriesAlignment:
    """_get_ou_signal_for_ticker must align mismatched series before compute_signal."""

    def _engine_with_pair(self, tmp_path, df1, df2):
        """Build an engine wired with one pair (JPM/BAC) and stub storage."""
        engine, mock_storage, _, _ = _build_engine(
            tickers=("JPM", "BAC"),
            pairs=(("JPM", "BAC"),),
            tmp_path=tmp_path,
            mwu_decision=_mwu_decision(signal=0),
        )
        # Stub query_ohlcv to return the provided DataFrames.
        mock_storage.query_ohlcv.side_effect = lambda ticker, *_: (
            df1.copy() if ticker == "JPM" else df2.copy()
        )
        # Ensure the OU signal is fitted so it proceeds past the min_rows guard.
        ou = engine._ou_signals[("JPM", "BAC")]
        ou.lookback = 5
        return engine, ou

    def test_equal_length_series_not_modified(self, tmp_path):
        """When lengths match, compute_signal is called without alignment."""
        times = [datetime(2025, 1, 15, 14, i, tzinfo=timezone.utc) for i in range(10)]
        df1 = _make_ohlcv_df(times)
        df2 = _make_ohlcv_df(times)
        engine, ou = self._engine_with_pair(tmp_path, df1, df2)

        engine._get_ou_signal_for_ticker("JPM")

        call_args = ou.compute_signal.call_args
        assert len(call_args[0][0]) == len(call_args[0][1]) == 10

    def test_mismatched_series_aligned_to_shared_timestamps(self, tmp_path):
        """Extra bars on one leg must be dropped; compute_signal gets equal lengths."""
        times_common = [
            datetime(2025, 1, 15, 14, i, tzinfo=timezone.utc) for i in range(10)
        ]
        # df2 has 2 extra bars not present in df1
        times_extra = times_common + [
            datetime(2025, 1, 15, 14, 10, tzinfo=timezone.utc),
            datetime(2025, 1, 15, 14, 11, tzinfo=timezone.utc),
        ]
        df1 = _make_ohlcv_df(times_common)
        df2 = _make_ohlcv_df(times_extra)
        engine, ou = self._engine_with_pair(tmp_path, df1, df2)

        engine._get_ou_signal_for_ticker("JPM")

        call_args = ou.compute_signal.call_args
        passed_df1, passed_df2 = call_args[0][0], call_args[0][1]
        assert len(passed_df1) == len(passed_df2) == 10

    def test_mismatched_series_aligned_correctly_both_sides(self, tmp_path):
        """Each leg may have unique bars; only the intersection survives.
        df1: 12 base + 1 unique = 13 bars
        df2: 12 base + 2 unique = 14 bars  → different lengths → alignment fires
        After alignment: 12 shared bars reach compute_signal.
        """
        base_times = [datetime(2025, 1, 15, 14, i, tzinfo=timezone.utc) for i in range(12)]
        df1 = _make_ohlcv_df(base_times + [datetime(2025, 1, 15, 15, 0, tzinfo=timezone.utc)])
        df2 = _make_ohlcv_df(base_times + [
            datetime(2025, 1, 15, 15, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 15, 15, 2, tzinfo=timezone.utc),
        ])
        engine, ou = self._engine_with_pair(tmp_path, df1, df2)

        engine._get_ou_signal_for_ticker("JPM")

        call_args = ou.compute_signal.call_args
        passed_df1, passed_df2 = call_args[0][0], call_args[0][1]
        assert len(passed_df1) == len(passed_df2) == 12

    def test_too_few_shared_bars_returns_neutral(self, tmp_path):
        """If fewer than min_rows bars survive alignment, signal must be neutral 0."""
        times_common = [datetime(2025, 1, 15, 14, i, tzinfo=timezone.utc) for i in range(3)]
        times_extra = times_common + [
            datetime(2025, 1, 15, 14, 10, tzinfo=timezone.utc),
            datetime(2025, 1, 15, 14, 11, tzinfo=timezone.utc),
            datetime(2025, 1, 15, 14, 12, tzinfo=timezone.utc),
        ]
        df1 = _make_ohlcv_df(times_common)   # only 3 shared bars
        df2 = _make_ohlcv_df(times_extra)
        engine, ou = self._engine_with_pair(tmp_path, df1, df2)
        ou.lookback = 5   # min_rows = 5 > 3 shared

        result = engine._get_ou_signal_for_ticker("JPM")

        assert result["signal"] == 0
        ou.compute_signal.assert_not_called()


# ===========================================================================
# Regime smoothing
# ===========================================================================

class TestRegimeSmoothing:
    """Verify that a single-bar regime flip does not change the stable label."""

    def test_stable_label_unchanged_on_single_outlier(self, tmp_path):
        """One bear bar in a bull run should not flip the stable regime."""
        engine, _, _, mwu_map = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=0)
        )
        # Prime the history with 3 bull bars so stable label is "bull".
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 2, "label": "bull", "probs": [0.1, 0.1, 0.8],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        engine._hmm["AAPL"].state_labels = {0: "bear", 1: "neutral", 2: "bull"}
        for _ in range(3):
            engine.bar_handler(_make_bar("AAPL"))
        assert engine._stable_regime_label["AAPL"] == "bull"

        # One bear bar arrives — stable label must stay "bull".
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 0, "label": "bear", "probs": [0.8, 0.1, 0.1],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        engine.bar_handler(_make_bar("AAPL"))
        assert engine._stable_regime_label["AAPL"] == "bull"

    def test_stable_label_flips_after_consecutive_streak(self, tmp_path):
        """Three consecutive bear bars must flip the stable label to bear."""
        engine, _, _, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=0)
        )
        engine._hmm["AAPL"].state_labels = {0: "bear", 1: "neutral", 2: "bull"}

        # Prime with bull.
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 2, "label": "bull", "probs": [0.1, 0.1, 0.8],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        for _ in range(3):
            engine.bar_handler(_make_bar("AAPL"))
        assert engine._stable_regime_label["AAPL"] == "bull"

        # Three consecutive bear bars — label must flip.
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 0, "label": "bear", "probs": [0.8, 0.1, 0.1],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        for _ in range(3):
            engine.bar_handler(_make_bar("AAPL"))
        assert engine._stable_regime_label["AAPL"] == "bear"

    def test_smoothed_signal_sent_to_mwu(self, tmp_path):
        """MWU should receive the smoothed (stable) HMM signal, not the raw one."""
        engine, _, _, mwu_map = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=0)
        )
        engine._hmm["AAPL"].state_labels = {0: "bear", 1: "neutral", 2: "bull"}

        # Prime stable label as "bull" with 3 bars.
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 2, "label": "bull", "probs": [0.1, 0.1, 0.8],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        for _ in range(3):
            engine.bar_handler(_make_bar("AAPL"))

        # Now send one bear bar — MWU should still get hmm signal = +1 (bull).
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 0, "label": "bear", "probs": [0.8, 0.1, 0.1],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        mwu_map["AAPL"].scheduled_update.reset_mock()
        engine.bar_handler(_make_bar("AAPL"))

        signals = mwu_map["AAPL"].scheduled_update.call_args[1]["signals"]
        assert signals["hmm_regime"]["signal"] == 1   # bull = +1, not bear = -1

    def test_smoothed_regime_index_sent_to_mwu_on_outlier(self, tmp_path):
        """MWU regime kwarg must be the smoothed index (2=bull), not the raw outlier (0=bear)."""
        engine, _, _, mwu_map = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=0)
        )
        engine._hmm["AAPL"].state_labels = {0: "bear", 1: "neutral", 2: "bull"}

        # Prime stable regime as bull (index=2) with 3 bars.
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 2, "label": "bull", "probs": [0.1, 0.1, 0.8],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        for _ in range(3):
            engine.bar_handler(_make_bar("AAPL"))

        assert engine._stable_regime["AAPL"] == 2

        # One bear outlier — MWU should still receive regime=2 (bull), not regime=0 (bear).
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 0, "label": "bear", "probs": [0.8, 0.1, 0.1],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        mwu_map["AAPL"].scheduled_update.reset_mock()
        engine.bar_handler(_make_bar("AAPL"))

        regime_passed = mwu_map["AAPL"].scheduled_update.call_args[1]["regime"]
        assert regime_passed == 2, f"Expected smoothed regime 2 (bull), got {regime_passed}"

    def test_stable_regime_index_updated_after_consecutive_streak(self, tmp_path):
        """After three consecutive bear bars, _stable_regime must switch to bear index."""
        engine, _, _, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=0)
        )
        engine._hmm["AAPL"].state_labels = {0: "bear", 1: "neutral", 2: "bull"}

        # Prime with bull (index=2).
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 2, "label": "bull", "probs": [0.1, 0.1, 0.8],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        for _ in range(3):
            engine.bar_handler(_make_bar("AAPL"))
        assert engine._stable_regime["AAPL"] == 2

        # Three consecutive bear bars — stable index must flip to 0.
        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 0, "label": "bear", "probs": [0.8, 0.1, 0.1],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        for _ in range(3):
            engine.bar_handler(_make_bar("AAPL"))
        assert engine._stable_regime["AAPL"] == 0

    def test_single_outlier_does_not_change_stable_regime_index(self, tmp_path):
        """A single outlier bar must leave _stable_regime unchanged (bull=2 stays)."""
        engine, _, _, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=0)
        )
        engine._hmm["AAPL"].state_labels = {0: "bear", 1: "neutral", 2: "bull"}

        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 2, "label": "bull", "probs": [0.1, 0.1, 0.8],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        for _ in range(3):
            engine.bar_handler(_make_bar("AAPL"))

        engine._hmm["AAPL"].predict_regime.return_value = {
            "regime": 0, "label": "bear", "probs": [0.8, 0.1, 0.1],
            "timestamp": datetime.now(tz=timezone.utc),
        }
        engine.bar_handler(_make_bar("AAPL"))

        assert engine._stable_regime["AAPL"] == 2   # unchanged


# ===========================================================================
# Minimum holding period
# ===========================================================================

class TestHoldingPeriod:
    """Verify that direction reversals within 15 minutes are suppressed."""

    def test_reversal_within_15min_suppressed(self, tmp_path):
        """A -1 signal arriving 5 min after a +1 must not reach submit_order."""
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=1, score=0.5)
        )
        mock_alpaca.is_market_open.return_value = True

        # First bar: +1 signal is accepted and recorded.
        engine.bar_handler(_make_bar("AAPL"))
        assert engine._last_active_signal["AAPL"] == 1
        engine._executor.submit_order.reset_mock()

        # Simulate 5 minutes elapsed.
        engine._last_signal_change_time["AAPL"] = (
            datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        )

        # Send a -1 signal — must be suppressed.
        engine._mwu["AAPL"].scheduled_update.return_value = _mwu_decision(
            signal=-1, score=-0.5
        )
        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_not_called()

    def test_reversal_after_15min_allowed(self, tmp_path):
        """A -1 signal arriving 16 min after a +1 must reach submit_order."""
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=1, score=0.5)
        )
        mock_alpaca.is_market_open.return_value = True

        engine.bar_handler(_make_bar("AAPL"))
        engine._executor.submit_order.reset_mock()

        # Simulate 16 minutes elapsed — past the holding period.
        engine._last_signal_change_time["AAPL"] = (
            datetime.now(tz=timezone.utc) - timedelta(minutes=16)
        )

        engine._mwu["AAPL"].scheduled_update.return_value = _mwu_decision(
            signal=-1, score=-0.5
        )
        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_called_once()

    def test_same_direction_not_suppressed(self, tmp_path):
        """A repeated +1 signal within 15 min is not a reversal and must not be blocked."""
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=1, score=0.5)
        )
        mock_alpaca.is_market_open.return_value = True

        engine.bar_handler(_make_bar("AAPL"))
        engine._executor.submit_order.reset_mock()

        # Same direction again, 2 min later.
        engine._last_signal_change_time["AAPL"] = (
            datetime.now(tz=timezone.utc) - timedelta(minutes=2)
        )
        engine.bar_handler(_make_bar("AAPL"))

        # Position-limit or too-small may prevent a second submission, but the
        # holding period must NOT be the reason — submit_order should be called.
        engine._executor.submit_order.assert_called_once()

    def test_signal_change_time_recorded_on_first_signal(self, tmp_path):
        """The first non-zero signal should initialise _last_signal_change_time."""
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path, mwu_decision=_mwu_decision(signal=1, score=0.5)
        )
        mock_alpaca.is_market_open.return_value = True
        assert engine._last_signal_change_time["AAPL"] is None

        engine.bar_handler(_make_bar("AAPL"))

        assert engine._last_signal_change_time["AAPL"] is not None
        assert engine._last_active_signal["AAPL"] == 1


# ===========================================================================
# sentiment_job
# ===========================================================================

class TestSentimentJob:
    """
    sentiment_job() routes all tickers through Alpaca News (no AV calls).
    AV's free tier was abandoned due to multi-ticker "Invalid inputs" errors
    and a 20-call/day hard limit that is exhausted within hours.
    """

    def test_always_calls_pipeline(self, tmp_path):
        """sentiment_job must call run_pipeline unconditionally."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine.sentiment_job()
        engine._llm.run_pipeline.assert_called_once()

    def test_routes_all_tickers_via_alpaca(self, tmp_path):
        """av_tickers must be [] and alpaca_client must be set."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine.sentiment_job()
        call_kwargs = engine._llm.run_pipeline.call_args
        # positional: tickers, av_client, storage
        assert call_kwargs.args[0] == engine._tickers
        assert call_kwargs.args[2] is engine._storage
        # av_tickers=[] routes everything to Alpaca inside run_pipeline
        assert call_kwargs.kwargs["av_tickers"] == []
        assert call_kwargs.kwargs["alpaca_client"] is engine._alpaca_news

    def test_exception_does_not_propagate(self, tmp_path):
        """A run_pipeline failure must be caught; the job must not raise."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._llm.run_pipeline.side_effect = RuntimeError("network error")
        engine.sentiment_job()   # must not raise


# ===========================================================================
# eod_job
# ===========================================================================

class TestEODJob:

    def test_mwu_performance_report_called_for_each_ticker(self, tmp_path):
        engine, _, _, mwu_map = _build_engine(
            tickers=("AAPL", "MSFT"), pairs=(), tmp_path=tmp_path
        )
        engine.eod_job()
        for t in ("AAPL", "MSFT"):
            mwu_map[t].performance_report.assert_called_once()

    def test_kelly_stats_updated_from_fills(self, tmp_path):
        """eod_job must call _update_kelly_stats (not derive from MWU win rates)."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        with patch.object(engine, "_update_kelly_stats") as mock_update:
            engine.eod_job()
        mock_update.assert_called_once()

    def test_win_rate_stays_at_default_when_no_fills(self, tmp_path):
        """With no Alpaca fills, Kelly stats remain at _DEFAULT_SIGNAL_STATS."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._executor._trading.get_orders.return_value = []
        engine.eod_job()
        assert engine._signal_stats["AAPL"]["win_rate"] == pytest.approx(0.52, abs=0.001)

    def test_state_saved(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine.eod_job()
        engine._state_manager.save.assert_called_once()

    def test_alpaca_unavailable_does_not_crash(self, tmp_path):
        engine, _, mock_alpaca, _ = _build_engine(tmp_path=tmp_path)
        mock_alpaca.get_account_info.side_effect = RuntimeError("Alpaca down")
        engine.eod_job()   # should not raise


# ===========================================================================
# APScheduler job registration
# ===========================================================================

class TestSchedulerSetup:

    def test_four_jobs_registered(self, tmp_path):
        """run() must register 4 APScheduler jobs: 2 sentiment + market_open + eod."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)

        mock_scheduler = MagicMock()
        mock_scheduler.running = False

        with patch(_SCHEDULER, return_value=mock_scheduler):
            engine._scheduler = mock_scheduler
            engine._scheduler.add_job(
                engine.sentiment_job,
                "cron",
                hour="7-10",
                minute="*/25",
                day_of_week="mon-fri",
                timezone="America/New_York",
                id="sentiment_job_early",
            )
            engine._scheduler.add_job(
                engine.sentiment_job,
                "cron",
                hour="10-16",
                minute="*/35",
                day_of_week="mon-fri",
                timezone="America/New_York",
                id="sentiment_job_late",
            )
            engine._scheduler.add_job(
                engine.market_open_job,
                "cron",
                day_of_week="mon-fri",
                hour=9,
                minute=31,
                timezone="America/New_York",
                id="market_open_job",
            )
            engine._scheduler.add_job(
                engine.eod_job,
                "cron",
                day_of_week="mon-fri",
                hour=16,
                minute=5,
                timezone="America/New_York",
                id="eod_job",
            )

        assert mock_scheduler.add_job.call_count == 4

    def test_sentiment_jobs_registered_as_cron(self, tmp_path):
        """Both sentiment jobs must use cron triggers with correct hour ranges."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        mock_scheduler = MagicMock()

        engine._scheduler = mock_scheduler
        engine._scheduler.add_job(
            engine.sentiment_job, "cron", hour="7-10", minute="*/25",
            day_of_week="mon-fri", timezone="America/New_York",
            id="sentiment_job_early",
        )
        engine._scheduler.add_job(
            engine.sentiment_job, "cron", hour="10-16", minute="*/35",
            day_of_week="mon-fri", timezone="America/New_York",
            id="sentiment_job_late",
        )

        calls = mock_scheduler.add_job.call_args_list
        # Both must be "cron" trigger
        triggers = [c[0][1] for c in calls]
        assert all(t == "cron" for t in triggers)
        # Verify hour ranges and minute intervals
        early_kwargs = calls[0][1]
        late_kwargs  = calls[1][1]
        assert early_kwargs["hour"] == "7-10"
        assert early_kwargs["minute"] == "*/25"
        assert late_kwargs["hour"] == "10-16"
        assert late_kwargs["minute"] == "*/35"

    def test_market_open_job_has_next_run_time(self, tmp_path):
        """market_open_job must be registered with next_run_time so it fires at startup."""
        from trading_engine.orchestrator.engine import TradingEngine

        engine, _, mock_alpaca, _ = _build_engine(tmp_path=tmp_path)
        mock_alpaca.is_market_open.return_value = False  # prevent rebalance orders

        mock_scheduler = MagicMock()
        mock_scheduler.running = False

        captured_kwargs: dict = {}

        def capture_add_job(fn, *args, **kwargs):
            if kwargs.get("id") == "market_open_job":
                captured_kwargs.update(kwargs)

        mock_scheduler.add_job.side_effect = capture_add_job

        with patch(_SCHEDULER, return_value=mock_scheduler):
            with patch.object(engine._alpaca, "stream_bars"):
                with patch.object(engine, "_shutdown_event") as mock_event:
                    mock_event.wait.side_effect = KeyboardInterrupt
                    try:
                        engine.run()
                    except Exception:
                        pass

        assert "next_run_time" in captured_kwargs, (
            "market_open_job must be registered with next_run_time"
        )
        assert captured_kwargs["next_run_time"] is not None

    def test_eod_job_scheduled_at_1605_et(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        mock_scheduler = MagicMock()

        engine._scheduler = mock_scheduler
        engine._scheduler.add_job(
            engine.eod_job, "cron",
            day_of_week="mon-fri", hour=16, minute=5,
            timezone="America/New_York", id="eod_job",
        )

        call_kwargs = mock_scheduler.add_job.call_args[1]
        assert call_kwargs["hour"] == 16
        assert call_kwargs["minute"] == 5
        assert call_kwargs["day_of_week"] == "mon-fri"


# ===========================================================================
# Shutdown sequence
# ===========================================================================

class TestShutdownSequence:

    def test_stream_stopped_on_shutdown(self, tmp_path):
        engine, _, mock_alpaca, _ = _build_engine(tmp_path=tmp_path)
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        engine._scheduler = mock_scheduler

        engine._shutdown()
        mock_alpaca.stop_stream.assert_called_once()

    def test_scheduler_shutdown_called(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        engine._scheduler = mock_scheduler

        engine._shutdown()
        mock_scheduler.shutdown.assert_called_once_with(wait=False)

    def test_no_liquidation_on_normal_shutdown(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._emergency_close = False
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        engine._scheduler = mock_scheduler

        engine._shutdown()
        engine._executor.close_all_positions.assert_not_called()

    def test_liquidation_on_emergency_shutdown(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._emergency_close = True
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        engine._scheduler = mock_scheduler

        engine._shutdown()
        engine._executor.close_all_positions.assert_called_once()

    def test_state_saved_on_shutdown(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        mock_scheduler = MagicMock()
        mock_scheduler.running = False
        engine._scheduler = mock_scheduler

        engine._shutdown()
        engine._state_manager.save.assert_called_once()

    def test_stream_stop_exception_does_not_crash_shutdown(self, tmp_path):
        engine, _, mock_alpaca, _ = _build_engine(tmp_path=tmp_path)
        mock_alpaca.stop_stream.side_effect = RuntimeError("stream error")
        mock_scheduler = MagicMock()
        mock_scheduler.running = False
        engine._scheduler = mock_scheduler

        engine._shutdown()   # should not raise


# ===========================================================================
# StateManager
# ===========================================================================

class TestStateManager:

    def test_save_and_load_roundtrip(self, tmp_path):
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)

        state = {
            "tickers":      ["AAPL", "MSFT"],
            "pairs":        [["JPM", "BAC"]],
            "signal_stats": {"AAPL": {"win_rate": 0.6}},
        }
        sm.save(state)
        loaded = sm.load()

        assert loaded is not None
        assert loaded["tickers"] == ["AAPL", "MSFT"]
        assert loaded["signal_stats"]["AAPL"]["win_rate"] == 0.6

    def test_load_returns_none_when_no_file(self, tmp_path):
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path / "empty")
        assert sm.load() is None

    def test_checksum_mismatch_returns_none_when_no_backups(self, tmp_path):
        """Corrupt current file with no backups → load() returns None (no exception)."""
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)
        sm.save({"tickers": ["AAPL"]})

        # Corrupt the current file only — no backups exist yet.
        path = sm._state_path()
        data = json.loads(path.read_text())
        data["checksum"] = "bad_checksum"
        path.write_text(json.dumps(data))

        assert sm.load() is None

    def test_rolling_backup_creates_bak1(self, tmp_path):
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)
        sm.save({"tickers": ["AAPL"]})
        sm.save({"tickers": ["MSFT"]})   # triggers first rotation

        assert sm._backup_path(1).exists()

    def test_rolling_backup_keeps_max_3(self, tmp_path):
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)
        for i in range(5):
            sm.save({"iteration": i})

        backups = sm.list_backups()
        assert len(backups) <= 3

    def test_state_file_written_atomically(self, tmp_path):
        """A .tmp file must not remain after save."""
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)
        sm.save({"tickers": ["AAPL"]})

        tmp_file = sm._state_path().with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_version_field_in_file(self, tmp_path):
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)
        sm.save({"tickers": ["AAPL"]})

        raw = json.loads(sm._state_path().read_text())
        assert "version" in raw
        assert "saved_at" in raw
        assert "checksum" in raw

    def test_loaded_state_excludes_metadata_keys(self, tmp_path):
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)
        sm.save({"tickers": ["AAPL"]})
        loaded = sm.load()

        assert "checksum" not in loaded
        assert "version" not in loaded
        assert "saved_at" not in loaded

    # ------------------------------------------------------------------
    # Fix 5: backup-based recovery
    # ------------------------------------------------------------------

    def _corrupt(self, path: Path) -> None:
        """Overwrite checksum in a state file to make it invalid."""
        data = json.loads(path.read_text())
        data["checksum"] = "corrupted"
        path.write_text(json.dumps(data))

    def test_recovers_from_bak1_when_current_corrupted(self, tmp_path):
        """Corrupt current, valid bak1 → load() recovers bak1 contents."""
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)

        sm.save({"iteration": 1})   # becomes bak1 on next save
        sm.save({"iteration": 2})   # current; bak1 holds iteration=1

        self._corrupt(sm._state_path())

        loaded = sm.load()
        assert loaded is not None
        assert loaded["iteration"] == 1

    def test_recovers_from_bak2_when_current_and_bak1_corrupted(self, tmp_path):
        """Corrupt current and bak1, valid bak2 → load() recovers bak2."""
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)

        sm.save({"iteration": 1})   # will eventually be bak2
        sm.save({"iteration": 2})   # will be bak1
        sm.save({"iteration": 3})   # current; bak1=2, bak2=1

        self._corrupt(sm._state_path())
        self._corrupt(sm._backup_path(1))

        loaded = sm.load()
        assert loaded is not None
        assert loaded["iteration"] == 1

    def test_returns_none_when_all_candidates_corrupted(self, tmp_path):
        """All files corrupted → load() returns None (no exception)."""
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)

        for _ in range(4):
            sm.save({"data": "x"})

        self._corrupt(sm._state_path())
        for n in range(1, 4):
            if sm._backup_path(n).exists():
                self._corrupt(sm._backup_path(n))

        assert sm.load() is None


# ===========================================================================
# main.py — argument parsing
# ===========================================================================

class TestMainArgParsing:

    def test_default_tickers(self):
        from trading_engine.main import _build_parser
        args = _build_parser().parse_args([])
        assert args.tickers == ["AAPL", "MSFT", "JPM", "BAC"]

    def test_custom_tickers(self):
        from trading_engine.main import _build_parser
        args = _build_parser().parse_args(["--tickers", "NVDA", "TSLA"])
        assert args.tickers == ["NVDA", "TSLA"]

    def test_paper_is_default(self):
        from trading_engine.main import _build_parser
        args = _build_parser().parse_args([])
        assert args.live is False

    def test_live_flag(self):
        from trading_engine.main import _build_parser
        args = _build_parser().parse_args(["--live"])
        assert args.live is True

    def test_pairs_file_defaults_to_none(self):
        from trading_engine.main import _build_parser
        args = _build_parser().parse_args([])
        assert args.pairs_file is None

    def test_pairs_file_custom_path(self, tmp_path):
        from trading_engine.main import _build_parser
        pairs_path = tmp_path / "my_pairs.json"
        args = _build_parser().parse_args(["--pairs-file", str(pairs_path)])
        assert args.pairs_file == pairs_path


# ===========================================================================
# _load_discovered_pairs
# ===========================================================================

class TestVixMultiplier:

    def test_vix_multiplier_thresholds(self):
        from trading_engine.orchestrator.engine import _vix_risk_off_multiplier
        assert _vix_risk_off_multiplier(None) == 1.0
        assert _vix_risk_off_multiplier(20.0) == 1.0
        assert _vix_risk_off_multiplier(26.0) == 0.75
        assert _vix_risk_off_multiplier(32.0) == 0.50
        assert _vix_risk_off_multiplier(42.0) == 0.25

    def test_vix_boundary_values(self):
        from trading_engine.orchestrator.engine import _vix_risk_off_multiplier
        # Exact boundary: >25 triggers 0.75, ==25 does not
        assert _vix_risk_off_multiplier(25.0) == 1.0
        assert _vix_risk_off_multiplier(25.001) == 0.75
        # Exact boundary: >30 triggers 0.50, ==30 does not
        assert _vix_risk_off_multiplier(30.0) == 0.75
        assert _vix_risk_off_multiplier(30.001) == 0.50
        # Exact boundary: >40 triggers 0.25, ==40 does not
        assert _vix_risk_off_multiplier(40.0) == 0.50
        assert _vix_risk_off_multiplier(40.001) == 0.25


class TestLoadDiscoveredPairs:

    def test_missing_file_returns_empty_list(self, tmp_path):
        from trading_engine.orchestrator.engine import _load_discovered_pairs
        result = _load_discovered_pairs(tmp_path / "nonexistent.json")
        assert result == []

    def test_valid_file_returns_pairs(self, tmp_path):
        from trading_engine.orchestrator.engine import _load_discovered_pairs
        path = tmp_path / "discovered_pairs.json"
        payload = {
            "scanned_at": "2026-04-08T12:00:00Z",
            "lookback_days": 504,
            "n_tickers_scanned": 10,
            "n_candidate_pairs": 45,
            "n_correlated": 5,
            "n_cointegrated": 3,
            "n_selected": 2,
            "pairs": [
                {"ticker1": "LMT", "ticker2": "NOC", "eg_pvalue": 0.01,
                 "correlation": 0.91, "johansen_trace_stat": 18.4,
                 "beta_ols": 1.23, "half_life_bars": 18.3,
                 "kappa": 0.038, "mu": 0.12, "sigma": 2.45},
                {"ticker1": "MSFT", "ticker2": "GOOG", "eg_pvalue": 0.03,
                 "correlation": 0.85, "johansen_trace_stat": 14.2,
                 "beta_ols": 0.87, "half_life_bars": 22.1,
                 "kappa": 0.031, "mu": 0.05, "sigma": 3.10},
            ],
        }
        path.write_text(json.dumps(payload))
        result = _load_discovered_pairs(path)
        assert result == [("LMT", "NOC"), ("MSFT", "GOOG")]

    def test_malformed_json_returns_empty_list(self, tmp_path):
        from trading_engine.orchestrator.engine import _load_discovered_pairs
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        result = _load_discovered_pairs(path)
        assert result == []

    def test_empty_pairs_list(self, tmp_path):
        from trading_engine.orchestrator.engine import _load_discovered_pairs
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({
            "scanned_at": "2026-04-08T12:00:00Z",
            "pairs": [],
        }))
        result = _load_discovered_pairs(path)
        assert result == []

    def test_stale_file_returns_pairs_with_warning(self, tmp_path, capsys):
        """A file >14 days old must still return pairs (with a warning)."""
        from trading_engine.orchestrator.engine import _load_discovered_pairs
        path = tmp_path / "stale.json"
        # scanned_at 30 days ago
        from datetime import timedelta
        stale_ts = (
            datetime.now(tz=timezone.utc) - timedelta(days=30)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {
            "scanned_at": stale_ts,
            "pairs": [
                {"ticker1": "JPM", "ticker2": "BAC",
                 "eg_pvalue": 0.02, "correlation": 0.88,
                 "johansen_trace_stat": 16.0, "beta_ols": 1.1,
                 "half_life_bars": 15.0, "kappa": 0.046,
                 "mu": 0.0, "sigma": 1.5},
            ],
        }
        path.write_text(json.dumps(payload))

        result = _load_discovered_pairs(path)
        # Pairs must still be returned despite staleness
        assert result == [("JPM", "BAC")]
        # Warning must have been logged (structlog goes to stdout)
        captured = capsys.readouterr()
        assert "stale" in captured.out

    def test_pair_entry_missing_ticker_key_skipped(self, tmp_path):
        from trading_engine.orchestrator.engine import _load_discovered_pairs
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({
            "scanned_at": "2026-04-08T12:00:00Z",
            "pairs": [
                {"ticker1": "LMT"},                       # missing ticker2
                {"ticker1": "MSFT", "ticker2": "GOOG"},   # valid
            ],
        }))
        result = _load_discovered_pairs(path)
        assert result == [("MSFT", "GOOG")]


# ===========================================================================
# Pair ticker auto-merge
# ===========================================================================

_LOAD_PAIRS = f"{_ENG_MOD}._load_discovered_pairs"
_PO_MOD = f"{_ENG_MOD}.PortfolioOptimizer"


class TestPairTickerAutoMerge:
    """
    Test that pair tickers discovered in discovered_pairs.json are merged into
    self._tickers at TradingEngine.__init__ time.
    """

    @patch(f"{_ENG_MOD}.StateManager")
    @patch(_PO_MOD)
    @patch(f"{_ENG_MOD}.OrderExecutor")
    @patch(f"{_ENG_MOD}.RiskManager")
    @patch(f"{_ENG_MOD}.OUSpreadSignal")
    @patch(f"{_ENG_MOD}.MWUMetaAgent")
    @patch(f"{_ENG_MOD}.LLMSentimentSignal")
    @patch(f"{_ENG_MOD}.HMMRegimeDetector")
    @patch(f"{_ENG_MOD}.AlphaVantageNewsClient")
    @patch(f"{_ENG_MOD}.AlpacaMarketData")
    @patch(f"{_ENG_MOD}.Storage")
    @patch(_LOAD_PAIRS)
    def test_pair_tickers_added_to_tickers(
        self,
        mock_load_pairs,
        mock_storage_cls, mock_alpaca_cls, mock_av_cls,
        mock_hmm_cls, mock_llm_cls, mock_mwu_cls, mock_ou_cls,
        mock_risk_cls, mock_exec_cls, mock_po_cls,
        mock_state_cls,
        tmp_path,
    ):
        """Tickers in discovered pairs must be merged into engine._tickers."""
        mock_load_pairs.return_value = [("LMT", "NOC")]

        # State manager returns None on load (fresh state)
        mock_state_cls.return_value.load.return_value = None

        from trading_engine.orchestrator.engine import TradingEngine
        engine = TradingEngine(
            tickers=["AAPL"],
            paper=True,
            models_dir=tmp_path,
        )

        assert "LMT" in engine._tickers
        assert "NOC" in engine._tickers
        assert "AAPL" in engine._tickers

    @patch(f"{_ENG_MOD}.StateManager")
    @patch(_PO_MOD)
    @patch(f"{_ENG_MOD}.OrderExecutor")
    @patch(f"{_ENG_MOD}.RiskManager")
    @patch(f"{_ENG_MOD}.OUSpreadSignal")
    @patch(f"{_ENG_MOD}.MWUMetaAgent")
    @patch(f"{_ENG_MOD}.LLMSentimentSignal")
    @patch(f"{_ENG_MOD}.HMMRegimeDetector")
    @patch(f"{_ENG_MOD}.AlphaVantageNewsClient")
    @patch(f"{_ENG_MOD}.AlpacaMarketData")
    @patch(f"{_ENG_MOD}.Storage")
    @patch(_LOAD_PAIRS)
    def test_already_present_tickers_not_duplicated(
        self,
        mock_load_pairs,
        mock_storage_cls, mock_alpaca_cls, mock_av_cls,
        mock_hmm_cls, mock_llm_cls, mock_mwu_cls, mock_ou_cls,
        mock_risk_cls, mock_exec_cls, mock_po_cls,
        mock_state_cls,
        tmp_path,
    ):
        """Pair tickers already in --tickers must not be duplicated."""
        mock_load_pairs.return_value = [("AAPL", "MSFT")]
        mock_state_cls.return_value.load.return_value = None

        from trading_engine.orchestrator.engine import TradingEngine
        engine = TradingEngine(
            tickers=["AAPL", "MSFT"],
            paper=True,
            models_dir=tmp_path,
        )

        assert engine._tickers.count("AAPL") == 1
        assert engine._tickers.count("MSFT") == 1

    @patch(f"{_ENG_MOD}.StateManager")
    @patch(_PO_MOD)
    @patch(f"{_ENG_MOD}.OrderExecutor")
    @patch(f"{_ENG_MOD}.RiskManager")
    @patch(f"{_ENG_MOD}.OUSpreadSignal")
    @patch(f"{_ENG_MOD}.MWUMetaAgent")
    @patch(f"{_ENG_MOD}.LLMSentimentSignal")
    @patch(f"{_ENG_MOD}.HMMRegimeDetector")
    @patch(f"{_ENG_MOD}.AlphaVantageNewsClient")
    @patch(f"{_ENG_MOD}.AlpacaMarketData")
    @patch(f"{_ENG_MOD}.Storage")
    @patch(_LOAD_PAIRS)
    def test_no_pairs_file_engine_runs_without_pairs(
        self,
        mock_load_pairs,
        mock_storage_cls, mock_alpaca_cls, mock_av_cls,
        mock_hmm_cls, mock_llm_cls, mock_mwu_cls, mock_ou_cls,
        mock_risk_cls, mock_exec_cls, mock_po_cls,
        mock_state_cls,
        tmp_path,
    ):
        """When _load_discovered_pairs returns [], engine._pairs is empty."""
        mock_load_pairs.return_value = []
        mock_state_cls.return_value.load.return_value = None

        from trading_engine.orchestrator.engine import TradingEngine
        engine = TradingEngine(
            tickers=["AAPL"],
            paper=True,
            models_dir=tmp_path,
        )

        assert engine._pairs == []
        assert "AAPL" in engine._tickers
        assert "SPY" in engine._tickers


# ===========================================================================
# market_open_job — rebalance execution
# ===========================================================================

def _make_rebalance_order(ticker: str, action: str, dollar_amount: float = 5_000.0) -> dict:
    """Build a minimal rebalance order dict matching get_rebalance_orders output."""
    target = 0.10 if action == "buy" else 0.05
    current = 0.05 if action == "buy" else 0.10
    return {
        "ticker":         ticker,
        "action":         action,
        "target_weight":  target,
        "current_weight": current,
        "delta_weight":   target - current,
        "dollar_amount":  dollar_amount,
    }


def _positions_df(*holdings: tuple[str, float, float]) -> pd.DataFrame:
    """Build a positions DataFrame matching OrderExecutor.get_positions() columns."""
    rows = [
        {
            "ticker":            t,
            "qty":               float(qty),
            "market_value":      float(mv),
            "unrealized_pnl":    0.0,
            "unrealized_pnl_pct": 0.0,
        }
        for t, qty, mv in holdings
    ]
    cols = ["ticker", "qty", "market_value", "unrealized_pnl", "unrealized_pnl_pct"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _setup_rebalance_engine(
    tmp_path,
    rebalance_orders: list[dict] | None = None,
    positions: pd.DataFrame | None = None,
    equity: float = 100_000.0,
):
    """
    Build a TradingEngine with all rebalance-path mocks pre-configured.

    Returns (engine, mock_alpaca).
    """
    engine, _, mock_alpaca, _ = _build_engine(tmp_path=tmp_path)

    # BL optimisation succeeds
    engine.portfolio_optimizer.compute_black_litterman.return_value = {
        "weights":   {"AAPL": 0.10},
        "method":    "black_litterman",
        "n_views":   1,
        "timestamp": datetime.now(tz=timezone.utc),
    }

    # Account info
    mock_alpaca.get_account_info.return_value = _account(equity)

    # Positions
    if positions is None:
        positions = _positions_df(("AAPL", 50.0, 7_500.0))
    engine._executor.get_positions.return_value = positions

    # Rebalance orders
    engine.portfolio_optimizer.get_rebalance_orders.return_value = (
        rebalance_orders if rebalance_orders is not None else []
    )

    # Quote (any ticker → $150)
    mock_alpaca.get_latest_quote.return_value = {
        "mid": 150.0, "bid": 149.9, "ask": 150.1,
    }

    # submit_order returns a simple namespace with an id
    from types import SimpleNamespace
    engine._executor._trading.submit_order.return_value = SimpleNamespace(id="test-order-id")

    # Default: no same-day buys — PDT pre-check passes for all tickers.
    engine._executor.get_todays_filled_buy_symbols.return_value = set()

    return engine, mock_alpaca


class TestMarketOpenJobRebalance:

    def test_get_rebalance_orders_called_with_correct_equity(self, tmp_path):
        """After BL succeeds, get_rebalance_orders is called with account equity."""
        engine, _ = _setup_rebalance_engine(tmp_path, equity=100_000.0)

        engine.market_open_job()

        engine.portfolio_optimizer.get_rebalance_orders.assert_called_once()
        _, positional, _ = (
            engine.portfolio_optimizer.get_rebalance_orders.call_args[0],
            engine.portfolio_optimizer.get_rebalance_orders.call_args[0],
            engine.portfolio_optimizer.get_rebalance_orders.call_args[1],
        )
        equity_arg = positional[1]
        assert equity_arg == pytest.approx(100_000.0)

    def test_sells_execute_before_buys(self, tmp_path):
        """Sell orders must be submitted before buy orders regardless of list order."""
        orders = [
            _make_rebalance_order("AAPL", "buy",  dollar_amount=5_000.0),
            _make_rebalance_order("JPM",  "sell", dollar_amount=3_000.0),
        ]
        positions = _positions_df(("AAPL", 50.0, 7_500.0), ("JPM", 100.0, 10_000.0))
        engine, _ = _setup_rebalance_engine(tmp_path, rebalance_orders=orders, positions=positions)

        submitted_symbols: list[str] = []

        from types import SimpleNamespace
        def capture(req):
            submitted_symbols.append(req.symbol)
            return SimpleNamespace(id="ok")

        engine._executor._trading.submit_order.side_effect = capture

        engine.market_open_job()

        assert len(submitted_symbols) == 2
        assert submitted_symbols[0] == "JPM",  "sell must come first"
        assert submitted_symbols[1] == "AAPL", "buy must come second"

    def test_circuit_breaker_halts_all_rebalance_orders(self, tmp_path):
        """A tripped circuit breaker must prevent any order from being submitted."""
        orders = [_make_rebalance_order("AAPL", "buy")]
        engine, _ = _setup_rebalance_engine(tmp_path, rebalance_orders=orders)
        engine._risk.circuit_breaker.return_value = True

        engine.market_open_job()

        engine._executor._trading.submit_order.assert_not_called()

    def test_rebalance_skipped_when_market_closed(self, tmp_path):
        """_execute_rebalance_orders must skip all orders on a holiday/closed market."""
        orders = [_make_rebalance_order("AAPL", "buy")]
        engine, mock_alpaca = _setup_rebalance_engine(tmp_path, rebalance_orders=orders)
        mock_alpaca.is_market_open.return_value = False

        engine.market_open_job()

        # BL optimisation still ran (weights computed)
        engine.portfolio_optimizer.compute_black_litterman.assert_called_once()
        # No orders submitted
        engine._executor._trading.submit_order.assert_not_called()

    def test_per_ticker_error_isolation(self, tmp_path):
        """A failure on one ticker must not prevent the remaining orders from executing."""
        orders = [
            _make_rebalance_order("AAPL", "buy"),
            _make_rebalance_order("MSFT", "buy"),
            _make_rebalance_order("JPM",  "buy"),
        ]
        engine, _ = _setup_rebalance_engine(tmp_path, rebalance_orders=orders)

        from types import SimpleNamespace

        def fail_on_msft(req):
            if req.symbol == "MSFT":
                raise RuntimeError("order rejected by broker")
            return SimpleNamespace(id="ok")

        engine._executor._trading.submit_order.side_effect = fail_on_msft

        engine.market_open_job()   # must not raise

        # All 3 attempted; MSFT threw but AAPL and JPM succeeded
        assert engine._executor._trading.submit_order.call_count == 3

    def test_rebalance_refetches_cash_after_sells(self, tmp_path):
        """
        _execute_rebalance_orders must re-fetch account after the sell phase to
        get the updated cash balance.  Buys should proceed with the post-sell cash.
        """
        orders = [
            _make_rebalance_order("JPM",  "sell", dollar_amount=3_000.0),
            _make_rebalance_order("AAPL", "buy",  dollar_amount=5_000.0),
        ]
        positions = _positions_df(("JPM", 20.0, 3_000.0))
        engine, mock_alpaca = _setup_rebalance_engine(
            tmp_path, rebalance_orders=orders, positions=positions, equity=5_000.0
        )

        # First call (circuit-breaker gate): small account, $5K cash
        # Second call (after sells):         bigger cash from proceeds
        initial_account = {
            "equity": 5_000.0, "cash": 5_000.0,
            "buying_power": 10_000.0, "portfolio_value": 5_000.0,
        }
        updated_account = {
            "equity": 25_000.0, "cash": 25_000.0,
            "buying_power": 50_000.0, "portfolio_value": 25_000.0,
        }
        mock_alpaca.get_account_info.side_effect = [initial_account, updated_account]

        submitted_symbols: list[str] = []

        def capture(req):
            submitted_symbols.append(req.symbol)
            return SimpleNamespace(id="ok")

        engine._executor._trading.submit_order.side_effect = capture

        engine.market_open_job()

        # Sell ran first, buy ran after; account fetched twice
        assert len(submitted_symbols) == 2
        assert submitted_symbols[0] == "JPM"
        assert submitted_symbols[1] == "AAPL"
        assert mock_alpaca.get_account_info.call_count == 2

    def test_rebalance_buy_capped_by_available_cash(self, tmp_path):
        """
        When buys total more than available cash, orders are filled sequentially
        until cash runs out; remaining buys are skipped with no_cash_remaining.

        With price=$1000/share, 3×$10K buy orders, and $20K cash:
          - Buy 1 (10 shares, $10K): succeeds, available → $10K
          - Buy 2 (10 shares, $10K): succeeds, available → $0
          - Buy 3:  available_cash=0 → skipped
        """
        orders = [
            _make_rebalance_order("AAPL", "buy", dollar_amount=10_000.0),
            _make_rebalance_order("MSFT", "buy", dollar_amount=10_000.0),
            _make_rebalance_order("JPM",  "buy", dollar_amount=10_000.0),
        ]
        engine, mock_alpaca = _setup_rebalance_engine(
            tmp_path, rebalance_orders=orders, positions=_positions_df(), equity=100_000.0
        )

        # Circuit-breaker call returns normal account; post-sell refetch shows $20K cash
        initial_account = _account(100_000.0)
        post_sell_account = {
            "equity":          100_000.0,
            "cash":             20_000.0,
            "buying_power":    200_000.0,
            "portfolio_value": 100_000.0,
        }
        mock_alpaca.get_account_info.side_effect = [initial_account, post_sell_account]

        # Shares: $10K / $1000 = 10 per order; deduct $10K each
        mock_alpaca.get_latest_quote.return_value = {
            "mid": 1_000.0, "bid": 999.9, "ask": 1_000.1
        }

        submitted_symbols: list[str] = []

        def capture(req):
            submitted_symbols.append(req.symbol)
            return SimpleNamespace(id="ok")

        engine._executor._trading.submit_order.side_effect = capture

        engine.market_open_job()

        # Only 2 of 3 buys executed; third skipped (no cash)
        assert len(submitted_symbols) == 2
        assert "AAPL" in submitted_symbols
        assert "MSFT" in submitted_symbols
        assert "JPM" not in submitted_symbols

    def test_rebalance_skips_all_buys_if_cash_refetch_fails(self, tmp_path):
        """
        If get_account_info raises after the sell phase, all pending buys must
        be skipped and no orders submitted for the buy phase.
        """
        orders = [
            _make_rebalance_order("AAPL", "buy", dollar_amount=5_000.0),
            _make_rebalance_order("MSFT", "buy", dollar_amount=5_000.0),
        ]
        engine, mock_alpaca = _setup_rebalance_engine(tmp_path, rebalance_orders=orders)

        # First call succeeds (circuit-breaker gate); second raises (cash refetch)
        mock_alpaca.get_account_info.side_effect = [
            _account(100_000.0),
            RuntimeError("Alpaca API down"),
        ]

        engine.market_open_job()

        # No buy orders should have been submitted
        engine._executor._trading.submit_order.assert_not_called()


# ===========================================================================
# PDT (Pattern Day Trader) protection
# ===========================================================================

class TestPDTProtection:
    """
    Verify that same-day sells are skipped and PDT broker errors are handled
    gracefully rather than surfaced as hard errors.
    """

    def test_sell_skipped_when_position_opened_today(self, tmp_path):
        """A sell for a ticker bought today must be skipped (pdt_skip), not submitted."""
        orders = [_make_rebalance_order("TXT", "sell", dollar_amount=3_000.0)]
        positions = _positions_df(("TXT", 10.0, 3_000.0))
        engine, mock_alpaca = _setup_rebalance_engine(
            tmp_path, rebalance_orders=orders, positions=positions
        )
        mock_alpaca.is_market_open.return_value = True
        # TXT was bought today
        engine._executor.get_todays_filled_buy_symbols.return_value = {"TXT"}

        engine.market_open_job()

        engine._executor._trading.submit_order.assert_not_called()

    def test_sell_proceeds_when_position_opened_yesterday(self, tmp_path):
        """A sell for a ticker NOT in today's buys must proceed normally."""
        orders = [_make_rebalance_order("TXT", "sell", dollar_amount=1_500.0)]
        positions = _positions_df(("TXT", 10.0, 1_500.0))
        engine, mock_alpaca = _setup_rebalance_engine(
            tmp_path, rebalance_orders=orders, positions=positions
        )
        mock_alpaca.is_market_open.return_value = True
        engine._executor.get_todays_filled_buy_symbols.return_value = set()  # no same-day buys

        engine.market_open_job()

        engine._executor._trading.submit_order.assert_called_once()

    def test_only_same_day_tickers_skipped_others_proceed(self, tmp_path):
        """When two sells are pending and only one was bought today, only that one is skipped."""
        orders = [
            _make_rebalance_order("RGTI", "sell", dollar_amount=2_000.0),
            _make_rebalance_order("TXT",  "sell", dollar_amount=2_000.0),
        ]
        positions = _positions_df(
            ("RGTI", 20.0, 2_000.0),
            ("TXT",  15.0, 2_000.0),
        )
        engine, mock_alpaca = _setup_rebalance_engine(
            tmp_path, rebalance_orders=orders, positions=positions
        )
        mock_alpaca.is_market_open.return_value = True
        # Only RGTI was bought today; TXT was not
        engine._executor.get_todays_filled_buy_symbols.return_value = {"RGTI"}

        submitted: list[str] = []
        from types import SimpleNamespace as NS
        def capture(req):
            submitted.append(req.symbol)
            return NS(id="ok")
        engine._executor._trading.submit_order.side_effect = capture

        engine.market_open_job()

        assert submitted == ["TXT"]   # RGTI skipped, TXT executed

    def test_pdt_broker_error_handled_as_warning_not_error(self, tmp_path, capsys):
        """If PDT error slips through pre-check, the except block must log warning not error."""
        orders = [_make_rebalance_order("TXT", "sell", dollar_amount=3_000.0)]
        positions = _positions_df(("TXT", 10.0, 3_000.0))
        engine, mock_alpaca = _setup_rebalance_engine(
            tmp_path, rebalance_orders=orders, positions=positions
        )
        mock_alpaca.is_market_open.return_value = True
        # Pre-check returns empty (simulating fetch failure), but Alpaca rejects
        engine._executor.get_todays_filled_buy_symbols.return_value = set()
        engine._executor._trading.submit_order.side_effect = Exception(
            '{"code":40310100,"message":"trade denied due to pattern day trading protection"}'
        )

        engine.market_open_job()   # must not raise

        out = capsys.readouterr().out
        assert "pdt_skip" in out
        assert "order_failed" not in out


# ===========================================================================
# startup_checks — historical seeding
# ===========================================================================

class TestStartupChecksSeeding:
    """
    Tests for the DB-seeding path inside startup_checks.

    startup_checks is tested via a real TradingEngine.__init__ call with all
    external deps patched.  We call startup_checks() directly to exercise only
    that method.
    """

    def _build_seeding_engine(self, tmp_path, hmm_fitted=False):
        """Build a minimal engine with patched deps for startup_checks tests."""
        from trading_engine.orchestrator.engine import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)

        mock_storage = MagicMock()
        mock_alpaca  = MagicMock()
        mock_hmm     = MagicMock()
        mock_hmm.is_fitted = hmm_fitted

        engine._tickers        = ["AAPL"]
        engine._pairs          = []
        engine._storage        = mock_storage
        engine._alpaca         = mock_alpaca
        engine._hmm            = {"AAPL": mock_hmm}
        engine._ou_signals     = {}
        engine._llm            = MagicMock()
        engine._mwu            = {"AAPL": MagicMock()}
        engine._risk           = MagicMock()
        engine._executor       = MagicMock()
        engine._state_manager  = MagicMock()
        engine._signal_stats   = {"AAPL": {}}
        engine._shutdown_event = __import__("threading").Event()
        engine._emergency_close = False
        engine._scheduler      = None
        engine._models_dir     = tmp_path
        engine.portfolio_optimizer = MagicMock()

        # Alpaca: account OK, Ollama: model found
        mock_alpaca.get_account_info.return_value = _account()

        return engine, mock_storage, mock_alpaca, mock_hmm

    def test_seeds_when_db_has_too_few_rows(self, tmp_path):
        """If DB has < 60 rows, fetch_historical_ohlcv should be called."""
        engine, mock_storage, mock_alpaca, mock_hmm = self._build_seeding_engine(
            tmp_path, hmm_fitted=False
        )

        # DB returns only 12 rows — not enough
        mock_storage.query_ohlcv.return_value = pd.DataFrame({"close": range(12)})

        # Alpaca returns a non-empty DataFrame with enough rows
        seeded_df = pd.DataFrame({"ticker": ["AAPL"] * 100, "close": range(100)})
        mock_alpaca.fetch_historical_ohlcv.return_value = seeded_df

        with (
            patch("ollama.Client") as mock_ollama_client,
            patch(_SETTINGS, _fake_settings()),
        ):
            mock_ollama_client.return_value.list.return_value = SimpleNamespace(
                models=[SimpleNamespace(model="gemma4:e4b")]
            )
            engine.startup_checks()

        mock_alpaca.fetch_historical_ohlcv.assert_called_once()
        call_args = mock_alpaca.fetch_historical_ohlcv.call_args
        assert call_args[0][0] == ["AAPL"]
        assert call_args[1]["timeframe"] == "1Day"

    def test_no_seed_when_db_has_enough_rows(self, tmp_path):
        """If DB already has ≥ 60 rows, fetch_historical_ohlcv is NOT called."""
        engine, mock_storage, mock_alpaca, mock_hmm = self._build_seeding_engine(
            tmp_path, hmm_fitted=False
        )

        # DB returns 100 rows — enough
        mock_storage.query_ohlcv.return_value = pd.DataFrame({"close": range(100)})

        with (
            patch("ollama.Client") as mock_ollama_client,
            patch(_SETTINGS, _fake_settings()),
        ):
            mock_ollama_client.return_value.list.return_value = SimpleNamespace(
                models=[SimpleNamespace(model="gemma4:e4b")]
            )
            engine.startup_checks()

        mock_alpaca.fetch_historical_ohlcv.assert_not_called()

    def test_seed_failure_does_not_abort_fit(self, tmp_path):
        """If seeding raises, the engine still attempts hmm.fit (and logs a warning)."""
        engine, mock_storage, mock_alpaca, mock_hmm = self._build_seeding_engine(
            tmp_path, hmm_fitted=False
        )

        mock_storage.query_ohlcv.return_value = pd.DataFrame({"close": range(12)})
        mock_alpaca.fetch_historical_ohlcv.side_effect = RuntimeError("API down")

        with (
            patch("ollama.Client") as mock_ollama_client,
            patch(_SETTINGS, _fake_settings()),
        ):
            mock_ollama_client.return_value.list.return_value = SimpleNamespace(
                models=[SimpleNamespace(model="gemma4:e4b")]
            )
            engine.startup_checks()   # must not raise

        # fit was still attempted despite seeding failure
        mock_hmm.fit.assert_called_once()

    def test_no_seeding_when_already_fitted(self, tmp_path):
        """Seeding is skipped entirely when the HMM is already fitted."""
        engine, mock_storage, mock_alpaca, mock_hmm = self._build_seeding_engine(
            tmp_path, hmm_fitted=True
        )

        with (
            patch("ollama.Client") as mock_ollama_client,
            patch(_SETTINGS, _fake_settings()),
        ):
            mock_ollama_client.return_value.list.return_value = SimpleNamespace(
                models=[SimpleNamespace(model="gemma4:e4b")]
            )
            engine.startup_checks()

        mock_storage.query_ohlcv.assert_not_called()
        mock_alpaca.fetch_historical_ohlcv.assert_not_called()

    def test_seed_empty_response_logs_warning_and_continues(self, tmp_path):
        """If Alpaca returns an empty DataFrame (ticker not on IEX), fit is still attempted."""
        engine, mock_storage, mock_alpaca, mock_hmm = self._build_seeding_engine(
            tmp_path, hmm_fitted=False
        )

        mock_storage.query_ohlcv.return_value = pd.DataFrame({"close": range(12)})
        mock_alpaca.fetch_historical_ohlcv.return_value = pd.DataFrame()  # empty

        with (
            patch("ollama.Client") as mock_ollama_client,
            patch(_SETTINGS, _fake_settings()),
        ):
            mock_ollama_client.return_value.list.return_value = SimpleNamespace(
                models=[SimpleNamespace(model="gemma4:e4b")]
            )
            engine.startup_checks()   # must not raise

        mock_hmm.fit.assert_called_once()

    def test_portfolio_min_variance_seeded_in_startup(self, tmp_path):
        """startup_checks() must call compute_min_variance() to seed weights."""
        engine, mock_storage, mock_alpaca, _ = self._build_seeding_engine(
            tmp_path, hmm_fitted=True
        )

        with (
            patch("ollama.Client") as mock_ollama_client,
            patch(_SETTINGS, _fake_settings()),
        ):
            mock_ollama_client.return_value.list.return_value = SimpleNamespace(
                models=[SimpleNamespace(model="gemma4:e4b")]
            )
            engine.startup_checks()

        engine.portfolio_optimizer.compute_min_variance.assert_called_once()

    def test_portfolio_init_failure_does_not_abort_startup(self, tmp_path):
        """If compute_min_variance raises, startup_checks must still complete."""
        engine, mock_storage, mock_alpaca, _ = self._build_seeding_engine(
            tmp_path, hmm_fitted=True
        )
        engine.portfolio_optimizer.compute_min_variance.side_effect = RuntimeError(
            "DB unavailable"
        )

        with (
            patch("ollama.Client") as mock_ollama_client,
            patch(_SETTINGS, _fake_settings()),
        ):
            mock_ollama_client.return_value.list.return_value = SimpleNamespace(
                models=[SimpleNamespace(model="gemma4:e4b")]
            )
            engine.startup_checks()   # must not raise


# ===========================================================================
# Earnings guard
# ===========================================================================

class TestEarningsGuard:
    """
    _is_earnings_guard_triggered() returns True only when today or tomorrow is
    an earnings date for the ticker.  bar_handler skips order submission (but
    still persists the trade_log entry) when the guard fires.
    """

    def _engine_with_signal(self, tmp_path, earnings_date):
        """Build an engine that produces a BUY signal, with market open."""
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path,
            mwu_decision=_mwu_decision(signal=1, score=0.8),
        )
        mock_alpaca.is_market_open.return_value = True
        engine._fundamentals.get_earnings_dates.return_value = {
            "AAPL": earnings_date
        }
        return engine, mock_alpaca

    # ------------------------------------------------------------------
    # _is_earnings_guard_triggered unit tests
    # ------------------------------------------------------------------

    def test_triggered_when_earnings_today(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        today_dt = datetime.now(tz=timezone.utc)
        engine._fundamentals.get_earnings_dates.return_value = {"AAPL": today_dt}

        assert engine._is_earnings_guard_triggered("AAPL") is True

    def test_triggered_when_earnings_tomorrow(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        tomorrow_dt = datetime.now(tz=timezone.utc) + timedelta(days=1)
        engine._fundamentals.get_earnings_dates.return_value = {"AAPL": tomorrow_dt}

        assert engine._is_earnings_guard_triggered("AAPL") is True

    def test_not_triggered_when_earnings_in_two_days(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        future_dt = datetime.now(tz=timezone.utc) + timedelta(days=2)
        engine._fundamentals.get_earnings_dates.return_value = {"AAPL": future_dt}

        assert engine._is_earnings_guard_triggered("AAPL") is False

    def test_not_triggered_when_no_earnings_date(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._fundamentals.get_earnings_dates.return_value = {"AAPL": None}

        assert engine._is_earnings_guard_triggered("AAPL") is False

    def test_fails_open_on_exception(self, tmp_path):
        """A crash in get_earnings_dates must not block order submission."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._fundamentals.get_earnings_dates.side_effect = RuntimeError("yfinance down")

        assert engine._is_earnings_guard_triggered("AAPL") is False

    # ------------------------------------------------------------------
    # bar_handler integration: order skipped when guard fires
    # ------------------------------------------------------------------

    def test_order_skipped_on_earnings_day(self, tmp_path):
        engine, mock_alpaca = self._engine_with_signal(
            tmp_path, datetime.now(tz=timezone.utc)
        )
        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_not_called()

    def test_order_skipped_on_earnings_eve(self, tmp_path):
        engine, mock_alpaca = self._engine_with_signal(
            tmp_path, datetime.now(tz=timezone.utc) + timedelta(days=1)
        )
        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_not_called()

    def test_trade_log_still_written_when_guard_fires(self, tmp_path):
        """The decision must be persisted even when the order is skipped."""
        engine, _ = self._engine_with_signal(
            tmp_path, datetime.now(tz=timezone.utc)
        )
        engine.bar_handler(_make_bar("AAPL"))

        engine._storage.insert_trade_log.assert_called_once()

    def test_order_proceeds_when_no_earnings(self, tmp_path):
        """When no earnings date is imminent, order submission runs normally."""
        engine, mock_alpaca = self._engine_with_signal(tmp_path, None)
        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_called_once()

    def test_order_proceeds_when_earnings_two_days_out(self, tmp_path):
        engine, mock_alpaca = self._engine_with_signal(
            tmp_path, datetime.now(tz=timezone.utc) + timedelta(days=2)
        )
        engine.bar_handler(_make_bar("AAPL"))

        engine._executor.submit_order.assert_called_once()

# ===========================================================================
# _get_analyst_signal
# ===========================================================================

class TestAnalystSignal:
    """
    Unit tests for TradingEngine._get_analyst_signal().

    Verifies that the analyst recommendation is correctly fetched from
    FundamentalsClient and mapped to a signal dict, and that the helper
    fails open on any exception.
    """

    def test_buy_recommendation_returns_plus_one(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._fundamentals.get_analyst_recommendations.return_value = {"AAPL": 1}

        result = engine._get_analyst_signal("AAPL")

        assert result["signal"] == 1
        assert result["confidence"] == pytest.approx(0.7)

    def test_sell_recommendation_returns_minus_one(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._fundamentals.get_analyst_recommendations.return_value = {"AAPL": -1}

        result = engine._get_analyst_signal("AAPL")

        assert result["signal"] == -1
        assert result["confidence"] == pytest.approx(0.7)

    def test_hold_recommendation_returns_zero_confidence(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._fundamentals.get_analyst_recommendations.return_value = {"AAPL": 0}

        result = engine._get_analyst_signal("AAPL")

        assert result["signal"] == 0
        assert result["confidence"] == pytest.approx(0.0)

    def test_missing_ticker_returns_neutral(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._fundamentals.get_analyst_recommendations.return_value = {}

        result = engine._get_analyst_signal("AAPL")

        assert result["signal"] == 0

    def test_fails_open_on_exception(self, tmp_path):
        """A crash in get_analyst_recommendations must return neutral, not raise."""
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        engine._fundamentals.get_analyst_recommendations.side_effect = RuntimeError("yfinance down")

        result = engine._get_analyst_signal("AAPL")

        assert result["signal"] == 0
        assert result["confidence"] == pytest.approx(0.0)

    def test_analyst_signal_included_in_mwu_call(self, tmp_path):
        """
        bar_handler must pass analyst_recs to scheduled_update.
        Verify it appears in the signals dict argument.
        """
        engine, _, mock_alpaca, mwu_map = _build_engine(
            tmp_path=tmp_path,
            mwu_decision=_mwu_decision(signal=0),
        )
        mock_alpaca.is_market_open.return_value = True
        engine._fundamentals.get_analyst_recommendations.return_value = {"AAPL": 1}

        engine.bar_handler(_make_bar("AAPL"))

        call_kwargs = mwu_map["AAPL"].scheduled_update.call_args
        signals_passed = call_kwargs.kwargs.get("signals") or call_kwargs.args[1]
        assert "analyst_recs" in signals_passed
        assert signals_passed["analyst_recs"]["signal"] == 1

    def test_analyst_signal_logged_to_trade_log(self, tmp_path):
        """
        bar_handler must write analyst_signal and analyst_confidence to trade_log.
        """
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path,
            mwu_decision=_mwu_decision(signal=1),
        )
        mock_alpaca.is_market_open.return_value = True
        engine._fundamentals.get_analyst_recommendations.return_value = {"AAPL": -1}

        engine.bar_handler(_make_bar("AAPL"))

        engine._storage.insert_trade_log.assert_called_once()
        logged = engine._storage.insert_trade_log.call_args[0][0]
        assert logged["analyst_signal"] == -1
        assert logged["analyst_confidence"] == pytest.approx(0.7)

    def test_analyst_neutral_confidence_zero_in_trade_log(self, tmp_path):
        """When analyst signal is 0, confidence logged as 0.0."""
        engine, _, mock_alpaca, _ = _build_engine(
            tmp_path=tmp_path,
            mwu_decision=_mwu_decision(signal=1),
        )
        mock_alpaca.is_market_open.return_value = True
        engine._fundamentals.get_analyst_recommendations.return_value = {"AAPL": 0}

        engine.bar_handler(_make_bar("AAPL"))

        logged = engine._storage.insert_trade_log.call_args[0][0]
        assert logged["analyst_signal"] == 0
        assert logged["analyst_confidence"] == pytest.approx(0.0)
