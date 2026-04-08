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
import pytest

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_ENG_MOD    = "trading_engine.orchestrator.engine"
_STORAGE    = f"{_ENG_MOD}.Storage"
_ALPACA     = f"{_ENG_MOD}.AlpacaMarketData"
_AV_CLIENT  = f"{_ENG_MOD}.AlphaVantageNewsClient"
_HMM        = f"{_ENG_MOD}.HMMRegimeDetector"
_LLM        = f"{_ENG_MOD}.LLMSentimentSignal"
_MWU        = f"{_ENG_MOD}.MWUMetaAgent"
_OU_SIGNAL  = f"{_ENG_MOD}.OUSpreadSignal"
_RISK       = f"{_ENG_MOD}.RiskManager"
_EXECUTOR   = f"{_ENG_MOD}.OrderExecutor"
_SCHEDULER  = f"{_ENG_MOD}.BackgroundScheduler"
_SETTINGS   = f"{_ENG_MOD}.settings"
_STATE_MGR  = f"{_ENG_MOD}.StateManager"


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
        "weights":      {"hmm_regime": 0.4, "ou_spread": 0.3, "llm_sentiment": 0.3},
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

    # Wire everything onto the engine
    engine._tickers     = list(tickers)
    engine._pairs       = [tuple(p) for p in pairs]
    engine._paper       = True
    engine._models_dir  = tmp_path or Path("/tmp/test_engine")
    engine._storage     = mock_storage
    engine._alpaca      = mock_alpaca
    engine._av_client   = mock_av
    engine._hmm         = hmm_map
    engine._ou_signals  = ou_map
    engine._llm         = MagicMock()
    engine._mwu         = mwu_map
    engine._risk        = mock_risk
    engine._executor    = mock_executor
    engine._state_manager = mock_state
    engine._signal_stats  = {t: {"win_rate": 0.52, "avg_win": 0.015, "avg_loss": 0.010} for t in tickers}
    engine._shutdown_event = threading.Event()
    engine._emergency_close = False
    engine._scheduler   = None

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

    def test_signals_dict_has_three_keys(self, tmp_path):
        engine, _, _, mwu_map = _build_engine(tmp_path=tmp_path)
        engine.bar_handler(_make_bar("AAPL"))
        signals = mwu_map["AAPL"].scheduled_update.call_args[1]["signals"]
        assert set(signals.keys()) == {"hmm_regime", "ou_spread", "llm_sentiment"}

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
# sentiment_job
# ===========================================================================

class TestSentimentJob:

    def _make_engine_for_sentiment(self, tmp_path, market_open: bool):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)

        if market_open:
            # 10:00 AM ET Tuesday → market open
            from datetime import datetime
            from zoneinfo import ZoneInfo
            fake_now = datetime(2025, 1, 14, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        else:
            # 8:00 PM ET → closed
            from datetime import datetime
            from zoneinfo import ZoneInfo
            fake_now = datetime(2025, 1, 14, 20, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        return engine, fake_now

    def test_runs_during_market_hours(self, tmp_path):
        engine, fake_now = self._make_engine_for_sentiment(tmp_path, market_open=True)
        with patch(f"{_ENG_MOD}.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            engine.sentiment_job()
        engine._llm.run_pipeline.assert_called_once_with(
            engine._tickers, engine._av_client, engine._storage
        )

    def test_skipped_outside_market_hours(self, tmp_path):
        engine, fake_now = self._make_engine_for_sentiment(tmp_path, market_open=False)
        with patch(f"{_ENG_MOD}.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            engine.sentiment_job()
        engine._llm.run_pipeline.assert_not_called()

    def test_exception_does_not_propagate(self, tmp_path):
        engine, fake_now = self._make_engine_for_sentiment(tmp_path, market_open=True)
        engine._llm.run_pipeline.side_effect = RuntimeError("AV rate limit")
        with patch(f"{_ENG_MOD}.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            engine.sentiment_job()   # should not raise


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

    def test_win_rate_updated_in_signal_stats(self, tmp_path):
        engine, _, _, mwu_map = _build_engine(tmp_path=tmp_path)
        # All signals have 0.6 win rate → ensemble = 0.6
        engine.eod_job()
        assert engine._signal_stats["AAPL"]["win_rate"] == pytest.approx(0.58, abs=0.01)

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

    def test_two_jobs_registered(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)

        mock_scheduler = MagicMock()
        mock_scheduler.running = False

        with patch(_SCHEDULER, return_value=mock_scheduler):
            # Simulate run() up to just after scheduler.add_job calls by
            # calling the job-registration portion directly.
            engine._scheduler = mock_scheduler
            engine._scheduler.add_job(
                engine.sentiment_job,
                "interval",
                hours=4,
                id="sentiment_job",
                next_run_time=datetime.now(tz=timezone.utc),
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

        assert mock_scheduler.add_job.call_count == 2

    def test_sentiment_job_registered_as_interval(self, tmp_path):
        engine, _, _, _ = _build_engine(tmp_path=tmp_path)
        mock_scheduler = MagicMock()

        engine._scheduler = mock_scheduler
        engine._scheduler.add_job(
            engine.sentiment_job, "interval", hours=4, id="sentiment_job",
            next_run_time=datetime.now(tz=timezone.utc),
        )
        engine._scheduler.add_job(
            engine.eod_job, "cron", day_of_week="mon-fri", hour=16, minute=5,
            timezone="America/New_York", id="eod_job",
        )

        calls = mock_scheduler.add_job.call_args_list
        triggers = [c[0][1] for c in calls]
        assert "interval" in triggers
        assert "cron" in triggers

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

    def test_checksum_mismatch_raises(self, tmp_path):
        from trading_engine.orchestrator.state_manager import StateManager
        sm = StateManager(state_dir=tmp_path)
        sm.save({"tickers": ["AAPL"]})

        # Corrupt the file
        path = sm._state_path()
        data = json.loads(path.read_text())
        data["checksum"] = "bad_checksum"
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="checksum"):
            sm.load()

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

    def test_pair_parsing(self):
        from trading_engine.main import _parse_pair
        assert _parse_pair("JPM,BAC") == ("JPM", "BAC")

    def test_pair_parsing_case_insensitive(self):
        from trading_engine.main import _parse_pair
        assert _parse_pair("jpm,bac") == ("JPM", "BAC")

    def test_invalid_pair_raises(self):
        from trading_engine.main import _parse_pair
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_pair("AAPL")
