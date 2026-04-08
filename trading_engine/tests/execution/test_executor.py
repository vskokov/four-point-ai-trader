"""
Unit tests for execution/executor.py.

All Alpaca SDK calls are mocked — no live API access.
Run with:
    .venv/bin/pytest tests/execution/test_executor.py -v
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_MOD = "trading_engine.execution.executor"
_TRADING_CLIENT = f"{_MOD}.TradingClient"
_SETTINGS       = f"{_MOD}.settings"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        ALPACA_API_KEY="fake-key",
        ALPACA_SECRET_KEY="fake-secret",
    )


def _account(equity: float = 100_000.0) -> dict:
    return {
        "equity":          equity,
        "cash":            equity * 0.8,
        "buying_power":    equity * 1.6,
        "portfolio_value": equity,
    }


def _signal_stats(win_rate: float = 0.6, avg_win: float = 0.10, avg_loss: float = 0.05) -> dict:
    return {"win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss}


def _fake_position(symbol: str, qty: float, market_value: float) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        qty=str(qty),
        market_value=str(market_value),
        unrealized_pl=str(market_value * 0.02),
        unrealized_plpc=str(0.02),
    )


def _fake_order(order_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=uuid.UUID(order_id or "12345678-1234-5678-1234-567812345678"))


# ---------------------------------------------------------------------------
# Import helpers — deferred so patch targets are stable
# ---------------------------------------------------------------------------

def _make_risk_manager(**kwargs):
    from trading_engine.execution.executor import RiskManager
    return RiskManager(**kwargs)


def _make_executor(mock_trading, mock_alpaca):
    """Build an OrderExecutor with the TradingClient already patched."""
    from trading_engine.execution.executor import OrderExecutor, RiskManager
    risk = RiskManager()
    executor = OrderExecutor.__new__(OrderExecutor)
    executor._alpaca = mock_alpaca
    executor._risk = risk
    executor._trading = mock_trading
    executor.portfolio_optimizer = None   # no optimizer by default in unit tests
    return executor


# ===========================================================================
# RiskManager — kelly_size
# ===========================================================================

class TestKellySize:

    def test_basic_calculation(self):
        rm = _make_risk_manager(kelly_fraction=0.25, max_position_pct=0.10)
        # b = 0.10/0.05 = 2, f = (0.6*2 - 0.4)/2 = (1.2-0.4)/2 = 0.4
        # kelly* = 0.25 * 0.4 = 0.10
        result = rm.kelly_size(win_rate=0.6, avg_win=0.10, avg_loss=0.05)
        assert abs(result - 0.10) < 1e-9

    def test_clamped_to_max_position_pct(self):
        # Very high win rate → raw Kelly >> max_position_pct
        rm = _make_risk_manager(kelly_fraction=1.0, max_position_pct=0.10)
        result = rm.kelly_size(win_rate=0.99, avg_win=1.0, avg_loss=0.01)
        assert result == pytest.approx(0.10)

    def test_negative_kelly_returns_zero(self):
        # 20% win rate with 1:1 payoff → negative Kelly
        rm = _make_risk_manager()
        result = rm.kelly_size(win_rate=0.20, avg_win=0.05, avg_loss=0.05)
        assert result == 0.0

    def test_zero_avg_loss_returns_zero(self):
        rm = _make_risk_manager()
        assert rm.kelly_size(win_rate=0.6, avg_win=0.1, avg_loss=0.0) == 0.0

    def test_zero_avg_win_returns_zero(self):
        rm = _make_risk_manager()
        assert rm.kelly_size(win_rate=0.6, avg_win=0.0, avg_loss=0.05) == 0.0

    def test_zero_win_rate_returns_zero(self):
        rm = _make_risk_manager()
        assert rm.kelly_size(win_rate=0.0, avg_win=0.1, avg_loss=0.05) == 0.0

    def test_fractional_kelly_scales_down(self):
        rm_full  = _make_risk_manager(kelly_fraction=1.0,  max_position_pct=1.0)
        rm_half  = _make_risk_manager(kelly_fraction=0.5,  max_position_pct=1.0)
        rm_quarter = _make_risk_manager(kelly_fraction=0.25, max_position_pct=1.0)
        full = rm_full.kelly_size(0.6, 0.10, 0.05)
        assert rm_half.kelly_size(0.6, 0.10, 0.05) == pytest.approx(full * 0.5)
        assert rm_quarter.kelly_size(0.6, 0.10, 0.05) == pytest.approx(full * 0.25)

    def test_result_never_exceeds_max_position_pct(self):
        for max_pct in [0.05, 0.10, 0.20]:
            rm = _make_risk_manager(kelly_fraction=1.0, max_position_pct=max_pct)
            result = rm.kelly_size(0.9, 1.0, 0.01)
            assert result <= max_pct


# ===========================================================================
# RiskManager — circuit_breaker
# ===========================================================================

class TestCircuitBreaker:

    def test_no_breach_returns_false(self):
        rm = _make_risk_manager(max_drawdown_pct=0.15, max_daily_loss_pct=0.05)
        # Equity at full value — no drawdown or daily loss
        assert rm.circuit_breaker(_account(100_000)) is False

    def test_drawdown_breach_triggers(self):
        rm = _make_risk_manager(max_drawdown_pct=0.15)
        rm.circuit_breaker(_account(100_000))   # establish peak
        # Drop 20 % → exceeds 15 % limit
        assert rm.circuit_breaker(_account(80_000)) is True

    def test_daily_loss_breach_triggers(self):
        rm = _make_risk_manager(max_drawdown_pct=0.50, max_daily_loss_pct=0.05)
        rm.circuit_breaker(_account(100_000))   # sets daily_start_equity
        # Drop 6 % within same UTC day → exceeds 5 % daily limit
        assert rm.circuit_breaker(_account(94_000)) is True

    def test_small_drawdown_within_limits_ok(self):
        rm = _make_risk_manager(max_drawdown_pct=0.15, max_daily_loss_pct=0.05)
        rm.circuit_breaker(_account(100_000))
        # 3 % drop — within both limits
        assert rm.circuit_breaker(_account(97_000)) is False

    def test_critical_logged_on_drawdown(self, capsys):
        rm = _make_risk_manager(max_drawdown_pct=0.10)
        rm.circuit_breaker(_account(100_000))
        rm.circuit_breaker(_account(85_000))
        out = capsys.readouterr().out
        assert "drawdown" in out.lower()

    def test_critical_logged_on_daily_loss(self, capsys):
        rm = _make_risk_manager(max_drawdown_pct=0.50, max_daily_loss_pct=0.05)
        rm.circuit_breaker(_account(100_000))
        rm.circuit_breaker(_account(90_000))
        out = capsys.readouterr().out
        assert "daily" in out.lower()

    def test_peak_equity_updates_on_new_high(self):
        rm = _make_risk_manager(max_drawdown_pct=0.15)
        rm.circuit_breaker(_account(100_000))
        rm.circuit_breaker(_account(110_000))   # new peak
        assert rm._peak_equity == 110_000

    def test_daily_start_resets_on_new_day(self, monkeypatch):
        rm = _make_risk_manager(max_drawdown_pct=0.50, max_daily_loss_pct=0.05)

        day1 = date(2025, 1, 10)
        day2 = date(2025, 1, 11)

        # Simulate day 1 — set a low daily start
        monkeypatch.setattr(
            "trading_engine.execution.executor.datetime",
            _make_fixed_datetime(day1),
        )
        rm.circuit_breaker(_account(100_000))
        assert rm._daily_start_equity == 100_000

        # Simulate day 2 — daily start resets
        monkeypatch.setattr(
            "trading_engine.execution.executor.datetime",
            _make_fixed_datetime(day2),
        )
        rm.circuit_breaker(_account(95_000))
        assert rm._daily_start_equity == 95_000


def _make_fixed_datetime(fixed_date: date):
    """Return a datetime subclass whose now() always reports *fixed_date* at noon UTC."""
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(fixed_date.year, fixed_date.month, fixed_date.day, 12, 0, 0, tzinfo=timezone.utc)
    return _FixedDatetime


# ===========================================================================
# RiskManager — check_trade
# ===========================================================================

class TestCheckTrade:

    def test_approved_no_existing_position(self):
        rm = _make_risk_manager(max_position_pct=0.10)
        result = rm.check_trade("AAPL", 1, _account(100_000), {})
        assert result["approved"] is True
        assert result["max_size"] == pytest.approx(10_000.0)

    def test_approved_partial_existing_position(self):
        rm = _make_risk_manager(max_position_pct=0.10)
        positions = {"AAPL": {"market_value": 5_000.0, "qty": 50.0}}
        result = rm.check_trade("AAPL", 1, _account(100_000), positions)
        assert result["approved"] is True
        assert result["max_size"] == pytest.approx(5_000.0)

    def test_rejected_position_limit_exactly_at_cap(self):
        rm = _make_risk_manager(max_position_pct=0.10)
        positions = {"AAPL": {"market_value": 10_000.0, "qty": 100.0}}
        result = rm.check_trade("AAPL", 1, _account(100_000), positions)
        assert result["approved"] is False
        assert result["reason"] == "position_limit_exceeded"
        assert result["max_size"] == 0.0

    def test_rejected_position_limit_over_cap(self):
        rm = _make_risk_manager(max_position_pct=0.10)
        positions = {"AAPL": {"market_value": 12_000.0, "qty": 120.0}}
        result = rm.check_trade("AAPL", 1, _account(100_000), positions)
        assert result["approved"] is False
        assert result["reason"] == "position_limit_exceeded"

    def test_sell_bypasses_position_limit(self):
        # Even with a huge position, selling should not be blocked by position limit
        rm = _make_risk_manager(max_position_pct=0.10)
        positions = {"AAPL": {"market_value": 50_000.0, "qty": 500.0}}
        result = rm.check_trade("AAPL", -1, _account(100_000), positions)
        assert result["approved"] is True

    def test_rejected_circuit_breaker_overrides_all(self):
        rm = _make_risk_manager(max_drawdown_pct=0.15)
        rm.circuit_breaker(_account(100_000))
        # Large drawdown — circuit breaker fires regardless of position
        result = rm.check_trade("AAPL", 1, _account(80_000), {})
        assert result["approved"] is False
        assert result["reason"] == "circuit_breaker"

    def test_sell_rejected_by_circuit_breaker(self):
        rm = _make_risk_manager(max_drawdown_pct=0.15)
        rm.circuit_breaker(_account(100_000))
        result = rm.check_trade("AAPL", -1, _account(80_000), {})
        assert result["approved"] is False
        assert result["reason"] == "circuit_breaker"

    def test_other_ticker_position_does_not_affect_limit(self):
        rm = _make_risk_manager(max_position_pct=0.10)
        positions = {"MSFT": {"market_value": 50_000.0, "qty": 500.0}}
        result = rm.check_trade("AAPL", 1, _account(100_000), positions)
        assert result["approved"] is True
        assert result["max_size"] == pytest.approx(10_000.0)


# ===========================================================================
# OrderExecutor — submit_order
# ===========================================================================

class TestSubmitOrder:

    def _make_mock_alpaca(self, mid_price: float = 150.0) -> MagicMock:
        mock = MagicMock()
        mock.get_latest_quote.return_value = {"mid": mid_price, "bid": mid_price - 0.1, "ask": mid_price + 0.1}
        return mock

    def test_no_op_when_signal_zero(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        executor = _make_executor(mock_trading, self._make_mock_alpaca())

        result = executor.submit_order("AAPL", 0, 0.8, _account(), _signal_stats())
        assert result == {"status": "no_op"}
        mock_trading.submit_order.assert_not_called()

    def test_rejected_when_circuit_breaker_fires(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        executor = _make_executor(mock_trading, self._make_mock_alpaca())

        # Establish peak, then drop 20 %
        executor._risk.circuit_breaker(_account(100_000))
        result = executor.submit_order("AAPL", 1, 1.0, _account(80_000), _signal_stats())

        assert result["status"] == "rejected"
        assert result["reason"] == "circuit_breaker"
        mock_trading.submit_order.assert_not_called()

    def test_rejected_when_position_limit_exceeded(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = [
            _fake_position("AAPL", 100.0, 10_000.0)
        ]
        executor = _make_executor(mock_trading, self._make_mock_alpaca())

        result = executor.submit_order("AAPL", 1, 1.0, _account(100_000), _signal_stats())
        assert result["status"] == "rejected"
        assert result["reason"] == "position_limit_exceeded"

    def test_too_small_when_zero_shares(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        # mid price is $1000 per share but Kelly sizing produces tiny USD amount
        executor = _make_executor(mock_trading, self._make_mock_alpaca(mid_price=1_000_000.0))

        # equity=1000, Kelly * confidence = tiny fraction → floor = 0 shares
        result = executor.submit_order("AAPL", 1, 0.01, _account(1_000.0), _signal_stats(win_rate=0.51, avg_win=0.01, avg_loss=0.05))
        assert result["status"] == "too_small"
        mock_trading.submit_order.assert_not_called()

    def test_buy_order_submitted_correctly(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        executor = _make_executor(mock_trading, mock_alpaca)

        # equity=100_000, kelly_f=0.10, confidence=1.0 → size_usd=10_000 → 100 shares
        result = executor.submit_order("AAPL", 1, 1.0, _account(100_000), _signal_stats())

        assert result["status"] == "submitted"
        assert result["ticker"] == "AAPL"
        assert result["side"] == "buy"
        assert result["qty"] == 100
        assert result["estimated_price"] == 100.0
        assert "order_id" in result
        assert "timestamp" in result

        call_args = mock_trading.submit_order.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.qty == 100

    def test_sell_order_submitted_correctly(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = [
            _fake_position("AAPL", 200.0, 20_000.0)
        ]
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        executor = _make_executor(mock_trading, mock_alpaca)

        # sell signal — cap at min(100 computed, 200 held) = 100
        result = executor.submit_order("AAPL", -1, 1.0, _account(100_000), _signal_stats())

        assert result["status"] == "submitted"
        assert result["side"] == "sell"
        assert result["qty"] == 100

    def test_sell_capped_at_current_position(self):
        mock_trading = MagicMock()
        # Only 30 shares held, but Kelly would request more
        mock_trading.get_all_positions.return_value = [
            _fake_position("AAPL", 30.0, 3_000.0)
        ]
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        executor = _make_executor(mock_trading, mock_alpaca)

        result = executor.submit_order("AAPL", -1, 1.0, _account(100_000), _signal_stats())
        assert result["status"] == "submitted"
        assert result["qty"] == 30   # capped at held shares

    def test_sell_returns_no_position_when_not_held(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []   # no positions
        mock_alpaca = self._make_mock_alpaca()
        executor = _make_executor(mock_trading, mock_alpaca)

        result = executor.submit_order("AAPL", -1, 1.0, _account(100_000), _signal_stats())
        assert result["status"] == "no_position"
        mock_trading.submit_order.assert_not_called()

    def test_confidence_scales_share_count(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        executor = _make_executor(mock_trading, mock_alpaca)

        # confidence=0.5 should halve the share count vs confidence=1.0
        r_full = executor.submit_order("AAPL", 1, 1.0, _account(100_000), _signal_stats())
        mock_trading.get_all_positions.return_value = []
        r_half = executor.submit_order("AAPL", 1, 0.5, _account(100_000), _signal_stats())

        assert r_full["qty"] == r_half["qty"] * 2


# ===========================================================================
# OrderExecutor — get_positions
# ===========================================================================

class TestGetPositions:

    def test_returns_dataframe_with_correct_columns(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = [
            _fake_position("AAPL", 100.0, 15_000.0),
            _fake_position("MSFT", 50.0,  10_000.0),
        ]
        executor = _make_executor(mock_trading, MagicMock())
        df = executor.get_positions()

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"ticker", "qty", "market_value", "unrealized_pnl", "unrealized_pnl_pct"}
        assert len(df) == 2
        assert set(df["ticker"].tolist()) == {"AAPL", "MSFT"}

    def test_returns_empty_dataframe_when_no_positions(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        executor = _make_executor(mock_trading, MagicMock())
        df = executor.get_positions()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "ticker" in df.columns

    def test_market_value_is_float(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = [
            _fake_position("AAPL", 10.0, 1_500.0),
        ]
        executor = _make_executor(mock_trading, MagicMock())
        df = executor.get_positions()

        assert df["market_value"].dtype == float


# ===========================================================================
# OrderExecutor — close_all_positions
# ===========================================================================

class TestCloseAllPositions:

    def test_calls_alpaca_close_all(self):
        mock_trading = MagicMock()
        executor = _make_executor(mock_trading, MagicMock())

        executor.close_all_positions()
        mock_trading.close_all_positions.assert_called_once_with(cancel_orders=True)

    def test_critical_logged_before_close(self, capsys):
        mock_trading = MagicMock()
        executor = _make_executor(mock_trading, MagicMock())

        executor.close_all_positions()
        out = capsys.readouterr().out
        assert "close_all" in out.lower()

    def test_exception_propagates(self):
        mock_trading = MagicMock()
        mock_trading.close_all_positions.side_effect = RuntimeError("Alpaca down")
        executor = _make_executor(mock_trading, MagicMock())

        with pytest.raises(RuntimeError, match="Alpaca down"):
            executor.close_all_positions()


# ===========================================================================
# OrderExecutor — __init__ wires TradingClient
# ===========================================================================

class TestOrderExecutorInit:

    def test_trading_client_created_with_paper_flag(self):
        fake_s = _fake_settings()
        with (
            patch(_TRADING_CLIENT) as MockTC,
            patch(_SETTINGS, fake_s),
        ):
            from trading_engine.execution.executor import OrderExecutor, RiskManager
            mock_alpaca = MagicMock()
            risk = RiskManager()
            OrderExecutor(mock_alpaca, risk, paper=True)

            MockTC.assert_called_once_with(
                api_key="fake-key",
                secret_key="fake-secret",
                paper=True,
            )

    def test_trading_client_created_paper_false(self):
        fake_s = _fake_settings()
        with (
            patch(_TRADING_CLIENT) as MockTC,
            patch(_SETTINGS, fake_s),
        ):
            from trading_engine.execution.executor import OrderExecutor, RiskManager
            mock_alpaca = MagicMock()
            risk = RiskManager()
            OrderExecutor(mock_alpaca, risk, paper=False)

            MockTC.assert_called_once_with(
                api_key="fake-key",
                secret_key="fake-secret",
                paper=False,
            )
