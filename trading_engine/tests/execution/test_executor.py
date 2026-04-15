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

        # cash=80_000 (equity*0.8), kelly_f≈0.10, confidence=1.0 → size_usd≈8_000
        result = executor.submit_order("AAPL", 1, 1.0, _account(100_000), _signal_stats())

        assert result["status"] == "submitted"
        assert result["ticker"] == "AAPL"
        assert result["side"] == "buy"
        # Buy uses notional (fractional shares) — check dollar amount not share count
        assert result["notional"] > 0
        assert result["notional"] <= 80_000.0   # capped at cash
        assert result["estimated_price"] == 100.0
        assert "order_id" in result
        assert "timestamp" in result

        call_args = mock_trading.submit_order.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.notional == result["notional"]

    def test_sell_order_submitted_correctly(self):
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = [
            _fake_position("AAPL", 200.0, 20_000.0)
        ]
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        executor = _make_executor(mock_trading, mock_alpaca)

        # sell signal — sized off equity=100K (sells are not cash-constrained)
        # kelly_f≈0.10, confidence=1.0 → ~100 shares computed; cap at min(100, 200) = 100
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

        # confidence=0.5 should roughly halve the notional vs confidence=1.0
        r_full = executor.submit_order("AAPL", 1, 1.0, _account(100_000), _signal_stats())
        mock_trading.get_all_positions.return_value = []
        r_half = executor.submit_order("AAPL", 1, 0.5, _account(100_000), _signal_stats())

        # notional at half confidence should be ~half of full confidence
        assert abs(r_full["notional"] - r_half["notional"] * 2) <= 1.0

    def test_returns_skipped_when_market_closed(self):
        """submit_order must short-circuit with status=skipped when market is closed."""
        mock_trading = MagicMock()
        mock_alpaca = self._make_mock_alpaca()
        mock_alpaca.is_market_open.return_value = False
        executor = _make_executor(mock_trading, mock_alpaca)

        result = executor.submit_order("AAPL", 1, 0.8, _account(), _signal_stats())

        assert result == {"status": "skipped", "reason": "market_closed"}
        mock_trading.submit_order.assert_not_called()

    def test_proceeds_when_market_open(self):
        """submit_order must proceed normally when is_market_open() is True."""
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        mock_alpaca.is_market_open.return_value = True
        executor = _make_executor(mock_trading, mock_alpaca)

        result = executor.submit_order("AAPL", 1, 1.0, _account(100_000), _signal_stats())

        assert result["status"] == "submitted"
        mock_trading.submit_order.assert_called_once()


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
# OrderExecutor — get_todays_filled_buy_symbols
# ===========================================================================

def _fake_filled_order(symbol: str, side_value: str = "buy", filled_qty: float = 10.0):
    from alpaca.trading.enums import OrderSide
    return SimpleNamespace(
        symbol=symbol,
        side=OrderSide.BUY if side_value == "buy" else OrderSide.SELL,
        filled_qty=str(filled_qty),
    )


class TestGetTodaysFilledBuySymbols:

    def test_returns_buy_symbols_filled_today(self):
        mock_trading = MagicMock()
        mock_trading.get_orders.return_value = [
            _fake_filled_order("AAPL", "buy",  10.0),
            _fake_filled_order("MSFT", "buy",  5.0),
            _fake_filled_order("JPM",  "sell", 3.0),  # sell — must be excluded
        ]
        executor = _make_executor(mock_trading, MagicMock())

        result = executor.get_todays_filled_buy_symbols()

        assert result == {"AAPL", "MSFT"}

    def test_excludes_unfilled_orders(self):
        mock_trading = MagicMock()
        mock_trading.get_orders.return_value = [
            _fake_filled_order("AAPL", "buy", 0.0),   # filled_qty=0 → not filled
            _fake_filled_order("MSFT", "buy", 5.0),
        ]
        executor = _make_executor(mock_trading, MagicMock())

        result = executor.get_todays_filled_buy_symbols()

        assert result == {"MSFT"}

    def test_returns_empty_set_when_no_orders(self):
        mock_trading = MagicMock()
        mock_trading.get_orders.return_value = []
        executor = _make_executor(mock_trading, MagicMock())

        assert executor.get_todays_filled_buy_symbols() == set()

    def test_returns_empty_set_on_api_error(self):
        mock_trading = MagicMock()
        mock_trading.get_orders.side_effect = RuntimeError("Alpaca down")
        executor = _make_executor(mock_trading, MagicMock())

        # Must not raise — fallback returns empty set
        result = executor.get_todays_filled_buy_symbols()
        assert result == set()


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


# ===========================================================================
# Cash-only trading enforcement
# ===========================================================================

class TestCashOnlyTrading:
    """Verify that all buy-side sizing is based on cash, not equity."""

    def _make_mock_alpaca(self, mid_price: float = 100.0) -> MagicMock:
        mock = MagicMock()
        mock.get_latest_quote.return_value = {"mid": mid_price, "bid": mid_price - 0.1, "ask": mid_price + 0.1}
        return mock

    def test_buy_sized_off_cash_not_equity(self):
        """With equity=200K but cash=10K, buy qty must be much less than equity-based qty."""
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        executor = _make_executor(mock_trading, mock_alpaca)

        account = {
            "equity":          200_000.0,
            "cash":             10_000.0,
            "buying_power":    400_000.0,
            "portfolio_value": 200_000.0,
        }
        # kelly_f ≈ 0.10, price = 100
        # cash-based:   10_000 * ~0.10 / 100 ≈ 10 shares
        # equity-based: 200_000 * ~0.10 / 100 ≈ 200 shares (must NOT happen)
        result = executor.submit_order("AAPL", 1, 1.0, account, _signal_stats())

        assert result["status"] == "submitted"
        # Verify sizing is cash-based: notional must be well below equity-based value
        assert result["notional"] <= 10_000.0   # capped at cash, not equity (200K * 10% = 20K)

    def test_hard_cash_cap_limits_buy_shares(self):
        """Hard cap ensures order cost never exceeds available cash even after sizing."""
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = []
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        executor = _make_executor(mock_trading, mock_alpaca)

        account = {
            "equity":          100_000.0,
            "cash":              5_000.0,
            "buying_power":    200_000.0,
            "portfolio_value": 100_000.0,
        }
        result = executor.submit_order("AAPL", 1, 1.0, account, _signal_stats())

        assert result["status"] == "submitted"
        # Notional must never exceed available cash (hard cap guarantees this)
        assert result["notional"] <= 5_000.0

    def test_sell_not_blocked_by_zero_cash(self):
        """Sell orders must succeed even when cash=0 (sells free cash, not consume it)."""
        mock_trading = MagicMock()
        mock_trading.get_all_positions.return_value = [
            _fake_position("AAPL", 50.0, 5_000.0)
        ]
        mock_trading.submit_order.return_value = _fake_order()
        mock_alpaca = self._make_mock_alpaca(mid_price=100.0)
        executor = _make_executor(mock_trading, mock_alpaca)

        account = {
            "equity":          100_000.0,
            "cash":                  0.0,   # no cash
            "buying_power":          0.0,
            "portfolio_value":   100_000.0,
        }
        result = executor.submit_order("AAPL", -1, 1.0, account, _signal_stats())

        # Sell should go through; cash cap only applies to buys
        assert result["status"] == "submitted"
        assert result["side"] == "sell"
        mock_trading.submit_order.assert_called_once()

    def test_check_trade_max_size_capped_at_cash(self):
        """check_trade max_size must not exceed available cash."""
        rm = _make_risk_manager(max_position_pct=0.10)

        account = {
            "equity":          200_000.0,
            "cash":             10_000.0,   # far less than 10% of equity (20K)
            "buying_power":    400_000.0,
            "portfolio_value": 200_000.0,
        }
        result = rm.check_trade("AAPL", 1, account, {})

        assert result["approved"] is True
        # 10% of 200K = 20K, but cash is only 10K → max_size must be 10K
        assert result["max_size"] == pytest.approx(10_000.0)

    # ------------------------------------------------------------------
    # Regime / VIX multiplier tests
    # ------------------------------------------------------------------

    def test_regime_bear_halves_kelly(self):
        """bear regime → max_size ~50 % of neutral baseline."""
        rm = _make_risk_manager(max_position_pct=0.10)
        account = {"equity": 100_000.0, "cash": 100_000.0}

        neutral = rm.check_trade("AAPL", 1, account, {}, regime="neutral")
        bear    = rm.check_trade("AAPL", 1, account, {}, regime="bear")

        assert bear["approved"] is True
        assert bear["max_size"] == pytest.approx(neutral["max_size"] * 0.5)

    def test_regime_bull_increases_kelly(self):
        """bull regime → max_size ~120 % of neutral baseline."""
        rm = _make_risk_manager(max_position_pct=0.10)
        account = {"equity": 100_000.0, "cash": 100_000.0}

        neutral = rm.check_trade("AAPL", 1, account, {}, regime="neutral")
        bull    = rm.check_trade("AAPL", 1, account, {}, regime="bull")

        assert bull["approved"] is True
        assert bull["max_size"] == pytest.approx(neutral["max_size"] * 1.2)

    def test_vix_40_reduces_to_25_pct(self):
        """vix_multiplier=0.25 → max_size 25 % of base."""
        rm = _make_risk_manager(max_position_pct=0.10)
        account = {"equity": 100_000.0, "cash": 100_000.0}

        base    = rm.check_trade("AAPL", 1, account, {})
        reduced = rm.check_trade("AAPL", 1, account, {}, vix_multiplier=0.25)

        assert reduced["approved"] is True
        assert reduced["max_size"] == pytest.approx(base["max_size"] * 0.25)

    def test_combined_bear_vix40(self):
        """bear (0.5) × vix_multiplier=0.25 → 12.5 % of neutral base."""
        rm = _make_risk_manager(max_position_pct=0.10)
        account = {"equity": 100_000.0, "cash": 100_000.0}

        base     = rm.check_trade("AAPL", 1, account, {}, regime="neutral")
        combined = rm.check_trade("AAPL", 1, account, {}, regime="bear", vix_multiplier=0.25)

        assert combined["approved"] is True
        assert combined["max_size"] == pytest.approx(base["max_size"] * 0.5 * 0.25)
