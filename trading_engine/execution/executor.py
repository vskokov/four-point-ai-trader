"""
Execution layer — Phase 5.1.

RiskManager   — Kelly criterion sizing, position limits, circuit breakers.
OrderExecutor — Market order submission via Alpaca paper-trading.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import trading_engine.config.settings as settings
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Retry helper (reads-only; never retry order submission)
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3


def _read_with_retry(fn: Any, label: str = "") -> Any:
    """Retry a read-only Alpaca call up to _MAX_RETRIES times."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in {401, 403}:
                logger.error("executor.auth_error", label=label, status=status)
                raise
            if attempt == _MAX_RETRIES:
                logger.error("executor.retry_exhausted", label=label, attempts=attempt + 1)
                raise
            import time
            wait = 2 ** attempt
            logger.warning("executor.retry", label=label, attempt=attempt + 1, wait_s=wait)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Enforces pre-trade risk controls and computes fractional Kelly position sizes.

    State tracking
    --------------
    peak_equity       — high-water mark; updated on every check.
    daily_start_equity — equity at first check of each UTC trading day; resets
                         at midnight UTC.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_drawdown_pct: float = 0.15,
        max_daily_loss_pct: float = 0.05,
        kelly_fraction: float = 0.25,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.kelly_fraction = kelly_fraction

        self._peak_equity: float | None = None
        self._daily_start_equity: float | None = None
        self._daily_start_date: date | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_state(self, equity: float) -> None:
        """Update peak equity and daily-start equity."""
        today = datetime.now(timezone.utc).date()
        if self._daily_start_date != today:
            self._daily_start_date = today
            self._daily_start_equity = equity

        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def circuit_breaker(self, account_info: dict[str, Any]) -> bool:
        """
        Return *True* (halt all trading) if drawdown or daily-loss limits are
        breached.  Logs CRITICAL when triggered.
        """
        equity = float(account_info["equity"])
        self._update_state(equity)

        # Drawdown from peak
        if self._peak_equity and self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > self.max_drawdown_pct:
                logger.critical(
                    "risk.circuit_breaker.drawdown_exceeded",
                    drawdown_pct=round(drawdown * 100, 2),
                    limit_pct=self.max_drawdown_pct * 100,
                    peak_equity=self._peak_equity,
                    current_equity=equity,
                )
                return True

        # Daily loss from session open
        if self._daily_start_equity and self._daily_start_equity > 0:
            daily_loss = (self._daily_start_equity - equity) / self._daily_start_equity
            if daily_loss > self.max_daily_loss_pct:
                logger.critical(
                    "risk.circuit_breaker.daily_loss_exceeded",
                    daily_loss_pct=round(daily_loss * 100, 2),
                    limit_pct=self.max_daily_loss_pct * 100,
                    daily_start_equity=self._daily_start_equity,
                    current_equity=equity,
                )
                return True

        return False

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Fractional Kelly position size as a fraction of equity.

        Formula
        -------
        f  = (p * b - q) / b     where b = avg_win / avg_loss
        f* = kelly_fraction * f

        Returns the value clamped to ``[0, max_position_pct]``.
        """
        if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
            return 0.0

        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p
        f = (p * b - q) / b

        if f <= 0:
            return 0.0

        return min(self.kelly_fraction * f, self.max_position_pct)

    def check_trade(
        self,
        ticker: str,
        signal: int,
        account_info: dict[str, Any],
        current_positions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Pre-trade check.

        Parameters
        ----------
        ticker:
            Symbol being traded.
        signal:
            +1 buy, -1 sell, 0 no-op.
        account_info:
            Dict with at least ``equity`` key (from ``AlpacaMarketData.get_account_info``).
        current_positions:
            ``{ticker: {"market_value": float, ...}}``  — keyed by symbol.

        Returns
        -------
        dict
            ``{"approved": bool, "reason": str, "max_size": float}``
            *max_size* is the maximum additional purchase in USD (0 for sells /
            rejected orders).
        """
        if self.circuit_breaker(account_info):
            return {"approved": False, "reason": "circuit_breaker", "max_size": 0.0}

        if signal == 1:  # buy — enforce per-position size limit
            equity = float(account_info["equity"])
            pos = current_positions.get(ticker, {})
            current_mv = float(pos.get("market_value", 0.0))
            current_pct = current_mv / equity if equity > 0 else 0.0

            if current_pct >= self.max_position_pct:
                return {
                    "approved": False,
                    "reason": "position_limit_exceeded",
                    "max_size": 0.0,
                }

            max_size = (self.max_position_pct - current_pct) * equity
            return {"approved": True, "reason": "ok", "max_size": max_size}

        # sell (-1) — circuit breaker already checked; no size limit on reducing
        return {"approved": True, "reason": "ok", "max_size": 0.0}


# ---------------------------------------------------------------------------
# OrderExecutor
# ---------------------------------------------------------------------------

class OrderExecutor:
    """
    Submits market orders to Alpaca paper trading and manages open positions.

    Parameters
    ----------
    alpaca_client:
        An initialised ``AlpacaMarketData`` instance used for price quotes.
    risk_manager:
        A ``RiskManager`` instance that enforces pre-trade controls.
    paper:
        If *True* (default) connects to the Alpaca paper-trading endpoint.
    """

    def __init__(
        self,
        alpaca_client: Any,   # AlpacaMarketData — Any avoids circular import
        risk_manager: RiskManager,
        paper: bool = True,
    ) -> None:
        self._alpaca = alpaca_client
        self._risk = risk_manager
        self._trading = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=paper,
        )
        logger.info("executor.init", paper=paper)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> pd.DataFrame:
        """
        Fetch all open positions from Alpaca.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker``, ``qty``, ``market_value``,
            ``unrealized_pnl``, ``unrealized_pnl_pct``.
            Empty DataFrame (same columns) when no positions are open.
        """
        _COLS = ["ticker", "qty", "market_value", "unrealized_pnl", "unrealized_pnl_pct"]

        positions = _read_with_retry(
            self._trading.get_all_positions,
            label="get_all_positions",
        )

        if not positions:
            return pd.DataFrame(columns=_COLS)

        rows = [
            {
                "ticker":            pos.symbol,
                "qty":               float(pos.qty),
                "market_value":      float(pos.market_value),
                "unrealized_pnl":    float(pos.unrealized_pl),
                "unrealized_pnl_pct": float(pos.unrealized_plpc),
            }
            for pos in positions
        ]
        return pd.DataFrame(rows, columns=_COLS)

    # ------------------------------------------------------------------
    # Emergency liquidation
    # ------------------------------------------------------------------

    def close_all_positions(self) -> None:
        """
        Emergency liquidation — submit market sell orders for every open position.

        Uses Alpaca's ``close_all_positions`` API call (atomic, cancels open
        orders first).  Logs a CRITICAL event before and after.
        """
        logger.critical("executor.close_all_positions.initiated")
        try:
            self._trading.close_all_positions(cancel_orders=True)
            logger.critical("executor.close_all_positions.submitted")
        except Exception as exc:
            logger.critical(
                "executor.close_all_positions.failed",
                error=str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        ticker: str,
        signal: int,
        confidence: float,
        account_info: dict[str, Any],
        signal_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Size and submit a market order for *ticker*.

        Parameters
        ----------
        ticker:
            Symbol, e.g. ``"AAPL"``.
        signal:
            +1 buy, -1 sell, 0 no-op.
        confidence:
            LLM or ensemble confidence in [0, 1].  Scales the dollar size.
        account_info:
            Dict with ``equity`` and other keys from
            ``AlpacaMarketData.get_account_info()``.
        signal_stats:
            Dict with ``win_rate``, ``avg_win``, ``avg_loss`` keys used for
            Kelly sizing.

        Returns
        -------
        dict
            One of::

                {"status": "no_op"}
                {"status": "rejected", "reason": str}
                {"status": "too_small"}
                {"status": "no_position"}   # sell with no open position
                {"status": "submitted", "ticker": str, "side": str,
                 "qty": int, "estimated_price": float,
                 "timestamp": str, "order_id": str}
        """
        if signal == 0:
            return {"status": "no_op"}

        # Fetch current open positions for risk check and sell cap
        pos_df = self.get_positions()
        current_positions: dict[str, dict[str, Any]] = {
            row["ticker"]: {"market_value": row["market_value"], "qty": row["qty"]}
            for _, row in pos_df.iterrows()
        }

        check = self._risk.check_trade(ticker, signal, account_info, current_positions)
        if not check["approved"]:
            logger.warning(
                "executor.order_rejected",
                ticker=ticker,
                signal=signal,
                reason=check["reason"],
            )
            return {"status": "rejected", "reason": check["reason"]}

        # Current mid price
        quote = self._alpaca.get_latest_quote(ticker)
        current_price: float = quote["mid"]

        # Fractional Kelly sizing
        kelly_f = self._risk.kelly_size(
            win_rate=float(signal_stats["win_rate"]),
            avg_win=float(signal_stats["avg_win"]),
            avg_loss=float(signal_stats["avg_loss"]),
        )
        size_usd = float(account_info["equity"]) * kelly_f * confidence
        n_shares = math.floor(size_usd / current_price) if current_price > 0 else 0

        if n_shares == 0:
            logger.info("executor.order_too_small", ticker=ticker, size_usd=size_usd)
            return {"status": "too_small"}

        if signal == -1:
            # Cap sell quantity at current long position
            pos_info = current_positions.get(ticker)
            if pos_info is None:
                logger.warning("executor.sell_no_position", ticker=ticker)
                return {"status": "no_position"}
            held = int(pos_info["qty"])
            n_shares = min(n_shares, held)
            if n_shares == 0:
                return {"status": "too_small"}

        side = OrderSide.BUY if signal == 1 else OrderSide.SELL
        order_req = MarketOrderRequest(
            symbol=ticker,
            qty=n_shares,
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        order = self._trading.submit_order(order_req)

        result: dict[str, Any] = {
            "status":          "submitted",
            "ticker":          ticker,
            "side":            side.value,
            "qty":             n_shares,
            "estimated_price": current_price,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "order_id":        str(order.id),
        }
        logger.info("executor.order_submitted", **result)
        return result
