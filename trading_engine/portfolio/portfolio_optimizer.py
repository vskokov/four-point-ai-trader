"""Portfolio optimization layer — Black-Litterman + Min-Variance.

Uses PyPortfolioOpt with LedoitWolf covariance shrinkage to produce
cross-asset target weights conditioned on MWU ensemble signals.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

import trading_engine.config.settings as settings
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Cross-asset portfolio optimizer.

    Combines Black-Litterman views derived from MWU signal scores with
    LedoitWolf-shrunk sample covariance to produce target portfolio weights.
    Falls back to minimum-variance when no views pass the confidence threshold.

    Parameters
    ----------
    tickers:
        Universe of equity symbols.
    risk_free_rate:
        Annualised risk-free rate used for Sharpe maximisation.
    max_weight:
        Per-ticker weight ceiling (fraction of portfolio, e.g. 0.10 = 10 %).
    min_weight:
        Per-ticker weight floor.  0.0 = long-only.
    lookback_days:
        Number of trading days of history used to estimate covariance.
        Calendar days fetched = lookback_days * 1.4 (generous buffer).
    """

    def __init__(
        self,
        tickers: list[str],
        risk_free_rate: float = 0.05,
        max_weight: float = 0.10,
        min_weight: float = 0.0,
        lookback_days: int = 252,
    ) -> None:
        self.tickers = list(tickers)
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.lookback_days = lookback_days

        self.target_weights: dict[str, float] = {t: 0.0 for t in self.tickers}
        self.last_optimized: datetime | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_return_matrix(self, end: datetime | None = None) -> pd.DataFrame:
        """
        Fetch OHLCV from TimescaleDB and compute daily log-return matrix.

        Returns
        -------
        pd.DataFrame
            Index = date, columns = available tickers.
            Log returns, forward-filled and NaN-dropped.
        """
        from trading_engine.data.storage import Storage

        if end is None:
            end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=int(self.lookback_days * 1.4))

        storage = Storage(settings.DB_URL)
        close_prices: dict[str, pd.Series] = {}

        try:
            for ticker in self.tickers:
                df = storage.query_ohlcv(ticker, start, end)
                if df.empty:
                    logger.warning("portfolio.no_data_for_ticker", ticker=ticker)
                    continue
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.set_index("time")
                daily_close = df["close"].resample("1D").last().dropna()
                if len(daily_close) < 10:
                    logger.warning(
                        "portfolio.insufficient_bars",
                        ticker=ticker,
                        n=len(daily_close),
                    )
                    continue
                close_prices[ticker] = daily_close
        finally:
            storage.dispose()

        if not close_prices:
            raise ValueError("No OHLCV data available for any ticker in universe")

        price_df = pd.DataFrame(close_prices)
        returns = np.log(price_df / price_df.shift(1)).iloc[1:]

        # Drop tickers with >10 % missing data
        missing_pct = returns.isna().mean()
        drop_tickers = missing_pct[missing_pct > 0.10].index.tolist()
        if drop_tickers:
            logger.warning(
                "portfolio.dropping_high_missing_tickers",
                tickers=drop_tickers,
                threshold_pct=10,
            )
            returns = returns.drop(columns=drop_tickers)

        # Forward-fill up to 3 days, then drop remaining NaNs
        returns = returns.ffill(limit=3).dropna()

        return returns

    def _equal_weight_prior(self, tickers: list[str]) -> pd.Series:
        """Uniform market prior: 1/n for each ticker in *tickers*."""
        n = len(tickers)
        return pd.Series([1.0 / n] * n, index=tickers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_black_litterman(
        self,
        mwu_scores: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compute Black-Litterman portfolio weights conditioned on MWU signals.

        Parameters
        ----------
        mwu_scores:
            Mapping of ticker → {"score": float, "confidence": float,
            "final_signal": int}.  Only entries with abs(score) >= 0.3 AND
            confidence >= 0.4 are used as views.

        Returns
        -------
        dict
            Keys: weights (dict[str, float]), method (str),
            n_views (int), timestamp (datetime).
        """
        from pypfopt import BlackLittermanModel, EfficientFrontier
        from sklearn.covariance import LedoitWolf

        returns = self._get_return_matrix()
        available = list(returns.columns)

        # LedoitWolf covariance shrinkage
        lw = LedoitWolf().fit(returns.dropna())
        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=available,
            columns=available,
        )

        # Build absolute views — only confident signals
        views: dict[str, float] = {}
        view_conf: dict[str, float] = {}
        for ticker in available:
            if ticker not in mwu_scores:
                continue
            info = mwu_scores[ticker]
            score = float(info.get("score", 0.0))
            confidence = float(info.get("confidence", 0.0))
            if abs(score) >= 0.3 and confidence >= 0.4:
                # Map ±1 score to ±20 % annual return view
                views[ticker] = score * 0.20
                # Lower confidence → higher uncertainty → wider omega
                view_conf[ticker] = 1.0 - confidence

        # Fall back to min-variance if no views pass threshold
        if not views:
            logger.info(
                "portfolio.bl_no_views_fallback",
                reason="no signals passed abs(score)>=0.3 and confidence>=0.4",
            )
            return self.compute_min_variance()

        # Omega: diagonal matrix of view uncertainties (n_views × n_views)
        omega = np.diag([view_conf[t] for t in views])

        prior = self._equal_weight_prior(available)

        bl = BlackLittermanModel(
            cov_matrix,
            pi=prior,
            absolute_views=views,
            omega=omega,
        )
        bl_return = bl.bl_returns()
        bl_cov = bl.bl_cov()

        # Ensure max_weight >= 1/n so the weight sum constraint is feasible
        effective_max = max(self.max_weight, 1.0 / len(available))
        ef = EfficientFrontier(
            bl_return,
            bl_cov,
            weight_bounds=(self.min_weight, effective_max),
        )
        try:
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        except Exception as exc:
            logger.warning(
                "portfolio.max_sharpe_infeasible",
                error=str(exc),
                fallback="min_volatility",
            )
            ef = EfficientFrontier(
                bl_return,
                bl_cov,
                weight_bounds=(self.min_weight, effective_max),
            )
            ef.min_volatility()

        weights = dict(ef.clean_weights())

        self.target_weights = {t: weights.get(t, 0.0) for t in self.tickers}
        self.last_optimized = datetime.now(tz=timezone.utc)

        try:
            perf = ef.portfolio_performance(
                verbose=False, risk_free_rate=self.risk_free_rate
            )
            logger.info(
                "portfolio.bl_optimized",
                weights=weights,
                expected_return=round(float(perf[0]), 4),
                expected_vol=round(float(perf[1]), 4),
                sharpe=round(float(perf[2]), 4),
                n_views=len(views),
            )
        except Exception:
            logger.info("portfolio.bl_optimized", weights=weights, n_views=len(views))

        return {
            "weights": weights,
            "method": "black_litterman",
            "n_views": len(views),
            "timestamp": self.last_optimized,
        }

    def compute_min_variance(self) -> dict[str, Any]:
        """
        Compute minimum-variance portfolio weights (no return views required).

        Returns
        -------
        dict
            Keys: weights (dict[str, float]), method (str),
            n_views (int = 0), timestamp (datetime).
        """
        from pypfopt import EfficientFrontier
        from sklearn.covariance import LedoitWolf

        returns = self._get_return_matrix()
        available = list(returns.columns)

        lw = LedoitWolf().fit(returns.dropna())
        cov_matrix = pd.DataFrame(
            lw.covariance_,
            index=available,
            columns=available,
        )

        # Ensure max_weight >= 1/n so the weight sum constraint is feasible
        effective_max = max(self.max_weight, 1.0 / len(available))
        ef = EfficientFrontier(
            None,
            cov_matrix,
            weight_bounds=(self.min_weight, effective_max),
        )
        ef.min_volatility()
        weights = dict(ef.clean_weights())

        self.target_weights = {t: weights.get(t, 0.0) for t in self.tickers}
        self.last_optimized = datetime.now(tz=timezone.utc)

        try:
            perf = ef.portfolio_performance(verbose=False)
            logger.info(
                "portfolio.min_variance_optimized",
                weights=weights,
                expected_vol=round(float(perf[1]), 4),
            )
        except Exception:
            logger.info("portfolio.min_variance_optimized", weights=weights)

        return {
            "weights": weights,
            "method": "min_variance",
            "n_views": 0,
            "timestamp": self.last_optimized,
        }

    def get_rebalance_orders(
        self,
        current_positions: pd.DataFrame,
        account_equity: float,
        min_trade_pct: float = 0.005,
    ) -> list[dict[str, Any]]:
        """
        Compute the rebalance orders needed to reach target weights.

        Parameters
        ----------
        current_positions:
            DataFrame with columns ``ticker`` and ``market_value``.
            Matches the output of ``OrderExecutor.get_positions()``.
        account_equity:
            Total account equity in USD.
        min_trade_pct:
            Minimum |delta_weight| to generate a buy/sell; smaller deltas
            are labelled "hold" to avoid unnecessary churn.

        Returns
        -------
        list[dict]
            Each entry contains: ticker, action, target_weight,
            current_weight, delta_weight, dollar_amount.
        """
        if account_equity <= 0:
            return []

        # Build current weight map
        current_weights: dict[str, float] = {}
        if not current_positions.empty and "ticker" in current_positions.columns:
            for _, row in current_positions.iterrows():
                ticker = str(row["ticker"])
                mv = float(row.get("market_value", 0.0))
                current_weights[ticker] = mv / account_equity

        orders: list[dict[str, Any]] = []
        n_buys = n_sells = n_holds = 0

        for ticker, target_w in self.target_weights.items():
            current_w = current_weights.get(ticker, 0.0)
            delta = target_w - current_w

            if abs(delta) < min_trade_pct:
                action = "hold"
                n_holds += 1
            elif delta > 0:
                action = "buy"
                n_buys += 1
            else:
                action = "sell"
                n_sells += 1

            orders.append(
                {
                    "ticker": ticker,
                    "action": action,
                    "target_weight": target_w,
                    "current_weight": current_w,
                    "delta_weight": delta,
                    "dollar_amount": abs(delta * account_equity),
                }
            )

        total_turnover = sum(
            abs(o["delta_weight"]) for o in orders if o["action"] != "hold"
        )
        logger.info(
            "portfolio.rebalance_orders",
            n_buys=n_buys,
            n_sells=n_sells,
            n_holds=n_holds,
            total_turnover_pct=round(total_turnover * 100, 2),
        )

        return orders

    def get_target_weight(self, ticker: str) -> float:
        """
        Return the current target weight for *ticker*.

        Logs a warning if weights are uninitialized or stale (> 24 h old).

        Returns
        -------
        float
            Weight in [0, max_weight], or 0.0 for unknown tickers.
        """
        if self.last_optimized is None:
            logger.warning(
                "portfolio.weights_never_optimized",
                ticker=ticker,
                hint="call compute_black_litterman or compute_min_variance first",
            )
        else:
            age = datetime.now(tz=timezone.utc) - self.last_optimized
            if age.total_seconds() > 86_400:
                logger.warning(
                    "portfolio.weights_stale",
                    ticker=ticker,
                    age_hours=round(age.total_seconds() / 3600, 1),
                )

        return self.target_weights.get(ticker, 0.0)
