"""
Portfolio Optimizer

Implements portfolio optimization methods:
- Mean-Variance (Markowitz) - maximize Sharpe ratio
- Risk Parity - equal risk contribution from each asset
- Maximum Diversification - maximize diversification ratio

Usage:
    optimizer = PortfolioOptimizer(broker)
    weights = await optimizer.optimize_risk_parity(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
        max_weight=0.25,
    )

    # weights = {"AAPL": 0.25, "MSFT": 0.30, "GOOGL": 0.20, "AMZN": 0.25}
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: Dict[str, float]
    method: str
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    timestamp: datetime

    # Risk metrics
    max_weight: float
    min_weight: float
    effective_n: float  # Effective number of holdings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "method": self.method,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "diversification_ratio": self.diversification_ratio,
            "timestamp": self.timestamp.isoformat(),
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "effective_n": self.effective_n,
        }


class PortfolioOptimizer:
    """
    Portfolio optimization with multiple methods.
    """

    def __init__(
        self,
        broker,
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
    ):
        """
        Initialize portfolio optimizer.

        Args:
            broker: Trading broker instance
            lookback_days: Days of history for covariance estimation
            risk_free_rate: Annual risk-free rate
        """
        self.broker = broker
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate

        # Cache for covariance matrix
        self._cov_cache = None
        self._cov_cache_time = None
        self._cov_cache_ttl = 3600  # 1 hour

    async def optimize_mean_variance(
        self,
        symbols: List[str],
        target_return: Optional[float] = None,
        max_weight: float = 0.25,
        min_weight: float = 0.02,
    ) -> OptimizationResult:
        """
        Mean-variance optimization (Markowitz).

        Maximizes Sharpe ratio subject to constraints.

        Args:
            symbols: List of symbols to include
            target_return: Target annual return (None = max Sharpe)
            max_weight: Maximum weight per position
            min_weight: Minimum weight per position

        Returns:
            OptimizationResult with optimal weights
        """
        # Get returns and covariance
        returns, cov_matrix = await self._get_returns_and_covariance(symbols)

        if returns is None or cov_matrix is None:
            return self._equal_weight_fallback(symbols, "mean_variance")

        n = len(symbols)

        try:
            # Simple optimization using analytical solution for max Sharpe
            # For production, use scipy.optimize.minimize

            # Inverse covariance weighted
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n)

            # Max Sharpe weights (unconstrained)
            excess_returns = returns - self.risk_free_rate / 252
            raw_weights = inv_cov @ excess_returns
            raw_weights = raw_weights / np.sum(raw_weights)

            # Apply constraints
            weights = np.clip(raw_weights, min_weight, max_weight)
            weights = weights / np.sum(weights)  # Renormalize

            # Calculate metrics
            port_return = np.sum(weights * returns) * 252
            port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
            sharpe = (port_return - self.risk_free_rate) / port_vol

            # Diversification ratio
            weighted_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
            div_ratio = (weights @ weighted_vols) / port_vol

            # Effective N
            effective_n = 1 / np.sum(weights ** 2)

            return OptimizationResult(
                weights=dict(zip(symbols, weights)),
                method="mean_variance",
                expected_return=port_return,
                expected_volatility=port_vol,
                sharpe_ratio=sharpe,
                diversification_ratio=div_ratio,
                timestamp=datetime.now(),
                max_weight=float(np.max(weights)),
                min_weight=float(np.min(weights)),
                effective_n=effective_n,
            )

        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return self._equal_weight_fallback(symbols, "mean_variance")

    async def optimize_risk_parity(
        self,
        symbols: List[str],
        max_weight: float = 0.25,
        iterations: int = 100,
    ) -> OptimizationResult:
        """
        Risk parity optimization.

        Each asset contributes equally to portfolio risk.

        Args:
            symbols: List of symbols
            max_weight: Maximum weight per position
            iterations: Optimization iterations

        Returns:
            OptimizationResult with risk parity weights
        """
        returns, cov_matrix = await self._get_returns_and_covariance(symbols)

        if returns is None or cov_matrix is None:
            return self._equal_weight_fallback(symbols, "risk_parity")

        n = len(symbols)

        try:
            # Initialize with inverse volatility weights
            vols = np.sqrt(np.diag(cov_matrix))
            weights = (1 / vols) / np.sum(1 / vols)

            # Iterate to find risk parity
            for _ in range(iterations):
                port_var = weights @ cov_matrix @ weights
                marginal_contrib = cov_matrix @ weights
                risk_contrib = weights * marginal_contrib / port_var

                # Target: equal risk contribution
                target_risk = 1 / n

                # Update weights
                adjustment = target_risk / (risk_contrib + 1e-8)
                weights = weights * adjustment
                weights = weights / np.sum(weights)

                # Apply max weight constraint
                weights = np.minimum(weights, max_weight)
                weights = weights / np.sum(weights)

            # Calculate metrics
            port_return = np.sum(weights * returns) * 252
            port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

            weighted_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
            div_ratio = (weights @ weighted_vols) / port_vol if port_vol > 0 else 1

            effective_n = 1 / np.sum(weights ** 2)

            return OptimizationResult(
                weights=dict(zip(symbols, weights)),
                method="risk_parity",
                expected_return=port_return,
                expected_volatility=port_vol,
                sharpe_ratio=sharpe,
                diversification_ratio=div_ratio,
                timestamp=datetime.now(),
                max_weight=float(np.max(weights)),
                min_weight=float(np.min(weights)),
                effective_n=effective_n,
            )

        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return self._equal_weight_fallback(symbols, "risk_parity")

    async def optimize_max_diversification(
        self,
        symbols: List[str],
        max_weight: float = 0.25,
        min_weight: float = 0.02,
    ) -> OptimizationResult:
        """
        Maximum diversification optimization.

        Maximizes the diversification ratio (weighted avg vol / portfolio vol).

        Args:
            symbols: List of symbols
            max_weight: Maximum weight per position
            min_weight: Minimum weight per position

        Returns:
            OptimizationResult with max diversification weights
        """
        returns, cov_matrix = await self._get_returns_and_covariance(symbols)

        if returns is None or cov_matrix is None:
            return self._equal_weight_fallback(symbols, "max_diversification")

        n = len(symbols)
        vols = np.sqrt(np.diag(cov_matrix))

        try:
            # Correlation matrix
            corr_matrix = cov_matrix / np.outer(vols, vols)

            # Max diversification = minimize portfolio vol / weighted avg vol
            # Equivalent to inverse correlation weighted

            inv_corr = np.linalg.inv(corr_matrix)
            ones = np.ones(n)

            raw_weights = inv_corr @ ones
            raw_weights = raw_weights / np.sum(raw_weights)

            # Apply constraints
            weights = np.clip(raw_weights, min_weight, max_weight)
            weights = weights / np.sum(weights)

            # Calculate metrics
            port_return = np.sum(weights * returns) * 252
            port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

            weighted_vols = vols * np.sqrt(252)
            div_ratio = (weights @ weighted_vols) / port_vol if port_vol > 0 else 1

            effective_n = 1 / np.sum(weights ** 2)

            return OptimizationResult(
                weights=dict(zip(symbols, weights)),
                method="max_diversification",
                expected_return=port_return,
                expected_volatility=port_vol,
                sharpe_ratio=sharpe,
                diversification_ratio=div_ratio,
                timestamp=datetime.now(),
                max_weight=float(np.max(weights)),
                min_weight=float(np.min(weights)),
                effective_n=effective_n,
            )

        except Exception as e:
            logger.error(f"Max diversification optimization failed: {e}")
            return self._equal_weight_fallback(symbols, "max_diversification")

    async def _get_returns_and_covariance(
        self, symbols: List[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get returns and covariance matrix for symbols."""
        returns_data = {}

        for symbol in symbols:
            try:
                end_date = date.today()
                start_date = end_date - timedelta(days=int(self.lookback_days * 1.5))

                bars = await self.broker.get_bars(
                    symbol,
                    timeframe="1Day",
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )

                if bars and len(bars) > 50:
                    prices = np.array([float(b.close) for b in bars])
                    rets = np.diff(np.log(prices))
                    returns_data[symbol] = rets[-self.lookback_days:]

            except Exception as e:
                logger.debug(f"Error fetching data for {symbol}: {e}")

        if len(returns_data) < 2:
            return None, None

        # Align returns (use shortest)
        min_len = min(len(r) for r in returns_data.values())
        aligned_returns = {s: r[-min_len:] for s, r in returns_data.items()}

        # Build matrix
        symbols_with_data = list(aligned_returns.keys())
        returns_matrix = np.column_stack([aligned_returns[s] for s in symbols_with_data])

        # Calculate expected returns and covariance
        expected_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix, rowvar=False)

        return expected_returns, cov_matrix

    def _equal_weight_fallback(
        self, symbols: List[str], method: str
    ) -> OptimizationResult:
        """Return equal-weight portfolio as fallback."""
        n = len(symbols)
        weights = {s: 1.0 / n for s in symbols}

        logger.warning(f"Falling back to equal weights for {method}")

        return OptimizationResult(
            weights=weights,
            method=f"{method}_fallback",
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            diversification_ratio=1.0,
            timestamp=datetime.now(),
            max_weight=1.0 / n,
            min_weight=1.0 / n,
            effective_n=n,
        )

    async def get_rebalance_trades(
        self,
        target_weights: Dict[str, float],
        current_positions: Dict[str, float],
        total_value: float,
        threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Calculate trades needed to rebalance to target weights.

        Args:
            target_weights: Target portfolio weights
            current_positions: Current position values
            total_value: Total portfolio value
            threshold: Minimum weight difference to trigger trade

        Returns:
            List of trade dicts with symbol, side, value
        """
        trades = []

        # Calculate current weights
        current_weights = {
            s: v / total_value for s, v in current_positions.items()
        }

        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < threshold:
                continue

            trade_value = weight_diff * total_value

            trades.append({
                "symbol": symbol,
                "side": "buy" if trade_value > 0 else "sell",
                "value": abs(trade_value),
                "target_weight": target_weight,
                "current_weight": current_weight,
                "weight_change": weight_diff,
            })

        # Sort by absolute value (largest trades first)
        trades.sort(key=lambda x: x["value"], reverse=True)

        return trades


class AlphaMonitor:
    """
    Monitors strategy alpha decay and performance degradation.

    Detects when strategies lose their edge:
    - Rolling Sharpe ratio decline
    - Signal decay correlation
    - Trend detection in performance
    """

    def __init__(
        self,
        lookback_trades: int = 100,
        min_sharpe: float = 0.5,
        decay_threshold: float = 0.3,
    ):
        """
        Initialize alpha monitor.

        Args:
            lookback_trades: Number of trades to analyze
            min_sharpe: Minimum Sharpe ratio before alert
            decay_threshold: Decline threshold for decay detection
        """
        self.lookback_trades = lookback_trades
        self.min_sharpe = min_sharpe
        self.decay_threshold = decay_threshold

        self._trade_history: List[Dict[str, Any]] = []

    def record_trade(
        self,
        strategy_name: str,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        signal_strength: float,
        timestamp: Optional[datetime] = None,
    ):
        """Record a completed trade for monitoring."""
        self._trade_history.append({
            "strategy": strategy_name,
            "symbol": symbol,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "signal_strength": signal_strength,
            "timestamp": timestamp or datetime.now(),
        })

        # Keep only recent trades
        if len(self._trade_history) > self.lookback_trades * 2:
            self._trade_history = self._trade_history[-self.lookback_trades:]

    def check_alpha_decay(
        self, strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check for alpha decay in strategy performance.

        Returns:
            Dict with decay status and metrics
        """
        trades = self._trade_history
        if strategy_name:
            trades = [t for t in trades if t["strategy"] == strategy_name]

        if len(trades) < 20:
            return {
                "has_sufficient_data": False,
                "alert": False,
                "message": "Insufficient trade history",
            }

        # Calculate rolling Sharpe
        pnls = np.array([t["pnl_pct"] for t in trades[-self.lookback_trades:]])

        if len(pnls) < 30:
            window = len(pnls) // 2
        else:
            window = 30

        rolling_sharpe = []
        for i in range(window, len(pnls)):
            period_pnls = pnls[i-window:i]
            if np.std(period_pnls) > 0:
                sharpe = np.mean(period_pnls) / np.std(period_pnls) * np.sqrt(252)
                rolling_sharpe.append(sharpe)

        if len(rolling_sharpe) < 3:
            return {
                "has_sufficient_data": False,
                "alert": False,
                "message": "Insufficient rolling history",
            }

        current_sharpe = rolling_sharpe[-1]
        peak_sharpe = max(rolling_sharpe)
        avg_sharpe = np.mean(rolling_sharpe)

        # Check for decay
        sharpe_decline = (peak_sharpe - current_sharpe) / max(abs(peak_sharpe), 0.01)

        # Trend detection using linear regression
        x = np.arange(len(rolling_sharpe))
        coeffs = np.polyfit(x, rolling_sharpe, 1)
        trend_slope = coeffs[0]

        # Signal-to-performance correlation
        if len(trades) >= 20:
            signal_strengths = [t["signal_strength"] for t in trades[-20:]]
            pnl_pcts = [t["pnl_pct"] for t in trades[-20:]]
            signal_correlation = np.corrcoef(signal_strengths, pnl_pcts)[0, 1]
            if np.isnan(signal_correlation):
                signal_correlation = 0
        else:
            signal_correlation = 0

        # Determine if alert needed
        alert = False
        alert_reasons = []

        if current_sharpe < self.min_sharpe:
            alert = True
            alert_reasons.append(f"Sharpe below minimum ({current_sharpe:.2f} < {self.min_sharpe})")

        if sharpe_decline > self.decay_threshold:
            alert = True
            alert_reasons.append(f"Sharpe declined {sharpe_decline:.0%} from peak")

        if trend_slope < -0.01:
            alert = True
            alert_reasons.append(f"Declining Sharpe trend (slope: {trend_slope:.3f})")

        if signal_correlation < 0.2:
            alert = True
            alert_reasons.append(f"Low signal correlation ({signal_correlation:.2f})")

        return {
            "has_sufficient_data": True,
            "alert": alert,
            "alert_reasons": alert_reasons,
            "current_sharpe": current_sharpe,
            "peak_sharpe": peak_sharpe,
            "avg_sharpe": avg_sharpe,
            "sharpe_decline": sharpe_decline,
            "trend_slope": trend_slope,
            "signal_correlation": signal_correlation,
            "trade_count": len(trades),
            "recommendation": self._get_recommendation(alert, alert_reasons),
        }

    def _get_recommendation(
        self, alert: bool, reasons: List[str]
    ) -> str:
        """Get recommendation based on alpha decay analysis."""
        if not alert:
            return "Strategy performing normally. Continue monitoring."

        if "Sharpe below minimum" in str(reasons):
            return "Consider reducing position sizes or pausing strategy for review."

        if "Declining" in str(reasons):
            return "Strategy may be losing edge. Investigate market regime changes."

        if "signal correlation" in str(reasons):
            return "Signals not predicting returns well. Consider retraining model."

        return "Monitor closely and consider reducing exposure."

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if len(self._trade_history) < 10:
            return {"message": "Insufficient trade history"}

        trades = self._trade_history[-self.lookback_trades:]
        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]

        return {
            "trade_count": len(trades),
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            "avg_win": np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
            "avg_loss": np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
            "sharpe": np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(252) if np.std(pnl_pcts) > 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(pnls),
            "alpha_decay_check": self.check_alpha_decay(),
        }

    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from P&L series."""
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        return float(np.min(drawdown)) if len(drawdown) > 0 else 0
