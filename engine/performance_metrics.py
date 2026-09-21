"""
Performance Metrics Module

Calculates trading strategy performance metrics with institutional-grade
statistical validation including:
- Multiple testing correction (Bonferroni, Benjamini-Hochberg FDR)
- Effect size reporting (Cohen's d, Hedge's g)
- Bootstrap confidence intervals
- Permutation test integration

The pure statistical primitives (Bonferroni / FDR correction, Cohen's d,
Hedge's g, dataclasses) live in ``engine.statistical_testing`` and are
re-exported here for backwards compatibility. ``calculate_effect_size`` is
defined in this module so that monkeypatch-based tests can rebind the
helper functions in ``engine.performance_metrics`` and have those overrides
flow through to the composed calculation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from engine.statistical_testing import (
    EffectSizeResult,
    SignificanceResult,
    apply_bonferroni_correction,
    apply_fdr_correction,
    calculate_adjusted_significance,
    calculate_cohens_d,
    calculate_hedges_g,
)

__all__ = [
    "EffectSizeResult",
    "PerformanceMetrics",
    "SignificanceResult",
    "apply_bonferroni_correction",
    "apply_fdr_correction",
    "calculate_adjusted_significance",
    "calculate_cohens_d",
    "calculate_effect_size",
    "calculate_hedges_g",
]

logger = logging.getLogger(__name__)


def calculate_effect_size(
    returns: np.ndarray,
    population_mean: float = 0.0,
    confidence_level: float = 0.95,
) -> EffectSizeResult:
    """
    Calculate effect sizes with confidence intervals and interpretation.

    Reports PRACTICAL significance alongside statistical significance.
    A p-value tells you if an effect exists; effect size tells you if
    it matters.

    The helper calls (``calculate_cohens_d``, ``calculate_hedges_g``) are
    looked up via module globals at call time, which lets tests monkeypatch
    them on ``engine.performance_metrics`` and have the overrides flow
    through to the composed result.

    Args:
        returns: Array of returns
        population_mean: Benchmark (default 0)
        confidence_level: For CI calculation (default 0.95)

    Returns:
        EffectSizeResult with Cohen's d, Hedge's g, CI, and interpretation
    """
    n = len(returns)
    if n < 4:
        return EffectSizeResult(
            cohens_d=0.0,
            hedges_g=0.0,
            interpretation="Insufficient data for effect size calculation",
            confidence_interval=(0.0, 0.0),
        )

    d = calculate_cohens_d(returns, population_mean)
    g = calculate_hedges_g(returns, population_mean)

    # Approximate CI for Cohen's d using non-central t distribution
    # Simplified: use normal approximation for large n
    se_d = np.sqrt((1 / n) + (d**2 / (2 * n)))
    z = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = d - z * se_d
    ci_upper = d + z * se_d

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    direction = "positive" if d > 0 else "negative" if d < 0 else "zero"

    interpretation = (
        f"{magnitude.capitalize()} {direction} effect (d={d:.3f}, g={g:.3f}). "
        f"Returns are {abs_d:.2f} standard deviations "
        f"{'above' if d > 0 else 'below'} the benchmark."
    )

    return EffectSizeResult(
        cohens_d=d,
        hedges_g=g,
        interpretation=interpretation,
        confidence_interval=(ci_lower, ci_upper),
    )


class PerformanceMetrics:
    """
    Class for calculating performance metrics for trading strategies.
    """

    def __init__(self, risk_free_rate=0.02):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%).
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.

        Args:
            backtest_result: Dictionary with backtest results containing:
                - 'equity_curve': List of portfolio values over time
                - 'trades': List of trade dictionaries
                - 'start_date': Start date of backtest
                - 'end_date': End date of backtest

        Returns:
            Dictionary with calculated metrics.
        """
        try:
            # Extract required data
            equity_curve = backtest_result.get("equity_curve", [])
            trades = backtest_result.get("trades", [])
            start_date_raw = backtest_result.get("start_date")
            end_date_raw = backtest_result.get("end_date")
            start_date: Optional[datetime] = (
                start_date_raw if isinstance(start_date_raw, datetime) else None
            )
            end_date: Optional[datetime] = (
                end_date_raw if isinstance(end_date_raw, datetime) else None
            )
            initial_capital = backtest_result.get("initial_capital", 100000)

            if not equity_curve or len(equity_curve) < 2:
                logger.warning("Insufficient equity curve data for metrics calculation")
                return self._empty_metrics()

            # Create numpy arrays for calculations
            equity_array = np.array(equity_curve, dtype=float)

            # Calculate basic returns
            total_return = float((equity_array[-1] / equity_array[0]) - 1)

            # Calculate daily returns
            daily_returns = np.diff(equity_array) / equity_array[:-1]

            # Calculate metrics
            avg_win, avg_loss = self._calculate_avg_win_loss(trades)
            metrics = {
                "total_return": total_return,
                "annualized_return": self._calculate_annualized_return(
                    total_return, start_date, end_date
                ),
                "max_drawdown": self._calculate_max_drawdown(equity_array),
                "sharpe_ratio": self._calculate_sharpe_ratio(daily_returns),
                "sortino_ratio": self._calculate_sortino_ratio(daily_returns),
                "win_rate": self._calculate_win_rate(trades),
                "profit_factor": self._calculate_profit_factor(trades),
                "avg_trade": self._calculate_avg_trade(trades),
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "volatility": float(np.std(daily_returns)) if len(daily_returns) > 0 else 0.0,
                "trade_count": len(trades),
                "num_trades": len(trades),  # Alias for compatibility
                "final_equity": (
                    float(equity_array[-1]) if len(equity_array) > 0 else float(initial_capital)
                ),
            }

            # Calculate additional metrics
            metrics["calmar_ratio"] = self._calculate_calmar_ratio(
                metrics["annualized_return"], metrics["max_drawdown"]
            )

            # Calculate recovery factor
            metrics["recovery_factor"] = (
                metrics["total_return"] / metrics["max_drawdown"]
                if metrics["max_drawdown"] > 0
                else 0
            )

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary with zeros."""
        return {
            "total_return": 0,
            "annualized_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "volatility": 0,
            "calmar_ratio": 0,
            "recovery_factor": 0,
            "trade_count": 0,
            "num_trades": 0,
            "final_equity": 0,
        }

    def _calculate_annualized_return(
        self, total_return: float, start_date: object, end_date: object
    ) -> float:
        """Calculate annualized return."""
        # Defensive: callers/tests may pass non-datetime values; treat as invalid.
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            return 0.0

        years = (end_date - start_date).days / 365.25
        if years <= 0:
            return 0.0

        return float((1 + total_return) ** (1 / years) - 1)

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return float(abs(np.min(drawdown))) if len(drawdown) > 0 else 0.0

    def _calculate_sharpe_ratio(self, returns: np.ndarray, period=252) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) == 0:
            return 0.0

        # Convert annual risk-free rate to period risk-free rate
        period_risk_free = (1 + self.risk_free_rate) ** (1 / period) - 1

        excess_returns = returns - period_risk_free
        std = float(np.std(returns))
        if std == 0:
            return 0.0

        return float(float(np.mean(excess_returns)) / std * np.sqrt(period))

    def _calculate_sortino_ratio(self, returns: np.ndarray, period=252) -> float:
        """Calculate Sortino ratio (annualized)."""
        if len(returns) == 0:
            return 0.0

        # Convert annual risk-free rate to period risk-free rate
        period_risk_free = (1 + self.risk_free_rate) ** (1 / period) - 1

        excess_returns = returns - period_risk_free
        downside_returns = returns[returns < 0]

        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
        if len(downside_returns) == 0 or downside_std == 0:
            return 0 if np.mean(excess_returns) <= 0 else float("inf")

        return float(float(np.mean(excess_returns)) / downside_std * np.sqrt(period))

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0 if annualized_return <= 0 else float("inf")

        return annualized_return / max_drawdown

    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0

        winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
        return winning_trades / len(trades)

    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not trades:
            return 0.0

        pnls = [float(trade.get("pnl") or 0.0) for trade in trades]
        gross_profit = sum(p for p in pnls if p > 0.0)
        gross_loss = sum(-p for p in pnls if p < 0.0)

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0

        return float(gross_profit / gross_loss)

    def _calculate_avg_trade(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average trade P&L."""
        if not trades:
            return 0.0

        pnls = [float(trade.get("pnl") or 0.0) for trade in trades]
        total_pnl = sum(pnls)
        return float(total_pnl / len(pnls))

    def _calculate_avg_win_loss(self, trades: List[Dict[str, Any]]) -> tuple[float, float]:
        """Calculate average win and average loss separately.

        Returns:
            Tuple of (avg_win, avg_loss) as percentages
        """
        if not trades:
            return 0.0, 0.0

        pnls = [float(trade.get("pnl") or 0.0) for trade in trades]
        wins = [p for p in pnls if p > 0.0]
        losses = [p for p in pnls if p < 0.0]

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Convert to percentage relative to trade size if possible
        # For now return as raw P&L values normalized
        total_pnl = sum(abs(p) for p in pnls)
        if total_pnl > 0:
            avg_win = avg_win / total_pnl if avg_win else 0
            avg_loss = abs(avg_loss) / total_pnl if avg_loss else 0

        return float(avg_win), float(avg_loss)


@dataclass(frozen=True)
class Verdict:
    """Whether a backtest's numbers are quotable. Computed here, rendered by every script."""

    status: str  # "REPORTED" | "INCONCLUSIVE" | "DATA_UNAVAILABLE"
    reasons: Tuple[str, ...] = ()

    @property
    def quotable(self) -> bool:
        return self.status == "REPORTED"


def verdict(
    metrics: Dict[str, Any],
    n_trades: int,
    data_quality: Optional[Dict[str, Any]] = None,
    *,
    min_trades: int = 50,
) -> Verdict:
    """One judgement of a run: data complete, enough trades, nothing suspicious."""
    reasons = []
    dq = data_quality or {}
    requested = int(dq.get("symbols_requested", 0) or 0)
    loaded = int(dq.get("symbols_loaded", 0) or 0)
    if requested and loaded < requested:
        reasons.append(f"only {loaded} of {requested} symbols loaded")
        return Verdict("DATA_UNAVAILABLE", tuple(reasons))
    if n_trades < min_trades:
        reasons.append(f"{n_trades} trades, below the {min_trades}-trade significance bar")
    sharpe = float(metrics.get("sharpe_ratio", 0.0) or 0.0)
    if sharpe > 3:
        reasons.append(f"Sharpe {sharpe:.2f} is implausibly high; check for lookahead")
    win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
    if win_rate > 0.8:
        reasons.append(f"win rate {win_rate:.0%} is implausibly high; check for lookahead")
    max_dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    total = float(metrics.get("total_return", 0.0) or 0.0)
    if max_dd < 0.01 and total > 0.1:
        reasons.append("near-zero drawdown with a positive return is implausible")
    return Verdict("INCONCLUSIVE" if reasons else "REPORTED", tuple(reasons))
