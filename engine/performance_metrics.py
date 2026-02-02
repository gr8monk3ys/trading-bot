import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


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
            start_date = backtest_result.get("start_date")
            end_date = backtest_result.get("end_date")
            initial_capital = backtest_result.get("initial_capital", 100000)

            if not equity_curve or len(equity_curve) < 2:
                logger.warning("Insufficient equity curve data for metrics calculation")
                return self._empty_metrics()

            # Create numpy arrays for calculations
            equity_array = np.array(equity_curve)

            # Calculate basic returns
            total_return = (equity_array[-1] / equity_array[0]) - 1

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
                "volatility": np.std(daily_returns) if len(daily_returns) > 0 else 0,
                "trade_count": len(trades),
                "num_trades": len(trades),  # Alias for compatibility
                "final_equity": equity_array[-1] if len(equity_array) > 0 else initial_capital,
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
        self, total_return: float, start_date: datetime, end_date: datetime
    ) -> float:
        """Calculate annualized return."""
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            return 0

        years = (end_date - start_date).days / 365.25
        if years <= 0:
            return 0

        return (1 + total_return) ** (1 / years) - 1

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0

    def _calculate_sharpe_ratio(self, returns: np.ndarray, period=252) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) == 0:
            return 0

        # Convert annual risk-free rate to period risk-free rate
        period_risk_free = (1 + self.risk_free_rate) ** (1 / period) - 1

        excess_returns = returns - period_risk_free
        if np.std(returns) == 0:
            return 0

        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(period)
        return sharpe

    def _calculate_sortino_ratio(self, returns: np.ndarray, period=252) -> float:
        """Calculate Sortino ratio (annualized)."""
        if len(returns) == 0:
            return 0

        # Convert annual risk-free rate to period risk-free rate
        period_risk_free = (1 + self.risk_free_rate) ** (1 / period) - 1

        excess_returns = returns - period_risk_free
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0 if np.mean(excess_returns) <= 0 else float("inf")

        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(period)
        return sortino

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
            return 0

        gross_profit = sum(trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) > 0)
        gross_loss = sum(abs(trade.get("pnl", 0)) for trade in trades if trade.get("pnl", 0) < 0)

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0

        return gross_profit / gross_loss

    def _calculate_avg_trade(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average trade P&L."""
        if not trades:
            return 0

        total_pnl = sum(trade.get("pnl", 0) for trade in trades)
        return total_pnl / len(trades)

    def _calculate_avg_win_loss(self, trades: List[Dict[str, Any]]) -> tuple:
        """Calculate average win and average loss separately.

        Returns:
            Tuple of (avg_win, avg_loss) as percentages
        """
        if not trades:
            return 0.0, 0.0

        wins = [trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) > 0]
        losses = [trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) < 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Convert to percentage relative to trade size if possible
        # For now return as raw P&L values normalized
        total_pnl = sum(abs(trade.get("pnl", 0)) for trade in trades)
        if total_pnl > 0:
            avg_win = avg_win / total_pnl if avg_win else 0
            avg_loss = abs(avg_loss) / total_pnl if avg_loss else 0

        return avg_win, avg_loss

    def analyze_strategy(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a strategy's performance and provide insights.

        Args:
            backtest_result: Backtest result dictionary.

        Returns:
            Dictionary with performance metrics and insights.
        """
        # Calculate basic metrics
        metrics = self.calculate_metrics(backtest_result)

        # Generate insights based on metrics
        insights = self._generate_insights(metrics)

        return {"metrics": metrics, "insights": insights}

    def _generate_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate insights from metrics."""
        insights = []

        # Return insights
        if metrics["total_return"] <= 0:
            insights.append("Strategy is not profitable over the tested period.")
        elif metrics["total_return"] > 0 and metrics["total_return"] < 0.05:
            insights.append("Strategy shows minimal profitability. Consider optimization.")
        elif metrics["total_return"] >= 0.05:
            insights.append("Strategy shows positive returns.")

        # Risk insights
        if metrics["max_drawdown"] > 0.2:
            insights.append("High maximum drawdown indicates significant risk.")

        # Sharpe ratio insights
        if metrics["sharpe_ratio"] < 1:
            insights.append("Low Sharpe ratio indicates poor risk-adjusted returns.")
        elif metrics["sharpe_ratio"] >= 1 and metrics["sharpe_ratio"] < 2:
            insights.append("Moderate Sharpe ratio indicates acceptable risk-adjusted returns.")
        elif metrics["sharpe_ratio"] >= 2:
            insights.append("High Sharpe ratio indicates strong risk-adjusted returns.")

        # Win rate insights
        if metrics["win_rate"] < 0.4:
            insights.append("Low win rate. Consider improving entry/exit criteria.")
        elif metrics["win_rate"] >= 0.4 and metrics["win_rate"] < 0.6:
            insights.append("Moderate win rate.")
        elif metrics["win_rate"] >= 0.6:
            insights.append("High win rate. Ensure you're not overoptimizing.")

        # Profit factor insights
        if metrics["profit_factor"] < 1:
            insights.append("Profit factor below 1 indicates losses exceed profits.")
        elif metrics["profit_factor"] >= 1 and metrics["profit_factor"] < 1.5:
            insights.append("Profit factor indicates marginally profitable strategy.")
        elif metrics["profit_factor"] >= 1.5:
            insights.append("Strong profit factor indicates good profitability.")

        return insights

    def compare_strategies(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple strategies.

        Args:
            results: Dictionary mapping strategy names to backtest results.

        Returns:
            Dictionary with comparison results.
        """
        if not results:
            return {"error": "No strategy results provided"}

        # Calculate metrics for each strategy
        metrics = {}
        for name, result in results.items():
            metrics[name] = self.calculate_metrics(result)

        # Rank strategies by different metrics
        rankings = {}
        for metric in ["sharpe_ratio", "total_return", "max_drawdown", "profit_factor"]:
            if metric == "max_drawdown":
                # Lower is better for drawdown
                ranked = sorted(metrics.items(), key=lambda x: x[1][metric])
            else:
                # Higher is better for other metrics
                ranked = sorted(metrics.items(), key=lambda x: x[1][metric], reverse=True)

            rankings[metric] = [name for name, _ in ranked]

        # Calculate overall rank (average rank across metrics)
        strategy_ranks = {name: [] for name in metrics}

        for metric, ranked_names in rankings.items():
            for i, name in enumerate(ranked_names):
                strategy_ranks[name].append(i + 1)  # 1-based ranking

        # Calculate average rank
        avg_ranks = {name: sum(ranks) / len(ranks) for name, ranks in strategy_ranks.items()}

        # Sort by average rank
        overall_ranking = sorted(avg_ranks.items(), key=lambda x: x[1])

        return {"metrics": metrics, "rankings": rankings, "overall_ranking": overall_ranking}

    def calculate_significance(
        self,
        trades: List[Dict[str, Any]],
        min_trades: int = 50,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance of trading results.

        CRITICAL: Use this before trusting backtest results!

        A strategy with only 9 trades showing 42% return is NOT statistically
        significant - it could easily be luck. This method helps identify
        whether results are reliable.

        Args:
            trades: List of trade dictionaries with 'pnl' field
            min_trades: Minimum trades required for significance (default 50)
            confidence_level: Confidence level for intervals (default 0.95)

        Returns:
            Dictionary with significance analysis:
            - is_significant: Whether results are statistically significant
            - warnings: List of issues found
            - trade_count: Number of trades analyzed
            - mean_return: Mean trade return
            - t_statistic: T-test statistic
            - p_value: P-value for mean > 0 test
            - confidence_interval: CI for mean return
            - sharpe_ci: Bootstrap CI for Sharpe ratio
        """
        result = {
            "is_significant": False,
            "warnings": [],
            "trade_count": len(trades),
            "mean_return": 0,
            "std_return": 0,
            "t_statistic": 0,
            "p_value": 1.0,
            "confidence_interval": (0, 0),
            "sharpe_ci": (0, 0),
        }

        # Check minimum trade count
        if len(trades) < min_trades:
            result["warnings"].append(
                f"Insufficient trades: {len(trades)} < {min_trades} minimum. "
                "Results are NOT statistically reliable."
            )
            return result

        # Extract returns
        returns = np.array([trade.get("pnl", 0) for trade in trades])

        if len(returns) == 0 or np.all(returns == 0):
            result["warnings"].append("No valid trade returns found.")
            return result

        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # Sample std

        result["mean_return"] = mean_return
        result["std_return"] = std_return

        # T-test: Is mean return significantly different from zero?
        if std_return > 0:
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            result["t_statistic"] = t_stat
            result["p_value"] = p_value / 2  # One-tailed (we want return > 0)

            # Confidence interval for mean
            se = std_return / np.sqrt(len(returns))
            t_crit = stats.t.ppf((1 + confidence_level) / 2, len(returns) - 1)
            ci_lower = mean_return - t_crit * se
            ci_upper = mean_return + t_crit * se
            result["confidence_interval"] = (ci_lower, ci_upper)

        # Bootstrap confidence interval for Sharpe ratio
        if len(returns) >= 30:
            sharpe_ci = self._bootstrap_sharpe_ci(returns, confidence_level)
            result["sharpe_ci"] = sharpe_ci

        # Determine significance
        alpha = 1 - confidence_level
        is_significant = (
            len(trades) >= min_trades
            and result["p_value"] < alpha
            and mean_return > 0
        )
        result["is_significant"] = is_significant

        # Generate warnings
        if result["p_value"] >= alpha:
            result["warnings"].append(
                f"Returns not statistically significant (p={result['p_value']:.4f} >= {alpha}). "
                "Could be random chance."
            )

        if mean_return <= 0:
            result["warnings"].append(
                f"Mean return is not positive ({mean_return:.4f}). "
                "Strategy is not profitable on average."
            )

        # Check for high variance (unreliable results)
        if std_return > 0 and abs(mean_return / std_return) < 0.5:
            result["warnings"].append(
                f"High variance relative to mean (CV={std_return/abs(mean_return):.2f}). "
                "Results are unstable."
            )

        # Check for outlier dependency
        outlier_impact = self._check_outlier_dependency(returns)
        if outlier_impact > 0.5:
            result["warnings"].append(
                f"Results depend heavily on outliers ({outlier_impact:.0%} of profit from top 10% trades). "
                "May not be reproducible."
            )

        return result

    def _bootstrap_sharpe_ci(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for Sharpe ratio.

        Bootstrap is preferred for Sharpe because it doesn't assume
        normality of returns.

        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound) for Sharpe ratio
        """
        if len(returns) < 10:
            return (0, 0)

        sharpe_samples = []
        n = len(returns)

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(returns, size=n, replace=True)

            # Calculate Sharpe for this sample
            if np.std(sample) > 0:
                sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252)
                sharpe_samples.append(sharpe)

        if not sharpe_samples:
            return (0, 0)

        # Calculate percentiles
        alpha = (1 - confidence_level) / 2
        lower = np.percentile(sharpe_samples, alpha * 100)
        upper = np.percentile(sharpe_samples, (1 - alpha) * 100)

        return (lower, upper)

    def _check_outlier_dependency(self, returns: np.ndarray) -> float:
        """
        Check how dependent profits are on outlier trades.

        A strategy that depends on a few big wins is less reliable
        than one with consistent returns.

        Args:
            returns: Array of returns

        Returns:
            Fraction of total profit from top 10% of trades (0-1)
        """
        if len(returns) < 10:
            return 0

        # Only consider winning trades
        wins = returns[returns > 0]
        if len(wins) == 0:
            return 0

        total_profit = np.sum(wins)
        if total_profit <= 0:
            return 0

        # Top 10% of winning trades
        top_n = max(1, len(wins) // 10)
        top_profits = np.sum(np.sort(wins)[-top_n:])

        return top_profits / total_profit

    def validate_backtest_results(
        self,
        backtest_result: Dict[str, Any],
        min_trades: int = 50,
        max_overfit_ratio: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of backtest results.

        Checks for:
        1. Statistical significance
        2. Overfitting indicators
        3. Unrealistic performance

        Args:
            backtest_result: Backtest results dictionary
            min_trades: Minimum trades required
            max_overfit_ratio: Maximum in-sample/out-of-sample ratio

        Returns:
            Validation report with pass/fail status
        """
        trades = backtest_result.get("trades", [])
        metrics = self.calculate_metrics(backtest_result)
        significance = self.calculate_significance(trades, min_trades)

        validation = {
            "passed": True,
            "issues": [],
            "warnings": significance["warnings"].copy(),
            "metrics": metrics,
            "significance": significance,
        }

        # Check statistical significance
        if not significance["is_significant"]:
            validation["passed"] = False
            validation["issues"].append(
                "Results are not statistically significant. "
                "Cannot trust these numbers for live trading."
            )

        # Check for unrealistic Sharpe
        if metrics["sharpe_ratio"] > 3:
            validation["warnings"].append(
                f"Sharpe ratio of {metrics['sharpe_ratio']:.2f} is unusually high. "
                "Possible overfitting or data issue."
            )

        # Check for too few trades
        if len(trades) < min_trades:
            validation["passed"] = False
            validation["issues"].append(
                f"Only {len(trades)} trades (need {min_trades}+). "
                "Insufficient data for reliable conclusions."
            )

        # Check for extreme win rate
        if metrics["win_rate"] > 0.8:
            validation["warnings"].append(
                f"Win rate of {metrics['win_rate']:.0%} is suspiciously high. "
                "May indicate overfitting or lookahead bias."
            )

        # Check for very low drawdown (unrealistic)
        if metrics["max_drawdown"] < 0.01 and metrics["total_return"] > 0.1:
            validation["warnings"].append(
                f"Max drawdown of {metrics['max_drawdown']:.1%} with {metrics['total_return']:.1%} return "
                "is unrealistic. Check for bias in simulation."
            )

        return validation
