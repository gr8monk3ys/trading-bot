"""
Strategy Validator - Comprehensive strategy validation before live trading.

CRITICAL: Run this validation before paper trading or live trading!

This module provides a systematic validation pipeline that checks:
1. Walk-forward validation (detects overfitting)
2. Statistical significance (enough trades, reliable results)
3. In-sample vs out-of-sample ratio (performance degradation)
4. Market regime testing (works in different conditions)

Usage:
    from utils.strategy_validator import StrategyValidator

    validator = StrategyValidator()
    result = await validator.validate_strategy(
        strategy_class=MomentumStrategy,
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
    )

    if result["valid"]:
        print("Strategy passed validation - safe to paper trade")
    else:
        print(f"Strategy failed: {result['blockers']}")
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of strategy validation."""

    valid: bool
    strategy_name: str
    validation_date: datetime = field(default_factory=datetime.now)

    # Scores (0-100)
    overall_score: float = 0
    significance_score: float = 0
    overfit_score: float = 0
    robustness_score: float = 0

    # Details
    metrics: Dict[str, Any] = field(default_factory=dict)
    walk_forward_result: Dict[str, Any] = field(default_factory=dict)
    significance_result: Dict[str, Any] = field(default_factory=dict)
    regime_results: Dict[str, Any] = field(default_factory=dict)

    # Issues
    blockers: List[str] = field(default_factory=list)  # Must fix before trading
    warnings: List[str] = field(default_factory=list)  # Should investigate
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "strategy_name": self.strategy_name,
            "validation_date": self.validation_date.isoformat(),
            "overall_score": self.overall_score,
            "significance_score": self.significance_score,
            "overfit_score": self.overfit_score,
            "robustness_score": self.robustness_score,
            "metrics": self.metrics,
            "walk_forward_result": self.walk_forward_result,
            "significance_result": self.significance_result,
            "regime_results": self.regime_results,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class StrategyValidator:
    """
    Comprehensive strategy validation pipeline.

    Validates strategies through multiple tests before allowing
    paper or live trading.
    """

    # Validation thresholds
    MIN_TRADES = 50  # Minimum trades for statistical significance
    MAX_OVERFIT_RATIO = 1.5  # Max in-sample / out-of-sample ratio
    MIN_SHARPE = 0.5  # Minimum Sharpe ratio
    MAX_DRAWDOWN = 0.15  # Maximum acceptable drawdown
    MIN_WIN_RATE = 0.35  # Minimum win rate

    def __init__(
        self,
        broker=None,
        min_trades: int = 50,
        max_overfit_ratio: float = 1.5,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.15,
    ):
        """
        Initialize the validator.

        Args:
            broker: Broker instance for data fetching
            min_trades: Minimum trades required
            max_overfit_ratio: Maximum IS/OOS ratio
            min_sharpe: Minimum Sharpe ratio
            max_drawdown: Maximum acceptable drawdown
        """
        self.broker = broker
        self.min_trades = min_trades
        self.max_overfit_ratio = max_overfit_ratio
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown

    async def validate_strategy(
        self,
        strategy_class: Type,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100000,
        config: Optional[Dict] = None,
    ) -> ValidationResult:
        """
        Run full validation pipeline on a strategy.

        Args:
            strategy_class: Strategy class to validate
            symbols: Symbols to test
            start_date: Start date for validation
            end_date: End date for validation
            initial_capital: Starting capital for backtests
            config: Optional strategy configuration

        Returns:
            ValidationResult with pass/fail status and details
        """
        from engine.backtest_engine import BacktestEngine
        from engine.performance_metrics import PerformanceMetrics
        from engine.walk_forward import WalkForwardValidator

        result = ValidationResult(
            valid=False,
            strategy_name=getattr(strategy_class, "NAME", strategy_class.__name__),
        )

        logger.info(
            f"Starting validation for {result.strategy_name} "
            f"({start_date} to {end_date}, {len(symbols)} symbols)"
        )

        try:
            # Initialize engines
            engine = BacktestEngine(broker=self.broker)
            metrics_calc = PerformanceMetrics()

            # 1. Run basic backtest first
            logger.info("Step 1: Running basic backtest...")
            backtest_result = await engine.run_backtest(
                strategy_class=strategy_class,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
            )

            result.metrics = metrics_calc.calculate_metrics(backtest_result)
            trades = backtest_result.get("trades", [])

            # 2. Check statistical significance
            logger.info("Step 2: Checking statistical significance...")
            significance = metrics_calc.calculate_significance(
                trades, min_trades=self.min_trades
            )
            result.significance_result = significance
            result.significance_score = self._calculate_significance_score(significance)

            if not significance["is_significant"]:
                result.blockers.append(
                    f"Results not statistically significant: "
                    f"{len(trades)} trades (need {self.min_trades}+)"
                )

            for warning in significance["warnings"]:
                result.warnings.append(warning)

            # 3. Run walk-forward validation
            logger.info("Step 3: Running walk-forward validation...")
            try:
                wf_validator = WalkForwardValidator(engine)
                wf_result = await wf_validator.validate(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    n_splits=5,
                    train_ratio=0.7,
                )
                result.walk_forward_result = wf_result

                # Check overfit ratio
                oos_ratio = wf_result.get("oos_ratio", 0)
                if oos_ratio > self.max_overfit_ratio:
                    result.blockers.append(
                        f"Overfitting detected: OOS ratio {oos_ratio:.2f} > {self.max_overfit_ratio}"
                    )
                    result.overfit_score = max(0, 100 - (oos_ratio - 1) * 50)
                else:
                    result.overfit_score = min(100, 100 - (oos_ratio - 1) * 30)

            except Exception as e:
                logger.warning(f"Walk-forward validation failed: {e}")
                result.warnings.append(f"Walk-forward validation error: {str(e)}")
                result.overfit_score = 50  # Unknown

            # 4. Check basic metrics thresholds
            logger.info("Step 4: Checking performance thresholds...")
            self._check_metric_thresholds(result)

            # 5. Test robustness (optional - different market regimes)
            logger.info("Step 5: Testing robustness across regimes...")
            await self._test_market_regimes(
                result, engine, strategy_class, symbols, start_date, end_date
            )

            # Calculate overall score
            result.overall_score = self._calculate_overall_score(result)

            # Determine if valid
            result.valid = len(result.blockers) == 0 and result.overall_score >= 60

            # Generate recommendations
            self._generate_recommendations(result)

            logger.info(
                f"Validation complete: {'PASSED' if result.valid else 'FAILED'} "
                f"(score: {result.overall_score:.0f}/100)"
            )

        except Exception as e:
            logger.error(f"Validation failed with error: {e}", exc_info=True)
            result.blockers.append(f"Validation error: {str(e)}")

        return result

    def _calculate_significance_score(self, significance: Dict) -> float:
        """Calculate significance score (0-100)."""
        if not significance["is_significant"]:
            return 30

        score = 60  # Base score for being significant

        # Bonus for more trades
        trade_count = significance["trade_count"]
        if trade_count >= 100:
            score += 20
        elif trade_count >= 75:
            score += 15
        elif trade_count >= 50:
            score += 10

        # Bonus for lower p-value
        p_value = significance["p_value"]
        if p_value < 0.01:
            score += 20
        elif p_value < 0.05:
            score += 10

        return min(100, score)

    def _check_metric_thresholds(self, result: ValidationResult):
        """Check if metrics meet minimum thresholds."""
        metrics = result.metrics

        # Sharpe ratio
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe < self.min_sharpe:
            result.blockers.append(
                f"Sharpe ratio {sharpe:.2f} below minimum {self.min_sharpe}"
            )

        # Max drawdown
        max_dd = metrics.get("max_drawdown", 1)
        if max_dd > self.max_drawdown:
            result.blockers.append(
                f"Max drawdown {max_dd:.1%} exceeds maximum {self.max_drawdown:.1%}"
            )

        # Win rate (warning only)
        win_rate = metrics.get("win_rate", 0)
        if win_rate < self.MIN_WIN_RATE:
            result.warnings.append(
                f"Low win rate: {win_rate:.1%} (minimum recommended: {self.MIN_WIN_RATE:.1%})"
            )

        # Total return (must be positive)
        total_return = metrics.get("total_return", 0)
        if total_return <= 0:
            result.blockers.append(
                f"Strategy is not profitable: {total_return:.1%} return"
            )

        # Profit factor
        profit_factor = metrics.get("profit_factor", 0)
        if profit_factor < 1.0:
            result.blockers.append(
                f"Profit factor {profit_factor:.2f} < 1.0 (losses exceed profits)"
            )

    async def _test_market_regimes(
        self,
        result: ValidationResult,
        engine,
        strategy_class: Type,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ):
        """Test strategy across different market regimes."""
        from engine.performance_metrics import PerformanceMetrics

        metrics_calc = PerformanceMetrics()

        # Split period into halves to test different conditions
        mid_date = start_date + (end_date - start_date) / 2

        regime_results = {}
        regime_scores = []

        periods = [
            ("first_half", start_date, mid_date),
            ("second_half", mid_date, end_date),
        ]

        for period_name, p_start, p_end in periods:
            try:
                bt_result = await engine.run_backtest(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=p_start,
                    end_date=p_end,
                    initial_capital=100000,
                )

                period_metrics = metrics_calc.calculate_metrics(bt_result)
                regime_results[period_name] = {
                    "return": period_metrics.get("total_return", 0),
                    "sharpe": period_metrics.get("sharpe_ratio", 0),
                    "drawdown": period_metrics.get("max_drawdown", 0),
                    "trades": period_metrics.get("trade_count", 0),
                }

                # Score this period
                if period_metrics.get("total_return", 0) > 0:
                    regime_scores.append(70)
                else:
                    regime_scores.append(30)

            except Exception as e:
                logger.warning(f"Regime test failed for {period_name}: {e}")
                regime_results[period_name] = {"error": str(e)}
                regime_scores.append(50)

        result.regime_results = regime_results

        # Check for consistency
        if regime_results:
            returns = [
                r.get("return", 0)
                for r in regime_results.values()
                if isinstance(r, dict) and "return" in r
            ]
            if len(returns) >= 2:
                # Check if performance is consistent
                if all(r > 0 for r in returns):
                    result.robustness_score = 80
                elif any(r > 0 for r in returns):
                    result.robustness_score = 60
                    result.warnings.append(
                        "Strategy is not profitable in all tested periods"
                    )
                else:
                    result.robustness_score = 30
                    result.blockers.append(
                        "Strategy unprofitable in all tested periods"
                    )
            else:
                result.robustness_score = 50

        result.robustness_score = np.mean(regime_scores) if regime_scores else 50

    def _calculate_overall_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score."""
        # Weighted average of component scores
        weights = {
            "significance": 0.30,
            "overfit": 0.30,
            "robustness": 0.20,
            "metrics": 0.20,
        }

        # Calculate metrics score
        metrics = result.metrics
        metrics_score = 50  # Base

        if metrics.get("sharpe_ratio", 0) >= 1.0:
            metrics_score += 20
        if metrics.get("max_drawdown", 1) < 0.10:
            metrics_score += 15
        if metrics.get("profit_factor", 0) >= 1.5:
            metrics_score += 15

        scores = {
            "significance": result.significance_score,
            "overfit": result.overfit_score,
            "robustness": result.robustness_score,
            "metrics": min(100, metrics_score),
        }

        overall = sum(scores[k] * weights[k] for k in weights)

        # Penalty for blockers
        if result.blockers:
            overall = min(overall, 50)  # Cap at 50 if any blockers

        return overall

    def _generate_recommendations(self, result: ValidationResult):
        """Generate recommendations based on validation results."""
        if result.valid:
            result.recommendations.append(
                "Strategy passed validation. Safe to proceed with paper trading."
            )
            result.recommendations.append(
                "Monitor paper trading for 30+ days before live trading."
            )
        else:
            if len(result.significance_result.get("warnings", [])) > 0:
                result.recommendations.append(
                    "Collect more trades before trusting results. "
                    f"Current: {result.significance_result.get('trade_count', 0)}, "
                    f"Need: {self.min_trades}+"
                )

            if result.overfit_score < 60:
                result.recommendations.append(
                    "Reduce strategy parameters to avoid overfitting. "
                    "Consider using walk-forward optimization."
                )

            if result.metrics.get("sharpe_ratio", 0) < self.min_sharpe:
                result.recommendations.append(
                    "Improve risk-adjusted returns. Consider tighter stops or "
                    "better entry signals."
                )

    async def quick_validate(
        self,
        trades: List[Dict],
        equity_curve: List[float],
    ) -> Dict[str, Any]:
        """
        Quick validation of existing backtest results.

        Use this when you already have backtest results and just want
        to check significance.

        Args:
            trades: List of trade dictionaries
            equity_curve: List of equity values

        Returns:
            Quick validation result
        """
        from engine.performance_metrics import PerformanceMetrics

        metrics_calc = PerformanceMetrics()

        # Create pseudo backtest result
        backtest_result = {
            "trades": trades,
            "equity_curve": equity_curve,
        }

        validation = metrics_calc.validate_backtest_results(
            backtest_result,
            min_trades=self.min_trades,
            max_overfit_ratio=self.max_overfit_ratio,
        )

        return validation


async def validate_before_live(
    strategy_class: Type,
    symbols: List[str],
    broker=None,
) -> bool:
    """
    Convenience function to validate a strategy before live trading.

    Runs validation with sensible defaults and returns simple pass/fail.

    Args:
        strategy_class: Strategy class to validate
        symbols: Symbols to test
        broker: Broker instance (optional)

    Returns:
        True if strategy passed validation, False otherwise
    """
    from datetime import date

    validator = StrategyValidator(broker=broker)

    # Use last year of data
    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    result = await validator.validate_strategy(
        strategy_class=strategy_class,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    if not result.valid:
        logger.warning(
            f"Strategy {result.strategy_name} FAILED validation:\n"
            + "\n".join(f"  - {b}" for b in result.blockers)
        )

    return result.valid
