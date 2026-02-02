"""
Validated Backtest Runner - Integrates walk-forward validation with backtesting

Automatically runs walk-forward validation alongside backtests to detect overfitting
and provide regime-stratified performance metrics.

Usage:
    from engine.validated_backtest import ValidatedBacktestRunner

    runner = ValidatedBacktestRunner(broker)
    result = await runner.run_validated_backtest(
        strategy_class=MomentumStrategy,
        symbols=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    if result.overfit_warning:
        print("Strategy may be overfit!")
    print(f"Regime performance: {result.regime_metrics}")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from engine.walk_forward import WalkForwardValidator, WalkForwardResult
from engine.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class RegimePerformance:
    """Performance metrics for a specific market regime."""

    regime: str
    num_days: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int


@dataclass
class ValidatedBacktestResult:
    """Result of a validated backtest with walk-forward and regime analysis."""

    # Basic backtest results
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float

    # Walk-forward validation
    walk_forward_validated: bool
    overfit_warning: bool
    overfit_ratio: float
    is_return: float  # In-sample average return
    oos_return: float  # Out-of-sample average return
    consistency_score: float  # % of folds where OOS > 0
    walk_forward_folds: List[WalkForwardResult] = field(default_factory=list)

    # Regime-stratified performance
    regime_metrics: Dict[str, RegimePerformance] = field(default_factory=dict)

    # Statistical significance
    statistically_significant: bool = False
    p_value: Optional[float] = None

    # Raw data
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None


class ValidatedBacktestRunner:
    """
    Backtest runner with integrated walk-forward validation and regime analysis.

    Automatically detects overfitting and provides regime-stratified metrics.
    """

    def __init__(
        self,
        broker,
        walk_forward_enabled: bool = True,
        regime_analysis_enabled: bool = True,
        n_splits: int = 5,
        overfit_threshold: float = 1.5,
    ):
        """
        Initialize validated backtest runner.

        Args:
            broker: Trading broker instance
            walk_forward_enabled: Run walk-forward validation
            regime_analysis_enabled: Calculate regime-stratified metrics
            n_splits: Number of walk-forward folds
            overfit_threshold: IS/OOS ratio above this triggers warning
        """
        self.broker = broker
        self.walk_forward_enabled = walk_forward_enabled
        self.regime_analysis_enabled = regime_analysis_enabled
        self.n_splits = n_splits
        self.overfit_threshold = overfit_threshold

    async def run_validated_backtest(
        self,
        strategy_class: Type,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        **strategy_kwargs,
    ) -> ValidatedBacktestResult:
        """
        Run backtest with validation.

        Args:
            strategy_class: Strategy class to backtest
            symbols: List of symbols to trade
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital
            **strategy_kwargs: Additional strategy parameters

        Returns:
            ValidatedBacktestResult with all metrics
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        logger.info(
            f"Running validated backtest: {strategy_class.__name__} "
            f"from {start_date} to {end_date}"
        )

        # 1. Run main backtest
        backtest_result = await self._run_backtest(
            strategy_class, symbols, start_dt, end_dt, initial_capital, **strategy_kwargs
        )

        # 2. Run walk-forward validation
        wf_result = None
        if self.walk_forward_enabled:
            wf_result = await self._run_walk_forward(
                strategy_class, symbols, start_dt, end_dt, initial_capital, **strategy_kwargs
            )

        # 3. Calculate regime-stratified metrics
        regime_metrics = {}
        if self.regime_analysis_enabled and backtest_result.get("equity_curve") is not None:
            regime_metrics = await self._calculate_regime_metrics(
                backtest_result["equity_curve"],
                backtest_result["daily_returns"],
                start_dt,
                end_dt,
            )

        # 4. Calculate statistical significance
        stat_sig = False
        p_value = None
        if backtest_result.get("daily_returns") is not None:
            stat_sig, p_value = self._calculate_significance(
                backtest_result["daily_returns"]
            )

        # Build result
        result = ValidatedBacktestResult(
            strategy_name=strategy_class.__name__,
            start_date=start_dt,
            end_date=end_dt,
            total_return=backtest_result.get("total_return", 0),
            sharpe_ratio=backtest_result.get("sharpe_ratio", 0),
            max_drawdown=backtest_result.get("max_drawdown", 0),
            num_trades=backtest_result.get("num_trades", 0),
            win_rate=backtest_result.get("win_rate", 0),
            walk_forward_validated=wf_result is not None,
            overfit_warning=wf_result["overfit_warning"] if wf_result else False,
            overfit_ratio=wf_result["overfit_ratio"] if wf_result else 1.0,
            is_return=wf_result["is_return"] if wf_result else 0,
            oos_return=wf_result["oos_return"] if wf_result else 0,
            consistency_score=wf_result["consistency"] if wf_result else 0,
            walk_forward_folds=wf_result["folds"] if wf_result else [],
            regime_metrics=regime_metrics,
            statistically_significant=stat_sig,
            p_value=p_value,
            equity_curve=backtest_result.get("equity_curve"),
            daily_returns=backtest_result.get("daily_returns"),
        )

        # Log summary
        self._log_result_summary(result)

        return result

    async def _run_backtest(
        self,
        strategy_class: Type,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        **strategy_kwargs,
    ) -> Dict[str, Any]:
        """Run the main backtest."""
        try:
            from engine.backtest_engine import BacktestEngine
            from brokers.backtest_broker import BacktestBroker

            # Create backtest broker
            backtest_broker = BacktestBroker(
                data_broker=self.broker,
                initial_capital=initial_capital,
            )

            # Initialize strategy
            strategy = strategy_class(broker=backtest_broker, **strategy_kwargs)

            # Create engine and run
            engine = BacktestEngine(broker=backtest_broker)
            results = await engine.run([strategy], start_date, end_date)

            if results and len(results) > 0:
                result_df = results[0]

                # Extract metrics
                equity_curve = result_df["equity"].dropna()
                daily_returns = result_df["returns"].dropna()

                total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1 if len(equity_curve) > 0 else 0
                sharpe = self._calculate_sharpe(daily_returns)
                max_dd = result_df["drawdown"].min() if "drawdown" in result_df else 0

                return {
                    "total_return": total_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "num_trades": int(result_df["trades"].sum()),
                    "win_rate": 0.5,  # Would need trade-level data
                    "equity_curve": equity_curve,
                    "daily_returns": daily_returns,
                }

        except Exception as e:
            logger.error(f"Backtest failed: {e}")

        return {}

    async def _run_walk_forward(
        self,
        strategy_class: Type,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        **strategy_kwargs,
    ) -> Dict[str, Any]:
        """Run walk-forward validation."""
        try:
            validator = WalkForwardValidator(n_splits=self.n_splits)

            # Create time splits
            splits = validator.create_time_splits(start_date, end_date)

            is_returns = []
            oos_returns = []
            folds = []

            for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
                logger.debug(f"Walk-forward fold {i+1}/{len(splits)}")

                # Run backtest on training period
                is_result = await self._run_backtest(
                    strategy_class, symbols, train_start, train_end,
                    initial_capital, **strategy_kwargs
                )

                # Run backtest on test period
                oos_result = await self._run_backtest(
                    strategy_class, symbols, test_start, test_end,
                    initial_capital, **strategy_kwargs
                )

                is_ret = is_result.get("total_return", 0)
                oos_ret = oos_result.get("total_return", 0)

                is_returns.append(is_ret)
                oos_returns.append(oos_ret)

                # Create fold result
                fold = WalkForwardResult(
                    fold_num=i + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    is_return=is_ret,
                    is_sharpe=is_result.get("sharpe_ratio", 0),
                    is_trades=is_result.get("num_trades", 0),
                    is_win_rate=is_result.get("win_rate", 0),
                    oos_return=oos_ret,
                    oos_sharpe=oos_result.get("sharpe_ratio", 0),
                    oos_trades=oos_result.get("num_trades", 0),
                    oos_win_rate=oos_result.get("win_rate", 0),
                    overfitting_ratio=is_ret / oos_ret if oos_ret != 0 else float("inf"),
                    degradation=is_ret - oos_ret,
                )
                folds.append(fold)

            # Calculate aggregates
            avg_is = np.mean(is_returns) if is_returns else 0
            avg_oos = np.mean(oos_returns) if oos_returns else 0

            if avg_oos != 0:
                overfit_ratio = avg_is / avg_oos
            elif avg_is > 0:
                overfit_ratio = float("inf")
            else:
                overfit_ratio = 1.0

            consistency = sum(1 for r in oos_returns if r > 0) / len(oos_returns) if oos_returns else 0

            overfit_warning = overfit_ratio > self.overfit_threshold or consistency < 0.5

            if overfit_warning:
                logger.warning(
                    f"⚠️ OVERFITTING DETECTED: IS/OOS ratio = {overfit_ratio:.2f}, "
                    f"consistency = {consistency:.0%}"
                )

            return {
                "overfit_warning": overfit_warning,
                "overfit_ratio": overfit_ratio,
                "is_return": avg_is,
                "oos_return": avg_oos,
                "consistency": consistency,
                "folds": folds,
            }

        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")
            return None

    async def _calculate_regime_metrics(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, RegimePerformance]:
        """Calculate performance by market regime."""
        try:
            from utils.market_regime import MarketRegimeDetector

            detector = MarketRegimeDetector(self.broker)

            # Group returns by regime
            regime_returns: Dict[str, List[float]] = {
                "BULL": [],
                "BEAR": [],
                "SIDEWAYS": [],
                "VOLATILE": [],
            }

            for date, ret in daily_returns.items():
                if pd.isna(ret):
                    continue

                # Detect regime for this date
                try:
                    regime_info = await detector.detect_regime(date)
                    regime = regime_info.get("regime", "UNKNOWN") if isinstance(regime_info, dict) else "UNKNOWN"
                except Exception:
                    regime = "UNKNOWN"

                if regime in regime_returns:
                    regime_returns[regime].append(ret)

            # Calculate metrics per regime
            regime_metrics = {}
            for regime, returns in regime_returns.items():
                if not returns:
                    continue

                returns_arr = np.array(returns)
                total_ret = np.prod(1 + returns_arr) - 1
                sharpe = self._calculate_sharpe(pd.Series(returns_arr))

                # Calculate drawdown for this regime
                cumulative = np.cumprod(1 + returns_arr)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0

                regime_metrics[regime] = RegimePerformance(
                    regime=regime,
                    num_days=len(returns),
                    total_return=total_ret,
                    sharpe_ratio=sharpe,
                    max_drawdown=max_dd,
                    win_rate=sum(1 for r in returns if r > 0) / len(returns),
                    num_trades=0,  # Would need trade-level data
                )

            return regime_metrics

        except ImportError:
            logger.debug("Market regime detector not available")
            return {}
        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return {}

    def _calculate_sharpe(self, returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0

        returns = returns.dropna()
        if len(returns) < 2:
            return 0

        excess_returns = returns - rf_rate / 252
        if returns.std() == 0:
            return 0

        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_significance(
        self, daily_returns: pd.Series
    ) -> tuple[bool, Optional[float]]:
        """Calculate statistical significance of returns."""
        try:
            from scipy import stats

            returns = daily_returns.dropna()
            if len(returns) < 30:
                return False, None

            # T-test: Are returns significantly different from zero?
            t_stat, p_value = stats.ttest_1samp(returns, 0)

            return p_value < 0.05, p_value

        except ImportError:
            return False, None
        except Exception:
            return False, None

    def _log_result_summary(self, result: ValidatedBacktestResult):
        """Log a summary of backtest results."""
        logger.info("=" * 60)
        logger.info(f"VALIDATED BACKTEST RESULTS: {result.strategy_name}")
        logger.info("=" * 60)
        logger.info(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        logger.info(f"Total Return: {result.total_return:.2%}")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")

        if result.walk_forward_validated:
            logger.info("-" * 40)
            logger.info("WALK-FORWARD VALIDATION:")
            logger.info(f"  In-Sample Return: {result.is_return:.2%}")
            logger.info(f"  Out-of-Sample Return: {result.oos_return:.2%}")
            logger.info(f"  Overfitting Ratio: {result.overfit_ratio:.2f}")
            logger.info(f"  Consistency Score: {result.consistency_score:.0%}")
            if result.overfit_warning:
                logger.warning("  ⚠️ OVERFITTING WARNING: Strategy may be overfit!")
            else:
                logger.info("  ✓ Walk-forward validation passed")

        if result.regime_metrics:
            logger.info("-" * 40)
            logger.info("REGIME PERFORMANCE:")
            for regime, perf in result.regime_metrics.items():
                logger.info(
                    f"  {regime}: {perf.total_return:.2%} return, "
                    f"{perf.sharpe_ratio:.2f} Sharpe ({perf.num_days} days)"
                )

        if result.statistically_significant:
            logger.info(f"  ✓ Statistically significant (p={result.p_value:.4f})")

        logger.info("=" * 60)
