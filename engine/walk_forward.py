"""
Walk-Forward Validation Engine

Implements walk-forward optimization to detect overfitting and validate
strategy performance on out-of-sample data.

Key Concepts:
- Train/Test Splits: Strategy is trained on one period, tested on another
- Rolling Windows: Multiple train/test periods to capture different market conditions
- Overfitting Detection: Compares in-sample vs out-of-sample performance

Industry standard: If OOS performance < 50% of IS performance, strategy is overfit.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from config import BACKTEST_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""
    fold_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # In-sample (training) metrics
    is_return: float
    is_sharpe: float
    is_trades: int
    is_win_rate: float

    # Out-of-sample (testing) metrics
    oos_return: float
    oos_sharpe: float
    oos_trades: int
    oos_win_rate: float

    # Comparison metrics
    overfitting_ratio: float  # IS return / OOS return
    degradation: float  # How much worse OOS is compared to IS


class WalkForwardValidator:
    """
    Validates trading strategies using walk-forward optimization.

    Walk-forward optimization splits data into training and testing periods:
    1. Train strategy parameters on training data
    2. Test strategy on out-of-sample test data
    3. Roll forward and repeat

    This detects overfitting by comparing in-sample vs out-of-sample performance.
    """

    def __init__(
        self,
        train_ratio: float = None,
        n_splits: int = None,
        min_train_days: int = None,
        gap_days: int = 0
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_ratio: Ratio of data for training (default 0.7)
            n_splits: Number of walk-forward folds (default 5)
            min_train_days: Minimum days required for training (default 30)
            gap_days: Gap between train and test to prevent lookahead (default 0)
        """
        self.train_ratio = train_ratio or BACKTEST_PARAMS.get("TRAIN_RATIO", 0.7)
        self.n_splits = n_splits or BACKTEST_PARAMS.get("N_SPLITS", 5)
        self.min_train_days = min_train_days or BACKTEST_PARAMS.get("MIN_TRAIN_DAYS", 30)
        self.gap_days = gap_days
        self.overfitting_threshold = BACKTEST_PARAMS.get("OVERFITTING_RATIO_THRESHOLD", 2.0)

        self.results: List[WalkForwardResult] = []

    def create_time_splits(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Create time-based train/test splits.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        total_days = (end_date - start_date).days

        if total_days < self.min_train_days * 2:
            raise ValueError(
                f"Date range too short for walk-forward validation. "
                f"Need at least {self.min_train_days * 2} days, got {total_days}"
            )

        splits = []
        fold_size = total_days // self.n_splits

        for i in range(self.n_splits):
            # Each fold uses expanding window for training
            fold_start = start_date
            fold_train_end = start_date + timedelta(days=int(fold_size * (i + 1) * self.train_ratio))

            # Test period starts after gap
            test_start = fold_train_end + timedelta(days=self.gap_days)
            test_end = min(
                test_start + timedelta(days=int(fold_size * (1 - self.train_ratio))),
                end_date
            )

            # Skip if test period is too short
            if (test_end - test_start).days < 5:
                continue

            splits.append((fold_start, fold_train_end, test_start, test_end))

        return splits

    async def validate(
        self,
        backtest_fn,
        symbols: List[str],
        start_date_str: str,
        end_date_str: str,
        **backtest_kwargs
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation on a backtest function.

        Args:
            backtest_fn: Async function that runs backtest
                        Must accept (symbols, start_date_str, end_date_str, **kwargs)
            symbols: List of symbols to trade
            start_date_str: Start date (YYYY-MM-DD)
            end_date_str: End date (YYYY-MM-DD)
            **backtest_kwargs: Additional arguments for backtest function

        Returns:
            Dictionary with aggregated validation results
        """
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        splits = self.create_time_splits(start_date, end_date)

        if not splits:
            raise ValueError("Could not create valid train/test splits")

        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION")
        print("="*80)
        print(f"Total period: {start_date_str} to {end_date_str}")
        print(f"Number of folds: {len(splits)}")
        print(f"Train/Test ratio: {self.train_ratio:.0%}/{1-self.train_ratio:.0%}")
        print("="*80 + "\n")

        self.results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            print(f"\n--- Fold {i+1}/{len(splits)} ---")
            print(f"Train: {train_start.date()} to {train_end.date()}")
            print(f"Test:  {test_start.date()} to {test_end.date()}")

            # Run in-sample (training) backtest
            print("  Running in-sample backtest...")
            is_result = await backtest_fn(
                symbols,
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                **backtest_kwargs
            )

            # Run out-of-sample (testing) backtest
            print("  Running out-of-sample backtest...")
            oos_result = await backtest_fn(
                symbols,
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
                **backtest_kwargs
            )

            # Calculate comparison metrics
            is_return = is_result.get('total_return', 0)
            oos_return = oos_result.get('total_return', 0)

            # P1 Fix: Overfitting ratio with proper handling of edge cases
            # Ratio > 2 suggests overfitting
            # Only calculate meaningful ratio when both returns are positive
            if is_return > 0 and oos_return > 0:
                overfitting_ratio = is_return / oos_return
            elif is_return > 0 and oos_return <= 0:
                # Positive IS, negative/zero OOS = severe overfitting
                overfitting_ratio = self.overfitting_threshold * 2  # Mark as overfit, not inf
            elif is_return <= 0 and oos_return > 0:
                # Negative IS, positive OOS = unusual but OK
                overfitting_ratio = 0.5  # Not overfit
            else:
                # Both negative or zero = neutral
                overfitting_ratio = 1.0

            # P2 Fix: Degradation - meaningful only when IS is positive
            if is_return > 0:
                degradation = (is_return - oos_return) / is_return
            elif is_return < 0 and oos_return < 0:
                # Both negative: less loss in OOS is actually good
                degradation = (abs(is_return) - abs(oos_return)) / abs(is_return)
            else:
                degradation = 0

            result = WalkForwardResult(
                fold_num=i+1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                is_return=is_return,
                is_sharpe=is_result.get('sharpe_ratio', 0),
                is_trades=is_result.get('num_trades', 0),
                is_win_rate=is_result.get('win_rate', 0),
                oos_return=oos_return,
                oos_sharpe=oos_result.get('sharpe_ratio', 0),
                oos_trades=oos_result.get('num_trades', 0),
                oos_win_rate=oos_result.get('win_rate', 0),
                overfitting_ratio=overfitting_ratio,
                degradation=degradation
            )

            self.results.append(result)

            print(f"  IS Return: {is_return:+.2%} | OOS Return: {oos_return:+.2%}")
            print(f"  Degradation: {degradation:.1%} | Overfit Ratio: {overfitting_ratio:.2f}")

        return self._aggregate_results()

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all folds."""
        if not self.results:
            return {}

        # Calculate averages
        avg_is_return = np.mean([r.is_return for r in self.results])
        avg_oos_return = np.mean([r.oos_return for r in self.results])
        avg_is_sharpe = np.mean([r.is_sharpe for r in self.results])
        avg_oos_sharpe = np.mean([r.oos_sharpe for r in self.results])
        avg_degradation = np.mean([r.degradation for r in self.results])

        # P1 Fix: Handle empty list and NaN values for overfitting ratio
        valid_ratios = [r.overfitting_ratio for r in self.results
                       if r.overfitting_ratio != float('inf') and not np.isnan(r.overfitting_ratio)]
        avg_overfit_ratio = np.mean(valid_ratios) if valid_ratios else self.overfitting_threshold

        # Count how many folds show overfitting
        overfit_folds = sum(1 for r in self.results if r.overfitting_ratio > self.overfitting_threshold)

        # Calculate consistency (OOS positive in what % of folds)
        oos_positive_folds = sum(1 for r in self.results if r.oos_return > 0)
        consistency = oos_positive_folds / len(self.results) if self.results else 0

        # Total trades
        total_oos_trades = sum(r.oos_trades for r in self.results)

        # Determine if strategy passes validation
        passes_validation = (
            avg_oos_return > 0 and
            avg_overfit_ratio < self.overfitting_threshold and
            consistency >= 0.5
        )

        # Print summary
        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*80)

        print("\nIn-Sample Performance (Training):")
        print(f"  Average Return:    {avg_is_return:+.2%}")
        print(f"  Average Sharpe:    {avg_is_sharpe:.2f}")

        print("\nOut-of-Sample Performance (Testing):")
        print(f"  Average Return:    {avg_oos_return:+.2%}")
        print(f"  Average Sharpe:    {avg_oos_sharpe:.2f}")
        print(f"  Total Trades:      {total_oos_trades}")
        print(f"  Consistency:       {consistency:.0%} of folds profitable")

        print("\nOverfitting Analysis:")
        print(f"  Average Degradation:  {avg_degradation:.1%}")
        print(f"  Average Overfit Ratio: {avg_overfit_ratio:.2f}")
        print(f"  Overfit Folds:        {overfit_folds}/{len(self.results)}")

        print("\n" + "="*80)
        if passes_validation:
            print("✅ VALIDATION PASSED")
            print("   Strategy shows consistent out-of-sample performance")
        else:
            print("❌ VALIDATION FAILED")
            if avg_oos_return <= 0:
                print("   - Out-of-sample returns are negative")
            if avg_overfit_ratio >= self.overfitting_threshold:
                print(f"   - Overfitting detected (ratio {avg_overfit_ratio:.2f} > {self.overfitting_threshold})")
            if consistency < 0.5:
                print(f"   - Inconsistent results ({consistency:.0%} folds profitable)")

        print("="*80 + "\n")

        return {
            'passes_validation': passes_validation,
            'n_folds': len(self.results),

            # In-sample metrics
            'is_avg_return': avg_is_return,
            'is_avg_sharpe': avg_is_sharpe,

            # Out-of-sample metrics
            'oos_avg_return': avg_oos_return,
            'oos_avg_sharpe': avg_oos_sharpe,
            'oos_total_trades': total_oos_trades,
            'oos_consistency': consistency,

            # Overfitting metrics
            'avg_degradation': avg_degradation,
            'avg_overfit_ratio': avg_overfit_ratio,
            'overfit_folds': overfit_folds,

            # Raw results
            'fold_results': self.results
        }


async def run_walk_forward_validation(
    symbols: List[str],
    start_date: str,
    end_date: str,
    **kwargs
):
    """
    Convenience function to run walk-forward validation.

    Usage:
        result = await run_walk_forward_validation(
            symbols=['AAPL', 'MSFT'],
            start_date='2024-01-01',
            end_date='2024-12-01'
        )

        if result['passes_validation']:
            print("Strategy is robust!")
    """
    from simple_backtest import simple_backtest

    validator = WalkForwardValidator()
    result = await validator.validate(
        simple_backtest,
        symbols,
        start_date,
        end_date,
        **kwargs
    )

    return result


if __name__ == "__main__":
    # Example usage
    async def main():
        result = await run_walk_forward_validation(
            symbols=['AAPL', 'MSFT', 'NVDA'],
            start_date='2024-01-01',
            end_date='2024-11-01'
        )

        print(f"Validation passed: {result['passes_validation']}")
        print(f"OOS Return: {result['oos_avg_return']:+.2%}")
        print(f"Overfit Ratio: {result['avg_overfit_ratio']:.2f}")

    asyncio.run(main())
