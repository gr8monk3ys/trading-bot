"""
Walk-Forward Validation Engine

Implements walk-forward optimization to detect overfitting and validate
strategy performance on out-of-sample data.

Key Concepts:
- Train/Test Splits: Strategy is trained on one period, tested on another
- Rolling Windows: Multiple train/test periods to capture different market conditions
- Overfitting Detection: Compares in-sample vs out-of-sample performance
- Statistical Significance: Wilcoxon test for IS vs OOS degradation

Industry standard: If OOS performance < 50% of IS performance, strategy is overfit.

INSTITUTIONAL STANDARD: Degradation should be tested for statistical significance,
not just compared against arbitrary thresholds.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from config import BACKTEST_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class DegradationSignificanceResult:
    """Result of statistical significance test for IS→OOS degradation."""

    is_returns: List[float]
    oos_returns: List[float]

    # Wilcoxon signed-rank test (paired test for IS > OOS)
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    degradation_significant: bool  # True if IS significantly > OOS

    # Bootstrap confidence interval for mean degradation
    mean_degradation: float
    degradation_ci_lower: float  # 95% CI lower bound
    degradation_ci_upper: float  # 95% CI upper bound

    # Effect size (rank-biserial correlation for Wilcoxon)
    effect_size: float
    effect_magnitude: str  # 'negligible', 'small', 'medium', 'large'

    interpretation: str


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


def check_degradation_significance(
    is_returns: List[float],
    oos_returns: List[float],
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> DegradationSignificanceResult:
    """
    Test if IS→OOS degradation is statistically significant.

    Uses Wilcoxon signed-rank test (paired, non-parametric) to test whether
    in-sample returns are significantly greater than out-of-sample returns.

    INSTITUTIONAL STANDARD: Don't just use arbitrary thresholds like "2x".
    Use statistical tests to determine if degradation is real or noise.

    The Wilcoxon signed-rank test is appropriate because:
    1. Paired data (same folds, different conditions)
    2. Non-parametric (doesn't assume normal distribution)
    3. Robust to outliers (common in trading returns)

    Args:
        is_returns: In-sample returns from each fold
        oos_returns: Out-of-sample returns from each fold (same length)
        alpha: Significance threshold (default 0.05)
        n_bootstrap: Number of bootstrap samples for CI (default 1000)
        random_state: Random seed for reproducibility

    Returns:
        DegradationSignificanceResult with test statistics and interpretation
    """
    n = len(is_returns)

    if n < 5:
        return DegradationSignificanceResult(
            is_returns=list(is_returns),
            oos_returns=list(oos_returns),
            wilcoxon_statistic=0.0,
            wilcoxon_p_value=1.0,
            degradation_significant=False,
            mean_degradation=0.0,
            degradation_ci_lower=0.0,
            degradation_ci_upper=0.0,
            effect_size=0.0,
            effect_magnitude="unknown",
            interpretation="Insufficient folds for statistical test (need >= 5)",
        )

    if len(is_returns) != len(oos_returns):
        raise ValueError("is_returns and oos_returns must have same length")

    is_arr = np.array(is_returns)
    oos_arr = np.array(oos_returns)

    # Calculate paired differences
    differences = is_arr - oos_arr

    # If all differences are zero, can't perform test
    if np.all(differences == 0):
        return DegradationSignificanceResult(
            is_returns=list(is_returns),
            oos_returns=list(oos_returns),
            wilcoxon_statistic=0.0,
            wilcoxon_p_value=1.0,
            degradation_significant=False,
            mean_degradation=0.0,
            degradation_ci_lower=0.0,
            degradation_ci_upper=0.0,
            effect_size=0.0,
            effect_magnitude="negligible",
            interpretation="No difference between IS and OOS returns",
        )

    # Wilcoxon signed-rank test
    # alternative='greater' tests if IS > OOS (degradation exists)
    try:
        stat, p_value = stats.wilcoxon(is_arr, oos_arr, alternative="greater", zero_method="wilcox")
    except ValueError:
        # Can happen if too few non-zero differences
        stat, p_value = 0.0, 1.0

    degradation_significant = p_value < alpha

    # Calculate mean degradation and bootstrap CI
    mean_degradation = np.mean(differences)

    rng = np.random.default_rng(random_state)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_diff = differences[indices]
        bootstrap_means.append(np.mean(bootstrap_diff))

    bootstrap_means = np.array(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    # Calculate effect size (rank-biserial correlation for Wilcoxon)
    # r = 1 - (2W / (n(n+1)/2)) where W is the test statistic
    n_pairs = n
    max_stat = n_pairs * (n_pairs + 1) / 2
    effect_size = 1 - (2 * stat / max_stat) if max_stat > 0 else 0

    # Interpret effect size magnitude
    abs_effect = abs(effect_size)
    if abs_effect < 0.1:
        effect_magnitude = "negligible"
    elif abs_effect < 0.3:
        effect_magnitude = "small"
    elif abs_effect < 0.5:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"

    # Build interpretation
    if degradation_significant:
        interpretation = (
            f"SIGNIFICANT DEGRADATION DETECTED (p={p_value:.4f} < {alpha}). "
            f"In-sample returns are significantly higher than out-of-sample. "
            f"Mean degradation: {mean_degradation:.2%} (95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]). "
            f"Effect size: {effect_size:.3f} ({effect_magnitude}). "
            f"This suggests the strategy is OVERFIT."
        )
    else:
        interpretation = (
            f"No significant degradation detected (p={p_value:.4f} >= {alpha}). "
            f"Mean degradation: {mean_degradation:.2%} (95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]). "
            f"Effect size: {effect_size:.3f} ({effect_magnitude}). "
            f"IS→OOS difference may be due to chance variation."
        )

    return DegradationSignificanceResult(
        is_returns=list(is_returns),
        oos_returns=list(oos_returns),
        wilcoxon_statistic=stat,
        wilcoxon_p_value=p_value,
        degradation_significant=degradation_significant,
        mean_degradation=mean_degradation,
        degradation_ci_lower=ci_lower,
        degradation_ci_upper=ci_upper,
        effect_size=effect_size,
        effect_magnitude=effect_magnitude,
        interpretation=interpretation,
    )


def calculate_sharpe_confidence_interval(
    returns: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Calculate Sharpe ratio with bootstrap confidence interval.

    INSTITUTIONAL STANDARD: Point estimates of Sharpe ratio are misleading.
    Always report confidence intervals.

    Args:
        returns: Array of returns
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Tuple of (sharpe_ratio, ci_lower, ci_upper)
    """
    if len(returns) < 10:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        return sharpe, sharpe - 1, sharpe + 1  # Wide CI for small samples

    rng = np.random.default_rng(random_state)
    n = len(returns)

    # Calculate observed Sharpe
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    # Bootstrap
    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_returns = returns[indices]
        boot_std = np.std(boot_returns)
        if boot_std > 0:
            boot_sharpe = np.mean(boot_returns) / boot_std * np.sqrt(252)
            bootstrap_sharpes.append(boot_sharpe)

    if not bootstrap_sharpes:
        return sharpe, sharpe - 1, sharpe + 1

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_sharpes, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2))

    return sharpe, ci_lower, ci_upper


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
        gap_days: int = 5,
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_ratio: Ratio of data for training (default 0.7)
            n_splits: Number of walk-forward folds (default 5)
            min_train_days: Minimum days required for training (default 30)
            gap_days: Gap (embargo period) between train and test to prevent
                     information leakage from pending orders, market impact,
                     and other temporal effects. Default 5 days.

                     INSTITUTIONAL STANDARD: 3-10 days depending on strategy
                     holding period. Longer for strategies with longer horizons.
        """
        self.train_ratio = train_ratio or BACKTEST_PARAMS.get("TRAIN_RATIO", 0.7)
        self.n_splits = n_splits or BACKTEST_PARAMS.get("N_SPLITS", 5)
        self.min_train_days = min_train_days or BACKTEST_PARAMS.get("MIN_TRAIN_DAYS", 30)
        self.gap_days = gap_days
        self.overfitting_threshold = BACKTEST_PARAMS.get("OVERFITTING_RATIO_THRESHOLD", 2.0)

        self.results: List[WalkForwardResult] = []

    def create_time_splits(
        self, start_date: datetime, end_date: datetime
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
            fold_train_end = start_date + timedelta(
                days=int(fold_size * (i + 1) * self.train_ratio)
            )

            # Test period starts after gap
            test_start = fold_train_end + timedelta(days=self.gap_days)
            test_end = min(
                test_start + timedelta(days=int(fold_size * (1 - self.train_ratio))), end_date
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
        **backtest_kwargs,
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

        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION")
        print("=" * 80)
        print(f"Total period: {start_date_str} to {end_date_str}")
        print(f"Number of folds: {len(splits)}")
        print(f"Train/Test ratio: {self.train_ratio:.0%}/{1-self.train_ratio:.0%}")
        print("=" * 80 + "\n")

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
                **backtest_kwargs,
            )

            # Run out-of-sample (testing) backtest
            print("  Running out-of-sample backtest...")
            oos_result = await backtest_fn(
                symbols,
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
                **backtest_kwargs,
            )

            # Calculate comparison metrics
            is_return = is_result.get("total_return", 0)
            oos_return = oos_result.get("total_return", 0)

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
                fold_num=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                is_return=is_return,
                is_sharpe=is_result.get("sharpe_ratio", 0),
                is_trades=is_result.get("num_trades", 0),
                is_win_rate=is_result.get("win_rate", 0),
                oos_return=oos_return,
                oos_sharpe=oos_result.get("sharpe_ratio", 0),
                oos_trades=oos_result.get("num_trades", 0),
                oos_win_rate=oos_result.get("win_rate", 0),
                overfitting_ratio=overfitting_ratio,
                degradation=degradation,
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
        valid_ratios = [
            r.overfitting_ratio
            for r in self.results
            if r.overfitting_ratio != float("inf") and not np.isnan(r.overfitting_ratio)
        ]
        avg_overfit_ratio = np.mean(valid_ratios) if valid_ratios else self.overfitting_threshold

        # Count how many folds show overfitting
        overfit_folds = sum(
            1 for r in self.results if r.overfitting_ratio > self.overfitting_threshold
        )

        # Calculate consistency (OOS positive in what % of folds)
        oos_positive_folds = sum(1 for r in self.results if r.oos_return > 0)
        consistency = oos_positive_folds / len(self.results) if self.results else 0

        # Total trades
        total_oos_trades = sum(r.oos_trades for r in self.results)

        # === STATISTICAL SIGNIFICANCE TESTING ===
        # INSTITUTIONAL STANDARD: Don't rely on arbitrary thresholds
        # Use Wilcoxon signed-rank test for paired IS vs OOS comparison
        is_returns = [r.is_return for r in self.results]
        oos_returns = [r.oos_return for r in self.results]

        degradation_test = check_degradation_significance(is_returns, oos_returns)

        # Calculate Sharpe confidence intervals
        np.array([r.is_return for r in self.results])
        oos_returns_arr = np.array([r.oos_return for r in self.results])

        oos_sharpe, oos_sharpe_ci_lower, oos_sharpe_ci_upper = calculate_sharpe_confidence_interval(
            oos_returns_arr
        )

        # === UPDATED VALIDATION CRITERIA ===
        # Old: Arbitrary threshold-based
        # New: Statistical significance-based
        #
        # Strategy passes if:
        # 1. OOS returns are positive (basic requirement)
        # 2. No STATISTICALLY SIGNIFICANT degradation (Wilcoxon p > 0.05)
        #    OR degradation effect size is small
        # 3. Consistent across folds (>= 50% profitable)

        # Use statistical test results for validation
        passes_statistical_test = not degradation_test.degradation_significant or (
            degradation_test.effect_magnitude in ["negligible", "small"]
        )

        # Keep a hard overfitting-ratio guard even when sample size is too small
        # for significance testing (e.g., a single fold).
        passes_overfit_ratio = bool(avg_overfit_ratio <= self.overfitting_threshold)

        passes_validation = bool(
            avg_oos_return > 0
            and passes_statistical_test
            and consistency >= 0.5
            and passes_overfit_ratio
        )

        # Print summary
        print("\n" + "=" * 80)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 80)

        print("\nIn-Sample Performance (Training):")
        print(f"  Average Return:    {avg_is_return:+.2%}")
        print(f"  Average Sharpe:    {avg_is_sharpe:.2f}")

        print("\nOut-of-Sample Performance (Testing):")
        print(f"  Average Return:    {avg_oos_return:+.2%}")
        print(
            f"  Average Sharpe:    {avg_oos_sharpe:.2f} (95% CI: [{oos_sharpe_ci_lower:.2f}, {oos_sharpe_ci_upper:.2f}])"
        )
        print(f"  Total Trades:      {total_oos_trades}")
        print(f"  Consistency:       {consistency:.0%} of folds profitable")

        print("\nOverfitting Analysis (Legacy Metrics):")
        print(f"  Average Degradation:  {avg_degradation:.1%}")
        print(f"  Average Overfit Ratio: {avg_overfit_ratio:.2f}")
        print(f"  Overfit Folds:        {overfit_folds}/{len(self.results)}")

        print("\n🔬 STATISTICAL SIGNIFICANCE TEST (Wilcoxon):")
        print(f"  Test Statistic:     {degradation_test.wilcoxon_statistic:.2f}")
        print(f"  P-value:            {degradation_test.wilcoxon_p_value:.4f}")
        print(
            f"  Significant:        {'YES' if degradation_test.degradation_significant else 'NO'}"
        )
        print(
            f"  Effect Size:        {degradation_test.effect_size:.3f} ({degradation_test.effect_magnitude})"
        )
        print(f"  Mean Degradation:   {degradation_test.mean_degradation:.2%}")
        print(
            f"  95% CI:             [{degradation_test.degradation_ci_lower:.2%}, {degradation_test.degradation_ci_upper:.2%}]"
        )

        print("\n" + "=" * 80)
        if passes_validation:
            print("✅ VALIDATION PASSED")
            print("   Strategy shows consistent out-of-sample performance")
            if not degradation_test.degradation_significant:
                print("   IS→OOS degradation is NOT statistically significant")
        else:
            print("❌ VALIDATION FAILED")
            if avg_oos_return <= 0:
                print("   - Out-of-sample returns are negative")
            if (
                degradation_test.degradation_significant
                and degradation_test.effect_magnitude not in ["negligible", "small"]
            ):
                print(
                    f"   - STATISTICALLY SIGNIFICANT overfitting detected "
                    f"(p={degradation_test.wilcoxon_p_value:.4f}, effect={degradation_test.effect_magnitude})"
                )
            if consistency < 0.5:
                print(f"   - Inconsistent results ({consistency:.0%} folds profitable)")
            if not passes_overfit_ratio:
                print(
                    f"   - Overfit ratio too high ({avg_overfit_ratio:.2f} > "
                    f"{self.overfitting_threshold:.2f})"
                )

        print("=" * 80 + "\n")

        return {
            "passes_validation": passes_validation,
            "n_folds": len(self.results),
            # In-sample metrics
            "is_avg_return": avg_is_return,
            "is_avg_sharpe": avg_is_sharpe,
            # Out-of-sample metrics
            "oos_avg_return": avg_oos_return,
            "oos_avg_sharpe": avg_oos_sharpe,
            "oos_sharpe_ci": (oos_sharpe_ci_lower, oos_sharpe_ci_upper),
            "oos_total_trades": total_oos_trades,
            "oos_consistency": consistency,
            # Overfitting metrics (legacy)
            "avg_degradation": avg_degradation,
            "avg_overfit_ratio": avg_overfit_ratio,
            "overfit_folds": overfit_folds,
            # Statistical significance (NEW)
            "degradation_test": {
                "wilcoxon_statistic": degradation_test.wilcoxon_statistic,
                "wilcoxon_p_value": degradation_test.wilcoxon_p_value,
                "degradation_significant": degradation_test.degradation_significant,
                "mean_degradation": degradation_test.mean_degradation,
                "degradation_ci": (
                    degradation_test.degradation_ci_lower,
                    degradation_test.degradation_ci_upper,
                ),
                "effect_size": degradation_test.effect_size,
                "effect_magnitude": degradation_test.effect_magnitude,
                "interpretation": degradation_test.interpretation,
            },
            # Raw results
            "fold_results": self.results,
        }


async def run_walk_forward_validation(symbols: List[str], start_date: str, end_date: str, **kwargs):
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
    result = await validator.validate(simple_backtest, symbols, start_date, end_date, **kwargs)

    return result


if __name__ == "__main__":
    # Example usage
    async def main():
        result = await run_walk_forward_validation(
            symbols=["AAPL", "MSFT", "NVDA"], start_date="2024-01-01", end_date="2024-11-01"
        )

        print(f"Validation passed: {result['passes_validation']}")
        print(f"OOS Return: {result['oos_avg_return']:+.2%}")
        print(f"Overfit Ratio: {result['avg_overfit_ratio']:.2f}")

    asyncio.run(main())
