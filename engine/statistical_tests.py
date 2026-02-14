"""
Statistical Tests Module

Institutional-grade statistical validation for trading strategies including:
- Permutation testing for strategy returns
- Runs test for independence
- Maximum drawdown significance
- Autocorrelation tests

These tests help validate that observed performance is not due to chance.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# PERMUTATION TESTING
# =============================================================================


@dataclass
class PermutationTestResult:
    """Result of permutation test for strategy returns."""

    observed_statistic: float
    null_distribution: np.ndarray
    p_value: float
    n_permutations: int
    is_significant: bool
    alpha: float
    interpretation: str


def permutation_test_returns(
    returns: np.ndarray,
    n_permutations: int = 10000,
    statistic: str = "mean",
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> PermutationTestResult:
    """
    Permutation test to validate that strategy returns are not due to chance.

    This is the GOLD STANDARD for validating trading strategy significance.
    Unlike t-tests, permutation tests:
    - Make no assumptions about return distribution
    - Generate an empirical null distribution
    - Directly answer: "What's the probability of getting this result by chance?"

    How it works:
    1. Calculate the observed statistic (e.g., mean return)
    2. Randomly shuffle the returns many times
    3. Calculate the statistic for each shuffled version
    4. Count how often shuffled results >= observed result
    5. p-value = proportion of times chance beats reality

    Args:
        returns: Array of strategy returns
        n_permutations: Number of permutations (10,000+ recommended)
        statistic: 'mean', 'sharpe', or 'total' (default 'mean')
        alpha: Significance threshold (default 0.05)
        random_state: Random seed for reproducibility

    Returns:
        PermutationTestResult with p-value and null distribution
    """
    if len(returns) < 10:
        return PermutationTestResult(
            observed_statistic=0.0,
            null_distribution=np.array([]),
            p_value=1.0,
            n_permutations=0,
            is_significant=False,
            alpha=alpha,
            interpretation="Insufficient data for permutation test (need >= 10 returns)",
        )

    rng = np.random.default_rng(random_state)

    # Calculate observed statistic
    observed = _calculate_statistic(returns, statistic)

    # Generate null distribution by permuting returns
    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle returns (breaks any temporal structure)
        permuted = rng.permutation(returns)
        null_distribution[i] = _calculate_statistic(permuted, statistic)

    # Calculate p-value (one-tailed: observed > null)
    # Add 1 to numerator and denominator for continuity correction
    n_exceeds = np.sum(null_distribution >= observed)
    p_value = (n_exceeds + 1) / (n_permutations + 1)

    is_significant = p_value < alpha

    # Interpretation
    percentile = 100 * (1 - p_value)
    if is_significant:
        interpretation = (
            f"Strategy {statistic} ({observed:.4f}) is in the {percentile:.1f}th percentile "
            f"of chance results. This is statistically significant (p={p_value:.4f} < {alpha}). "
            f"Only {p_value:.2%} of random strategies would perform this well."
        )
    else:
        interpretation = (
            f"Strategy {statistic} ({observed:.4f}) is in the {percentile:.1f}th percentile "
            f"of chance results. This is NOT significant (p={p_value:.4f} >= {alpha}). "
            f"{p_value:.0%} of random strategies would perform equally well or better."
        )

    return PermutationTestResult(
        observed_statistic=observed,
        null_distribution=null_distribution,
        p_value=p_value,
        n_permutations=n_permutations,
        is_significant=is_significant,
        alpha=alpha,
        interpretation=interpretation,
    )


def _calculate_statistic(returns: np.ndarray, statistic: str) -> float:
    """Calculate the specified statistic from returns."""
    if statistic == "mean":
        return np.mean(returns)
    elif statistic == "sharpe":
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    elif statistic == "total":
        return np.sum(returns)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")


def permutation_test_strategy(
    equity_curve: np.ndarray,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, PermutationTestResult]:
    """
    Comprehensive permutation test suite for a strategy's equity curve.

    Tests multiple statistics to catch different types of alpha:
    - Mean: Average return (most common)
    - Sharpe: Risk-adjusted return
    - Total: Cumulative return

    Args:
        equity_curve: Array of portfolio values over time
        n_permutations: Number of permutations per test
        alpha: Significance threshold
        random_state: Random seed

    Returns:
        Dictionary with results for each statistic
    """
    # Calculate returns from equity curve
    returns = np.diff(equity_curve) / equity_curve[:-1]

    results = {}
    for stat in ["mean", "sharpe", "total"]:
        results[stat] = permutation_test_returns(returns, n_permutations, stat, alpha, random_state)

    return results


# =============================================================================
# RUNS TEST FOR INDEPENDENCE
# =============================================================================


@dataclass
class RunsTestResult:
    """Result of runs test for trade independence."""

    n_runs: int
    n_expected: float
    z_statistic: float
    p_value: float
    is_random: bool
    interpretation: str


def runs_test(returns: np.ndarray, alpha: float = 0.05) -> RunsTestResult:
    """
    Runs test to check if wins/losses are randomly distributed.

    A "run" is a sequence of consecutive wins or losses.
    - Too few runs = momentum (wins follow wins, losses follow losses)
    - Too many runs = mean reversion (wins follow losses)
    - Expected runs = random

    Why it matters:
    - Non-random patterns might indicate exploitable market structure
    - OR might indicate data leakage / lookahead bias
    - Random patterns suggest strategy captures true alpha

    Args:
        returns: Array of returns (positive = win, negative = loss)
        alpha: Significance threshold

    Returns:
        RunsTestResult with test statistics and interpretation
    """
    if len(returns) < 10:
        return RunsTestResult(
            n_runs=0,
            n_expected=0,
            z_statistic=0,
            p_value=1.0,
            is_random=True,
            interpretation="Insufficient data for runs test (need >= 10 returns)",
        )

    # Convert to binary: 1 for positive, 0 for negative
    binary = (returns > 0).astype(int)

    # Count runs
    n_runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i - 1]:
            n_runs += 1

    # Count wins and losses
    n_wins = np.sum(binary)
    n_losses = len(binary) - n_wins

    if n_wins == 0 or n_losses == 0:
        return RunsTestResult(
            n_runs=n_runs,
            n_expected=0,
            z_statistic=0,
            p_value=1.0,
            is_random=True,
            interpretation="All wins or all losses - cannot perform runs test",
        )

    # Expected number of runs under null (random)
    n = len(binary)
    n_expected = (2 * n_wins * n_losses) / n + 1

    # Standard deviation of runs under null
    numerator = 2 * n_wins * n_losses * (2 * n_wins * n_losses - n)
    denominator = n**2 * (n - 1)
    if denominator == 0:
        std_runs = 1.0
    else:
        std_runs = np.sqrt(numerator / denominator)

    if std_runs == 0:
        std_runs = 1.0

    # Z-statistic
    z_stat = (n_runs - n_expected) / std_runs

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    is_random = p_value >= alpha

    # Interpretation
    if is_random:
        interpretation = (
            f"Runs test passed (p={p_value:.4f}). "
            f"Trade outcomes appear randomly distributed. "
            f"Observed {n_runs} runs, expected {n_expected:.1f}."
        )
    elif n_runs < n_expected:
        interpretation = (
            f"Runs test FAILED (p={p_value:.4f}). "
            f"Too few runs ({n_runs} vs {n_expected:.1f} expected) suggests momentum pattern. "
            f"Check for autocorrelation or data issues."
        )
    else:
        interpretation = (
            f"Runs test FAILED (p={p_value:.4f}). "
            f"Too many runs ({n_runs} vs {n_expected:.1f} expected) suggests mean reversion pattern. "
            f"Check for overfitting or data issues."
        )

    return RunsTestResult(
        n_runs=n_runs,
        n_expected=n_expected,
        z_statistic=z_stat,
        p_value=p_value,
        is_random=is_random,
        interpretation=interpretation,
    )


# =============================================================================
# MAXIMUM DRAWDOWN SIGNIFICANCE
# =============================================================================


@dataclass
class DrawdownSignificanceResult:
    """Result of maximum drawdown significance test."""

    observed_max_dd: float
    expected_max_dd: float
    dd_percentile: float
    p_value: float
    is_significant: bool
    interpretation: str


def max_drawdown_significance(
    returns: np.ndarray,
    n_simulations: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> DrawdownSignificanceResult:
    """
    Test if maximum drawdown is significantly better than chance.

    Simulates random walks with same mean/volatility to establish
    expected drawdown distribution. Lower drawdown than expected
    suggests genuine risk management skill.

    Args:
        returns: Array of strategy returns
        n_simulations: Number of simulations for null distribution
        alpha: Significance threshold
        random_state: Random seed

    Returns:
        DrawdownSignificanceResult with comparison to null
    """
    if len(returns) < 20:
        return DrawdownSignificanceResult(
            observed_max_dd=0.0,
            expected_max_dd=0.0,
            dd_percentile=50.0,
            p_value=0.5,
            is_significant=False,
            interpretation="Insufficient data for drawdown significance test",
        )

    rng = np.random.default_rng(random_state)

    # Calculate observed max drawdown
    observed_dd = _calculate_max_drawdown(returns)

    # Estimate return distribution parameters
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    n = len(returns)

    # Simulate random strategies with same characteristics
    simulated_dds = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Generate random returns with same mean/std
        sim_returns = rng.normal(mean_ret, std_ret, n)
        simulated_dds[i] = _calculate_max_drawdown(sim_returns)

    # Calculate percentile (lower is better for drawdown)
    dd_percentile = 100 * np.mean(simulated_dds <= observed_dd)
    p_value = dd_percentile / 100

    expected_dd = np.median(simulated_dds)
    is_significant = p_value < alpha

    # Interpretation
    if is_significant:
        interpretation = (
            f"Maximum drawdown ({observed_dd:.2%}) is in the {dd_percentile:.1f}th percentile - "
            f"significantly better than expected ({expected_dd:.2%}). "
            f"Suggests genuine drawdown control, not just luck."
        )
    else:
        interpretation = (
            f"Maximum drawdown ({observed_dd:.2%}) is in the {dd_percentile:.1f}th percentile - "
            f"not significantly different from expected ({expected_dd:.2%}). "
            f"Drawdown is consistent with random chance given the return distribution."
        )

    return DrawdownSignificanceResult(
        observed_max_dd=observed_dd,
        expected_max_dd=expected_dd,
        dd_percentile=dd_percentile,
        p_value=p_value,
        is_significant=is_significant,
        interpretation=interpretation,
    )


def _calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return abs(np.min(drawdown))


# =============================================================================
# AUTOCORRELATION TEST
# =============================================================================


@dataclass
class AutocorrelationResult:
    """Result of return autocorrelation analysis."""

    lag1_autocorr: float
    lag1_p_value: float
    significant_lags: List[int]
    ljung_box_statistic: float
    ljung_box_p_value: float
    has_autocorrelation: bool
    interpretation: str


def autocorrelation_test(
    returns: np.ndarray,
    max_lag: int = 10,
    alpha: float = 0.05,
) -> AutocorrelationResult:
    """
    Test for autocorrelation in strategy returns.

    Autocorrelation can indicate:
    - Momentum patterns (positive autocorr)
    - Mean reversion (negative autocorr)
    - Data issues or lookahead bias

    Independent returns (no autocorrelation) are generally preferred
    as they suggest strategy captures fresh alpha each period.

    Args:
        returns: Array of returns
        max_lag: Maximum lag to test (default 10)
        alpha: Significance threshold

    Returns:
        AutocorrelationResult with lag analysis
    """
    n = len(returns)
    if n < max_lag + 10:
        return AutocorrelationResult(
            lag1_autocorr=0.0,
            lag1_p_value=1.0,
            significant_lags=[],
            ljung_box_statistic=0.0,
            ljung_box_p_value=1.0,
            has_autocorrelation=False,
            interpretation="Insufficient data for autocorrelation test",
        )

    # Demean returns
    demeaned = returns - np.mean(returns)

    # Calculate autocorrelations for each lag
    autocorrs = []
    p_values = []
    significant_lags = []

    for lag in range(1, max_lag + 1):
        # Autocorrelation at this lag
        if n - lag > 0:
            acf = np.corrcoef(demeaned[:-lag], demeaned[lag:])[0, 1]
        else:
            acf = 0.0
        autocorrs.append(acf)

        # Standard error under null (no autocorrelation)
        se = 1 / np.sqrt(n)

        # Z-test for significance
        z = acf / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        p_values.append(p_val)

        if p_val < alpha:
            significant_lags.append(lag)

    # Ljung-Box test for overall autocorrelation
    q_stat = n * (n + 2) * sum((acf**2) / (n - lag) for lag, acf in enumerate(autocorrs, 1))
    lb_p_value = 1 - stats.chi2.cdf(q_stat, max_lag)

    has_autocorr = lb_p_value < alpha or len(significant_lags) > 0

    # Interpretation
    lag1_acf = autocorrs[0] if autocorrs else 0.0
    lag1_p = p_values[0] if p_values else 1.0

    if has_autocorr:
        if lag1_acf > 0:
            pattern = "momentum (positive autocorrelation)"
        else:
            pattern = "mean reversion (negative autocorrelation)"

        interpretation = (
            f"Significant autocorrelation detected ({pattern}). "
            f"Lag-1 autocorr: {lag1_acf:.3f} (p={lag1_p:.4f}). "
            f"Significant at lags: {significant_lags}. "
            f"Ljung-Box test: Q={q_stat:.2f}, p={lb_p_value:.4f}. "
            f"Check for data issues or potential alpha source."
        )
    else:
        interpretation = (
            f"No significant autocorrelation detected. "
            f"Lag-1 autocorr: {lag1_acf:.3f} (p={lag1_p:.4f}). "
            f"Ljung-Box test: Q={q_stat:.2f}, p={lb_p_value:.4f}. "
            f"Returns appear independent - good sign for strategy validity."
        )

    return AutocorrelationResult(
        lag1_autocorr=lag1_acf,
        lag1_p_value=lag1_p,
        significant_lags=significant_lags,
        ljung_box_statistic=q_stat,
        ljung_box_p_value=lb_p_value,
        has_autocorrelation=has_autocorr,
        interpretation=interpretation,
    )


# =============================================================================
# MULTIPLE TESTING CORRECTION
# =============================================================================


@dataclass
class MultipleTestingResult:
    """Result of multiple testing correction."""

    original_p_values: List[float]
    adjusted_p_values: List[float]
    original_alpha: float
    adjusted_alpha: Optional[float]  # For Bonferroni
    significant_original: List[bool]
    significant_adjusted: List[bool]
    correction_method: str
    n_tests: int
    n_significant_original: int
    n_significant_adjusted: int
    interpretation: str


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> MultipleTestingResult:
    """
    Apply Bonferroni correction for multiple testing.

    The Bonferroni correction is the most conservative multiple testing correction.
    It controls the family-wise error rate (FWER) - the probability of making
    at least one Type I error across all tests.

    Method: Divide alpha by the number of tests, or multiply p-values by n_tests.

    Example:
        If you run 4 tests at alpha=0.05:
        - Original: p < 0.05 is significant
        - Bonferroni: p < 0.0125 (0.05/4) is significant

    When to use:
        - When ANY false positive is unacceptable
        - When tests are independent
        - When you have few tests (< 10)

    Args:
        p_values: List of p-values from multiple tests
        alpha: Original significance threshold (default 0.05)

    Returns:
        MultipleTestingResult with adjusted alpha and significance
    """
    n_tests = len(p_values)

    if n_tests == 0:
        return MultipleTestingResult(
            original_p_values=[],
            adjusted_p_values=[],
            original_alpha=alpha,
            adjusted_alpha=alpha,
            significant_original=[],
            significant_adjusted=[],
            correction_method="bonferroni",
            n_tests=0,
            n_significant_original=0,
            n_significant_adjusted=0,
            interpretation="No tests to correct",
        )

    # Bonferroni correction: divide alpha by number of tests
    adjusted_alpha = alpha / n_tests

    # Adjusted p-values (multiply by n_tests, cap at 1.0)
    adjusted_p_values = [min(p * n_tests, 1.0) for p in p_values]

    # Determine significance
    significant_original = [p < alpha for p in p_values]
    significant_adjusted = [p < adjusted_alpha for p in p_values]

    n_sig_orig = sum(significant_original)
    n_sig_adj = sum(significant_adjusted)

    # Interpretation
    if n_sig_orig == n_sig_adj:
        interpretation = (
            f"Bonferroni correction: {n_sig_adj}/{n_tests} tests remain significant. "
            f"Adjusted α = {adjusted_alpha:.4f} (original α = {alpha}). "
            f"All originally significant results survive correction."
        )
    elif n_sig_adj == 0:
        interpretation = (
            f"Bonferroni correction: {n_sig_orig}/{n_tests} originally significant tests "
            f"reduced to 0/{n_tests} after correction. "
            f"Adjusted α = {adjusted_alpha:.4f} is very strict. "
            f"Consider using FDR correction for more power."
        )
    else:
        interpretation = (
            f"Bonferroni correction: {n_sig_orig} → {n_sig_adj} significant tests. "
            f"Adjusted α = {adjusted_alpha:.4f}. "
            f"{n_sig_orig - n_sig_adj} results may be false positives."
        )

    return MultipleTestingResult(
        original_p_values=list(p_values),
        adjusted_p_values=adjusted_p_values,
        original_alpha=alpha,
        adjusted_alpha=adjusted_alpha,
        significant_original=significant_original,
        significant_adjusted=significant_adjusted,
        correction_method="bonferroni",
        n_tests=n_tests,
        n_significant_original=n_sig_orig,
        n_significant_adjusted=n_sig_adj,
        interpretation=interpretation,
    )


def benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> MultipleTestingResult:
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.

    The Benjamini-Hochberg (BH) procedure controls the False Discovery Rate (FDR) -
    the expected proportion of false positives among all positive results.

    FDR is less conservative than Bonferroni, providing more statistical power
    while still controlling error rates in a meaningful way.

    Method:
        1. Sort p-values from smallest to largest
        2. For rank i (1 to n), compare p[i] to (i/n) * alpha
        3. Find the largest i where p[i] <= (i/n) * alpha
        4. All tests with p <= p[i] are significant

    Example:
        4 tests with p-values [0.02, 0.03, 0.04, 0.06] at alpha=0.05:
        - Bonferroni: adjusted α = 0.0125, 0 significant
        - BH-FDR: 0.02 < 0.0125 (1/4*0.05), 0.03 < 0.025 (2/4*0.05),
                 0.04 > 0.0375 (3/4*0.05) → 2 significant

    When to use:
        - When some false positives are acceptable
        - When you want more statistical power
        - When tests may be correlated
        - Standard for genomics, neuroscience, finance

    Args:
        p_values: List of p-values from multiple tests
        alpha: Target FDR (default 0.05 = expect 5% false positives among positives)

    Returns:
        MultipleTestingResult with BH-adjusted significance
    """
    n_tests = len(p_values)

    if n_tests == 0:
        return MultipleTestingResult(
            original_p_values=[],
            adjusted_p_values=[],
            original_alpha=alpha,
            adjusted_alpha=None,
            significant_original=[],
            significant_adjusted=[],
            correction_method="benjamini_hochberg_fdr",
            n_tests=0,
            n_significant_original=0,
            n_significant_adjusted=0,
            interpretation="No tests to correct",
        )

    # Create index-value pairs and sort by p-value
    indexed_p = list(enumerate(p_values))
    sorted_p = sorted(indexed_p, key=lambda x: x[1])

    # Calculate BH critical values and adjusted p-values
    adjusted_p_values = [0.0] * n_tests
    significant_adjusted = [False] * n_tests

    # BH procedure: find largest k where p[k] <= (k/n) * alpha
    # Then reject all hypotheses with p <= p[k]

    # Calculate adjusted p-values (method: start from largest, take cumulative min)
    prev_adj_p = 1.0
    for i in range(n_tests - 1, -1, -1):
        orig_idx, p_val = sorted_p[i]
        rank = i + 1

        # Adjusted p-value = p * (n / rank), capped at previous adjusted p
        adj_p = min(p_val * n_tests / rank, prev_adj_p)
        adj_p = min(adj_p, 1.0)  # Cap at 1.0
        adjusted_p_values[orig_idx] = adj_p
        prev_adj_p = adj_p

    # Determine significance using adjusted p-values
    significant_adjusted = [adj_p < alpha for adj_p in adjusted_p_values]
    significant_original = [p < alpha for p in p_values]

    n_sig_orig = sum(significant_original)
    n_sig_adj = sum(significant_adjusted)

    # Interpretation
    if n_sig_orig == n_sig_adj:
        interpretation = (
            f"BH-FDR correction: {n_sig_adj}/{n_tests} tests remain significant. "
            f"All originally significant results survive FDR correction at q={alpha}. "
            f"Expected false discovery rate ≤ {alpha:.0%}."
        )
    elif n_sig_adj == 0:
        interpretation = (
            f"BH-FDR correction: {n_sig_orig}/{n_tests} originally significant tests "
            f"reduced to 0/{n_tests} after correction. "
            f"None of the results survive FDR correction at q={alpha}."
        )
    else:
        interpretation = (
            f"BH-FDR correction: {n_sig_orig} → {n_sig_adj} significant tests. "
            f"Expected false discovery rate among positives ≤ {alpha:.0%}. "
            f"More powerful than Bonferroni while controlling FDR."
        )

    return MultipleTestingResult(
        original_p_values=list(p_values),
        adjusted_p_values=adjusted_p_values,
        original_alpha=alpha,
        adjusted_alpha=None,  # FDR doesn't use adjusted alpha in same way
        significant_original=significant_original,
        significant_adjusted=significant_adjusted,
        correction_method="benjamini_hochberg_fdr",
        n_tests=n_tests,
        n_significant_original=n_sig_orig,
        n_significant_adjusted=n_sig_adj,
        interpretation=interpretation,
    )


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""

    cohens_d: float
    hedges_g: float
    interpretation: str
    magnitude: str  # 'negligible', 'small', 'medium', 'large'


def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> EffectSizeResult:
    """
    Calculate Cohen's d and Hedge's g effect sizes.

    Effect size quantifies the MAGNITUDE of a difference, not just whether
    it exists. A statistically significant result can have a tiny effect,
    and a non-significant result can have a large effect (due to small sample).

    Cohen's d:
        d = (mean1 - mean2) / pooled_std
        Interpretation:
        - |d| < 0.2: Negligible
        - 0.2 ≤ |d| < 0.5: Small
        - 0.5 ≤ |d| < 0.8: Medium
        - |d| ≥ 0.8: Large

    Hedge's g:
        Bias-corrected version of Cohen's d for small samples.
        g = d * (1 - 3 / (4n - 9))
        Use when n1 + n2 < 50.

    For trading:
        - Compare strategy returns vs benchmark returns
        - Compare in-sample vs out-of-sample performance
        - Quantify regime differences

    Args:
        group1: First group of observations (e.g., strategy returns)
        group2: Second group of observations (e.g., benchmark returns)

    Returns:
        EffectSizeResult with Cohen's d, Hedge's g, and interpretation
    """
    n1, n2 = len(group1), len(group2)

    if n1 < 2 or n2 < 2:
        return EffectSizeResult(
            cohens_d=0.0,
            hedges_g=0.0,
            interpretation="Insufficient data for effect size calculation",
            magnitude="unknown",
        )

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return EffectSizeResult(
            cohens_d=0.0,
            hedges_g=0.0,
            interpretation="Zero variance - cannot calculate effect size",
            magnitude="unknown",
        )

    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std

    # Hedge's g (bias correction)
    n_total = n1 + n2
    correction_factor = 1 - (3 / (4 * n_total - 9))
    hedges_g = cohens_d * correction_factor

    # Interpret magnitude (using absolute value)
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    # Direction
    direction = "higher" if cohens_d > 0 else "lower"

    interpretation = (
        f"Effect size: Cohen's d = {cohens_d:.3f}, Hedge's g = {hedges_g:.3f}. "
        f"Magnitude: {magnitude.upper()}. "
        f"Group 1 mean is {abs_d:.2f} pooled SDs {direction} than Group 2. "
        f"Practical significance: {'meaningful' if abs_d >= 0.5 else 'may not be practically meaningful'}."
    )

    return EffectSizeResult(
        cohens_d=cohens_d,
        hedges_g=hedges_g,
        interpretation=interpretation,
        magnitude=magnitude,
    )


def calculate_strategy_effect_size(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> EffectSizeResult:
    """
    Calculate effect size for strategy vs benchmark comparison.

    This answers: "How big is the difference between strategy and benchmark?"

    Args:
        strategy_returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns (same length)

    Returns:
        EffectSizeResult comparing strategy to benchmark
    """
    return calculate_effect_size(strategy_returns, benchmark_returns)


# =============================================================================
# COMPREHENSIVE STATISTICAL VALIDATION
# =============================================================================


def comprehensive_statistical_validation(
    returns: np.ndarray,
    equity_curve: Optional[np.ndarray] = None,
    benchmark_returns: Optional[np.ndarray] = None,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    apply_multiple_testing_correction: bool = True,
) -> Dict[str, Any]:
    """
    Run all statistical validation tests on a strategy.

    This is the main entry point for institutional-grade validation.
    Combines:
    1. Permutation testing (is alpha real?)
    2. Runs test (are trades independent?)
    3. Autocorrelation test (any patterns in returns?)
    4. Drawdown significance (is risk management skill or luck?)
    5. Multiple testing correction (Bonferroni and BH-FDR)
    6. Effect size (Cohen's d, Hedge's g) vs benchmark

    IMPORTANT: When running multiple tests, raw p-values can be misleading.
    With 4 tests at α=0.05, there's ~19% chance of at least one false positive.
    Multiple testing correction adjusts for this.

    Args:
        returns: Array of strategy returns
        equity_curve: Optional equity curve (derived from returns if not provided)
        benchmark_returns: Optional benchmark returns for effect size calculation
        n_permutations: Number of permutations for tests
        alpha: Significance threshold
        random_state: Random seed for reproducibility
        apply_multiple_testing_correction: If True, apply Bonferroni and BH-FDR

    Returns:
        Dictionary with all test results and overall assessment
    """
    results = {
        "n_returns": len(returns),
        "tests": {},
        "p_values": {},  # Collect all p-values for correction
        "passed_tests": [],
        "failed_tests": [],
        "warnings": [],
        "overall_valid": False,
    }

    if len(returns) < 20:
        results["warnings"].append(
            "Insufficient data for comprehensive validation (need >= 20 returns)"
        )
        return results

    # 1. Permutation test
    perm_result = permutation_test_returns(returns, n_permutations, "sharpe", alpha, random_state)
    results["tests"]["permutation"] = {
        "observed_sharpe": perm_result.observed_statistic,
        "p_value": perm_result.p_value,
        "is_significant": perm_result.is_significant,
        "interpretation": perm_result.interpretation,
    }
    results["p_values"]["permutation"] = perm_result.p_value
    if perm_result.is_significant:
        results["passed_tests"].append("permutation")
    else:
        results["failed_tests"].append("permutation")

    # 2. Runs test
    runs_result = runs_test(returns, alpha)
    results["tests"]["runs"] = {
        "n_runs": runs_result.n_runs,
        "n_expected": runs_result.n_expected,
        "p_value": runs_result.p_value,
        "is_random": runs_result.is_random,
        "interpretation": runs_result.interpretation,
    }
    results["p_values"]["runs"] = runs_result.p_value
    if runs_result.is_random:
        results["passed_tests"].append("runs")
    else:
        results["failed_tests"].append("runs")
        results["warnings"].append("Non-random trade patterns detected - investigate further")

    # 3. Autocorrelation test
    acf_result = autocorrelation_test(returns, alpha=alpha)
    results["tests"]["autocorrelation"] = {
        "lag1_autocorr": acf_result.lag1_autocorr,
        "significant_lags": acf_result.significant_lags,
        "ljung_box_p_value": acf_result.ljung_box_p_value,
        "has_autocorrelation": acf_result.has_autocorrelation,
        "interpretation": acf_result.interpretation,
    }
    results["p_values"]["autocorrelation"] = acf_result.ljung_box_p_value
    if not acf_result.has_autocorrelation:
        results["passed_tests"].append("autocorrelation")
    else:
        results["failed_tests"].append("autocorrelation")

    # 4. Drawdown significance
    dd_result = max_drawdown_significance(returns, n_permutations, alpha, random_state)
    results["tests"]["drawdown"] = {
        "observed_max_dd": dd_result.observed_max_dd,
        "expected_max_dd": dd_result.expected_max_dd,
        "dd_percentile": dd_result.dd_percentile,
        "p_value": dd_result.p_value,
        "is_significant": dd_result.is_significant,
        "interpretation": dd_result.interpretation,
    }
    results["p_values"]["drawdown"] = dd_result.p_value
    # For drawdown, lower percentile is better (not failing this test)
    results["tests"]["drawdown"]["note"] = "Drawdown test: low percentile = good risk management"

    # 5. Multiple testing correction
    if apply_multiple_testing_correction:
        p_value_list = list(results["p_values"].values())
        test_names = list(results["p_values"].keys())

        # Bonferroni correction
        bonf_result = bonferroni_correction(p_value_list, alpha)
        results["multiple_testing"] = {
            "bonferroni": {
                "adjusted_alpha": bonf_result.adjusted_alpha,
                "n_significant_original": bonf_result.n_significant_original,
                "n_significant_adjusted": bonf_result.n_significant_adjusted,
                "interpretation": bonf_result.interpretation,
                "significant_tests": [
                    test_names[i] for i, sig in enumerate(bonf_result.significant_adjusted) if sig
                ],
            }
        }

        # Benjamini-Hochberg FDR correction
        fdr_result = benjamini_hochberg_fdr(p_value_list, alpha)
        results["multiple_testing"]["fdr"] = {
            "n_significant_original": fdr_result.n_significant_original,
            "n_significant_adjusted": fdr_result.n_significant_adjusted,
            "interpretation": fdr_result.interpretation,
            "significant_tests": [
                test_names[i] for i, sig in enumerate(fdr_result.significant_adjusted) if sig
            ],
            "adjusted_p_values": {
                test_names[i]: fdr_result.adjusted_p_values[i] for i in range(len(test_names))
            },
        }

        # Add warning if corrections change conclusions
        if bonf_result.n_significant_adjusted < bonf_result.n_significant_original:
            results["warnings"].append(
                f"Bonferroni correction reduced significant tests from "
                f"{bonf_result.n_significant_original} to {bonf_result.n_significant_adjusted}. "
                f"Some results may be false positives."
            )

    # 6. Effect size vs benchmark (if provided)
    if benchmark_returns is not None and len(benchmark_returns) >= 10:
        # Align lengths if needed
        min_len = min(len(returns), len(benchmark_returns))
        effect_result = calculate_effect_size(returns[:min_len], benchmark_returns[:min_len])
        results["effect_size"] = {
            "cohens_d": effect_result.cohens_d,
            "hedges_g": effect_result.hedges_g,
            "magnitude": effect_result.magnitude,
            "interpretation": effect_result.interpretation,
        }

        # Add effect size to assessment
        if effect_result.magnitude in ["negligible", "small"]:
            results["warnings"].append(
                f"Effect size vs benchmark is {effect_result.magnitude} "
                f"(d={effect_result.cohens_d:.3f}). "
                f"Even if significant, practical difference may be minimal."
            )

    # Overall assessment
    # Strategy is valid if permutation test passes (core requirement)
    # But we now also consider multiple testing correction
    permutation_significant_raw = perm_result.is_significant

    # After multiple testing correction, check if permutation still significant
    if apply_multiple_testing_correction:
        perm_idx = test_names.index("permutation")
        permutation_significant_corrected = fdr_result.significant_adjusted[perm_idx]

        results["overall_valid"] = permutation_significant_corrected

        if permutation_significant_raw and not permutation_significant_corrected:
            results["warnings"].append(
                "WARNING: Permutation test was significant at raw α but NOT after "
                "FDR correction. This suggests the result may be a false positive."
            )
    else:
        results["overall_valid"] = permutation_significant_raw

    if results["overall_valid"]:
        correction_note = " (survives FDR correction)" if apply_multiple_testing_correction else ""
        results["summary"] = (
            f"Strategy PASSES validation{correction_note}. "
            f"Permutation test confirms alpha is statistically significant (p={perm_result.p_value:.4f}). "
            f"Passed {len(results['passed_tests'])}/{len(results['tests'])} tests."
        )
    else:
        if apply_multiple_testing_correction and permutation_significant_raw:
            results["summary"] = (
                f"Strategy FAILS validation after multiple testing correction. "
                f"Permutation test significant at raw α (p={perm_result.p_value:.4f}) "
                f"but NOT after FDR correction. "
                f"Do NOT deploy to live trading without further investigation."
            )
        else:
            results["summary"] = (
                f"Strategy FAILS validation. "
                f"Permutation test shows results could be due to chance (p={perm_result.p_value:.4f}). "
                f"Do NOT deploy to live trading without further investigation."
            )

    return results
