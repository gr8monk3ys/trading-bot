"""
Statistical Testing Module

Standalone statistical-testing primitives used by ``engine.performance_metrics``:

* Multiple-testing correction (Bonferroni, Benjamini-Hochberg FDR).
* Effect-size measures (Cohen's d, Hedge's g).
* ``SignificanceResult`` / ``EffectSizeResult`` dataclasses.

These helpers are deliberately pure functions (no I/O, no class state) so they
can be reused by any caller that needs principled statistical validation —
permutation tests, walk-forward harnesses, batch strategy comparisons, etc.

``calculate_effect_size`` (which composes ``calculate_cohens_d`` and
``calculate_hedges_g``) is intentionally kept in ``engine.performance_metrics``
so that monkeypatch-based tests can rebind the helper functions in that
module's namespace and have those overrides flow through to the composed
calculation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MULTIPLE TESTING CORRECTION
# =============================================================================


@dataclass
class SignificanceResult:
    """Result of statistical significance testing with corrections."""

    raw_p_value: float
    adjusted_p_value: float
    is_significant: bool
    correction_method: str
    n_tests: int
    alpha: float


def apply_bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> List[SignificanceResult]:
    """
    Apply Bonferroni correction for family-wise error rate control.

    The Bonferroni correction is conservative but guarantees the family-wise
    error rate (probability of at least one false positive) stays below alpha.

    CRITICAL: When testing multiple strategies or parameters, use this to avoid
    finding "significant" results by chance.

    Args:
        p_values: List of raw p-values from individual tests
        alpha: Desired family-wise error rate (default 0.05)

    Returns:
        List of SignificanceResult with adjusted p-values and significance
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []

    # Bonferroni adjustment: multiply each p-value by number of tests
    adjusted_alpha = alpha / n_tests

    results = []
    for p in p_values:
        adjusted_p = min(p * n_tests, 1.0)  # Cap at 1.0
        results.append(
            SignificanceResult(
                raw_p_value=p,
                adjusted_p_value=adjusted_p,
                is_significant=p < adjusted_alpha,
                correction_method="bonferroni",
                n_tests=n_tests,
                alpha=alpha,
            )
        )

    return results


def apply_fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[SignificanceResult]:
    """
    Apply Benjamini-Hochberg False Discovery Rate (FDR) correction.

    FDR controls the expected proportion of false positives among rejected
    hypotheses. Less conservative than Bonferroni, more appropriate when
    testing many hypotheses (e.g., screening many strategies).

    Use FDR when:
    - Testing many strategies and expect some to be truly good
    - Willing to accept some false positives in exchange for power

    Use Bonferroni when:
    - Need strict control (live trading decisions)
    - Testing few hypotheses

    Args:
        p_values: List of raw p-values from individual tests
        alpha: Desired false discovery rate (default 0.05)

    Returns:
        List of SignificanceResult with adjusted p-values and significance
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return []

    # Sort p-values and track original indices
    indexed_pvals = [(i, p) for i, p in enumerate(p_values)]
    indexed_pvals.sort(key=lambda x: x[1])

    # Calculate BH-adjusted p-values
    adjusted_pvals = [0.0] * n_tests

    # Start from largest p-value
    prev_adjusted = 1.0
    for rank in range(n_tests, 0, -1):
        original_idx, raw_p = indexed_pvals[rank - 1]
        # BH formula: p * n / rank
        adjusted = raw_p * n_tests / rank
        # Enforce monotonicity (adjusted p-values should be non-decreasing)
        adjusted = min(adjusted, prev_adjusted)
        adjusted = min(adjusted, 1.0)  # Cap at 1.0
        adjusted_pvals[original_idx] = adjusted
        prev_adjusted = adjusted

    # Determine significance using step-up procedure
    significant = [False] * n_tests
    for rank, (_original_idx, raw_p) in enumerate(indexed_pvals, 1):
        threshold = (rank / n_tests) * alpha
        if raw_p <= threshold:
            # Mark this and all smaller p-values as significant
            for j in range(rank):
                significant[indexed_pvals[j][0]] = True

    results = []
    for i, p in enumerate(p_values):
        results.append(
            SignificanceResult(
                raw_p_value=p,
                adjusted_p_value=adjusted_pvals[i],
                is_significant=significant[i],
                correction_method="benjamini-hochberg",
                n_tests=n_tests,
                alpha=alpha,
            )
        )

    return results


def calculate_adjusted_significance(
    raw_p: float, n_tests: int, method: str = "bonferroni", alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate significance for a single p-value with multiple testing adjustment.

    Convenience function when you have a single p-value but know the total
    number of tests performed.

    Args:
        raw_p: Raw p-value from test
        n_tests: Total number of tests performed (including this one)
        method: Correction method ('bonferroni' or 'fdr')
        alpha: Significance threshold (default 0.05)

    Returns:
        Dictionary with adjusted significance information
    """
    if method == "bonferroni":
        adjusted_p = min(raw_p * n_tests, 1.0)
        is_significant = raw_p < (alpha / n_tests)
    elif method == "fdr":
        # For single test, FDR = raw (would need rank for proper adjustment)
        adjusted_p = raw_p
        is_significant = raw_p < alpha
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return {
        "raw_p_value": raw_p,
        "adjusted_p_value": adjusted_p,
        "is_significant": is_significant,
        "correction_method": method,
        "n_tests": n_tests,
        "alpha": alpha,
        "adjusted_alpha": alpha / n_tests if method == "bonferroni" else alpha,
        "interpretation": (
            f"After {method} correction for {n_tests} tests, "
            f"p={adjusted_p:.4f} is {'significant' if is_significant else 'not significant'} "
            f"at alpha={alpha}"
        ),
    }


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""

    cohens_d: float
    hedges_g: float
    interpretation: str
    confidence_interval: tuple


def calculate_cohens_d(returns: np.ndarray, population_mean: float = 0.0) -> float:
    """
    Calculate Cohen's d effect size for returns vs a benchmark (default 0).

    Cohen's d measures how many standard deviations the mean is from the
    benchmark. This tells you the PRACTICAL significance, not just
    statistical significance.

    Interpretation (Cohen's conventions):
    - |d| < 0.2: negligible effect
    - 0.2 <= |d| < 0.5: small effect
    - 0.5 <= |d| < 0.8: medium effect
    - |d| >= 0.8: large effect

    For trading:
    - A strategy with d=0.3 has a small but real edge
    - A strategy with d=0.8 has a substantial edge

    Args:
        returns: Array of returns
        population_mean: Benchmark to compare against (default 0)

    Returns:
        Cohen's d effect size
    """
    if len(returns) == 0:
        return 0.0

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1))  # Sample std

    if std_return == 0:
        return 0.0

    return float((mean_return - population_mean) / std_return)


def calculate_hedges_g(returns: np.ndarray, population_mean: float = 0.0) -> float:
    """
    Calculate Hedge's g (bias-corrected Cohen's d).

    Hedge's g applies a correction factor for small sample sizes.
    Use this instead of Cohen's d when n < 50.

    The correction factor is: 1 - 3/(4n - 9)

    Args:
        returns: Array of returns
        population_mean: Benchmark to compare against (default 0)

    Returns:
        Hedge's g effect size
    """
    n = len(returns)
    if n < 4:
        return 0.0

    d = calculate_cohens_d(returns, population_mean)

    # Correction factor for small samples
    correction = 1 - (3 / (4 * n - 9))

    return d * correction
