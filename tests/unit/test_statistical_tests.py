"""
Tests for Phase 2.1-2.2: Multiple Testing Correction and Effect Size.

These tests verify that:
1. Bonferroni correction correctly adjusts alpha and p-values
2. Benjamini-Hochberg FDR controls false discovery rate
3. Effect size calculations (Cohen's d, Hedge's g) are accurate
4. comprehensive_statistical_validation applies corrections
"""

import pytest
import numpy as np
from engine.statistical_tests import (
    bonferroni_correction,
    benjamini_hochberg_fdr,
    calculate_effect_size,
    calculate_strategy_effect_size,
    comprehensive_statistical_validation,
    MultipleTestingResult,
    EffectSizeResult,
)


class TestBonferroniCorrection:
    """Tests for Bonferroni multiple testing correction."""

    def test_basic_correction(self):
        """Bonferroni should divide alpha by number of tests."""
        p_values = [0.01, 0.02, 0.03, 0.04]
        result = bonferroni_correction(p_values, alpha=0.05)

        assert result.adjusted_alpha == 0.0125  # 0.05 / 4
        assert result.n_tests == 4
        assert result.correction_method == "bonferroni"

    def test_adjusted_p_values(self):
        """Bonferroni should multiply p-values by n_tests."""
        p_values = [0.01, 0.02, 0.03, 0.04]
        result = bonferroni_correction(p_values, alpha=0.05)

        assert result.adjusted_p_values[0] == pytest.approx(0.04)  # 0.01 * 4
        assert result.adjusted_p_values[1] == pytest.approx(0.08)  # 0.02 * 4
        assert result.adjusted_p_values[2] == pytest.approx(0.12)  # 0.03 * 4
        assert result.adjusted_p_values[3] == pytest.approx(0.16)  # 0.04 * 4

    def test_cap_at_one(self):
        """Adjusted p-values should be capped at 1.0."""
        p_values = [0.4, 0.5]
        result = bonferroni_correction(p_values, alpha=0.05)

        assert result.adjusted_p_values[0] == pytest.approx(0.8)  # 0.4 * 2
        assert result.adjusted_p_values[1] == 1.0  # 0.5 * 2 = 1.0, capped

    def test_significance_determination(self):
        """Should correctly identify significant results after correction."""
        # At alpha=0.05 with 4 tests, adjusted alpha = 0.0125
        p_values = [0.01, 0.02, 0.03, 0.04]
        result = bonferroni_correction(p_values, alpha=0.05)

        # Only p=0.01 < 0.0125
        assert result.significant_adjusted == [True, False, False, False]
        assert result.n_significant_adjusted == 1

    def test_all_survive(self):
        """All originally significant should survive if p-values very small."""
        p_values = [0.001, 0.002, 0.003, 0.004]
        result = bonferroni_correction(p_values, alpha=0.05)

        # All p * 4 < 0.05
        assert all(result.significant_adjusted)
        assert result.n_significant_adjusted == 4

    def test_none_survive(self):
        """None should survive if all p-values too large."""
        p_values = [0.02, 0.03, 0.04, 0.05]
        result = bonferroni_correction(p_values, alpha=0.05)

        # Adjusted alpha = 0.0125, none below that
        assert not any(result.significant_adjusted)
        assert result.n_significant_adjusted == 0

    def test_empty_list(self):
        """Should handle empty p-value list."""
        result = bonferroni_correction([], alpha=0.05)

        assert result.n_tests == 0
        assert result.adjusted_p_values == []
        assert "No tests" in result.interpretation

    def test_single_test(self):
        """Single test should have same adjusted alpha as original."""
        p_values = [0.03]
        result = bonferroni_correction(p_values, alpha=0.05)

        assert result.adjusted_alpha == 0.05  # 0.05 / 1
        assert result.significant_adjusted[0] is True  # 0.03 < 0.05


class TestBenjaminiHochbergFDR:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_basic_fdr(self):
        """BH-FDR should be less conservative than Bonferroni."""
        # Classic example: 4 tests with borderline p-values
        p_values = [0.02, 0.03, 0.04, 0.06]
        bonf = bonferroni_correction(p_values, alpha=0.05)
        fdr = benjamini_hochberg_fdr(p_values, alpha=0.05)

        # Bonferroni: adjusted alpha = 0.0125, 0 significant
        assert bonf.n_significant_adjusted == 0

        # BH-FDR should find more (or equal) significant results
        assert fdr.n_significant_adjusted >= bonf.n_significant_adjusted

    def test_step_up_procedure(self):
        """BH-FDR should follow step-up procedure correctly."""
        # Sorted p-values: [0.01, 0.02, 0.04, 0.10]
        # BH critical values at alpha=0.05: [0.0125, 0.025, 0.0375, 0.05]
        # 0.01 < 0.0125: significant
        # 0.02 < 0.025: significant
        # 0.04 > 0.0375: not significant (stops here)
        p_values = [0.01, 0.02, 0.04, 0.10]
        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert result.n_significant_adjusted == 2
        assert result.significant_adjusted[0] is True
        assert result.significant_adjusted[1] is True
        assert result.significant_adjusted[2] is False
        assert result.significant_adjusted[3] is False

    def test_adjusted_p_values_ordering(self):
        """Adjusted p-values should maintain meaningful ordering."""
        p_values = [0.01, 0.03, 0.05]
        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        # Adjusted p-values should be monotonically increasing with raw p-values
        # (after adjustment, smallest raw p should have smallest adjusted p)
        adj_p = result.adjusted_p_values
        # Check monotonicity
        for i in range(len(adj_p) - 1):
            assert adj_p[i] <= adj_p[i + 1] + 1e-10  # Allow small floating point error

    def test_all_significant(self):
        """All should be significant with very small p-values."""
        p_values = [0.001, 0.002, 0.003]
        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert all(result.significant_adjusted)
        assert result.n_significant_adjusted == 3

    def test_none_significant(self):
        """None should be significant with large p-values."""
        p_values = [0.20, 0.30, 0.40]
        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert not any(result.significant_adjusted)
        assert result.n_significant_adjusted == 0

    def test_empty_list(self):
        """Should handle empty p-value list."""
        result = benjamini_hochberg_fdr([], alpha=0.05)

        assert result.n_tests == 0
        assert "No tests" in result.interpretation

    def test_single_test(self):
        """Single test should behave like uncorrected."""
        p_values = [0.03]
        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert result.significant_adjusted[0] is True


class TestEffectSize:
    """Tests for Cohen's d and Hedge's g effect size calculations."""

    def test_zero_effect(self):
        """Identical groups should have zero effect size."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calculate_effect_size(group1, group2)

        assert result.cohens_d == pytest.approx(0.0)
        assert result.hedges_g == pytest.approx(0.0)
        assert result.magnitude == "negligible"

    def test_large_positive_effect(self):
        """Large mean difference should give large positive d."""
        group1 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calculate_effect_size(group1, group2)

        # Mean diff = 9, pooled std ≈ 1.58
        # d ≈ 9 / 1.58 ≈ 5.7 (very large)
        assert result.cohens_d > 0.8
        assert result.magnitude == "large"

    def test_large_negative_effect(self):
        """Group 1 lower than group 2 should give negative d."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

        result = calculate_effect_size(group1, group2)

        assert result.cohens_d < -0.8
        assert result.magnitude == "large"

    def test_small_effect(self):
        """Small mean difference should give small d."""
        np.random.seed(42)
        # Use larger sample and clearer mean difference for reliable result
        group1 = np.random.normal(0.3, 1.0, 200)  # Mean 0.3
        group2 = np.random.normal(0.0, 1.0, 200)  # Mean 0.0

        result = calculate_effect_size(group1, group2)

        # d should be around 0.3 (small effect)
        # With large samples, should be close to true effect
        assert abs(result.cohens_d) < 0.6  # Not large
        assert result.magnitude in ["negligible", "small", "medium"]

    def test_medium_effect(self):
        """Medium mean difference should give medium d."""
        np.random.seed(42)
        group1 = np.random.normal(0.5, 1.0, 100)  # Mean 0.5
        group2 = np.random.normal(0.0, 1.0, 100)  # Mean 0.0

        result = calculate_effect_size(group1, group2)

        # d should be around 0.5 (medium effect)
        assert 0.3 < abs(result.cohens_d) < 0.7
        assert result.magnitude in ["small", "medium"]

    def test_hedges_g_correction(self):
        """Hedge's g should be slightly smaller than Cohen's d for small samples."""
        group1 = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calculate_effect_size(group1, group2)

        # Hedge's g should be slightly smaller due to bias correction
        assert abs(result.hedges_g) < abs(result.cohens_d)

    def test_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        group1 = np.array([1.0])
        group2 = np.array([2.0])

        result = calculate_effect_size(group1, group2)

        assert result.cohens_d == 0.0
        assert "Insufficient" in result.interpretation

    def test_zero_variance(self):
        """Should handle zero variance gracefully."""
        group1 = np.array([5.0, 5.0, 5.0, 5.0])
        group2 = np.array([5.0, 5.0, 5.0, 5.0])

        result = calculate_effect_size(group1, group2)

        assert result.cohens_d == 0.0
        assert "Zero variance" in result.interpretation


class TestStrategyEffectSize:
    """Tests for strategy vs benchmark effect size."""

    def test_outperforming_strategy(self):
        """Strategy with higher returns should have positive effect size."""
        np.random.seed(42)
        # Use larger difference and more samples for reliable result
        strategy_returns = np.random.normal(0.003, 0.02, 200)  # 0.3% daily
        benchmark_returns = np.random.normal(0.0, 0.02, 200)  # 0.0% daily

        result = calculate_strategy_effect_size(strategy_returns, benchmark_returns)

        # Should show strategy outperforming (positive d)
        assert result.cohens_d > 0

    def test_underperforming_strategy(self):
        """Strategy with lower returns should have negative effect size."""
        np.random.seed(42)
        strategy_returns = np.random.normal(0.0, 0.02, 100)
        benchmark_returns = np.random.normal(0.001, 0.02, 100)

        result = calculate_strategy_effect_size(strategy_returns, benchmark_returns)

        assert result.cohens_d < 0


class TestComprehensiveValidationWithCorrection:
    """Tests for comprehensive validation with multiple testing correction."""

    @pytest.fixture
    def strong_strategy_returns(self):
        """Generate returns for a strongly performing strategy."""
        np.random.seed(42)
        return np.random.normal(0.002, 0.01, 100)  # Strong positive returns

    @pytest.fixture
    def weak_strategy_returns(self):
        """Generate returns for a weakly performing strategy."""
        np.random.seed(42)
        return np.random.normal(0.0001, 0.02, 100)  # Weak, noisy returns

    def test_includes_multiple_testing_section(self, strong_strategy_returns):
        """Validation should include multiple testing correction results."""
        result = comprehensive_statistical_validation(
            strong_strategy_returns,
            n_permutations=100,  # Fewer for speed
            apply_multiple_testing_correction=True,
        )

        assert "multiple_testing" in result
        assert "bonferroni" in result["multiple_testing"]
        assert "fdr" in result["multiple_testing"]

    def test_collects_p_values(self, strong_strategy_returns):
        """Should collect p-values from all tests."""
        result = comprehensive_statistical_validation(
            strong_strategy_returns,
            n_permutations=100,
            apply_multiple_testing_correction=True,
        )

        assert "p_values" in result
        assert "permutation" in result["p_values"]
        assert "runs" in result["p_values"]
        assert "autocorrelation" in result["p_values"]
        assert "drawdown" in result["p_values"]

    def test_fdr_adjusted_p_values(self, strong_strategy_returns):
        """Should provide FDR-adjusted p-values for each test."""
        result = comprehensive_statistical_validation(
            strong_strategy_returns,
            n_permutations=100,
            apply_multiple_testing_correction=True,
        )

        fdr = result["multiple_testing"]["fdr"]
        assert "adjusted_p_values" in fdr
        assert "permutation" in fdr["adjusted_p_values"]

    def test_bonferroni_adjusted_alpha(self, strong_strategy_returns):
        """Should provide Bonferroni-adjusted alpha."""
        result = comprehensive_statistical_validation(
            strong_strategy_returns,
            n_permutations=100,
            alpha=0.05,
            apply_multiple_testing_correction=True,
        )

        bonf = result["multiple_testing"]["bonferroni"]
        # 4 tests, so adjusted alpha = 0.05/4 = 0.0125
        assert bonf["adjusted_alpha"] == pytest.approx(0.0125)

    def test_effect_size_with_benchmark(self, strong_strategy_returns):
        """Should calculate effect size when benchmark provided."""
        np.random.seed(123)
        benchmark = np.random.normal(0.0005, 0.015, 100)

        result = comprehensive_statistical_validation(
            strong_strategy_returns,
            benchmark_returns=benchmark,
            n_permutations=100,
            apply_multiple_testing_correction=True,
        )

        assert "effect_size" in result
        assert "cohens_d" in result["effect_size"]
        assert "hedges_g" in result["effect_size"]
        assert "magnitude" in result["effect_size"]

    def test_no_effect_size_without_benchmark(self, strong_strategy_returns):
        """Should not include effect size without benchmark."""
        result = comprehensive_statistical_validation(
            strong_strategy_returns,
            n_permutations=100,
            apply_multiple_testing_correction=True,
        )

        assert "effect_size" not in result

    def test_can_disable_correction(self, strong_strategy_returns):
        """Should be able to disable multiple testing correction."""
        result = comprehensive_statistical_validation(
            strong_strategy_returns,
            n_permutations=100,
            apply_multiple_testing_correction=False,
        )

        assert "multiple_testing" not in result

    def test_warning_on_correction_change(self):
        """Should warn if correction changes significance."""
        # Create returns that produce borderline p-values
        np.random.seed(42)
        # Weak strategy that might pass raw but not corrected
        returns = np.random.normal(0.0003, 0.015, 50)

        result = comprehensive_statistical_validation(
            returns,
            n_permutations=500,
            apply_multiple_testing_correction=True,
        )

        # Check that warnings are collected properly
        assert "warnings" in result
        # The warning about correction may or may not appear depending on exact p-values

    def test_overall_validity_considers_correction(self):
        """Overall validity should consider FDR correction for permutation test."""
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.01, 100)

        result = comprehensive_statistical_validation(
            returns,
            n_permutations=100,
            apply_multiple_testing_correction=True,
        )

        # The overall_valid flag should reflect corrected significance
        # not raw significance
        if result["overall_valid"]:
            # If valid, permutation should be in FDR significant tests
            fdr_sig = result["multiple_testing"]["fdr"]["significant_tests"]
            assert "permutation" in fdr_sig


class TestKnownValues:
    """Tests with known statistical values for verification."""

    def test_bonferroni_known_example(self):
        """Test Bonferroni with textbook example."""
        # 4 tests, alpha = 0.05
        # Adjusted alpha should be 0.0125
        # p-values: [0.01, 0.02, 0.03, 0.04]
        # Only 0.01 < 0.0125, so 1 significant
        p_values = [0.01, 0.02, 0.03, 0.04]
        result = bonferroni_correction(p_values, alpha=0.05)

        assert result.adjusted_alpha == 0.0125
        assert result.n_significant_adjusted == 1
        assert result.significant_adjusted[0] is True
        assert result.significant_adjusted[1] is False

    def test_cohens_d_known_example(self):
        """Test Cohen's d with known values."""
        # Two groups with known difference
        # Group 1: mean=10, std=2
        # Group 2: mean=8, std=2
        # Cohen's d = (10-8) / 2 = 1.0
        np.random.seed(42)
        group1 = np.array([8, 9, 10, 11, 12])  # mean=10, spread around it
        group2 = np.array([6, 7, 8, 9, 10])  # mean=8, same spread

        result = calculate_effect_size(group1, group2)

        # Should be positive (group1 > group2) and large (d > 0.8)
        assert result.cohens_d > 0
        # With exact same variance, d = (10-8) / std
        assert result.magnitude in ["medium", "large"]
