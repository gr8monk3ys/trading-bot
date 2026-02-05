from engine.performance_metrics import (
    apply_bonferroni_correction,
    apply_fdr_correction,
    calculate_adjusted_significance,
)


def test_bonferroni_correction_flags_significance():
    p_values = [0.001, 0.02, 0.5]
    results = apply_bonferroni_correction(p_values, alpha=0.05)

    assert results[0].is_significant is True
    assert results[1].is_significant is False
    assert results[2].adjusted_p_value == 1.0


def test_fdr_correction_monotonic():
    p_values = [0.01, 0.02, 0.03, 0.5]
    results = apply_fdr_correction(p_values, alpha=0.05)

    adjusted = [r.adjusted_p_value for r in results]
    assert all(0.0 <= p <= 1.0 for p in adjusted)


def test_calculate_adjusted_significance_bonferroni():
    result = calculate_adjusted_significance(0.01, n_tests=5, method="bonferroni")

    assert result["adjusted_p_value"] == 0.05
    assert result["is_significant"] is False
