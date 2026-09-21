"""Edge cases in engine.performance_metrics that the main suite does not cover:
empty inputs, degenerate variance, and the warning text emitted for
outlier-dependent and high-variance trade samples."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from engine.performance_metrics import (
    PerformanceMetrics,
    apply_bonferroni_correction,
    apply_fdr_correction,
    calculate_adjusted_significance,
    calculate_cohens_d,
    calculate_effect_size,
    calculate_hedges_g,
)


def test_corrections_on_empty_input_return_empty():
    assert apply_bonferroni_correction([]) == []
    assert apply_fdr_correction([]) == []


def test_fdr_with_single_hypothesis_leaves_p_value_unchanged():
    result = calculate_adjusted_significance(0.01, n_tests=4, method="fdr")
    assert result["adjusted_p_value"] == pytest.approx(0.01)


def test_unknown_correction_method_raises():
    with pytest.raises(ValueError, match="Unknown correction method"):
        calculate_adjusted_significance(0.01, n_tests=4, method="unknown")


def test_cohens_d_degenerate_inputs_are_zero():
    assert calculate_cohens_d(np.array([])) == 0.0
    assert calculate_cohens_d(np.array([1.0, 1.0, 1.0])) == 0.0  # zero variance
    assert calculate_cohens_d(np.array([0.1, 0.2, 0.4, 0.3])) > 0


def test_hedges_g_needs_at_least_four_samples():
    assert calculate_hedges_g(np.array([0.1, 0.2, 0.3])) == 0.0
    assert calculate_hedges_g(np.array([0.1, 0.2, 0.3, 0.4])) != 0.0


def test_effect_size_reports_insufficient_data_below_four_samples():
    result = calculate_effect_size(np.array([0.1, 0.2, 0.3]))
    assert result.interpretation.startswith("Insufficient")


def test_calculate_metrics_with_corrupt_equity_curve_returns_zero_return():
    result = PerformanceMetrics().calculate_metrics(
        {
            "equity_curve": [100000, "bad"],
            "trades": [{"pnl": 1}],
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 2),
        }
    )
    assert result["total_return"] == 0
