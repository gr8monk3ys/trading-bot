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


def test_insights_flag_minimal_profitability():
    insights = PerformanceMetrics()._generate_insights(
        {
            "total_return": 0.03,
            "max_drawdown": 0.1,
            "sharpe_ratio": 1.1,
            "win_rate": 0.5,
            "profit_factor": 1.2,
        }
    )
    assert any("minimal profitability" in msg.lower() for msg in insights)


def test_significance_warns_on_high_variance_and_outliers():
    # 54 small wins, 5 small losses, one win 200x the rest: the mean is
    # carried by a single trade.
    trades = [{"pnl": 1.0}] * 54 + [{"pnl": -1.0}] * 5 + [{"pnl": 200.0}]
    result = PerformanceMetrics().calculate_significance(trades, min_trades=50)
    warnings = " | ".join(result["warnings"])
    assert "High variance relative to mean" in warnings
    assert "outliers" in warnings


def test_outlier_dependency_is_zero_for_tiny_or_all_losing_samples():
    metrics = PerformanceMetrics()
    assert metrics._check_outlier_dependency(np.array([1.0, 2.0, 3.0])) == 0.0
    assert metrics._check_outlier_dependency(np.array([-1.0] * 20)) == 0.0
