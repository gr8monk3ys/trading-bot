from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from engine.performance_metrics import (
    EffectSizeResult,
    PerformanceMetrics,
    apply_bonferroni_correction,
    apply_fdr_correction,
    calculate_adjusted_significance,
    calculate_cohens_d,
    calculate_effect_size,
    calculate_hedges_g,
)


def test_correction_helpers_and_adjusted_significance_modes():
    assert apply_bonferroni_correction([]) == []
    assert apply_fdr_correction([]) == []

    fdr_result = calculate_adjusted_significance(0.01, n_tests=4, method="fdr")
    assert fdr_result["adjusted_p_value"] == pytest.approx(0.01)

    with pytest.raises(ValueError, match="Unknown correction method"):
        calculate_adjusted_significance(0.01, n_tests=4, method="unknown")


def test_effect_size_helpers_and_all_interpretation_branches(monkeypatch):
    assert calculate_cohens_d(np.array([])) == 0.0
    assert calculate_cohens_d(np.array([1.0, 1.0, 1.0])) == 0.0
    assert calculate_cohens_d(np.array([0.1, 0.2, 0.4, 0.3])) > 0

    assert calculate_hedges_g(np.array([0.1, 0.2, 0.3])) == 0.0
    assert calculate_hedges_g(np.array([0.1, 0.2, 0.3, 0.4])) != 0.0

    insufficient = calculate_effect_size(np.array([0.1, 0.2, 0.3]))
    assert insufficient.interpretation.startswith("Insufficient")

    for d in (0.1, 0.3, 0.7, 1.1):
        monkeypatch.setattr(
            "engine.performance_metrics.calculate_cohens_d",
            lambda _returns, _population_mean=0.0, _d=d: _d,
        )
        monkeypatch.setattr(
            "engine.performance_metrics.calculate_hedges_g",
            lambda _returns, _population_mean=0.0, _d=d: _d,
        )
        result = calculate_effect_size(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert "effect" in result.interpretation.lower()
        assert result.confidence_interval[0] <= result.confidence_interval[1]


def test_calculate_metrics_exception_and_minimal_profitability_insight():
    metrics_calc = PerformanceMetrics()
    result = metrics_calc.calculate_metrics(
        {
            "equity_curve": [100000, "bad"],
            "trades": [{"pnl": 1}],
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 2),
        }
    )
    assert result["total_return"] == 0

    insights = metrics_calc._generate_insights(
        {
            "total_return": 0.03,
            "max_drawdown": 0.1,
            "sharpe_ratio": 1.1,
            "win_rate": 0.5,
            "profit_factor": 1.2,
        }
    )
    assert any("minimal profitability" in msg.lower() for msg in insights)


def test_calculate_significance_high_variance_and_outlier_warnings():
    metrics_calc = PerformanceMetrics()
    trades = [{"pnl": 1.0}] * 54 + [{"pnl": -1.0}] * 5 + [{"pnl": 200.0}]

    result = metrics_calc.calculate_significance(trades, min_trades=50)
    all_warnings = " | ".join(result["warnings"])

    assert "High variance relative to mean" in all_warnings
    assert "outliers" in all_warnings


def test_check_outlier_dependency_edge_cases(monkeypatch):
    metrics_calc = PerformanceMetrics()

    assert metrics_calc._check_outlier_dependency(np.array([1.0, 2.0, 3.0])) == 0.0
    assert metrics_calc._check_outlier_dependency(np.array([-1.0] * 20)) == 0.0

    monkeypatch.setattr("engine.performance_metrics.np.sum", lambda _wins: 0.0)
    assert metrics_calc._check_outlier_dependency(np.array([1.0] * 20)) == 0.0


def test_validate_backtest_results_and_batch_significance_bonferroni(monkeypatch):
    metrics_calc = PerformanceMetrics()

    monkeypatch.setattr(
        metrics_calc,
        "calculate_metrics",
        lambda _result: {
            "sharpe_ratio": 4.1,
            "total_return": 0.25,
            "max_drawdown": 0.005,
            "win_rate": 0.9,
            "profit_factor": 2.5,
        },
    )
    monkeypatch.setattr(
        metrics_calc,
        "calculate_significance",
        lambda _trades, _min_trades, _confidence_level=0.95: {
            "is_significant": True,
            "warnings": [],
            "p_value": 0.01,
        },
    )

    validation = metrics_calc.validate_backtest_results({"trades": [{"pnl": 1.0}] * 60}, min_trades=50)
    warning_blob = " | ".join(validation["warnings"])
    assert "unusually high" in warning_blob
    assert "suspiciously high" in warning_blob
    assert "unrealistic" in warning_blob

    batch = metrics_calc.batch_significance_test(
        strategy_results={
            "A": [{"pnl": 1.0}] * 60,
            "B": [{"pnl": 0.5}] * 60,
        },
        min_trades=50,
        correction_method="bonferroni",
    )
    assert batch["correction_method"] == "bonferroni"


def test_calculate_comprehensive_significance_false_positive_warning(monkeypatch):
    metrics_calc = PerformanceMetrics()
    trades = [{"pnl": 0.02}] * 59 + [{"pnl": "bad"}]

    monkeypatch.setattr("engine.performance_metrics.stats.ttest_1samp", lambda _r, _m: (2.5, 0.02))
    result = metrics_calc.calculate_comprehensive_significance(
        trades,
        n_total_tests=10,
        min_trades=50,
        confidence_level=0.95,
        correction_method="bonferroni",
    )
    assert result["mean_return"] >= 0
    assert any("false positive" in warning.lower() for warning in result["warnings"])


def test_calculate_comprehensive_significance_negligible_and_outlier_warnings(monkeypatch):
    metrics_calc = PerformanceMetrics()
    trades = [{"pnl": 0.05}] * 60

    monkeypatch.setattr("engine.performance_metrics.stats.ttest_1samp", lambda _r, _m: (5.0, 0.0002))
    monkeypatch.setattr(
        "engine.performance_metrics.calculate_effect_size",
        lambda _returns, _population_mean=0.0, _confidence_level=0.95: EffectSizeResult(
            cohens_d=0.1,
            hedges_g=0.1,
            interpretation="small effect",
            confidence_interval=(0.01, 0.2),
        ),
    )
    monkeypatch.setattr(metrics_calc, "_check_outlier_dependency", lambda _returns: 0.8)

    result = metrics_calc.calculate_comprehensive_significance(
        trades,
        n_total_tests=1,
        min_trades=50,
        confidence_level=0.95,
        correction_method="bonferroni",
    )
    warning_blob = " | ".join(result["warnings"])
    assert "negligible effect size" in warning_blob
    assert "outliers" in warning_blob


def test_calculate_comprehensive_significance_large_effect_warning(monkeypatch):
    metrics_calc = PerformanceMetrics()
    trades = [{"pnl": 0.1}] * 60

    monkeypatch.setattr("engine.performance_metrics.stats.ttest_1samp", lambda _r, _m: (5.0, 0.0002))
    monkeypatch.setattr(
        "engine.performance_metrics.calculate_effect_size",
        lambda _returns, _population_mean=0.0, _confidence_level=0.95: EffectSizeResult(
            cohens_d=1.2,
            hedges_g=1.1,
            interpretation="large effect",
            confidence_interval=(0.8, 1.4),
        ),
    )
    monkeypatch.setattr(metrics_calc, "_check_outlier_dependency", lambda _returns: 0.0)

    result = metrics_calc.calculate_comprehensive_significance(
        trades,
        n_total_tests=1,
        min_trades=50,
        confidence_level=0.95,
        correction_method="bonferroni",
    )
    assert any("Large effect size" in warning for warning in result["warnings"])
