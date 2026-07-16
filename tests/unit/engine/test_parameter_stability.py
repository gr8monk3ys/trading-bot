"""Tests for engine.parameter_stability.

Covers ParameterStabilityAnalyzer.analyze / _analyze_parameter / _generate_report
and the standalone quick_stability_check helper, exercising real computation
(no mocking of the logic under test) with fake async backtest functions.
"""

import numpy as np
import pytest

from engine.parameter_stability import (
    ParameterSensitivity,
    ParameterStabilityAnalyzer,
    StabilityReport,
    quick_stability_check,
)


# ---------------------------------------------------------------------------
# Fake backtest functions
# ---------------------------------------------------------------------------


async def _stable_backtest_fn(params, **_kwargs):
    """Performance barely changes regardless of parameter perturbation."""
    return {"sharpe_ratio": 1.0}


async def _unstable_backtest_fn(params, **_kwargs):
    """Performance swings wildly based on the perturbed 'window' param."""
    window = params.get("window", 20)
    # Big cliff effect: far from base (20) -> performance craters or spikes.
    deviation = abs(window - 20)
    return {"sharpe_ratio": max(-2.0, 2.0 - deviation * 0.5)}


async def _raises_for_low_values_fn(params, **_kwargs):
    """Raises an exception whenever the perturbed value drops below base."""
    value = params.get("threshold", 0)
    if value < 10:
        raise ValueError(f"backtest blew up for threshold={value}")
    return {"sharpe_ratio": 1.5}


async def _zero_base_performance_fn(params, **_kwargs):
    """Base backtest always returns 0 performance."""
    return {"sharpe_ratio": 0}


# ---------------------------------------------------------------------------
# ParameterStabilityAnalyzer.analyze - normal / stable path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_stable_strategy_reports_high_stability():
    analyzer = ParameterStabilityAnalyzer()

    report = await analyzer.analyze(
        backtest_fn=_stable_backtest_fn,
        base_params={"rsi_period": 14, "stop_loss": 0.05},
        strategy_name="StableStrat",
    )

    assert isinstance(report, StabilityReport)
    assert report.strategy_name == "StableStrat"
    assert set(report.parameter_sensitivities.keys()) == {"rsi_period", "stop_loss"}

    for sens in report.parameter_sensitivities.values():
        assert isinstance(sens, ParameterSensitivity)
        # Performance is identical (1.0) for every perturbation -> zero sensitivity.
        assert sens.sensitivity_score == pytest.approx(0.0)
        assert sens.stability_score == pytest.approx(1.0)

    assert report.overall_stability_score == pytest.approx(1.0)
    assert bool(report.is_stable) is True
    # No warnings triggered -> falls back to the "robust" message.
    assert report.warnings == ["No stability warnings - parameters appear robust"]
    assert report.recommendations == ["Strategy parameters appear stable for production use"]
    assert report.most_sensitive_parameter in {"rsi_period", "stop_loss"}
    assert report.least_sensitive_parameter in {"rsi_period", "stop_loss"}


# ---------------------------------------------------------------------------
# ParameterStabilityAnalyzer.analyze - unstable path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_unstable_strategy_reports_low_stability_and_warnings():
    analyzer = ParameterStabilityAnalyzer()

    report = await analyzer.analyze(
        backtest_fn=_unstable_backtest_fn,
        base_params={"window": 20},
        strategy_name="UnstableStrat",
    )

    sens = report.parameter_sensitivities["window"]
    # base_performance = 2.0 (deviation=0); perturbations of +-10%/20% on 20
    # produce large relative swings.
    assert sens.base_performance == pytest.approx(2.0)
    assert sens.sensitivity_score > 0
    assert sens.stability_score < 1.0

    assert report.overall_stability_score < analyzer.stability_threshold
    assert bool(report.is_stable) is False
    assert report.most_sensitive_parameter == "window"
    assert report.least_sensitive_parameter == "window"

    # If stability dropped below 0.5 for this single param, expect specific
    # warning/recommendation content; otherwise still expect an overfit note.
    assert any("overfit" in r.lower() for r in report.recommendations)


# ---------------------------------------------------------------------------
# ParameterStabilityAnalyzer._analyze_parameter - exception handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_parameter_backtest_exception_defaults_to_zero_performance():
    analyzer = ParameterStabilityAnalyzer()

    base_params = {"threshold": 15}
    base_performance = 1.5

    sensitivity = await analyzer._analyze_parameter(
        backtest_fn=_raises_for_low_values_fn,
        base_params=base_params,
        param_name="threshold",
        param_range=None,
        base_performance=base_performance,
    )

    # perturbations: -20% -> 12, -10% -> 13.5, +10% -> 16.5, +20% -> 18
    # all are >= 10 so none should raise here (raises only when < 10).
    assert len(sensitivity.test_values) == 4
    assert all(p == pytest.approx(1.5) for p in sensitivity.performance_values)

    # Now use a base value that produces perturbations below 10 to trigger
    # the exception-catching path.
    sensitivity_low = await analyzer._analyze_parameter(
        backtest_fn=_raises_for_low_values_fn,
        base_params={"threshold": 10},
        param_name="threshold",
        param_range=None,
        base_performance=1.5,
    )
    # -20% -> 8, -10% -> 9 (both < 10 -> raise -> performance 0)
    # +10% -> 11, +20% -> 12 (>= 10 -> performance 1.5)
    assert sensitivity_low.performance_values.count(0) == 2
    assert sensitivity_low.performance_values.count(1.5) == 2
    # Should not have crashed and should still produce a real interpretation.
    assert sensitivity_low.parameter_name == "threshold"
    assert isinstance(sensitivity_low.interpretation, str)
    assert sensitivity_low.interpretation


# ---------------------------------------------------------------------------
# Integer param handling + clamping/collapse-to-base skip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_parameter_int_rounding_and_collapse_skip():
    analyzer = ParameterStabilityAnalyzer(perturbation_pcts=[-0.01, 0.01])

    async def _fn(params, **_kwargs):
        return {"sharpe_ratio": 1.0}

    # base_value=14 (int); perturbations of +-1% round back to 14 -> skipped.
    sensitivity = await analyzer._analyze_parameter(
        backtest_fn=_fn,
        base_params={"rsi_period": 14},
        param_name="rsi_period",
        param_range=None,
        base_performance=1.0,
    )

    assert sensitivity.test_values == []
    assert sensitivity.performance_values == []
    # no valid results -> the "could not analyze" fallback branch
    assert sensitivity.sensitivity_score == 1.0
    assert sensitivity.max_degradation == 1.0
    assert sensitivity.stability_score == 0.0
    assert "Could not analyze" in sensitivity.interpretation


@pytest.mark.asyncio
async def test_analyze_parameter_param_range_clamping():
    analyzer = ParameterStabilityAnalyzer(perturbation_pcts=[-0.20, -0.10, 0.10, 0.20])

    async def _fn(params, **_kwargs):
        return {"sharpe_ratio": 1.0}

    # base_value=10, range clamps everything below 9 up to 9, and -10%/-20%
    # perturbations (9.0, 8.0) both clamp to 9 -> only one distinct test value.
    sensitivity = await analyzer._analyze_parameter(
        backtest_fn=_fn,
        base_params={"stop_loss": 10.0},
        param_name="stop_loss",
        param_range=(9.0, 20.0),
        base_performance=1.0,
    )

    # -20% -> 8.0 clamped to 9.0; -10% -> 9.0 clamped to 9.0 (dup, still kept,
    # since not equal to base 10.0); +10% -> 11.0; +20% -> 12.0
    assert 9.0 in sensitivity.test_values
    assert sensitivity.test_values.count(9.0) == 2
    assert 11.0 in sensitivity.test_values
    assert 12.0 in sensitivity.test_values
    assert len(sensitivity.test_values) == 4


@pytest.mark.asyncio
async def test_analyze_parameter_int_clamp_collapses_to_base_is_skipped():
    analyzer = ParameterStabilityAnalyzer(perturbation_pcts=[-0.20, -0.10, 0.10, 0.20])

    async def _fn(params, **_kwargs):
        return {"sharpe_ratio": 1.0}

    # base_value=14 (int), range clamps low end to 13; -20% -> 11.2 clamp to
    # 13, round -> 13 (differs from base 14, kept). -10% -> 12.6 clamp to 13,
    # round -> 13 (kept, duplicate). +10% -> 15.4 round -> 15. +20% -> 16.8
    # round -> 17.
    sensitivity = await analyzer._analyze_parameter(
        backtest_fn=_fn,
        base_params={"rsi_period": 14},
        param_name="rsi_period",
        param_range=(13, 30),
        base_performance=1.0,
    )

    assert 14 not in sensitivity.test_values
    assert all(isinstance(v, int) for v in sensitivity.test_values)
    assert len(sensitivity.test_values) == 4


# ---------------------------------------------------------------------------
# base_performance == 0 fallback (division-by-zero guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_zero_base_performance_forces_zero_stability_no_crash():
    analyzer = ParameterStabilityAnalyzer()

    report = await analyzer.analyze(
        backtest_fn=_zero_base_performance_fn,
        base_params={"rsi_period": 14},
        strategy_name="ZeroBaseStrat",
    )

    sens = report.parameter_sensitivities["rsi_period"]
    assert sens.base_performance == 0
    assert sens.sensitivity_score == 1.0
    assert sens.max_degradation == 1.0
    assert sens.stability_score == 0.0
    assert "Could not analyze" in sens.interpretation

    assert report.overall_stability_score == pytest.approx(0.0)
    assert bool(report.is_stable) is False


# ---------------------------------------------------------------------------
# _generate_report edge cases
# ---------------------------------------------------------------------------


def test_generate_report_empty_sensitivities_early_return():
    analyzer = ParameterStabilityAnalyzer()

    report = analyzer._generate_report("EmptyStrat", {}, base_performance=1.0)

    assert report.strategy_name == "EmptyStrat"
    assert report.parameter_sensitivities == {}
    assert report.overall_stability_score == 0
    assert report.most_sensitive_parameter == "N/A"
    assert report.least_sensitive_parameter == "N/A"
    assert report.is_stable is False
    assert report.warnings == ["No parameters analyzed"]
    assert report.recommendations == ["Provide parameters to analyze"]


def test_generate_report_most_and_least_sensitive_selection():
    analyzer = ParameterStabilityAnalyzer()

    sensitivities = {
        "very_sensitive": ParameterSensitivity(
            parameter_name="very_sensitive",
            base_value=10,
            test_values=[8, 9, 11, 12],
            performance_values=[0.1, 0.2, 0.3, 0.2],
            base_performance=1.0,
            sensitivity_score=0.9,
            max_degradation=0.9,
            stability_score=0.1,
            interpretation="very_sensitive is UNSTABLE",
        ),
        "very_stable": ParameterSensitivity(
            parameter_name="very_stable",
            base_value=5,
            test_values=[4, 4.5, 5.5, 6],
            performance_values=[0.99, 1.0, 1.0, 1.01],
            base_performance=1.0,
            sensitivity_score=0.01,
            max_degradation=0.01,
            stability_score=0.99,
            interpretation="very_stable is STABLE",
        ),
    }

    report = analyzer._generate_report("MixedStrat", sensitivities, base_performance=1.0)

    assert report.most_sensitive_parameter == "very_sensitive"
    assert report.least_sensitive_parameter == "very_stable"
    assert report.overall_stability_score == pytest.approx((0.1 + 0.99) / 2)
    assert bool(report.is_stable) is False  # (0.1+0.99)/2 = 0.545 < 0.7 threshold

    assert any("very_sensitive" in w for w in report.warnings)
    assert any(">50% performance degradation" in w for w in report.warnings)
    assert any("very_sensitive" in r for r in report.recommendations)
    assert any("overfit" in r.lower() for r in report.recommendations)


def test_generate_report_print_summary_does_not_raise(capsys):
    analyzer = ParameterStabilityAnalyzer()
    sensitivities = {
        "param_a": ParameterSensitivity(
            parameter_name="param_a",
            base_value=10,
            test_values=[9, 11],
            performance_values=[1.0, 1.0],
            base_performance=1.0,
            sensitivity_score=0.0,
            max_degradation=0.0,
            stability_score=1.0,
            interpretation="param_a is STABLE",
        )
    }
    report = analyzer._generate_report("PrintStrat", sensitivities, base_performance=1.0)

    analyzer._print_summary(report)

    captured = capsys.readouterr()
    assert "PARAMETER STABILITY SUMMARY" in captured.out
    assert "param_a" in captured.out


# ---------------------------------------------------------------------------
# quick_stability_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_stability_check_normal_case_varying_returns():
    returns_by_param = {
        "rsi=10": np.array([0.01, 0.02, -0.01, 0.015, 0.005]),
        "rsi=14": np.array([0.012, 0.018, -0.008, 0.014, 0.006]),
        "rsi=20": np.array([0.011, 0.019, -0.009, 0.013, 0.007]),
    }

    result = await quick_stability_check(returns_by_param)

    assert set(result["metrics"].keys()) == set(returns_by_param.keys())
    assert isinstance(result["mean_metric"], (float, np.floating))
    assert result["std_metric"] >= 0
    assert result["coefficient_of_variation"] >= 0
    assert 0 <= result["stability_score"] <= 1
    assert isinstance(result["is_stable"], (bool, np.bool_))
    assert result["is_stable"] == (result["stability_score"] >= 0.7)


@pytest.mark.asyncio
async def test_quick_stability_check_zero_variance_per_param_uses_metric_guard():
    # Every returns array has zero standard deviation, so the default
    # metric_fn's `if np.std(r) > 0 else 0` guard forces each metric to 0.
    returns_by_param = {
        "a": np.array([0.01, 0.01, 0.01]),
        "b": np.array([0.02, 0.02, 0.02]),
    }

    result = await quick_stability_check(returns_by_param)

    assert result["metrics"] == {"a": 0, "b": 0}
    assert result["mean_metric"] == pytest.approx(0.0)
    assert result["std_metric"] == pytest.approx(0.0)
    # mean_metric == 0 -> cv falls back to float('inf') per the guard.
    assert result["coefficient_of_variation"] == float("inf")
    # stability = max(0, 1 - inf) = 0
    assert result["stability_score"] == 0
    assert result["is_stable"] is False


@pytest.mark.asyncio
async def test_quick_stability_check_custom_metric_fn_zero_mean_across_params():
    # Custom metric_fn returns values that average to zero but aren't all
    # identical, so std_metric > 0 while mean_metric == 0, still hitting the
    # `mean_metric != 0` False branch (inf guard) without needing std to be 0.
    returns_by_param = {
        "x": np.array([1.0, 2.0, 3.0]),
        "y": np.array([1.0, 2.0, 3.0]),
    }

    def metric_fn(r):
        # Symmetric metric values around zero: e.g. +1 for 'x', -1 for 'y'
        return float(r[0]) - 2.0 if r is returns_by_param["x"] else -(float(r[0]) - 2.0)

    result = await quick_stability_check(returns_by_param, metric_fn=metric_fn)

    assert result["mean_metric"] == pytest.approx(0.0)
    assert result["std_metric"] > 0
    assert result["coefficient_of_variation"] == float("inf")
    assert result["stability_score"] == 0
    assert result["is_stable"] is False


@pytest.mark.asyncio
async def test_quick_stability_check_single_param_no_variation_is_stable():
    # A single param entry with itself equal to mean has std_metric == 0
    # and mean_metric != 0, giving cv == 0 -> perfectly stable.
    returns_by_param = {
        "only": np.array([0.01, 0.02, 0.03, -0.01, 0.015]),
    }

    result = await quick_stability_check(returns_by_param)

    assert result["std_metric"] == pytest.approx(0.0)
    assert result["coefficient_of_variation"] == pytest.approx(0.0)
    assert result["stability_score"] == pytest.approx(1.0)
    assert bool(result["is_stable"]) is True
