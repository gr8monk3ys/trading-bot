"""
Parameter Stability Analysis

Tests how sensitive strategy performance is to parameter changes.
Strategies that are highly sensitive to exact parameter values are
likely overfit and won't perform well in live trading.

Key Concepts:
- Stable strategies: Performance degrades gracefully with parameter changes
- Unstable strategies: Small changes cause large performance swings
- Cliff effects: Performance falls off sharply at certain thresholds
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParameterSensitivity:
    """Sensitivity analysis for a single parameter."""

    parameter_name: str
    base_value: float
    test_values: List[float]
    performance_values: List[float]
    base_performance: float

    # Sensitivity metrics
    sensitivity_score: float  # Higher = more sensitive (bad)
    max_degradation: float  # Worst performance drop
    stability_score: float  # 0-1 score (higher = more stable)

    interpretation: str


@dataclass
class StabilityReport:
    """Complete parameter stability report for a strategy."""

    strategy_name: str
    parameter_sensitivities: Dict[str, ParameterSensitivity]

    # Overall metrics
    overall_stability_score: float  # 0-1 (higher = more stable)
    most_sensitive_parameter: str
    least_sensitive_parameter: str

    # Risk assessment
    is_stable: bool
    warnings: List[str]
    recommendations: List[str]


class ParameterStabilityAnalyzer:
    """
    Analyzes parameter stability for trading strategies.

    A strategy is considered stable if:
    1. Small parameter changes (±10%) cause small performance changes (<20%)
    2. There are no "cliff effects" where performance drops suddenly
    3. The optimal parameters aren't at extreme values

    Usage:
        analyzer = ParameterStabilityAnalyzer()
        report = await analyzer.analyze(
            backtest_fn=run_backtest,
            base_params={'rsi_period': 14, 'stop_loss': 0.05},
            param_ranges={'rsi_period': (7, 28), 'stop_loss': (0.02, 0.10)},
            **backtest_kwargs
        )
    """

    def __init__(
        self,
        perturbation_pcts: List[float] = None,
        metric: str = "sharpe_ratio",
        stability_threshold: float = 0.7,
    ):
        """
        Initialize parameter stability analyzer.

        Args:
            perturbation_pcts: Percentages to perturb parameters by
                              Default: [-20%, -10%, +10%, +20%]
            metric: Performance metric to track (default 'sharpe_ratio')
            stability_threshold: Minimum stability score to pass (default 0.7)
        """
        self.perturbation_pcts = perturbation_pcts or [-0.20, -0.10, 0.10, 0.20]
        self.metric = metric
        self.stability_threshold = stability_threshold

    async def analyze(
        self,
        backtest_fn: Callable,
        base_params: Dict[str, float],
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        strategy_name: str = "Strategy",
        **backtest_kwargs,
    ) -> StabilityReport:
        """
        Analyze parameter stability for a strategy.

        Args:
            backtest_fn: Async function that runs backtest
                        Must accept params dict and return results dict
            base_params: Base parameter values (optimal/chosen values)
            param_ranges: Optional min/max ranges for each parameter
                         Used to clamp perturbed values
            strategy_name: Name for reporting
            **backtest_kwargs: Additional args passed to backtest_fn

        Returns:
            StabilityReport with sensitivity analysis
        """
        print(f"\n{'='*60}")
        print(f"PARAMETER STABILITY ANALYSIS: {strategy_name}")
        print(f"{'='*60}")
        print(f"Base parameters: {base_params}")
        print(f"Testing perturbations: {[f'{p:+.0%}' for p in self.perturbation_pcts]}")

        # Run base backtest
        print("\nRunning base backtest...")
        base_result = await backtest_fn(params=base_params, **backtest_kwargs)
        base_performance = base_result.get(self.metric, 0)
        print(f"Base {self.metric}: {base_performance:.4f}")

        # Analyze each parameter
        sensitivities = {}

        for param_name, _base_value in base_params.items():
            print(f"\nAnalyzing sensitivity to: {param_name}")

            sensitivity = await self._analyze_parameter(
                backtest_fn=backtest_fn,
                base_params=base_params,
                param_name=param_name,
                param_range=param_ranges.get(param_name) if param_ranges else None,
                base_performance=base_performance,
                **backtest_kwargs,
            )

            sensitivities[param_name] = sensitivity
            print(f"  Stability score: {sensitivity.stability_score:.2f}")
            print(f"  Max degradation: {sensitivity.max_degradation:.1%}")

        # Generate report
        report = self._generate_report(strategy_name, sensitivities, base_performance)

        # Print summary
        self._print_summary(report)

        return report

    async def _analyze_parameter(
        self,
        backtest_fn: Callable,
        base_params: Dict[str, float],
        param_name: str,
        param_range: Optional[Tuple[float, float]],
        base_performance: float,
        **backtest_kwargs,
    ) -> ParameterSensitivity:
        """Analyze sensitivity to a single parameter."""
        base_value = base_params[param_name]
        test_values = []
        performance_values = []

        for pct in self.perturbation_pcts:
            # Calculate perturbed value
            perturbed_value = base_value * (1 + pct)

            # Clamp to range if provided
            if param_range:
                perturbed_value = max(param_range[0], min(param_range[1], perturbed_value))

            # Handle integer parameters
            if isinstance(base_value, int):
                perturbed_value = int(round(perturbed_value))

            # Skip if same as base (can happen with clamping)
            if perturbed_value == base_value:
                continue

            test_values.append(perturbed_value)

            # Run backtest with perturbed parameter
            test_params = base_params.copy()
            test_params[param_name] = perturbed_value

            try:
                result = await backtest_fn(params=test_params, **backtest_kwargs)
                performance = result.get(self.metric, 0)
            except Exception as e:
                logger.warning(f"Backtest failed for {param_name}={perturbed_value}: {e}")
                performance = 0

            performance_values.append(performance)
            print(f"    {param_name}={perturbed_value}: {self.metric}={performance:.4f}")

        # Calculate sensitivity metrics
        if not performance_values or base_performance == 0:
            return ParameterSensitivity(
                parameter_name=param_name,
                base_value=base_value,
                test_values=test_values,
                performance_values=performance_values,
                base_performance=base_performance,
                sensitivity_score=1.0,
                max_degradation=1.0,
                stability_score=0.0,
                interpretation=f"Could not analyze {param_name} - no valid results",
            )

        # Calculate relative performance changes
        rel_changes = [
            abs((p - base_performance) / base_performance) if base_performance != 0 else 0
            for p in performance_values
        ]

        # Sensitivity score: average relative change (higher = more sensitive)
        sensitivity_score = np.mean(rel_changes) if rel_changes else 0

        # Max degradation: worst case performance drop
        if base_performance > 0:
            degradations = [(base_performance - p) / base_performance for p in performance_values]
            max_degradation = max(degradations) if degradations else 0
        else:
            max_degradation = 0

        # Stability score: inverse of sensitivity (0-1, higher = more stable)
        # Transform: sensitivity of 0 -> stability of 1
        #           sensitivity of 1 -> stability of 0
        stability_score = max(0, 1 - sensitivity_score)

        # Generate interpretation
        if stability_score >= 0.8:
            interpretation = f"{param_name} is STABLE: ±20% changes cause only {sensitivity_score:.0%} performance variation."
        elif stability_score >= 0.5:
            interpretation = f"{param_name} is MODERATELY STABLE: Some sensitivity to parameter changes detected."
        else:
            interpretation = f"{param_name} is UNSTABLE: High sensitivity ({sensitivity_score:.0%} avg change). May be overfit to exact value."

        return ParameterSensitivity(
            parameter_name=param_name,
            base_value=base_value,
            test_values=test_values,
            performance_values=performance_values,
            base_performance=base_performance,
            sensitivity_score=sensitivity_score,
            max_degradation=max_degradation,
            stability_score=stability_score,
            interpretation=interpretation,
        )

    def _generate_report(
        self,
        strategy_name: str,
        sensitivities: Dict[str, ParameterSensitivity],
        base_performance: float,
    ) -> StabilityReport:
        """Generate stability report from parameter sensitivities."""
        if not sensitivities:
            return StabilityReport(
                strategy_name=strategy_name,
                parameter_sensitivities={},
                overall_stability_score=0,
                most_sensitive_parameter="N/A",
                least_sensitive_parameter="N/A",
                is_stable=False,
                warnings=["No parameters analyzed"],
                recommendations=["Provide parameters to analyze"],
            )

        # Calculate overall stability
        stability_scores = [s.stability_score for s in sensitivities.values()]
        overall_stability = np.mean(stability_scores)

        # Find most/least sensitive
        sorted_by_sensitivity = sorted(
            sensitivities.items(), key=lambda x: x[1].sensitivity_score, reverse=True
        )
        most_sensitive = sorted_by_sensitivity[0][0]
        least_sensitive = sorted_by_sensitivity[-1][0]

        # Determine if stable
        is_stable = overall_stability >= self.stability_threshold

        # Generate warnings
        warnings = []
        for name, sens in sensitivities.items():
            if sens.stability_score < 0.5:
                warnings.append(
                    f"Parameter '{name}' is highly sensitive - consider widening acceptable range"
                )
            if sens.max_degradation > 0.5:
                warnings.append(
                    f"Parameter '{name}' has >50% performance degradation risk at ±20%"
                )

        # Generate recommendations
        recommendations = []
        if not is_stable:
            recommendations.append(
                "Strategy may be overfit. Consider simplifying or using regularization."
            )
        if overall_stability < 0.5:
            recommendations.append(
                "High parameter sensitivity suggests overfitting. Walk-forward optimization recommended."
            )

        for name, sens in sensitivities.items():
            if sens.stability_score < 0.5:
                recommendations.append(
                    f"Review '{name}' parameter - may need constraint or removal"
                )

        if not warnings:
            warnings.append("No stability warnings - parameters appear robust")

        if not recommendations:
            recommendations.append("Strategy parameters appear stable for production use")

        return StabilityReport(
            strategy_name=strategy_name,
            parameter_sensitivities=sensitivities,
            overall_stability_score=overall_stability,
            most_sensitive_parameter=most_sensitive,
            least_sensitive_parameter=least_sensitive,
            is_stable=is_stable,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _print_summary(self, report: StabilityReport):
        """Print stability report summary."""
        print(f"\n{'='*60}")
        print("PARAMETER STABILITY SUMMARY")
        print(f"{'='*60}")

        print(f"\nOverall Stability Score: {report.overall_stability_score:.2f}")
        print(f"Status: {'✅ STABLE' if report.is_stable else '❌ UNSTABLE'}")

        print(f"\nMost sensitive parameter: {report.most_sensitive_parameter}")
        print(f"Least sensitive parameter: {report.least_sensitive_parameter}")

        print("\nParameter Stability Scores:")
        for name, sens in sorted(
            report.parameter_sensitivities.items(),
            key=lambda x: x[1].stability_score,
        ):
            bar = "█" * int(sens.stability_score * 20) + "░" * (20 - int(sens.stability_score * 20))
            print(f"  {name:20s} [{bar}] {sens.stability_score:.2f}")

        print("\nWarnings:")
        for w in report.warnings:
            print(f"  ⚠️  {w}")

        print("\nRecommendations:")
        for r in report.recommendations:
            print(f"  📋 {r}")

        print(f"\n{'='*60}\n")


async def quick_stability_check(
    returns_by_param: Dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray], float] = None,
) -> Dict[str, float]:
    """
    Quick stability check from pre-computed backtest results.

    Use when you've already run backtests with different parameters
    and want to assess stability without re-running.

    Args:
        returns_by_param: Dict mapping param description to returns array
                         e.g., {'rsi=10': returns1, 'rsi=14': returns2, ...}
        metric_fn: Function to compute metric from returns
                   Default: Sharpe ratio

    Returns:
        Dict with stability metrics
    """
    if metric_fn is None:
        def metric_fn(r):
            return np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0

    metrics = {k: metric_fn(v) for k, v in returns_by_param.items()}

    values = list(metrics.values())
    mean_metric = np.mean(values)
    std_metric = np.std(values)

    # Coefficient of variation (lower is more stable)
    cv = std_metric / abs(mean_metric) if mean_metric != 0 else float("inf")

    # Stability score
    stability = max(0, 1 - cv)

    return {
        "metrics": metrics,
        "mean_metric": mean_metric,
        "std_metric": std_metric,
        "coefficient_of_variation": cv,
        "stability_score": stability,
        "is_stable": stability >= 0.7,
    }
