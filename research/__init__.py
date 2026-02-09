"""
Research Package - Research/Production Code Separation

This package provides tools for managing the research-to-production pipeline:

1. ResearchRegistry: Track experiments from inception to production
2. Validation gates: Ensure models pass rigorous testing before deployment
3. Version control: Track git commits and config hashes for reproducibility

Usage:
    from research.research_registry import ResearchRegistry

    # Initialize registry
    registry = ResearchRegistry()

    # Create experiment
    exp_id = registry.create_experiment(
        name="enhanced_momentum",
        description="Momentum with sector-relative component",
        author="quant_team",
    )

    # Record results as you progress
    registry.record_backtest_results(exp_id, backtest_results)
    registry.record_validation_results(exp_id, walkforward_results)
    registry.record_paper_results(exp_id, paper_results)

    # Check promotion readiness
    if registry.is_promotion_ready(exp_id):
        registry.promote_to_production(exp_id)
    else:
        blockers = registry.get_promotion_blockers(exp_id)
        print(f"Cannot promote: {blockers}")
"""

from research.research_registry import (
    Experiment,
    ExperimentStatus,
    ResearchRegistry,
    ValidationGate,
    ValidationResult,
    print_experiment_summary,
)

__all__ = [
    "ResearchRegistry",
    "Experiment",
    "ExperimentStatus",
    "ValidationGate",
    "ValidationResult",
    "print_experiment_summary",
]
