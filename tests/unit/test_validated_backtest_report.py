from datetime import datetime

from engine.validated_backtest import ValidatedBacktestResult, format_validated_backtest_report


def test_format_report_with_permutation_error():
    result = ValidatedBacktestResult(
        strategy_name="TestStrategy",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        total_return=0.1,
        sharpe_ratio=1.0,
        max_drawdown=-0.05,
        num_trades=20,
        win_rate=0.5,
        walk_forward_validated=False,
        overfit_warning=False,
        overfit_ratio=1.0,
        is_return=0.0,
        oos_return=0.0,
        consistency_score=0.0,
        walk_forward_folds=[],
        regime_metrics={},
        statistically_significant=False,
        p_value=None,
    )
    result.validation_gates = {
        "blockers": [],
        "permutation": {"error": "Insufficient returns"},
    }

    report = format_validated_backtest_report(result)

    assert "VALIDATED BACKTEST REPORT" in report
    assert "PERMUTATION TESTS" not in report


def test_format_report_with_permutation_tests():
    result = ValidatedBacktestResult(
        strategy_name="TestStrategy",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        total_return=0.1,
        sharpe_ratio=1.0,
        max_drawdown=-0.05,
        num_trades=20,
        win_rate=0.5,
        walk_forward_validated=True,
        overfit_warning=False,
        overfit_ratio=1.0,
        is_return=0.05,
        oos_return=0.04,
        consistency_score=0.6,
        walk_forward_folds=[],
        regime_metrics={},
        statistically_significant=True,
        p_value=0.01,
    )
    result.validation_gates = {
        "blockers": [],
        "permutation": {
            "method": "bonferroni",
            "alpha": 0.05,
            "tests": {
                "mean": {"p_value": 0.01, "adjusted_p_value": 0.02, "is_significant": True},
                "sharpe": {"p_value": 0.02, "adjusted_p_value": 0.04, "is_significant": True},
            },
        },
    }

    report = format_validated_backtest_report(result)

    assert "PERMUTATION TESTS" in report
