from datetime import datetime

from engine.validated_backtest import ValidatedBacktestResult, format_validated_backtest_report


def test_report_without_walk_forward_or_significance():
    result = ValidatedBacktestResult(
        strategy_name="TestStrategy",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        total_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        num_trades=0,
        win_rate=0.0,
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

    report = format_validated_backtest_report(result)
    assert "WALK-FORWARD" not in report
    assert "Significance" not in report
