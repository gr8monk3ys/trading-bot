from datetime import datetime

import pandas as pd

from engine.validated_backtest import (
    ValidatedBacktestResult,
    ValidatedBacktestRunner,
    format_validated_backtest_report,
)


class DummyPermResult:
    def __init__(self, p_value: float, n_permutations: int = 1000):
        self.p_value = p_value
        self.n_permutations = n_permutations
        self.is_significant = p_value < 0.05


def _base_result(**overrides) -> ValidatedBacktestResult:
    base = ValidatedBacktestResult(
        strategy_name="TestStrategy",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        total_return=0.25,
        sharpe_ratio=1.2,
        max_drawdown=-0.08,
        num_trades=120,
        win_rate=0.55,
        walk_forward_validated=True,
        overfit_warning=False,
        overfit_ratio=1.1,
        is_return=0.18,
        oos_return=0.14,
        consistency_score=0.7,
        walk_forward_folds=[],
        regime_metrics={},
        statistically_significant=True,
        p_value=0.01,
        equity_curve=None,
        daily_returns=pd.Series([0.001] * 40),
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_validation_gates_pass_with_mocked_permutation(monkeypatch):
    def _mock_perm(*_args, **_kwargs):
        return DummyPermResult(p_value=0.001)

    monkeypatch.setattr("engine.validated_backtest.permutation_test_returns", _mock_perm)

    runner = ValidatedBacktestRunner(broker=None)
    result = _base_result(
        equity_curve=pd.Series([100000, 101000, 103000, 104000]),
        daily_returns=pd.Series([0.01] * 20),
    )

    gates, eligible = runner._evaluate_validation_gates(result)

    assert eligible is True
    assert gates["blockers"] == []
    assert gates["permutation"]["tests"]["mean"]["is_significant"] is True
    assert gates["permutation"]["tests"]["sharpe"]["is_significant"] is True


def test_validation_gates_blocks_when_no_walk_forward(monkeypatch):
    def _mock_perm(*_args, **_kwargs):
        return DummyPermResult(p_value=0.001)

    monkeypatch.setattr("engine.validated_backtest.permutation_test_returns", _mock_perm)

    runner = ValidatedBacktestRunner(broker=None)
    result = _base_result(walk_forward_validated=False)

    gates, eligible = runner._evaluate_validation_gates(result)

    assert eligible is False
    assert "walk_forward_overfit" in gates["blockers"]
    assert "walk_forward_consistency" in gates["blockers"]


def test_validation_gates_fail_for_core_thresholds(monkeypatch):
    def _mock_perm(*_args, **_kwargs):
        return DummyPermResult(p_value=0.001)

    monkeypatch.setattr("engine.validated_backtest.permutation_test_returns", _mock_perm)

    runner = ValidatedBacktestRunner(broker=None)
    result = _base_result(
        num_trades=10,
        sharpe_ratio=0.1,
        max_drawdown=-0.4,
        win_rate=0.2,
        overfit_ratio=3.0,
        consistency_score=0.2,
        statistically_significant=False,
        p_value=0.5,
        equity_curve=pd.Series([100000, 99000, 98000, 97000]),
        daily_returns=pd.Series([-0.01] * 20),
    )

    gates, eligible = runner._evaluate_validation_gates(result)

    assert eligible is False
    assert "min_trades" in gates["blockers"]
    assert "min_sharpe" in gates["blockers"]
    assert "max_drawdown" in gates["blockers"]
    assert "min_win_rate" in gates["blockers"]
    assert "walk_forward_overfit" in gates["blockers"]
    assert "walk_forward_consistency" in gates["blockers"]
    assert "t_test" in gates["blockers"]


def test_permutation_test_error_when_insufficient_returns():
    runner = ValidatedBacktestRunner(broker=None)
    result = runner._calculate_permutation_tests(pd.Series([0.01] * 5))

    assert "error" in result


def test_format_report_handles_missing_data():
    result = _base_result(daily_returns=None, validation_gates={})
    report = format_validated_backtest_report(result)

    assert "VALIDATED BACKTEST REPORT" in report
