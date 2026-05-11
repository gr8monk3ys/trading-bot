import numpy as np

from engine.performance_metrics import PerformanceMetrics


def test_bootstrap_sharpe_ci_short_returns():
    metrics_calc = PerformanceMetrics()
    ci = metrics_calc._bootstrap_sharpe_ci(np.array([0.01, 0.02, 0.01]))

    assert ci == (0, 0)


def test_check_outlier_dependency():
    metrics_calc = PerformanceMetrics()
    returns = np.array([0.1] * 20 + [10.0])
    outlier_ratio = metrics_calc._check_outlier_dependency(returns)

    assert outlier_ratio > 0.5


def test_validate_backtest_results_flags_issues():
    metrics_calc = PerformanceMetrics()
    backtest_result = {
        "equity_curve": [100, 120],
        "trades": [{"pnl": 1}] * 5,
        "start_date": None,
        "end_date": None,
    }

    validation = metrics_calc.validate_backtest_results(backtest_result, min_trades=50)

    assert validation["passed"] is False
    assert validation["issues"]


def test_batch_significance_test_empty():
    metrics_calc = PerformanceMetrics()
    result = metrics_calc.batch_significance_test({})

    assert "error" in result
