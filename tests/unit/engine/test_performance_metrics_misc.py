import numpy as np

from engine.performance_metrics import PerformanceMetrics


def test_calculate_annualized_return_invalid_dates():
    metrics_calc = PerformanceMetrics()
    assert metrics_calc._calculate_annualized_return(0.1, None, None) == 0


def test_calculate_max_drawdown_basic():
    metrics_calc = PerformanceMetrics()
    drawdown = metrics_calc._calculate_max_drawdown(np.array([100, 90, 95, 80]))
    assert drawdown == 0.2


def test_calculate_sharpe_ratio_empty():
    metrics_calc = PerformanceMetrics()
    assert metrics_calc._calculate_sharpe_ratio(np.array([])) == 0


def test_calculate_sortino_ratio_negative_excess():
    metrics_calc = PerformanceMetrics()
    returns = np.array([-0.01, -0.02, -0.01])
    sortino = metrics_calc._calculate_sortino_ratio(returns)
    assert sortino < 0


def test_calculate_profit_factor_no_losses():
    metrics_calc = PerformanceMetrics()
    trades = [{"pnl": 10}, {"pnl": 5}]
    assert metrics_calc._calculate_profit_factor(trades) == float("inf")
