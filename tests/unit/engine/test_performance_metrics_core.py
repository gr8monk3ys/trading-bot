from datetime import datetime

import numpy as np

from engine.performance_metrics import PerformanceMetrics


def test_calculate_metrics_basic():
    metrics_calc = PerformanceMetrics()

    backtest_result = {
        "equity_curve": [100000, 105000, 103000, 110000],
        "trades": [
            {"pnl": 500},
            {"pnl": -200},
            {"pnl": 700},
        ],
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 2, 1),
        "initial_capital": 100000,
    }

    metrics = metrics_calc.calculate_metrics(backtest_result)

    assert metrics["total_return"] > 0
    assert metrics["trade_count"] == 3
    assert 0 <= metrics["win_rate"] <= 1
    assert metrics["final_equity"] == 110000


def test_sortino_ratio_no_downside_returns():
    metrics_calc = PerformanceMetrics()
    returns = np.array([0.01, 0.02, 0.015])

    sortino = metrics_calc._calculate_sortino_ratio(returns)

    assert sortino == float("inf")


def test_calmar_ratio_zero_drawdown():
    metrics_calc = PerformanceMetrics()

    assert metrics_calc._calculate_calmar_ratio(0.1, 0) == float("inf")
    assert metrics_calc._calculate_calmar_ratio(0.0, 0) == 0


def test_win_rate_profit_factor_avg_trade_edges():
    metrics_calc = PerformanceMetrics()
    trades = [{"pnl": 10}, {"pnl": -5}, {"pnl": 0}]

    assert metrics_calc._calculate_win_rate(trades) == 1 / 3
    assert metrics_calc._calculate_profit_factor(trades) == 2.0
    assert metrics_calc._calculate_avg_trade(trades) == 5 / 3

    assert metrics_calc._calculate_profit_factor([]) == 0


def test_calculate_metrics_empty_equity_curve():
    metrics_calc = PerformanceMetrics()
    metrics = metrics_calc.calculate_metrics({"equity_curve": []})

    assert metrics["total_return"] == 0
    assert metrics["trade_count"] == 0
