from datetime import datetime

from engine.performance_metrics import PerformanceMetrics


def test_analyze_strategy_generates_insights():
    metrics_calc = PerformanceMetrics()

    backtest_result = {
        "equity_curve": [100, 90, 95],
        "trades": [{"pnl": -5}, {"pnl": 2}],
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 2, 1),
    }

    report = metrics_calc.analyze_strategy(backtest_result)

    assert "metrics" in report
    assert "insights" in report
    assert any("not profitable" in s.lower() for s in report["insights"])
