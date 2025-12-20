from engine.performance_metrics import PerformanceMetrics


def test_generate_insights_thresholds():
    metrics_calc = PerformanceMetrics()

    metrics = {
        "total_return": 0.10,
        "max_drawdown": 0.25,
        "sharpe_ratio": 2.1,
        "win_rate": 0.65,
        "profit_factor": 1.8,
    }

    insights = metrics_calc._generate_insights(metrics)

    assert any("positive returns" in s.lower() for s in insights)
    assert any("high maximum drawdown" in s.lower() for s in insights)
    assert any("high sharpe" in s.lower() for s in insights)
    assert any("high win rate" in s.lower() for s in insights)
    assert any("strong profit factor" in s.lower() for s in insights)
