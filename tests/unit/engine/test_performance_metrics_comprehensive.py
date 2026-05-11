from engine.performance_metrics import PerformanceMetrics


def test_calculate_comprehensive_significance_insufficient_trades():
    metrics_calc = PerformanceMetrics()
    trades = [{"pnl": 1}] * 5

    result = metrics_calc.calculate_comprehensive_significance(trades, min_trades=50)

    assert result["warnings"]
    assert result["is_significant"] is False


def test_calculate_comprehensive_significance_zero_returns():
    metrics_calc = PerformanceMetrics()
    trades = [{"pnl": 0}] * 60

    result = metrics_calc.calculate_comprehensive_significance(trades, min_trades=50)

    assert result["warnings"]
    assert result["is_significant"] is False


def test_batch_significance_test_basic():
    metrics_calc = PerformanceMetrics()
    strategy_results = {
        "A": [{"pnl": 1}] * 60,
        "B": [{"pnl": -1}] * 60,
    }

    result = metrics_calc.batch_significance_test(strategy_results, min_trades=50)

    assert result["n_strategies"] == 2
    assert "individual_results" in result
