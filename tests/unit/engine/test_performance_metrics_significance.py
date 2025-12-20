import numpy as np

from engine.performance_metrics import PerformanceMetrics


def test_calculate_significance_insufficient_trades():
    metrics = PerformanceMetrics()
    trades = [{"pnl": 0.01}] * 10

    result = metrics.calculate_significance(trades, min_trades=50)

    assert result["is_significant"] is False
    assert result["trade_count"] == 10
    assert result["warnings"]


def test_calculate_significance_zero_returns():
    metrics = PerformanceMetrics()
    trades = [{"pnl": 0.0}] * 60

    result = metrics.calculate_significance(trades, min_trades=50)

    assert result["is_significant"] is False
    assert "No valid trade returns found." in result["warnings"]


def test_calculate_significance_positive_returns(monkeypatch):
    def _fake_ttest(_returns, _mu):
        return 2.0, 0.01

    monkeypatch.setattr("engine.performance_metrics.stats.ttest_1samp", _fake_ttest)

    metrics = PerformanceMetrics()
    rng = np.random.default_rng(42)
    trades = [{"pnl": float(x)} for x in rng.normal(loc=0.01, scale=0.005, size=80)]

    result = metrics.calculate_significance(trades, min_trades=50)

    assert result["trade_count"] == 80
    assert result["p_value"] <= 1.0
