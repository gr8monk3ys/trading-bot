"""
Unit tests for PerformanceMetrics.

Tests the performance metrics calculation including:
- Total return, annualized return
- Drawdown calculations
- Sharpe and Sortino ratios
- Win rate and profit factor
- Strategy analysis and comparison
"""

from datetime import datetime, timedelta

import numpy as np
import pytest


class TestPerformanceMetricsInit:
    """Test PerformanceMetrics initialization."""

    def test_init_default_risk_free_rate(self):
        """Test initialization with default risk-free rate."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        assert metrics.risk_free_rate == 0.02

    def test_init_custom_risk_free_rate(self):
        """Test initialization with custom risk-free rate."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics(risk_free_rate=0.05)

        assert metrics.risk_free_rate == 0.05


class TestCalculateMetrics:
    """Test calculate_metrics method."""

    def test_empty_equity_curve(self):
        """Test with empty equity curve."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics.calculate_metrics({"equity_curve": []})

        assert result["total_return"] == 0
        assert result["trade_count"] == 0

    def test_insufficient_equity_curve(self):
        """Test with single-point equity curve."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics.calculate_metrics({"equity_curve": [100000]})

        assert result["total_return"] == 0

    def test_positive_return(self):
        """Test with positive return equity curve."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics.calculate_metrics({
            "equity_curve": [100000, 105000, 110000, 115000, 120000],
            "trades": [],
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 12, 31),
        })

        assert result["total_return"] == pytest.approx(0.20, rel=0.01)
        assert result["final_equity"] == 120000

    def test_negative_return(self):
        """Test with negative return equity curve."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics.calculate_metrics({
            "equity_curve": [100000, 95000, 90000, 85000],
            "trades": [],
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 12, 31),
        })

        assert result["total_return"] < 0

    def test_with_trades(self):
        """Test with trade data."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [
            {"pnl": 500},
            {"pnl": -200},
            {"pnl": 300},
            {"pnl": 100},
            {"pnl": -150},
        ]

        result = metrics.calculate_metrics({
            "equity_curve": [100000, 100500, 100300, 100600, 100700, 100550],
            "trades": trades,
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 6, 30),
        })

        assert result["trade_count"] == 5
        assert result["win_rate"] == 0.6  # 3 wins out of 5


class TestEmptyMetrics:
    """Test _empty_metrics method."""

    def test_returns_all_zeros(self):
        """Test empty metrics returns zeros for all fields."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics._empty_metrics()

        assert result["total_return"] == 0
        assert result["sharpe_ratio"] == 0
        assert result["win_rate"] == 0
        assert result["trade_count"] == 0
        assert result["final_equity"] == 0


class TestAnnualizedReturn:
    """Test annualized return calculation."""

    def test_one_year_period(self):
        """Test annualized return for one year period."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        start = datetime(2024, 1, 1)
        end = datetime(2025, 1, 1)

        result = metrics._calculate_annualized_return(0.10, start, end)

        assert result == pytest.approx(0.10, rel=0.01)

    def test_two_year_period(self):
        """Test annualized return for two year period."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        start = datetime(2024, 1, 1)
        end = datetime(2026, 1, 1)

        # 20% total return over 2 years
        result = metrics._calculate_annualized_return(0.21, start, end)

        # ~10% annualized
        assert result == pytest.approx(0.10, rel=0.05)

    def test_invalid_dates(self):
        """Test with non-datetime dates."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        result = metrics._calculate_annualized_return(0.10, "2024-01-01", "2025-01-01")

        assert result == 0

    def test_zero_duration(self):
        """Test with same start and end date."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        date = datetime(2024, 1, 1)

        result = metrics._calculate_annualized_return(0.10, date, date)

        assert result == 0


class TestMaxDrawdown:
    """Test max drawdown calculation."""

    def test_no_drawdown(self):
        """Test with constantly increasing equity."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        equity = np.array([100, 110, 120, 130, 140])

        result = metrics._calculate_max_drawdown(equity)

        assert result == 0

    def test_single_drawdown(self):
        """Test with single drawdown."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        # Peak at 120, drops to 100, then recovers
        equity = np.array([100, 110, 120, 100, 115])

        result = metrics._calculate_max_drawdown(equity)

        # Drawdown = (120-100)/120 = 16.67%
        assert result == pytest.approx(0.1667, rel=0.01)

    def test_multiple_drawdowns(self):
        """Test with multiple drawdowns, returns max."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        # Multiple peaks and valleys
        equity = np.array([100, 120, 100, 130, 91])

        result = metrics._calculate_max_drawdown(equity)

        # Max drawdown = (130-91)/130 = 30%
        assert result == pytest.approx(0.30, rel=0.01)


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_empty_returns(self):
        """Test with empty returns array."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics._calculate_sharpe_ratio(np.array([]))

        assert result == 0

    def test_zero_volatility(self):
        """Test with zero volatility (constant returns)."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        returns = np.array([0.01, 0.01, 0.01, 0.01])

        result = metrics._calculate_sharpe_ratio(returns)

        assert result == 0  # Std = 0, so Sharpe undefined

    def test_positive_sharpe(self):
        """Test with positive excess returns."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics(risk_free_rate=0)
        # Consistent positive returns
        returns = np.array([0.01, 0.02, 0.01, 0.02, 0.01])

        result = metrics._calculate_sharpe_ratio(returns)

        assert result > 0


class TestSortinoRatio:
    """Test Sortino ratio calculation."""

    def test_empty_returns(self):
        """Test with empty returns array."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics._calculate_sortino_ratio(np.array([]))

        assert result == 0

    def test_no_downside(self):
        """Test with no negative returns."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics(risk_free_rate=0)
        returns = np.array([0.01, 0.02, 0.01, 0.02])

        result = metrics._calculate_sortino_ratio(returns)

        assert result == float("inf")

    def test_with_downside(self):
        """Test with negative returns."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics(risk_free_rate=0)
        returns = np.array([0.02, -0.01, 0.02, -0.01, 0.02])

        result = metrics._calculate_sortino_ratio(returns)

        assert result > 0


class TestCalmarRatio:
    """Test Calmar ratio calculation."""

    def test_zero_drawdown_positive_return(self):
        """Test with zero drawdown and positive return."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        result = metrics._calculate_calmar_ratio(0.10, 0)

        assert result == float("inf")

    def test_zero_drawdown_negative_return(self):
        """Test with zero drawdown and negative return."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        result = metrics._calculate_calmar_ratio(-0.10, 0)

        assert result == 0

    def test_normal_calmar(self):
        """Test normal Calmar calculation."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        result = metrics._calculate_calmar_ratio(0.20, 0.10)

        assert result == 2.0


class TestWinRate:
    """Test win rate calculation."""

    def test_empty_trades(self):
        """Test with no trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        result = metrics._calculate_win_rate([])

        assert result == 0

    def test_all_winners(self):
        """Test with all winning trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [{"pnl": 100}, {"pnl": 200}, {"pnl": 50}]

        result = metrics._calculate_win_rate(trades)

        assert result == 1.0

    def test_all_losers(self):
        """Test with all losing trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [{"pnl": -100}, {"pnl": -200}, {"pnl": -50}]

        result = metrics._calculate_win_rate(trades)

        assert result == 0

    def test_mixed_trades(self):
        """Test with mixed winning/losing trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 200},
            {"pnl": -75},
        ]

        result = metrics._calculate_win_rate(trades)

        assert result == 0.5


class TestProfitFactor:
    """Test profit factor calculation."""

    def test_empty_trades(self):
        """Test with no trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        result = metrics._calculate_profit_factor([])

        assert result == 0

    def test_all_winners(self):
        """Test with all winning trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [{"pnl": 100}, {"pnl": 200}]

        result = metrics._calculate_profit_factor(trades)

        assert result == float("inf")

    def test_no_winners(self):
        """Test with all losing trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [{"pnl": -100}, {"pnl": -200}]

        result = metrics._calculate_profit_factor(trades)

        assert result == 0

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [
            {"pnl": 200},   # Win
            {"pnl": -100},  # Loss
            {"pnl": 100},   # Win
        ]

        result = metrics._calculate_profit_factor(trades)

        # Gross profit = 300, gross loss = 100
        assert result == 3.0


class TestAvgTrade:
    """Test average trade calculation."""

    def test_empty_trades(self):
        """Test with no trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        result = metrics._calculate_avg_trade([])

        assert result == 0

    def test_avg_trade_calculation(self):
        """Test average trade P&L."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 200},
        ]

        result = metrics._calculate_avg_trade(trades)

        assert result == pytest.approx(83.33, rel=0.01)


class TestAvgWinLoss:
    """Test average win/loss calculation."""

    def test_empty_trades(self):
        """Test with no trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        avg_win, avg_loss = metrics._calculate_avg_win_loss([])

        assert avg_win == 0
        assert avg_loss == 0

    def test_only_winners(self):
        """Test with only winning trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [{"pnl": 100}, {"pnl": 200}]

        avg_win, avg_loss = metrics._calculate_avg_win_loss(trades)

        assert avg_win > 0
        assert avg_loss == 0

    def test_only_losers(self):
        """Test with only losing trades."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        trades = [{"pnl": -100}, {"pnl": -200}]

        avg_win, avg_loss = metrics._calculate_avg_win_loss(trades)

        assert avg_win == 0
        assert avg_loss > 0


class TestAnalyzeStrategy:
    """Test analyze_strategy method."""

    def test_analyze_returns_metrics_and_insights(self):
        """Test that analyze returns both metrics and insights."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics.analyze_strategy({
            "equity_curve": [100000, 110000, 120000],
            "trades": [{"pnl": 5000}, {"pnl": 5000}],
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 12, 31),
        })

        assert "metrics" in result
        assert "insights" in result
        assert isinstance(result["insights"], list)


class TestGenerateInsights:
    """Test insight generation."""

    def test_unprofitable_strategy(self):
        """Test insights for unprofitable strategy."""
        from engine.performance_metrics import PerformanceMetrics

        metrics_calc = PerformanceMetrics()
        metrics = {"total_return": -0.05, "max_drawdown": 0.1, "sharpe_ratio": -0.5, "win_rate": 0.3, "profit_factor": 0.8}

        insights = metrics_calc._generate_insights(metrics)

        assert any("not profitable" in insight for insight in insights)

    def test_high_drawdown_warning(self):
        """Test warning for high drawdown."""
        from engine.performance_metrics import PerformanceMetrics

        metrics_calc = PerformanceMetrics()
        metrics = {"total_return": 0.10, "max_drawdown": 0.25, "sharpe_ratio": 1.0, "win_rate": 0.5, "profit_factor": 1.2}

        insights = metrics_calc._generate_insights(metrics)

        assert any("drawdown" in insight.lower() for insight in insights)

    def test_high_sharpe_positive(self):
        """Test positive insight for high Sharpe."""
        from engine.performance_metrics import PerformanceMetrics

        metrics_calc = PerformanceMetrics()
        metrics = {"total_return": 0.20, "max_drawdown": 0.05, "sharpe_ratio": 2.5, "win_rate": 0.6, "profit_factor": 2.0}

        insights = metrics_calc._generate_insights(metrics)

        assert any("strong" in insight.lower() or "high sharpe" in insight.lower() for insight in insights)


class TestCompareStrategies:
    """Test strategy comparison."""

    def test_empty_results(self):
        """Test with no strategies."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        result = metrics.compare_strategies({})

        assert "error" in result

    def test_single_strategy(self):
        """Test with single strategy."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics.compare_strategies({
            "StrategyA": {
                "equity_curve": [100000, 110000],
                "trades": [],
                "start_date": datetime(2024, 1, 1),
                "end_date": datetime(2024, 12, 31),
            }
        })

        assert "metrics" in result
        assert "rankings" in result
        assert "overall_ranking" in result

    def test_multiple_strategies(self):
        """Test comparing multiple strategies."""
        from engine.performance_metrics import PerformanceMetrics

        metrics = PerformanceMetrics()
        result = metrics.compare_strategies({
            "StrategyA": {
                "equity_curve": [100000, 120000],
                "trades": [{"pnl": 10000}, {"pnl": 10000}],
                "start_date": datetime(2024, 1, 1),
                "end_date": datetime(2024, 12, 31),
            },
            "StrategyB": {
                "equity_curve": [100000, 105000],
                "trades": [{"pnl": 2500}, {"pnl": 2500}],
                "start_date": datetime(2024, 1, 1),
                "end_date": datetime(2024, 12, 31),
            },
        })

        # StrategyA should rank higher (better returns)
        assert result["metrics"]["StrategyA"]["total_return"] > result["metrics"]["StrategyB"]["total_return"]
        assert len(result["overall_ranking"]) == 2
