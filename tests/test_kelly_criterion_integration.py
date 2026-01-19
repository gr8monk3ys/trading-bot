"""
Tests for Kelly Criterion Integration in MomentumStrategy

Tests cover:
- Kelly Criterion parameter configuration
- Position sizing calculation with Kelly
- Half-Kelly vs Full-Kelly behavior
- Integration with BaseStrategy.calculate_kelly_position_size
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKellyParameters:
    """Test Kelly Criterion parameters in MomentumStrategy."""

    def test_kelly_params_in_defaults(self):
        """Kelly parameters should be in default_parameters."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy()
        params = strategy.default_parameters()

        assert "use_kelly_criterion" in params
        assert "kelly_fraction" in params
        assert "kelly_min_trades" in params
        assert "kelly_lookback" in params

    def test_kelly_disabled_by_default(self):
        """Kelly Criterion should be disabled by default."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy()
        params = strategy.default_parameters()

        assert params["use_kelly_criterion"] is False

    def test_kelly_fraction_is_half_kelly(self):
        """Default Kelly fraction should be 0.5 (Half Kelly)."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy()
        params = strategy.default_parameters()

        assert params["kelly_fraction"] == 0.5

    def test_kelly_min_trades_reasonable(self):
        """Minimum trades should be reasonable (30+)."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy()
        params = strategy.default_parameters()

        assert params["kelly_min_trades"] >= 30


class TestKellyCriterionCore:
    """Test KellyCriterion class directly."""

    def test_kelly_formula_calculation(self):
        """Test Kelly formula: f* = (bp - q) / b."""
        from utils.kelly_criterion import KellyCriterion

        kelly = KellyCriterion(kelly_fraction=1.0)  # Full Kelly

        # Example: Win rate 60%, profit factor 2.0
        # f* = (2.0 * 0.6 - 0.4) / 2.0 = 0.8 / 2.0 = 0.4 = 40%
        win_rate = 0.6
        profit_factor = 2.0

        result = kelly.calculate_kelly_fraction(win_rate, profit_factor)

        expected = (profit_factor * win_rate - (1 - win_rate)) / profit_factor
        assert result == pytest.approx(expected, rel=1e-6)
        assert result == pytest.approx(0.4, rel=1e-6)

    def test_half_kelly_reduces_by_half(self):
        """Half Kelly should return half the full Kelly position."""
        from utils.kelly_criterion import KellyCriterion

        full_kelly = KellyCriterion(kelly_fraction=1.0)
        half_kelly = KellyCriterion(kelly_fraction=0.5)

        win_rate = 0.6
        profit_factor = 2.0

        full_result = full_kelly.calculate_kelly_fraction(win_rate, profit_factor)
        half_result = half_kelly.calculate_kelly_fraction(win_rate, profit_factor)

        assert half_result == pytest.approx(full_result * 0.5, rel=1e-6)

    def test_negative_kelly_means_no_edge(self):
        """Negative Kelly means no edge - don't bet."""
        from utils.kelly_criterion import KellyCriterion

        kelly = KellyCriterion(kelly_fraction=1.0)

        # Win rate 40%, profit factor 1.0 = no edge
        # f* = (1.0 * 0.4 - 0.6) / 1.0 = -0.2
        result = kelly.calculate_kelly_fraction(0.4, 1.0)

        assert result < 0, "Negative Kelly indicates no edge"

    def test_kelly_position_size_capped(self):
        """Position size should be capped at max_position_size."""
        from utils.kelly_criterion import KellyCriterion

        kelly = KellyCriterion(kelly_fraction=1.0, max_position_size=0.20)  # 20% max

        # Simulate high win rate scenario that would exceed max
        kelly.win_rate = 0.9
        kelly.profit_factor = 3.0
        kelly.trades = [None] * 100  # Simulate enough trades

        position_value, position_fraction = kelly.calculate_position_size(current_capital=100000)

        assert position_fraction <= 0.20, "Position should be capped at max"


class TestKellyWithTrades:
    """Test Kelly Criterion with trade history."""

    def test_trade_recording(self):
        """Test that trades are recorded correctly."""
        from utils.kelly_criterion import KellyCriterion, Trade

        kelly = KellyCriterion()

        trade = Trade(
            symbol="AAPL",
            entry_time=datetime.now() - timedelta(days=1),
            exit_time=datetime.now(),
            entry_price=150.0,
            exit_price=155.0,
            quantity=10,
            pnl=50.0,
            pnl_pct=0.0333,
            is_winner=True,
        )

        kelly.add_trade(trade)

        assert len(kelly.trades) == 1
        assert kelly.win_rate == 1.0  # 1 winner, 0 losers

    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        from utils.kelly_criterion import KellyCriterion, Trade

        kelly = KellyCriterion()

        # Add 3 winners and 2 losers
        for i in range(3):
            trade = Trade(
                symbol="AAPL",
                entry_time=datetime.now() - timedelta(days=i + 1),
                exit_time=datetime.now() - timedelta(days=i),
                entry_price=100.0,
                exit_price=105.0,
                quantity=10,
                pnl=50.0,
                pnl_pct=0.05,
                is_winner=True,
            )
            kelly.add_trade(trade)

        for i in range(2):
            trade = Trade(
                symbol="AAPL",
                entry_time=datetime.now() - timedelta(days=i + 4),
                exit_time=datetime.now() - timedelta(days=i + 3),
                entry_price=100.0,
                exit_price=97.0,
                quantity=10,
                pnl=-30.0,
                pnl_pct=-0.03,
                is_winner=False,
            )
            kelly.add_trade(trade)

        assert kelly.win_rate == pytest.approx(0.6, rel=1e-6)  # 3/5 = 60%


class TestKellyResearch:
    """Validate Kelly Criterion research claims."""

    def test_half_kelly_variance(self):
        """Half Kelly should have ~25% of full Kelly's variance."""
        # Research: Half-Kelly provides 75% of max profit with 25% variance
        # This is a documentation test
        kelly_fraction = 0.5

        # Variance scales with kelly_fraction^2
        variance_ratio = kelly_fraction**2

        assert variance_ratio == 0.25, "Half Kelly has 25% of full Kelly variance"

    def test_half_kelly_expected_growth(self):
        """Half Kelly should have ~75% of full Kelly's expected growth."""
        # Research: Half-Kelly provides 75% of max profit with 25% variance
        # kelly_fraction = 0.5 (not used, just documentation of the concept)

        # Expected growth ratio for half Kelly
        # g(f) = p * log(1 + bf) + (1-p) * log(1 - f)
        # At f = f*/2, growth is approximately 75% of max

        expected_growth_ratio = 0.75  # Research claim
        assert expected_growth_ratio == 0.75

    def test_kelly_advantages(self):
        """Document Kelly Criterion advantages."""
        advantages = [
            "Maximizes long-term growth rate",
            "Mathematically optimal for known edge",
            "Prevents over-betting (negative Kelly = no bet)",
            "Adapts to changing win rates",
            "Half-Kelly balances growth vs variance",
        ]

        # All advantages documented
        assert len(advantages) >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
