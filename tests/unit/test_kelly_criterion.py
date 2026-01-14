#!/usr/bin/env python3
"""
Unit tests for Kelly Criterion position sizing.

Tests cover:
1. Kelly formula calculation correctness
2. Position size limits (min/max)
3. Trade history tracking
4. Half-Kelly and Quarter-Kelly variations
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.kelly_criterion import KellyCriterion, Trade


class TestKellyFormula:
    """Test the core Kelly formula calculations."""

    def test_kelly_formula_basic(self):
        """Test basic Kelly formula: f* = (bp - q) / b"""
        kelly = KellyCriterion(kelly_fraction=1.0)  # Full Kelly

        # Win rate 60%, profit factor 2.0
        # Kelly = (2.0 * 0.6 - 0.4) / 2.0 = (1.2 - 0.4) / 2.0 = 0.4 = 40%
        result = kelly.calculate_kelly_fraction(win_rate=0.6, profit_factor=2.0)
        assert abs(result - 0.4) < 0.01, f"Expected 0.40, got {result}"

    def test_kelly_formula_high_win_rate(self):
        """Test Kelly with high win rate."""
        kelly = KellyCriterion(kelly_fraction=1.0)

        # Win rate 80%, profit factor 1.5
        # Kelly = (1.5 * 0.8 - 0.2) / 1.5 = (1.2 - 0.2) / 1.5 = 0.667
        result = kelly.calculate_kelly_fraction(win_rate=0.8, profit_factor=1.5)
        assert abs(result - 0.667) < 0.01, f"Expected ~0.667, got {result}"

    def test_kelly_formula_no_edge(self):
        """Test Kelly with no edge (should be 0 or negative)."""
        kelly = KellyCriterion(kelly_fraction=1.0)

        # Win rate 50%, profit factor 1.0 = no edge
        # Kelly = (1.0 * 0.5 - 0.5) / 1.0 = 0
        result = kelly.calculate_kelly_fraction(win_rate=0.5, profit_factor=1.0)
        assert abs(result) < 0.01, f"Expected ~0, got {result}"

    def test_kelly_formula_negative_edge(self):
        """Test Kelly with negative edge (should be negative)."""
        kelly = KellyCriterion(kelly_fraction=1.0)

        # Win rate 40%, profit factor 1.0 = negative edge
        # Kelly = (1.0 * 0.4 - 0.6) / 1.0 = -0.2
        result = kelly.calculate_kelly_fraction(win_rate=0.4, profit_factor=1.0)
        assert result < 0, f"Expected negative, got {result}"

    def test_half_kelly(self):
        """Test Half-Kelly fraction (conservative approach)."""
        kelly = KellyCriterion(kelly_fraction=0.5)

        # Full Kelly would be 40%, Half Kelly = 20%
        result = kelly.calculate_kelly_fraction(win_rate=0.6, profit_factor=2.0)
        assert abs(result - 0.2) < 0.01, f"Expected 0.20, got {result}"

    def test_quarter_kelly(self):
        """Test Quarter-Kelly fraction (very conservative)."""
        kelly = KellyCriterion(kelly_fraction=0.25)

        # Full Kelly would be 40%, Quarter Kelly = 10%
        result = kelly.calculate_kelly_fraction(win_rate=0.6, profit_factor=2.0)
        assert abs(result - 0.1) < 0.01, f"Expected 0.10, got {result}"


class TestPositionSizing:
    """Test position size calculations with limits."""

    def test_position_size_basic(self):
        """Test basic position size calculation."""
        kelly = KellyCriterion(
            kelly_fraction=0.5,
            max_position_size=0.25,
            min_position_size=0.05
        )

        # Add enough trades to enable Kelly calculation
        for i in range(35):
            kelly.add_trade(Trade(
                symbol='AAPL',
                entry_time=datetime.now() - timedelta(days=i),
                exit_time=datetime.now() - timedelta(days=i-1),
                entry_price=100,
                exit_price=105 if i % 5 != 0 else 95,  # 80% win rate
                quantity=10,
                pnl=50 if i % 5 != 0 else -50,
                pnl_pct=0.05 if i % 5 != 0 else -0.05,
                is_winner=i % 5 != 0
            ))

        position_value, position_fraction = kelly.calculate_position_size(
            current_capital=100000,
            win_rate=0.6,
            profit_factor=2.0
        )

        # Half Kelly of 40% = 20%
        assert position_fraction <= 0.25, "Should not exceed max position size"
        assert position_fraction >= 0.05, "Should not go below min position size"
        assert position_value > 0, "Position value should be positive"

    def test_position_size_respects_max(self):
        """Test that position size respects maximum limit."""
        kelly = KellyCriterion(
            kelly_fraction=1.0,  # Full Kelly
            max_position_size=0.15
        )

        # Add enough trades
        for i in range(35):
            kelly.add_trade(Trade(
                symbol='TEST',
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100,
                exit_price=110,
                quantity=10,
                pnl=100,
                pnl_pct=0.10,
                is_winner=True
            ))

        _, position_fraction = kelly.calculate_position_size(
            current_capital=100000,
            win_rate=0.9,  # Very high win rate would suggest large position
            profit_factor=5.0
        )

        assert position_fraction <= 0.15, f"Should not exceed max (got {position_fraction})"

    def test_position_size_respects_min(self):
        """Test that position size respects minimum limit."""
        kelly = KellyCriterion(
            kelly_fraction=1.0,
            min_position_size=0.05
        )

        # With insufficient trades, should use minimum
        _, position_fraction = kelly.calculate_position_size(
            current_capital=100000
        )

        assert position_fraction >= 0.05, f"Should not go below min (got {position_fraction})"


class TestTradeHistory:
    """Test trade history tracking and metrics."""

    def test_add_trade(self):
        """Test adding trades to history."""
        kelly = KellyCriterion()

        trade = Trade(
            symbol='AAPL',
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500.0,
            pnl_pct=0.0333,
            is_winner=True
        )

        kelly.add_trade(trade)
        assert len(kelly.trades) == 1

    def test_win_rate_calculation(self):
        """Test win rate calculation from trade history."""
        kelly = KellyCriterion()

        # Add 10 trades: 6 winners, 4 losers = 60% win rate
        for i in range(10):
            is_winner = i < 6
            kelly.add_trade(Trade(
                symbol='TEST',
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100,
                exit_price=105 if is_winner else 95,
                quantity=10,
                pnl=50 if is_winner else -50,
                pnl_pct=0.05 if is_winner else -0.05,
                is_winner=is_winner
            ))

        assert abs(kelly.win_rate - 0.6) < 0.01, f"Expected 60% win rate, got {kelly.win_rate}"

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        kelly = KellyCriterion()

        # Add trades: avg win 5%, avg loss 2.5% = profit factor 2.0
        for i in range(10):
            is_winner = i < 6
            kelly.add_trade(Trade(
                symbol='TEST',
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100,
                exit_price=105 if is_winner else 97.5,
                quantity=10,
                pnl=50 if is_winner else -25,
                pnl_pct=0.05 if is_winner else -0.025,
                is_winner=is_winner
            ))

        # Profit factor = avg_win / avg_loss = 0.05 / 0.025 = 2.0
        assert kelly.profit_factor > 1.5, f"Expected profit factor > 1.5, got {kelly.profit_factor}"

    def test_add_trade_from_position(self):
        """Test adding trade from position details."""
        kelly = KellyCriterion()

        kelly.add_trade_from_position(
            symbol='TSLA',
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            entry_price=200.0,
            exit_price=210.0,
            quantity=50,
            side='long'
        )

        assert len(kelly.trades) == 1
        trade = kelly.trades[0]
        assert trade.is_winner
        assert trade.pnl == 500.0  # (210-200) * 50
        assert abs(trade.pnl_pct - 0.05) < 0.01

    def test_short_trade(self):
        """Test short trade P/L calculation."""
        kelly = KellyCriterion()

        kelly.add_trade_from_position(
            symbol='SPY',
            entry_time=datetime.now() - timedelta(hours=2),
            exit_time=datetime.now(),
            entry_price=450.0,
            exit_price=440.0,  # Price went down, short wins
            quantity=10,
            side='short'
        )

        trade = kelly.trades[0]
        assert trade.is_winner
        assert trade.pnl == 100.0  # (450-440) * 10


class TestPerformanceSummary:
    """Test performance summary generation."""

    def test_empty_summary(self):
        """Test summary with no trades."""
        kelly = KellyCriterion()
        summary = kelly.get_performance_summary()

        assert summary['total_trades'] == 0
        assert summary['win_rate'] == 0.0

    def test_summary_with_trades(self):
        """Test summary with trade history."""
        kelly = KellyCriterion()

        # Add trades
        for i in range(20):
            is_winner = i % 3 != 0  # ~67% win rate
            kelly.add_trade(Trade(
                symbol='TEST',
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100,
                exit_price=105 if is_winner else 95,
                quantity=10,
                pnl=50 if is_winner else -50,
                pnl_pct=0.05 if is_winner else -0.05,
                is_winner=is_winner
            ))

        summary = kelly.get_performance_summary()

        assert summary['total_trades'] == 20
        assert summary['win_rate'] > 0.5
        assert 'profit_factor' in summary
        assert 'kelly_fraction' in summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_profit_factor(self):
        """Test handling of zero profit factor."""
        kelly = KellyCriterion()

        # All losing trades = profit factor 0
        for i in range(10):
            kelly.add_trade(Trade(
                symbol='TEST',
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100,
                exit_price=95,
                quantity=10,
                pnl=-50,
                pnl_pct=-0.05,
                is_winner=False
            ))

        result = kelly.calculate_kelly_fraction(win_rate=0.0, profit_factor=0.0)
        assert result == 0.0, "Should return 0 for zero profit factor"

    def test_lookback_limit(self):
        """Test that lookback limit is respected."""
        kelly = KellyCriterion(lookback_trades=10)

        # Add 20 trades
        for i in range(20):
            kelly.add_trade(Trade(
                symbol='TEST',
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100,
                exit_price=105,
                quantity=10,
                pnl=50,
                pnl_pct=0.05,
                is_winner=True
            ))

        # Metrics should only consider last 10 trades
        summary = kelly.get_performance_summary()
        assert summary['recent_trades'] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
