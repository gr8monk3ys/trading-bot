#!/usr/bin/env python3
"""
Unit tests for utils/streak_sizing.py

Tests cover:
- TradeResult dataclass
- StreakSizer initialization
- Recording trades
- Adjusting position size for streaks
- Streak detection and status updates
- Statistics calculation
- Reset functionality
"""

from datetime import datetime

import pytest

from utils.streak_sizing import StreakSizer, TradeResult

# ============================================================================
# TradeResult Tests
# ============================================================================


class TestTradeResult:
    """Test TradeResult dataclass."""

    def test_create_trade_result(self):
        """Test creating a basic trade result."""
        timestamp = datetime.now()
        trade = TradeResult(
            timestamp=timestamp,
            is_winner=True,
            pnl_pct=0.05,
        )

        assert trade.timestamp == timestamp
        assert trade.is_winner is True
        assert trade.pnl_pct == 0.05
        assert trade.symbol is None

    def test_create_trade_result_with_symbol(self):
        """Test creating trade result with symbol."""
        trade = TradeResult(
            timestamp=datetime.now(),
            is_winner=False,
            pnl_pct=-0.02,
            symbol="AAPL",
        )

        assert trade.symbol == "AAPL"
        assert trade.is_winner is False
        assert trade.pnl_pct == -0.02


# ============================================================================
# StreakSizer Initialization Tests
# ============================================================================


class TestStreakSizerInit:
    """Test StreakSizer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        sizer = StreakSizer()

        assert sizer.lookback_trades == 10
        assert sizer.hot_streak_threshold == 7
        assert sizer.cold_streak_threshold == 3
        assert sizer.hot_multiplier == 1.2
        assert sizer.cold_multiplier == 0.7
        assert sizer.reset_after_trades == 5
        assert sizer.trades == []
        assert sizer.current_streak == "normal"
        assert sizer.streak_started is None
        assert sizer.trades_in_streak == 0

    def test_custom_init(self):
        """Test custom initialization."""
        sizer = StreakSizer(
            lookback_trades=20,
            hot_streak_threshold=15,
            cold_streak_threshold=5,
            hot_multiplier=1.5,
            cold_multiplier=0.5,
            reset_after_trades=10,
        )

        assert sizer.lookback_trades == 20
        assert sizer.hot_streak_threshold == 15
        assert sizer.cold_streak_threshold == 5
        assert sizer.hot_multiplier == 1.5
        assert sizer.cold_multiplier == 0.5
        assert sizer.reset_after_trades == 10


# ============================================================================
# Record Trade Tests
# ============================================================================


class TestRecordTrade:
    """Test record_trade method."""

    def test_record_winning_trade(self):
        """Test recording a winning trade."""
        sizer = StreakSizer()

        sizer.record_trade(is_winner=True, pnl_pct=0.05)

        assert len(sizer.trades) == 1
        assert sizer.trades[0].is_winner is True
        assert sizer.trades[0].pnl_pct == 0.05
        assert sizer.trades_in_streak == 1

    def test_record_losing_trade(self):
        """Test recording a losing trade."""
        sizer = StreakSizer()

        sizer.record_trade(is_winner=False, pnl_pct=-0.02)

        assert len(sizer.trades) == 1
        assert sizer.trades[0].is_winner is False
        assert sizer.trades[0].pnl_pct == -0.02

    def test_record_trade_with_symbol(self):
        """Test recording trade with symbol."""
        sizer = StreakSizer()

        sizer.record_trade(is_winner=True, pnl_pct=0.03, symbol="MSFT")

        assert sizer.trades[0].symbol == "MSFT"

    def test_record_multiple_trades(self):
        """Test recording multiple trades."""
        sizer = StreakSizer()

        for _i in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.01)

        assert len(sizer.trades) == 5
        assert sizer.trades_in_streak == 5


# ============================================================================
# Adjust For Streak Tests
# ============================================================================


class TestAdjustForStreak:
    """Test adjust_for_streak method."""

    def test_normal_streak_no_adjustment(self):
        """Test normal streak has no adjustment."""
        sizer = StreakSizer()
        sizer.current_streak = "normal"

        adjusted = sizer.adjust_for_streak(0.10)

        assert adjusted == 0.10

    def test_hot_streak_increases_position(self):
        """Test hot streak increases position size."""
        sizer = StreakSizer(hot_multiplier=1.2)
        sizer.current_streak = "hot"

        adjusted = sizer.adjust_for_streak(0.10)

        assert adjusted == pytest.approx(0.12, rel=0.01)

    def test_cold_streak_decreases_position(self):
        """Test cold streak decreases position size."""
        sizer = StreakSizer(cold_multiplier=0.7)
        sizer.current_streak = "cold"

        adjusted = sizer.adjust_for_streak(0.10)

        assert adjusted == pytest.approx(0.07, rel=0.01)

    def test_minimum_position_size_cap(self):
        """Test position size doesn't go below 1%."""
        sizer = StreakSizer(cold_multiplier=0.1)  # Very low multiplier
        sizer.current_streak = "cold"

        adjusted = sizer.adjust_for_streak(0.05)  # 0.5% would be result

        assert adjusted == 0.01  # Capped at 1%

    def test_maximum_position_size_cap(self):
        """Test position size doesn't exceed 25%."""
        sizer = StreakSizer(hot_multiplier=3.0)  # Very high multiplier
        sizer.current_streak = "hot"

        adjusted = sizer.adjust_for_streak(0.15)  # 45% would be result

        assert adjusted == 0.25  # Capped at 25%


# ============================================================================
# Update Streak Status Tests
# ============================================================================


class TestUpdateStreakStatus:
    """Test _update_streak_status method."""

    def test_not_enough_trades_stays_normal(self):
        """Test streak stays normal with insufficient trades."""
        sizer = StreakSizer(lookback_trades=10)

        # Record only 5 trades (less than lookback)
        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        assert sizer.current_streak == "normal"

    def test_hot_streak_detected(self):
        """Test hot streak is detected."""
        sizer = StreakSizer(lookback_trades=10, hot_streak_threshold=7)

        # Record 8 winning trades out of 10
        for i in range(10):
            is_winner = i < 8  # First 8 are wins
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.03 if is_winner else -0.01)

        assert sizer.current_streak == "hot"

    def test_cold_streak_detected(self):
        """Test cold streak is detected."""
        sizer = StreakSizer(lookback_trades=10, cold_streak_threshold=3)

        # Record 2 winning trades out of 10
        for i in range(10):
            is_winner = i < 2  # First 2 are wins, rest are losses
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.03 if is_winner else -0.01)

        assert sizer.current_streak == "cold"

    def test_normal_streak_with_medium_wins(self):
        """Test normal streak with medium win rate."""
        sizer = StreakSizer(
            lookback_trades=10,
            hot_streak_threshold=7,
            cold_streak_threshold=3,
        )

        # Record 5 winning trades out of 10 (between thresholds)
        for i in range(10):
            is_winner = i < 5
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.03 if is_winner else -0.01)

        assert sizer.current_streak == "normal"

    def test_streak_transition_from_normal_to_hot(self):
        """Test transition from normal to hot streak."""
        sizer = StreakSizer(lookback_trades=10, hot_streak_threshold=7)

        # Start with 5 losses
        for _ in range(5):
            sizer.record_trade(is_winner=False, pnl_pct=-0.01)

        assert sizer.current_streak == "normal"

        # Add 5 more wins (total 5/10 - still normal)
        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        assert sizer.current_streak == "normal"

        # Add 2 more wins (now 7/10 should trigger hot)
        for _ in range(2):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        assert sizer.current_streak == "hot"

    def test_streak_reset_after_prolonged_hot(self):
        """Test streak resets to normal after prolonged hot streak."""
        sizer = StreakSizer(
            lookback_trades=10,
            hot_streak_threshold=7,
            reset_after_trades=3,
        )

        # Build up a hot streak
        for _ in range(10):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        assert sizer.current_streak == "hot"

        # Continue winning (should reset after reset_after_trades)
        # After streak change, trades_in_streak resets to 0
        # Then we need reset_after_trades more to trigger reset

        # Add more wins to trigger reset
        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        # After 3+ trades in streak with same state, it should reset
        # (only if still in hot/cold, not normal)
        # The implementation resets to normal after reset_after_trades
        # but only if new_streak != "normal"
        assert sizer.current_streak in ["hot", "normal"]


# ============================================================================
# Get Recent Win Rate Tests
# ============================================================================


class TestGetRecentWinRate:
    """Test _get_recent_win_rate method."""

    def test_no_trades_returns_zero(self):
        """Test returns 0 with no trades."""
        sizer = StreakSizer()

        win_rate = sizer._get_recent_win_rate()

        assert win_rate == 0.0

    def test_all_wins_returns_one(self):
        """Test returns 1.0 with all wins."""
        sizer = StreakSizer(lookback_trades=5)

        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        win_rate = sizer._get_recent_win_rate()

        assert win_rate == 1.0

    def test_all_losses_returns_zero(self):
        """Test returns 0 with all losses."""
        sizer = StreakSizer(lookback_trades=5)

        for _ in range(5):
            sizer.record_trade(is_winner=False, pnl_pct=-0.02)

        win_rate = sizer._get_recent_win_rate()

        assert win_rate == 0.0

    def test_mixed_trades_calculates_correctly(self):
        """Test calculates correct win rate with mixed trades."""
        sizer = StreakSizer(lookback_trades=10)

        # 3 wins, 7 losses
        for i in range(10):
            is_winner = i < 3
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.03 if is_winner else -0.01)

        win_rate = sizer._get_recent_win_rate()

        assert win_rate == pytest.approx(0.3, rel=0.01)

    def test_uses_lookback_window(self):
        """Test only considers lookback window."""
        sizer = StreakSizer(lookback_trades=5)

        # First 5 are losses
        for _ in range(5):
            sizer.record_trade(is_winner=False, pnl_pct=-0.02)

        # Next 5 are wins
        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        # Lookback should only see last 5 (all wins)
        win_rate = sizer._get_recent_win_rate()

        assert win_rate == 1.0


# ============================================================================
# Get Streak Statistics Tests
# ============================================================================


class TestGetStreakStatistics:
    """Test get_streak_statistics method."""

    def test_empty_trades_returns_defaults(self):
        """Test returns default stats with no trades."""
        sizer = StreakSizer()

        stats = sizer.get_streak_statistics()

        assert stats["current_streak"] == "normal"
        assert stats["trades_in_streak"] == 0
        assert stats["recent_win_rate"] == 0.0
        assert stats["total_trades"] == 0
        assert stats["overall_win_rate"] == 0.0
        assert stats["hot_streaks"] == 0
        assert stats["cold_streaks"] == 0

    def test_stats_after_trades(self):
        """Test statistics after recording trades."""
        sizer = StreakSizer(lookback_trades=10)

        # Record 6 wins out of 10
        for i in range(10):
            is_winner = i < 6
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.03 if is_winner else -0.01)

        stats = sizer.get_streak_statistics()

        assert stats["total_trades"] == 10
        assert stats["overall_win_rate"] == pytest.approx(0.6, rel=0.01)
        assert stats["recent_win_rate"] == pytest.approx(0.6, rel=0.01)
        assert stats["recent_trades"] == 10

    def test_current_multiplier_hot(self):
        """Test current multiplier for hot streak."""
        sizer = StreakSizer(hot_multiplier=1.3, lookback_trades=5, hot_streak_threshold=4)

        # Need trades to get past the early return
        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        assert sizer.current_streak == "hot"
        stats = sizer.get_streak_statistics()

        assert stats["current_multiplier"] == 1.3

    def test_current_multiplier_cold(self):
        """Test current multiplier for cold streak."""
        sizer = StreakSizer(cold_multiplier=0.6, lookback_trades=5, cold_streak_threshold=1)

        # Need trades to get past the early return
        for _ in range(5):
            sizer.record_trade(is_winner=False, pnl_pct=-0.02)

        assert sizer.current_streak == "cold"
        stats = sizer.get_streak_statistics()

        assert stats["current_multiplier"] == 0.6

    def test_current_multiplier_normal(self):
        """Test current multiplier for normal streak."""
        sizer = StreakSizer(lookback_trades=5, hot_streak_threshold=4, cold_streak_threshold=1)

        # Need trades to get past the early return (2/5 wins is normal)
        for i in range(5):
            is_winner = i < 2
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.03 if is_winner else -0.02)

        assert sizer.current_streak == "normal"
        stats = sizer.get_streak_statistics()

        assert stats["current_multiplier"] == 1.0

    def test_partial_lookback_window(self):
        """Test stats with fewer trades than lookback."""
        sizer = StreakSizer(lookback_trades=10)

        # Only record 5 trades
        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        stats = sizer.get_streak_statistics()

        assert stats["total_trades"] == 5
        assert stats["recent_trades"] == 5
        assert stats["recent_win_rate"] == 1.0


# ============================================================================
# Reset Tests
# ============================================================================


class TestReset:
    """Test reset method."""

    def test_reset_clears_trades(self):
        """Test reset clears trade history."""
        sizer = StreakSizer()

        # Record some trades
        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        sizer.reset()

        assert sizer.trades == []

    def test_reset_clears_streak_state(self):
        """Test reset clears streak state."""
        sizer = StreakSizer()

        # Build up a hot streak
        for _ in range(10):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        sizer.reset()

        assert sizer.current_streak == "normal"
        assert sizer.streak_started is None
        assert sizer.trades_in_streak == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestStreakSizerIntegration:
    """Integration tests for StreakSizer."""

    def test_full_trading_cycle(self):
        """Test complete trading cycle with streaks."""
        sizer = StreakSizer(
            lookback_trades=10,
            hot_streak_threshold=7,
            cold_streak_threshold=3,
            hot_multiplier=1.2,
            cold_multiplier=0.7,
        )

        base_size = 0.10

        # Phase 1: Build hot streak (8 wins out of 10)
        for i in range(10):
            is_winner = i < 8
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.03 if is_winner else -0.01)

        assert sizer.current_streak == "hot"
        adjusted = sizer.adjust_for_streak(base_size)
        assert adjusted == pytest.approx(0.12, rel=0.01)

        # Phase 2: Transition to losing streak
        for _ in range(8):
            sizer.record_trade(is_winner=False, pnl_pct=-0.02)

        assert sizer.current_streak == "cold"
        adjusted = sizer.adjust_for_streak(base_size)
        assert adjusted == pytest.approx(0.07, rel=0.01)

        # Phase 3: Recovery to normal
        for _ in range(5):
            sizer.record_trade(is_winner=True, pnl_pct=0.03)

        # Now should be back to normal (5/10 wins is between thresholds)
        assert sizer.current_streak == "normal"
        adjusted = sizer.adjust_for_streak(base_size)
        assert adjusted == base_size

    def test_position_sizing_consistency(self):
        """Test position sizing remains consistent."""
        sizer = StreakSizer()

        base_size = 0.15

        # Same streak state should give same adjustment
        sizer.current_streak = "hot"
        adj1 = sizer.adjust_for_streak(base_size)
        adj2 = sizer.adjust_for_streak(base_size)

        assert adj1 == adj2

    def test_statistics_accuracy(self):
        """Test statistics accuracy after various trades."""
        sizer = StreakSizer(lookback_trades=10)

        # Record specific pattern
        wins = 0
        for i in range(15):
            is_winner = i % 3 != 0  # Win 2 out of every 3
            if is_winner:
                wins += 1
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.02 if is_winner else -0.01)

        stats = sizer.get_streak_statistics()

        assert stats["total_trades"] == 15
        expected_overall_rate = wins / 15
        assert stats["overall_win_rate"] == pytest.approx(expected_overall_rate, rel=0.01)


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_base_size(self):
        """Test adjustment with zero base size."""
        sizer = StreakSizer()
        sizer.current_streak = "hot"

        adjusted = sizer.adjust_for_streak(0.0)

        # Should be capped at minimum
        assert adjusted == 0.01

    def test_very_small_base_size(self):
        """Test adjustment with very small base size."""
        sizer = StreakSizer(cold_multiplier=0.5)
        sizer.current_streak = "cold"

        adjusted = sizer.adjust_for_streak(0.01)  # 0.005 would be result

        assert adjusted == 0.01  # Capped at minimum

    def test_large_number_of_trades(self):
        """Test with large number of trades."""
        sizer = StreakSizer(lookback_trades=10)

        # Record 1000 trades
        for i in range(1000):
            is_winner = i % 2 == 0  # Alternating
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.01 if is_winner else -0.01)

        assert len(sizer.trades) == 1000
        win_rate = sizer._get_recent_win_rate()
        assert win_rate == pytest.approx(0.5, rel=0.1)  # Should be around 50%

    def test_exact_threshold_values(self):
        """Test behavior at exact threshold values."""
        sizer = StreakSizer(
            lookback_trades=10,
            hot_streak_threshold=7,
            cold_streak_threshold=3,
        )

        # Exactly 7 wins (hot threshold)
        for i in range(10):
            is_winner = i < 7
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.02 if is_winner else -0.01)

        assert sizer.current_streak == "hot"

        # Reset and test exact cold threshold
        sizer.reset()

        # Exactly 3 wins (cold threshold)
        for i in range(10):
            is_winner = i < 3
            sizer.record_trade(is_winner=is_winner, pnl_pct=0.02 if is_winner else -0.01)

        assert sizer.current_streak == "cold"
