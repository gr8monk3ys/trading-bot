#!/usr/bin/env python3
"""
Unit tests for VWAP Execution Algorithm.

Tests cover:
1. Slice creation and weighting
2. VWAP calculation
3. Participation rate limits
4. Execution statistics
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.vwap_executor import VWAPExecutor, VWAPSlice, VWAPResult


class MockBar:
    """Mock bar for testing."""
    def __init__(self, volume=1000000):
        self.volume = volume


class MockQuote:
    """Mock quote for testing."""
    def __init__(self, bid=100.0, ask=100.10):
        self.bid_price = bid
        self.ask_price = ask


class TestSliceCreation:
    """Test order slice creation."""

    def test_creates_correct_number_of_slices(self):
        """Test that correct number of slices are created."""
        executor = VWAPExecutor(MagicMock())

        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        slices = executor._create_slices(
            symbol='AAPL',
            side='buy',
            total_qty=100,
            duration_minutes=60,
            num_slices=5,
            weights=weights
        )

        assert len(slices) == 5

    def test_slice_quantities_sum_to_total(self):
        """Test that slice quantities sum to total order."""
        executor = VWAPExecutor(MagicMock())

        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        slices = executor._create_slices(
            symbol='AAPL',
            side='buy',
            total_qty=100,
            duration_minutes=60,
            num_slices=5,
            weights=weights
        )

        total = sum(s.target_qty for s in slices)
        assert abs(total - 100) < 0.1, f"Total qty {total} != 100"

    def test_slices_are_time_ordered(self):
        """Test that slices are scheduled in order."""
        executor = VWAPExecutor(MagicMock())

        weights = [0.25, 0.25, 0.25, 0.25]
        slices = executor._create_slices(
            symbol='AAPL',
            side='buy',
            total_qty=100,
            duration_minutes=60,
            num_slices=4,
            weights=weights
        )

        for i in range(len(slices) - 1):
            assert slices[i].scheduled_time < slices[i+1].scheduled_time

    def test_minimum_slice_quantity_enforced(self):
        """Test minimum slice quantity is enforced."""
        executor = VWAPExecutor(MagicMock(), min_slice_qty=5.0)

        weights = [0.01, 0.99]  # First slice would be tiny
        slices = executor._create_slices(
            symbol='AAPL',
            side='buy',
            total_qty=100,
            duration_minutes=60,
            num_slices=2,
            weights=weights
        )

        # First slice should be at least min_slice_qty
        assert slices[0].target_qty >= 5.0


class TestVWAPCalculation:
    """Test VWAP calculation."""

    def test_vwap_equal_volumes(self):
        """Test VWAP with equal volumes = simple average."""
        executor = VWAPExecutor(MagicMock())

        prices = [100, 102, 104]
        volumes = [100, 100, 100]

        vwap = executor._calculate_vwap(prices, volumes)

        # With equal volumes, VWAP = simple average
        assert vwap == 102.0

    def test_vwap_weighted_by_volume(self):
        """Test VWAP is properly weighted by volume."""
        executor = VWAPExecutor(MagicMock())

        prices = [100, 110]
        volumes = [900, 100]  # 90% at $100, 10% at $110

        vwap = executor._calculate_vwap(prices, volumes)

        # Expected: (100*900 + 110*100) / 1000 = 101
        assert vwap == 101.0

    def test_vwap_empty_data(self):
        """Test VWAP handles empty data."""
        executor = VWAPExecutor(MagicMock())

        vwap = executor._calculate_vwap([], [])

        assert vwap == 0.0

    def test_vwap_zero_volume(self):
        """Test VWAP handles zero total volume."""
        executor = VWAPExecutor(MagicMock())

        prices = [100, 105, 110]
        volumes = [0, 0, 0]

        vwap = executor._calculate_vwap(prices, volumes)

        # Should return simple average when no volume
        assert vwap == 105.0


class TestVolumeProfile:
    """Test volume profile weighting."""

    def test_default_profile_exists(self):
        """Test default volume profile has all trading hours."""
        executor = VWAPExecutor(MagicMock())

        assert '09:30' in executor.DEFAULT_VOLUME_PROFILE
        assert '15:30' in executor.DEFAULT_VOLUME_PROFILE

    def test_profile_weights_sum_to_reasonable_value(self):
        """Test profile weights sum to ~1.0."""
        executor = VWAPExecutor(MagicMock())

        total = sum(executor.DEFAULT_VOLUME_PROFILE.values())

        # Should be close to 1.0 (allowing for partial day)
        assert 0.5 < total < 1.5

    def test_opening_has_high_volume(self):
        """Test that market open has high volume weight."""
        executor = VWAPExecutor(MagicMock())

        # Opening 30 min should have higher volume than midday
        assert executor.DEFAULT_VOLUME_PROFILE['09:30'] > executor.DEFAULT_VOLUME_PROFILE['12:00']

    def test_closing_has_high_volume(self):
        """Test that market close has high volume weight."""
        executor = VWAPExecutor(MagicMock())

        # Closing 30 min should have higher volume than midday
        assert executor.DEFAULT_VOLUME_PROFILE['15:30'] > executor.DEFAULT_VOLUME_PROFILE['12:00']


class TestExecutionStats:
    """Test execution statistics."""

    def test_empty_stats(self):
        """Test stats with no history."""
        executor = VWAPExecutor(MagicMock())

        stats = executor.get_execution_stats()

        assert stats['total_executions'] == 0
        assert stats['avg_slippage_bps'] == 0

    def test_stats_calculation(self):
        """Test stats are calculated correctly."""
        executor = VWAPExecutor(MagicMock())

        # Add some fake history
        executor.execution_history = [
            VWAPResult('AAPL', 'buy', 100, 100, 150.0, 150.0, 5.0, 60, 5, 5, 'completed'),
            VWAPResult('MSFT', 'buy', 200, 200, 300.0, 300.0, -3.0, 60, 5, 5, 'completed'),
        ]

        stats = executor.get_execution_stats()

        assert stats['total_executions'] == 2
        assert stats['avg_slippage_bps'] == 1.0  # (5 + -3) / 2
        assert stats['completion_rate'] == 1.0


class TestParticipationRate:
    """Test participation rate limiting."""

    @pytest.mark.asyncio
    async def test_participation_reduces_quantity(self):
        """Test that participation rate limits slice size."""
        broker = MagicMock()
        broker.get_bars = AsyncMock(return_value=[
            MockBar(volume=10000),  # Avg volume = 10000
        ])

        executor = VWAPExecutor(broker)

        # Want 1000 shares but participation rate is 5%
        # Max = 10000 * 0.05 = 500
        adjusted = await executor._adjust_for_participation('AAPL', 1000, 0.05)

        assert adjusted == 500

    @pytest.mark.asyncio
    async def test_participation_no_reduction_needed(self):
        """Test that small orders aren't reduced."""
        broker = MagicMock()
        broker.get_bars = AsyncMock(return_value=[
            MockBar(volume=100000),  # High volume
        ])

        executor = VWAPExecutor(broker)

        # Want 100 shares, participation allows 10000
        adjusted = await executor._adjust_for_participation('AAPL', 100, 0.10)

        assert adjusted == 100  # Not reduced


class TestSlippageCalculation:
    """Test slippage calculation."""

    def test_buy_positive_slippage(self):
        """Test buy order slippage when price is higher than VWAP."""
        # Buy at $101, VWAP was $100
        # Slippage = (101 - 100) / 100 * 10000 = 100 bps
        avg_price = 101
        vwap = 100
        slippage = ((avg_price - vwap) / vwap) * 10000

        assert slippage == 100  # 100 bps = 1%

    def test_buy_negative_slippage(self):
        """Test buy order slippage when price is lower than VWAP (good!)."""
        # Buy at $99, VWAP was $100
        # Slippage = (99 - 100) / 100 * 10000 = -100 bps
        avg_price = 99
        vwap = 100
        slippage = ((avg_price - vwap) / vwap) * 10000

        assert slippage == -100  # -100 bps = beat VWAP by 1%

    def test_sell_slippage_inverted(self):
        """Test sell order slippage is inverted."""
        # Sell at $99, VWAP was $100
        # For sells: slippage = (VWAP - price) / VWAP * 10000 = 100 bps (bad)
        avg_price = 99
        vwap = 100
        slippage = ((vwap - avg_price) / vwap) * 10000

        assert slippage == 100  # 100 bps worse than VWAP


class TestCancelOrder:
    """Test order cancellation."""

    def test_cancel_active_order(self):
        """Test cancelling pending slices."""
        executor = VWAPExecutor(MagicMock())

        executor.active_orders['AAPL'] = [
            VWAPSlice(datetime.now(), 100, status='filled'),
            VWAPSlice(datetime.now() + timedelta(minutes=10), 100, status='pending'),
            VWAPSlice(datetime.now() + timedelta(minutes=20), 100, status='pending'),
        ]

        result = executor.cancel_active_order('AAPL')

        assert result is True
        assert executor.active_orders['AAPL'][0].status == 'filled'  # Not changed
        assert executor.active_orders['AAPL'][1].status == 'cancelled'
        assert executor.active_orders['AAPL'][2].status == 'cancelled'

    def test_cancel_nonexistent_order(self):
        """Test cancelling order that doesn't exist."""
        executor = VWAPExecutor(MagicMock())

        result = executor.cancel_active_order('NONEXISTENT')

        assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
