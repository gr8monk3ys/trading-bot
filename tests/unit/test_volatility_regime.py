#!/usr/bin/env python3
"""
Unit tests for Volatility Regime Detection.

Tests cover:
1. Regime classification by VIX levels
2. Position size adjustments
3. Stop-loss adjustments
4. Regime change detection
5. VIX caching
6. Edge cases
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.volatility_regime import VolatilityRegimeDetector


class MockQuote:
    """Mock quote for VIX testing."""
    def __init__(self, price):
        self.ask_price = price


class MockBroker:
    """Mock broker for testing."""

    def __init__(self, vix_value=None):
        self.vix_value = vix_value

    async def get_latest_quote(self, symbol):
        if self.vix_value is None:
            raise Exception("VIX not available")
        return MockQuote(self.vix_value)


class TestRegimeClassification:
    """Test VIX-based regime classification."""

    @pytest.mark.asyncio
    async def test_very_low_regime(self):
        """Test very low volatility regime (VIX < 12)."""
        broker = MockBroker(vix_value=10.5)
        detector = VolatilityRegimeDetector(broker)

        regime, adjustments = await detector.get_current_regime()

        assert regime == 'very_low'
        assert adjustments['pos_mult'] == 1.4  # 40% larger positions
        assert adjustments['stop_mult'] == 0.7  # 30% tighter stops
        assert adjustments['max_positions'] == 12
        assert adjustments['trade'] == True

    @pytest.mark.asyncio
    async def test_low_regime(self):
        """Test low volatility regime (VIX 12-15)."""
        broker = MockBroker(vix_value=13.5)
        detector = VolatilityRegimeDetector(broker)

        regime, adjustments = await detector.get_current_regime()

        assert regime == 'low'
        assert adjustments['pos_mult'] == 1.2  # 20% larger positions
        assert adjustments['stop_mult'] == 0.8  # 20% tighter stops
        assert adjustments['max_positions'] == 10

    @pytest.mark.asyncio
    async def test_normal_regime(self):
        """Test normal volatility regime (VIX 15-20)."""
        broker = MockBroker(vix_value=17.5)
        detector = VolatilityRegimeDetector(broker)

        regime, adjustments = await detector.get_current_regime()

        assert regime == 'normal'
        assert adjustments['pos_mult'] == 1.0  # Standard positions
        assert adjustments['stop_mult'] == 1.0  # Standard stops
        assert adjustments['max_positions'] == 8

    @pytest.mark.asyncio
    async def test_elevated_regime(self):
        """Test elevated volatility regime (VIX 20-30)."""
        broker = MockBroker(vix_value=25.0)
        detector = VolatilityRegimeDetector(broker)

        regime, adjustments = await detector.get_current_regime()

        assert regime == 'elevated'
        assert adjustments['pos_mult'] == 0.7  # 30% smaller positions
        assert adjustments['stop_mult'] == 1.2  # 20% wider stops
        assert adjustments['max_positions'] == 5

    @pytest.mark.asyncio
    async def test_high_regime(self):
        """Test high volatility regime (VIX > 30)."""
        broker = MockBroker(vix_value=35.0)
        detector = VolatilityRegimeDetector(broker)

        regime, adjustments = await detector.get_current_regime()

        assert regime == 'high'
        assert adjustments['pos_mult'] == 0.4  # 60% smaller positions
        assert adjustments['stop_mult'] == 1.5  # 50% wider stops
        assert adjustments['max_positions'] == 3
        assert adjustments['trade'] == True  # Still allowed to trade

    @pytest.mark.asyncio
    async def test_boundary_values(self):
        """Test exact boundary values."""
        # Test exactly at thresholds
        test_cases = [
            (12.0, 'low'),       # Exactly 12 -> low (not very_low)
            (15.0, 'normal'),    # Exactly 15 -> normal (not low)
            (20.0, 'elevated'),  # Exactly 20 -> elevated (not normal)
            (30.0, 'high'),      # Exactly 30 -> high (not elevated)
        ]

        for vix, expected_regime in test_cases:
            broker = MockBroker(vix_value=vix)
            detector = VolatilityRegimeDetector(broker)
            regime, _ = await detector.get_current_regime()
            assert regime == expected_regime, f"VIX={vix} should be {expected_regime}, got {regime}"


class TestPositionSizeAdjustment:
    """Test position size adjustment calculations."""

    def test_position_size_increase_low_vol(self):
        """Test position increase in low volatility."""
        detector = VolatilityRegimeDetector(MagicMock())

        base_size = 0.10  # 10%
        adjusted = detector.adjust_position_size(base_size, 1.4)

        assert adjusted == 0.14  # 14%

    def test_position_size_decrease_high_vol(self):
        """Test position decrease in high volatility."""
        detector = VolatilityRegimeDetector(MagicMock())

        base_size = 0.10  # 10%
        adjusted = detector.adjust_position_size(base_size, 0.4)

        assert adjusted == 0.04  # 4%

    def test_position_size_respects_minimum(self):
        """Test position size doesn't go below minimum."""
        detector = VolatilityRegimeDetector(MagicMock())

        base_size = 0.01  # 1%
        adjusted = detector.adjust_position_size(base_size, 0.4)

        assert adjusted >= 0.01  # Min 1%

    def test_position_size_respects_maximum(self):
        """Test position size doesn't exceed maximum."""
        detector = VolatilityRegimeDetector(MagicMock())

        base_size = 0.25  # 25%
        adjusted = detector.adjust_position_size(base_size, 2.0)  # Would be 50%

        assert adjusted <= 0.25  # Max 25%


class TestStopLossAdjustment:
    """Test stop-loss adjustment calculations."""

    def test_stop_loss_tighter_low_vol(self):
        """Test tighter stops in low volatility."""
        detector = VolatilityRegimeDetector(MagicMock())

        base_stop = 0.03  # 3%
        adjusted = detector.adjust_stop_loss(base_stop, 0.7)

        assert adjusted == 0.021  # 2.1%

    def test_stop_loss_wider_high_vol(self):
        """Test wider stops in high volatility."""
        detector = VolatilityRegimeDetector(MagicMock())

        base_stop = 0.03  # 3%
        adjusted = detector.adjust_stop_loss(base_stop, 1.5)

        assert adjusted == 0.045  # 4.5%

    def test_stop_loss_respects_minimum(self):
        """Test stop-loss doesn't go below minimum."""
        detector = VolatilityRegimeDetector(MagicMock())

        base_stop = 0.01  # 1%
        adjusted = detector.adjust_stop_loss(base_stop, 0.5)  # Would be 0.5%

        assert adjusted >= 0.01  # Min 1%

    def test_stop_loss_respects_maximum(self):
        """Test stop-loss doesn't exceed maximum."""
        detector = VolatilityRegimeDetector(MagicMock())

        base_stop = 0.08  # 8%
        adjusted = detector.adjust_stop_loss(base_stop, 2.0)  # Would be 16%

        assert adjusted <= 0.10  # Max 10%


class TestRegimeChangeDetection:
    """Test regime change tracking."""

    @pytest.mark.asyncio
    async def test_regime_change_recorded(self):
        """Test that regime changes are recorded."""
        broker = MockBroker(vix_value=15.0)  # Normal
        detector = VolatilityRegimeDetector(broker, cache_minutes=0)

        # First call - normal regime
        await detector.get_current_regime()
        assert detector.last_regime == 'normal'

        # Change VIX to elevated
        broker.vix_value = 25.0
        detector.last_vix_value = None  # Clear cache

        # Second call - elevated regime
        await detector.get_current_regime()

        # Check change was recorded
        assert len(detector.regime_changes) == 1
        change = detector.regime_changes[0]
        assert change['from'] == 'normal'
        assert change['to'] == 'elevated'
        assert change['vix'] == 25.0

    @pytest.mark.asyncio
    async def test_no_duplicate_regime_changes(self):
        """Test that same regime doesn't create change event."""
        broker = MockBroker(vix_value=17.0)  # Normal
        detector = VolatilityRegimeDetector(broker, cache_minutes=0)

        # Multiple calls at same regime
        await detector.get_current_regime()
        detector.last_vix_value = None  # Clear cache
        await detector.get_current_regime()

        # No regime changes recorded (same regime)
        assert len(detector.regime_changes) == 0


class TestVIXCaching:
    """Test VIX value caching."""

    @pytest.mark.asyncio
    async def test_vix_caching(self):
        """Test that VIX is cached for specified duration."""
        broker = MockBroker(vix_value=20.0)
        detector = VolatilityRegimeDetector(broker, cache_minutes=5)

        # First call fetches VIX
        await detector.get_current_regime()
        assert detector.last_vix_value == 20.0

        # Change broker value
        broker.vix_value = 30.0

        # Second call should use cached value
        await detector.get_current_regime()

        # Should still be 20.0 (cached)
        assert detector.last_vix_value == 20.0

    @pytest.mark.asyncio
    async def test_cache_expires(self):
        """Test that cache expires after duration."""
        broker = MockBroker(vix_value=20.0)
        detector = VolatilityRegimeDetector(broker, cache_minutes=5)

        # First call
        await detector.get_current_regime()

        # Manually expire cache
        detector.last_vix_time = datetime.now() - timedelta(minutes=10)
        broker.vix_value = 30.0

        # Should fetch new value
        await detector.get_current_regime()
        assert detector.last_vix_value == 30.0


class TestCrisisMode:
    """Test crisis mode detection."""

    @pytest.mark.asyncio
    async def test_crisis_mode_when_vix_above_40(self):
        """Test crisis mode is detected when VIX > 40."""
        broker = MockBroker(vix_value=45.0)
        detector = VolatilityRegimeDetector(broker)

        is_crisis = await detector.is_crisis_mode()
        assert is_crisis == True

    @pytest.mark.asyncio
    async def test_no_crisis_mode_when_vix_below_40(self):
        """Test no crisis mode when VIX < 40."""
        broker = MockBroker(vix_value=35.0)
        detector = VolatilityRegimeDetector(broker)

        is_crisis = await detector.is_crisis_mode()
        assert is_crisis == False

    @pytest.mark.asyncio
    async def test_crisis_mode_handles_vix_unavailable(self):
        """Test crisis mode returns False when VIX unavailable."""
        broker = MockBroker(vix_value=None)  # VIX unavailable
        detector = VolatilityRegimeDetector(broker)

        is_crisis = await detector.is_crisis_mode()
        assert is_crisis == False


class TestReduceExposure:
    """Test exposure reduction logic."""

    @pytest.mark.asyncio
    async def test_should_reduce_in_elevated_regime(self):
        """Test should_reduce_exposure returns True in elevated regime."""
        broker = MockBroker(vix_value=25.0)  # Elevated
        detector = VolatilityRegimeDetector(broker)

        should_reduce = await detector.should_reduce_exposure()
        assert should_reduce == True

    @pytest.mark.asyncio
    async def test_should_reduce_in_high_regime(self):
        """Test should_reduce_exposure returns True in high regime."""
        broker = MockBroker(vix_value=35.0)  # High
        detector = VolatilityRegimeDetector(broker)

        should_reduce = await detector.should_reduce_exposure()
        assert should_reduce == True

    @pytest.mark.asyncio
    async def test_should_not_reduce_in_normal_regime(self):
        """Test should_reduce_exposure returns False in normal regime."""
        broker = MockBroker(vix_value=17.0)  # Normal
        detector = VolatilityRegimeDetector(broker)

        should_reduce = await detector.should_reduce_exposure()
        assert should_reduce == False


class TestVIXStatistics:
    """Test VIX historical statistics."""

    def test_empty_statistics(self):
        """Test statistics with no history."""
        detector = VolatilityRegimeDetector(MagicMock())

        stats = detector.get_vix_statistics()

        assert stats['current'] is None
        assert stats['avg_30d'] is None
        assert stats['min_30d'] is None
        assert stats['max_30d'] is None

    @pytest.mark.asyncio
    async def test_statistics_with_history(self):
        """Test statistics with VIX history."""
        broker = MockBroker(vix_value=20.0)
        detector = VolatilityRegimeDetector(broker, cache_minutes=0)

        # Simulate multiple VIX readings
        detector.vix_history = [
            {'time': datetime.now(), 'value': 15.0},
            {'time': datetime.now(), 'value': 20.0},
            {'time': datetime.now(), 'value': 25.0}
        ]
        detector.last_vix_value = 20.0

        stats = detector.get_vix_statistics()

        assert stats['current'] == 20.0
        assert stats['avg_30d'] == 20.0  # (15+20+25)/3
        assert stats['min_30d'] == 15.0
        assert stats['max_30d'] == 25.0
        assert stats['data_points'] == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_broker_exception(self):
        """Test graceful handling when broker throws exception."""
        broker = MockBroker(vix_value=None)  # Will throw exception
        detector = VolatilityRegimeDetector(broker)

        regime, adjustments = await detector.get_current_regime()

        # Should fall back to normal regime
        assert regime == 'normal'
        assert adjustments['pos_mult'] == 1.0

    @pytest.mark.asyncio
    async def test_handles_invalid_vix_value(self):
        """Test handling of invalid VIX values."""
        broker = MockBroker(vix_value=500.0)  # Unrealistic VIX
        detector = VolatilityRegimeDetector(broker)

        # Should reject and use cached/fallback
        vix = await detector._get_vix()

        # Either None or a previously cached valid value
        assert vix is None or vix < 150

    @pytest.mark.asyncio
    async def test_alternative_vix_symbol(self):
        """Test fallback to alternative VIX symbol."""
        broker = MagicMock()

        # First symbol fails, second succeeds
        call_count = 0

        async def mock_get_quote(symbol):
            nonlocal call_count
            call_count += 1
            if symbol == 'VIX' and call_count == 1:
                raise Exception("VIX not found")
            return MockQuote(20.0)

        broker.get_latest_quote = mock_get_quote

        detector = VolatilityRegimeDetector(broker)
        vix = await detector._get_vix()

        # Should have tried alternative symbol
        assert vix == 20.0

    def test_regime_history_tracking(self):
        """Test that regime history is tracked."""
        detector = VolatilityRegimeDetector(MagicMock())

        # Simulate some regime changes
        detector.regime_changes = [
            {'time': datetime.now() - timedelta(hours=2), 'from': 'normal', 'to': 'elevated', 'vix': 25.0},
            {'time': datetime.now() - timedelta(hours=1), 'from': 'elevated', 'to': 'high', 'vix': 35.0}
        ]

        history = detector.get_regime_history()

        assert len(history) == 2
        assert history[0]['from'] == 'normal'
        assert history[1]['to'] == 'high'


class TestThresholdConstants:
    """Test threshold constant values."""

    def test_thresholds_are_ordered(self):
        """Test that thresholds are properly ordered."""
        assert VolatilityRegimeDetector.VERY_LOW_THRESHOLD < VolatilityRegimeDetector.LOW_THRESHOLD
        assert VolatilityRegimeDetector.LOW_THRESHOLD < VolatilityRegimeDetector.NORMAL_THRESHOLD
        assert VolatilityRegimeDetector.NORMAL_THRESHOLD < VolatilityRegimeDetector.ELEVATED_THRESHOLD

    def test_thresholds_are_positive(self):
        """Test that all thresholds are positive."""
        assert VolatilityRegimeDetector.VERY_LOW_THRESHOLD > 0
        assert VolatilityRegimeDetector.LOW_THRESHOLD > 0
        assert VolatilityRegimeDetector.NORMAL_THRESHOLD > 0
        assert VolatilityRegimeDetector.ELEVATED_THRESHOLD > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
