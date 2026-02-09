#!/usr/bin/env python3
"""
Unit tests for utils/vwap_executor.py

Tests VWAPExecutor class for:
- VWAP order execution
- Slice creation and scheduling
- Volume profile weighting
- Participation rate adjustment
- Execution statistics
- Order cancellation
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from utils.vwap_executor import (
    VWAPExecutor,
    VWAPResult,
    VWAPSlice,
    execute_vwap,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_broker():
    """Create mock broker with async methods."""
    broker = MagicMock()
    broker.get_bars = AsyncMock()
    broker.get_latest_quote = AsyncMock()
    broker.get_last_price = AsyncMock()
    broker.submit_order_advanced = AsyncMock()
    return broker


@pytest.fixture
def executor(mock_broker):
    """Create VWAP executor with mock broker."""
    return VWAPExecutor(mock_broker)


@pytest.fixture
def executor_custom(mock_broker):
    """Create VWAP executor with custom settings."""
    return VWAPExecutor(
        mock_broker,
        default_slices=5,
        min_slice_qty=10.0,
        max_participation_rate=0.05,
    )


@pytest.fixture
def mock_quote():
    """Create mock quote."""
    quote = MagicMock()
    quote.ask_price = 151.0
    quote.bid_price = 150.0
    return quote


@pytest.fixture
def mock_bar():
    """Create mock bar factory."""

    def _make_bar(volume=100000, close=150.0):
        bar = MagicMock()
        bar.volume = volume
        bar.close = close
        return bar

    return _make_bar


# ============================================================================
# VWAPSlice Tests
# ============================================================================


class TestVWAPSlice:
    """Test VWAPSlice dataclass."""

    def test_slice_creation(self):
        """Test creating a VWAP slice."""
        scheduled = datetime.now()
        slice_obj = VWAPSlice(scheduled_time=scheduled, target_qty=100.0)
        assert slice_obj.scheduled_time == scheduled
        assert slice_obj.target_qty == 100.0
        assert slice_obj.executed_qty == 0.0
        assert slice_obj.avg_price == 0.0
        assert slice_obj.status == "pending"

    def test_slice_with_all_fields(self):
        """Test creating slice with all fields."""
        scheduled = datetime.now()
        slice_obj = VWAPSlice(
            scheduled_time=scheduled,
            target_qty=100.0,
            executed_qty=95.0,
            avg_price=150.50,
            status="filled",
        )
        assert slice_obj.executed_qty == 95.0
        assert slice_obj.avg_price == 150.50
        assert slice_obj.status == "filled"


# ============================================================================
# VWAPResult Tests
# ============================================================================


class TestVWAPResult:
    """Test VWAPResult dataclass."""

    def test_result_creation(self):
        """Test creating a VWAP result."""
        result = VWAPResult(
            symbol="AAPL",
            side="buy",
            total_qty=1000.0,
            executed_qty=995.0,
            avg_price=150.25,
            vwap_benchmark=150.00,
            slippage_bps=16.67,
            duration_minutes=60,
            num_slices=10,
            slices_filled=9,
            status="completed",
        )
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.total_qty == 1000.0
        assert result.executed_qty == 995.0
        assert result.avg_price == 150.25
        assert result.vwap_benchmark == 150.00
        assert result.slippage_bps == 16.67
        assert result.status == "completed"


# ============================================================================
# VWAPExecutor Initialization Tests
# ============================================================================


class TestVWAPExecutorInit:
    """Test VWAPExecutor initialization."""

    def test_default_init(self, executor):
        """Test default initialization values."""
        assert executor.default_slices == 10
        assert executor.min_slice_qty == 1.0
        assert executor.max_participation_rate == 0.10
        assert executor.active_orders == {}
        assert executor.execution_history == []

    def test_custom_init(self, executor_custom):
        """Test custom initialization values."""
        assert executor_custom.default_slices == 5
        assert executor_custom.min_slice_qty == 10.0
        assert executor_custom.max_participation_rate == 0.05

    def test_default_volume_profile_exists(self, executor):
        """Test that default volume profile is defined."""
        assert hasattr(executor, "DEFAULT_VOLUME_PROFILE")
        assert len(executor.DEFAULT_VOLUME_PROFILE) > 0
        # Opening should have high volume
        assert executor.DEFAULT_VOLUME_PROFILE.get("09:30", 0) > 0
        # Lunch should have low volume
        assert executor.DEFAULT_VOLUME_PROFILE.get("12:00", 0) < executor.DEFAULT_VOLUME_PROFILE.get(
            "09:30", 0
        )


# ============================================================================
# Create Slices Tests
# ============================================================================


class TestCreateSlices:
    """Test _create_slices method."""

    def test_equal_weight_slices(self, executor):
        """Test creating slices with equal weights."""
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        slices = executor._create_slices("AAPL", "buy", 100.0, 60, 5, weights)

        assert len(slices) == 5
        total_qty = sum(s.target_qty for s in slices)
        assert abs(total_qty - 100.0) < 0.1

    def test_varied_weight_slices(self, executor):
        """Test creating slices with varied weights."""
        weights = [0.1, 0.1, 0.05, 0.05, 0.3, 0.4]  # Higher at end
        slices = executor._create_slices("AAPL", "buy", 100.0, 60, 6, weights)

        assert len(slices) == 6
        total_qty = sum(s.target_qty for s in slices)
        assert abs(total_qty - 100.0) < 0.1

    def test_slice_scheduling(self, executor):
        """Test that slices are scheduled correctly."""
        weights = [0.25, 0.25, 0.25, 0.25]
        slices = executor._create_slices("AAPL", "buy", 100.0, 60, 4, weights)

        # Check scheduling interval (60 min / 4 slices = 15 min apart)
        for i in range(1, len(slices)):
            time_diff = (slices[i].scheduled_time - slices[i - 1].scheduled_time).total_seconds()
            assert abs(time_diff - 900) < 60  # 15 minutes +/- 1 min tolerance

    def test_minimum_slice_qty(self, executor):
        """Test minimum slice quantity is enforced."""
        weights = [0.01, 0.99]  # First slice would be 1 share
        slices = executor._create_slices("AAPL", "buy", 100.0, 60, 2, weights)

        # First slice should be at least min_slice_qty
        assert slices[0].target_qty >= executor.min_slice_qty

    def test_single_slice(self, executor):
        """Test creating a single slice."""
        weights = [1.0]
        slices = executor._create_slices("AAPL", "buy", 100.0, 60, 1, weights)

        assert len(slices) == 1
        assert slices[0].target_qty == 100.0

    def test_slices_all_pending(self, executor):
        """Test all slices start as pending."""
        weights = [0.33, 0.33, 0.34]
        slices = executor._create_slices("AAPL", "buy", 100.0, 60, 3, weights)

        for slice_obj in slices:
            assert slice_obj.status == "pending"


# ============================================================================
# Volume Weights Tests
# ============================================================================


class TestGetVolumeWeights:
    """Test _get_volume_weights method."""

    @pytest.mark.asyncio
    async def test_uses_historical_data(self, executor, mock_broker, mock_bar):
        """Test using historical bar data for weights."""
        # Create bars with different volumes
        bars = [mock_bar(v) for v in [100, 200, 150, 250, 300]]
        mock_broker.get_bars.return_value = bars

        weights = await executor._get_volume_weights("AAPL", 5, 60)

        assert len(weights) == 5
        assert sum(weights) == pytest.approx(1.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_falls_back_to_default_profile(self, executor, mock_broker):
        """Test fallback to default profile when no bars available."""
        mock_broker.get_bars.return_value = None

        weights = await executor._get_volume_weights("AAPL", 5, 60)

        assert len(weights) == 5

    @pytest.mark.asyncio
    async def test_handles_bar_exception(self, executor, mock_broker):
        """Test handling of exceptions when fetching bars."""
        mock_broker.get_bars.side_effect = Exception("API Error")

        weights = await executor._get_volume_weights("AAPL", 5, 60)

        # Should fall back to default and not raise
        assert len(weights) == 5

    @pytest.mark.asyncio
    async def test_insufficient_bars(self, executor, mock_broker, mock_bar):
        """Test with fewer bars than requested slices."""
        bars = [mock_bar(100), mock_bar(200)]  # Only 2 bars
        mock_broker.get_bars.return_value = bars

        weights = await executor._get_volume_weights("AAPL", 5, 60)

        # Should fall back to default profile
        assert len(weights) == 5


# ============================================================================
# Participation Rate Tests
# ============================================================================


class TestAdjustForParticipation:
    """Test _adjust_for_participation method."""

    @pytest.mark.asyncio
    async def test_reduces_qty_when_exceeds_rate(self, executor, mock_broker, mock_bar):
        """Test quantity reduction when exceeding participation rate."""
        bars = [mock_bar(1000)] * 5  # Average volume 1000
        mock_broker.get_bars.return_value = bars

        # Participation rate of 10%, so max qty should be 100
        adjusted = await executor._adjust_for_participation("AAPL", 200.0, 0.10)

        assert adjusted <= 100.0

    @pytest.mark.asyncio
    async def test_keeps_qty_when_under_rate(self, executor, mock_broker, mock_bar):
        """Test quantity unchanged when under participation rate."""
        bars = [mock_bar(10000)] * 5  # High volume
        mock_broker.get_bars.return_value = bars

        # Request 100 shares, rate is 10%, max would be 1000
        adjusted = await executor._adjust_for_participation("AAPL", 100.0, 0.10)

        assert adjusted == 100.0

    @pytest.mark.asyncio
    async def test_handles_no_bars(self, executor, mock_broker):
        """Test handling when no bars available."""
        mock_broker.get_bars.return_value = None

        adjusted = await executor._adjust_for_participation("AAPL", 100.0, 0.10)

        # Should return original quantity
        assert adjusted == 100.0

    @pytest.mark.asyncio
    async def test_handles_exception(self, executor, mock_broker):
        """Test handling of exceptions."""
        mock_broker.get_bars.side_effect = Exception("API Error")

        adjusted = await executor._adjust_for_participation("AAPL", 100.0, 0.10)

        # Should return original quantity
        assert adjusted == 100.0


# ============================================================================
# Calculate VWAP Tests
# ============================================================================


class TestCalculateVWAP:
    """Test _calculate_vwap method."""

    def test_basic_vwap(self, executor):
        """Test basic VWAP calculation."""
        prices = [100.0, 102.0, 101.0]
        volumes = [1000, 2000, 1000]

        # VWAP = (100*1000 + 102*2000 + 101*1000) / 4000
        # = (100000 + 204000 + 101000) / 4000 = 101.25
        vwap = executor._calculate_vwap(prices, volumes)

        assert vwap == pytest.approx(101.25, rel=0.01)

    def test_equal_volumes(self, executor):
        """Test VWAP with equal volumes (should equal simple average)."""
        prices = [100.0, 110.0, 105.0]
        volumes = [100, 100, 100]

        vwap = executor._calculate_vwap(prices, volumes)
        expected = np.mean(prices)

        assert vwap == pytest.approx(expected, rel=0.01)

    def test_empty_prices(self, executor):
        """Test VWAP with empty inputs."""
        vwap = executor._calculate_vwap([], [])
        assert vwap == 0.0

    def test_zero_total_volume(self, executor):
        """Test VWAP with zero total volume."""
        prices = [100.0, 110.0, 105.0]
        volumes = [0, 0, 0]

        vwap = executor._calculate_vwap(prices, volumes)

        # Should fall back to simple average
        assert vwap == pytest.approx(np.mean(prices), rel=0.01)

    def test_single_price(self, executor):
        """Test VWAP with single price."""
        vwap = executor._calculate_vwap([150.0], [1000])
        assert vwap == 150.0


# ============================================================================
# Get Current Price Tests
# ============================================================================


class TestGetCurrentPrice:
    """Test _get_current_price method."""

    @pytest.mark.asyncio
    async def test_uses_quote(self, executor, mock_broker, mock_quote):
        """Test getting price from quote (bid/ask midpoint)."""
        mock_broker.get_latest_quote.return_value = mock_quote

        price = await executor._get_current_price("AAPL")

        assert price == 150.5  # (151 + 150) / 2

    @pytest.mark.asyncio
    async def test_falls_back_to_last_price(self, executor, mock_broker):
        """Test fallback to last price when quote fails."""
        mock_broker.get_latest_quote.side_effect = Exception("No quote")
        mock_broker.get_last_price.return_value = 148.0

        price = await executor._get_current_price("AAPL")

        assert price == 148.0

    @pytest.mark.asyncio
    async def test_returns_zero_on_all_failures(self, executor, mock_broker):
        """Test returns zero when all price methods fail."""
        mock_broker.get_latest_quote.side_effect = Exception("No quote")
        mock_broker.get_last_price.side_effect = Exception("No price")

        price = await executor._get_current_price("AAPL")

        assert price == 0.0


# ============================================================================
# Execute Slice Tests
# ============================================================================


class TestExecuteSlice:
    """Test _execute_slice method."""

    @pytest.mark.asyncio
    async def test_successful_slice_execution(self, executor, mock_broker, mock_quote):
        """Test successful slice execution."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_result = MagicMock()
        mock_result.id = "order123"
        mock_broker.submit_order_advanced.return_value = mock_result

        result = await executor._execute_slice("AAPL", "buy", 100.0)

        assert result is not None
        assert result["qty"] == 100.0
        assert result["price"] == 150.5
        assert result["order_id"] == "order123"

    @pytest.mark.asyncio
    async def test_failed_slice_execution(self, executor, mock_broker):
        """Test failed slice execution."""
        mock_broker.submit_order_advanced.return_value = None

        result = await executor._execute_slice("AAPL", "buy", 100.0)

        assert result is None

    @pytest.mark.asyncio
    async def test_exception_during_slice(self, executor, mock_broker):
        """Test handling of exceptions during slice execution."""
        mock_broker.submit_order_advanced.side_effect = Exception("Order error")

        result = await executor._execute_slice("AAPL", "buy", 100.0)

        assert result is None


# ============================================================================
# Execute Single Order Tests
# ============================================================================


class TestExecuteSingleOrder:
    """Test _execute_single_order method."""

    @pytest.mark.asyncio
    async def test_successful_single_order(self, executor, mock_broker, mock_quote):
        """Test successful single order execution."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_broker.submit_order_advanced.return_value = MagicMock()

        result = await executor._execute_single_order("AAPL", "buy", 10.0)

        assert result.status == "completed"
        assert result.executed_qty == 10.0
        assert result.num_slices == 1
        assert result.slices_filled == 1

    @pytest.mark.asyncio
    async def test_failed_single_order(self, executor, mock_broker, mock_quote):
        """Test failed single order execution."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_broker.submit_order_advanced.return_value = None

        result = await executor._execute_single_order("AAPL", "buy", 10.0)

        assert result.status == "cancelled"
        assert result.executed_qty == 0
        assert result.slices_filled == 0


# ============================================================================
# Execute VWAP Order Tests
# ============================================================================


class TestExecuteVWAPOrder:
    """Test execute_vwap_order method."""

    @pytest.mark.asyncio
    async def test_small_order_uses_single_execution(self, executor, mock_broker, mock_quote):
        """Test small orders execute as single order."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_broker.submit_order_advanced.return_value = MagicMock()

        # Order less than min_slice_qty (1.0)
        result = await executor.execute_vwap_order("AAPL", "buy", 0.5, duration_minutes=60)

        assert result.num_slices == 1

    @pytest.mark.asyncio
    async def test_calculates_slices_automatically(self, executor, mock_broker, mock_quote):
        """Test automatic slice calculation."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_broker.get_bars.return_value = None
        mock_result = MagicMock()
        mock_result.id = "order123"
        mock_broker.submit_order_advanced.return_value = mock_result

        with patch("asyncio.sleep", return_value=None):
            result = await executor.execute_vwap_order(
                "AAPL", "buy", 100.0, duration_minutes=60, num_slices=2
            )

        assert result.num_slices == 2

    @pytest.mark.asyncio
    async def test_status_completed(self, executor, mock_broker, mock_quote):
        """Test completed status when fully executed."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_broker.get_bars.return_value = None
        mock_result = MagicMock()
        mock_result.id = "order123"
        mock_broker.submit_order_advanced.return_value = mock_result

        with patch("asyncio.sleep", return_value=None):
            result = await executor.execute_vwap_order(
                "AAPL", "buy", 10.0, duration_minutes=10, num_slices=2
            )

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_status_partial(self, executor, mock_broker, mock_quote):
        """Test partial status when some slices fail."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_broker.get_bars.return_value = None
        # First slice succeeds, second fails
        mock_result = MagicMock()
        mock_result.id = "order123"
        mock_broker.submit_order_advanced.side_effect = [mock_result, None]

        with patch("asyncio.sleep", return_value=None):
            result = await executor.execute_vwap_order(
                "AAPL", "buy", 10.0, duration_minutes=10, num_slices=2
            )

        assert result.status == "partial"

    @pytest.mark.asyncio
    async def test_status_cancelled(self, executor, mock_broker, mock_quote):
        """Test cancelled status when all slices fail."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_broker.get_bars.return_value = None
        mock_broker.submit_order_advanced.return_value = None

        with patch("asyncio.sleep", return_value=None):
            result = await executor.execute_vwap_order(
                "AAPL", "buy", 10.0, duration_minutes=10, num_slices=2
            )

        assert result.status == "cancelled"


# ============================================================================
# Execution Stats Tests
# ============================================================================


class TestGetExecutionStats:
    """Test get_execution_stats method."""

    def test_empty_history(self, executor):
        """Test stats with empty history."""
        stats = executor.get_execution_stats()
        assert stats["total_executions"] == 0
        assert stats["avg_slippage_bps"] == 0
        assert stats["completion_rate"] == 0

    def test_with_executions(self, executor):
        """Test stats with execution history."""
        executor.execution_history = [
            VWAPResult(
                symbol="AAPL",
                side="buy",
                total_qty=100,
                executed_qty=100,
                avg_price=150,
                vwap_benchmark=149.5,
                slippage_bps=10,
                duration_minutes=60,
                num_slices=5,
                slices_filled=5,
                status="completed",
            ),
            VWAPResult(
                symbol="MSFT",
                side="buy",
                total_qty=50,
                executed_qty=50,
                avg_price=300,
                vwap_benchmark=299,
                slippage_bps=20,
                duration_minutes=30,
                num_slices=3,
                slices_filled=3,
                status="completed",
            ),
        ]

        stats = executor.get_execution_stats()

        assert stats["total_executions"] == 2
        assert stats["avg_slippage_bps"] == 15  # (10 + 20) / 2
        assert stats["min_slippage_bps"] == 10
        assert stats["max_slippage_bps"] == 20
        assert stats["completion_rate"] == 1.0
        assert stats["total_volume"] == 150


# ============================================================================
# Cancel Active Order Tests
# ============================================================================


class TestCancelActiveOrder:
    """Test cancel_active_order method."""

    def test_cancel_existing_order(self, executor):
        """Test cancelling an existing active order."""
        # Set up active order
        slices = [
            VWAPSlice(scheduled_time=datetime.now(), target_qty=50, status="pending"),
            VWAPSlice(scheduled_time=datetime.now(), target_qty=50, status="pending"),
        ]
        executor.active_orders["AAPL"] = slices

        result = executor.cancel_active_order("AAPL")

        assert result
        for slice_obj in executor.active_orders["AAPL"]:
            assert slice_obj.status == "cancelled"

    def test_cancel_nonexistent_order(self, executor):
        """Test cancelling a non-existent order."""
        result = executor.cancel_active_order("AAPL")
        assert not result

    def test_only_pending_slices_cancelled(self, executor):
        """Test that only pending slices are cancelled."""
        slices = [
            VWAPSlice(
                scheduled_time=datetime.now(), target_qty=50, status="filled", executed_qty=50
            ),
            VWAPSlice(scheduled_time=datetime.now(), target_qty=50, status="pending"),
        ]
        executor.active_orders["AAPL"] = slices

        executor.cancel_active_order("AAPL")

        assert slices[0].status == "filled"  # Unchanged
        assert slices[1].status == "cancelled"


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestExecuteVWAPFunction:
    """Test execute_vwap convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function(self, mock_broker, mock_quote):
        """Test the convenience function creates executor and executes."""
        mock_broker.get_latest_quote.return_value = mock_quote
        mock_broker.get_bars.return_value = None
        mock_broker.submit_order_advanced.return_value = MagicMock()

        with patch("asyncio.sleep", return_value=None):
            result = await execute_vwap(mock_broker, "AAPL", "buy", 10.0, duration_minutes=10)

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.side == "buy"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_vwap_with_single_execution(self, executor):
        """Test VWAP calculation with single trade."""
        vwap = executor._calculate_vwap([150.0], [1000])
        assert vwap == 150.0

    def test_vwap_with_large_volume_differences(self, executor):
        """Test VWAP when volumes are vastly different."""
        prices = [100.0, 200.0]
        volumes = [1, 1000000]  # Second trade dominates

        vwap = executor._calculate_vwap(prices, volumes)

        # Should be very close to 200 since second trade has much more volume
        assert vwap > 199.0

    def test_slices_weight_normalization(self, executor):
        """Test that weights are properly normalized."""
        # Weights don't sum to 1
        weights = [1.0, 2.0, 3.0, 4.0]  # Sum = 10
        slices = executor._create_slices("AAPL", "buy", 100.0, 60, 4, weights)

        total_qty = sum(s.target_qty for s in slices)
        assert abs(total_qty - 100.0) < 0.1  # Should still total to 100
