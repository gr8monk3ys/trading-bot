"""
Tests for the Partial Fill Tracker.

Tests cover:
- Policy configuration
- Order tracking
- Fill recording
- Callback notifications
- Auto-resubmit logic
- Statistics calculation
"""

from datetime import datetime, timedelta

import pytest

from utils.partial_fill_tracker import (
    OrderTrackingRecord,
    PartialFillEvent,
    PartialFillPolicy,
    PartialFillStatistics,
    PartialFillTracker,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tracker():
    """Create a basic PartialFillTracker."""
    return PartialFillTracker()


@pytest.fixture
def tracker_auto_resubmit():
    """Create a tracker with auto-resubmit policy."""
    return PartialFillTracker(policy=PartialFillPolicy.AUTO_RESUBMIT)


@pytest.fixture
def tracker_with_order(tracker):
    """Create a tracker with a tracked order."""
    tracker.track_order(
        order_id="order-123",
        symbol="AAPL",
        side="buy",
        requested_qty=100,
    )
    return tracker


# ============================================================================
# DATACLASS TESTS
# ============================================================================


class TestPartialFillEvent:
    """Tests for PartialFillEvent dataclass."""

    def test_event_creation(self):
        """Test creating a PartialFillEvent."""
        event = PartialFillEvent(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            requested_qty=100,
            filled_qty=60,
            unfilled_qty=40,
            fill_price=150.50,
            timestamp=datetime.now(),
            event_type="partial_fill",
        )

        assert event.order_id == "order-123"
        assert event.symbol == "AAPL"
        assert event.filled_qty == 60
        assert event.unfilled_qty == 40

    def test_fill_rate_property(self):
        """Test fill_rate calculation."""
        event = PartialFillEvent(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            requested_qty=100,
            filled_qty=75,
            unfilled_qty=25,
            fill_price=150.0,
            timestamp=datetime.now(),
            event_type="partial_fill",
        )

        assert event.fill_rate == 0.75

    def test_fill_rate_zero_requested(self):
        """Test fill_rate with zero requested quantity."""
        event = PartialFillEvent(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            requested_qty=0,
            filled_qty=0,
            unfilled_qty=0,
            fill_price=0,
            timestamp=datetime.now(),
            event_type="rejected",
        )

        assert event.fill_rate == 0.0

    def test_is_complete_property(self):
        """Test is_complete property."""
        partial_event = PartialFillEvent(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            requested_qty=100,
            filled_qty=60,
            unfilled_qty=40,
            fill_price=150.0,
            timestamp=datetime.now(),
            event_type="partial_fill",
        )
        assert partial_event.is_complete is False

        complete_event = PartialFillEvent(
            order_id="order-456",
            symbol="AAPL",
            side="buy",
            requested_qty=100,
            filled_qty=100,
            unfilled_qty=0,
            fill_price=150.0,
            timestamp=datetime.now(),
            event_type="fill",
        )
        assert complete_event.is_complete is True


class TestOrderTrackingRecord:
    """Tests for OrderTrackingRecord dataclass."""

    def test_record_creation(self):
        """Test creating an OrderTrackingRecord."""
        record = OrderTrackingRecord(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            requested_qty=100,
        )

        assert record.order_id == "order-123"
        assert record.filled_qty == 0.0
        assert record.status == "pending"

    def test_unfilled_qty_property(self):
        """Test unfilled_qty calculation."""
        record = OrderTrackingRecord(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            requested_qty=100,
            filled_qty=60,
        )

        assert record.unfilled_qty == 40

    def test_unfilled_qty_never_negative(self):
        """Test unfilled_qty never goes negative (overfill case)."""
        record = OrderTrackingRecord(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            requested_qty=100,
            filled_qty=150,  # Overfill (shouldn't happen but handle gracefully)
        )

        assert record.unfilled_qty == 0


class TestPartialFillStatistics:
    """Tests for PartialFillStatistics dataclass."""

    def test_statistics_creation(self):
        """Test creating PartialFillStatistics."""
        stats = PartialFillStatistics(
            total_orders=10,
            fully_filled_orders=8,
            partially_filled_orders=2,
            canceled_orders=0,
            rejected_orders=0,
            total_requested_qty=1000,
            total_filled_qty=920,
            total_unfilled_qty=80,
            average_fill_rate=0.92,
            orders_below_90_pct_fill=2,
            auto_resubmits_triggered=1,
        )

        assert stats.total_orders == 10
        assert stats.average_fill_rate == 0.92

    def test_statistics_to_dict(self):
        """Test converting statistics to dict."""
        stats = PartialFillStatistics(
            total_orders=5,
            fully_filled_orders=4,
            partially_filled_orders=1,
            canceled_orders=0,
            rejected_orders=0,
            total_requested_qty=500,
            total_filled_qty=480,
            total_unfilled_qty=20,
            average_fill_rate=0.96,
            orders_below_90_pct_fill=1,
            auto_resubmits_triggered=0,
        )

        d = stats.to_dict()
        assert isinstance(d, dict)
        assert d["total_orders"] == 5
        assert d["average_fill_rate"] == 0.96


# ============================================================================
# TRACKER INITIALIZATION TESTS
# ============================================================================


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_default_policy(self):
        """Test default policy is ALERT_ONLY."""
        tracker = PartialFillTracker()
        assert tracker.policy == PartialFillPolicy.ALERT_ONLY

    def test_custom_policy(self):
        """Test setting custom policy."""
        tracker = PartialFillTracker(policy=PartialFillPolicy.AUTO_RESUBMIT)
        assert tracker.policy == PartialFillPolicy.AUTO_RESUBMIT

    def test_custom_parameters(self):
        """Test setting custom parameters."""
        tracker = PartialFillTracker(
            policy=PartialFillPolicy.TRACK_ONLY,
            max_resubmit_attempts=5,
            min_resubmit_qty=10.0,
            fill_rate_threshold=0.80,
        )

        assert tracker.max_resubmit_attempts == 5
        assert tracker.min_resubmit_qty == 10.0
        assert tracker.fill_rate_threshold == 0.80


class TestPolicyConfiguration:
    """Tests for policy configuration."""

    def test_set_policy(self, tracker):
        """Test changing policy."""
        tracker.set_policy(PartialFillPolicy.CANCEL_REMAINDER)
        assert tracker.policy == PartialFillPolicy.CANCEL_REMAINDER

    def test_policy_enum_values(self):
        """Test all policy enum values."""
        assert PartialFillPolicy.ALERT_ONLY.value == "alert_only"
        assert PartialFillPolicy.AUTO_RESUBMIT.value == "auto_resubmit"
        assert PartialFillPolicy.CANCEL_REMAINDER.value == "cancel_remainder"
        assert PartialFillPolicy.TRACK_ONLY.value == "track_only"


# ============================================================================
# ORDER TRACKING TESTS
# ============================================================================


class TestOrderTracking:
    """Tests for order tracking."""

    def test_track_order(self, tracker):
        """Test tracking an order."""
        tracker.track_order(
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            requested_qty=100,
        )

        status = tracker.get_order_status("order-123")
        assert status is not None
        assert status["symbol"] == "AAPL"
        assert status["requested_qty"] == 100
        assert status["filled_qty"] == 0
        assert status["status"] == "pending"

    def test_track_multiple_orders(self, tracker):
        """Test tracking multiple orders."""
        tracker.track_order("order-1", "AAPL", "buy", 100)
        tracker.track_order("order-2", "MSFT", "sell", 50)

        assert tracker.get_order_status("order-1") is not None
        assert tracker.get_order_status("order-2") is not None

    def test_get_nonexistent_order(self, tracker):
        """Test getting status of untracked order."""
        status = tracker.get_order_status("nonexistent")
        assert status is None


# ============================================================================
# FILL RECORDING TESTS
# ============================================================================


class TestFillRecording:
    """Tests for fill recording."""

    @pytest.mark.asyncio
    async def test_record_full_fill(self, tracker_with_order):
        """Test recording a full fill."""
        event = await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=100,
            fill_price=150.0,
            is_final=True,
        )

        assert event is not None
        assert event.event_type == "fill"
        assert event.filled_qty == 100
        assert event.unfilled_qty == 0

        status = tracker_with_order.get_order_status("order-123")
        assert status["status"] == "filled"

    @pytest.mark.asyncio
    async def test_record_partial_fill(self, tracker_with_order):
        """Test recording a partial fill."""
        event = await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=60,
            fill_price=150.0,
            is_final=False,
        )

        assert event is not None
        assert event.event_type == "partial_fill"
        assert event.filled_qty == 60
        assert event.unfilled_qty == 40

        status = tracker_with_order.get_order_status("order-123")
        assert status["status"] == "partial"

    @pytest.mark.asyncio
    async def test_record_final_partial_fill(self, tracker_with_order):
        """Test recording a final partial fill (order closed but not fully filled)."""
        # First partial
        await tracker_with_order.record_fill("order-123", 60, 150.0, is_final=False)

        # Final partial (no more fills coming, but not complete)
        event = await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=70,  # Only 70 of 100 filled
            fill_price=150.0,
            is_final=True,
            status="partial",
        )

        assert event.unfilled_qty == 30
        status = tracker_with_order.get_order_status("order-123")
        assert status["status"] == "partial"

    @pytest.mark.asyncio
    async def test_record_canceled_order(self, tracker_with_order):
        """Test recording a canceled order."""
        # Partial fill before cancel
        await tracker_with_order.record_fill("order-123", 50, 150.0, is_final=False)

        # Then canceled
        event = await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=50,
            fill_price=150.0,
            is_final=True,
            status="canceled",
        )

        assert event.event_type == "canceled"
        status = tracker_with_order.get_order_status("order-123")
        assert status["status"] == "canceled"

    @pytest.mark.asyncio
    async def test_record_rejected_order(self, tracker_with_order):
        """Test recording a rejected order."""
        event = await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=0,
            fill_price=0,
            is_final=True,
            status="rejected",
        )

        assert event.event_type == "rejected"
        status = tracker_with_order.get_order_status("order-123")
        assert status["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_record_fill_untracked_order(self, tracker):
        """Test recording fill for untracked order returns None."""
        event = await tracker.record_fill("unknown-order", 100, 150.0, is_final=True)
        assert event is None


# ============================================================================
# CALLBACK TESTS
# ============================================================================


class TestCallbacks:
    """Tests for callback functionality."""

    @pytest.mark.asyncio
    async def test_callback_on_partial_fill(self, tracker_with_order):
        """Test callback is called on partial fill."""
        callback_called = []

        async def callback(event):
            callback_called.append(event)

        tracker_with_order.register_callback(callback)
        tracker_with_order.fill_rate_threshold = 0.95  # Ensure partial triggers

        await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=60,  # 60% fill
            fill_price=150.0,
            is_final=True,
        )

        assert len(callback_called) == 1
        assert callback_called[0].filled_qty == 60

    @pytest.mark.asyncio
    async def test_sync_callback(self, tracker_with_order):
        """Test synchronous callback works."""
        callback_called = []

        def callback(event):
            callback_called.append(event)

        tracker_with_order.register_callback(callback)
        tracker_with_order.fill_rate_threshold = 0.95

        await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=60,
            fill_price=150.0,
            is_final=True,
        )

        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self, tracker_with_order):
        """Test multiple callbacks all called."""
        results = []

        async def callback1(event):
            results.append("cb1")

        async def callback2(event):
            results.append("cb2")

        tracker_with_order.register_callback(callback1)
        tracker_with_order.register_callback(callback2)
        tracker_with_order.fill_rate_threshold = 0.95

        await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=60,
            fill_price=150.0,
            is_final=True,
        )

        assert "cb1" in results
        assert "cb2" in results

    @pytest.mark.asyncio
    async def test_callback_above_threshold_not_called(self, tracker_with_order):
        """Test callback not called if fill rate above threshold."""
        callback_called = []

        async def callback(event):
            callback_called.append(event)

        tracker_with_order.register_callback(callback)
        tracker_with_order.fill_rate_threshold = 0.50  # 50% threshold

        await tracker_with_order.record_fill(
            order_id="order-123",
            filled_qty=60,  # 60% fill - above threshold
            fill_price=150.0,
            is_final=True,
        )

        # Callback should NOT be called because fill rate (60%) >= threshold (50%)
        assert len(callback_called) == 0


# ============================================================================
# AUTO-RESUBMIT TESTS
# ============================================================================


class TestAutoResubmit:
    """Tests for auto-resubmit functionality."""

    @pytest.mark.asyncio
    async def test_auto_resubmit_triggered(self, tracker_auto_resubmit):
        """Test auto-resubmit is triggered on partial fill."""
        tracker_auto_resubmit.track_order("order-123", "AAPL", "buy", 100)

        resubmit_calls = []

        async def resubmit_callback(symbol, side, qty):
            resubmit_calls.append((symbol, side, qty))
            return "new-order-456"

        tracker_auto_resubmit.set_resubmit_callback(resubmit_callback)

        await tracker_auto_resubmit.record_fill(
            order_id="order-123",
            filled_qty=60,  # 60% fill, 40 unfilled
            fill_price=150.0,
            is_final=True,
        )

        assert len(resubmit_calls) == 1
        assert resubmit_calls[0] == ("AAPL", "buy", 40.0)

    @pytest.mark.asyncio
    async def test_auto_resubmit_tracks_new_order(self, tracker_auto_resubmit):
        """Test auto-resubmit creates tracking for new order."""
        tracker_auto_resubmit.track_order("order-123", "AAPL", "buy", 100)

        async def resubmit_callback(symbol, side, qty):
            return "new-order-456"

        tracker_auto_resubmit.set_resubmit_callback(resubmit_callback)

        await tracker_auto_resubmit.record_fill(
            order_id="order-123",
            filled_qty=60,
            fill_price=150.0,
            is_final=True,
        )

        # New order should be tracked
        new_status = tracker_auto_resubmit.get_order_status("new-order-456")
        assert new_status is not None
        assert new_status["requested_qty"] == 40  # Unfilled from original

    @pytest.mark.asyncio
    async def test_auto_resubmit_max_attempts(self, tracker_auto_resubmit):
        """Test auto-resubmit stops after max attempts on same original order."""
        tracker_auto_resubmit.max_resubmit_attempts = 2
        tracker_auto_resubmit.track_order("order-123", "AAPL", "buy", 100)

        resubmit_count = [0]

        async def resubmit_callback(symbol, side, qty):
            resubmit_count[0] += 1
            return f"new-order-{resubmit_count[0]}"

        tracker_auto_resubmit.set_resubmit_callback(resubmit_callback)

        # First partial fill - resubmit #1 (original order, attempt 1)
        await tracker_auto_resubmit.record_fill("order-123", 60, 150.0, is_final=True)

        # The original order had 1 resubmit attempt recorded
        original_status = tracker_auto_resubmit.get_order_status("order-123")
        assert original_status["resubmit_attempts"] == 1

        # Second partial fill on child order - this is a NEW order with its own count
        # Each child order has its own resubmit counter starting at 0
        await tracker_auto_resubmit.record_fill("new-order-1", 20, 150.0, is_final=True)

        child1_status = tracker_auto_resubmit.get_order_status("new-order-1")
        assert child1_status["resubmit_attempts"] == 1

        # Total resubmits should be 2 (original's child + child's child)
        assert resubmit_count[0] == 2

        # Now the grandchild order will also get partial fill and resubmit
        await tracker_auto_resubmit.record_fill("new-order-2", 10, 150.0, is_final=True)

        # Total resubmits = 3 (each order can resubmit up to max_attempts times)
        assert resubmit_count[0] == 3

    @pytest.mark.asyncio
    async def test_auto_resubmit_min_qty(self, tracker_auto_resubmit):
        """Test auto-resubmit not triggered below minimum quantity."""
        tracker_auto_resubmit.min_resubmit_qty = 50
        tracker_auto_resubmit.track_order("order-123", "AAPL", "buy", 100)

        resubmit_calls = []

        async def resubmit_callback(symbol, side, qty):
            resubmit_calls.append(qty)
            return "new-order"

        tracker_auto_resubmit.set_resubmit_callback(resubmit_callback)

        # 70% fill, 30 unfilled - below min of 50
        await tracker_auto_resubmit.record_fill("order-123", 70, 150.0, is_final=True)

        assert len(resubmit_calls) == 0

    @pytest.mark.asyncio
    async def test_auto_resubmit_no_callback_set(self, tracker_auto_resubmit):
        """Test auto-resubmit doesn't crash without callback."""
        tracker_auto_resubmit.track_order("order-123", "AAPL", "buy", 100)
        # No callback set

        # Should not crash
        await tracker_auto_resubmit.record_fill("order-123", 60, 150.0, is_final=True)


# ============================================================================
# STATISTICS TESTS
# ============================================================================


class TestStatistics:
    """Tests for statistics calculation."""

    @pytest.mark.asyncio
    async def test_empty_statistics(self, tracker):
        """Test statistics with no orders."""
        stats = tracker.get_statistics()

        assert stats.total_orders == 0
        assert stats.average_fill_rate == 0.0

    @pytest.mark.asyncio
    async def test_statistics_with_orders(self, tracker):
        """Test statistics with multiple orders."""
        tracker.track_order("order-1", "AAPL", "buy", 100)
        tracker.track_order("order-2", "MSFT", "buy", 100)
        tracker.track_order("order-3", "GOOGL", "buy", 100)

        await tracker.record_fill("order-1", 100, 150.0, is_final=True)  # Full fill
        await tracker.record_fill("order-2", 80, 250.0, is_final=True, status="partial")  # Partial
        await tracker.record_fill("order-3", 0, 0, is_final=True, status="rejected")  # Rejected

        stats = tracker.get_statistics()

        assert stats.total_orders == 3
        assert stats.fully_filled_orders == 1
        assert stats.partially_filled_orders == 1
        assert stats.rejected_orders == 1
        assert stats.total_requested_qty == 300
        assert stats.total_filled_qty == 180

    @pytest.mark.asyncio
    async def test_orders_below_threshold(self, tracker):
        """Test counting orders below 90% fill rate."""
        tracker.track_order("order-1", "AAPL", "buy", 100)
        tracker.track_order("order-2", "MSFT", "buy", 100)

        await tracker.record_fill("order-1", 95, 150.0, is_final=True)  # 95% - above 90
        await tracker.record_fill(
            "order-2", 70, 250.0, is_final=True, status="partial"
        )  # 70% - below 90

        stats = tracker.get_statistics()

        assert stats.orders_below_90_pct_fill == 1


# ============================================================================
# UTILITY METHOD TESTS
# ============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_unfilled_qty(self, tracker_with_order):
        """Test getting unfilled quantity."""
        assert tracker_with_order.get_unfilled_qty("order-123") == 100

    def test_get_unfilled_qty_unknown_order(self, tracker):
        """Test getting unfilled qty for unknown order."""
        assert tracker.get_unfilled_qty("unknown") == 0.0

    @pytest.mark.asyncio
    async def test_get_pending_orders(self, tracker):
        """Test getting pending orders."""
        tracker.track_order("order-1", "AAPL", "buy", 100)
        tracker.track_order("order-2", "MSFT", "buy", 100)

        await tracker.record_fill("order-1", 100, 150.0, is_final=True)  # Complete
        await tracker.record_fill("order-2", 50, 250.0, is_final=False)  # Still pending

        pending = tracker.get_pending_orders()

        assert len(pending) == 1
        assert pending[0]["order_id"] == "order-2"

    @pytest.mark.asyncio
    async def test_detect_stalled_orders(self, tracker):
        """Test stalled pending order detection."""
        tracker.track_order("order-1", "AAPL", "buy", 100)
        await tracker.record_fill("order-1", 40, 150.0, is_final=False)

        # Force order into stale state.
        tracker._orders["order-1"].updated_at = datetime.now() - timedelta(minutes=10)
        stalled = tracker.detect_stalled_orders(max_stall_seconds=60)

        assert len(stalled) == 1
        assert stalled[0]["order_id"] == "order-1"
        assert stalled[0]["stall_seconds"] >= 60

    @pytest.mark.asyncio
    async def test_get_fill_events(self, tracker_with_order):
        """Test getting fill events."""
        await tracker_with_order.record_fill("order-123", 50, 150.0, is_final=False)
        await tracker_with_order.record_fill("order-123", 100, 150.0, is_final=True)

        events = tracker_with_order.get_fill_events()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_fill_events_filtered(self, tracker):
        """Test filtering fill events."""
        tracker.track_order("order-1", "AAPL", "buy", 100)
        tracker.track_order("order-2", "MSFT", "buy", 100)

        await tracker.record_fill("order-1", 60, 150.0, is_final=True, status="partial")
        await tracker.record_fill("order-2", 100, 250.0, is_final=True)

        # Filter by symbol
        aapl_events = tracker.get_fill_events(symbol="AAPL")
        assert len(aapl_events) == 1

        # Filter by event type
        partial_events = tracker.get_fill_events(event_type="partial_fill")
        assert len(partial_events) == 1

    def test_clear(self, tracker_with_order):
        """Test clearing tracker data."""
        tracker_with_order.clear()

        assert tracker_with_order.get_order_status("order-123") is None
        assert len(tracker_with_order.get_fill_events()) == 0
