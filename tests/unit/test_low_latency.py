"""
Tests for Low-Latency Execution Framework

Tests:
- Latency monitoring
- Connection pooling
- Order queue
- Message serialization
- Executor operations
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from execution.low_latency import (
    LowLatencyExecutor,
    LatencyMonitor,
    LatencyStats,
    LatencyBucket,
    ConnectionPool,
    OrderQueue,
    OrderMessage,
    ExecutionConfig,
    LatencyOptimizer,
    FastJSONSerializer,
    create_low_latency_executor,
    get_monotonic_us,
)


class TestLatencyBucket:
    """Tests for LatencyBucket enum."""

    def test_all_buckets_exist(self):
        """Test all latency buckets exist."""
        expected = [
            "ORDER_CREATION",
            "SERIALIZATION",
            "NETWORK_SEND",
            "NETWORK_RECV",
            "VENUE_ACK",
            "TOTAL_ROUND_TRIP",
        ]
        for name in expected:
            assert hasattr(LatencyBucket, name)


class TestLatencyMonitor:
    """Tests for LatencyMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a monitor."""
        return LatencyMonitor()

    def test_record_latency(self, monitor):
        """Test recording latency."""
        monitor.record(LatencyBucket.ORDER_CREATION, 100)
        monitor.record(LatencyBucket.ORDER_CREATION, 150)
        monitor.record(LatencyBucket.ORDER_CREATION, 200)

        stats = monitor.compute_stats(LatencyBucket.ORDER_CREATION)
        assert stats is not None
        assert stats.count == 3

    def test_start_stop_timer(self, monitor):
        """Test timer operations."""
        start = monitor.start_timer()

        # Small delay
        time.sleep(0.001)

        latency = monitor.stop_timer(LatencyBucket.NETWORK_SEND, start)

        # Should be at least 1000 us (1 ms)
        assert latency >= 500  # Allow some tolerance

    def test_compute_stats(self, monitor):
        """Test computing statistics."""
        for i in range(100):
            monitor.record(LatencyBucket.ORDER_CREATION, 100 + i)

        stats = monitor.compute_stats(LatencyBucket.ORDER_CREATION)

        assert stats.count == 100
        assert stats.min_us == 100
        assert stats.max_us == 199
        assert 145 < stats.mean_us < 155
        assert stats.p95_us > stats.median_us

    def test_alert_callback(self, monitor):
        """Test alert callback on high latency."""
        alerts = []

        def callback(bucket, latency):
            alerts.append((bucket, latency))

        monitor.set_alert_callback(callback)

        # Record high latency
        monitor.record(LatencyBucket.NETWORK_SEND, 5000)

        assert len(alerts) == 1
        assert alerts[0][1] == 5000

    def test_get_all_stats(self, monitor):
        """Test getting all bucket stats."""
        monitor.record(LatencyBucket.ORDER_CREATION, 100)
        monitor.record(LatencyBucket.NETWORK_SEND, 200)

        all_stats = monitor.get_all_stats()

        assert LatencyBucket.ORDER_CREATION in all_stats
        assert LatencyBucket.NETWORK_SEND in all_stats

    def test_clear(self, monitor):
        """Test clearing measurements."""
        monitor.record(LatencyBucket.ORDER_CREATION, 100)
        monitor.clear()

        stats = monitor.compute_stats(LatencyBucket.ORDER_CREATION)
        assert stats is None


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_create_stats(self):
        """Test creating stats."""
        stats = LatencyStats(
            bucket=LatencyBucket.ORDER_CREATION,
            count=100,
            min_us=50,
            max_us=500,
            mean_us=150.0,
            median_us=140.0,
            p95_us=300.0,
            p99_us=450.0,
            std_us=50.0,
        )

        assert stats.count == 100
        assert stats.p99_us == 450.0

    def test_to_dict(self):
        """Test serialization."""
        stats = LatencyStats(
            bucket=LatencyBucket.ORDER_CREATION,
            count=100,
            min_us=50,
            max_us=500,
            mean_us=150.0,
            median_us=140.0,
            p95_us=300.0,
            p99_us=450.0,
            std_us=50.0,
        )

        d = stats.to_dict()
        assert d["bucket"] == "order_creation"
        assert d["count"] == 100


class TestOrderMessage:
    """Tests for OrderMessage dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = OrderMessage(
            message_id=1,
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
            venue="NYSE",
        )

        assert msg.symbol == "AAPL"
        assert msg.quantity == 100

    def test_timing_properties(self):
        """Test timing calculations."""
        msg = OrderMessage(
            message_id=1,
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
            venue="NYSE",
        )

        # Simulate timing
        msg.created_at_us = 1000000
        msg.sent_at_us = 1000100
        msg.acked_at_us = 1000300

        assert msg.creation_to_send_us == 100
        assert msg.send_to_ack_us == 200
        assert msg.total_latency_us == 300


class TestOrderQueue:
    """Tests for OrderQueue class."""

    @pytest.fixture
    def queue(self):
        """Create a queue."""
        return OrderQueue()

    def test_enqueue_dequeue(self, queue):
        """Test enqueueing and dequeueing."""
        msg = OrderMessage(
            message_id=0,
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
            venue="NYSE",
        )

        msg_id = queue.enqueue(msg)
        assert msg_id == 1

        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.symbol == "AAPL"

    def test_peek(self, queue):
        """Test peeking at queue."""
        msg = OrderMessage(
            message_id=0,
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
            venue="NYSE",
        )

        queue.enqueue(msg)

        peeked = queue.peek()
        assert peeked is not None

        # Queue should still have item
        assert not queue.is_empty

    def test_empty_queue(self, queue):
        """Test empty queue operations."""
        assert queue.is_empty
        assert queue.size == 0
        assert queue.dequeue() is None
        assert queue.peek() is None


class TestConnectionPool:
    """Tests for ConnectionPool class."""

    @pytest.fixture
    def pool(self):
        """Create a pool."""
        return ConnectionPool(min_connections=2, max_connections=5)

    @pytest.mark.asyncio
    async def test_initialize(self, pool):
        """Test initializing pool."""
        await pool.initialize(["NYSE", "NASDAQ"])

        # Should have connections for both venues
        assert "NYSE" in pool._connections
        assert "NASDAQ" in pool._connections
        assert len(pool._connections["NYSE"]) >= 2

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_release(self, pool):
        """Test acquiring and releasing connections."""
        await pool.initialize(["NYSE"])

        conn = pool.acquire("NYSE")
        assert conn is not None

        pool.release("NYSE", conn)

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_nonexistent_venue(self, pool):
        """Test acquiring from nonexistent venue."""
        await pool.initialize(["NYSE"])

        conn = pool.acquire("INVALID")
        assert conn is None

        await pool.shutdown()


class TestFastJSONSerializer:
    """Tests for FastJSONSerializer class."""

    @pytest.fixture
    def serializer(self):
        """Create a serializer."""
        return FastJSONSerializer()

    def test_serialize(self, serializer):
        """Test serializing a message."""
        msg = OrderMessage(
            message_id=1,
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="limit",
            venue="NYSE",
        )

        data = serializer.serialize(msg)
        assert isinstance(data, bytes)
        assert b"AAPL" in data

    def test_deserialize(self, serializer):
        """Test deserializing."""
        data = b'{"id": 1, "symbol": "AAPL", "qty": 100}'
        result = serializer.deserialize(data)

        assert result["symbol"] == "AAPL"
        assert result["qty"] == 100


class TestExecutionConfig:
    """Tests for ExecutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExecutionConfig()

        assert config.min_connections_per_venue == 2
        assert config.max_queue_size == 10000
        assert config.latency_alert_threshold_us == 1000

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutionConfig(
            min_connections_per_venue=5,
            latency_alert_threshold_us=500,
        )

        assert config.min_connections_per_venue == 5
        assert config.latency_alert_threshold_us == 500


class TestLowLatencyExecutor:
    """Tests for LowLatencyExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create an executor."""
        return LowLatencyExecutor()

    @pytest.mark.asyncio
    async def test_initialize(self, executor):
        """Test initializing executor."""
        await executor.initialize(["NYSE", "NASDAQ"])

        assert executor._running
        assert executor.connection_pool is not None

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_submit_order(self, executor):
        """Test submitting an order."""
        await executor.initialize(["NYSE"])

        msg_id = await executor.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            venue="NYSE",
        )

        assert msg_id > 0

        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_callbacks(self, executor):
        """Test fill/reject callbacks."""
        fills = []
        rejects = []

        executor.set_callbacks(
            on_fill=lambda msg: fills.append(msg),
            on_reject=lambda msg, reason: rejects.append((msg, reason)),
        )

        await executor.initialize(["NYSE"])

        await executor.submit_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            venue="NYSE",
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        await executor.shutdown()

    def test_get_latency_stats(self, executor):
        """Test getting latency stats."""
        # Record some latencies
        executor.latency_monitor.record(LatencyBucket.ORDER_CREATION, 100)
        executor.latency_monitor.record(LatencyBucket.NETWORK_SEND, 200)

        stats = executor.get_latency_stats()

        assert "order_creation" in stats
        assert "network_send" in stats

    def test_get_execution_stats(self, executor):
        """Test getting execution stats."""
        stats = executor.get_execution_stats()

        assert "orders_sent" in stats
        assert "orders_filled" in stats
        assert "fill_rate" in stats


class TestLatencyOptimizer:
    """Tests for LatencyOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create an optimizer."""
        executor = LowLatencyExecutor()
        return LatencyOptimizer(executor)

    def test_analyze(self, optimizer):
        """Test analyzing latencies."""
        # Add some measurements
        optimizer.executor.latency_monitor.record(
            LatencyBucket.ORDER_CREATION, 100
        )
        optimizer.executor.latency_monitor.record(
            LatencyBucket.NETWORK_SEND, 200
        )
        optimizer.executor.latency_monitor.record(
            LatencyBucket.TOTAL_ROUND_TRIP, 300
        )

        analysis = optimizer.analyze()

        assert "timestamp" in analysis
        assert "stats" in analysis
        assert "grade" in analysis

    def test_recommendations_on_high_latency(self, optimizer):
        """Test recommendations for high latency."""
        # Add high latency measurement
        for _ in range(100):
            optimizer.executor.latency_monitor.record(
                LatencyBucket.NETWORK_SEND, 5000
            )

        analysis = optimizer.analyze()

        # Should have recommendations
        assert "recommendations" in analysis


class TestCreateLowLatencyExecutor:
    """Tests for create_low_latency_executor factory."""

    def test_create_with_defaults(self):
        """Test creating with defaults."""
        executor = create_low_latency_executor()
        assert executor is not None

    def test_create_with_venues(self):
        """Test creating with specific venues."""
        executor = create_low_latency_executor(venues=["NYSE", "NASDAQ"])
        assert executor is not None

    def test_create_with_config(self):
        """Test creating with custom config."""
        config = ExecutionConfig(
            latency_alert_threshold_us=500
        )
        executor = create_low_latency_executor(config=config)
        assert executor.config.latency_alert_threshold_us == 500


class TestGetMonotonicUs:
    """Tests for get_monotonic_us function."""

    def test_returns_positive(self):
        """Test returns positive value."""
        us = get_monotonic_us()
        assert us > 0

    def test_monotonically_increasing(self):
        """Test values are monotonically increasing."""
        t1 = get_monotonic_us()
        time.sleep(0.001)
        t2 = get_monotonic_us()

        assert t2 > t1
