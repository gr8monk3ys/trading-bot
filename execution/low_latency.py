"""
Low-Latency Execution Framework

Provides infrastructure for sub-millisecond order execution:
1. Connection Pool: Pre-established venue connections
2. Message Queue: Lock-free order queuing
3. Latency Monitoring: Microsecond-level tracking
4. Hot Path Optimization: Pre-allocated buffers, minimal allocations

Why low-latency matters:
- Co-located HFT achieves <10 microsecond execution
- Retail typically sees 50-100ms latency
- Each millisecond of latency can cost 0.5-1 bps
"""

import asyncio
import logging
import queue
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_monotonic_ns() -> int:
    """Get monotonic time in nanoseconds."""
    return time.monotonic_ns()


def get_monotonic_us() -> int:
    """Get monotonic time in microseconds."""
    return time.monotonic_ns() // 1000


class LatencyBucket(Enum):
    """Latency measurement categories."""

    ORDER_CREATION = "order_creation"
    SERIALIZATION = "serialization"
    NETWORK_SEND = "network_send"
    NETWORK_RECV = "network_recv"
    DESERIALIZATION = "deserialization"
    VENUE_ACK = "venue_ack"
    TOTAL_ROUND_TRIP = "total_round_trip"


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    bucket: LatencyBucket
    latency_us: int
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None


@dataclass
class LatencyStats:
    """Aggregated latency statistics."""

    bucket: LatencyBucket
    count: int
    min_us: int
    max_us: int
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float
    std_us: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket": self.bucket.value,
            "count": self.count,
            "min_us": self.min_us,
            "max_us": self.max_us,
            "mean_us": self.mean_us,
            "median_us": self.median_us,
            "p95_us": self.p95_us,
            "p99_us": self.p99_us,
        }


class LatencyMonitor:
    """
    Tracks execution latencies with minimal overhead.

    Uses pre-allocated circular buffers to avoid allocations
    in the hot path.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        alert_threshold_us: int = 1000,  # 1ms
    ):
        self.buffer_size = buffer_size
        self.alert_threshold_us = alert_threshold_us

        # Pre-allocated buffers per bucket
        self._buffers: Dict[LatencyBucket, Deque[int]] = {
            bucket: deque(maxlen=buffer_size) for bucket in LatencyBucket
        }

        # Running stats (updated periodically, not per-sample)
        self._stats: Dict[LatencyBucket, LatencyStats] = {}
        self._stats_lock = threading.Lock()

        # Alert callback
        self._alert_callback: Optional[Callable[[LatencyBucket, int], None]] = None

    def record(self, bucket: LatencyBucket, latency_us: int) -> None:
        """
        Record a latency measurement.

        This is the hot path - must be fast.
        """
        self._buffers[bucket].append(latency_us)

        # Check for alert (simple comparison, no lock)
        if latency_us > self.alert_threshold_us and self._alert_callback:
            self._alert_callback(bucket, latency_us)

    def start_timer(self) -> int:
        """Start a timer, returns start time in microseconds."""
        return get_monotonic_us()

    def stop_timer(self, bucket: LatencyBucket, start_us: int) -> int:
        """Stop timer and record measurement."""
        latency_us = get_monotonic_us() - start_us
        self.record(bucket, latency_us)
        return latency_us

    def set_alert_callback(
        self,
        callback: Callable[[LatencyBucket, int], None],
    ) -> None:
        """Set callback for latency alerts."""
        self._alert_callback = callback

    def compute_stats(self, bucket: LatencyBucket) -> Optional[LatencyStats]:
        """Compute statistics for a bucket."""
        samples = list(self._buffers[bucket])
        if not samples:
            return None

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return LatencyStats(
            bucket=bucket,
            count=n,
            min_us=sorted_samples[0],
            max_us=sorted_samples[-1],
            mean_us=statistics.mean(sorted_samples),
            median_us=statistics.median(sorted_samples),
            p95_us=sorted_samples[int(n * 0.95)] if n > 20 else sorted_samples[-1],
            p99_us=sorted_samples[int(n * 0.99)] if n > 100 else sorted_samples[-1],
            std_us=statistics.stdev(sorted_samples) if n > 1 else 0,
        )

    def get_all_stats(self) -> Dict[LatencyBucket, LatencyStats]:
        """Get stats for all buckets."""
        stats = {}
        for bucket in LatencyBucket:
            stat = self.compute_stats(bucket)
            if stat:
                stats[bucket] = stat
        return stats

    def clear(self) -> None:
        """Clear all measurements."""
        for buffer in self._buffers.values():
            buffer.clear()


@dataclass
class OrderMessage:
    """Pre-formatted order message for fast sending."""

    message_id: int
    symbol: str
    side: str
    quantity: int
    price: Optional[float]
    order_type: str
    venue: str

    # Timing
    created_at_us: int = field(default_factory=get_monotonic_us)
    sent_at_us: int = 0
    acked_at_us: int = 0

    # Pre-serialized payload (avoid runtime serialization)
    payload: Optional[bytes] = None

    @property
    def creation_to_send_us(self) -> int:
        """Time from creation to network send."""
        if self.sent_at_us > 0:
            return self.sent_at_us - self.created_at_us
        return 0

    @property
    def send_to_ack_us(self) -> int:
        """Time from send to acknowledgment."""
        if self.acked_at_us > 0 and self.sent_at_us > 0:
            return self.acked_at_us - self.sent_at_us
        return 0

    @property
    def total_latency_us(self) -> int:
        """Total round-trip latency."""
        if self.acked_at_us > 0:
            return self.acked_at_us - self.created_at_us
        return 0


class MessageSerializer(ABC):
    """Abstract message serializer."""

    @abstractmethod
    def serialize(self, message: OrderMessage) -> bytes:
        """Serialize message to bytes."""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize bytes to dict."""
        pass


class FastJSONSerializer(MessageSerializer):
    """
    Fast JSON serializer using pre-allocation.

    For production, consider:
    - orjson (Rust-based, 10x faster)
    - msgpack (binary, smaller)
    - FIX protocol (industry standard)
    """

    def __init__(self):
        # Pre-allocate common templates
        self._templates = {
            "market_buy": b'{"type":"market","side":"buy","symbol":"',
            "market_sell": b'{"type":"market","side":"sell","symbol":"',
            "limit_buy": b'{"type":"limit","side":"buy","symbol":"',
            "limit_sell": b'{"type":"limit","side":"sell","symbol":"',
        }

    def serialize(self, message: OrderMessage) -> bytes:
        """Fast serialize using templates."""
        # In production, use orjson or similar
        import json

        return json.dumps(
            {
                "id": message.message_id,
                "symbol": message.symbol,
                "side": message.side,
                "qty": message.quantity,
                "price": message.price,
                "type": message.order_type,
                "venue": message.venue,
            }
        ).encode()

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize response."""
        import json

        return json.loads(data)


class ConnectionPool:
    """
    Pre-established connection pool for venues.

    Maintains warm connections to avoid connection latency
    on the order path.
    """

    def __init__(
        self,
        min_connections: int = 2,
        max_connections: int = 10,
        health_check_interval: float = 5.0,
    ):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval

        self._connections: Dict[str, List[Any]] = {}
        self._available: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._running = False
        self._health_task: Optional[asyncio.Task] = None

    async def initialize(self, venues: List[str]) -> None:
        """Pre-establish connections to venues."""
        for venue in venues:
            self._connections[venue] = []
            self._available[venue] = queue.Queue()

            # Create minimum connections
            for _ in range(self.min_connections):
                conn = await self._create_connection(venue)
                if conn:
                    self._connections[venue].append(conn)
                    self._available[venue].put(conn)

        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def _create_connection(self, venue: str) -> Optional[Any]:
        """Create a new connection to venue."""
        # Placeholder - in production, create actual TCP/WebSocket connection
        return {"venue": venue, "connected": True, "created_at": datetime.now()}

    async def _health_check_loop(self) -> None:
        """Periodically check connection health."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_connections(self) -> None:
        """Check and refresh unhealthy connections."""
        for venue, connections in self._connections.items():
            for conn in connections:
                if not conn.get("connected"):
                    # Replace unhealthy connection
                    new_conn = await self._create_connection(venue)
                    if new_conn:
                        connections.remove(conn)
                        connections.append(new_conn)
                        self._available[venue].put(new_conn)

    def acquire(self, venue: str, timeout: float = 0.001) -> Optional[Any]:
        """
        Acquire a connection (non-blocking with very short timeout).

        Hot path - must be fast.
        """
        try:
            return self._available[venue].get(timeout=timeout)
        except (KeyError, queue.Empty):
            return None

    def release(self, venue: str, conn: Any) -> None:
        """Release a connection back to pool."""
        if venue in self._available:
            self._available[venue].put(conn)

    async def shutdown(self) -> None:
        """Shutdown connection pool."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass


class OrderQueue:
    """
    Lock-free order queue for high-throughput.

    Uses a simple deque with atomic operations.
    For true lock-free, consider disruptor pattern.
    """

    def __init__(self, max_size: int = 10000):
        self._queue: Deque[OrderMessage] = deque(maxlen=max_size)
        self._message_id = 0
        self._lock = threading.Lock()

    def enqueue(self, message: OrderMessage) -> int:
        """Add message to queue. Returns message ID."""
        with self._lock:
            self._message_id += 1
            message.message_id = self._message_id
            self._queue.append(message)
            return self._message_id

    def dequeue(self) -> Optional[OrderMessage]:
        """Remove and return next message."""
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def peek(self) -> Optional[OrderMessage]:
        """Look at next message without removing."""
        try:
            return self._queue[0]
        except IndexError:
            return None

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0


@dataclass
class ExecutionConfig:
    """Configuration for low-latency execution."""

    # Connection settings
    min_connections_per_venue: int = 2
    max_connections_per_venue: int = 10
    connection_timeout_ms: int = 100

    # Queue settings
    max_queue_size: int = 10000
    batch_size: int = 10

    # Timing
    health_check_interval_s: float = 5.0
    latency_alert_threshold_us: int = 1000

    # Retry
    max_retries: int = 3
    retry_delay_us: int = 100


class LowLatencyExecutor:
    """
    Low-latency order execution engine.

    Optimizations:
    - Pre-established connection pool
    - Pre-allocated message buffers
    - Batched network I/O
    - Microsecond-level latency tracking
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
    ):
        self.config = config or ExecutionConfig()

        # Components
        self.latency_monitor = LatencyMonitor(
            alert_threshold_us=self.config.latency_alert_threshold_us
        )
        self.connection_pool = ConnectionPool(
            min_connections=self.config.min_connections_per_venue,
            max_connections=self.config.max_connections_per_venue,
            health_check_interval=self.config.health_check_interval_s,
        )
        self.order_queue = OrderQueue(max_size=self.config.max_queue_size)
        self.serializer = FastJSONSerializer()

        # State
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_fill: Optional[Callable] = None
        self._on_reject: Optional[Callable] = None

        # Stats
        self._orders_sent = 0
        self._orders_filled = 0
        self._orders_rejected = 0

    async def initialize(self, venues: List[str]) -> None:
        """Initialize executor with venue connections."""
        await self.connection_pool.initialize(venues)
        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info(f"Low-latency executor initialized for {len(venues)} venues")

    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "market",
        venue: str = "default",
    ) -> int:
        """
        Submit order for execution.

        Returns message ID for tracking.
        """
        start = self.latency_monitor.start_timer()

        message = OrderMessage(
            message_id=0,  # Will be set by queue
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            venue=venue,
        )

        # Pre-serialize
        message.payload = self.serializer.serialize(message)

        self.latency_monitor.stop_timer(LatencyBucket.ORDER_CREATION, start)

        # Enqueue
        message_id = self.order_queue.enqueue(message)
        return message_id

    async def _process_queue(self) -> None:
        """Process order queue (background task)."""
        while self._running:
            try:
                # Batch processing for efficiency
                batch: List[OrderMessage] = []

                for _ in range(self.config.batch_size):
                    message = self.order_queue.dequeue()
                    if message:
                        batch.append(message)
                    else:
                        break

                if batch:
                    await self._send_batch(batch)
                else:
                    # No orders, brief sleep
                    await asyncio.sleep(0.0001)  # 100 microseconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    async def _send_batch(self, batch: List[OrderMessage]) -> None:
        """Send batch of orders."""
        for message in batch:
            await self._send_order(message)

    async def _send_order(self, message: OrderMessage) -> None:
        """Send single order with latency tracking."""
        # Get connection
        conn = self.connection_pool.acquire(message.venue)
        if not conn:
            logger.warning(f"No connection available for {message.venue}")
            self._orders_rejected += 1
            if self._on_reject:
                self._on_reject(message, "no_connection")
            return

        try:
            # Record send time
            send_start = self.latency_monitor.start_timer()
            message.sent_at_us = send_start

            # Simulate network send (replace with actual send)
            await asyncio.sleep(0.0001)  # Simulated network latency

            self.latency_monitor.stop_timer(LatencyBucket.NETWORK_SEND, send_start)

            # Simulate response
            await asyncio.sleep(0.0001)

            message.acked_at_us = get_monotonic_us()
            self.latency_monitor.record(LatencyBucket.TOTAL_ROUND_TRIP, message.total_latency_us)

            self._orders_sent += 1
            self._orders_filled += 1

            if self._on_fill:
                self._on_fill(message)

        except Exception as e:
            logger.error(f"Order send error: {e}")
            self._orders_rejected += 1
            if self._on_reject:
                self._on_reject(message, str(e))

        finally:
            self.connection_pool.release(message.venue, conn)

    def set_callbacks(
        self,
        on_fill: Optional[Callable] = None,
        on_reject: Optional[Callable] = None,
    ) -> None:
        """Set fill and rejection callbacks."""
        self._on_fill = on_fill
        self._on_reject = on_reject

    def get_latency_stats(self) -> Dict[str, LatencyStats]:
        """Get latency statistics."""
        stats = self.latency_monitor.get_all_stats()
        return {bucket.value: stat for bucket, stat in stats.items()}

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "orders_sent": self._orders_sent,
            "orders_filled": self._orders_filled,
            "orders_rejected": self._orders_rejected,
            "fill_rate": (self._orders_filled / self._orders_sent if self._orders_sent > 0 else 0),
            "queue_size": self.order_queue.size,
        }

    async def shutdown(self) -> None:
        """Shutdown executor."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        await self.connection_pool.shutdown()
        logger.info("Low-latency executor shutdown complete")


class LatencyOptimizer:
    """
    Analyzes and optimizes execution latency.

    Provides recommendations based on latency patterns.
    """

    def __init__(self, executor: LowLatencyExecutor):
        self.executor = executor
        self._history: List[Dict[str, Any]] = []

    def analyze(self) -> Dict[str, Any]:
        """Analyze current latency patterns."""
        stats = self.executor.get_latency_stats()

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "stats": {k: v.to_dict() for k, v in stats.items()},
            "recommendations": [],
        }

        # Check for issues
        for bucket, stat in stats.items():
            if stat.p99_us > 1000:  # >1ms at p99
                analysis["recommendations"].append(
                    {
                        "issue": f"High {bucket} latency",
                        "p99_us": stat.p99_us,
                        "recommendation": self._get_recommendation(bucket, stat),
                    }
                )

        # Overall assessment
        total = stats.get("total_round_trip")
        if total:
            if total.median_us < 100:
                analysis["grade"] = "A"
            elif total.median_us < 500:
                analysis["grade"] = "B"
            elif total.median_us < 1000:
                analysis["grade"] = "C"
            else:
                analysis["grade"] = "D"
        else:
            analysis["grade"] = "N/A"

        self._history.append(analysis)
        return analysis

    def _get_recommendation(
        self,
        bucket: str,
        stat: LatencyStats,
    ) -> str:
        """Get optimization recommendation."""
        recommendations = {
            "order_creation": "Consider pre-allocating order objects",
            "serialization": "Use binary format (msgpack, protobuf) instead of JSON",
            "network_send": "Check network path, consider co-location",
            "network_recv": "Increase receive buffer size",
            "venue_ack": "Contact venue about slow acknowledgments",
            "total_round_trip": "Review entire order path for bottlenecks",
        }
        return recommendations.get(bucket, "Investigate latency source")


def create_low_latency_executor(
    venues: Optional[List[str]] = None,
    config: Optional[ExecutionConfig] = None,
) -> LowLatencyExecutor:
    """
    Factory function to create low-latency executor.

    Args:
        venues: List of venue identifiers
        config: Execution configuration

    Returns:
        Configured LowLatencyExecutor
    """
    venues = venues or ["NYSE", "NASDAQ", "IEX", "BATS"]
    executor = LowLatencyExecutor(config=config)

    # Note: Must call executor.initialize(venues) before use

    return executor


def print_latency_report(executor: LowLatencyExecutor) -> None:
    """Print formatted latency report."""
    stats = executor.get_latency_stats()
    exec_stats = executor.get_execution_stats()

    print("\n" + "=" * 60)
    print("LOW-LATENCY EXECUTION REPORT")
    print("=" * 60)

    print(f"\n{'Execution Summary':-^40}")
    print(f"  Orders Sent: {exec_stats['orders_sent']:,}")
    print(f"  Orders Filled: {exec_stats['orders_filled']:,}")
    print(f"  Orders Rejected: {exec_stats['orders_rejected']:,}")
    print(f"  Fill Rate: {exec_stats['fill_rate']:.2%}")
    print(f"  Queue Size: {exec_stats['queue_size']}")

    print(f"\n{'Latency by Stage (microseconds)':-^40}")

    for bucket, stat in stats.items():
        print(f"\n  {bucket}:")
        print(f"    Count: {stat.count:,}")
        print(f"    Min: {stat.min_us:,} us")
        print(f"    Median: {stat.median_us:.0f} us")
        print(f"    Mean: {stat.mean_us:.0f} us")
        print(f"    P95: {stat.p95_us:.0f} us")
        print(f"    P99: {stat.p99_us:.0f} us")
        print(f"    Max: {stat.max_us:,} us")

    # Performance grade
    total = stats.get("total_round_trip")
    if total:
        grade = "A" if total.median_us < 100 else "B" if total.median_us < 500 else "C"
        print(f"\n{'Performance Grade':-^40}")
        print(f"  Grade: {grade}")
        print(f"  Median Round-Trip: {total.median_us:.0f} us ({total.median_us/1000:.2f} ms)")

    print("=" * 60 + "\n")
