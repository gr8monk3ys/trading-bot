"""
Multi-Broker Failover Manager

Provides automatic failover between multiple brokers:
- Primary broker (e.g., Alpaca) handles all operations normally
- Health monitoring detects failures
- Automatic failover to backup broker(s) on failure
- Automatic failback when primary recovers

Usage:
    from brokers.multi_broker import MultiBrokerManager
    from brokers.alpaca_broker import AlpacaBroker

    # Initialize brokers
    primary = AlpacaBroker(api_key, secret_key)
    backup = InteractiveBrokersBroker(...)  # Or another broker

    # Create manager
    manager = MultiBrokerManager(
        primary=primary,
        backups=[backup],
        health_check_interval=30,
    )

    # Use like a normal broker
    account = await manager.get_account()
    order = await manager.submit_order(request)

Features:
- Automatic health monitoring
- Configurable failover thresholds
- Position synchronization warnings
- Audit logging of all failovers
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from brokers.broker_interface import (
    AccountInfo,
    Bar,
    BrokerConnectionError,
    BrokerError,
    BrokerInterface,
    BrokerStatus,
    Order,
    OrderRequest,
    OrderStatus,
    Position,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FailoverEvent(Enum):
    """Types of failover events."""

    PRIMARY_FAILED = "primary_failed"
    FAILOVER_TO_BACKUP = "failover_to_backup"
    FAILBACK_TO_PRIMARY = "failback_to_primary"
    BACKUP_FAILED = "backup_failed"
    ALL_BROKERS_FAILED = "all_brokers_failed"
    HEALTH_CHECK_PASSED = "health_check_passed"
    HEALTH_CHECK_FAILED = "health_check_failed"


@dataclass
class BrokerHealth:
    """Health status for a single broker."""

    broker_name: str
    status: BrokerStatus
    last_check: datetime
    last_success: Optional[datetime]
    consecutive_failures: int
    response_time_ms: Optional[float]
    error_message: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self.status == BrokerStatus.CONNECTED and self.consecutive_failures == 0


@dataclass
class FailoverLog:
    """Log entry for failover events."""

    timestamp: datetime
    event: FailoverEvent
    from_broker: Optional[str]
    to_broker: Optional[str]
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class MultiBrokerManager(BrokerInterface):
    """
    Manages multiple brokers with automatic failover.

    Implements BrokerInterface so it can be used as a drop-in replacement
    for any single broker.
    """

    # Default configuration
    DEFAULT_HEALTH_CHECK_INTERVAL = 30  # seconds
    DEFAULT_FAILURE_THRESHOLD = 3  # consecutive failures before failover
    DEFAULT_RECOVERY_THRESHOLD = 5  # consecutive successes before failback
    DEFAULT_OPERATION_TIMEOUT = 30.0  # seconds

    def __init__(
        self,
        primary: BrokerInterface,
        backups: Optional[List[BrokerInterface]] = None,
        health_check_interval: int = DEFAULT_HEALTH_CHECK_INTERVAL,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_threshold: int = DEFAULT_RECOVERY_THRESHOLD,
        operation_timeout: float = DEFAULT_OPERATION_TIMEOUT,
        on_failover: Optional[Callable[[FailoverLog], None]] = None,
        auto_start_monitoring: bool = True,
    ):
        """
        Initialize multi-broker manager.

        Args:
            primary: Primary broker instance
            backups: List of backup broker instances (in priority order)
            health_check_interval: Seconds between health checks
            failure_threshold: Consecutive failures before failover
            recovery_threshold: Consecutive successes before failback
            operation_timeout: Timeout for broker operations
            on_failover: Callback function for failover events
            auto_start_monitoring: Start health monitoring on init
        """
        self._primary = primary
        self._backups = backups or []
        self._all_brokers = [primary] + self._backups

        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.operation_timeout = operation_timeout
        self.on_failover = on_failover

        # State
        self._active_broker: BrokerInterface = primary
        self._broker_health: Dict[str, BrokerHealth] = {}
        self._failover_log: List[FailoverLog] = []
        self._is_failed_over = False

        # Health monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = asyncio.Event()

        # Initialize health tracking
        for broker in self._all_brokers:
            self._broker_health[broker.name] = BrokerHealth(
                broker_name=broker.name,
                status=BrokerStatus.UNKNOWN,
                last_check=datetime.now(),
                last_success=None,
                consecutive_failures=0,
                response_time_ms=None,
            )

        # Recovery tracking for failback
        self._primary_recovery_count = 0

        if auto_start_monitoring:
            # Note: caller should await start_monitoring() after init
            pass

    @property
    def name(self) -> str:
        return f"MultiBroker({self._active_broker.name})"

    @property
    def is_paper(self) -> bool:
        return self._active_broker.is_paper

    @property
    def active_broker(self) -> BrokerInterface:
        """Currently active broker."""
        return self._active_broker

    @property
    def is_failed_over(self) -> bool:
        """Whether we're currently using a backup broker."""
        return self._is_failed_over

    # === Monitoring ===

    async def start_monitoring(self):
        """Start health monitoring background task."""
        if self._monitoring_task is not None:
            return

        self._stop_monitoring.clear()
        self._monitoring_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started broker health monitoring")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        if self._monitoring_task is None:
            return

        self._stop_monitoring.set()
        self._monitoring_task.cancel()
        try:
            await self._monitoring_task
        except asyncio.CancelledError:
            pass
        self._monitoring_task = None
        logger.info("Stopped broker health monitoring")

    async def _health_monitor_loop(self):
        """Background loop for health monitoring."""
        while not self._stop_monitoring.is_set():
            try:
                await self._check_all_brokers()
                await self._evaluate_failover()

                # Wait for next check interval
                await asyncio.wait_for(
                    self._stop_monitoring.wait(),
                    timeout=self.health_check_interval,
                )
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue loop
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _check_all_brokers(self):
        """Check health of all brokers."""
        for broker in self._all_brokers:
            await self._check_broker_health(broker)

    async def _check_broker_health(self, broker: BrokerInterface):
        """Check health of a single broker."""
        start_time = datetime.now()
        health = self._broker_health[broker.name]

        try:
            is_healthy = await asyncio.wait_for(
                broker.health_check(),
                timeout=self.operation_timeout,
            )

            response_time = (datetime.now() - start_time).total_seconds() * 1000

            if is_healthy:
                health.status = BrokerStatus.CONNECTED
                health.last_success = datetime.now()
                health.consecutive_failures = 0
                health.error_message = None
            else:
                health.status = BrokerStatus.DEGRADED
                health.consecutive_failures += 1
                health.error_message = "Health check returned False"

            health.response_time_ms = response_time

        except asyncio.TimeoutError:
            health.status = BrokerStatus.DISCONNECTED
            health.consecutive_failures += 1
            health.error_message = "Health check timed out"
            health.response_time_ms = None

        except Exception as e:
            health.status = BrokerStatus.DISCONNECTED
            health.consecutive_failures += 1
            health.error_message = str(e)
            health.response_time_ms = None

        health.last_check = datetime.now()

    async def _evaluate_failover(self):
        """Evaluate whether failover or failback is needed."""
        primary_health = self._broker_health[self._primary.name]

        if not self._is_failed_over:
            # Check if primary has failed
            if primary_health.consecutive_failures >= self.failure_threshold:
                await self._do_failover()
        else:
            # Check if primary has recovered (for failback)
            if primary_health.is_healthy:
                self._primary_recovery_count += 1
                if self._primary_recovery_count >= self.recovery_threshold:
                    await self._do_failback()
            else:
                self._primary_recovery_count = 0

    async def _do_failover(self):
        """Execute failover to backup broker."""
        # Find healthy backup
        for backup in self._backups:
            health = self._broker_health[backup.name]
            if health.is_healthy or health.consecutive_failures < self.failure_threshold:
                old_broker = self._active_broker.name
                self._active_broker = backup
                self._is_failed_over = True

                log_entry = FailoverLog(
                    timestamp=datetime.now(),
                    event=FailoverEvent.FAILOVER_TO_BACKUP,
                    from_broker=old_broker,
                    to_broker=backup.name,
                    reason=f"Primary failed ({self._broker_health[self._primary.name].error_message})",
                )
                self._failover_log.append(log_entry)

                logger.warning(f"FAILOVER: {old_broker} -> {backup.name} " f"({log_entry.reason})")

                if self.on_failover:
                    try:
                        self.on_failover(log_entry)
                    except Exception as e:
                        logger.error(f"Failover callback error: {e}")

                return

        # No healthy backup found
        log_entry = FailoverLog(
            timestamp=datetime.now(),
            event=FailoverEvent.ALL_BROKERS_FAILED,
            from_broker=self._active_broker.name,
            to_broker=None,
            reason="All brokers unavailable",
        )
        self._failover_log.append(log_entry)
        logger.critical("ALL BROKERS FAILED - no healthy backup available")

        if self.on_failover:
            try:
                self.on_failover(log_entry)
            except Exception as e:
                logger.error(f"Failover callback error: {e}")

    async def _do_failback(self):
        """Execute failback to primary broker."""
        old_broker = self._active_broker.name
        self._active_broker = self._primary
        self._is_failed_over = False
        self._primary_recovery_count = 0

        log_entry = FailoverLog(
            timestamp=datetime.now(),
            event=FailoverEvent.FAILBACK_TO_PRIMARY,
            from_broker=old_broker,
            to_broker=self._primary.name,
            reason="Primary recovered",
        )
        self._failover_log.append(log_entry)

        logger.info(f"FAILBACK: {old_broker} -> {self._primary.name}")

        if self.on_failover:
            try:
                self.on_failover(log_entry)
            except Exception as e:
                logger.error(f"Failover callback error: {e}")

    # === Broker Interface Implementation ===

    async def _execute_with_failover(
        self,
        operation_name: str,
        operation: Callable[[BrokerInterface], Any],
    ) -> Any:
        """
        Execute an operation with automatic failover on failure.

        Args:
            operation_name: Name of operation (for logging)
            operation: Async callable that takes a broker and returns result

        Returns:
            Operation result

        Raises:
            BrokerError if all brokers fail
        """
        last_error = None

        # Try active broker first
        brokers_to_try = [self._active_broker]

        # Add other brokers if active fails
        for broker in self._all_brokers:
            if broker != self._active_broker:
                brokers_to_try.append(broker)

        for broker in brokers_to_try:
            try:
                result = await asyncio.wait_for(
                    operation(broker),
                    timeout=self.operation_timeout,
                )

                # If this wasn't the active broker, we've failed over
                if broker != self._active_broker:
                    logger.warning(f"Operation {operation_name} succeeded on backup {broker.name}")
                    # Update health for the broker that worked
                    health = self._broker_health[broker.name]
                    health.consecutive_failures = 0
                    health.last_success = datetime.now()

                return result

            except asyncio.TimeoutError:
                last_error = BrokerConnectionError(f"{operation_name} timed out on {broker.name}")
                logger.warning(f"{operation_name} timed out on {broker.name}")
                health = self._broker_health[broker.name]
                health.consecutive_failures += 1

            except Exception as e:
                last_error = e
                logger.warning(f"{operation_name} failed on {broker.name}: {e}")
                health = self._broker_health[broker.name]
                health.consecutive_failures += 1

        # All brokers failed
        raise last_error or BrokerError(f"All brokers failed for {operation_name}")

    async def connect(self) -> bool:
        """Connect to all brokers."""
        results = []
        for broker in self._all_brokers:
            try:
                result = await broker.connect()
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to connect to {broker.name}: {e}")
                results.append(False)

        # Return True if at least one broker connected
        return any(results)

    async def disconnect(self) -> None:
        """Disconnect from all brokers."""
        await self.stop_monitoring()
        for broker in self._all_brokers:
            try:
                await broker.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting from {broker.name}: {e}")

    async def get_status(self) -> BrokerStatus:
        """Get status of active broker."""
        return await self._active_broker.get_status()

    async def health_check(self) -> bool:
        """Check health of active broker."""
        return await self._active_broker.health_check()

    async def get_account(self) -> AccountInfo:
        """Get account info with failover."""
        return await self._execute_with_failover(
            "get_account",
            lambda b: b.get_account(),
        )

    async def get_positions(self) -> List[Position]:
        """Get positions with failover."""
        return await self._execute_with_failover(
            "get_positions",
            lambda b: b.get_positions(),
        )

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol with failover."""
        return await self._execute_with_failover(
            f"get_position({symbol})",
            lambda b: b.get_position(symbol),
        )

    async def submit_order(self, request: OrderRequest) -> Order:
        """Submit order with failover."""
        # Warning: Order submission on backup broker creates position sync issues
        if self._is_failed_over:
            logger.warning(
                f"Submitting order on BACKUP broker {self._active_broker.name}. "
                "Positions may not sync with primary when it recovers."
            )

        return await self._execute_with_failover(
            f"submit_order({request.symbol})",
            lambda b: b.submit_order(request),
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with failover."""
        return await self._execute_with_failover(
            f"cancel_order({order_id})",
            lambda b: b.cancel_order(order_id),
        )

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order with failover."""
        return await self._execute_with_failover(
            f"get_order({order_id})",
            lambda b: b.get_order(order_id),
        )

    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbols: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get orders with failover."""
        return await self._execute_with_failover(
            "get_orders",
            lambda b: b.get_orders(status, symbols, limit),
        )

    async def cancel_all_orders(self) -> int:
        """Cancel all orders with failover."""
        return await self._execute_with_failover(
            "cancel_all_orders",
            lambda b: b.cancel_all_orders(),
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Bar]:
        """Get bars with failover."""
        return await self._execute_with_failover(
            f"get_bars({symbol})",
            lambda b: b.get_bars(symbol, timeframe, start, end, limit),
        )

    async def get_latest_quote(self, symbol: str) -> Dict[str, float]:
        """Get quote with failover."""
        return await self._execute_with_failover(
            f"get_latest_quote({symbol})",
            lambda b: b.get_latest_quote(symbol),
        )

    async def get_clock(self) -> Dict[str, Any]:
        """Get market clock with failover."""
        return await self._execute_with_failover(
            "get_clock",
            lambda b: b.get_clock(),
        )

    async def close_position(self, symbol: str) -> Optional[Order]:
        """Close position with failover."""
        return await self._execute_with_failover(
            f"close_position({symbol})",
            lambda b: b.close_position(symbol),
        )

    async def close_all_positions(self) -> List[Order]:
        """Close all positions with failover."""
        return await self._execute_with_failover(
            "close_all_positions",
            lambda b: b.close_all_positions(),
        )

    # === Status & Reporting ===

    def get_broker_health(self) -> Dict[str, BrokerHealth]:
        """Get health status for all brokers."""
        return self._broker_health.copy()

    def get_failover_log(self, limit: int = 50) -> List[FailoverLog]:
        """Get recent failover events."""
        return self._failover_log[-limit:]

    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary for monitoring."""
        return {
            "active_broker": self._active_broker.name,
            "is_failed_over": self._is_failed_over,
            "primary_health": self._broker_health[self._primary.name].__dict__,
            "backup_health": [self._broker_health[b.name].__dict__ for b in self._backups],
            "recent_failovers": len(
                [
                    f
                    for f in self._failover_log
                    if f.timestamp > datetime.now() - timedelta(hours=24)
                ]
            ),
            "total_failovers": len(
                [f for f in self._failover_log if f.event == FailoverEvent.FAILOVER_TO_BACKUP]
            ),
        }


def print_broker_status(manager: MultiBrokerManager):
    """Print formatted broker status."""
    status = manager.get_status_summary()

    print("\n" + "=" * 60)
    print("MULTI-BROKER STATUS")
    print("=" * 60)
    print(f"Active Broker: {status['active_broker']}")
    print(f"Failed Over: {'Yes' if status['is_failed_over'] else 'No'}")

    print("\n--- Broker Health ---")
    primary = status["primary_health"]
    print(f"Primary ({primary['broker_name']}): {primary['status']}")
    print(f"  Last check: {primary['last_check']}")
    print(f"  Consecutive failures: {primary['consecutive_failures']}")
    if primary["response_time_ms"]:
        print(f"  Response time: {primary['response_time_ms']:.0f}ms")

    for backup in status["backup_health"]:
        print(f"\nBackup ({backup['broker_name']}): {backup['status']}")
        print(f"  Consecutive failures: {backup['consecutive_failures']}")

    print(f"\nFailovers (last 24h): {status['recent_failovers']}")
    print(f"Total failovers: {status['total_failovers']}")
    print("=" * 60 + "\n")
