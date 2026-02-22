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
import inspect
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from types import SimpleNamespace
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
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
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
        for broker in self._all_brokers:
            if not hasattr(broker, "name"):
                derived_name = getattr(broker, "NAME", broker.__class__.__name__)
                try:
                    broker.name = str(derived_name)
                except Exception:
                    # Fallback: leave object unchanged; later attribute access may fail loudly.
                    pass

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
        self._gateway_token: str | None = None
        self._gateway_enforced = False

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

    @staticmethod
    async def _maybe_await(result: Any) -> Any:
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _normalize_account_shape(account: Any) -> Any:
        if account is None:
            return account
        if isinstance(account, dict):
            payload = dict(account)
            payload.setdefault("id", payload.get("account_id") or payload.get("id") or "unknown")
            payload.setdefault("equity", payload.get("equity", 0))
            payload.setdefault("cash", payload.get("cash", 0))
            payload.setdefault("buying_power", payload.get("buying_power", 0))
            return SimpleNamespace(**payload)
        if hasattr(account, "account_id") and not hasattr(account, "id"):
            account.id = str(account.account_id)
        if not hasattr(account, "buying_power"):
            account.buying_power = 0
        return account

    @staticmethod
    def _normalize_position_shape(position: Any) -> Any:
        if position is None:
            return position
        if isinstance(position, dict):
            payload = dict(position)
            qty = payload.get("qty", payload.get("quantity", 0))
            payload["qty"] = qty
            payload.setdefault("quantity", qty)
            payload.setdefault("avg_entry_price", payload.get("entry_price", 0))
            payload.setdefault("current_price", payload.get("avg_entry_price", 0))
            payload.setdefault("unrealized_pl", payload.get("unrealized_pnl", 0))
            payload.setdefault("unrealized_plpc", payload.get("unrealized_pnl_pct", 0))
            return SimpleNamespace(**payload)
        if hasattr(position, "quantity") and not hasattr(position, "qty"):
            position.qty = position.quantity
        if hasattr(position, "unrealized_pnl") and not hasattr(position, "unrealized_pl"):
            position.unrealized_pl = position.unrealized_pnl
        if hasattr(position, "unrealized_pnl_pct") and not hasattr(position, "unrealized_plpc"):
            position.unrealized_plpc = position.unrealized_pnl_pct
        if not hasattr(position, "current_price"):
            position.current_price = getattr(position, "avg_entry_price", 0)
        return position

    @staticmethod
    def _normalize_order_shape(order: Any) -> Any:
        if order is None:
            return order
        if isinstance(order, dict):
            payload = dict(order)
            order_id = payload.get("id", payload.get("order_id"))
            payload["id"] = str(order_id) if order_id is not None else ""
            payload.setdefault("order_id", payload["id"])
            qty = payload.get("qty", payload.get("quantity", 0))
            payload["qty"] = qty
            payload.setdefault("quantity", qty)
            payload.setdefault("filled_qty", payload.get("filled_quantity", 0))
            return SimpleNamespace(**payload)
        if hasattr(order, "order_id") and not hasattr(order, "id"):
            order.id = str(order.order_id)
        if hasattr(order, "quantity") and not hasattr(order, "qty"):
            order.qty = order.quantity
        if hasattr(order, "filled_quantity") and not hasattr(order, "filled_qty"):
            order.filled_qty = order.filled_quantity
        return order

    @staticmethod
    def _normalize_clock_shape(clock: Any) -> Any:
        if isinstance(clock, dict):
            return SimpleNamespace(**clock)
        return clock

    def _append_failover_event(
        self,
        *,
        event: FailoverEvent,
        reason: str,
        from_broker: str | None,
        to_broker: str | None,
        details: Dict[str, Any] | None = None,
    ) -> None:
        entry = FailoverLog(
            timestamp=datetime.now(),
            event=event,
            from_broker=from_broker,
            to_broker=to_broker,
            reason=reason,
            details=details or {},
        )
        self._failover_log.append(entry)
        if self.on_failover:
            try:
                self.on_failover(entry)
            except Exception as e:
                logger.error(f"Failover callback error: {e}")

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

    # === Compatibility helpers for existing Alpaca-oriented runtime ===

    def set_audit_log(self, audit_log: Any) -> None:
        for broker in self._all_brokers:
            handler = getattr(broker, "set_audit_log", None)
            if callable(handler):
                try:
                    handler(audit_log)
                except Exception as e:
                    logger.warning(f"Failed to set audit log on {broker.name}: {e}")

    def set_position_manager(self, position_manager: Any) -> None:
        for broker in self._all_brokers:
            handler = getattr(broker, "set_position_manager", None)
            if callable(handler):
                try:
                    handler(position_manager)
                except Exception as e:
                    logger.warning(f"Failed to set position manager on {broker.name}: {e}")

    def set_lifecycle_tracker(self, lifecycle_tracker: Any) -> None:
        for broker in self._all_brokers:
            handler = getattr(broker, "set_lifecycle_tracker", None)
            if callable(handler):
                try:
                    handler(lifecycle_tracker)
                except Exception as e:
                    logger.warning(f"Failed to set lifecycle tracker on {broker.name}: {e}")

    def register_order_metadata(self, order_id: str, metadata: Dict[str, Any]) -> None:
        handler = getattr(self._active_broker, "register_order_metadata", None)
        if callable(handler):
            try:
                handler(order_id, metadata)
            except Exception as e:
                logger.warning(
                    f"Failed to register order metadata on {self._active_broker.name}: {e}"
                )

    def track_order_for_fills(self, order_id: str, symbol: str, side: str, qty: float) -> None:
        handler = getattr(self._active_broker, "track_order_for_fills", None)
        if callable(handler):
            try:
                handler(order_id, symbol, side, qty)
            except Exception as e:
                logger.warning(
                    f"Failed to track order for fills on {self._active_broker.name}: {e}"
                )

    def enable_gateway_requirement(self) -> str:
        handler = getattr(self._primary, "enable_gateway_requirement", None)
        if callable(handler):
            token = str(handler())
        else:
            token = secrets.token_urlsafe(32)
        self._gateway_token = token
        self._gateway_enforced = True
        return token

    def disable_gateway_requirement(self) -> None:
        self._gateway_enforced = False
        self._gateway_token = None
        handler = getattr(self._primary, "disable_gateway_requirement", None)
        if callable(handler):
            try:
                handler()
            except Exception as e:
                logger.warning(f"Failed to disable gateway requirement on primary: {e}")

    async def start_websocket(self, symbols: Optional[List[str]] = None) -> None:
        handler = getattr(self._primary, "start_websocket", None)
        if callable(handler):
            if symbols is not None:
                try:
                    if len(inspect.signature(handler).parameters) > 0:
                        await self._maybe_await(handler(symbols))
                        return
                except (TypeError, ValueError):
                    pass
            await self._maybe_await(handler())

    async def stop_websocket(self) -> None:
        handler = getattr(self._primary, "stop_websocket", None)
        if callable(handler):
            await self._maybe_await(handler())

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

                reason = f"Primary failed ({self._broker_health[self._primary.name].error_message})"
                logger.warning(f"FAILOVER: {old_broker} -> {backup.name} ({reason})")
                self._append_failover_event(
                    event=FailoverEvent.FAILOVER_TO_BACKUP,
                    from_broker=old_broker,
                    to_broker=backup.name,
                    reason=reason,
                )

                return

        # No healthy backup found
        self._append_failover_event(
            event=FailoverEvent.ALL_BROKERS_FAILED,
            from_broker=self._active_broker.name,
            to_broker=None,
            reason="All brokers unavailable",
        )
        logger.critical("ALL BROKERS FAILED - no healthy backup available")

    async def _do_failback(self):
        """Execute failback to primary broker."""
        old_broker = self._active_broker.name
        self._active_broker = self._primary
        self._is_failed_over = False
        self._primary_recovery_count = 0

        logger.info(f"FAILBACK: {old_broker} -> {self._primary.name}")
        self._append_failover_event(
            event=FailoverEvent.FAILBACK_TO_PRIMARY,
            from_broker=old_broker,
            to_broker=self._primary.name,
            reason="Primary recovered",
        )

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
        account = await self._execute_with_failover(
            "get_account",
            lambda b: b.get_account(),
        )
        return self._normalize_account_shape(account)

    async def get_positions(self) -> List[Position]:
        """Get positions with failover."""
        positions = await self._execute_with_failover(
            "get_positions",
            lambda b: b.get_positions(),
        )
        return [self._normalize_position_shape(p) for p in positions]

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol with failover."""
        position = await self._execute_with_failover(
            f"get_position({symbol})",
            lambda b: b.get_position(symbol),
        )
        return self._normalize_position_shape(position)

    @staticmethod
    def _extract_value(value: Any) -> str:
        raw = getattr(value, "value", value)
        return str(raw).strip().lower()

    @classmethod
    def _map_order_type(cls, order_request: Any) -> OrderType:
        value = cls._extract_value(
            getattr(order_request, "order_type", None) or getattr(order_request, "type", "market")
        )
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
            "trailing_stop": OrderType.TRAILING_STOP,
        }
        return mapping.get(value, OrderType.MARKET)

    @classmethod
    def _map_time_in_force(cls, order_request: Any) -> TimeInForce:
        value = cls._extract_value(getattr(order_request, "time_in_force", TimeInForce.DAY))
        mapping = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
            "opg": TimeInForce.OPG,
            "cls": TimeInForce.CLS,
        }
        return mapping.get(value, TimeInForce.DAY)

    @classmethod
    def _to_standard_order_request(cls, order_request: Any) -> OrderRequest:
        symbol = str(getattr(order_request, "symbol", "") or "").strip()
        if not symbol:
            raise BrokerError("Order request missing symbol")

        quantity = getattr(order_request, "qty", None) or getattr(order_request, "quantity", None)
        if quantity is None:
            raise BrokerError("Backup broker failover requires explicit quantity-based orders")
        try:
            quantity = float(quantity)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"Invalid order quantity for failover: {quantity}") from exc
        if quantity <= 0:
            raise BrokerError("Order quantity must be positive for failover")

        raw_side = getattr(order_request, "side", "buy")
        side = OrderSide.SELL if cls._extract_value(raw_side) == "sell" else OrderSide.BUY

        order_class = cls._extract_value(getattr(order_request, "order_class", "simple"))
        if order_class not in {"", "simple"}:
            raise BrokerError(f"Backup failover does not support order_class={order_class}")
        if (
            getattr(order_request, "take_profit", None) is not None
            or getattr(order_request, "stop_loss", None) is not None
        ):
            raise BrokerError("Backup failover does not support bracket/OCO/OTO legs")

        return OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=cls._map_order_type(order_request),
            limit_price=getattr(order_request, "limit_price", None),
            stop_price=getattr(order_request, "stop_price", None),
            time_in_force=cls._map_time_in_force(order_request),
            client_order_id=getattr(order_request, "client_order_id", None),
            extended_hours=bool(getattr(order_request, "extended_hours", False)),
        )

    async def _submit_on_backup_brokers(self, order_request: Any) -> Any:
        standard_request = self._to_standard_order_request(order_request)
        last_error: Exception | None = None

        for backup in self._backups:
            try:
                result = await self._maybe_await(backup.submit_order(standard_request))
            except Exception as exc:
                last_error = exc
                logger.warning("Backup order submission failed on %s: %s", backup.name, exc)
                continue

            old_broker = self._active_broker.name
            self._active_broker = backup
            self._is_failed_over = True
            health = self._broker_health.get(backup.name)
            if health:
                health.consecutive_failures = 0
                health.last_success = datetime.now()
                health.status = BrokerStatus.CONNECTED
            self._append_failover_event(
                event=FailoverEvent.FAILOVER_TO_BACKUP,
                from_broker=old_broker,
                to_broker=backup.name,
                reason="Primary order submission failed; backup accepted order",
                details={"operation": "submit_order_advanced"},
            )
            return self._normalize_order_shape(result)

        if last_error:
            raise last_error
        raise BrokerError("No backup broker configured for failover order submission")

    async def submit_order_advanced(self, order_request, check_impact: bool = True):
        primary_method = getattr(self._primary, "submit_order_advanced", None)
        if callable(primary_method):
            try:
                result = await self._maybe_await(primary_method(order_request, check_impact))
                return self._normalize_order_shape(result)
            except Exception as exc:
                logger.warning("Primary submit_order_advanced failed: %s", exc)
        return await self._submit_on_backup_brokers(order_request)

    async def _internal_submit_order(
        self,
        order_request,
        gateway_token: str,
        check_impact: bool = True,
    ):
        if self._gateway_enforced and (not gateway_token or gateway_token != self._gateway_token):
            raise BrokerError("Invalid gateway authorization token for multi-broker order path")

        primary_internal = getattr(self._primary, "_internal_submit_order", None)
        if callable(primary_internal):
            try:
                result = await self._maybe_await(
                    primary_internal(
                        order_request,
                        gateway_token=gateway_token,
                        check_impact=check_impact,
                    )
                )
                return self._normalize_order_shape(result)
            except Exception as exc:
                logger.warning("Primary _internal_submit_order failed: %s", exc)
        return await self._submit_on_backup_brokers(order_request)

    async def submit_order(self, request: OrderRequest) -> Order:
        """Submit order with failover."""
        # Warning: Order submission on backup broker creates position sync issues
        if self._is_failed_over:
            logger.warning(
                f"Submitting order on BACKUP broker {self._active_broker.name}. "
                "Positions may not sync with primary when it recovers."
            )

        order = await self._execute_with_failover(
            f"submit_order({request.symbol})",
            lambda b: b.submit_order(request),
        )
        return self._normalize_order_shape(order)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with failover."""
        return await self._execute_with_failover(
            f"cancel_order({order_id})",
            lambda b: b.cancel_order(order_id),
        )

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order with failover."""
        order = await self._execute_with_failover(
            f"get_order({order_id})",
            lambda b: b.get_order(order_id),
        )
        return self._normalize_order_shape(order)

    async def get_order_by_id(self, order_id: str) -> Optional[Any]:
        """Compatibility alias for Alpaca-style order lookup."""
        return await self.get_order(order_id)

    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbols: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get orders with failover."""
        orders = await self._execute_with_failover(
            "get_orders",
            lambda b: b.get_orders(status, symbols, limit),
        )
        return [self._normalize_order_shape(order) for order in orders]

    async def cancel_all_orders(self) -> int:
        """Cancel all orders with failover."""
        return await self._execute_with_failover(
            "cancel_all_orders",
            lambda b: b.cancel_all_orders(),
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Any = None,
        end: Optional[Any] = None,
        limit: int = 1000,
    ) -> List[Bar]:
        """Get bars with failover."""
        bars = await self._execute_with_failover(
            f"get_bars({symbol})",
            lambda b: b.get_bars(symbol, timeframe, start, end, limit),
        )
        return bars

    async def get_latest_quote(self, symbol: str) -> Dict[str, float]:
        """Get quote with failover."""
        return await self._execute_with_failover(
            f"get_latest_quote({symbol})",
            lambda b: b.get_latest_quote(symbol),
        )

    async def get_clock(self) -> Dict[str, Any]:
        """Get market clock with failover."""
        clock = await self._execute_with_failover(
            "get_clock",
            lambda b: b.get_clock(),
        )
        return self._normalize_clock_shape(clock)

    async def get_market_status(self) -> Dict[str, Any]:
        """Compatibility method for Alpaca-style market status payload."""
        handler = getattr(self._active_broker, "get_market_status", None)
        if callable(handler):
            try:
                result = await self._maybe_await(handler())
                if isinstance(result, dict):
                    return result
                return {
                    "is_open": bool(getattr(result, "is_open", False)),
                    "next_open": str(getattr(result, "next_open", "")),
                    "next_close": str(getattr(result, "next_close", "")),
                }
            except Exception as exc:
                logger.warning(
                    "Primary get_market_status failed on %s: %s", self._active_broker.name, exc
                )

        clock = await self.get_clock()
        return {
            "is_open": bool(getattr(clock, "is_open", False)),
            "next_open": str(getattr(clock, "next_open", "")),
            "next_close": str(getattr(clock, "next_close", "")),
        }

    async def get_last_price(self, symbol: str) -> Optional[float]:
        """Compatibility method for Alpaca-style latest trade price."""
        handler = getattr(self._active_broker, "get_last_price", None)
        if callable(handler):
            try:
                price = await self._maybe_await(handler(symbol))
                if price is not None:
                    return float(price)
            except Exception as exc:
                logger.warning(
                    "Primary get_last_price failed on %s: %s", self._active_broker.name, exc
                )

        quote = await self.get_latest_quote(symbol)
        if isinstance(quote, dict):
            for key in ("last", "price", "ask", "bid"):
                value = quote.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    async def get_last_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Compatibility method for batch latest prices."""
        prices: Dict[str, Optional[float]] = {}
        for symbol in symbols:
            prices[str(symbol)] = await self.get_last_price(str(symbol))
        return prices

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
