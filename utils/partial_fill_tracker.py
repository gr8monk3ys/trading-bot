"""
Partial Fill Tracking System for Institutional-Grade Order Management.

This module tracks order fills and handles partial fills according to
configurable policies. It ensures position tracking remains accurate
even when orders don't fill completely.

Key Features:
- Track requested vs filled quantities
- Configurable policies for unfilled portions
- Callbacks for strategy-level handling
- Auto-resubmit option for unfilled quantities
- Detailed fill statistics for analysis
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PartialFillPolicy(Enum):
    """
    Policy for handling partial fills.

    ALERT_ONLY: Log and notify, but take no automatic action
    AUTO_RESUBMIT: Automatically resubmit order for unfilled quantity
    CANCEL_REMAINDER: Cancel any unfilled portion
    TRACK_ONLY: Just track the fill, no alerts or actions
    """

    ALERT_ONLY = "alert_only"
    AUTO_RESUBMIT = "auto_resubmit"
    CANCEL_REMAINDER = "cancel_remainder"
    TRACK_ONLY = "track_only"


@dataclass
class PartialFillEvent:
    """Record of a partial fill event."""

    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    requested_qty: float
    filled_qty: float
    unfilled_qty: float
    fill_price: float
    timestamp: datetime
    event_type: str  # 'partial_fill', 'fill', 'canceled', 'rejected'
    delta_qty: float = 0.0

    @property
    def fill_rate(self) -> float:
        """Percentage of order that was filled."""
        if self.requested_qty <= 0:
            return 0.0
        return self.filled_qty / self.requested_qty

    @property
    def is_complete(self) -> bool:
        """Whether the order is fully filled."""
        return self.unfilled_qty <= 0


@dataclass
class OrderTrackingRecord:
    """Internal record for tracking an order's fill status."""

    order_id: str
    symbol: str
    side: str
    requested_qty: float
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)
    status: str = "pending"  # pending, partial, filled, canceled, rejected
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resubmit_attempts: int = 0
    child_order_ids: List[str] = field(default_factory=list)

    @property
    def unfilled_qty(self) -> float:
        """Calculate unfilled quantity."""
        return max(0, self.requested_qty - self.filled_qty)


@dataclass
class PartialFillStatistics:
    """Aggregate statistics on partial fills."""

    total_orders: int
    fully_filled_orders: int
    partially_filled_orders: int
    canceled_orders: int
    rejected_orders: int
    total_requested_qty: float
    total_filled_qty: float
    total_unfilled_qty: float
    average_fill_rate: float
    orders_below_90_pct_fill: int
    auto_resubmits_triggered: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_orders": self.total_orders,
            "fully_filled_orders": self.fully_filled_orders,
            "partially_filled_orders": self.partially_filled_orders,
            "canceled_orders": self.canceled_orders,
            "rejected_orders": self.rejected_orders,
            "total_requested_qty": self.total_requested_qty,
            "total_filled_qty": self.total_filled_qty,
            "total_unfilled_qty": self.total_unfilled_qty,
            "average_fill_rate": self.average_fill_rate,
            "orders_below_90_pct_fill": self.orders_below_90_pct_fill,
            "auto_resubmits_triggered": self.auto_resubmits_triggered,
        }


class PartialFillTracker:
    """
    Tracks order fills and handles partial fills according to policy.

    Usage:
        tracker = PartialFillTracker(policy=PartialFillPolicy.ALERT_ONLY)
        tracker.register_callback(my_strategy.on_partial_fill)

        # When submitting an order
        tracker.track_order(order_id, symbol, side, qty)

        # When receiving fill updates
        tracker.record_fill(order_id, filled_qty, fill_price)
    """

    def __init__(
        self,
        policy: PartialFillPolicy = PartialFillPolicy.ALERT_ONLY,
        max_resubmit_attempts: int = 3,
        min_resubmit_qty: float = 1.0,
        fill_rate_threshold: float = 0.90,
    ):
        """
        Initialize the partial fill tracker.

        Args:
            policy: How to handle partial fills
            max_resubmit_attempts: Maximum times to resubmit unfilled portion
            min_resubmit_qty: Minimum quantity to resubmit (avoid 1-share orders)
            fill_rate_threshold: Below this fill rate, take action per policy
        """
        self.policy = policy
        self.max_resubmit_attempts = max_resubmit_attempts
        self.min_resubmit_qty = min_resubmit_qty
        self.fill_rate_threshold = fill_rate_threshold

        # Order tracking
        self._orders: Dict[str, OrderTrackingRecord] = {}
        self._fill_events: List[PartialFillEvent] = []

        # Callbacks
        self._callbacks: List[Callable] = []
        self._resubmit_callback: Optional[Callable] = None

        # Statistics
        self._auto_resubmits_triggered = 0

    def set_policy(self, policy: PartialFillPolicy) -> None:
        """Change the partial fill policy."""
        self.policy = policy
        logger.info(f"Partial fill policy changed to: {policy.value}")

    def register_callback(self, callback: Callable) -> None:
        """
        Register a callback for partial fill events.

        The callback will be called with a PartialFillEvent when a partial
        fill is detected.

        Args:
            callback: Async function taking PartialFillEvent
        """
        self._callbacks.append(callback)

    def set_resubmit_callback(self, callback: Callable) -> None:
        """
        Set the callback for resubmitting orders.

        This callback will be called when AUTO_RESUBMIT policy triggers
        and there's unfilled quantity to resubmit.

        Args:
            callback: Async function taking (symbol, side, qty)
                     Should return the new order_id or None on failure
        """
        self._resubmit_callback = callback

    def track_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        requested_qty: float,
    ) -> None:
        """
        Start tracking an order.

        Call this when submitting a new order.

        Args:
            order_id: Unique order identifier
            symbol: Stock symbol
            side: 'buy' or 'sell'
            requested_qty: Quantity requested
        """
        self._orders[order_id] = OrderTrackingRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            requested_qty=requested_qty,
            status="pending",
        )
        logger.debug(f"Tracking order {order_id}: {side} {requested_qty} {symbol}")

    async def record_fill(
        self,
        order_id: str,
        filled_qty: float,
        fill_price: float,
        is_final: bool = False,
        status: Optional[str] = None,
    ) -> Optional[PartialFillEvent]:
        """
        Record a fill for a tracked order.

        Call this when receiving fill updates from the broker.

        Args:
            order_id: The order that received a fill
            filled_qty: Cumulative filled quantity (total filled so far)
            fill_price: Average fill price
            is_final: Whether this is the final fill update (order closed)
            status: Order status (filled, partial, canceled, rejected)

        Returns:
            PartialFillEvent if this was a partial fill, None otherwise
        """
        if order_id not in self._orders:
            logger.warning(f"Fill for untracked order {order_id}")
            return None

        record = self._orders[order_id]
        prev_filled = record.filled_qty

        # Update tracking record
        record.filled_qty = filled_qty
        record.avg_fill_price = fill_price
        record.updated_at = datetime.now()

        if status:
            record.status = status

        # Calculate fill metrics
        new_fill_qty = filled_qty - prev_filled
        unfilled_qty = record.unfilled_qty

        # Record individual fill
        record.fills.append(
            {
                "qty": new_fill_qty,
                "price": fill_price,
                "timestamp": datetime.now(),
            }
        )

        # Determine event type
        if unfilled_qty <= 0:
            event_type = "fill"
            record.status = "filled"
        elif is_final:
            if status == "canceled":
                event_type = "canceled"
                record.status = "canceled"
            elif status == "rejected":
                event_type = "rejected"
                record.status = "rejected"
            else:
                event_type = "partial_fill"
                record.status = "partial"
        else:
            event_type = "partial_fill" if unfilled_qty > 0 else "fill"
            if unfilled_qty > 0:
                record.status = "partial"

        # Create event
        event = PartialFillEvent(
            order_id=order_id,
            symbol=record.symbol,
            side=record.side,
            requested_qty=record.requested_qty,
            filled_qty=filled_qty,
            unfilled_qty=unfilled_qty,
            delta_qty=new_fill_qty,
            fill_price=fill_price,
            timestamp=datetime.now(),
            event_type=event_type,
        )
        self._fill_events.append(event)

        # Handle partial fill based on policy
        if unfilled_qty > 0 and is_final:
            await self._handle_partial_fill(record, event)

        return event

    async def _handle_partial_fill(
        self,
        record: OrderTrackingRecord,
        event: PartialFillEvent,
    ) -> None:
        """Handle a partial fill according to policy."""
        fill_rate = event.fill_rate

        # Check if below threshold
        if fill_rate >= self.fill_rate_threshold:
            logger.info(
                f"Order {event.order_id} fill rate {fill_rate:.1%} >= threshold {self.fill_rate_threshold:.1%}"
            )
            return

        # Log the partial fill
        logger.warning(
            f"PARTIAL FILL: {event.symbol} order {event.order_id} - "
            f"filled {event.filled_qty}/{event.requested_qty} ({fill_rate:.1%}), "
            f"unfilled: {event.unfilled_qty}"
        )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in partial fill callback: {e}")

        # Apply policy
        if self.policy == PartialFillPolicy.TRACK_ONLY:
            pass  # Do nothing

        elif self.policy == PartialFillPolicy.ALERT_ONLY:
            # Already logged above
            pass

        elif self.policy == PartialFillPolicy.AUTO_RESUBMIT:
            await self._auto_resubmit(record, event)

        elif self.policy == PartialFillPolicy.CANCEL_REMAINDER:
            logger.info(f"Remainder for {event.order_id} not resubmitted (CANCEL_REMAINDER policy)")

    async def _auto_resubmit(
        self,
        record: OrderTrackingRecord,
        event: PartialFillEvent,
    ) -> None:
        """Automatically resubmit order for unfilled quantity."""
        # Check if we can resubmit
        if record.resubmit_attempts >= self.max_resubmit_attempts:
            logger.warning(
                f"Max resubmit attempts ({self.max_resubmit_attempts}) reached for {event.order_id}"
            )
            return

        if event.unfilled_qty < self.min_resubmit_qty:
            logger.info(
                f"Unfilled qty {event.unfilled_qty} below minimum {self.min_resubmit_qty}, not resubmitting"
            )
            return

        if not self._resubmit_callback:
            logger.warning("AUTO_RESUBMIT policy but no resubmit callback set")
            return

        # Attempt resubmit
        try:
            logger.info(
                f"Auto-resubmitting {event.unfilled_qty} {event.symbol} "
                f"(attempt {record.resubmit_attempts + 1}/{self.max_resubmit_attempts})"
            )

            new_order_id = await self._resubmit_callback(
                event.symbol,
                event.side,
                event.unfilled_qty,
            )

            if new_order_id:
                record.resubmit_attempts += 1
                record.child_order_ids.append(new_order_id)
                self._auto_resubmits_triggered += 1

                # Track the new order
                self.track_order(
                    new_order_id,
                    event.symbol,
                    event.side,
                    event.unfilled_qty,
                )

                logger.info(f"Successfully resubmitted as order {new_order_id}")
            else:
                logger.error(f"Resubmit callback returned None for {event.order_id}")

        except Exception as e:
            logger.error(f"Error auto-resubmitting order: {e}")

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a tracked order.

        Returns:
            Dict with order status or None if not tracked
        """
        if order_id not in self._orders:
            return None

        record = self._orders[order_id]
        return {
            "order_id": record.order_id,
            "symbol": record.symbol,
            "side": record.side,
            "requested_qty": record.requested_qty,
            "filled_qty": record.filled_qty,
            "unfilled_qty": record.unfilled_qty,
            "avg_fill_price": record.avg_fill_price,
            "fill_rate": record.filled_qty / record.requested_qty if record.requested_qty > 0 else 0,
            "status": record.status,
            "fills_count": len(record.fills),
            "resubmit_attempts": record.resubmit_attempts,
            "child_orders": record.child_order_ids,
        }

    def get_unfilled_qty(self, order_id: str) -> float:
        """Get the unfilled quantity for an order."""
        if order_id not in self._orders:
            return 0.0
        return self._orders[order_id].unfilled_qty

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all orders that still have unfilled quantities."""
        return [
            self.get_order_status(oid)
            for oid, record in self._orders.items()
            if record.status in ("pending", "partial")
        ]

    def detect_stalled_orders(self, max_stall_seconds: float = 300.0) -> List[Dict[str, Any]]:
        """
        Detect pending/partial orders with no recent fill update.

        Args:
            max_stall_seconds: Age threshold since last update.

        Returns:
            List of order status dicts augmented with `stall_seconds`.
        """
        threshold = max(1.0, float(max_stall_seconds))
        now = datetime.now()
        stalled: List[Dict[str, Any]] = []
        for order_id, record in self._orders.items():
            if record.status not in ("pending", "partial"):
                continue
            if record.unfilled_qty <= 0:
                continue
            stall_seconds = (now - record.updated_at).total_seconds()
            if stall_seconds < threshold:
                continue

            status = self.get_order_status(order_id) or {"order_id": order_id}
            status["stall_seconds"] = stall_seconds
            stalled.append(status)

        return stalled

    def get_statistics(self) -> PartialFillStatistics:
        """Calculate aggregate statistics on partial fills."""
        orders = list(self._orders.values())

        if not orders:
            return PartialFillStatistics(
                total_orders=0,
                fully_filled_orders=0,
                partially_filled_orders=0,
                canceled_orders=0,
                rejected_orders=0,
                total_requested_qty=0.0,
                total_filled_qty=0.0,
                total_unfilled_qty=0.0,
                average_fill_rate=0.0,
                orders_below_90_pct_fill=0,
                auto_resubmits_triggered=0,
            )

        fully_filled = sum(1 for o in orders if o.status == "filled")
        partially_filled = sum(1 for o in orders if o.status == "partial")
        canceled = sum(1 for o in orders if o.status == "canceled")
        rejected = sum(1 for o in orders if o.status == "rejected")

        total_requested = sum(o.requested_qty for o in orders)
        total_filled = sum(o.filled_qty for o in orders)

        fill_rates = [o.filled_qty / o.requested_qty for o in orders if o.requested_qty > 0]
        avg_fill_rate = sum(fill_rates) / len(fill_rates) if fill_rates else 0.0

        below_90_pct = sum(1 for rate in fill_rates if rate < 0.90)

        return PartialFillStatistics(
            total_orders=len(orders),
            fully_filled_orders=fully_filled,
            partially_filled_orders=partially_filled,
            canceled_orders=canceled,
            rejected_orders=rejected,
            total_requested_qty=total_requested,
            total_filled_qty=total_filled,
            total_unfilled_qty=total_requested - total_filled,
            average_fill_rate=avg_fill_rate,
            orders_below_90_pct_fill=below_90_pct,
            auto_resubmits_triggered=self._auto_resubmits_triggered,
        )

    def get_fill_events(
        self,
        symbol: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[PartialFillEvent]:
        """
        Get recorded fill events with optional filtering.

        Args:
            symbol: Filter by symbol
            event_type: Filter by event type (partial_fill, fill, canceled, rejected)

        Returns:
            List of matching PartialFillEvent objects
        """
        events = self._fill_events

        if symbol:
            events = [e for e in events if e.symbol == symbol]
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events

    def clear(self) -> None:
        """Clear all tracking data (for new backtest run)."""
        self._orders.clear()
        self._fill_events.clear()
        self._auto_resubmits_triggered = 0
        logger.debug("Partial fill tracker cleared")
