"""
Order Lifecycle State Machine.

Tracks order status transitions and validates allowed transitions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OrderState(Enum):
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


ALLOWED_TRANSITIONS = {
    OrderState.SUBMITTED: {
        OrderState.PARTIAL,
        OrderState.FILLED,
        OrderState.CANCELED,
        OrderState.REJECTED,
    },
    OrderState.PARTIAL: {
        OrderState.PARTIAL,
        OrderState.FILLED,
        OrderState.CANCELED,
        OrderState.REJECTED,
    },
    OrderState.FILLED: set(),
    OrderState.CANCELED: set(),
    OrderState.REJECTED: set(),
}


@dataclass
class OrderLifecycleRecord:
    order_id: str
    symbol: str
    side: str
    quantity: float
    strategy_name: Optional[str]
    client_order_id: Optional[str] = None
    state: OrderState = OrderState.SUBMITTED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class OrderLifecycleTracker:
    def __init__(self):
        self._orders: Dict[str, OrderLifecycleRecord] = {}

    def register_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        strategy_name: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> None:
        self._orders[order_id] = OrderLifecycleRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            strategy_name=strategy_name,
            client_order_id=client_order_id,
        )

    def update_state(self, order_id: str, new_state: OrderState) -> None:
        record = self._orders.get(order_id)
        if not record:
            logger.warning(f"Lifecycle update for unknown order {order_id}")
            return

        allowed = ALLOWED_TRANSITIONS.get(record.state, set())
        if new_state not in allowed and new_state != record.state:
            logger.warning(
                f"Invalid order transition {record.state.value} -> {new_state.value} for {order_id}"
            )
            return

        record.state = new_state
        record.updated_at = datetime.now()

    def get_state(self, order_id: str) -> Optional[OrderState]:
        record = self._orders.get(order_id)
        return record.state if record else None

    def export_state(self) -> Dict[str, Dict]:
        """Export lifecycle state for persistence."""
        return {
            oid: {
                "order_id": rec.order_id,
                "symbol": rec.symbol,
                "side": rec.side,
                "quantity": rec.quantity,
                "strategy_name": rec.strategy_name,
                "client_order_id": rec.client_order_id,
                "state": rec.state.value,
                "created_at": rec.created_at.isoformat(),
                "updated_at": rec.updated_at.isoformat(),
            }
            for oid, rec in self._orders.items()
        }

    def import_state(self, state: Dict[str, Dict]) -> None:
        """Restore lifecycle state from persistence."""
        self._orders.clear()
        for oid, data in state.items():
            self._orders[oid] = OrderLifecycleRecord(
                order_id=data["order_id"],
                symbol=data["symbol"],
                side=data["side"],
                quantity=float(data["quantity"]),
                strategy_name=data.get("strategy_name"),
                client_order_id=data.get("client_order_id"),
                state=OrderState(data["state"]),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
            )
