"""
Order Reconciliation - verify broker orders vs internal lifecycle tracker.
"""

import logging
from typing import Optional

from utils.order_lifecycle import OrderState

logger = logging.getLogger(__name__)


class OrderReconciler:
    def __init__(self, broker, lifecycle_tracker, audit_log=None):
        self.broker = broker
        self.lifecycle_tracker = lifecycle_tracker
        self.audit_log = audit_log

    async def reconcile(self) -> None:
        """
        Reconcile broker open orders with internal lifecycle tracker.
        Marks missing orders as canceled if not filled.
        """
        try:
            broker_orders = await self.broker.get_orders()
        except Exception as e:
            logger.error(f"Order reconciliation failed to fetch broker orders: {e}")
            return

        open_ids = {str(o.id): o for o in broker_orders}

        for order_id, record in list(self.lifecycle_tracker._orders.items()):
            state = record.state
            if state in (OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED):
                continue
            if order_id not in open_ids:
                logger.warning(f"Order {order_id} missing from broker open orders; marking canceled")
                self.lifecycle_tracker.update_state(order_id, OrderState.CANCELED)
