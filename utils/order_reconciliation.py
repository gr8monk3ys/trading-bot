"""
Order Reconciliation - verify broker orders vs internal lifecycle tracker.
"""

import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from utils.audit_log import AuditEventType
from utils.order_lifecycle import OrderState
from utils.run_artifacts import JsonlWriter

logger = logging.getLogger(__name__)


class OrderReconciler:
    _TERMINAL_STATES = {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED}

    def __init__(
        self,
        broker,
        lifecycle_tracker,
        audit_log=None,
        mismatch_halt_threshold: int = 3,
        events_path: str | Path | None = None,
        run_id: Optional[str] = None,
    ):
        self.broker = broker
        self.lifecycle_tracker = lifecycle_tracker
        self.audit_log = audit_log
        self.mismatch_halt_threshold = max(1, int(mismatch_halt_threshold))
        self.run_id = run_id
        self._events_writer = JsonlWriter(events_path) if events_path else None

        self._runs_total = 0
        self._total_mismatches = 0
        self._consecutive_mismatch_runs = 0
        self._last_run_mismatches: list[dict[str, Any]] = []
        self._halt_recommended = False
        self._last_halt_reason: Optional[str] = None
        self._last_error: Optional[str] = None

    async def reconcile(self) -> None:
        """
        Reconcile broker order status/fills with internal lifecycle tracker.

        Behavior:
        - If an internal non-terminal order is missing from broker open orders,
          fetch by ID and reconcile to terminal state when possible.
        - If order is not found at all, conservatively mark as canceled.
        - If broker reports partial/fill state that differs from lifecycle,
          update lifecycle to match broker reality.
        - If internal state is terminal but broker still reports open, emit
          mismatch warnings for operator investigation.
        """
        broker_orders = await self._fetch_open_orders()
        if broker_orders is None:
            self._runs_total += 1
            self._consecutive_mismatch_runs = 0
            self._halt_recommended = False
            self._last_halt_reason = None
            self._persist_health_snapshot(status="fetch_failed")
            return
        self._runs_total += 1
        self._last_error = None
        run_mismatches: list[dict[str, Any]] = []

        open_orders = {
            self._extract_order_id(order): order
            for order in broker_orders
            if self._extract_order_id(order)
        }

        for order_id, record in list(self.lifecycle_tracker._orders.items()):
            state = record.state
            open_order = open_orders.get(order_id)

            if state in self._TERMINAL_STATES:
                if open_order is not None:
                    logger.warning(
                        "Order %s is terminal internally (%s) but still open at broker",
                        order_id,
                        state.value,
                    )
                    self._record_mismatch(
                        run_mismatches,
                        order_id,
                        mismatch_type="terminal_but_open",
                        internal_state=state.value,
                        broker_status=self._extract_status(open_order),
                        broker_filled_qty=self._extract_float(open_order, "filled_qty"),
                    )
                continue

            if open_order is not None:
                self._reconcile_order_state(
                    order_id,
                    record,
                    open_order,
                    source="open_orders",
                    run_mismatches=run_mismatches,
                )
                continue

            broker_order = await self._fetch_order_by_id(order_id)
            if broker_order is None:
                logger.warning(
                    "Order %s missing from broker open orders and lookup; marking canceled",
                    order_id,
                )
                self.lifecycle_tracker.update_state(order_id, OrderState.CANCELED)
                self._record_mismatch(
                    run_mismatches,
                    order_id,
                    mismatch_type="missing_from_broker",
                    internal_state=state.value,
                )
                continue

            self._reconcile_order_state(
                order_id,
                record,
                broker_order,
                source="order_lookup",
                run_mismatches=run_mismatches,
            )

        self._update_health(run_mismatches)
        self._persist_health_snapshot(status="ok")

    async def _fetch_open_orders(self) -> Optional[list]:
        """Fetch current broker open orders."""
        try:
            return await self._call_broker(self.broker.get_orders)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Order reconciliation failed to fetch broker orders: {e}")
            return None

    async def _fetch_order_by_id(self, order_id: str) -> Optional[Any]:
        """Best-effort fetch of an order by ID across broker interfaces."""
        for method_name in ("get_order_by_id", "get_order"):
            if not self._has_real_method(self.broker, method_name):
                continue
            method = getattr(self.broker, method_name)
            try:
                result = await self._call_broker(method, order_id)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(
                    "Order reconciliation lookup failed via %s for %s: %s",
                    method_name,
                    order_id,
                    e,
                )
        return None

    @staticmethod
    async def _call_broker(method, *args, **kwargs):
        """Call broker methods that may be sync or async."""
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _has_real_method(obj: Any, method_name: str) -> bool:
        """
        Check whether a method is explicitly defined on the object.

        This avoids mock frameworks auto-creating arbitrary callables for
        undefined attributes.
        """
        instance_attrs = getattr(obj, "__dict__", {})
        if method_name in instance_attrs and callable(instance_attrs[method_name]):
            return True
        return callable(getattr(type(obj), method_name, None))

    @staticmethod
    def _extract_order_id(order: Any) -> Optional[str]:
        """Extract order ID from broker order object."""
        order_id = getattr(order, "id", None) or getattr(order, "order_id", None)
        if order_id is None:
            return None
        return str(order_id)

    @staticmethod
    def _extract_status(order: Any) -> Optional[str]:
        """Extract normalized broker status string."""
        status = getattr(order, "status", None)
        if status is None:
            return None
        value = getattr(status, "value", status)
        normalized = str(value).strip().lower()
        if "." in normalized:
            normalized = normalized.split(".")[-1]
        return normalized or None

    @staticmethod
    def _extract_float(order: Any, field: str) -> Optional[float]:
        """Safely parse float fields from broker order objects."""
        value = getattr(order, field, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _map_broker_state(self, order: Any) -> Optional[OrderState]:
        """Map broker-native status to internal lifecycle state."""
        status = self._extract_status(order)
        if not status:
            return None

        if status == "filled":
            return OrderState.FILLED
        if "partial" in status:
            return OrderState.PARTIAL
        if status in {
            "new",
            "accepted",
            "pending_new",
            "accepted_for_bidding",
            "pending_replace",
            "pending_cancel",
            "calculated",
        }:
            return OrderState.SUBMITTED
        if status in {"canceled", "cancelled", "done_for_day", "expired", "replaced"}:
            return OrderState.CANCELED
        if status in {"rejected", "suspended", "stopped"}:
            return OrderState.REJECTED

        return None

    def _reconcile_order_state(
        self,
        order_id: str,
        record,
        broker_order: Any,
        source: str,
        run_mismatches: list[dict[str, Any]],
    ) -> None:
        """Apply broker state/fill reconciliation for a single order."""
        prior_state = record.state
        expected_qty = float(record.quantity)
        filled_qty = self._extract_float(broker_order, "filled_qty")
        broker_status = self._extract_status(broker_order)
        mapped_state = self._map_broker_state(broker_order)
        target_state = None
        reason = None

        if filled_qty is not None and expected_qty > 0:
            if filled_qty >= expected_qty and prior_state != OrderState.FILLED:
                target_state = OrderState.FILLED
                reason = "filled_qty_complete"
            elif 0 < filled_qty < expected_qty and prior_state == OrderState.SUBMITTED:
                target_state = OrderState.PARTIAL
                reason = "filled_qty_partial"

        if target_state is None and mapped_state and mapped_state != prior_state:
            target_state = mapped_state
            reason = "status_mismatch"

        if target_state is None:
            return

        self.lifecycle_tracker.update_state(order_id, target_state)
        current_state = self.lifecycle_tracker.get_state(order_id)

        if current_state == target_state:
            logger.warning(
                "Reconciled order %s: %s -> %s via %s (%s)",
                order_id,
                prior_state.value,
                target_state.value,
                source,
                reason,
            )
            self._record_mismatch(
                run_mismatches,
                order_id,
                mismatch_type=reason or "state_mismatch",
                internal_state=prior_state.value,
                reconciled_state=target_state.value,
                broker_status=broker_status,
                broker_filled_qty=filled_qty,
                expected_qty=expected_qty,
                source=source,
            )
            return

        logger.warning(
            "Order %s reconciliation mismatch could not transition %s -> %s (broker status=%s)",
            order_id,
            prior_state.value,
            target_state.value,
            broker_status,
        )
        self._record_mismatch(
            run_mismatches,
            order_id,
            mismatch_type="transition_blocked",
            internal_state=prior_state.value,
            target_state=target_state.value,
            broker_status=broker_status,
            broker_filled_qty=filled_qty,
            expected_qty=expected_qty,
            source=source,
        )

    def _record_mismatch(
        self,
        run_mismatches: list[dict[str, Any]],
        order_id: str,
        mismatch_type: str,
        **data,
    ) -> None:
        """Track mismatch for health assessment and write audit trail event."""
        mismatch = {"order_id": order_id, "mismatch_type": mismatch_type}
        mismatch.update(data)
        run_mismatches.append(mismatch)
        self._emit_mismatch_event(order_id, mismatch_type, **data)

    def _update_health(self, run_mismatches: list[dict[str, Any]]) -> None:
        """Update reconciliation health metrics and halt recommendation."""
        mismatch_count = len(run_mismatches)
        self._last_run_mismatches = run_mismatches
        self._total_mismatches += mismatch_count

        if mismatch_count == 0:
            self._consecutive_mismatch_runs = 0
            self._halt_recommended = False
            self._last_halt_reason = None
            return

        self._consecutive_mismatch_runs += 1
        if self._consecutive_mismatch_runs >= self.mismatch_halt_threshold:
            self._halt_recommended = True
            self._last_halt_reason = (
                "Order reconciliation mismatch threshold breached: "
                f"{self._consecutive_mismatch_runs} consecutive runs "
                f"with mismatches (threshold={self.mismatch_halt_threshold})"
            )
            logger.error(self._last_halt_reason)

    def should_halt_trading(self) -> bool:
        """Whether reconciliation drift currently recommends halting entries."""
        return self._halt_recommended

    def get_health_snapshot(self) -> dict[str, Any]:
        """Return reconciliation health and recent mismatch context."""
        return {
            "runs_total": self._runs_total,
            "total_mismatches": self._total_mismatches,
            "consecutive_mismatch_runs": self._consecutive_mismatch_runs,
            "mismatch_halt_threshold": self.mismatch_halt_threshold,
            "halt_recommended": self._halt_recommended,
            "halt_reason": self._last_halt_reason,
            "last_run_mismatches": [m.copy() for m in self._last_run_mismatches],
            "last_error": self._last_error,
        }

    def close(self) -> None:
        if self._events_writer:
            self._events_writer.close()

    def _persist_health_snapshot(self, status: str) -> None:
        if not self._events_writer:
            return

        self._events_writer.write(
            {
                "event_type": "order_reconciliation_snapshot",
                "timestamp": datetime.utcnow().isoformat(),
                "run_id": self.run_id,
                "status": status,
                **self.get_health_snapshot(),
            }
        )

    def _emit_mismatch_event(self, order_id: str, mismatch_type: str, **data) -> None:
        """Emit reconciliation mismatch to immutable audit trail."""
        if not self.audit_log:
            return

        payload = {
            "type": "order_reconciliation",
            "mismatch_type": mismatch_type,
            "order_id": order_id,
        }
        payload.update(data)

        try:
            self.audit_log.log(AuditEventType.RISK_WARNING, payload)
        except Exception as e:
            logger.warning(f"Failed to write order reconciliation audit event for {order_id}: {e}")
