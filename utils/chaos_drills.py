"""
Deterministic chaos-drill scenarios for operational resilience validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

from utils.data_quality import should_halt_trading_for_data_quality
from utils.order_lifecycle import OrderLifecycleTracker
from utils.order_gateway import OrderGateway
from utils.order_reconciliation import OrderReconciler
from utils.partial_fill_tracker import PartialFillTracker
from utils.slo_monitor import SLOMonitor
from utils.websocket_manager import WebSocketManager


@dataclass
class ChaosDrillCheck:
    name: str
    passed: bool
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }


class _FailingBroker:
    async def get_orders(self):
        raise RuntimeError("chaos: broker order fetch failed")


class _FailingNotifier:
    def notify(self, breach: Dict[str, Any]):
        raise RuntimeError("chaos: pager endpoint failed")


class _AlwaysFailStream:
    def subscribe_bars(self, *args, **kwargs):
        return None

    def subscribe_quotes(self, *args, **kwargs):
        return None

    def subscribe_trades(self, *args, **kwargs):
        return None

    def run(self):
        raise RuntimeError("chaos: websocket disconnect storm")


class _ChaosOrderRequest:
    def __init__(
        self,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
        order_type: str = "market",
    ):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.client_order_id = client_order_id
        self.type = order_type


class _CrashRecoveryBroker:
    def __init__(self):
        self.submit_calls = 0
        self.orders_by_client_id: Dict[str, str] = {}

    async def submit_order_advanced(self, order_request):
        self.submit_calls += 1
        order_id = f"ord-chaos-{self.submit_calls}"
        client_order_id = str(getattr(order_request, "client_order_id", "") or "").strip()
        if client_order_id:
            self.orders_by_client_id[client_order_id] = order_id

        class _MockOrder:
            def __init__(self, oid: str):
                self.id = oid
                self.filled_avg_price = "100.0"

        return _MockOrder(order_id)

    async def get_order_by_client_id(self, client_order_id: str):
        order_id = self.orders_by_client_id.get(client_order_id)
        if not order_id:
            return None

        class _ExistingOrder:
            def __init__(self, oid: str):
                self.id = oid

        return _ExistingOrder(order_id)


async def _drill_order_reconciliation_fetch_failure() -> ChaosDrillCheck:
    lifecycle = OrderLifecycleTracker()
    lifecycle.register_order("ord-chaos-1", "AAPL", "buy", 10, "ChaosDrill")
    reconciler = OrderReconciler(
        broker=_FailingBroker(),
        lifecycle_tracker=lifecycle,
        mismatch_halt_threshold=2,
    )
    await reconciler.reconcile()
    health = reconciler.get_health_snapshot()
    passed = (
        health.get("runs_total") == 1
        and isinstance(health.get("last_error"), str)
        and "chaos: broker order fetch failed" in str(health.get("last_error"))
    )
    details = (
        "Reconciler retained health snapshot after broker failure"
        if passed
        else f"Unexpected reconciliation health snapshot: {health}"
    )
    return ChaosDrillCheck(
        name="order_reconciliation_fetch_failure",
        passed=passed,
        details=details,
    )


def _drill_data_quality_halt() -> ChaosDrillCheck:
    should_halt, reason = should_halt_trading_for_data_quality(
        {"total_errors": 2, "stale_warnings": 0},
        max_errors=0,
        max_stale_warnings=0,
    )
    passed = should_halt and isinstance(reason, str) and "Data quality errors" in reason
    details = reason or "No halt reason returned"
    return ChaosDrillCheck(
        name="data_quality_halt_on_critical_errors",
        passed=passed,
        details=details,
    )


def _drill_alert_path_failure_tolerance() -> ChaosDrillCheck:
    monitor = SLOMonitor(
        alert_notifier=_FailingNotifier(),
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )
    breaches = monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    status = monitor.get_status_snapshot()
    monitor.close()

    passed = (
        monitor.has_critical_breach(breaches)
        and status.get("alerting", {}).get("attempts", 0) >= 1
        and status.get("alerting", {}).get("failures", 0) >= 1
    )
    details = (
        "SLO monitor continued after notifier failure"
        if passed
        else f"Unexpected alerting snapshot: {status.get('alerting')}"
    )
    return ChaosDrillCheck(
        name="slo_alert_failure_tolerance",
        passed=passed,
        details=details,
    )


async def _drill_websocket_reconnect_storm_tolerance() -> ChaosDrillCheck:
    manager = WebSocketManager(
        api_key="chaos_key",
        secret_key="chaos_secret",
        feed="iex",
    )
    manager._max_reconnect_attempts = 3
    manager._base_reconnect_delay = 0
    manager._max_reconnect_delay = 0
    manager._create_stream = lambda: _AlwaysFailStream()  # type: ignore[assignment]
    manager._running = True
    manager.subscribe_bars(["AAPL"])

    # Should exit after max reconnection attempts with no unhandled exception.
    await manager._run_stream()
    stats = manager.get_connection_stats()
    passed = (
        stats.get("is_running") is False
        and stats.get("is_connected") is False
        and int(stats.get("reconnect_attempts", 0) or 0) > manager._max_reconnect_attempts
    )
    details = (
        "WebSocket manager capped reconnect attempts and exited cleanly"
        if passed
        else f"Unexpected reconnect storm stats: {stats}"
    )
    return ChaosDrillCheck(
        name="websocket_reconnect_storm_tolerance",
        passed=passed,
        details=details,
    )


async def _drill_partial_fill_stall_detection() -> ChaosDrillCheck:
    tracker = PartialFillTracker()
    tracker.track_order("order-stall-1", "AAPL", "buy", 100)
    await tracker.record_fill(
        order_id="order-stall-1",
        filled_qty=40,
        fill_price=100.0,
        is_final=False,
    )
    # Simulate stalled fill updates.
    tracker._orders["order-stall-1"].updated_at = datetime.now().replace(microsecond=0)
    tracker._orders["order-stall-1"].updated_at = (
        tracker._orders["order-stall-1"].updated_at
        - timedelta(seconds=600)
    )
    stalled = tracker.detect_stalled_orders(max_stall_seconds=60)
    passed = len(stalled) == 1 and stalled[0].get("order_id") == "order-stall-1"
    details = (
        f"Detected {len(stalled)} stalled order(s) with threshold=60s"
        if passed
        else f"Stalled orders not detected as expected: {stalled}"
    )
    return ChaosDrillCheck(
        name="partial_fill_stall_detection",
        passed=passed,
        details=details,
    )


async def _drill_crash_recovery_idempotent_replay() -> ChaosDrillCheck:
    broker = _CrashRecoveryBroker()
    request = _ChaosOrderRequest(
        symbol="AAPL",
        qty=10,
        side="buy",
        client_order_id="chaos-restart-001",
    )

    first_gateway = OrderGateway(broker=broker, enforce_gateway=False)
    first = await first_gateway.submit_order(request, strategy_name="ChaosDrill")
    lifecycle_state = first_gateway.lifecycle_tracker.export_state()

    # Simulate process crash + restart by booting a new gateway and restoring state.
    second_gateway = OrderGateway(broker=broker, enforce_gateway=False)
    second_gateway.lifecycle_tracker.import_state(lifecycle_state)
    second = await second_gateway.submit_order(request, strategy_name="ChaosDrill")
    stats = second_gateway.get_statistics()

    passed = (
        first.success
        and second.success
        and first.order_id == second.order_id
        and broker.submit_calls == 1
        and int(stats.get("duplicate_orders_suppressed", 0) or 0) == 1
    )
    details = (
        "Restart replay suppressed duplicate submission and preserved order identity"
        if passed
        else (
            "Unexpected crash-recovery replay state: "
            f"first={first.order_id}/{first.success}, "
            f"second={second.order_id}/{second.success}, "
            f"submit_calls={broker.submit_calls}, stats={stats}"
        )
    )
    return ChaosDrillCheck(
        name="crash_recovery_idempotent_replay",
        passed=passed,
        details=details,
    )


async def run_chaos_drills() -> Dict[str, Any]:
    """Run built-in chaos drills and return machine-readable report."""
    checks: List[ChaosDrillCheck] = []
    checks.append(await _drill_order_reconciliation_fetch_failure())
    checks.append(_drill_data_quality_halt())
    checks.append(_drill_alert_path_failure_tolerance())
    checks.append(await _drill_websocket_reconnect_storm_tolerance())
    checks.append(await _drill_partial_fill_stall_detection())
    checks.append(await _drill_crash_recovery_idempotent_replay())

    passed = all(check.passed for check in checks)
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "passed": passed,
        "checks": [check.to_dict() for check in checks],
    }


def format_chaos_drill_report(report: Dict[str, Any]) -> str:
    """Render chaos-drill report for CLI output."""
    status = "PASS" if report.get("passed") else "FAIL"
    lines = [
        "=" * 72,
        f"CHAOS DRILL REPORT | {status}",
        "=" * 72,
    ]
    for check in report.get("checks", []):
        label = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"[{label}] {check.get('name')}")
        lines.append(f"  {check.get('details')}")
    lines.append("=" * 72)
    return "\n".join(lines)
