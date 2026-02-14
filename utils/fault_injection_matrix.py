"""
Deterministic broker/API fault injection matrix with SLO assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

from utils.incident_tracker import IncidentTracker
from utils.order_gateway import OrderGateway
from utils.order_lifecycle import OrderLifecycleTracker
from utils.order_reconciliation import OrderReconciler
from utils.slo_monitor import SLOMonitor


@dataclass
class FaultInjectionCheck:
    name: str
    passed: bool
    details: str
    expected_slo: List[str]
    observed_slo: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "expected_slo": self.expected_slo,
            "observed_slo": self.observed_slo,
        }


class _FaultOrderRequest:
    def __init__(self, symbol: str = "AAPL", qty: float = 10, side: str = "buy"):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.type = "market"


class _SubmitFailBroker:
    async def submit_order_advanced(self, order_request):
        raise TimeoutError("simulated submit timeout")

    async def get_positions(self):
        return []


class _ReconcileFailBroker:
    async def get_orders(self):
        raise RuntimeError("simulated broker order fetch failure")


async def _scenario_broker_submit_timeout() -> FaultInjectionCheck:
    broker = _SubmitFailBroker()
    gateway = OrderGateway(broker=broker, enforce_gateway=False)
    result = await gateway.submit_order(_FaultOrderRequest(), strategy_name="Matrix")
    observed = []
    if not result.success:
        observed.append("order_submit_rejected")
    passed = (not result.success) and "Broker rejected order" in str(result.rejection_reason or "")
    return FaultInjectionCheck(
        name="broker_submit_timeout_rejected",
        passed=passed,
        details=str(result.rejection_reason or ""),
        expected_slo=["order_submit_rejected"],
        observed_slo=observed,
    )


async def _scenario_data_quality_critical_slo(tmp_dir: str) -> FaultInjectionCheck:
    incident_tracker = IncidentTracker(
        events_path=f"{tmp_dir}/incident_events.jsonl",
        run_id="matrix",
        ack_sla_minutes=15,
    )
    monitor = SLOMonitor(
        incident_tracker=incident_tracker,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )
    breaches = monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    observed = [b.name for b in breaches]
    status = monitor.get_status_snapshot()
    monitor.close()
    passed = (
        any(name == "data_quality_errors" for name in observed)
        and status.get("incidents", {}).get("total_incidents", 0) >= 1
    )
    return FaultInjectionCheck(
        name="data_quality_breach_opens_incident",
        passed=passed,
        details=f"observed={observed} incidents={status.get('incidents', {})}",
        expected_slo=["data_quality_errors"],
        observed_slo=observed,
    )


async def _scenario_incident_ack_timeout(tmp_dir: str) -> FaultInjectionCheck:
    incident_tracker = IncidentTracker(
        events_path=f"{tmp_dir}/incident_events_ack.jsonl",
        run_id="matrix",
        ack_sla_minutes=1,
    )
    monitor = SLOMonitor(
        incident_tracker=incident_tracker,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )
    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    incident_id = next(iter(incident_tracker._incidents.keys()))
    created_at = incident_tracker._incidents[incident_id].created_at
    breaches = monitor.check_incident_ack_sla(now=created_at + timedelta(minutes=2))
    observed = [b.name for b in breaches]
    monitor.close()
    passed = any(name == "incident_ack_sla_breach" for name in observed)
    return FaultInjectionCheck(
        name="incident_ack_timeout_critical_slo",
        passed=passed,
        details=f"observed={observed}",
        expected_slo=["incident_ack_sla_breach"],
        observed_slo=observed,
    )


async def _scenario_shadow_drift_critical() -> FaultInjectionCheck:
    monitor = SLOMonitor(
        shadow_drift_warning_threshold=0.10,
        shadow_drift_critical_threshold=0.15,
    )
    breaches = monitor.record_shadow_drift_summary({"paper_live_shadow_drift": 0.21})
    observed = [b.name for b in breaches]
    monitor.close()
    passed = any(name == "paper_live_shadow_drift" for name in observed)
    return FaultInjectionCheck(
        name="shadow_drift_critical_threshold",
        passed=passed,
        details=f"observed={observed}",
        expected_slo=["paper_live_shadow_drift"],
        observed_slo=observed,
    )


async def _scenario_reconciliation_fetch_failure_health() -> FaultInjectionCheck:
    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-matrix-1", "AAPL", "buy", 5, "Matrix")
    reconciler = OrderReconciler(
        broker=_ReconcileFailBroker(),
        lifecycle_tracker=tracker,
        mismatch_halt_threshold=2,
    )
    await reconciler.reconcile()
    health = reconciler.get_health_snapshot()
    observed = ["order_reconciliation_fetch_error"] if health.get("last_error") else []
    passed = bool(health.get("last_error"))
    return FaultInjectionCheck(
        name="order_reconciliation_fetch_failure_health_signal",
        passed=passed,
        details=str(health.get("last_error")),
        expected_slo=["order_reconciliation_fetch_error"],
        observed_slo=observed,
    )


async def run_fault_injection_matrix(tmp_dir: str = "results") -> Dict[str, Any]:
    """
    Run fault-injection scenarios and assert expected SLO outcomes.
    """
    checks: List[FaultInjectionCheck] = []
    checks.append(await _scenario_broker_submit_timeout())
    checks.append(await _scenario_data_quality_critical_slo(tmp_dir))
    checks.append(await _scenario_incident_ack_timeout(tmp_dir))
    checks.append(await _scenario_shadow_drift_critical())
    checks.append(await _scenario_reconciliation_fetch_failure_health())
    passed = all(check.passed for check in checks)
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "passed": passed,
        "checks": [check.to_dict() for check in checks],
    }


def format_fault_injection_report(report: Dict[str, Any]) -> str:
    status = "PASS" if report.get("passed") else "FAIL"
    lines = [
        "=" * 72,
        f"FAULT INJECTION MATRIX | {status}",
        "=" * 72,
    ]
    for check in report.get("checks", []):
        label = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"[{label}] {check.get('name')}")
        lines.append(f"  expected={check.get('expected_slo')} observed={check.get('observed_slo')}")
        lines.append(f"  {check.get('details')}")
    lines.append("=" * 72)
    return "\n".join(lines)
