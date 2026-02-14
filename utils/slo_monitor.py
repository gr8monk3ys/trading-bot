"""
Operational SLO monitoring and alert emission for live/paper trading sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.audit_log import AuditEventType, AuditLog
from utils.execution_quality_gate import extract_paper_live_shadow_drift
from utils.incident_tracker import IncidentTracker
from utils.run_artifacts import JsonlWriter

logger = logging.getLogger(__name__)


@dataclass
class SLOBreach:
    """Represents one SLO breach event."""

    name: str
    severity: str  # "warning" | "critical"
    message: str
    value: Any = None
    threshold: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


class SLOMonitor:
    """
    Tracks operational SLOs and emits warnings/critical alerts.

    This monitor is intentionally lightweight and side-effect free except for:
    - structured logging
    - optional audit trail entries
    - optional JSONL event emission for replay/incident analysis
    """

    def __init__(
        self,
        *,
        audit_log: Optional[AuditLog] = None,
        events_path: str | Path | None = None,
        alert_notifier: Any | None = None,
        incident_tracker: IncidentTracker | None = None,
        recon_mismatch_halt_runs: int = 3,
        max_data_quality_errors: int = 0,
        max_stale_data_warnings: int = 0,
        shadow_drift_warning_threshold: float = 0.12,
        shadow_drift_critical_threshold: float = 0.15,
    ):
        self.audit_log = audit_log
        self.alert_notifier = alert_notifier
        self.incident_tracker = incident_tracker
        self.recon_mismatch_halt_runs = max(1, int(recon_mismatch_halt_runs))
        self.max_data_quality_errors = max(0, int(max_data_quality_errors))
        self.max_stale_data_warnings = max(0, int(max_stale_data_warnings))
        self.shadow_drift_warning_threshold = max(
            0.0,
            float(shadow_drift_warning_threshold),
        )
        self.shadow_drift_critical_threshold = max(
            self.shadow_drift_warning_threshold,
            float(shadow_drift_critical_threshold),
        )
        self._events_writer = JsonlWriter(events_path) if events_path else None
        self._breach_history: List[SLOBreach] = []
        self._alert_attempts = 0
        self._alert_sent = 0
        self._alert_failures = 0

    def close(self) -> None:
        if self._events_writer:
            self._events_writer.close()
        if self.incident_tracker:
            self.incident_tracker.close()

    def record_order_reconciliation_health(self, health: Dict[str, Any]) -> List[SLOBreach]:
        """
        Evaluate order reconciliation health snapshot and emit SLO breaches.
        """
        breaches: List[SLOBreach] = []
        mismatch_runs = int(health.get("consecutive_mismatch_runs", 0) or 0)
        mismatch_total = int(health.get("total_mismatches", 0) or 0)

        if mismatch_total > 0:
            breaches.append(
                SLOBreach(
                    name="order_reconciliation_mismatch_total",
                    severity="warning",
                    message=f"Order reconciliation has {mismatch_total} total mismatches",
                    value=mismatch_total,
                    threshold="0",
                    context={"health": health},
                )
            )

        if mismatch_runs >= self.recon_mismatch_halt_runs:
            breaches.append(
                SLOBreach(
                    name="order_reconciliation_consecutive_mismatch_runs",
                    severity="critical",
                    message=(
                        "Order reconciliation mismatch runs breached threshold: "
                        f"{mismatch_runs} >= {self.recon_mismatch_halt_runs}"
                    ),
                    value=mismatch_runs,
                    threshold=self.recon_mismatch_halt_runs,
                    context={"health": health},
                )
            )

        self._emit(breaches)
        return breaches

    def record_data_quality_summary(self, summary: Dict[str, Any]) -> List[SLOBreach]:
        """
        Evaluate data quality summary and emit SLO breaches.
        """
        breaches: List[SLOBreach] = []
        error_count = int(summary.get("total_errors", 0) or 0)
        stale_warnings = int(summary.get("stale_warnings", 0) or 0)

        if error_count > self.max_data_quality_errors:
            breaches.append(
                SLOBreach(
                    name="data_quality_errors",
                    severity="critical",
                    message=(
                        "Data quality error threshold breached: "
                        f"{error_count} > {self.max_data_quality_errors}"
                    ),
                    value=error_count,
                    threshold=self.max_data_quality_errors,
                    context={"summary": summary},
                )
            )

        if stale_warnings > self.max_stale_data_warnings:
            breaches.append(
                SLOBreach(
                    name="data_quality_stale_warnings",
                    severity="critical",
                    message=(
                        "Data staleness threshold breached: "
                        f"{stale_warnings} > {self.max_stale_data_warnings}"
                    ),
                    value=stale_warnings,
                    threshold=self.max_stale_data_warnings,
                    context={"summary": summary},
                )
            )

        self._emit(breaches)
        return breaches

    def get_status_snapshot(self) -> Dict[str, Any]:
        """
        Return monitor status and recent breach summary.
        """
        critical = sum(1 for b in self._breach_history if b.severity == "critical")
        warnings = sum(1 for b in self._breach_history if b.severity == "warning")
        last = self._breach_history[-1].to_dict() if self._breach_history else None

        return {
            "breaches_total": len(self._breach_history),
            "critical_breaches": critical,
            "warning_breaches": warnings,
            "last_breach": last,
            "alerting": {
                "attempts": self._alert_attempts,
                "sent": self._alert_sent,
                "failures": self._alert_failures,
            },
            "incidents": (
                self.incident_tracker.get_status_snapshot() if self.incident_tracker else {}
            ),
            "thresholds": {
                "recon_mismatch_halt_runs": self.recon_mismatch_halt_runs,
                "max_data_quality_errors": self.max_data_quality_errors,
                "max_stale_data_warnings": self.max_stale_data_warnings,
                "shadow_drift_warning_threshold": self.shadow_drift_warning_threshold,
                "shadow_drift_critical_threshold": self.shadow_drift_critical_threshold,
            },
        }

    def record_shadow_drift_summary(self, summary: Dict[str, Any]) -> List[SLOBreach]:
        """
        Evaluate paper/live shadow drift and emit SLO warnings or critical breaches.
        """
        breaches: List[SLOBreach] = []
        drift = extract_paper_live_shadow_drift(summary)
        if drift is None:
            return breaches

        warning_threshold = self.shadow_drift_warning_threshold
        critical_threshold = self.shadow_drift_critical_threshold
        if drift >= critical_threshold:
            breaches.append(
                SLOBreach(
                    name="paper_live_shadow_drift",
                    severity="critical",
                    message=(
                        "Paper/live shadow drift breached critical threshold: "
                        f"{drift:.3f} >= {critical_threshold:.3f}"
                    ),
                    value=drift,
                    threshold=critical_threshold,
                    context={"summary": summary},
                )
            )
        elif drift >= warning_threshold:
            breaches.append(
                SLOBreach(
                    name="paper_live_shadow_drift_warning",
                    severity="warning",
                    message=(
                        "Paper/live shadow drift breached warning threshold: "
                        f"{drift:.3f} >= {warning_threshold:.3f}"
                    ),
                    value=drift,
                    threshold=warning_threshold,
                    context={"summary": summary},
                )
            )

        self._emit(breaches)
        return breaches

    def acknowledge_incident(self, incident_id: str, acknowledged_by: str, notes: str = "") -> bool:
        """Acknowledge a tracked incident."""
        if not self.incident_tracker:
            return False
        return self.incident_tracker.acknowledge(incident_id, acknowledged_by, notes)

    def check_incident_ack_sla(self, now: Optional[datetime] = None) -> List[SLOBreach]:
        """
        Emit critical breaches when incident acknowledgment SLA is exceeded.
        """
        if not self.incident_tracker:
            return []

        overdue = self.incident_tracker.evaluate_ack_sla(now=now)
        breaches: List[SLOBreach] = []
        for incident in overdue:
            age = float(incident.get("age_minutes", 0.0) or 0.0)
            threshold = float(incident.get("ack_sla_minutes", 0.0) or 0.0)
            incident_id = str(incident.get("incident_id", "unknown"))
            breaches.append(
                SLOBreach(
                    name="incident_ack_sla_breach",
                    severity="critical",
                    message=(
                        "Incident acknowledgment SLA breached "
                        f"(incident={incident_id}, age={age:.1f}m, sla={threshold:.1f}m)"
                    ),
                    value=age,
                    threshold=threshold,
                    context={"incident": incident},
                )
            )

        self._emit(breaches)
        return breaches

    @staticmethod
    def has_critical_breach(breaches: List[SLOBreach]) -> bool:
        return any(b.severity == "critical" for b in breaches)

    def _emit(self, breaches: List[SLOBreach]) -> None:
        for breach in breaches:
            self._breach_history.append(breach)
            breach_payload = breach.to_dict()

            if (
                self.incident_tracker
                and breach.severity == "critical"
                and breach.name != "incident_ack_sla_breach"
            ):
                try:
                    incident = self.incident_tracker.open_incident(breach_payload)
                    context = dict(breach_payload.get("context", {}) or {})
                    context["incident_id"] = incident.get("incident_id")
                    breach_payload["context"] = context
                except Exception as e:
                    logger.warning(f"Failed to open incident from SLO breach: {e}")

            if breach.severity == "critical":
                logger.error("SLO BREACH [%s] %s", breach.name, breach.message)
            else:
                logger.warning("SLO WARNING [%s] %s", breach.name, breach.message)

            if self.audit_log:
                try:
                    self.audit_log.log(
                        AuditEventType.RISK_WARNING,
                        {
                            "type": "slo_breach",
                            **breach_payload,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to write SLO breach audit event: {e}")

            if self._events_writer:
                self._events_writer.write(
                    {
                        "event_type": "slo_breach",
                        **breach_payload,
                    }
                )

            if self.alert_notifier:
                try:
                    result = self.alert_notifier.notify(breach_payload)
                    if result is not None:
                        self._alert_attempts += 1
                        if result:
                            self._alert_sent += 1
                        else:
                            self._alert_failures += 1
                except Exception as e:
                    self._alert_attempts += 1
                    self._alert_failures += 1
                    logger.warning(f"Failed to send SLO breach alert: {e}")
