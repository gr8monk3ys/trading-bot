"""
Incident acknowledgment tracking for critical operational alerts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from utils.run_artifacts import JsonlWriter, read_jsonl


@dataclass
class Incident:
    incident_id: str
    created_at: datetime
    severity: str
    name: str
    message: str
    context: Dict[str, Any]
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledgment_notes: str = ""
    sla_breached_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "created_at": self.created_at.isoformat(),
            "severity": self.severity,
            "name": self.name,
            "message": self.message,
            "context": self.context,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "acknowledgment_notes": self.acknowledgment_notes,
            "sla_breached_at": self.sla_breached_at.isoformat() if self.sla_breached_at else None,
        }


class IncidentTracker:
    """
    Tracks critical incidents and enforces acknowledgment SLA.

    Uses an append-only JSONL stream so acknowledgments can be written from
    external processes (for example, a CLI or operational automation task).
    """

    def __init__(
        self,
        *,
        events_path: str | Path | None = None,
        run_id: str | None = None,
        ack_sla_minutes: int = 15,
    ):
        self.events_path = Path(events_path) if events_path else None
        self.run_id = run_id
        self.ack_sla_minutes = max(1, int(ack_sla_minutes))
        self._writer = JsonlWriter(self.events_path) if self.events_path else None
        self._incidents: dict[str, Incident] = {}

    def close(self) -> None:
        if self._writer:
            self._writer.close()

    def _sync_from_events(self) -> None:
        if not self.events_path or not self.events_path.exists():
            return

        incidents: dict[str, Incident] = {}
        for event in read_jsonl(self.events_path):
            event_type = str(event.get("event_type", "")).strip().lower()
            if event_type == "incident_open":
                incident_id = str(event.get("incident_id", "")).strip()
                if not incident_id:
                    continue
                created_at = self._parse_dt(event.get("created_at")) or datetime.utcnow()
                incidents[incident_id] = Incident(
                    incident_id=incident_id,
                    created_at=created_at,
                    severity=str(event.get("severity", "critical")),
                    name=str(event.get("name", "unknown_incident")),
                    message=str(event.get("message", "")),
                    context=event.get("context", {}) or {},
                )
            elif event_type == "incident_ack":
                incident_id = str(event.get("incident_id", "")).strip()
                incident = incidents.get(incident_id)
                if incident is None:
                    continue
                incident.acknowledged_at = self._parse_dt(event.get("acknowledged_at"))
                incident.acknowledged_by = str(event.get("acknowledged_by", "")).strip() or None
                incident.acknowledgment_notes = str(event.get("acknowledgment_notes", "") or "")
            elif event_type == "incident_sla_breach":
                incident_id = str(event.get("incident_id", "")).strip()
                incident = incidents.get(incident_id)
                if incident is None:
                    continue
                incident.sla_breached_at = (
                    self._parse_dt(event.get("breached_at")) or datetime.utcnow()
                )

        self._incidents = incidents

    @staticmethod
    def _parse_dt(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def open_incident(self, breach: Dict[str, Any]) -> Dict[str, Any]:
        """Create and persist a new incident from a critical breach payload."""
        self._sync_from_events()
        incident_id = f"inc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        incident = Incident(
            incident_id=incident_id,
            created_at=datetime.utcnow(),
            severity=str(breach.get("severity", "critical")),
            name=str(breach.get("name", "unknown_incident")),
            message=str(breach.get("message", "")),
            context=dict(breach.get("context", {}) or {}),
        )
        self._incidents[incident_id] = incident

        if self._writer:
            self._writer.write(
                {
                    "event_type": "incident_open",
                    "run_id": self.run_id,
                    **incident.to_dict(),
                }
            )

        return incident.to_dict()

    def acknowledge(self, incident_id: str, acknowledged_by: str, notes: str = "") -> bool:
        """Acknowledge an incident and persist acknowledgment event."""
        self._sync_from_events()
        incident = self._incidents.get(incident_id)
        if incident is None:
            return False
        if incident.acknowledged_at is not None:
            return True

        incident.acknowledged_at = datetime.utcnow()
        incident.acknowledged_by = acknowledged_by.strip() or "unknown"
        incident.acknowledgment_notes = notes or ""

        if self._writer:
            self._writer.write(
                {
                    "event_type": "incident_ack",
                    "run_id": self.run_id,
                    "incident_id": incident_id,
                    "acknowledged_at": incident.acknowledged_at.isoformat(),
                    "acknowledged_by": incident.acknowledged_by,
                    "acknowledgment_notes": incident.acknowledgment_notes,
                }
            )

        return True

    def evaluate_ack_sla(self, now: Optional[datetime] = None) -> list[Dict[str, Any]]:
        """
        Return newly breached incidents where acknowledgment SLA has elapsed.
        """
        self._sync_from_events()
        ts = now or datetime.utcnow()
        breached: list[Dict[str, Any]] = []

        for incident in self._incidents.values():
            if incident.acknowledged_at is not None:
                continue
            age_minutes = (ts - incident.created_at).total_seconds() / 60.0
            if age_minutes < self.ack_sla_minutes:
                continue
            if incident.sla_breached_at is not None:
                continue

            incident.sla_breached_at = ts
            payload = {
                "incident_id": incident.incident_id,
                "name": incident.name,
                "severity": incident.severity,
                "age_minutes": age_minutes,
                "ack_sla_minutes": self.ack_sla_minutes,
                "created_at": incident.created_at.isoformat(),
                "message": incident.message,
                "context": incident.context,
            }
            breached.append(payload)

            if self._writer:
                self._writer.write(
                    {
                        "event_type": "incident_sla_breach",
                        "run_id": self.run_id,
                        "incident_id": incident.incident_id,
                        "breached_at": ts.isoformat(),
                        "age_minutes": age_minutes,
                        "ack_sla_minutes": self.ack_sla_minutes,
                    }
                )

        return breached

    def get_status_snapshot(self) -> Dict[str, Any]:
        self._sync_from_events()
        incidents = list(self._incidents.values())
        total = len(incidents)
        acknowledged = sum(1 for i in incidents if i.acknowledged_at is not None)
        unacknowledged = total - acknowledged
        sla_breaches = sum(1 for i in incidents if i.sla_breached_at is not None)
        oldest_unack = None
        for incident in incidents:
            if incident.acknowledged_at is not None:
                continue
            if oldest_unack is None or incident.created_at < oldest_unack:
                oldest_unack = incident.created_at

        return {
            "total_incidents": total,
            "acknowledged_incidents": acknowledged,
            "unacknowledged_incidents": unacknowledged,
            "sla_breaches": sla_breaches,
            "ack_sla_minutes": self.ack_sla_minutes,
            "oldest_unacknowledged_at": oldest_unack.isoformat() if oldest_unack else None,
        }
