#!/usr/bin/env python3
"""
Unit tests for operational SLO monitor.
"""

from datetime import timedelta

from utils.incident_tracker import IncidentTracker
from utils.run_artifacts import read_jsonl
from utils.slo_monitor import SLOMonitor


def test_order_reconciliation_health_critical_breach(tmp_path):
    monitor = SLOMonitor(
        events_path=tmp_path / "slo_events.jsonl",
        recon_mismatch_halt_runs=2,
    )
    breaches = monitor.record_order_reconciliation_health(
        {
            "consecutive_mismatch_runs": 2,
            "total_mismatches": 5,
        }
    )
    assert any(b.severity == "critical" for b in breaches)
    assert monitor.has_critical_breach(breaches) is True
    monitor.close()


def test_data_quality_health_critical_breach(tmp_path):
    monitor = SLOMonitor(
        events_path=tmp_path / "slo_events.jsonl",
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )
    breaches = monitor.record_data_quality_summary(
        {
            "total_errors": 1,
            "stale_warnings": 2,
        }
    )
    assert any(b.name == "data_quality_errors" for b in breaches)
    assert any(b.name == "data_quality_stale_warnings" for b in breaches)
    monitor.close()


def test_slo_monitor_sends_alerts_for_critical_breaches():
    class RecordingNotifier:
        def __init__(self):
            self.events = []

        def notify(self, breach):
            self.events.append(breach)
            return True

    notifier = RecordingNotifier()
    monitor = SLOMonitor(
        alert_notifier=notifier,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )

    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    status = monitor.get_status_snapshot()

    assert len(notifier.events) == 1
    assert status["alerting"]["attempts"] == 1
    assert status["alerting"]["sent"] == 1
    assert status["alerting"]["failures"] == 0
    monitor.close()


def test_slo_monitor_tolerates_alert_delivery_failures():
    class FailingNotifier:
        def notify(self, breach):
            raise RuntimeError("downstream pager unavailable")

    monitor = SLOMonitor(
        alert_notifier=FailingNotifier(),
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )

    breaches = monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    status = monitor.get_status_snapshot()

    assert monitor.has_critical_breach(breaches) is True
    assert status["alerting"]["attempts"] == 1
    assert status["alerting"]["failures"] == 1
    monitor.close()


def test_slo_monitor_records_and_acknowledges_incidents(tmp_path):
    incident_tracker = IncidentTracker(
        events_path=tmp_path / "incident_events.jsonl",
        run_id="run_test",
        ack_sla_minutes=15,
    )
    monitor = SLOMonitor(
        incident_tracker=incident_tracker,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )

    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    status = monitor.get_status_snapshot()
    assert status["incidents"]["total_incidents"] == 1
    assert status["incidents"]["unacknowledged_incidents"] == 1

    incident_id = next(iter(incident_tracker._incidents.keys()))
    assert monitor.acknowledge_incident(incident_id, "oncall", "investigating")
    status_after_ack = monitor.get_status_snapshot()
    assert status_after_ack["incidents"]["acknowledged_incidents"] == 1
    monitor.close()


def test_slo_monitor_emits_incident_ack_sla_breach(tmp_path, monkeypatch):
    incident_tracker = IncidentTracker(
        events_path=tmp_path / "incident_events.jsonl",
        run_id="run_test",
        ack_sla_minutes=1,
    )
    monitor = SLOMonitor(
        incident_tracker=incident_tracker,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )
    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    incident_id = next(iter(incident_tracker._incidents.keys()))
    monkeypatch.setattr(
        incident_tracker,
        "evaluate_ack_sla",
        lambda now=None: [
            {
                "incident_id": incident_id,
                "name": "data_quality_errors",
                "severity": "critical",
                "age_minutes": 20.0,
                "ack_sla_minutes": 1.0,
                "created_at": "2026-01-01T00:00:00",
                "message": "Data quality threshold breached",
                "context": {},
            }
        ],
    )
    breaches = monitor.check_incident_ack_sla()
    assert any(b.name == "incident_ack_sla_breach" for b in breaches)
    monitor.close()


def test_slo_monitor_incident_lifecycle_e2e_with_ack_clears_sla(tmp_path):
    incident_events_path = tmp_path / "incident_events.jsonl"
    slo_events_path = tmp_path / "slo_events.jsonl"
    incident_tracker = IncidentTracker(
        events_path=incident_events_path,
        run_id="run_e2e",
        ack_sla_minutes=1,
    )
    monitor = SLOMonitor(
        events_path=slo_events_path,
        incident_tracker=incident_tracker,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )

    # 1) Critical breach opens an incident.
    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    status_open = monitor.get_status_snapshot()
    assert status_open["incidents"]["total_incidents"] == 1
    assert status_open["incidents"]["unacknowledged_incidents"] == 1

    incident_id = next(iter(incident_tracker._incidents.keys()))
    created_at = incident_tracker._incidents[incident_id].created_at

    # 2) SLA check at deterministic future timestamp emits breach.
    overdue_breaches = monitor.check_incident_ack_sla(
        now=created_at + timedelta(minutes=2),
    )
    assert len(overdue_breaches) == 1
    assert overdue_breaches[0].name == "incident_ack_sla_breach"
    status_overdue = monitor.get_status_snapshot()
    assert status_overdue["incidents"]["sla_breaches"] == 1

    # 3) Acknowledge the incident and verify no further SLA breaches.
    assert monitor.acknowledge_incident(incident_id, "oncall", "mitigating")
    status_acked = monitor.get_status_snapshot()
    assert status_acked["incidents"]["acknowledged_incidents"] == 1
    assert status_acked["incidents"]["unacknowledged_incidents"] == 0
    assert status_acked["incidents"]["oldest_unacknowledged_at"] is None

    post_ack_breaches = monitor.check_incident_ack_sla(
        now=created_at + timedelta(minutes=5),
    )
    assert post_ack_breaches == []

    incident_events = read_jsonl(incident_events_path)
    incident_event_types = [event.get("event_type") for event in incident_events]
    assert incident_event_types.count("incident_open") == 1
    assert incident_event_types.count("incident_sla_breach") == 1
    assert incident_event_types.count("incident_ack") == 1

    slo_events = read_jsonl(slo_events_path)
    slo_names = [
        event.get("name") for event in slo_events if event.get("event_type") == "slo_breach"
    ]
    assert "data_quality_errors" in slo_names
    assert "incident_ack_sla_breach" in slo_names

    monitor.close()


def test_slo_monitor_shadow_drift_warning_breach():
    monitor = SLOMonitor(
        shadow_drift_warning_threshold=0.10,
        shadow_drift_critical_threshold=0.15,
    )
    breaches = monitor.record_shadow_drift_summary({"paper_live_shadow_drift": 0.12})
    assert len(breaches) == 1
    assert breaches[0].name == "paper_live_shadow_drift_warning"
    assert breaches[0].severity == "warning"
    monitor.close()


def test_slo_monitor_shadow_drift_critical_breach():
    monitor = SLOMonitor(
        shadow_drift_warning_threshold=0.10,
        shadow_drift_critical_threshold=0.15,
    )
    breaches = monitor.record_shadow_drift_summary({"paper_live_shadow_drift": 0.18})
    assert len(breaches) == 1
    assert breaches[0].name == "paper_live_shadow_drift"
    assert breaches[0].severity == "critical"
    monitor.close()
