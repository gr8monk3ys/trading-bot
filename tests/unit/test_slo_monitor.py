#!/usr/bin/env python3
"""
Unit tests for operational SLO monitor.
"""

from datetime import datetime, timedelta

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


def test_order_reconciliation_health_ignores_historical_totals_when_run_is_clean(tmp_path):
    monitor = SLOMonitor(
        events_path=tmp_path / "slo_events.jsonl",
        recon_mismatch_halt_runs=2,
    )
    breaches = monitor.record_order_reconciliation_health(
        {
            "consecutive_mismatch_runs": 0,
            "total_mismatches": 7,
            "mismatch_count": 0,
            "last_run_mismatches": [],
        }
    )
    assert breaches == []
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


def test_slo_monitor_creates_ticket_for_incident_ack_sla_breach(tmp_path):
    class RecordingTicketNotifier:
        def __init__(self):
            self.events = []

        def notify(self, breach):
            self.events.append(breach)
            return True

    ticket_notifier = RecordingTicketNotifier()
    incident_tracker = IncidentTracker(
        events_path=tmp_path / "incident_events.jsonl",
        run_id="run_ticket",
        ack_sla_minutes=1,
    )
    monitor = SLOMonitor(
        incident_tracker=incident_tracker,
        incident_ticket_notifier=ticket_notifier,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )

    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    incident_id = next(iter(incident_tracker._incidents.keys()))
    created_at = incident_tracker._incidents[incident_id].created_at

    breaches = monitor.check_incident_ack_sla(now=created_at + timedelta(minutes=2))
    status = monitor.get_status_snapshot()

    assert any(b.name == "incident_ack_sla_breach" for b in breaches)
    assert len(ticket_notifier.events) == 1
    assert status["ticketing"]["attempts"] == 1
    assert status["ticketing"]["created"] == 1
    assert status["ticketing"]["failures"] == 0
    monitor.close()


def test_slo_monitor_queues_dead_letter_on_alert_failure(tmp_path):
    class FailingAlertNotifier:
        def notify(self, breach):
            return False

    dead_letter_path = tmp_path / "notification_dead_letters.jsonl"
    monitor = SLOMonitor(
        alert_notifier=FailingAlertNotifier(),
        notification_dead_letter_path=dead_letter_path,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )

    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    status = monitor.get_status_snapshot()
    monitor.close()

    assert status["dead_letters"]["queued"] == 1
    assert status["dead_letters"]["slo_alert"] == 1
    assert status["dead_letters"]["incident_ticket"] == 0
    rows = read_jsonl(dead_letter_path)
    assert len(rows) == 1
    assert rows[0]["channel"] == "slo_alert"
    assert rows[0]["event_type"] == "notification_dead_letter"


def test_slo_monitor_queues_dead_letter_on_ticket_failure(tmp_path):
    class FailingTicketNotifier:
        def notify(self, breach):
            return False

    incident_tracker = IncidentTracker(
        events_path=tmp_path / "incident_events.jsonl",
        run_id="run_dead_letter",
        ack_sla_minutes=1,
    )
    dead_letter_path = tmp_path / "notification_dead_letters.jsonl"
    monitor = SLOMonitor(
        incident_tracker=incident_tracker,
        incident_ticket_notifier=FailingTicketNotifier(),
        notification_dead_letter_path=dead_letter_path,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )

    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    incident_id = next(iter(incident_tracker._incidents.keys()))
    created_at = incident_tracker._incidents[incident_id].created_at
    monitor.check_incident_ack_sla(now=created_at + timedelta(minutes=2))
    status = monitor.get_status_snapshot()
    monitor.close()

    assert status["dead_letters"]["queued"] == 1
    assert status["dead_letters"]["slo_alert"] == 0
    assert status["dead_letters"]["incident_ticket"] == 1
    rows = read_jsonl(dead_letter_path)
    assert len(rows) == 1
    assert rows[0]["channel"] == "incident_ticket"
    assert rows[0]["event_type"] == "notification_dead_letter"


def test_slo_monitor_emits_dead_letter_backlog_warning_after_persist_window(tmp_path):
    monitor = SLOMonitor(
        notification_dead_letter_warning_threshold=2,
        notification_dead_letter_critical_threshold=5,
        notification_dead_letter_persist_minutes=2,
    )
    base_time = datetime(2026, 1, 1, 10, 0, 0)

    first = monitor.record_notification_dead_letter_backlog(2, now=base_time)
    second = monitor.record_notification_dead_letter_backlog(
        2,
        now=base_time + timedelta(minutes=1),
    )
    third = monitor.record_notification_dead_letter_backlog(
        2,
        now=base_time + timedelta(minutes=2),
    )

    assert first == []
    assert second == []
    assert len(third) == 1
    assert third[0].name == "notification_dead_letter_backlog_warning"
    assert third[0].severity == "warning"
    monitor.close()


def test_slo_monitor_dead_letter_backlog_alert_is_edge_triggered():
    monitor = SLOMonitor(
        notification_dead_letter_warning_threshold=1,
        notification_dead_letter_critical_threshold=3,
        notification_dead_letter_persist_minutes=1,
    )
    base_time = datetime(2026, 1, 1, 10, 0, 0)

    first = monitor.record_notification_dead_letter_backlog(1, now=base_time)
    second = monitor.record_notification_dead_letter_backlog(
        1,
        now=base_time + timedelta(minutes=1),
    )
    third = monitor.record_notification_dead_letter_backlog(
        1,
        now=base_time + timedelta(minutes=3),
    )
    reset = monitor.record_notification_dead_letter_backlog(0, now=base_time + timedelta(minutes=4))
    fourth = monitor.record_notification_dead_letter_backlog(
        1,
        now=base_time + timedelta(minutes=5),
    )
    fifth = monitor.record_notification_dead_letter_backlog(
        1,
        now=base_time + timedelta(minutes=6),
    )

    assert first == []
    assert len(second) == 1
    assert third == []
    assert reset == []
    assert fourth == []
    assert len(fifth) == 1
    monitor.close()


def test_slo_monitor_backlog_alert_failure_does_not_requeue_dead_letter(tmp_path):
    class FailingAlertNotifier:
        def notify(self, breach):
            return False

    dead_letter_path = tmp_path / "notification_dead_letters.jsonl"
    monitor = SLOMonitor(
        alert_notifier=FailingAlertNotifier(),
        notification_dead_letter_path=dead_letter_path,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
        notification_dead_letter_warning_threshold=1,
        notification_dead_letter_critical_threshold=2,
        notification_dead_letter_persist_minutes=1,
    )

    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    base_time = datetime(2026, 1, 1, 10, 0, 0)
    monitor.record_notification_dead_letter_backlog(1, now=base_time)
    breaches = monitor.record_notification_dead_letter_backlog(
        1,
        now=base_time + timedelta(minutes=1),
    )
    status = monitor.get_status_snapshot()
    monitor.close()

    assert len(breaches) == 1
    assert breaches[0].name == "notification_dead_letter_backlog_warning"
    assert status["dead_letters"]["queued"] == 1
    rows = read_jsonl(dead_letter_path)
    assert len(rows) == 1
