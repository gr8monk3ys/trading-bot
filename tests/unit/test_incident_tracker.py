#!/usr/bin/env python3
"""
Unit tests for incident acknowledgment tracking.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from utils.incident_tracker import IncidentTracker


def test_incident_tracker_open_and_acknowledge(tmp_path):
    events_path = tmp_path / "incident_events.jsonl"
    tracker = IncidentTracker(events_path=events_path, run_id="run_1", ack_sla_minutes=15)
    try:
        incident = tracker.open_incident(
            {
                "name": "data_quality_errors",
                "severity": "critical",
                "message": "Data quality threshold breached",
                "context": {"symbol": "AAPL"},
            }
        )
        assert incident["incident_id"].startswith("inc_")

        snapshot = tracker.get_status_snapshot()
        assert snapshot["total_incidents"] == 1
        assert snapshot["unacknowledged_incidents"] == 1

        assert tracker.acknowledge(incident["incident_id"], "oncall", "triage started") is True
        snapshot = tracker.get_status_snapshot()
        assert snapshot["acknowledged_incidents"] == 1
        assert snapshot["unacknowledged_incidents"] == 0
    finally:
        tracker.close()


def test_incident_tracker_ack_sla_breach_emits_once(tmp_path):
    events_path = tmp_path / "incident_events.jsonl"
    tracker = IncidentTracker(events_path=events_path, run_id="run_2", ack_sla_minutes=5)
    try:
        incident = tracker.open_incident(
            {
                "name": "order_reconciliation_consecutive_mismatch_runs",
                "severity": "critical",
                "message": "Recon mismatch threshold breached",
                "context": {},
            }
        )
        created_at = datetime.fromisoformat(incident["created_at"])
        breached = tracker.evaluate_ack_sla(now=created_at + timedelta(minutes=6))
        assert len(breached) == 1
        assert breached[0]["incident_id"] == incident["incident_id"]

        # Should not duplicate the same SLA breach.
        breached_again = tracker.evaluate_ack_sla(now=created_at + timedelta(minutes=7))
        assert breached_again == []
    finally:
        tracker.close()


def test_incident_tracker_external_ack_round_trip(tmp_path):
    events_path = tmp_path / "incident_events.jsonl"
    tracker_a = IncidentTracker(events_path=events_path, run_id="run_3", ack_sla_minutes=10)
    tracker_b = IncidentTracker(events_path=events_path, run_id="run_3", ack_sla_minutes=10)
    try:
        incident = tracker_a.open_incident(
            {
                "name": "data_quality_stale_warnings",
                "severity": "critical",
                "message": "Stale warning threshold breached",
                "context": {},
            }
        )
        assert tracker_b.acknowledge(incident["incident_id"], "secondary_oncall", "acked via cli")

        # Tracker A should pick up external ack from event stream.
        snapshot = tracker_a.get_status_snapshot()
        assert snapshot["acknowledged_incidents"] == 1
        assert snapshot["unacknowledged_incidents"] == 0
    finally:
        tracker_a.close()
        tracker_b.close()
