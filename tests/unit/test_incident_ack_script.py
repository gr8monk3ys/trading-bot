#!/usr/bin/env python3
"""
Integration-style tests for incident acknowledgment CLI.
"""

from __future__ import annotations

import subprocess
import sys

from utils.incident_tracker import IncidentTracker


def test_incident_ack_script_acknowledges_incident(tmp_path):
    events_path = tmp_path / "incident_events.jsonl"
    tracker = IncidentTracker(events_path=events_path, run_id="run_script", ack_sla_minutes=15)
    try:
        incident = tracker.open_incident(
            {
                "name": "order_reconciliation_consecutive_mismatch_runs",
                "severity": "critical",
                "message": "Recon mismatch threshold breached",
                "context": {},
            }
        )
    finally:
        tracker.close()

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/incident_ack.py",
            "--events-path",
            str(events_path),
            "--incident-id",
            incident["incident_id"],
            "--ack-by",
            "unit_test_oncall",
            "--notes",
            "ack from test",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Acknowledged incident" in proc.stdout

    verify = IncidentTracker(events_path=events_path, run_id="run_script", ack_sla_minutes=15)
    try:
        status = verify.get_status_snapshot()
        assert status["acknowledged_incidents"] == 1
        assert status["unacknowledged_incidents"] == 0
    finally:
        verify.close()
