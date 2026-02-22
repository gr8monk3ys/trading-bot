from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone


def _build_report(
    *,
    webhook_url: str,
    delivery_succeeded: bool = True,
    runbook_url: str = "https://ops.example.com/runbooks/trading",
    escalation_roster_url: str = "https://ops.example.com/escalation/trading",
) -> dict:
    return {
        "drill_id": "20260221T000000Z",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "delivery": {
            "attempted": True,
            "succeeded": delivery_succeeded,
            "events_recorded": 1,
        },
        "ticketing": {"attempts": 1, "created": 1, "failures": 0},
        "dead_letters": {"queued": 0},
        "webhook": {"url": webhook_url},
        "response_links": {
            "runbook_url": runbook_url,
            "escalation_roster_url": escalation_roster_url,
        },
    }


def test_validate_incident_ticket_drill_evidence_script_passes_for_non_test_target(tmp_path):
    report_path = tmp_path / "incident_ticket_drill_report.json"
    output_path = tmp_path / "evidence_gate.json"
    report_path.write_text(
        json.dumps(
            _build_report(webhook_url="https://hooks.ops.example.com/incident-ticket"),
            indent=2,
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_incident_ticket_drill_evidence.py",
            "--report-path",
            str(report_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    gate = json.loads(output_path.read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["failures"] == []


def test_validate_incident_ticket_drill_evidence_script_fails_for_test_target(tmp_path):
    report_path = tmp_path / "incident_ticket_drill_report.json"
    output_path = tmp_path / "evidence_gate.json"
    report_path.write_text(
        json.dumps(
            _build_report(webhook_url="http://127.0.0.1:8123/incident-ticket"),
            indent=2,
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_incident_ticket_drill_evidence.py",
            "--report-path",
            str(report_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    gate = json.loads(output_path.read_text(encoding="utf-8"))
    assert gate["passed"] is False
    assert any("non_test_target" in failure for failure in gate["failures"])
