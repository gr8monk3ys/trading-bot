from __future__ import annotations

import json
import subprocess
import sys


def _read_payloads(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_staging_incident_ticket_drill_script_delivers_webhook(tmp_path):
    report_path = tmp_path / "staging_ticket_drill_report.json"
    webhook_payload_path = tmp_path / "incident_ticket_payloads.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/staging_incident_ticket_drill.py",
            "--webhook-url",
            webhook_payload_path.as_uri(),
            "--artifact-dir",
            str(tmp_path),
            "--output",
            str(report_path),
            "--require-delivery",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "STAGING INCIDENT TICKET DRILL" in proc.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["delivery"]["attempted"] is True
    assert report["delivery"]["succeeded"] is True
    assert report["ticketing"]["created"] == 1
    assert report["dead_letters"]["queued"] == 0

    payloads = _read_payloads(webhook_payload_path)
    assert len(payloads) >= 1
    webhook_payload = payloads[0]["payload"]
    assert webhook_payload["event_type"] == "incident_ticket"
    assert webhook_payload["breach"]["name"] == "incident_ack_sla_breach"


def test_staging_incident_ticket_drill_script_fails_when_non_test_target_required(tmp_path):
    report_path = tmp_path / "staging_ticket_drill_report.json"
    webhook_payload_path = tmp_path / "incident_ticket_payloads.jsonl"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/staging_incident_ticket_drill.py",
            "--webhook-url",
            webhook_payload_path.as_uri(),
            "--artifact-dir",
            str(tmp_path),
            "--output",
            str(report_path),
            "--require-delivery",
            "--require-non-test-target",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["webhook"]["is_non_test_target"] is False
