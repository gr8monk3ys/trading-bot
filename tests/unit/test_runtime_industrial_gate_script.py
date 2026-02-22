from __future__ import annotations

import json
import subprocess
import sys


def test_runtime_industrial_gate_script_passes_when_optional_checks_skipped(tmp_path):
    output_path = tmp_path / "runtime_gate.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_industrial_gate.py",
            "--no-run-chaos-drill",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "RUNTIME INDUSTRIAL READINESS GATE" in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ready"] is True


def test_runtime_industrial_gate_script_fails_when_ticket_drill_requested_without_webhook(tmp_path):
    output_path = tmp_path / "runtime_gate.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_industrial_gate.py",
            "--no-run-chaos-drill",
            "--run-ticket-drill",
            "--ticket-webhook-url",
            "",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["staging_incident_ticket_drill"]["passed"] is False
