from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_incident_response_automation_script_ready_path(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    watchdog = tmp_path / "runtime_watchdog.json"
    gate = tmp_path / "runtime_gate.json"
    precheck = tmp_path / "precheck.json"
    output = tmp_path / "incident_response.json"
    _write_json(watchdog, {"ready": True})
    _write_json(gate, {"ready": True})
    _write_json(precheck, {"ready": True})

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/incident_response_automation.py",
            "--run-dir",
            str(run_dir),
            "--runtime-watchdog-json",
            str(watchdog),
            "--runtime-gate-json",
            str(gate),
            "--go-live-summary-json",
            str(precheck),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Ready: YES" in proc.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True


def test_incident_response_automation_script_critical_path(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        run_dir / "incident_events.jsonl",
        [{"event_type": "incident_sla_breach", "incident_id": "inc_1"}],
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/incident_response_automation.py",
            "--run-dir",
            str(run_dir),
            "--incident-sla-breach-threshold",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "Severity: critical" in proc.stdout
