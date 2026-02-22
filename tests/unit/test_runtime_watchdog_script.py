from __future__ import annotations

import json
import subprocess
import sys


def test_runtime_watchdog_script_passes_when_all_checks_skipped(tmp_path):
    output_path = tmp_path / "runtime_watchdog.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_watchdog.py",
            "--no-check-alpaca",
            "--no-check-ticket-webhook",
            "--no-check-ib-port",
            "--no-check-ib-api",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "RUNTIME WATCHDOG" in proc.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert len(payload["checks"]) == 4


def test_runtime_watchdog_script_fails_when_alpaca_credentials_missing(tmp_path):
    output_path = tmp_path / "runtime_watchdog.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/runtime_watchdog.py",
            "--check-alpaca",
            "--no-check-ticket-webhook",
            "--no-check-ib-port",
            "--no-check-ib-api",
            "--alpaca-api-key",
            "",
            "--alpaca-secret-key",
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
    assert checks["alpaca_connectivity"]["passed"] is False
