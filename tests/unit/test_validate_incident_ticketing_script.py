from __future__ import annotations

import json
import subprocess
import sys


def test_validate_incident_ticketing_script_passes(tmp_path):
    output = tmp_path / "ticketing_validation.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_incident_ticketing.py",
            "--tmp-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "ACK-SLA TICKETING VALIDATION" in proc.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["ticketing"]["created"] == 1
