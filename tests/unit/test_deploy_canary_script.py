from __future__ import annotations

import json
import subprocess
import sys


def test_deploy_canary_script_dry_run(tmp_path):
    output = tmp_path / "canary.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/deploy_canary.py",
            "--candidate-cmd",
            f"{sys.executable} -c \"print('candidate')\"",
            "--health-check-cmd",
            f"{sys.executable} -c \"print('healthy')\"",
            "--output",
            str(output),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["dry_run"] is True


def test_deploy_canary_script_rolls_back_on_health_failure(tmp_path):
    output = tmp_path / "canary.json"
    rollback_marker = tmp_path / "rollback_marker.txt"
    rollback_cmd = f"{sys.executable} -c \"from pathlib import Path; Path(r'{rollback_marker}').write_text('rolled', encoding='utf-8')\""
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/deploy_canary.py",
            "--candidate-cmd",
            f"{sys.executable} -c \"print('candidate')\"",
            "--health-check-cmd",
            f'{sys.executable} -c "import sys; sys.exit(2)"',
            "--rollback-cmd",
            rollback_cmd,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    steps = [step["step"] for step in payload["steps"]]
    assert "health_check" in steps
    assert "rollback_after_health_failure" in steps
    assert rollback_marker.exists()
