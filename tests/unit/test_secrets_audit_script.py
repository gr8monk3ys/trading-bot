from __future__ import annotations

import json
import subprocess
import sys


def test_secrets_audit_script_reports_ready(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (docs / "SECRETS_ROTATION_INVENTORY.json").write_text(
        json.dumps(
            {
                "secrets": [
                    {"name": "ALPACA_API_KEY", "last_rotated": "2026-02-01", "max_age_days": 365}
                ]
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "secrets.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/secrets_audit.py",
            "--repo-root",
            str(repo),
            "--inventory-path",
            "docs/SECRETS_ROTATION_INVENTORY.json",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
