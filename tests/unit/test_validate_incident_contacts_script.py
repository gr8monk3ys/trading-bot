from __future__ import annotations

import json
import subprocess
import sys


def test_validate_incident_contacts_script_fails_on_placeholders(tmp_path):
    ownership = tmp_path / "ownership.md"
    escalation = tmp_path / "escalation.md"
    output = tmp_path / "report.json"

    ownership.write_text("- owner: `REPLACE_WITH_OWNER_TEAM`\n", encoding="utf-8")
    escalation.write_text("- pager: `REPLACE_WITH_PAGER_POLICY_URL`\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_incident_contacts.py",
            "--ownership-doc",
            str(ownership),
            "--escalation-doc",
            str(escalation),
            "--json-output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "Valid: NO" in proc.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["valid"] is False
    assert payload["placeholder_count"] == 2


def test_validate_incident_contacts_script_passes_with_real_values(tmp_path):
    ownership = tmp_path / "ownership.md"
    escalation = tmp_path / "escalation.md"

    ownership.write_text("- owner: `Trading Ops`\n", encoding="utf-8")
    escalation.write_text("- pager: `https://pager.example.internal/trading`\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_incident_contacts.py",
            "--ownership-doc",
            str(ownership),
            "--escalation-doc",
            str(escalation),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Valid: YES" in proc.stdout
