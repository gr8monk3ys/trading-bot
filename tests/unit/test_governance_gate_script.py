from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone


def test_governance_gate_script_live_mode(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "COMPLIANCE_GOVERNANCE.md").write_text("# policy\n", encoding="utf-8")

    approval_dir = repo / "results" / "governance"
    approval_dir.mkdir(parents=True, exist_ok=True)
    approval = {
        "approvers": ["ops_lead", "risk_lead"],
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(),
        "max_notional_usd": 100000,
        "kyc_attestation": True,
        "aml_attestation": True,
        "best_execution_policy_ack": True,
        "strategy_allowlist": ["momentum"],
    }
    (approval_dir / "live_approval.json").write_text(json.dumps(approval), encoding="utf-8")

    output = tmp_path / "gate.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/governance_gate.py",
            "--repo-root",
            str(repo),
            "--mode",
            "live",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "GOVERNANCE GATE" in proc.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
