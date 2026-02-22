from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from utils.governance_gate import run_governance_gate


def test_governance_gate_paper_mode_passes_with_policy_doc(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "COMPLIANCE_GOVERNANCE.md").write_text("# policy\n", encoding="utf-8")

    report = run_governance_gate(repo_root=repo, mode="paper")

    assert report["ready"] is True
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["compliance_policy_doc_present"]["passed"] is True


def test_governance_gate_live_mode_requires_dual_approval(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "COMPLIANCE_GOVERNANCE.md").write_text("# policy\n", encoding="utf-8")
    approval_dir = repo / "results" / "governance"
    approval_dir.mkdir(parents=True, exist_ok=True)
    approval = {
        "approvers": ["ops_lead"],
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat(),
        "max_notional_usd": 100000,
        "kyc_attestation": True,
        "aml_attestation": True,
        "best_execution_policy_ack": True,
        "strategy_allowlist": ["momentum"],
    }
    (approval_dir / "live_approval.json").write_text(json.dumps(approval), encoding="utf-8")

    report = run_governance_gate(repo_root=repo, mode="live")

    assert report["ready"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["dual_approval_present"]["passed"] is False


def test_governance_gate_live_mode_passes_with_valid_artifact(tmp_path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "COMPLIANCE_GOVERNANCE.md").write_text("# policy\n", encoding="utf-8")
    approval_dir = repo / "results" / "governance"
    approval_dir.mkdir(parents=True, exist_ok=True)
    approval = {
        "approvers": ["ops_lead", "risk_lead"],
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat(),
        "max_notional_usd": 100000,
        "kyc_attestation": True,
        "aml_attestation": True,
        "best_execution_policy_ack": True,
        "strategy_allowlist": ["momentum"],
    }
    (approval_dir / "live_approval.json").write_text(json.dumps(approval), encoding="utf-8")

    report = run_governance_gate(repo_root=repo, mode="live")

    assert report["ready"] is True
