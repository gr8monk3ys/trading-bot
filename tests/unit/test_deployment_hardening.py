#!/usr/bin/env python3
"""
Unit tests for deployment hardening checks.
"""

from utils.deployment_hardening import run_deployment_preflight, run_runtime_rollback_drill


def test_deployment_preflight_reports_missing_required_env_var(monkeypatch):
    monkeypatch.delenv("MUST_HAVE_ENV", raising=False)
    report = run_deployment_preflight(
        repo_root=".",
        required_env_vars=["MUST_HAVE_ENV"],
    )
    env_check = next(c for c in report["checks"] if c["name"] == "required_env_vars_present")
    assert env_check["passed"] is False
    assert "MUST_HAVE_ENV" in env_check["message"]


def test_deployment_preflight_reports_ready_when_no_required_env_vars():
    report = run_deployment_preflight(repo_root=".", required_env_vars=[])
    assert "ready" in report
    assert isinstance(report["checks"], list)


def test_deployment_preflight_can_run_rollback_drill(tmp_path):
    report = run_deployment_preflight(
        repo_root=".",
        required_env_vars=[],
        run_rollback_drill=True,
        rollback_drill_workdir=tmp_path,
    )
    drill = next(c for c in report["checks"] if c["name"] == "runtime_rollback_drill")
    assert drill["passed"] is True


def test_runtime_rollback_drill_reports_success(tmp_path):
    report = run_runtime_rollback_drill(workdir=tmp_path)
    assert report["passed"] is True
    assert "Recovered runtime snapshot" in report["message"]


def test_deployment_preflight_includes_incident_ownership_checks():
    report = run_deployment_preflight(repo_root=".", required_env_vars=[])
    checks = {c["name"]: c for c in report["checks"]}
    assert "runbook_links_incident_ownership" in checks
    assert "runbook_links_escalation_roster" in checks
    assert "incident_ownership_doc_present" in checks
    assert "incident_escalation_roster_present" in checks
    assert "canary_rollout_script_present" in checks
    assert "infra_as_code_manifests_present" in checks
    assert "governance_gate_script_present" in checks
    assert "secrets_audit_script_present" in checks
    assert "ops_metrics_push_script_present" in checks
    assert "incident_response_automation_script_present" in checks


def test_deployment_preflight_can_run_secrets_audit(monkeypatch):
    monkeypatch.setattr(
        "utils.deployment_hardening.run_secrets_audit",
        lambda **kwargs: {"ready": True, "checks": []},
    )
    report = run_deployment_preflight(
        repo_root=".",
        required_env_vars=[],
        run_secrets_audit_check=True,
    )
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["secrets_audit"]["passed"] is True
