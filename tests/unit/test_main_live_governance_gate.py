from __future__ import annotations

from pathlib import Path

from main import _evaluate_live_governance_gate


def test_live_governance_gate_can_be_explicitly_skipped():
    ready, report = _evaluate_live_governance_gate(
        enforce=False,
        repo_root=".",
        mode="live",
    )

    assert ready is True
    assert report["ready"] is True
    assert report["skipped"] is True


def test_live_governance_gate_passes_when_underlying_gate_is_ready(monkeypatch, tmp_path):
    captured: dict[str, str] = {}

    def _fake_run_governance_gate(**kwargs):
        captured.update({k: str(v) for k, v in kwargs.items()})
        return {"ready": True, "checks": [{"name": "x", "passed": True, "severity": "critical"}]}

    monkeypatch.setattr("main.run_governance_gate", _fake_run_governance_gate)

    approval = tmp_path / "approval.json"
    policy = tmp_path / "policy.md"
    ready, report = _evaluate_live_governance_gate(
        enforce=True,
        repo_root=tmp_path,
        mode="live",
        approval_path=approval,
        policy_doc_path=policy,
    )

    assert ready is True
    assert report["ready"] is True
    assert captured["repo_root"] == str(tmp_path)
    assert captured["mode"] == "live"
    assert captured["approval_path"] == str(approval)
    assert captured["policy_doc_path"] == str(policy)


def test_live_governance_gate_fails_when_underlying_gate_not_ready(monkeypatch):
    def _fake_run_governance_gate(**kwargs):
        return {
            "ready": False,
            "checks": [
                {
                    "name": "dual_approval_present",
                    "passed": False,
                    "severity": "critical",
                    "message": "missing",
                }
            ],
        }

    monkeypatch.setattr("main.run_governance_gate", _fake_run_governance_gate)

    ready, report = _evaluate_live_governance_gate(
        enforce=True,
        repo_root=Path("."),
        mode="live",
    )

    assert ready is False
    assert report["ready"] is False
