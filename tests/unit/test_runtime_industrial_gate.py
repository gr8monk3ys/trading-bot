from __future__ import annotations

import asyncio

from utils import runtime_industrial_gate


def test_runtime_gate_passes_when_base_checks_pass(monkeypatch):
    monkeypatch.setattr(
        runtime_industrial_gate,
        "validate_incident_contacts",
        lambda ownership_doc, escalation_doc: {
            "valid": True,
            "placeholder_count": 0,
            "findings": [],
            "ownership_doc": ownership_doc,
            "escalation_doc": escalation_doc,
        },
    )

    async def _chaos_ok():
        return {"passed": True, "checks": [{"name": "dummy", "passed": True}]}

    monkeypatch.setattr(runtime_industrial_gate, "run_chaos_drills", _chaos_ok)

    report = asyncio.run(
        runtime_industrial_gate.run_runtime_industrial_gate(
            run_ticket_drill=False,
            run_failover_probe=False,
        )
    )

    assert report["ready"] is True
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["incident_contacts"]["passed"] is True
    assert checks["chaos_drills"]["passed"] is True


def test_runtime_gate_fails_when_incident_contacts_invalid(monkeypatch):
    monkeypatch.setattr(
        runtime_industrial_gate,
        "validate_incident_contacts",
        lambda ownership_doc, escalation_doc: {
            "valid": False,
            "placeholder_count": 2,
            "findings": [
                {"doc": ownership_doc, "line": 1, "placeholder": "REPLACE_WITH_OWNER_TEAM"},
                {"doc": escalation_doc, "line": 1, "placeholder": "REPLACE_WITH_CONTACT"},
            ],
            "ownership_doc": ownership_doc,
            "escalation_doc": escalation_doc,
        },
    )

    async def _chaos_ok():
        return {"passed": True, "checks": []}

    monkeypatch.setattr(runtime_industrial_gate, "run_chaos_drills", _chaos_ok)

    report = asyncio.run(
        runtime_industrial_gate.run_runtime_industrial_gate(
            run_ticket_drill=False,
            run_failover_probe=False,
        )
    )

    assert report["ready"] is False
    incident = next(check for check in report["checks"] if check["name"] == "incident_contacts")
    assert incident["passed"] is False
    assert incident["details"]["placeholder_count"] == 2


def test_runtime_gate_runs_ticket_and_failover_checks(monkeypatch):
    monkeypatch.setattr(
        runtime_industrial_gate,
        "validate_incident_contacts",
        lambda ownership_doc, escalation_doc: {
            "valid": True,
            "placeholder_count": 0,
            "findings": [],
            "ownership_doc": ownership_doc,
            "escalation_doc": escalation_doc,
        },
    )

    async def _chaos_ok():
        return {"passed": True, "checks": []}

    monkeypatch.setattr(runtime_industrial_gate, "run_chaos_drills", _chaos_ok)
    monkeypatch.setattr(
        runtime_industrial_gate,
        "_check_staging_ticket_drill",
        lambda **kwargs: runtime_industrial_gate.RuntimeGateCheck(
            name="staging_incident_ticket_drill",
            passed=True,
            message="ok",
        ),
    )

    async def _failover_ok(**kwargs):
        return runtime_industrial_gate.RuntimeGateCheck(
            name="multi_broker_runtime_failover_probe",
            passed=True,
            message="ok",
        )

    monkeypatch.setattr(
        runtime_industrial_gate,
        "_check_multi_broker_failover_probe",
        _failover_ok,
    )

    report = asyncio.run(
        runtime_industrial_gate.run_runtime_industrial_gate(
            run_ticket_drill=True,
            ticket_webhook_url="https://hooks.ops.example.com/incident-ticket",
            run_failover_probe=True,
        )
    )

    assert report["ready"] is True
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["staging_incident_ticket_drill"]["passed"] is True
    assert checks["multi_broker_runtime_failover_probe"]["passed"] is True


def test_runtime_gate_ticket_drill_fails_without_webhook(monkeypatch):
    monkeypatch.setattr(
        runtime_industrial_gate,
        "validate_incident_contacts",
        lambda ownership_doc, escalation_doc: {
            "valid": True,
            "placeholder_count": 0,
            "findings": [],
            "ownership_doc": ownership_doc,
            "escalation_doc": escalation_doc,
        },
    )

    async def _chaos_ok():
        return {"passed": True, "checks": []}

    monkeypatch.setattr(runtime_industrial_gate, "run_chaos_drills", _chaos_ok)

    report = asyncio.run(
        runtime_industrial_gate.run_runtime_industrial_gate(
            run_ticket_drill=True,
            ticket_webhook_url="",
            run_failover_probe=False,
        )
    )

    assert report["ready"] is False
    ticket = next(
        check for check in report["checks"] if check["name"] == "staging_incident_ticket_drill"
    )
    assert ticket["passed"] is False
    assert "Missing webhook URL" in ticket["message"]
