from __future__ import annotations

import json
from pathlib import Path

from utils.incident_response_automation import (
    evaluate_incident_response_plan,
    run_incident_response_automation,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_evaluate_incident_response_plan_ready_when_no_triggers():
    plan = evaluate_incident_response_plan(
        {
            "metrics": {
                "runtime_watchdog_ready": 1,
                "runtime_industrial_gate_ready": 1,
                "go_live_precheck_ready": 1,
                "slo_breaches_critical_total": 0,
                "incident_sla_breaches_total": 0,
                "notification_dead_letters_queued_total": 0,
                "incidents_unacknowledged_estimate": 0,
                "slo_breaches_warning_total": 0,
            }
        }
    )

    assert plan["ready"] is True
    assert plan["severity"] == "ok"
    assert plan["trigger_count"] == 0


def test_evaluate_incident_response_plan_critical_conditions():
    plan = evaluate_incident_response_plan(
        {
            "metrics": {
                "runtime_watchdog_ready": 0,
                "runtime_industrial_gate_ready": 1,
                "go_live_precheck_ready": 1,
                "slo_breaches_critical_total": 2,
                "incident_sla_breaches_total": 0,
                "notification_dead_letters_queued_total": 30,
                "incidents_unacknowledged_estimate": 0,
                "slo_breaches_warning_total": 0,
            }
        },
        critical_slo_breach_threshold=1,
        dead_letter_critical_threshold=25,
    )

    assert plan["ready"] is False
    assert plan["severity"] == "critical"
    trigger_names = {trigger["name"] for trigger in plan["triggers"]}
    assert "runtime_watchdog_not_ready" in trigger_names
    assert "critical_slo_breach_threshold_exceeded" in trigger_names
    assert "notification_dead_letter_backlog_critical" in trigger_names


def test_run_incident_response_automation_executes_actions_for_critical(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        run_dir / "incident_events.jsonl",
        [{"event_type": "incident_sla_breach", "incident_id": "inc_1"}],
    )
    watchdog = tmp_path / "runtime_watchdog.json"
    _write_json(watchdog, {"ready": False})

    monkeypatch.setattr(
        "utils.incident_response_automation._deliver_incident_webhook",
        lambda **kwargs: {
            "attempted": True,
            "delivered": True,
            "status_code": 200,
            "message": "ok",
        },
    )
    monkeypatch.setattr(
        "utils.incident_response_automation._run_rollback_command",
        lambda command, timeout_seconds: {
            "attempted": True,
            "succeeded": True,
            "returncode": 0,
            "message": "rollback_ok",
        },
    )

    report = run_incident_response_automation(
        run_dir=run_dir,
        runtime_watchdog_json=watchdog,
        incident_sla_breach_threshold=1,
        webhook_url="https://example.com/webhook",
        rollback_cmd="echo rollback",
    )

    assert report["ready"] is False
    assert report["plan"]["severity"] == "critical"
    assert report["actions"]["webhook"]["delivered"] is True
    assert report["actions"]["rollback"]["succeeded"] is True


def test_run_incident_response_automation_skips_actions_when_ready(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    watchdog = tmp_path / "runtime_watchdog.json"
    gate = tmp_path / "runtime_gate.json"
    precheck = tmp_path / "precheck.json"
    _write_json(watchdog, {"ready": True})
    _write_json(gate, {"ready": True})
    _write_json(precheck, {"ready": True})

    report = run_incident_response_automation(
        run_dir=run_dir,
        runtime_watchdog_json=watchdog,
        runtime_gate_json=gate,
        go_live_summary_json=precheck,
    )

    assert report["ready"] is True
    assert report["plan"]["severity"] == "ok"
    assert report["actions"]["webhook"]["attempted"] is False
    assert report["actions"]["rollback"]["attempted"] is False
