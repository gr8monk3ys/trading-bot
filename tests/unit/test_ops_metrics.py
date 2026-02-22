from __future__ import annotations

import json
from pathlib import Path

from utils.ops_metrics import build_ops_metrics_snapshot, format_prometheus_text


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def test_build_ops_metrics_snapshot_counts_expected_metrics(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "incident_events.jsonl",
        [
            {"event_type": "incident_open"},
            {"event_type": "incident_open"},
            {"event_type": "incident_ack"},
            {"event_type": "incident_sla_breach"},
        ],
    )
    _write_jsonl(
        run_dir / "ops_slo_events.jsonl",
        [
            {"event_type": "slo_breach", "severity": "critical"},
            {"event_type": "slo_breach", "severity": "warning"},
            {"event_type": "other"},
        ],
    )
    _write_jsonl(
        run_dir / "position_reconciliation_events.jsonl",
        [{"has_mismatch": True}, {"has_mismatch": False}],
    )
    _write_jsonl(
        run_dir / "order_reconciliation_events.jsonl",
        [{"mismatch_count": 0}, {"mismatch_count": 2}],
    )
    _write_jsonl(
        run_dir / "notification_dead_letters.jsonl",
        [{"event_type": "notification_dead_letter"}],
    )

    watchdog_json = tmp_path / "watchdog.json"
    watchdog_json.write_text(json.dumps({"ready": True}), encoding="utf-8")
    gate_json = tmp_path / "runtime_gate.json"
    gate_json.write_text(json.dumps({"ready": False}), encoding="utf-8")
    precheck_json = tmp_path / "go_live_summary.json"
    precheck_json.write_text(json.dumps({"ready": True}), encoding="utf-8")

    snapshot = build_ops_metrics_snapshot(
        run_dir=run_dir,
        runtime_watchdog_json=watchdog_json,
        runtime_gate_json=gate_json,
        go_live_summary_json=precheck_json,
    )

    metrics = snapshot["metrics"]
    assert metrics["incidents_opened_total"] == 2
    assert metrics["incidents_acknowledged_total"] == 1
    assert metrics["incidents_unacknowledged_estimate"] == 1
    assert metrics["incident_sla_breaches_total"] == 1
    assert metrics["slo_breaches_critical_total"] == 1
    assert metrics["slo_breaches_warning_total"] == 1
    assert metrics["position_reconciliation_failures_total"] == 1
    assert metrics["order_reconciliation_mismatch_runs_total"] == 1
    assert metrics["notification_dead_letters_queued_total"] == 1
    assert metrics["runtime_watchdog_ready"] == 1
    assert metrics["runtime_industrial_gate_ready"] == 0
    assert metrics["go_live_precheck_ready"] == 1


def test_format_prometheus_text_renders_namespace():
    snapshot = {
        "metrics": {
            "runtime_watchdog_ready": 1,
            "slo_breaches_critical_total": 2,
        }
    }
    text = format_prometheus_text(snapshot, namespace="trade_ops")
    assert "trade_ops_runtime_watchdog_ready 1" in text
    assert "trade_ops_slo_breaches_critical_total 2" in text
    assert "# HELP trade_ops_runtime_watchdog_ready" in text
