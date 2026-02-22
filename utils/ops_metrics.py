"""
Operational metrics snapshot and Prometheus text exposition helpers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from utils.run_artifacts import read_jsonl


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _count_slo(rows: list[dict[str, Any]]) -> tuple[int, int]:
    critical = 0
    warning = 0
    for row in rows:
        if str(row.get("event_type", "")).strip().lower() != "slo_breach":
            continue
        severity = str(row.get("severity", "")).strip().lower()
        if severity == "critical":
            critical += 1
        elif severity == "warning":
            warning += 1
    return critical, warning


def build_ops_metrics_snapshot(
    *,
    run_dir: str | Path,
    runtime_watchdog_json: str | Path | None = None,
    runtime_gate_json: str | Path | None = None,
    go_live_summary_json: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build one metrics snapshot from run artifacts and readiness reports.
    """
    run_path = Path(run_dir)
    incident_rows = read_jsonl(run_path / "incident_events.jsonl")
    slo_rows = read_jsonl(run_path / "ops_slo_events.jsonl")
    position_rows = read_jsonl(run_path / "position_reconciliation_events.jsonl")
    order_rows = read_jsonl(run_path / "order_reconciliation_events.jsonl")
    dead_letter_rows = read_jsonl(run_path / "notification_dead_letters.jsonl")

    incident_types = [str(row.get("event_type", "")).strip().lower() for row in incident_rows]
    incidents_opened = incident_types.count("incident_open")
    incidents_acknowledged = incident_types.count("incident_ack")
    incident_sla_breaches = incident_types.count("incident_sla_breach")

    slo_critical, slo_warning = _count_slo(slo_rows)
    position_recon_failures = sum(1 for row in position_rows if bool(row.get("has_mismatch")))
    order_recon_mismatch_runs = sum(
        1 for row in order_rows if _safe_int(row.get("mismatch_count")) > 0
    )

    watchdog_report = _read_json(Path(runtime_watchdog_json) if runtime_watchdog_json else None)
    gate_report = _read_json(Path(runtime_gate_json) if runtime_gate_json else None)
    go_live_report = _read_json(Path(go_live_summary_json) if go_live_summary_json else None)

    metrics: dict[str, int] = {
        "incidents_opened_total": incidents_opened,
        "incidents_acknowledged_total": incidents_acknowledged,
        "incidents_unacknowledged_estimate": max(0, incidents_opened - incidents_acknowledged),
        "incident_sla_breaches_total": incident_sla_breaches,
        "slo_breaches_critical_total": slo_critical,
        "slo_breaches_warning_total": slo_warning,
        "position_reconciliation_failures_total": position_recon_failures,
        "order_reconciliation_mismatch_runs_total": order_recon_mismatch_runs,
        "notification_dead_letters_queued_total": len(dead_letter_rows),
        "runtime_watchdog_ready": 1 if bool(watchdog_report.get("ready")) else 0,
        "runtime_industrial_gate_ready": 1 if bool(gate_report.get("ready")) else 0,
        "go_live_precheck_ready": 1 if bool(go_live_report.get("ready")) else 0,
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_path),
        "metrics": metrics,
        "sources": {
            "runtime_watchdog_json": str(runtime_watchdog_json) if runtime_watchdog_json else None,
            "runtime_gate_json": str(runtime_gate_json) if runtime_gate_json else None,
            "go_live_summary_json": str(go_live_summary_json) if go_live_summary_json else None,
        },
    }


def format_prometheus_text(
    snapshot: Mapping[str, Any],
    *,
    namespace: str = "trading_bot",
) -> str:
    """
    Render snapshot metrics as Prometheus text exposition format.
    """
    metrics = snapshot.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}

    ns = str(namespace or "trading_bot").strip().lower().replace("-", "_")
    if not ns:
        ns = "trading_bot"

    help_map = {
        "incidents_opened_total": "Total incidents opened.",
        "incidents_acknowledged_total": "Total incidents acknowledged.",
        "incidents_unacknowledged_estimate": "Estimated currently unacknowledged incidents.",
        "incident_sla_breaches_total": "Total incident acknowledgment SLA breaches.",
        "slo_breaches_critical_total": "Total critical SLO breaches.",
        "slo_breaches_warning_total": "Total warning SLO breaches.",
        "position_reconciliation_failures_total": "Total position reconciliation failure runs.",
        "order_reconciliation_mismatch_runs_total": "Total order reconciliation mismatch runs.",
        "notification_dead_letters_queued_total": "Current queued notification dead letters.",
        "runtime_watchdog_ready": "Runtime watchdog readiness flag (1=ready,0=not ready).",
        "runtime_industrial_gate_ready": "Runtime industrial gate readiness flag (1=ready,0=not ready).",
        "go_live_precheck_ready": "Go-live precheck readiness flag (1=ready,0=not ready).",
    }

    lines: list[str] = []
    for key, raw_value in metrics.items():
        metric_name = f"{ns}_{str(key).strip().lower().replace('-', '_')}"
        value = _safe_int(raw_value)
        help_text = help_map.get(str(key), f"{key} metric.")
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")
        lines.append(f"{metric_name} {value}")
    lines.append("")
    return "\n".join(lines)
