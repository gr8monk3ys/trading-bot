"""
Incident response automation for operational readiness and SLO breaches.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request

from utils.ops_metrics import build_ops_metrics_snapshot


@dataclass
class IncidentTrigger:
    name: str
    severity: str
    message: str
    metric: str
    value: int
    threshold: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
        }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def evaluate_incident_response_plan(
    snapshot: Mapping[str, Any],
    *,
    critical_slo_breach_threshold: int = 1,
    incident_sla_breach_threshold: int = 1,
    dead_letter_critical_threshold: int = 25,
) -> dict[str, Any]:
    """
    Build incident response plan from one operational metrics snapshot.
    """
    metrics = snapshot.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}

    triggers: list[IncidentTrigger] = []

    def add_trigger(
        *,
        name: str,
        severity: str,
        message: str,
        metric: str,
        threshold: int,
    ) -> None:
        value = _safe_int(metrics.get(metric))
        triggers.append(
            IncidentTrigger(
                name=name,
                severity=severity,
                message=message,
                metric=metric,
                value=value,
                threshold=threshold,
            )
        )

    if _safe_int(metrics.get("runtime_watchdog_ready")) == 0:
        add_trigger(
            name="runtime_watchdog_not_ready",
            severity="critical",
            message="Runtime watchdog readiness failed",
            metric="runtime_watchdog_ready",
            threshold=1,
        )
    if _safe_int(metrics.get("runtime_industrial_gate_ready")) == 0:
        add_trigger(
            name="runtime_industrial_gate_not_ready",
            severity="critical",
            message="Runtime industrial gate readiness failed",
            metric="runtime_industrial_gate_ready",
            threshold=1,
        )
    if _safe_int(metrics.get("go_live_precheck_ready")) == 0:
        add_trigger(
            name="go_live_precheck_not_ready",
            severity="critical",
            message="Go-live precheck readiness failed",
            metric="go_live_precheck_ready",
            threshold=1,
        )

    critical_slo = _safe_int(metrics.get("slo_breaches_critical_total"))
    if critical_slo >= max(1, int(critical_slo_breach_threshold)):
        add_trigger(
            name="critical_slo_breach_threshold_exceeded",
            severity="critical",
            message="Critical SLO breach count exceeded threshold",
            metric="slo_breaches_critical_total",
            threshold=max(1, int(critical_slo_breach_threshold)),
        )

    incident_sla_breaches = _safe_int(metrics.get("incident_sla_breaches_total"))
    if incident_sla_breaches >= max(1, int(incident_sla_breach_threshold)):
        add_trigger(
            name="incident_ack_sla_breach_threshold_exceeded",
            severity="critical",
            message="Incident acknowledgment SLA breach count exceeded threshold",
            metric="incident_sla_breaches_total",
            threshold=max(1, int(incident_sla_breach_threshold)),
        )

    dead_letters = _safe_int(metrics.get("notification_dead_letters_queued_total"))
    if dead_letters >= max(1, int(dead_letter_critical_threshold)):
        add_trigger(
            name="notification_dead_letter_backlog_critical",
            severity="critical",
            message="Notification dead-letter backlog exceeded threshold",
            metric="notification_dead_letters_queued_total",
            threshold=max(1, int(dead_letter_critical_threshold)),
        )

    unacked_incidents = _safe_int(metrics.get("incidents_unacknowledged_estimate"))
    if unacked_incidents > 0:
        add_trigger(
            name="unacknowledged_incidents_present",
            severity="warning",
            message="Unacknowledged incidents remain open",
            metric="incidents_unacknowledged_estimate",
            threshold=0,
        )

    warning_slo = _safe_int(metrics.get("slo_breaches_warning_total"))
    if warning_slo > 0:
        add_trigger(
            name="warning_slo_breaches_present",
            severity="warning",
            message="Warning-level SLO breaches present",
            metric="slo_breaches_warning_total",
            threshold=0,
        )

    has_critical = any(trigger.severity == "critical" for trigger in triggers)
    has_warning = any(trigger.severity == "warning" for trigger in triggers)
    if has_critical:
        severity = "critical"
    elif has_warning:
        severity = "warning"
    else:
        severity = "ok"

    return {
        "ready": severity == "ok",
        "severity": severity,
        "trigger_count": len(triggers),
        "triggers": [trigger.to_dict() for trigger in triggers],
    }


def _authorization_header(auth_scheme: str, auth_token: str) -> str | None:
    token = str(auth_token or "").strip()
    if not token:
        return None
    scheme = str(auth_scheme or "").strip()
    if not scheme:
        return token
    return f"{scheme} {token}"


def _deliver_incident_webhook(
    *,
    webhook_url: str,
    payload: Mapping[str, Any],
    timeout_seconds: int,
    auth_token: str = "",
    auth_scheme: str = "Bearer",
) -> dict[str, Any]:
    url = str(webhook_url or "").strip()
    if not url:
        return {
            "attempted": False,
            "delivered": False,
            "status_code": None,
            "message": "Webhook URL not provided",
        }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "trading-bot-incident-automation/1.0",
    }
    auth_header = _authorization_header(auth_scheme, auth_token)
    if auth_header:
        headers["Authorization"] = auth_header

    req = request.Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=max(1, int(timeout_seconds))) as response:  # noqa: S310
            status = int(getattr(response, "status", 200) or 200)
        return {
            "attempted": True,
            "delivered": 200 <= status < 400,
            "status_code": status,
            "message": f"Webhook returned HTTP {status}",
        }
    except error.HTTPError as exc:
        return {
            "attempted": True,
            "delivered": False,
            "status_code": int(exc.code),
            "message": f"Webhook HTTP error: {exc.code}",
        }
    except Exception as exc:
        return {
            "attempted": True,
            "delivered": False,
            "status_code": None,
            "message": f"Webhook delivery failed: {exc}",
        }


def _run_rollback_command(command: str, timeout_seconds: int) -> dict[str, Any]:
    cmd = str(command or "").strip()
    if not cmd:
        return {
            "attempted": False,
            "succeeded": False,
            "returncode": None,
            "message": "Rollback command not provided",
        }

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
        )
        return {
            "attempted": True,
            "succeeded": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "stdout_tail": str(proc.stdout or "")[-800:],
            "stderr_tail": str(proc.stderr or "")[-800:],
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "message": (
                "Rollback command succeeded"
                if proc.returncode == 0
                else f"Rollback command failed with code {proc.returncode}"
            ),
        }
    except Exception as exc:
        return {
            "attempted": True,
            "succeeded": False,
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "message": f"Rollback command execution failed: {exc}",
        }


def run_incident_response_automation(
    *,
    run_dir: str | Path,
    runtime_watchdog_json: str | Path | None = None,
    runtime_gate_json: str | Path | None = None,
    go_live_summary_json: str | Path | None = None,
    critical_slo_breach_threshold: int = 1,
    incident_sla_breach_threshold: int = 1,
    dead_letter_critical_threshold: int = 25,
    webhook_url: str = "",
    webhook_timeout_seconds: int = 5,
    webhook_auth_token: str = "",
    webhook_auth_scheme: str = "Bearer",
    rollback_cmd: str = "",
    rollback_timeout_seconds: int = 120,
) -> dict[str, Any]:
    """
    Evaluate and execute incident response automation based on run metrics.
    """
    snapshot = build_ops_metrics_snapshot(
        run_dir=run_dir,
        runtime_watchdog_json=runtime_watchdog_json,
        runtime_gate_json=runtime_gate_json,
        go_live_summary_json=go_live_summary_json,
    )
    plan = evaluate_incident_response_plan(
        snapshot,
        critical_slo_breach_threshold=max(1, int(critical_slo_breach_threshold)),
        incident_sla_breach_threshold=max(1, int(incident_sla_breach_threshold)),
        dead_letter_critical_threshold=max(1, int(dead_letter_critical_threshold)),
    )

    actions: dict[str, Any] = {
        "webhook": {
            "attempted": False,
            "delivered": False,
            "status_code": None,
            "message": "No action required",
        },
        "rollback": {
            "attempted": False,
            "succeeded": False,
            "returncode": None,
            "message": "No action required",
        },
    }

    if plan.get("severity") == "critical":
        payload = {
            "event_type": "incident_response_automation_triggered",
            "severity": "critical",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "plan": plan,
            "snapshot": snapshot,
        }
        actions["webhook"] = _deliver_incident_webhook(
            webhook_url=webhook_url,
            payload=payload,
            timeout_seconds=max(1, int(webhook_timeout_seconds)),
            auth_token=webhook_auth_token,
            auth_scheme=webhook_auth_scheme,
        )
        actions["rollback"] = _run_rollback_command(
            rollback_cmd,
            timeout_seconds=max(1, int(rollback_timeout_seconds)),
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": bool(plan.get("ready")),
        "snapshot": snapshot,
        "plan": plan,
        "actions": actions,
    }
