"""
Runtime industrial-readiness gate checks.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from brokers.broker_interface import BrokerStatus
from incident_contacts import validate_incident_contacts
from utils.chaos_drills import run_chaos_drills
from utils.live_broker_factory import create_live_broker, shutdown_live_broker_failover


@dataclass
class RuntimeGateCheck:
    name: str
    passed: bool
    message: str
    severity: str = "critical"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
        }


def _resolve_path(repo_root: Path, candidate: str | Path) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _check_incident_contacts(
    *,
    repo_root: Path,
    ownership_doc: str | Path,
    escalation_doc: str | Path,
) -> RuntimeGateCheck:
    report = validate_incident_contacts(
        ownership_doc=str(_resolve_path(repo_root, ownership_doc)),
        escalation_doc=str(_resolve_path(repo_root, escalation_doc)),
    )
    passed = bool(report.get("valid"))
    findings = list(report.get("findings", []))
    return RuntimeGateCheck(
        name="incident_contacts",
        passed=passed,
        message=(
            "Incident ownership/escalation docs validated with no placeholders"
            if passed
            else f"Incident docs contain unresolved placeholders ({len(findings)} findings)"
        ),
        details={
            "ownership_doc": report.get("ownership_doc"),
            "escalation_doc": report.get("escalation_doc"),
            "placeholder_count": int(report.get("placeholder_count", 0) or 0),
            "findings": findings,
        },
    )


async def _check_chaos_drills() -> RuntimeGateCheck:
    report = await run_chaos_drills()
    checks = list(report.get("checks", []))
    failing = [check.get("name") for check in checks if not check.get("passed")]
    passed = bool(report.get("passed"))
    return RuntimeGateCheck(
        name="chaos_drills",
        passed=passed,
        message=(
            "Chaos drills passed"
            if passed
            else f"Chaos drills failed: {', '.join(str(name) for name in failing)}"
        ),
        details={"report": report},
    )


def _append_bool_flag(command: List[str], flag_name: str, enabled: bool) -> None:
    command.append(f"--{flag_name}" if enabled else f"--no-{flag_name}")


def _check_staging_ticket_drill(
    *,
    repo_root: Path,
    webhook_url: str,
    artifact_dir: str | Path,
    require_delivery: bool,
    require_non_test_target: bool,
    require_response_links: bool,
    runbook_url: str,
    escalation_roster_url: str,
    max_age_hours: int,
) -> RuntimeGateCheck:
    webhook = str(webhook_url or "").strip()
    if not webhook:
        return RuntimeGateCheck(
            name="staging_incident_ticket_drill",
            passed=False,
            message=(
                "Missing webhook URL. Set INCIDENT_TICKETING_WEBHOOK_URL or pass --ticket-webhook-url."
            ),
        )

    artifact_path = _resolve_path(repo_root, artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    report_path = artifact_path / "incident_ticket_drill_report.json"
    gate_path = artifact_path / "incident_ticket_drill_evidence_gate.json"

    drill_cmd = [
        sys.executable,
        "scripts/staging_incident_ticket_drill.py",
        "--webhook-url",
        webhook,
        "--artifact-dir",
        str(artifact_path),
        "--output",
        str(report_path),
        "--source",
        "scripts.runtime_industrial_gate",
    ]
    if runbook_url.strip():
        drill_cmd.extend(["--runbook-url", runbook_url.strip()])
    if escalation_roster_url.strip():
        drill_cmd.extend(["--escalation-roster-url", escalation_roster_url.strip()])
    if require_delivery:
        drill_cmd.append("--require-delivery")
    if require_non_test_target:
        drill_cmd.append("--require-non-test-target")

    drill_proc = subprocess.run(
        drill_cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if drill_proc.returncode != 0:
        failure_output = (drill_proc.stderr or drill_proc.stdout or "").strip()
        return RuntimeGateCheck(
            name="staging_incident_ticket_drill",
            passed=False,
            message="Staging incident ticket drill execution failed",
            details={
                "returncode": drill_proc.returncode,
                "output_tail": failure_output[-800:],
                "command": drill_cmd,
            },
        )

    evidence_cmd = [
        sys.executable,
        "scripts/validate_incident_ticket_drill_evidence.py",
        "--report-path",
        str(report_path),
        "--max-age-hours",
        str(max(1, int(max_age_hours))),
        "--output",
        str(gate_path),
    ]
    _append_bool_flag(evidence_cmd, "require-delivery", bool(require_delivery))
    _append_bool_flag(evidence_cmd, "require-non-test-target", bool(require_non_test_target))
    _append_bool_flag(evidence_cmd, "require-response-links", bool(require_response_links))

    evidence_proc = subprocess.run(
        evidence_cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if evidence_proc.returncode != 0:
        failure_output = (evidence_proc.stderr or evidence_proc.stdout or "").strip()
        return RuntimeGateCheck(
            name="staging_incident_ticket_drill",
            passed=False,
            message="Incident ticket drill evidence gate failed",
            details={
                "returncode": evidence_proc.returncode,
                "output_tail": failure_output[-800:],
                "report_path": str(report_path),
                "gate_path": str(gate_path),
            },
        )

    return RuntimeGateCheck(
        name="staging_incident_ticket_drill",
        passed=True,
        message="Staging incident ticket drill passed with evidence gate validation",
        details={
            "report_path": str(report_path),
            "gate_path": str(gate_path),
            "webhook_url": webhook,
        },
    )


async def _check_multi_broker_failover_probe(
    *,
    paper: bool,
    source: str,
) -> RuntimeGateCheck:
    manager = None
    try:
        broker, manager = await create_live_broker(paper=paper, source=source)
        if manager is None:
            return RuntimeGateCheck(
                name="multi_broker_runtime_failover_probe",
                passed=False,
                message=(
                    "Multi-broker manager unavailable. Ensure MULTI_BROKER_ENABLED=true and IB backup connectivity."
                ),
            )

        account = await broker.get_account()
        account_id = str(getattr(account, "id", getattr(account, "account_id", "")) or "").strip()

        await manager._check_broker_health(manager._primary)
        backup_health_ok = True
        for backup in list(getattr(manager, "_backups", []) or []):
            await manager._check_broker_health(backup)
            health = manager.get_broker_health().get(backup.name)
            backup_health_ok = backup_health_ok and bool(
                health
                and health.status == BrokerStatus.CONNECTED
                and health.consecutive_failures == 0
            )

        primary_health = manager._broker_health[manager._primary.name]
        primary_health.consecutive_failures = int(manager.failure_threshold)
        primary_health.status = BrokerStatus.DISCONNECTED
        primary_health.error_message = "runtime probe simulated primary outage"
        await manager._evaluate_failover()
        failover_triggered = bool(manager.is_failed_over)

        primary_health.consecutive_failures = 0
        primary_health.status = BrokerStatus.CONNECTED
        primary_health.error_message = None
        for _ in range(int(manager.recovery_threshold)):
            await manager._evaluate_failover()
        failback_triggered = not bool(manager.is_failed_over)

        passed = bool(account_id) and backup_health_ok and failover_triggered and failback_triggered
        return RuntimeGateCheck(
            name="multi_broker_runtime_failover_probe",
            passed=passed,
            message=(
                "Live multi-broker connectivity probe passed (including failover/failback control-plane simulation)"
                if passed
                else "Live multi-broker connectivity probe failed"
            ),
            details={
                "account_id": account_id,
                "backup_health_ok": backup_health_ok,
                "failover_triggered": failover_triggered,
                "failback_triggered": failback_triggered,
                "active_broker": manager.active_broker.name,
            },
        )
    except Exception as exc:
        return RuntimeGateCheck(
            name="multi_broker_runtime_failover_probe",
            passed=False,
            message=f"Live multi-broker connectivity probe raised: {exc}",
        )
    finally:
        await shutdown_live_broker_failover(manager)


async def run_runtime_industrial_gate(
    *,
    repo_root: str | Path = ".",
    ownership_doc: str | Path = "docs/INCIDENT_RESPONSE_OWNERSHIP.md",
    escalation_doc: str | Path = "docs/INCIDENT_ESCALATION_ROSTER.md",
    run_chaos_drill: bool = True,
    run_ticket_drill: bool = False,
    ticket_webhook_url: str = "",
    ticket_artifact_dir: str | Path = "results/validation",
    ticket_require_delivery: bool = True,
    ticket_require_non_test_target: bool = True,
    ticket_require_response_links: bool = True,
    ticket_runbook_url: str = "",
    ticket_escalation_roster_url: str = "",
    ticket_max_age_hours: int = 72,
    run_failover_probe: bool = False,
    failover_paper: bool = True,
    failover_source: str = "scripts.runtime_industrial_gate",
) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    checks: List[RuntimeGateCheck] = []

    checks.append(
        _check_incident_contacts(
            repo_root=root,
            ownership_doc=ownership_doc,
            escalation_doc=escalation_doc,
        )
    )

    if run_chaos_drill:
        checks.append(await _check_chaos_drills())
    else:
        checks.append(
            RuntimeGateCheck(
                name="chaos_drills",
                passed=True,
                severity="warning",
                message="Chaos drill check skipped by configuration",
            )
        )

    if run_ticket_drill:
        checks.append(
            _check_staging_ticket_drill(
                repo_root=root,
                webhook_url=ticket_webhook_url,
                artifact_dir=ticket_artifact_dir,
                require_delivery=ticket_require_delivery,
                require_non_test_target=ticket_require_non_test_target,
                require_response_links=ticket_require_response_links,
                runbook_url=ticket_runbook_url,
                escalation_roster_url=ticket_escalation_roster_url,
                max_age_hours=ticket_max_age_hours,
            )
        )
    else:
        checks.append(
            RuntimeGateCheck(
                name="staging_incident_ticket_drill",
                passed=True,
                severity="warning",
                message="Staging incident ticket drill skipped by configuration",
            )
        )

    if run_failover_probe:
        checks.append(
            await _check_multi_broker_failover_probe(
                paper=failover_paper,
                source=failover_source,
            )
        )
    else:
        checks.append(
            RuntimeGateCheck(
                name="multi_broker_runtime_failover_probe",
                passed=True,
                severity="warning",
                message="Live multi-broker failover probe skipped by configuration",
            )
        )

    ready = all(check.passed for check in checks if check.severity == "critical")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "checks": [check.to_dict() for check in checks],
    }
