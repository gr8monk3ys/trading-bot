"""
Compliance and governance gate for promotion beyond paper trading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class GovernanceCheck:
    name: str
    passed: bool
    message: str
    severity: str = "critical"
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "details": self.details or {},
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _parse_iso_timestamp(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def run_governance_gate(
    *,
    repo_root: str | Path = ".",
    mode: str = "paper",
    approval_path: str | Path = "results/governance/live_approval.json",
    policy_doc_path: str | Path = "docs/COMPLIANCE_GOVERNANCE.md",
) -> dict[str, Any]:
    """
    Evaluate governance readiness for paper/live trading modes.
    """
    root = Path(repo_root).resolve()
    normalized_mode = str(mode or "paper").strip().lower()
    if normalized_mode not in {"paper", "live"}:
        normalized_mode = "paper"

    approval_file = Path(approval_path)
    if not approval_file.is_absolute():
        approval_file = (root / approval_file).resolve()

    policy_file = Path(policy_doc_path)
    if not policy_file.is_absolute():
        policy_file = (root / policy_file).resolve()

    checks: list[GovernanceCheck] = []
    checks.append(
        GovernanceCheck(
            name="compliance_policy_doc_present",
            passed=policy_file.exists(),
            message=(
                f"Compliance governance policy found: {policy_file}"
                if policy_file.exists()
                else f"Missing compliance governance policy: {policy_file}"
            ),
        )
    )

    if normalized_mode == "paper":
        checks.append(
            GovernanceCheck(
                name="live_approval_not_required_for_paper",
                passed=True,
                severity="warning",
                message="Paper mode selected; live approval artifact is not required",
            )
        )
        ready = all(check.passed for check in checks if check.severity == "critical")
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": normalized_mode,
            "ready": ready,
            "checks": [check.to_dict() for check in checks],
        }

    approval = _load_json(approval_file)
    checks.append(
        GovernanceCheck(
            name="live_approval_artifact_present",
            passed=approval_file.exists(),
            message=(
                f"Live approval artifact found: {approval_file}"
                if approval_file.exists()
                else f"Missing live approval artifact: {approval_file}"
            ),
        )
    )

    approvers = approval.get("approvers", [])
    if not isinstance(approvers, list):
        approvers = []
    approver_names = sorted({str(value).strip() for value in approvers if str(value).strip()})
    checks.append(
        GovernanceCheck(
            name="dual_approval_present",
            passed=len(approver_names) >= 2,
            message=(
                f"Dual approval present ({len(approver_names)} approvers)"
                if len(approver_names) >= 2
                else "Dual approval missing; provide at least two unique approvers"
            ),
            details={"approvers": approver_names},
        )
    )

    required_flags = {
        "kyc_attestation": "KYC attestation",
        "aml_attestation": "AML attestation",
        "best_execution_policy_ack": "Best-execution policy acknowledgment",
    }
    missing_flags = [
        label for key, label in required_flags.items() if not bool(approval.get(key, False))
    ]
    checks.append(
        GovernanceCheck(
            name="compliance_attestations_present",
            passed=len(missing_flags) == 0,
            message=(
                "All compliance attestations are present"
                if not missing_flags
                else f"Missing compliance attestations: {', '.join(missing_flags)}"
            ),
        )
    )

    notional = approval.get("max_notional_usd")
    try:
        max_notional = float(notional)
    except (TypeError, ValueError):
        max_notional = 0.0
    checks.append(
        GovernanceCheck(
            name="max_notional_limit_present",
            passed=max_notional > 0,
            message=(
                f"Live notional cap set to ${max_notional:,.2f}"
                if max_notional > 0
                else "Missing or invalid max_notional_usd in live approval artifact"
            ),
            details={"max_notional_usd": max_notional},
        )
    )

    approved_at = _parse_iso_timestamp(str(approval.get("approved_at", "")).strip())
    expires_at = _parse_iso_timestamp(str(approval.get("expires_at", "")).strip())
    checks.append(
        GovernanceCheck(
            name="live_approval_timestamps_valid",
            passed=approved_at is not None and expires_at is not None,
            message=(
                "Approval timestamps are valid ISO-8601 values"
                if approved_at is not None and expires_at is not None
                else "approved_at/expires_at are missing or invalid in live approval artifact"
            ),
            details={
                "approved_at": approved_at.isoformat() if approved_at else None,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )
    )

    now = datetime.now(timezone.utc)
    not_expired = bool(expires_at and expires_at > now)
    checks.append(
        GovernanceCheck(
            name="live_approval_not_expired",
            passed=not_expired,
            message=(
                f"Live approval valid until {expires_at.isoformat()}"
                if not_expired and expires_at
                else "Live approval expired or missing expiry timestamp"
            ),
        )
    )

    strategy_allowlist = approval.get("strategy_allowlist", [])
    if not isinstance(strategy_allowlist, list):
        strategy_allowlist = []
    normalized_allowlist = sorted(
        {str(item).strip() for item in strategy_allowlist if str(item).strip()}
    )
    checks.append(
        GovernanceCheck(
            name="strategy_allowlist_present",
            passed=len(normalized_allowlist) > 0,
            message=(
                f"Strategy allowlist configured ({len(normalized_allowlist)} strategies)"
                if normalized_allowlist
                else "Missing strategy_allowlist in live approval artifact"
            ),
            details={"strategy_allowlist": normalized_allowlist},
        )
    )

    ready = all(check.passed for check in checks if check.severity == "critical")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": normalized_mode,
        "ready": ready,
        "checks": [check.to_dict() for check in checks],
        "approval_path": str(approval_file),
        "policy_doc_path": str(policy_file),
    }
