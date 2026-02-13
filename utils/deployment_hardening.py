"""
Deployment hardening checks for release readiness.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from utils.runtime_state import RuntimeStateStore


@dataclass
class DeploymentCheck:
    name: str
    passed: bool
    message: str
    severity: str = "critical"  # critical | warning

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
        }


def _git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return None
    return None


def run_deployment_preflight(
    repo_root: str | Path = ".",
    *,
    required_env_vars: Optional[List[str]] = None,
    run_rollback_drill: bool = False,
    rollback_drill_workdir: str | Path | None = None,
) -> dict:
    """
    Run deployment hardening checks.
    """
    root = Path(repo_root)
    required_env_vars = required_env_vars or []
    checks: List[DeploymentCheck] = []

    runbook_path = root / "docs" / "OPERATIONS_RUNBOOK.md"
    checks.append(
        DeploymentCheck(
            name="operations_runbook_present",
            passed=runbook_path.exists(),
            message=(
                "Operations runbook found"
                if runbook_path.exists()
                else f"Missing operations runbook: {runbook_path}"
            ),
        )
    )
    if runbook_path.exists():
        runbook_text = runbook_path.read_text(encoding="utf-8")
        has_drill = "Disaster Recovery Drill" in runbook_text
        checks.append(
            DeploymentCheck(
                name="runbook_contains_recovery_drill",
                passed=has_drill,
                severity="warning",
                message=(
                    "Runbook includes disaster recovery drill section"
                    if has_drill
                    else "Runbook missing 'Disaster Recovery Drill' section"
                ),
            )
        )

    rollback_script = root / "scripts" / "rollback_drill.py"
    checks.append(
        DeploymentCheck(
            name="rollback_drill_script_present",
            passed=rollback_script.exists(),
            severity="warning",
            message=(
                "Rollback drill script found"
                if rollback_script.exists()
                else f"Missing rollback drill script: {rollback_script}"
            ),
        )
    )

    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch is None:
        checks.append(
            DeploymentCheck(
                name="git_repository_detected",
                passed=False,
                severity="warning",
                message="Git repository not detected or git unavailable",
            )
        )
    else:
        checks.append(
            DeploymentCheck(
                name="branch_prefix",
                passed=branch.startswith("gr8monk3ys/"),
                severity="warning",
                message=(
                    f"Branch '{branch}' uses required prefix"
                    if branch.startswith("gr8monk3ys/")
                    else f"Branch '{branch}' is missing 'gr8monk3ys/' prefix"
                ),
            )
        )

        status = _git(root, "status", "--porcelain")
        dirty = bool(status)
        checks.append(
            DeploymentCheck(
                name="clean_worktree",
                passed=not dirty,
                message="Working tree clean" if not dirty else "Working tree has uncommitted changes",
            )
        )

        tracked_env = _git(root, "ls-files", ".env")
        checks.append(
            DeploymentCheck(
                name="env_not_tracked",
                passed=not bool(tracked_env),
                message=(
                    ".env is not tracked"
                    if not tracked_env
                    else ".env is tracked in git (security risk)"
                ),
            )
        )

    missing_env = [name for name in required_env_vars if not os.environ.get(name)]
    checks.append(
        DeploymentCheck(
            name="required_env_vars_present",
            passed=len(missing_env) == 0,
            message=(
                "All required environment variables are present"
                if not missing_env
                else f"Missing environment variables: {', '.join(missing_env)}"
            ),
        )
    )

    if run_rollback_drill:
        drill = run_runtime_rollback_drill(workdir=rollback_drill_workdir)
        checks.append(
            DeploymentCheck(
                name="runtime_rollback_drill",
                passed=bool(drill.get("passed")),
                message=str(drill.get("message", "rollback drill completed")),
            )
        )

    ready = all(c.passed for c in checks if c.severity == "critical")
    return {
        "ready": ready,
        "checks": [c.to_dict() for c in checks],
    }


def run_runtime_rollback_drill(
    workdir: str | Path | None = None,
) -> dict:
    """
    Execute deterministic runtime-state rollback drill.

    Drill plan:
    1) Persist a runtime snapshot with kill-switch state
    2) Corrupt primary snapshot
    3) Verify load() recovers via replay/backup path
    """

    class _DrillPositionManager:
        async def export_state(self):
            return {"ownership": {"AAPL": {"strategy": "rollback_drill"}}}

    base_dir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="rollback_drill_"))
    drill_dir = base_dir / ".rollback_drill"
    drill_dir.mkdir(parents=True, exist_ok=True)
    state_path = drill_dir / "runtime_state.json"
    store = RuntimeStateStore(str(state_path))

    async def _execute() -> tuple[bool, str]:
        await store.save(
            position_manager=_DrillPositionManager(),
            active_strategies={"rollback_drill": "running"},
            gateway_state={
                "halt_reason": "rollback_drill_halt",
                "trading_halted_until": "2099-01-01T00:00:00",
            },
        )
        # Corrupt primary snapshot to force recovery path.
        state_path.write_text("{corrupt", encoding="utf-8")
        recovered = await store.load()
        if not recovered:
            return False, "Recovery returned no runtime state"
        reason = (recovered.gateway_state or {}).get("halt_reason")
        if reason != "rollback_drill_halt":
            return False, f"Recovered state missing expected halt reason: {reason}"
        return True, "Recovered runtime snapshot from fallback source after primary corruption"

    try:
        passed, message = asyncio.run(_execute())
    except Exception as e:
        return {
            "passed": False,
            "message": f"Rollback drill execution failed: {e}",
            "state_path": str(state_path),
        }

    return {
        "passed": passed,
        "message": message,
        "state_path": str(state_path),
    }
