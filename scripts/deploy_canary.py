#!/usr/bin/env python3
"""Run a command-based canary rollout with health checks and rollback automation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canary rollout automation")
    parser.add_argument(
        "--candidate-cmd",
        required=True,
        help="Candidate deployment command",
    )
    parser.add_argument(
        "--health-check-cmd",
        required=True,
        help="Post-deploy health check command",
    )
    parser.add_argument(
        "--promote-cmd",
        default="",
        help="Optional promotion command after successful health checks",
    )
    parser.add_argument(
        "--rollback-cmd",
        default="",
        help="Optional rollback command executed on failure",
    )
    parser.add_argument(
        "--candidate-timeout-seconds",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--health-timeout-seconds",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--promote-timeout-seconds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--rollback-timeout-seconds",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute commands; only emit rollout plan",
    )
    return parser.parse_args()


def _run_command(
    command: str,
    *,
    timeout_seconds: int,
    step_name: str,
) -> dict:
    args = shlex.split(command)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
        )
        return {
            "step": step_name,
            "passed": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "stdout_tail": str(proc.stdout or "")[-800:],
            "stderr_tail": str(proc.stderr or "")[-800:],
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "step": step_name,
            "passed": False,
            "returncode": -1,
            "stdout_tail": str(exc.stdout or "")[-800:],
            "stderr_tail": str(exc.stderr or "")[-800:],
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "message": f"Command timed out after {timeout_seconds}s",
        }
    except Exception as exc:
        return {
            "step": step_name,
            "passed": False,
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "message": f"Command failed to execute: {exc}",
        }


def _write_output(path: str | None, report: dict) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "ready": False,
        "steps": [],
    }

    if args.dry_run:
        report["ready"] = True
        report["steps"] = [
            {"step": "candidate", "command": args.candidate_cmd, "planned": True},
            {"step": "health_check", "command": args.health_check_cmd, "planned": True},
            {
                "step": "promote",
                "command": args.promote_cmd or "",
                "planned": bool(args.promote_cmd),
            },
            {
                "step": "rollback",
                "command": args.rollback_cmd or "",
                "planned": bool(args.rollback_cmd),
            },
        ]
        _write_output(args.output, report)
        print("CANARY ROLLOUT (dry-run)")
        print(json.dumps(report, indent=2))
        return 0

    candidate = _run_command(
        args.candidate_cmd,
        timeout_seconds=args.candidate_timeout_seconds,
        step_name="candidate",
    )
    report["steps"].append(candidate)
    if not candidate["passed"]:
        if args.rollback_cmd:
            report["steps"].append(
                _run_command(
                    args.rollback_cmd,
                    timeout_seconds=args.rollback_timeout_seconds,
                    step_name="rollback_after_candidate_failure",
                )
            )
        _write_output(args.output, report)
        print(json.dumps(report, indent=2))
        return 1

    health = _run_command(
        args.health_check_cmd,
        timeout_seconds=args.health_timeout_seconds,
        step_name="health_check",
    )
    report["steps"].append(health)
    if not health["passed"]:
        if args.rollback_cmd:
            report["steps"].append(
                _run_command(
                    args.rollback_cmd,
                    timeout_seconds=args.rollback_timeout_seconds,
                    step_name="rollback_after_health_failure",
                )
            )
        _write_output(args.output, report)
        print(json.dumps(report, indent=2))
        return 1

    if args.promote_cmd:
        promote = _run_command(
            args.promote_cmd,
            timeout_seconds=args.promote_timeout_seconds,
            step_name="promote",
        )
        report["steps"].append(promote)
        if not promote["passed"]:
            if args.rollback_cmd:
                report["steps"].append(
                    _run_command(
                        args.rollback_cmd,
                        timeout_seconds=args.rollback_timeout_seconds,
                        step_name="rollback_after_promote_failure",
                    )
                )
            _write_output(args.output, report)
            print(json.dumps(report, indent=2))
            return 1

    report["ready"] = True
    _write_output(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
