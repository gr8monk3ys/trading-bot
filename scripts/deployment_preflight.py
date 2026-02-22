#!/usr/bin/env python3
"""
Deployment preflight checks for industrial-grade release hardening.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.deployment_hardening import run_deployment_preflight


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deployment preflight checks")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root path",
    )
    parser.add_argument(
        "--required-env",
        default="",
        help="Comma-separated required env vars",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output file path",
    )
    parser.add_argument(
        "--run-rollback-drill",
        action="store_true",
        help="Execute runtime rollback drill as part of preflight",
    )
    parser.add_argument(
        "--rollback-drill-workdir",
        default=None,
        help="Optional workdir for rollback drill artifacts",
    )
    parser.add_argument(
        "--run-secrets-audit",
        action="store_true",
        help="Run secrets leak/rotation audit as part of preflight",
    )
    parser.add_argument(
        "--secrets-inventory-path",
        default="docs/SECRETS_ROTATION_INVENTORY.json",
        help="Path to secrets rotation inventory JSON",
    )
    parser.add_argument(
        "--secrets-default-max-age-days",
        type=int,
        default=90,
        help="Default max age when inventory entries omit max_age_days",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    required_env = [x.strip() for x in args.required_env.split(",") if x.strip()]
    report = run_deployment_preflight(
        repo_root=args.repo_root,
        required_env_vars=required_env,
        run_rollback_drill=args.run_rollback_drill,
        rollback_drill_workdir=args.rollback_drill_workdir,
        run_secrets_audit_check=args.run_secrets_audit,
        secrets_inventory_path=args.secrets_inventory_path,
        secrets_default_max_age_days=max(1, int(args.secrets_default_max_age_days)),
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 72)
    print("DEPLOYMENT PREFLIGHT")
    print("=" * 72)
    print(f"Ready: {'YES' if report['ready'] else 'NO'}")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['message']}")

    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
