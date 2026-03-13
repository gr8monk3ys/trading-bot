#!/usr/bin/env python3
"""Run repository secrets leak + rotation audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.secrets_audit import run_secrets_audit, sanitize_audit_report  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secrets rotation and leak audit")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--inventory-path",
        default="docs/SECRETS_ROTATION_INVENTORY.json",
    )
    parser.add_argument(
        "--default-max-age-days",
        type=int,
        default=90,
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_secrets_audit(
        repo_root=args.repo_root,
        inventory_path=args.inventory_path,
        default_max_age_days=max(1, int(args.default_max_age_days)),
    )
    safe_report = sanitize_audit_report(report)

    print("=" * 72)
    print("SECRETS AUDIT")
    print("=" * 72)
    print(f"Ready: {'YES' if safe_report.get('ready') else 'NO'}")
    for check in safe_report.get("checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        print(f"[{status}] {check.get('name')}: {check.get('message')}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(safe_report, indent=2), encoding="utf-8")

    return 0 if report.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
