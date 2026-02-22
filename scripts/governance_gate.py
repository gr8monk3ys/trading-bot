#!/usr/bin/env python3
"""Run compliance/governance promotion gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.governance_gate import run_governance_gate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compliance and governance gate")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--mode",
        default="paper",
        choices=["paper", "live"],
        help="Trading mode under evaluation",
    )
    parser.add_argument(
        "--approval-path",
        default="results/governance/live_approval.json",
    )
    parser.add_argument(
        "--policy-doc-path",
        default="docs/COMPLIANCE_GOVERNANCE.md",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_governance_gate(
        repo_root=args.repo_root,
        mode=args.mode,
        approval_path=args.approval_path,
        policy_doc_path=args.policy_doc_path,
    )

    print("=" * 72)
    print("GOVERNANCE GATE")
    print("=" * 72)
    print(f"Mode: {report.get('mode')}")
    print(f"Ready: {'YES' if report.get('ready') else 'NO'}")
    for check in report.get("checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        severity = str(check.get("severity", "critical")).upper()
        print(f"[{status}][{severity}] {check.get('name')}: {check.get('message')}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if report.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
