#!/usr/bin/env python3
"""
Run runtime industrial-readiness gate checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime_industrial_gate import run_runtime_industrial_gate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime industrial-readiness gate")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--incident-ownership-doc",
        default="docs/INCIDENT_RESPONSE_OWNERSHIP.md",
    )
    parser.add_argument(
        "--incident-escalation-doc",
        default="docs/INCIDENT_ESCALATION_ROSTER.md",
    )
    parser.add_argument(
        "--run-chaos-drill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run-ticket-drill", action="store_true")
    parser.add_argument(
        "--ticket-webhook-url",
        default=str(os.environ.get("INCIDENT_TICKETING_WEBHOOK_URL", "") or "").strip(),
    )
    parser.add_argument(
        "--ticket-artifact-dir",
        default="results/validation",
    )
    parser.add_argument(
        "--ticket-require-delivery",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ticket-require-non-test-target",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ticket-require-response-links",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ticket-runbook-url",
        default=str(os.environ.get("INCIDENT_RESPONSE_RUNBOOK_URL", "") or "").strip(),
    )
    parser.add_argument(
        "--ticket-escalation-roster-url",
        default=str(os.environ.get("INCIDENT_ESCALATION_ROSTER_URL", "") or "").strip(),
    )
    parser.add_argument("--ticket-max-age-hours", type=int, default=72)
    parser.add_argument("--run-failover-probe", action="store_true")
    parser.add_argument("--failover-paper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = asyncio.run(
        run_runtime_industrial_gate(
            repo_root=args.repo_root,
            ownership_doc=args.incident_ownership_doc,
            escalation_doc=args.incident_escalation_doc,
            run_chaos_drill=bool(args.run_chaos_drill),
            run_ticket_drill=bool(args.run_ticket_drill),
            ticket_webhook_url=str(args.ticket_webhook_url or "").strip(),
            ticket_artifact_dir=args.ticket_artifact_dir,
            ticket_require_delivery=bool(args.ticket_require_delivery),
            ticket_require_non_test_target=bool(args.ticket_require_non_test_target),
            ticket_require_response_links=bool(args.ticket_require_response_links),
            ticket_runbook_url=str(args.ticket_runbook_url or "").strip(),
            ticket_escalation_roster_url=str(args.ticket_escalation_roster_url or "").strip(),
            ticket_max_age_hours=max(1, int(args.ticket_max_age_hours)),
            run_failover_probe=bool(args.run_failover_probe),
            failover_paper=bool(args.failover_paper),
        )
    )

    print("=" * 72)
    print("RUNTIME INDUSTRIAL READINESS GATE")
    print("=" * 72)
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
