#!/usr/bin/env python3
"""Validate incident ownership/escalation docs are fully populated."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from incident_contacts import validate_incident_contacts  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate incident ownership/escalation contacts")
    parser.add_argument(
        "--ownership-doc",
        default="docs/INCIDENT_RESPONSE_OWNERSHIP.md",
        help="Path to incident ownership document",
    )
    parser.add_argument(
        "--escalation-doc",
        default="docs/INCIDENT_ESCALATION_ROSTER.md",
        help="Path to escalation roster document",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write validation JSON",
    )
    return parser.parse_args()


def _print_report(report: dict) -> None:
    print("=" * 72)
    print("INCIDENT CONTACT VALIDATION")
    print("=" * 72)
    print(f"Ownership Doc: {report.get('ownership_doc')}")
    print(f"Escalation Doc: {report.get('escalation_doc')}")
    print(f"Valid: {'YES' if report.get('valid') else 'NO'}")
    print(f"Placeholder Findings: {int(report.get('placeholder_count', 0))}")
    if report.get("findings"):
        print("Findings:")
        for finding in report["findings"]:
            print(
                f"  - {finding.get('file')}:{int(finding.get('line', 0))} "
                f"{finding.get('token')} | {finding.get('context')}"
            )


def main() -> int:
    args = _parse_args()
    report = validate_incident_contacts(
        ownership_doc=args.ownership_doc,
        escalation_doc=args.escalation_doc,
    )
    _print_report(report)

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if bool(report.get("valid")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
