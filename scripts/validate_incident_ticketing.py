#!/usr/bin/env python3
"""Validate that ack-SLA breaches trigger incident ticket creation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.incident_tracker import IncidentTracker  # noqa: E402
from utils.slo_monitor import SLOMonitor  # noqa: E402


class _RecordingTicketNotifier:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def notify(self, breach: dict) -> bool:
        self.events.append(dict(breach))
        return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ack-SLA to ticket workflow")
    parser.add_argument("--tmp-dir", default="results", help="Directory for validation artifacts")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--ack-sla-minutes",
        type=int,
        default=1,
        help="Ack-SLA threshold used for simulation",
    )
    return parser.parse_args()


def _build_report(args: argparse.Namespace) -> dict:
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    incident_tracker = IncidentTracker(
        events_path=tmp_dir / "incident_events_ticketing_validation.jsonl",
        run_id="ticketing_validation",
        ack_sla_minutes=max(1, int(args.ack_sla_minutes)),
    )
    notifier = _RecordingTicketNotifier()
    monitor = SLOMonitor(
        incident_tracker=incident_tracker,
        incident_ticket_notifier=notifier,
        max_data_quality_errors=0,
        max_stale_data_warnings=0,
    )

    monitor.record_data_quality_summary({"total_errors": 1, "stale_warnings": 0})
    incident_id = next(iter(incident_tracker._incidents.keys()))
    created_at = incident_tracker._incidents[incident_id].created_at
    breaches = monitor.check_incident_ack_sla(now=created_at + timedelta(minutes=2))
    status = monitor.get_status_snapshot()
    monitor.close()

    breach_names = [b.name for b in breaches]
    passed = (
        "incident_ack_sla_breach" in breach_names
        and status.get("ticketing", {}).get("attempts") == 1
        and status.get("ticketing", {}).get("created") == 1
        and len(notifier.events) == 1
    )
    return {
        "passed": passed,
        "breaches": breach_names,
        "ticketing": status.get("ticketing", {}),
        "incident": status.get("incidents", {}),
    }


def main() -> int:
    args = _parse_args()
    report = _build_report(args)
    print("ACK-SLA TICKETING VALIDATION")
    print(json.dumps(report, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
