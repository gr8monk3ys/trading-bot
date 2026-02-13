#!/usr/bin/env python3
"""
Acknowledge an operational incident from a run artifact stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.incident_tracker import IncidentTracker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acknowledge run incident")
    parser.add_argument("--incident-id", required=True, help="Incident identifier")
    parser.add_argument("--ack-by", required=True, help="Responder/operator name")
    parser.add_argument("--notes", default="", help="Acknowledgment notes")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID under artifacts directory (required unless --events-path provided)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="results/runs",
        help="Base run artifacts directory",
    )
    parser.add_argument(
        "--events-path",
        default=None,
        help="Direct path to incident_events.jsonl (overrides --run-id resolution)",
    )
    return parser.parse_args()


def _resolve_events_path(args: argparse.Namespace) -> Path:
    if args.events_path:
        return Path(args.events_path)
    if not args.run_id:
        raise ValueError("Either --run-id or --events-path must be provided")
    return Path(args.artifacts_dir) / args.run_id / "incident_events.jsonl"


def main() -> None:
    args = _parse_args()
    events_path = _resolve_events_path(args)

    tracker = IncidentTracker(events_path=events_path)
    try:
        success = tracker.acknowledge(
            incident_id=args.incident_id,
            acknowledged_by=args.ack_by,
            notes=args.notes,
        )
    finally:
        tracker.close()

    if not success:
        print(f"Incident not found: {args.incident_id}")
        raise SystemExit(1)

    print(f"Acknowledged incident {args.incident_id} by {args.ack_by}")


if __name__ == "__main__":
    main()
