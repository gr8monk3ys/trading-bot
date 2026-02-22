#!/usr/bin/env python3
"""Replay failed SLO/ticket notifications from dead-letter JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RISK_PARAMS  # noqa: E402
from utils.run_artifacts import read_jsonl, to_jsonable  # noqa: E402
from utils.slo_alerting import (  # noqa: E402
    build_incident_ticket_notifier,
    build_slo_alert_notifier,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay notification dead-letter entries")
    parser.add_argument(
        "--dead-letter-path",
        required=True,
        help="Path to notification dead-letter JSONL file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON report path",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite dead-letter file in place with only remaining failed entries",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Maximum records to process (0 means process all)",
    )
    parser.add_argument(
        "--source",
        default="scripts.replay_notification_dead_letters",
        help="Source label used in outgoing replayed notifications",
    )
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(dict(row)), separators=(",", ":")) + "\n")


def _replay_dead_letters(
    records: list[dict[str, Any]],
    *,
    alert_notifier: Any | None,
    incident_ticket_notifier: Any | None,
    max_records: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    limit = max(0, int(max_records))
    processed = 0
    replayed = 0
    failed = 0
    skipped = 0
    remaining: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}

    def _bump_reason(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for row in records:
        if limit and processed >= limit:
            remaining.append(dict(row))
            continue

        if str(row.get("event_type", "")).strip() != "notification_dead_letter":
            remaining.append(dict(row))
            continue

        processed += 1
        channel = str(row.get("channel", "")).strip().lower()
        breach = dict(row.get("breach", {}) or {})

        notifier = None
        if channel == "slo_alert":
            notifier = alert_notifier
        elif channel == "incident_ticket":
            notifier = incident_ticket_notifier
        else:
            reason = "unknown_channel"
            failed += 1
            _bump_reason(reason)
            updated = dict(row)
            updated["last_replay_attempt_at"] = datetime.utcnow().isoformat()
            updated["replay_attempts"] = int(updated.get("replay_attempts", 0) or 0) + 1
            updated["last_replay_error"] = reason
            remaining.append(updated)
            continue

        if notifier is None:
            reason = "notifier_unavailable"
            skipped += 1
            _bump_reason(reason)
            updated = dict(row)
            updated["last_replay_attempt_at"] = datetime.utcnow().isoformat()
            updated["replay_attempts"] = int(updated.get("replay_attempts", 0) or 0) + 1
            updated["last_replay_error"] = reason
            remaining.append(updated)
            continue

        try:
            result = notifier.notify(breach)
        except Exception as exc:  # pragma: no cover - defensive path
            reason = f"exception:{type(exc).__name__}"
            failed += 1
            _bump_reason(reason)
            updated = dict(row)
            updated["last_replay_attempt_at"] = datetime.utcnow().isoformat()
            updated["replay_attempts"] = int(updated.get("replay_attempts", 0) or 0) + 1
            updated["last_replay_error"] = str(exc)
            remaining.append(updated)
            continue

        if result is True:
            replayed += 1
            continue

        if result is None:
            reason = "notifier_skipped"
            skipped += 1
        else:
            reason = "notifier_returned_false"
            failed += 1
        _bump_reason(reason)
        updated = dict(row)
        updated["last_replay_attempt_at"] = datetime.utcnow().isoformat()
        updated["replay_attempts"] = int(updated.get("replay_attempts", 0) or 0) + 1
        updated["last_replay_error"] = reason
        remaining.append(updated)

    report = {
        "records_total": len(records),
        "records_processed": processed,
        "records_replayed": replayed,
        "records_failed": failed,
        "records_skipped": skipped,
        "records_remaining": len(remaining),
        "reasons": reasons,
    }
    return report, remaining


def _build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dead_letter_path = Path(args.dead_letter_path)
    rows = read_jsonl(dead_letter_path)
    slo_notifier = build_slo_alert_notifier(RISK_PARAMS, source=args.source)
    incident_notifier = build_incident_ticket_notifier(RISK_PARAMS, source=args.source)
    report, remaining = _replay_dead_letters(
        rows,
        alert_notifier=slo_notifier,
        incident_ticket_notifier=incident_notifier,
        max_records=max(0, int(args.max_records)),
    )
    report["dead_letter_path"] = str(dead_letter_path)
    report["in_place"] = bool(args.in_place)
    return report, remaining


def main() -> int:
    args = _parse_args()
    dead_letter_path = Path(args.dead_letter_path)
    report, remaining = _build_report(args)

    if args.in_place:
        _write_jsonl(dead_letter_path, remaining)

    print("NOTIFICATION DEAD-LETTER REPLAY")
    print(json.dumps(report, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if report.get("records_remaining", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
