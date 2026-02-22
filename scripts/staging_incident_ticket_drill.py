#!/usr/bin/env python3
"""Run a staging-grade incident ticket webhook drill and persist evidence artifacts."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.incident_tracker import IncidentTracker  # noqa: E402
from utils.slo_alerting import WebhookIncidentTicketNotifier  # noqa: E402
from utils.slo_monitor import SLOMonitor  # noqa: E402


class _RecordingTicketNotifier:
    """Wrap a notifier to retain emitted payloads for evidence artifacts."""

    def __init__(self, delegate: WebhookIncidentTicketNotifier) -> None:
        self._delegate = delegate
        self.events: list[dict] = []

    def notify(self, breach: dict) -> bool | None:
        self.events.append(dict(breach))
        return self._delegate.notify(breach)


def _sanitize_webhook_url(raw_url: str) -> str:
    parsed = urlsplit(raw_url.strip())
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _is_non_test_webhook_target(raw_url: str) -> bool:
    parsed = urlsplit(str(raw_url or "").strip())
    if parsed.scheme.lower() != "https":
        return False

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False
    if hostname in {"localhost"} or hostname.endswith(".localhost") or hostname.endswith(".local"):
        return False
    if hostname.endswith(".test") or hostname.endswith(".example") or hostname.endswith(".invalid"):
        return False

    try:
        host_ip = ipaddress.ip_address(hostname)
    except ValueError:
        return True

    return not any(
        (
            host_ip.is_private,
            host_ip.is_loopback,
            host_ip.is_link_local,
            host_ip.is_reserved,
            host_ip.is_unspecified,
            host_ip.is_multicast,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real incident-ticket webhook drill in staging and persist evidence",
    )
    parser.add_argument(
        "--webhook-url",
        required=True,
        help="Incident-ticket webhook URL to test in staging",
    )
    parser.add_argument(
        "--artifact-dir",
        default="results/validation",
        help="Directory for evidence artifacts",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional report output path (default: artifact-dir with timestamped filename)",
    )
    parser.add_argument(
        "--ack-sla-minutes",
        type=int,
        default=1,
        help="Ack-SLA threshold used for simulation",
    )
    parser.add_argument("--timeout-seconds", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.5)
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("INCIDENT_TICKETING_AUTH_TOKEN", "").strip(),
        help="Optional auth token for webhook",
    )
    parser.add_argument(
        "--auth-scheme",
        default=str(os.environ.get("INCIDENT_TICKETING_AUTH_SCHEME", "Bearer")).strip(),
        help="Optional auth scheme for webhook",
    )
    parser.add_argument(
        "--source",
        default="scripts.staging_incident_ticket_drill",
        help="Source tag included in webhook payload",
    )
    parser.add_argument(
        "--runbook-url",
        default=os.environ.get("INCIDENT_RESPONSE_RUNBOOK_URL", "").strip(),
        help="Incident response runbook URL to include in ticket payload",
    )
    parser.add_argument(
        "--escalation-roster-url",
        default=os.environ.get("INCIDENT_ESCALATION_ROSTER_URL", "").strip(),
        help="Escalation roster URL to include in ticket payload",
    )
    parser.add_argument(
        "--require-delivery",
        action="store_true",
        help="Fail the drill if webhook delivery does not succeed",
    )
    parser.add_argument(
        "--require-non-test-target",
        action="store_true",
        help="Fail the drill if webhook target is not a non-test HTTPS endpoint",
    )
    return parser.parse_args()


def _build_report(args: argparse.Namespace) -> dict:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    drill_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    incident_events_path = artifact_dir / f"incident_events_ticket_drill_{drill_id}.jsonl"
    dead_letter_path = artifact_dir / f"notification_dead_letters_ticket_drill_{drill_id}.jsonl"

    notifier = _RecordingTicketNotifier(
        WebhookIncidentTicketNotifier(
            webhook_url=args.webhook_url,
            timeout_seconds=max(1, int(args.timeout_seconds)),
            source=args.source,
            max_retries=max(0, int(args.max_retries)),
            retry_backoff_seconds=max(0.0, float(args.retry_backoff_seconds)),
            auth_token=str(args.auth_token or "").strip(),
            auth_scheme=str(args.auth_scheme or "").strip(),
            runbook_url=str(args.runbook_url or "").strip(),
            escalation_roster_url=str(args.escalation_roster_url or "").strip(),
        )
    )

    incident_tracker = IncidentTracker(
        events_path=incident_events_path,
        run_id=f"staging_ticket_drill_{drill_id}",
        ack_sla_minutes=max(1, int(args.ack_sla_minutes)),
    )
    monitor = SLOMonitor(
        incident_tracker=incident_tracker,
        incident_ticket_notifier=notifier,
        notification_dead_letter_path=dead_letter_path,
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
    ticketing = status.get("ticketing", {})
    dead_letters = status.get("dead_letters", {})
    delivery_attempted = int(ticketing.get("attempts", 0)) > 0
    delivery_succeeded = (
        int(ticketing.get("created", 0)) > 0 and int(ticketing.get("failures", 0)) == 0
    )
    is_non_test_target = _is_non_test_webhook_target(args.webhook_url)
    passed = (
        "incident_ack_sla_breach" in breach_names
        and delivery_attempted
        and (delivery_succeeded or not bool(args.require_delivery))
        and (is_non_test_target or not bool(args.require_non_test_target))
    )

    return {
        "drill_id": drill_id,
        "timestamp": datetime.utcnow().isoformat(),
        "passed": passed,
        "require_delivery": bool(args.require_delivery),
        "breaches": breach_names,
        "delivery": {
            "attempted": delivery_attempted,
            "succeeded": delivery_succeeded,
            "events_recorded": len(notifier.events),
        },
        "ticketing": ticketing,
        "dead_letters": dead_letters,
        "incident": status.get("incidents", {}),
        "webhook": {
            "url": _sanitize_webhook_url(args.webhook_url),
            "source": args.source,
            "timeout_seconds": max(1, int(args.timeout_seconds)),
            "max_retries": max(0, int(args.max_retries)),
            "retry_backoff_seconds": max(0.0, float(args.retry_backoff_seconds)),
            "auth_configured": bool(str(args.auth_token or "").strip()),
            "is_non_test_target": is_non_test_target,
            "require_non_test_target": bool(args.require_non_test_target),
        },
        "response_links": {
            "runbook_url": str(args.runbook_url or "").strip(),
            "escalation_roster_url": str(args.escalation_roster_url or "").strip(),
            "configured": bool(
                str(args.runbook_url or "").strip()
                and str(args.escalation_roster_url or "").strip()
            ),
        },
        "artifacts": {
            "artifact_dir": str(artifact_dir),
            "incident_events_path": str(incident_events_path),
            "dead_letters_path": str(dead_letter_path),
        },
    }


def main() -> int:
    args = _parse_args()
    report = _build_report(args)

    output_path = Path(args.output) if args.output else None
    if output_path is None:
        output_path = (
            Path(args.artifact_dir) / f"incident_ticket_drill_report_{report['drill_id']}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("STAGING INCIDENT TICKET DRILL")
    print(json.dumps(report, indent=2))
    print(f"EVIDENCE_WRITTEN={output_path}")

    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
