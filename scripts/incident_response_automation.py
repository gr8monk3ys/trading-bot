#!/usr/bin/env python3
"""Run automated incident response based on operational metrics and readiness artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.incident_response_automation import run_incident_response_automation  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incident response automation")
    parser.add_argument("--run-dir", required=True, help="Run artifact directory")
    parser.add_argument("--runtime-watchdog-json", default=None)
    parser.add_argument("--runtime-gate-json", default=None)
    parser.add_argument("--go-live-summary-json", default=None)
    parser.add_argument("--critical-slo-breach-threshold", type=int, default=1)
    parser.add_argument("--incident-sla-breach-threshold", type=int, default=1)
    parser.add_argument("--dead-letter-critical-threshold", type=int, default=25)
    parser.add_argument(
        "--webhook-url",
        default=str(os.environ.get("INCIDENT_RESPONSE_AUTOMATION_WEBHOOK_URL", "") or "").strip(),
        help="Optional escalation webhook URL (or env INCIDENT_RESPONSE_AUTOMATION_WEBHOOK_URL)",
    )
    parser.add_argument("--webhook-timeout-seconds", type=int, default=5)
    parser.add_argument(
        "--webhook-auth-token",
        default=str(os.environ.get("INCIDENT_RESPONSE_AUTOMATION_AUTH_TOKEN", "") or "").strip(),
    )
    parser.add_argument(
        "--webhook-auth-scheme",
        default=str(
            os.environ.get("INCIDENT_RESPONSE_AUTOMATION_AUTH_SCHEME", "Bearer") or "Bearer"
        ).strip(),
    )
    parser.add_argument(
        "--rollback-cmd",
        default="",
        help="Optional rollback command executed on critical incident conditions",
    )
    parser.add_argument("--rollback-timeout-seconds", type=int, default=120)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_incident_response_automation(
        run_dir=args.run_dir,
        runtime_watchdog_json=args.runtime_watchdog_json,
        runtime_gate_json=args.runtime_gate_json,
        go_live_summary_json=args.go_live_summary_json,
        critical_slo_breach_threshold=max(1, int(args.critical_slo_breach_threshold)),
        incident_sla_breach_threshold=max(1, int(args.incident_sla_breach_threshold)),
        dead_letter_critical_threshold=max(1, int(args.dead_letter_critical_threshold)),
        webhook_url=str(args.webhook_url or "").strip(),
        webhook_timeout_seconds=max(1, int(args.webhook_timeout_seconds)),
        webhook_auth_token=str(args.webhook_auth_token or "").strip(),
        webhook_auth_scheme=str(args.webhook_auth_scheme or "").strip(),
        rollback_cmd=str(args.rollback_cmd or "").strip(),
        rollback_timeout_seconds=max(1, int(args.rollback_timeout_seconds)),
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plan = report.get("plan", {})
    actions = report.get("actions", {})
    print("=" * 72)
    print("INCIDENT RESPONSE AUTOMATION")
    print("=" * 72)
    print(f"Ready: {'YES' if report.get('ready') else 'NO'}")
    print(f"Severity: {plan.get('severity', 'unknown')}")
    print(f"Triggers: {plan.get('trigger_count', 0)}")
    webhook = actions.get("webhook", {})
    rollback = actions.get("rollback", {})
    print(
        "Webhook: "
        f"attempted={bool(webhook.get('attempted'))} "
        f"delivered={bool(webhook.get('delivered'))} "
        f"message={webhook.get('message')}"
    )
    print(
        "Rollback: "
        f"attempted={bool(rollback.get('attempted'))} "
        f"succeeded={bool(rollback.get('succeeded'))} "
        f"message={rollback.get('message')}"
    )
    return 0 if report.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
