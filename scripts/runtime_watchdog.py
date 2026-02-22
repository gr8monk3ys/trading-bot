#!/usr/bin/env python3
"""
Run runtime watchdog checks (Alpaca, incident webhook, IB socket).
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

from utils.runtime_watchdog import run_runtime_watchdog  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime watchdog checks")
    parser.add_argument(
        "--check-alpaca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Alpaca account connectivity check",
    )
    parser.add_argument(
        "--check-ticket-webhook",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable incident-ticket webhook check",
    )
    parser.add_argument(
        "--check-ib-port",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable IB API socket port check",
    )
    parser.add_argument(
        "--check-ib-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable IB API session handshake check",
    )

    parser.add_argument(
        "--alpaca-api-key",
        default=str(os.environ.get("ALPACA_API_KEY", "") or "").strip(),
    )
    parser.add_argument(
        "--alpaca-secret-key",
        default=str(os.environ.get("ALPACA_SECRET_KEY", "") or "").strip(),
    )
    parser.add_argument(
        "--alpaca-paper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Alpaca paper endpoint",
    )

    parser.add_argument(
        "--ticket-webhook-url",
        default=str(os.environ.get("INCIDENT_TICKETING_WEBHOOK_URL", "") or "").strip(),
    )
    parser.add_argument(
        "--ticket-timeout-seconds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ticket-auth-token",
        default=str(os.environ.get("INCIDENT_TICKETING_AUTH_TOKEN", "") or "").strip(),
    )
    parser.add_argument(
        "--ticket-auth-scheme",
        default=str(os.environ.get("INCIDENT_TICKETING_AUTH_SCHEME", "Bearer") or "").strip(),
    )

    parser.add_argument(
        "--ib-host",
        default=str(os.environ.get("IB_HOST", "127.0.0.1") or "127.0.0.1").strip(),
    )
    parser.add_argument(
        "--ib-port",
        type=int,
        default=int(str(os.environ.get("IB_PAPER_PORT", "7497") or "7497").strip()),
    )
    parser.add_argument(
        "--ib-timeout-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--ib-client-id",
        type=int,
        default=int(str(os.environ.get("IB_CLIENT_ID", "1") or "1").strip()),
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = asyncio.run(
        run_runtime_watchdog(
            check_alpaca=bool(args.check_alpaca),
            check_ticket_webhook=bool(args.check_ticket_webhook),
            check_ib_port=bool(args.check_ib_port),
            check_ib_api=bool(args.check_ib_api),
            alpaca_api_key=str(args.alpaca_api_key or "").strip(),
            alpaca_secret_key=str(args.alpaca_secret_key or "").strip(),
            alpaca_paper=bool(args.alpaca_paper),
            ticket_webhook_url=str(args.ticket_webhook_url or "").strip(),
            ticket_timeout_seconds=max(1, int(args.ticket_timeout_seconds)),
            ticket_auth_token=str(args.ticket_auth_token or "").strip(),
            ticket_auth_scheme=str(args.ticket_auth_scheme or "").strip(),
            ib_host=str(args.ib_host or "").strip(),
            ib_port=max(1, int(args.ib_port)),
            ib_client_id=max(1, int(args.ib_client_id)),
            ib_timeout_seconds=max(0.1, float(args.ib_timeout_seconds)),
        )
    )

    print("=" * 72)
    print("RUNTIME WATCHDOG")
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
