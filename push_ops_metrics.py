#!/usr/bin/env python3
"""Build ops metrics snapshot and optionally push to external Prometheus endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ops_metrics import build_ops_metrics_snapshot, format_prometheus_text  # noqa: E402
from utils.ops_metrics_push import push_prometheus_metrics  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push operational Prometheus metrics externally")
    parser.add_argument("--run-dir", required=True, help="Run artifact directory")
    parser.add_argument("--runtime-watchdog-json", default=None)
    parser.add_argument("--runtime-gate-json", default=None)
    parser.add_argument("--go-live-summary-json", default=None)
    parser.add_argument("--namespace", default="trading_bot")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--prom-output", default=None)
    parser.add_argument(
        "--pushgateway-url",
        default=str(os.environ.get("OPS_METRICS_PUSHGATEWAY_URL", "") or "").strip(),
        help="Pushgateway base URL (or set OPS_METRICS_PUSHGATEWAY_URL)",
    )
    parser.add_argument("--push-job", default="trading_bot")
    parser.add_argument(
        "--push-instance",
        default=str(os.environ.get("OPS_METRICS_PUSH_INSTANCE", "") or "").strip(),
    )
    parser.add_argument("--push-timeout-seconds", type=int, default=5)
    parser.add_argument("--push-method", choices=["PUT", "POST"], default="PUT")
    parser.add_argument(
        "--push-auth-token",
        default=str(os.environ.get("OPS_METRICS_PUSH_AUTH_TOKEN", "") or "").strip(),
    )
    parser.add_argument(
        "--push-auth-scheme",
        default=str(os.environ.get("OPS_METRICS_PUSH_AUTH_SCHEME", "Bearer") or "").strip(),
    )
    parser.add_argument(
        "--fail-on-push-error",
        action="store_true",
        help="Return non-zero when push attempt fails",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    snapshot = build_ops_metrics_snapshot(
        run_dir=args.run_dir,
        runtime_watchdog_json=args.runtime_watchdog_json,
        runtime_gate_json=args.runtime_gate_json,
        go_live_summary_json=args.go_live_summary_json,
    )
    prom_text = format_prometheus_text(snapshot, namespace=args.namespace)

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    if args.prom_output:
        prom_path = Path(args.prom_output)
        prom_path.parent.mkdir(parents=True, exist_ok=True)
        prom_path.write_text(prom_text, encoding="utf-8")

    push_result = None
    if args.pushgateway_url:
        push_result = push_prometheus_metrics(
            pushgateway_url=args.pushgateway_url,
            metrics_text=prom_text,
            job=args.push_job,
            instance=args.push_instance,
            timeout_seconds=max(1, int(args.push_timeout_seconds)),
            method=args.push_method,
            auth_token=args.push_auth_token,
            auth_scheme=args.push_auth_scheme,
        ).to_dict()

    print("=" * 72)
    print("OPS METRICS PUSH")
    print("=" * 72)
    print(f"Run Dir: {snapshot.get('run_dir')}")
    print(f"Metrics Count: {len(snapshot.get('metrics', {}))}")
    if push_result:
        print(
            "Push Status: "
            + ("DELIVERED" if push_result.get("delivered") else "FAILED")
            + f" ({push_result.get('message')})"
        )
    else:
        print("Push Status: SKIPPED (no --pushgateway-url provided)")

    if push_result and not bool(push_result.get("delivered")) and args.fail_on_push_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
