#!/usr/bin/env python3
"""Export operational run metrics to JSON and Prometheus text format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ops_metrics import build_ops_metrics_snapshot, format_prometheus_text  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ops metrics from run artifacts")
    parser.add_argument("--run-dir", required=True, help="Run artifact directory")
    parser.add_argument(
        "--runtime-watchdog-json",
        default=None,
        help="Optional runtime watchdog JSON report",
    )
    parser.add_argument(
        "--runtime-gate-json",
        default=None,
        help="Optional runtime industrial gate JSON report",
    )
    parser.add_argument(
        "--go-live-summary-json",
        default=None,
        help="Optional go-live precheck summary JSON report",
    )
    parser.add_argument(
        "--namespace",
        default="trading_bot",
        help="Prometheus namespace prefix",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional output path for raw JSON snapshot",
    )
    parser.add_argument(
        "--prom-output",
        default=None,
        help="Optional output path for Prometheus text metrics",
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

    print("=" * 72)
    print("OPS METRICS EXPORT")
    print("=" * 72)
    print(f"Run Dir: {snapshot.get('run_dir')}")
    for name, value in snapshot.get("metrics", {}).items():
        print(f"- {name}: {value}")

    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    if args.prom_output:
        path = Path(args.prom_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prom_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
