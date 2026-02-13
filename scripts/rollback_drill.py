#!/usr/bin/env python3
"""
Run deterministic runtime-state rollback drill.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.deployment_hardening import run_runtime_rollback_drill


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime rollback drill")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Optional workdir for drill artifacts",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_runtime_rollback_drill(workdir=args.workdir)
    print("=" * 72)
    print("RUNTIME ROLLBACK DRILL")
    print("=" * 72)
    print(f"Passed: {'YES' if report.get('passed') else 'NO'}")
    print(f"Message: {report.get('message')}")
    if report.get("state_path"):
        print(f"State Path: {report.get('state_path')}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    raise SystemExit(0 if report.get("passed") else 1)


if __name__ == "__main__":
    main()
