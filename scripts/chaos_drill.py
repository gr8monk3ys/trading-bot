#!/usr/bin/env python3
"""
Run deterministic chaos drills for reconciliation/data-quality/alerting paths.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from utils.chaos_drills import format_chaos_drill_report, run_chaos_drills


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run operational chaos drills")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    report = await run_chaos_drills()
    print(format_chaos_drill_report(report))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if report.get("passed") else 1


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
