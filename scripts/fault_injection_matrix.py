#!/usr/bin/env python3
"""
Run broker/API fault-injection matrix with SLO assertions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from utils.fault_injection_matrix import (
    format_fault_injection_report,
    run_fault_injection_matrix,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fault injection matrix")
    parser.add_argument(
        "--tmp-dir",
        default="results",
        help="Directory for temporary matrix artifacts",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    report = await run_fault_injection_matrix(tmp_dir=args.tmp_dir)
    print(format_fault_injection_report(report))

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
