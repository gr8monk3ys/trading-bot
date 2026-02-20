#!/usr/bin/env python3
"""Launch a conservative low-resource paper-trading profile."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a low-resource paper-trading profile suitable for Raspberry Pi-class hardware"
    )
    parser.add_argument(
        "--strategy",
        default="momentum",
        choices=["momentum", "mean_reversion", "bracket_momentum"],
        help="Strategy name passed to live_trader.py",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT"],
        help="Small symbol set to keep CPU/network usage low",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.02,
        help="Conservative position size fraction",
    )
    parser.add_argument("--stop-loss", type=float, default=0.01, help="Stop-loss fraction")
    parser.add_argument("--take-profit", type=float, default=0.02, help="Take-profit fraction")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print environment + command without starting live_trader.py",
    )
    return parser.parse_args(argv)


def _build_command(args: argparse.Namespace) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        str(root / "live_trader.py"),
        "--strategy",
        args.strategy,
        "--symbols",
        *args.symbols,
        "--position-size",
        str(args.position_size),
        "--stop-loss",
        str(args.stop_loss),
        "--take-profit",
        str(args.take_profit),
    ]


def _build_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PAPER", "true")
    env.setdefault("SLO_PAGING_ENABLED", "false")
    env.setdefault("INCIDENT_ACK_SLA_MINUTES", "15")
    env.setdefault("PAPER_LIVE_SHADOW_DRIFT_WARNING", "0.10")
    env.setdefault("PAPER_LIVE_SHADOW_DRIFT_MAX", "0.12")
    return env


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    env = _build_environment()
    cmd = _build_command(args)

    print("Low-resource profile settings:")
    print(f"  PAPER={env['PAPER']}")
    print(f"  SLO_PAGING_ENABLED={env['SLO_PAGING_ENABLED']}")
    print(f"  INCIDENT_ACK_SLA_MINUTES={env['INCIDENT_ACK_SLA_MINUTES']}")
    print(f"  PAPER_LIVE_SHADOW_DRIFT_WARNING={env['PAPER_LIVE_SHADOW_DRIFT_WARNING']}")
    print(f"  PAPER_LIVE_SHADOW_DRIFT_MAX={env['PAPER_LIVE_SHADOW_DRIFT_MAX']}")
    print("Command:")
    print("  " + shlex.join(cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
