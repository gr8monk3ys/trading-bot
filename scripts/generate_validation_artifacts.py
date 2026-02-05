#!/usr/bin/env python3
"""
Generate reproducible validation artifacts.

Outputs a timestamped folder with:
- validated_backtest_report.md
- validated_backtest_result.json
- manifest.json (env + git + config snapshot)
- paper_trading_summary.json (if available)
"""

import argparse
import asyncio
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from brokers.alpaca_broker import AlpacaBroker
from config import BACKTEST_PARAMS, RISK_PARAMS, SYMBOLS
from engine.strategy_manager import StrategyManager
from engine.validated_backtest import (
    ValidatedBacktestRunner,
    format_validated_backtest_report,
)
from utils.paper_trading_monitor import PaperTradingMonitor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation artifacts")
    parser.add_argument("--strategy", required=True, help="Strategy class name")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="results/validation", help="Base output dir")
    return parser.parse_args()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


async def _paper_trading_summary(strategy: str) -> Dict[str, Any]:
    monitor = PaperTradingMonitor(strategy_name=strategy)
    result = await monitor.is_go_live_ready()
    result["state"] = monitor.state.__dict__
    result["state"]["start_date"] = monitor.state.start_date.isoformat()
    return result


async def _run(args: argparse.Namespace) -> int:
    datetime.strptime(args.start_date, "%Y-%m-%d")
    datetime.strptime(args.end_date, "%Y-%m-%d")

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else SYMBOLS
    )

    broker = AlpacaBroker(paper=True)
    manager = StrategyManager(broker=broker)
    if args.strategy not in manager.available_strategies:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    strategy_class = manager.available_strategies[args.strategy]
    runner = ValidatedBacktestRunner(broker)

    result = await runner.run_validated_backtest(
        strategy_class=strategy_class,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save report + raw results
    report_path = out_dir / "validated_backtest_report.md"
    report_path.write_text(format_validated_backtest_report(result))

    result_path = out_dir / "validated_backtest_result.json"
    result_path.write_text(json.dumps(result.__dict__, default=str, indent=2))

    # Manifest with env + config snapshot
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "strategy": args.strategy,
        "symbols": symbols,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "config": {
            "RISK_PARAMS": RISK_PARAMS,
            "BACKTEST_PARAMS": BACKTEST_PARAMS,
        },
        "eligible_for_trading": result.eligible_for_trading,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Paper trading summary (optional)
    try:
        paper_summary = await _paper_trading_summary(args.strategy)
        (out_dir / "paper_trading_summary.json").write_text(
            json.dumps(paper_summary, default=str, indent=2)
        )
    except Exception:
        # No paper trading state available
        pass

    print(f"Artifacts written to: {out_dir}")
    return 0 if result.eligible_for_trading else 1


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
