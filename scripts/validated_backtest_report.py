#!/usr/bin/env python3
"""
Generate a validated backtest report with profitability gates.

Usage:
  python scripts/validated_backtest_report.py \
    --strategy MomentumStrategy \
    --symbols AAPL,MSFT \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --output results/validated_backtest_report.md \
    --json results/validated_backtest_report.json
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from brokers.alpaca_broker import AlpacaBroker
from engine.validated_backtest import (
    ValidatedBacktestRunner,
    format_validated_backtest_report,
)
from engine.strategy_manager import StrategyManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validated backtest report generator")
    parser.add_argument("--strategy", required=True, help="Strategy class name")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", required=False, help="Markdown output path")
    parser.add_argument("--json", dest="json_path", required=False, help="JSON output path")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    # Basic validation
    datetime.strptime(args.start_date, "%Y-%m-%d")
    datetime.strptime(args.end_date, "%Y-%m-%d")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")

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

    report_text = format_validated_backtest_report(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text)
    else:
        print(report_text)

    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result.__dict__, default=str, indent=2))

    return 0 if result.eligible_for_trading else 1


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
