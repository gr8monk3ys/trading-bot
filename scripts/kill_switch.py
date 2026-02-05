#!/usr/bin/env python3
"""
Emergency kill switch for live trading.

Usage:
  python scripts/kill_switch.py --confirm "HALT TRADING"
  python scripts/kill_switch.py --confirm "HALT TRADING" --cancel-orders --liquidate
"""

import argparse
import asyncio

from brokers.alpaca_broker import AlpacaBroker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergency kill switch")
    parser.add_argument(
        "--confirm",
        required=True,
        help='Type exactly: "HALT TRADING"',
    )
    parser.add_argument(
        "--cancel-orders",
        action="store_true",
        help="Cancel all open orders",
    )
    parser.add_argument(
        "--liquidate",
        action="store_true",
        help="Liquidate all open positions",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    if args.confirm != "HALT TRADING":
        raise SystemExit("Confirmation phrase did not match. Aborting.")

    broker = AlpacaBroker(paper=True)

    if args.cancel_orders:
        await broker.cancel_all_orders()

    if args.liquidate:
        positions = await broker.get_positions()
        for pos in positions:
            side = "sell" if float(pos.qty) > 0 else "buy"
            qty = abs(float(pos.qty))
            from brokers.order_builder import OrderBuilder

            order = OrderBuilder(pos.symbol, side, qty).market().day().build()
            try:
                await broker.submit_order_advanced(order)
            except Exception as e:
                print(f"Failed to submit liquidation for {pos.symbol}: {e}")

    print("Kill switch executed.")
    return 0


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
