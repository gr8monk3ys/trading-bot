#!/usr/bin/env python3
"""
Quick script to check current paper trading positions and account status
"""

import asyncio

from brokers.alpaca_broker import AlpacaBroker


async def main():
    broker = AlpacaBroker(paper=True)

    print("\n" + "=" * 80)
    print("PAPER TRADING ACCOUNT STATUS")
    print("=" * 80)

    # Get account info
    account = await broker.get_account()
    print(f"\nAccount ID: {account.id}")
    print(f"Status: {account.status}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")

    # Get positions
    positions = await broker.get_positions()

    if positions:
        print(f"\n{'='*80}")
        print(f"CURRENT POSITIONS ({len(positions)})")
        print(f"{'='*80}")
        print(f"{'Symbol':<10} {'Qty':<10} {'Entry':<12} {'Current':<12} {'P&L':<12} {'P&L %':<10}")
        print("-" * 80)

        total_pl = 0
        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            entry = float(pos.avg_entry_price)
            current = float(pos.current_price)
            pl = float(pos.unrealized_pl)
            pl_pct = float(pos.unrealized_plpc) * 100
            total_pl += pl

            print(
                f"{symbol:<10} {qty:<10.2f} ${entry:<11.2f} ${current:<11.2f} ${pl:<11.2f} {pl_pct:>8.2f}%"
            )

        print("-" * 80)
        print(f"{'TOTAL':<44} ${total_pl:<11.2f}")
    else:
        print("\n✓ No open positions")

    # Get open orders
    orders = await broker.get_orders(status="open")

    if orders:
        print(f"\n{'='*80}")
        print(f"OPEN ORDERS ({len(orders)})")
        print(f"{'='*80}")
        for order in orders:
            print(f"  {order.symbol}: {order.side} {order.qty} @ {order.type}")
    else:
        print("\n✓ No open orders")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
