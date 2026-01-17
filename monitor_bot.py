#!/usr/bin/env python3
"""
Real-time monitoring dashboard for the trading bot
Shows account status, positions, orders, and performance metrics
"""
import asyncio
import sys
from datetime import datetime
from brokers.alpaca_broker import AlpacaBroker

async def get_stats(broker):
    """Get current account statistics"""
    account = await broker.get_account()
    positions = await broker.get_positions()
    open_orders = await broker.get_orders(status='open')

    # Calculate P&L
    total_pl = sum(float(p.unrealized_pl) for p in positions)
    total_pl_pct = (total_pl / 100000.0) * 100  # Assuming $100k starting capital

    return {
        'cash': float(account.cash),
        'portfolio_value': float(account.portfolio_value),
        'buying_power': float(account.buying_power),
        'positions': len(positions),
        'open_orders': len(open_orders),
        'total_pl': total_pl,
        'total_pl_pct': total_pl_pct,
        'position_details': positions,
        'order_details': open_orders
    }

def print_dashboard(stats, iteration):
    """Print dashboard to console"""
    # Clear screen (works on most terminals)
    print("\033[2J\033[H", end="")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 100)
    print(f"🤖  TRADING BOT DASHBOARD  -  {timestamp}  -  Iteration #{iteration}")
    print("=" * 100)

    # Account Overview
    print("\n💰  ACCOUNT OVERVIEW")
    print(f"    Portfolio Value:  ${stats['portfolio_value']:>12,.2f}")
    print(f"    Cash Available:   ${stats['cash']:>12,.2f}")
    print(f"    Buying Power:     ${stats['buying_power']:>12,.2f}")

    # Performance
    pl_symbol = "📈" if stats['total_pl'] >= 0 else "📉"
    pl_color = "\033[92m" if stats['total_pl'] >= 0 else "\033[91m"  # Green/Red
    reset_color = "\033[0m"

    print(f"\n{pl_symbol}  PERFORMANCE")
    print(f"    Total P&L:        {pl_color}${stats['total_pl']:>12,.2f} ({stats['total_pl_pct']:>6.2f}%){reset_color}")
    print(f"    Open Positions:   {stats['positions']}")
    print(f"    Open Orders:      {stats['open_orders']}")

    # Positions Detail
    if stats['position_details']:
        print(f"\n📊  POSITIONS ({len(stats['position_details'])})")
        print(f"    {'Symbol':<8} {'Qty':<10} {'Entry':<12} {'Current':<12} {'P&L':<14} {'P&L %':<10}")
        print("    " + "-" * 80)
        for pos in stats['position_details']:
            qty = float(pos.qty)
            entry = float(pos.avg_entry_price)
            current = float(pos.current_price)
            pl = float(pos.unrealized_pl)
            pl_pct = float(pos.unrealized_plpc) * 100

            pl_color = "\033[92m" if pl >= 0 else "\033[91m"
            print(f"    {pos.symbol:<8} {qty:<10.2f} ${entry:<11.2f} ${current:<11.2f} "
                  f"{pl_color}${pl:<12.2f} {pl_pct:>8.2f}%{reset_color}")

    # Open Orders
    if stats['order_details']:
        print(f"\n📝  OPEN ORDERS ({len(stats['order_details'])})")
        for order in stats['order_details']:
            print(f"    {order.symbol}: {order.side.upper()} {order.qty} @ {order.type}")

    print("\n" + "=" * 100)
    print("Press Ctrl+C to stop monitoring")
    print("=" * 100)

async def monitor_loop(interval=60):
    """Main monitoring loop"""
    broker = AlpacaBroker(paper=True)
    iteration = 0

    try:
        while True:
            iteration += 1
            stats = await get_stats(broker)
            print_dashboard(stats, iteration)

            # Wait for next interval
            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    except Exception as e:
        print(f"\n\nError in monitoring loop: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Entry point"""
    # Get interval from command line or default to 60 seconds
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    print(f"\n🚀 Starting bot monitor (updating every {interval} seconds)...\n")
    await monitor_loop(interval)

if __name__ == "__main__":
    asyncio.run(main())
