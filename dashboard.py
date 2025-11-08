#!/usr/bin/env python3
"""
Real-Time Trading Dashboard

Live terminal dashboard showing:
- Current positions & P/L
- Recent trades
- Performance metrics
- Risk status
- Market status

Usage:
    python dashboard.py
"""

import asyncio
import os
from datetime import datetime
from brokers.alpaca_broker import AlpacaBroker
from utils.performance_tracker import PerformanceTracker


class TradingDashboard:
    """Real-time trading dashboard."""

    def __init__(self):
        self.broker = None
        self.tracker = PerformanceTracker()
        self.running = True

    async def initialize(self):
        """Initialize broker connection."""
        self.broker = AlpacaBroker(paper=True)
        return True

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')

    async def render(self):
        """Render dashboard."""
        self.clear_screen()

        # Header
        print("="*100)
        print(f"{'🤖 LIVE TRADING DASHBOARD':^100}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^100}")
        print("="*100)

        try:
            # Get account info
            account = await self.broker.get_account()
            equity = float(account.equity)
            cash = float(account.cash)
            buying_power = float(account.buying_power)

            # Get positions
            positions = await self.broker.get_positions()

            # Get market status
            clock = await self.broker.get_clock()

            # Account Summary
            print("\n📊 ACCOUNT SUMMARY")
            print("-"*100)
            print(f"  Equity: ${equity:>15,.2f}  |  Cash: ${cash:>15,.2f}  |  Buying Power: ${buying_power:>15,.2f}")
            print(f"  Market: {'🟢 OPEN' if clock.is_open else '🔴 CLOSED':>16}  |  Day P/L: ${float(account.equity) - 100000:>+12,.2f}")
            print("-"*100)

            # Positions
            print("\n💼 OPEN POSITIONS")
            print("-"*100)
            if positions:
                print(f"{'Symbol':<8} {'Qty':>8} {'Entry':>10} {'Current':>10} {'Value':>12} {'P/L $':>12} {'P/L %':>10}")
                print("-"*100)
                for pos in positions:
                    symbol = pos.symbol
                    qty = float(pos.qty)
                    entry = float(pos.avg_entry_price)
                    current = float(pos.current_price)
                    value = float(pos.market_value)
                    pnl = float(pos.unrealized_pl)
                    pnl_pct = float(pos.unrealized_plpc) * 100

                    color = '\033[92m' if pnl > 0 else '\033[91m'  # Green/Red
                    reset = '\033[0m'

                    print(f"{symbol:<8} {qty:>8.2f} ${entry:>9.2f} ${current:>9.2f} ${value:>11,.2f} "
                          f"{color}${pnl:>+11,.2f} {pnl_pct:>+9.2f}%{reset}")
            else:
                print("  No open positions")

            print("-"*100)

            # Recent Trades
            print("\n📈 RECENT TRADES")
            print("-"*100)
            recent_trades = self.tracker.trades[-10:]  # Last 10 trades
            if recent_trades:
                print(f"{'Time':<20} {'Symbol':<8} {'Side':<6} {'P/L $':>12} {'P/L %':>10}")
                print("-"*100)
                for trade in reversed(recent_trades):
                    time_str = trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')
                    color = '\033[92m' if trade.is_winner else '\033[91m'
                    reset = '\033[0m'
                    print(f"{time_str:<20} {trade.symbol:<8} {trade.side:<6} "
                          f"{color}${trade.pnl:>+11,.2f} {trade.pnl_pct*100:>+9.2f}%{reset}")
            else:
                print("  No trades yet")

            print("-"*100)

            # Performance Metrics
            if self.tracker.trades:
                metrics = self.tracker.calculate_metrics(100000)

                print("\n📊 PERFORMANCE METRICS")
                print("-"*100)
                print(f"  Total Trades: {metrics.total_trades:>4}  |  "
                      f"Win Rate: {metrics.win_rate:>6.1%}  |  "
                      f"Profit Factor: {metrics.profit_factor:>6.2f}  |  "
                      f"Sharpe: {metrics.sharpe_ratio:>6.2f}")
                print(f"  Total Return: ${metrics.total_return:>+10,.2f} ({metrics.total_return_pct:>+7.2%})  |  "
                      f"Max DD: {metrics.max_drawdown_pct:>7.2%}  |  "
                      f"Avg Win: ${metrics.avg_win:>8,.2f}")
                print("-"*100)

            # Controls
            print("\n⌨️  CONTROLS: Press Ctrl+C to exit")
            print("="*100)

        except Exception as e:
            print(f"\n❌ Error fetching data: {e}")
            print("="*100)

    async def run(self):
        """Run dashboard update loop."""
        await self.initialize()

        while self.running:
            await self.render()
            await asyncio.sleep(5)  # Update every 5 seconds


async def main():
    """Main entry point."""
    dashboard = TradingDashboard()

    try:
        await dashboard.run()
    except KeyboardInterrupt:
        print("\n\nShutting down dashboard...")


if __name__ == "__main__":
    asyncio.run(main())
