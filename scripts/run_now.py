#!/usr/bin/env python3
"""
ULTRA SIMPLE LIVE TRADER

Just monitors your account and positions in real-time.
No complex dependencies, just pure Alpaca API.
"""

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream

# Load credentials
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')

print("\n" + "="*80)
print("🤖 LIVE TRADING BOT - MONITORING MODE")
print("="*80)

# Initialize Alpaca
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Get account info
account = trading_client.get_account()

print(f"\n✅ Connected to Alpaca")
print(f"   Account ID: {account.id}")
print(f"   Status: {account.status}")
print(f"   Equity: ${float(account.equity):,.2f}")
print(f"   Cash: ${float(account.cash):,.2f}")
print(f"   Buying Power: ${float(account.buying_power):,.2f}")

# Check market status
clock = trading_client.get_clock()
print(f"\n   Market: {'🟢 OPEN' if clock.is_open else '🔴 CLOSED'}")
print(f"   Next Open: {clock.next_open}")
print(f"   Next Close: {clock.next_close}")

# Get current positions
positions = trading_client.get_all_positions()

print(f"\n📊 CURRENT POSITIONS: {len(positions)}")
print("-"*80)

if positions:
    print(f"{'Symbol':<8} {'Qty':>10} {'Entry':>12} {'Current':>12} {'P/L':>15}")
    print("-"*80)
    for pos in positions:
        symbol = pos.symbol
        qty = float(pos.qty)
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price)
        pnl = float(pos.unrealized_pl)
        pnl_pct = float(pos.unrealized_plpc) * 100

        color = '\033[92m' if pnl > 0 else '\033[91m'  # Green/Red
        reset = '\033[0m'

        print(f"{symbol:<8} {qty:>10.2f} ${entry:>11.2f} ${current:>11.2f} "
              f"{color}${pnl:>+9,.2f} ({pnl_pct:>+6.2f}%){reset}")
else:
    print("No open positions")

print("-"*80)

# Initialize WebSocket for live data
print("\n🔄 Starting real-time data stream...")
print("Press Ctrl+C to stop\n")

stream = StockDataStream(API_KEY, API_SECRET)

# Track symbols from positions
symbols_to_watch = [pos.symbol for pos in positions] if positions else ['SPY', 'QQQ']

print(f"Watching: {', '.join(symbols_to_watch)}\n")
print("="*80)


async def quote_handler(data):
    """Handle incoming quotes."""
    symbol = data.symbol
    bid = data.bid_price
    ask = data.ask_price
    timestamp = datetime.now().strftime('%H:%M:%S')

    print(f"{timestamp} | {symbol:<6} | Bid: ${bid:>8.2f} | Ask: ${ask:>8.2f} | Spread: ${ask-bid:.2f}")


async def run_stream():
    """Run the WebSocket stream."""
    try:
        # Subscribe to quotes
        stream.subscribe_quotes(quote_handler, *symbols_to_watch)

        # Start stream
        await stream._run()

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping stream...")
    except Exception as e:
        print(f"\n❌ Stream error: {e}")


# Run monitoring
try:
    print("\n📈 REAL-TIME QUOTES:")
    print("-"*80)
    asyncio.run(run_stream())
except KeyboardInterrupt:
    print("\n\n✅ Monitoring stopped")
    print("="*80)

print("\nFinal Account Status:")

# Get updated account
account = trading_client.get_account()
print(f"  Equity: ${float(account.equity):,.2f}")

positions = trading_client.get_all_positions()
print(f"  Positions: {len(positions)}")

if positions:
    total_pnl = sum(float(p.unrealized_pl) for p in positions)
    print(f"  Total Unrealized P/L: ${total_pnl:+,.2f}")

print("\n" + "="*80)
print("👋 Goodbye!")
print("="*80 + "\n")
