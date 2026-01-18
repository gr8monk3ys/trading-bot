#!/usr/bin/env python3
"""
Extended Hours Trading Example

Demonstrates pre-market (4:00 AM - 9:30 AM) and after-hours (4:00 PM - 8:00 PM)
trading with appropriate safeguards for low liquidity conditions.

Trading Sessions:
- Pre-Market: 4:00 AM - 9:30 AM EST (gap trading, earnings reactions)
- Regular: 9:30 AM - 4:00 PM EST (normal strategies)
- After-Hours: 4:00 PM - 8:00 PM EST (earnings reactions, news)

Strategies:
- Pre-Market: Gap trading on overnight news
- After-Hours: Earnings reaction trading

Usage:
    python examples/extended_hours_trading_example.py
"""

import asyncio
import logging

from brokers.alpaca_broker import AlpacaBroker
from utils.extended_hours import ExtendedHoursManager, GapTradingStrategy, EarningsReactionStrategy, format_session_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run extended hours trading example."""
    print("\n" + "="*80)
    print("🧪 EXTENDED HOURS TRADING EXAMPLE")
    print("="*80 + "\n")

    # Initialize broker (paper trading)
    broker = AlpacaBroker(paper=True)

    # Initialize extended hours manager
    ext_hours = ExtendedHoursManager(
        broker=broker,
        enable_pre_market=True,
        enable_after_hours=True
    )

    # Get current session info
    session_info = ext_hours.get_session_info()
    print(format_session_info(session_info))

    # Initialize strategies for different sessions
    gap_strategy = GapTradingStrategy(gap_threshold=0.02)  # 2% gap threshold
    earnings_strategy = EarningsReactionStrategy(min_move_pct=0.03)  # 3% move threshold

    # Stocks to monitor
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

    print("\n📊 Extended Hours Configuration:")
    print(f"   Pre-Market Trading: {'ENABLED' if ext_hours.enable_pre_market else 'DISABLED'}")
    print(f"   After-Hours Trading: {'ENABLED' if ext_hours.enable_after_hours else 'DISABLED'}")
    print(f"   Monitored Symbols: {', '.join(symbols)}")
    print("   Position Size: 50% of regular (more conservative)")
    print("   Order Type: LIMIT only (safer for low liquidity)")
    print("\n")

    print("⚠️  Extended Hours Safety Features:")
    print("   • Automatic position size reduction (50% of regular)")
    print("   • Limit orders only (no market orders)")
    print("   • Spread check (max 0.5% bid-ask spread)")
    print("   • Volume validation (minimum 10K daily volume)")
    print("   • Slippage protection (max 0.3% slippage)")
    print("\n")

    try:
        print("="*80)
        print("📈 EXTENDED HOURS MONITORING STARTED")
        print("="*80)
        print("\nPress Ctrl+C to stop\n")

        # Monitor loop
        while True:
            await asyncio.sleep(60)  # Check every minute

            # Get current session
            session = ext_hours.get_current_session()
            session_info = ext_hours.get_session_info()

            print("-" * 80)
            print(f"⏰ {session_info['current_time']}")
            print(f"📅 Session: {session_info['session_name']}")

            if session == 'closed':
                print("🔴 Market is CLOSED - no trading")
                print("-" * 80)
                continue

            # Get account status
            account = await broker.get_account()
            equity = float(account.equity)
            buying_power = float(account.buying_power)

            print(f"💰 Equity: ${equity:,.2f} | Buying Power: ${buying_power:,.2f}")

            # Different strategies for different sessions
            if ext_hours.is_pre_market():
                print("\n🌅 PRE-MARKET SESSION:")
                print("   Strategy: Gap Trading")
                print("   Focus: Overnight news, earnings, futures direction")

                # Example: Check for gaps (in real implementation, get previous close)
                for symbol in symbols:
                    # Placeholder - in real implementation:
                    # prev_close = await get_previous_close(symbol)
                    # current_price = await get_current_price(symbol)
                    # gap_signal = await gap_strategy.analyze_gap(symbol, prev_close, current_price)

                    print(f"\n   {symbol}:")
                    print("      Checking for overnight gaps...")
                    # if gap_signal:
                    #     print(f"      GAP DETECTED: {gap_signal['reason']}")
                    #     if gap_signal['signal'] == 'buy':
                    #         await ext_hours.execute_extended_hours_trade(
                    #             symbol, 'buy', quantity, strategy='limit'
                    #         )

            elif ext_hours.is_after_hours():
                print("\n🌆 AFTER-HOURS SESSION:")
                print("   Strategy: Earnings Reactions")
                print("   Focus: Post-earnings moves, guidance, analyst calls")

                # Example: Monitor earnings reactions
                for symbol in symbols:
                    print(f"\n   {symbol}:")
                    print("      Monitoring for earnings announcements...")
                    # In real implementation:
                    # if has_earnings_today(symbol):
                    #     close_price = await get_close_price(symbol)
                    #     ah_price = await get_current_price(symbol)
                    #     earnings_beat = await check_earnings_result(symbol)
                    #     signal = await earnings_strategy.analyze_earnings_move(
                    #         symbol, close_price, ah_price, earnings_beat
                    #     )
                    #     if signal:
                    #         print(f"      EARNINGS SIGNAL: {signal['reason']}")

            else:  # Regular hours
                print("\n☀️  REGULAR HOURS SESSION:")
                print("   Full strategies available")
                print("   Normal position sizing")

            # Show current positions
            positions = await broker.get_positions()
            if positions:
                print(f"\n📊 Current Positions: {len(positions)}")
                for pos in positions:
                    pnl = float(pos.unrealized_pl)
                    pnl_pct = float(pos.unrealized_plpc) * 100
                    print(f"   {pos.symbol}: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")

            print("-" * 80)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping extended hours trading...")

    finally:
        # Show final stats
        account = await broker.get_account()
        final_equity = float(account.equity)

        print("\n" + "="*80)
        print("📊 EXTENDED HOURS TRADING SESSION COMPLETE")
        print("="*80)
        print(f"Final Equity: ${final_equity:,.2f}")

        positions = await broker.get_positions()
        if positions:
            print(f"\nOpen Positions: {len(positions)}")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                print(f"  {pos.symbol}: ${pnl:+,.2f}")

        print("\n✅ Extended hours trading stopped cleanly")


async def example_gap_trade():
    """Example of a gap trade in pre-market."""
    print("\n" + "="*80)
    print("📈 GAP TRADING EXAMPLE")
    print("="*80 + "\n")

    broker = AlpacaBroker(paper=True)
    ext_hours = ExtendedHoursManager(broker)

    # Check if we can trade
    if not ext_hours.is_pre_market():
        print("❌ Not in pre-market session")
        return

    symbol = 'AAPL'
    prev_close = 175.00  # Previous day's close
    pre_market_price = 180.00  # Pre-market price (3% gap up!)

    gap_pct = (pre_market_price - prev_close) / prev_close
    print(f"📊 {symbol} Analysis:")
    print(f"   Previous Close: ${prev_close:.2f}")
    print(f"   Pre-Market: ${pre_market_price:.2f}")
    print(f"   Gap: {gap_pct:.2%} 🚀")

    if gap_pct > 0.02:  # 2% gap
        print("\n✅ Gap threshold met - executing FADE trade")
        print("   Strategy: Sell the gap (expect reversion to mean)")

        # Calculate position size (reduced for extended hours)
        account = await broker.get_account()
        buying_power = float(account.buying_power)
        position_value = buying_power * 0.10  # 10% of capital

        # Reduce for extended hours
        position_value = ext_hours.adjust_position_size_for_extended_hours(position_value)
        quantity = position_value / pre_market_price

        print(f"   Position: {quantity:.2f} shares (${position_value:,.2f})")

        # Execute extended hours trade
        result = await ext_hours.execute_extended_hours_trade(
            symbol=symbol,
            side='sell',  # Fade the gap
            quantity=quantity,
            strategy='limit'  # ALWAYS use limit in extended hours
        )

        if result:
            print("\n✅ Extended hours order submitted!")
        else:
            print("\n❌ Order failed (likely spread too wide or liquidity too low)")


if __name__ == "__main__":
    # Run main monitoring example
    asyncio.run(main())

    # Uncomment to run gap trading example:
    # asyncio.run(example_gap_trade())
