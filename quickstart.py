#!/usr/bin/env python3
"""
Trading Bot Quick Start

Interactive setup and launcher for the trading bot.

Helps you:
1. Test your Alpaca connection
2. Choose a strategy
3. Configure parameters
4. Start trading

Usage:
    python quickstart.py
"""

import asyncio
import os
import sys
from brokers.alpaca_broker import AlpacaBroker


async def test_connection():
    """Test Alpaca API connection."""
    print("\n" + "="*80)
    print("🔌 TESTING ALPACA CONNECTION")
    print("="*80 + "\n")

    try:
        broker = AlpacaBroker(paper=True)
        account = await broker.get_account()

        print("✅ Connection successful!")
        print(f"\n📊 Account Info:")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}\n")

        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}\n")
        print("Please check your .env file and API credentials")
        return False


def choose_strategy():
    """Interactive strategy selection."""
    print("\n" + "="*80)
    print("📈 CHOOSE YOUR STRATEGY")
    print("="*80 + "\n")

    strategies = {
        '1': {
            'name': 'momentum',
            'description': 'Follows upward price momentum with RSI confirmation',
            'best_for': 'Trending markets',
            'risk': 'Medium',
            'time_frame': '1-3 days'
        },
        '2': {
            'name': 'mean_reversion',
            'description': 'Buys oversold stocks, sells when they revert to mean',
            'best_for': 'Range-bound markets',
            'risk': 'Medium-High',
            'time_frame': '1-5 days'
        },
        '3': {
            'name': 'bracket_momentum',
            'description': 'Momentum with automatic take-profit and stop-loss',
            'best_for': 'Active trading',
            'risk': 'Medium',
            'time_frame': 'Hours to days'
        }
    }

    for key, strat in strategies.items():
        print(f"{key}. {strat['name'].replace('_', ' ').title()}")
        print(f"   Description: {strat['description']}")
        print(f"   Best for: {strat['best_for']}")
        print(f"   Risk Level: {strat['risk']}")
        print(f"   Time Frame: {strat['time_frame']}\n")

    while True:
        choice = input("Select strategy (1-3): ").strip()
        if choice in strategies:
            return strategies[choice]['name']
        print("Invalid choice. Please select 1, 2, or 3.")


def choose_symbols():
    """Interactive symbol selection."""
    print("\n" + "="*80)
    print("📊 CHOOSE STOCKS TO TRADE")
    print("="*80 + "\n")

    print("Popular options:")
    print("1. Tech Giants (AAPL, MSFT, GOOGL)")
    print("2. Growth Stocks (TSLA, NVDA, AMD)")
    print("3. S&P 500 Stalwarts (SPY, QQQ, DIA)")
    print("4. Custom (enter your own)\n")

    presets = {
        '1': ['AAPL', 'MSFT', 'GOOGL'],
        '2': ['TSLA', 'NVDA', 'AMD'],
        '3': ['SPY', 'QQQ', 'DIA']
    }

    choice = input("Select preset (1-3) or 4 for custom: ").strip()

    if choice in presets:
        return presets[choice]
    elif choice == '4':
        symbols_input = input("Enter symbols (comma-separated, e.g., AAPL,MSFT,TSLA): ").strip()
        return [s.strip().upper() for s in symbols_input.split(',')]
    else:
        print("Invalid choice. Using default: AAPL, MSFT, GOOGL")
        return ['AAPL', 'MSFT', 'GOOGL']


def configure_parameters():
    """Interactive parameter configuration."""
    print("\n" + "="*80)
    print("⚙️  CONFIGURE PARAMETERS")
    print("="*80 + "\n")

    print("Use recommended settings? (yes/no)")
    print("Recommended: 10% position size, 2% stop loss, 5% take profit")

    use_defaults = input("\nChoice (yes/no): ").strip().lower() == 'yes'

    if use_defaults:
        return {
            'position_size': 0.10,
            'stop_loss': 0.02,
            'take_profit': 0.05
        }

    print("\nCustom configuration:")

    while True:
        try:
            position_size = float(input("Position size (0.01-0.25, default 0.10): ") or "0.10")
            if 0.01 <= position_size <= 0.25:
                break
            print("Position size must be between 0.01 and 0.25")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            stop_loss = float(input("Stop loss (0.01-0.10, default 0.02): ") or "0.02")
            if 0.01 <= stop_loss <= 0.10:
                break
            print("Stop loss must be between 0.01 and 0.10")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            take_profit = float(input("Take profit (0.02-0.20, default 0.05): ") or "0.05")
            if 0.02 <= take_profit <= 0.20:
                break
            print("Take profit must be between 0.02 and 0.20")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return {
        'position_size': position_size,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }


def show_summary(strategy, symbols, params):
    """Show configuration summary."""
    print("\n" + "="*80)
    print("📋 CONFIGURATION SUMMARY")
    print("="*80 + "\n")

    print(f"Strategy: {strategy.replace('_', ' ').title()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"\nParameters:")
    print(f"  Position Size: {params['position_size']:.1%} of capital")
    print(f"  Stop Loss: {params['stop_loss']:.1%}")
    print(f"  Take Profit: {params['take_profit']:.1%}")

    print("\n⚠️  IMPORTANT NOTES:")
    print("  • This is PAPER TRADING (no real money)")
    print("  • Circuit breaker limits daily losses to 3%")
    print("  • Each position is limited to 5% of portfolio")
    print("  • Press Ctrl+C to stop trading gracefully")

    print("\n" + "="*80)


async def main():
    """Main quickstart flow."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "🤖 TRADING BOT QUICKSTART".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")

    # Step 1: Test connection
    if not await test_connection():
        print("\n❌ Setup failed. Please fix the connection issue and try again.\n")
        return 1

    # Step 2: Choose strategy
    strategy = choose_strategy()

    # Step 3: Choose symbols
    symbols = choose_symbols()

    # Step 4: Configure parameters
    params = configure_parameters()

    # Step 5: Show summary
    show_summary(strategy, symbols, params)

    # Step 6: Confirm and launch
    print("\nReady to start trading?")
    confirm = input("Type 'yes' to start, anything else to cancel: ").strip().lower()

    if confirm != 'yes':
        print("\n❌ Trading cancelled.\n")
        return 0

    # Launch trading
    print("\n🚀 Launching trading bot...\n")

    # Build command
    cmd = f"python live_trader.py --strategy {strategy} "
    cmd += f"--symbols {' '.join(symbols)} "
    cmd += f"--position-size {params['position_size']} "
    cmd += f"--stop-loss {params['stop_loss']} "
    cmd += f"--take-profit {params['take_profit']}"

    print(f"Executing: {cmd}\n")
    os.system(cmd)

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Quickstart cancelled.\n")
        sys.exit(0)
