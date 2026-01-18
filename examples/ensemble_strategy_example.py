#!/usr/bin/env python3
"""
Ensemble Strategy Example

Demonstrates how to use the ensemble strategy that combines multiple
trading approaches (mean reversion, momentum, trend following) with
intelligent market regime detection.

Expected Performance:
- Sharpe Ratio: 0.95-1.25
- Best For: All market conditions
- Adapts strategy weights based on regime

Usage:
    python examples/ensemble_strategy_example.py
"""

import asyncio
import logging

from brokers.alpaca_broker import AlpacaBroker
from strategies.ensemble_strategy import EnsembleStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run ensemble strategy example."""
    print("\n" + "="*80)
    print("🧪 ENSEMBLE STRATEGY EXAMPLE")
    print("="*80 + "\n")

    # Initialize broker (paper trading)
    broker = AlpacaBroker(paper=True)

    # Stock symbols to trade
    # Ensemble works best with diverse stocks
    symbols = [
        'SPY',   # S&P 500 ETF (follows market)
        'QQQ',   # Nasdaq 100 ETF (tech-heavy)
        'AAPL',  # Large cap tech
        'MSFT',  # Large cap tech
        'JPM',   # Financial
    ]

    print("📊 Strategy Configuration:")
    print(f"   Symbols: {', '.join(symbols)}")
    print("   Sub-strategies: Mean Reversion, Momentum, Trend Following")
    print("   Min Agreement: 60% (need 60% of strategies to agree)")
    print("   Regime Detection: Automatic (trending/ranging/volatile)")
    print("   Position Size: 10% per position")
    print("\n")

    # Initialize strategy with custom parameters
    strategy = EnsembleStrategy(
        broker=broker,
        symbols=symbols,
        parameters={
            'position_size': 0.10,  # 10% per position
            'max_positions': 3,     # Max 3 concurrent positions
            'stop_loss': 0.025,     # 2.5% stop loss
            'take_profit': 0.05,    # 5% take profit
            'min_agreement_pct': 0.60,  # 60% agreement needed
            'trailing_stop': 0.015,  # 1.5% trailing stop

            # Regime detection
            'adx_trending_threshold': 25,
            'adx_ranging_threshold': 20,
            'atr_volatility_threshold': 0.02,
        }
    )

    # Initialize strategy
    if not await strategy.initialize():
        print("❌ Failed to initialize strategy")
        return

    print("✅ Strategy initialized successfully")
    print("\n" + "="*80)
    print("📈 LIVE TRADING STARTED")
    print("="*80)
    print("\nHow it works:")
    print("1. Detects market regime (trending/ranging/volatile)")
    print("2. Weights strategies based on regime")
    print("   - Trending market: Boost momentum & trend following")
    print("   - Ranging market: Boost mean reversion")
    print("3. Requires 60% agreement across strategies")
    print("4. Enters only when confident signal")
    print("5. Exits with trailing stops to lock profits")
    print("\nPress Ctrl+C to stop\n")

    try:
        # Start WebSocket streaming
        await broker.start_websocket()

        # Run indefinitely
        while True:
            await asyncio.sleep(60)

            # Get account status
            account = await broker.get_account()
            equity = float(account.equity)

            # Get positions
            positions = await broker.get_positions()

            print("-" * 80)
            print(f"💰 Equity: ${equity:,.2f} | Positions: {len(positions)}")

            # Show active positions with regimes
            for pos in positions:
                symbol = pos.symbol
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100

                # Get regime for this symbol
                regime = strategy.market_regime.get(symbol, 'unknown')
                sub_signals = strategy.sub_strategy_signals.get(symbol, {})

                print(f"   {symbol}: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | Regime: {regime}")

                # Show which strategies are bullish/bearish
                if sub_signals:
                    signals_str = ", ".join([
                        f"{name.replace('_', ' ').title()}: {data['signal']}"
                        for name, data in sub_signals.items()
                    ])
                    print(f"      Sub-signals: {signals_str}")

            print("-" * 80)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping ensemble strategy...")

    finally:
        # Show final stats
        account = await broker.get_account()
        final_equity = float(account.equity)

        print("\n" + "="*80)
        print("📊 ENSEMBLE STRATEGY SESSION COMPLETE")
        print("="*80)
        print(f"Final Equity: ${final_equity:,.2f}")

        positions = await broker.get_positions()
        if positions:
            print(f"\nOpen Positions: {len(positions)}")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                print(f"  {pos.symbol}: ${pnl:+,.2f}")

        print("\n✅ Ensemble strategy stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
