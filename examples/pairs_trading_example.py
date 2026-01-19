#!/usr/bin/env python3
"""
Pairs Trading Strategy Example

Demonstrates market-neutral statistical arbitrage by trading cointegrated
stock pairs. Long one stock, short the other to profit from spread reversion.

Expected Performance:
- Sharpe Ratio: 0.80-1.20 (highest from research)
- Market-Neutral: Low correlation to market
- Best For: All market conditions

Common Pairs:
- KO/PEP (Coca-Cola / PepsiCo)
- GM/F (General Motors / Ford)
- WMT/TGT (Walmart / Target)
- JPM/BAC (JPMorgan / Bank of America)

Usage:
    python examples/pairs_trading_example.py
"""

import asyncio
import logging

from brokers.alpaca_broker import AlpacaBroker
from strategies.pairs_trading_strategy import PairsTradingStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def main():
    """Run pairs trading strategy example."""
    print("\n" + "=" * 80)
    print("🧪 PAIRS TRADING STRATEGY EXAMPLE")
    print("=" * 80 + "\n")

    # Initialize broker (paper trading)
    broker = AlpacaBroker(paper=True)

    # Define stock pairs to trade
    # Format: (stock1, stock2)
    # Should be companies in same sector with high correlation
    pairs = [
        ("KO", "PEP"),  # Coca-Cola / PepsiCo (beverages)
        ("JPM", "BAC"),  # JPMorgan / Bank of America (banks)
        ("WMT", "TGT"),  # Walmart / Target (retail)
    ]

    print("📊 Strategy Configuration:")
    print(f"   Pairs to trade: {len(pairs)}")
    for pair in pairs:
        print(f"      {pair[0]} / {pair[1]}")
    print("\n   Entry Z-Score: ±2.0 (enter when spread is 2 std devs from mean)")
    print("   Exit Z-Score: ±0.5 (exit when spread reverts)")
    print("   Market-Neutral: Yes (long + short = hedged)")
    print("   Position Size: 10% per pair (split between both stocks)")
    print("\n")

    # Initialize strategy with custom parameters
    strategy = PairsTradingStrategy(
        broker=broker,
        symbols=pairs,  # Pass pairs instead of individual symbols
        parameters={
            "position_size": 0.10,  # 10% per PAIR (5% each stock)
            "max_pairs": 2,  # Max 2 concurrent pairs
            # Statistical parameters
            "lookback_period": 60,  # 60 days for cointegration
            "min_correlation": 0.70,  # Min 70% correlation
            "cointegration_pvalue": 0.05,  # 5% p-value threshold
            # Entry/exit thresholds
            "entry_z_score": 2.0,  # Enter when |z| > 2
            "exit_z_score": 0.5,  # Exit when |z| < 0.5
            "stop_loss_z_score": 3.5,  # Stop if |z| > 3.5
            # Position management
            "max_holding_days": 10,  # Max 10 days per pair
            "take_profit_pct": 0.04,  # 4% profit target
            "stop_loss_pct": 0.03,  # 3% stop loss
        },
    )

    # Initialize strategy
    if not await strategy.initialize():
        print("❌ Failed to initialize strategy")
        return

    print("✅ Strategy initialized successfully")
    print("\n" + "=" * 80)
    print("📈 PAIRS TRADING STARTED")
    print("=" * 80)
    print("\nHow it works:")
    print("1. Tests each pair for cointegration (stable relationship)")
    print("2. Calculates spread = stock1 - (hedge_ratio * stock2)")
    print("3. Normalizes spread to z-score (std deviations from mean)")
    print("4. Entry signals:")
    print("   - Z-score > 2.0: Spread too wide → SHORT spread")
    print("     (Sell expensive stock, buy cheap stock)")
    print("   - Z-score < -2.0: Spread too narrow → LONG spread")
    print("     (Buy cheap stock, sell expensive stock)")
    print("5. Exit when spread reverts (z-score → 0)")
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

            print("-" * 80)
            print(f"💰 Equity: ${equity:,.2f} | Active Pairs: {len(strategy.pair_positions)}")

            # Show cointegration status
            print("\n📊 Pair Cointegration Status:")
            for pair in pairs:
                coint = strategy.cointegration_results.get(pair)
                if coint:
                    status = (
                        "✅ COINTEGRATED" if coint.get("cointegrated") else "❌ NOT COINTEGRATED"
                    )
                    corr = coint.get("correlation", 0)
                    pval = coint.get("coint_pvalue", 1)
                    hedge = coint.get("hedge_ratio", 0)

                    print(f"   {pair[0]}/{pair[1]}: {status}")
                    print(
                        f"      Correlation: {corr:.3f} | P-value: {pval:.4f} | Hedge: {hedge:.4f}"
                    )

                    # Show current z-score if available
                    stats = strategy.pair_stats.get(pair)
                    if stats and "z_score" in stats:
                        z = stats["z_score"]
                        signal = "LONG spread" if z < -2 else "SHORT spread" if z > 2 else "neutral"
                        print(f"      Z-Score: {z:.2f} → {signal}")

            # Show active pair positions
            if strategy.pair_positions:
                print("\n💼 Active Pair Positions:")
                for pair, position in strategy.pair_positions.items():
                    symbol1 = position["symbol1"]
                    symbol2 = position["symbol2"]
                    entry_z = position["entry_z_score"]

                    # Calculate current P/L
                    price1 = strategy.current_prices.get(symbol1)
                    price2 = strategy.current_prices.get(symbol2)

                    if price1 and price2:
                        entry_price1 = position["price1"]
                        entry_price2 = position["price2"]
                        quantity1 = position["quantity1"]
                        quantity2 = position["quantity2"]
                        side1 = position["side1"]
                        side2 = position["side2"]

                        # P/L calculation
                        if side1 == "buy":
                            pnl1 = (price1 - entry_price1) * quantity1
                        else:
                            pnl1 = (entry_price1 - price1) * quantity1

                        if side2 == "buy":
                            pnl2 = (price2 - entry_price2) * quantity2
                        else:
                            pnl2 = (entry_price2 - price2) * quantity2

                        total_pnl = pnl1 + pnl2
                        entry_value = entry_price1 * quantity1 + entry_price2 * quantity2
                        pnl_pct = (total_pnl / entry_value * 100) if entry_value > 0 else 0

                        # Current z-score
                        stats = strategy.pair_stats.get(pair)
                        current_z = stats["z_score"] if stats else 0

                        print(f"\n   {symbol1}/{symbol2}:")
                        print(f"      P/L: ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
                        print(f"      Z-Score: {entry_z:.2f} → {current_z:.2f}")
                        print(f"      {side1.upper()} {symbol1}: {quantity1:.2f} @ ${price1:.2f}")
                        print(f"      {side2.upper()} {symbol2}: {quantity2:.2f} @ ${price2:.2f}")

            print("-" * 80)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping pairs trading strategy...")

    finally:
        # Show final stats
        account = await broker.get_account()
        final_equity = float(account.equity)

        print("\n" + "=" * 80)
        print("📊 PAIRS TRADING SESSION COMPLETE")
        print("=" * 80)
        print(f"Final Equity: ${final_equity:,.2f}")

        if strategy.pair_positions:
            print(f"\nOpen Pair Positions: {len(strategy.pair_positions)}")
            for pair in strategy.pair_positions:
                print(f"  {pair[0]}/{pair[1]}")

        print("\n✅ Pairs trading strategy stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
