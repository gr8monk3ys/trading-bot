"""
Options Strategy Example

Demonstrates how to use the OptionsStrategy for income generation,
defined-risk trades, and portfolio hedging.

WARNING: Options trading is complex and risky. Only use with paper trading first!
Requires understanding of:
- Option mechanics (calls, puts, expiration, strike prices)
- Greeks (delta, theta, gamma, vega)
- Assignment risk
- Margin requirements

This example shows:
1. Basic options strategy setup
2. Parameter configuration
3. Strategy selection (covered calls, puts, spreads)
4. Risk management for options
"""

import asyncio
import logging
from datetime import datetime

from brokers.alpaca_broker import AlpacaBroker
from strategies.options_strategy import OptionsStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run options strategy example."""

    logger.info("=" * 80)
    logger.info("OPTIONS STRATEGY EXAMPLE")
    logger.info("=" * 80)
    logger.info("")

    # Initialize broker (PAPER TRADING ONLY for options!)
    broker = AlpacaBroker(paper=True)
    logger.info("✅ Initialized Alpaca broker (PAPER TRADING)")

    # Strategy 1: Conservative Options (Income Generation)
    # Best for: Generating income from existing stock positions
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY 1: Conservative Income Generation")
    logger.info("=" * 80)
    logger.info("")

    conservative_params = {
        'position_size': 0.10,
        'max_positions': 2,
        'option_allocation': 0.15,  # Only 15% in options

        # Enable only conservative strategies
        'enable_covered_calls': True,      # Sell calls against stock
        'enable_cash_secured_puts': True,  # Sell puts to enter positions
        'enable_call_spreads': False,
        'enable_put_spreads': False,
        'enable_iron_condor': False,
        'enable_protective_puts': True,    # Buy puts for insurance

        # Conservative strike selection
        'call_strike_otm_pct': 0.10,  # Sell calls 10% OTM (less likely to be called away)
        'put_strike_otm_pct': 0.10,   # Sell puts 10% OTM (more margin of safety)

        # Risk management
        'profit_target_pct': 0.50,  # Close at 50% of max profit
        'stop_loss_pct': 0.20,      # Stop at 20% loss
    }

    strategy_conservative = OptionsStrategy(
        broker=broker,
        symbols=['AAPL', 'MSFT'],  # Blue-chip stocks for covered calls
        parameters=conservative_params
    )

    await strategy_conservative.initialize()

    logger.info("Strategy Configuration:")
    logger.info(f"  Enabled: Covered Calls, Cash-Secured Puts, Protective Puts")
    logger.info(f"  Max allocation: 15%")
    logger.info(f"  Risk profile: CONSERVATIVE")
    logger.info("")

    # Strategy 2: Defined-Risk Directional (Spreads)
    # Best for: Directional trades with limited risk
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY 2: Defined-Risk Directional Trades")
    logger.info("=" * 80)
    logger.info("")

    spread_params = {
        'position_size': 0.08,
        'max_positions': 3,
        'option_allocation': 0.20,

        # Enable spread strategies
        'enable_covered_calls': False,
        'enable_cash_secured_puts': False,
        'enable_call_spreads': True,   # Bullish with defined risk
        'enable_put_spreads': True,    # Bearish with defined risk
        'enable_iron_condor': False,
        'enable_protective_puts': False,

        # Spread configuration
        'spread_width': 5,  # $5 spread width
        'call_strike_otm_pct': 0.02,  # Tighter strikes for spreads
        'put_strike_otm_pct': 0.02,

        # Only trade on high IV (better premiums for spreads)
        'high_iv_threshold': 0.30,  # 30% IV minimum

        # Risk management
        'profit_target_pct': 0.50,
        'stop_loss_pct': 0.30,
    }

    strategy_spreads = OptionsStrategy(
        broker=broker,
        symbols=['SPY', 'QQQ', 'IWM'],  # Liquid ETFs
        parameters=spread_params
    )

    await strategy_spreads.initialize()

    logger.info("Strategy Configuration:")
    logger.info(f"  Enabled: Call Spreads, Put Spreads")
    logger.info(f"  Max allocation: 20%")
    logger.info(f"  Spread width: $5")
    logger.info(f"  Risk profile: MODERATE")
    logger.info("")

    # Strategy 3: Advanced (Iron Condors for volatility selling)
    # Best for: Range-bound markets with high IV
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY 3: Advanced Volatility Selling")
    logger.info("=" * 80)
    logger.info("")

    advanced_params = {
        'position_size': 0.05,  # Smaller positions for advanced strategies
        'max_positions': 2,
        'option_allocation': 0.15,

        # Enable advanced strategies
        'enable_covered_calls': True,
        'enable_cash_secured_puts': True,
        'enable_call_spreads': True,
        'enable_put_spreads': True,
        'enable_iron_condor': True,  # Advanced - requires experience
        'enable_protective_puts': True,

        # Iron condor configuration
        'iron_condor_otm_pct': 0.15,  # 15% OTM for iron condor wings
        'spread_width': 10,  # Wider spreads for iron condors

        # Only trade on very high IV
        'high_iv_threshold': 0.40,  # 40% IV minimum

        # Risk management
        'profit_target_pct': 0.50,
        'stop_loss_pct': 0.30,
        'close_days_before_expiry': 10,  # Close 10 days before expiry
    }

    strategy_advanced = OptionsStrategy(
        broker=broker,
        symbols=['SPY'],  # Highly liquid for iron condors
        parameters=advanced_params
    )

    await strategy_advanced.initialize()

    logger.info("Strategy Configuration:")
    logger.info(f"  Enabled: ALL strategies including Iron Condor")
    logger.info(f"  Max allocation: 15%")
    logger.info(f"  Risk profile: ADVANCED")
    logger.info(f"  ⚠️  Requires experience with options!")
    logger.info("")

    # Show account info
    logger.info("\n" + "=" * 80)
    logger.info("ACCOUNT INFORMATION")
    logger.info("=" * 80)
    logger.info("")

    account = await broker.get_account()
    logger.info(f"Account Value: ${float(account.equity):,.2f}")
    logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
    logger.info(f"Cash: ${float(account.cash):,.2f}")
    logger.info("")

    # Important notes
    logger.info("\n" + "=" * 80)
    logger.info("IMPORTANT NOTES FOR OPTIONS TRADING")
    logger.info("=" * 80)
    logger.info("")
    logger.info("1. OPTIONS ARE COMPLEX:")
    logger.info("   - Require understanding of Greeks (delta, theta, gamma, vega)")
    logger.info("   - Time decay works against you (theta)")
    logger.info("   - Can expire worthless")
    logger.info("")
    logger.info("2. RISK MANAGEMENT:")
    logger.info("   - Never allocate more than 20% to options")
    logger.info("   - Use spreads for defined risk")
    logger.info("   - Close positions before expiration")
    logger.info("   - Monitor positions daily")
    logger.info("")
    logger.info("3. STRATEGY RECOMMENDATIONS:")
    logger.info("   - Beginners: Covered calls, cash-secured puts")
    logger.info("   - Intermediate: Call/put spreads")
    logger.info("   - Advanced: Iron condors")
    logger.info("")
    logger.info("4. PAPER TRADE FIRST:")
    logger.info("   - Test strategies for AT LEAST 3 months")
    logger.info("   - Verify you understand assignment risk")
    logger.info("   - Practice position management")
    logger.info("")
    logger.info("5. CURRENT IMPLEMENTATION:")
    logger.info("   - Strategy framework is complete")
    logger.info("   - Alpaca Options API integration pending")
    logger.info("   - Use for testing strategy logic")
    logger.info("")

    # Simulate one iteration (won't execute real trades yet)
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATING TRADING ITERATION")
    logger.info("=" * 80)
    logger.info("")

    try:
        # This will analyze symbols and select strategies
        # Actual option orders will not execute until API integration is complete
        await strategy_conservative.on_trading_iteration()
        logger.info("✅ Conservative strategy iteration complete")
    except Exception as e:
        logger.error(f"Error in conservative strategy: {e}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXAMPLE COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review strategy selection logic")
    logger.info("2. Test parameter configurations")
    logger.info("3. Complete Alpaca Options API integration")
    logger.info("4. Paper trade for 3+ months before live trading")


if __name__ == '__main__':
    asyncio.run(main())
