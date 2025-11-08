#!/usr/bin/env python3
"""
Portfolio Rebalancing Example

Demonstrates how to use the portfolio rebalancer to maintain target allocations.

This example shows:
1. Equal-weight portfolio (4 stocks, 25% each)
2. Automatic rebalancing when positions drift > 5%
3. Weekly rebalancing schedule
4. Detailed reporting

Real-world benefits:
- Discipline: Automatically sell winners, buy losers
- Risk control: Prevents over-concentration
- Mean reversion: Buy low, sell high systematically
- Reduced volatility: Maintains balanced risk profile
"""

import logging
import asyncio
from datetime import datetime
from utils.portfolio_rebalancer import PortfolioRebalancer

logger = logging.getLogger(__name__)


async def example_equal_weight_rebalancing():
    """
    Example 1: Equal-weight portfolio with automatic rebalancing.

    Goal: Keep 4 stocks at 25% each, rebalance weekly if drift > 5%
    """
    from brokers.alpaca_broker import AlpacaBroker

    # Initialize broker
    broker = AlpacaBroker(paper=True)

    # Define portfolio
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    logger.info("="*80)
    logger.info("EXAMPLE 1: Equal-Weight Portfolio Rebalancing")
    logger.info("="*80)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info("Target: 25% each")
    logger.info("Rebalance: Weekly, when drift > 5%")
    logger.info("="*80 + "\n")

    # Create rebalancer with equal weighting
    rebalancer = PortfolioRebalancer(
        broker=broker,
        equal_weight_symbols=symbols,
        rebalance_threshold=0.05,  # 5% drift threshold
        rebalance_frequency='weekly',
        min_trade_size=50.0,  # Minimum $50 trade
        dry_run=False  # Set True to test without executing
    )

    # Check current state
    logger.info("Current portfolio status:")
    report = await rebalancer.get_rebalance_report()
    print(report)

    # Check if rebalancing needed
    if await rebalancer.needs_rebalancing():
        logger.info("\n⚠️  Rebalancing needed!")

        # Generate orders
        orders = await rebalancer.generate_rebalance_orders()

        logger.info(f"\nGenerated {len(orders)} rebalancing orders:")
        for order in orders:
            logger.info(f"  {order['side'].upper()} {order['quantity']:.2f} {order['symbol']} @ ${order['price']:.2f}")
            logger.info(f"    Reason: {order['reason']}")

        # Ask for confirmation (in production, this would be automatic)
        response = input("\nExecute these orders? (yes/no): ")

        if response.lower() == 'yes':
            result = await rebalancer.execute_rebalancing(orders)
            logger.info(f"\nRebalancing result: {result}")

            # Show updated state
            logger.info("\nUpdated portfolio status:")
            report = await rebalancer.get_rebalance_report()
            print(report)
        else:
            logger.info("Rebalancing cancelled")
    else:
        logger.info("\n✅ Portfolio is balanced - no action needed")


async def example_custom_allocation_rebalancing():
    """
    Example 2: Custom allocation portfolio.

    Goal: Maintain specific target weights for each position
    """
    from brokers.alpaca_broker import AlpacaBroker

    # Initialize broker
    broker = AlpacaBroker(paper=True)

    # Define custom allocations (must sum to 1.0)
    target_allocations = {
        'AAPL': 0.30,   # 30% in Apple
        'MSFT': 0.30,   # 30% in Microsoft
        'GOOGL': 0.25,  # 25% in Google
        'AMZN': 0.15,   # 15% in Amazon
    }

    logger.info("="*80)
    logger.info("EXAMPLE 2: Custom Allocation Portfolio")
    logger.info("="*80)
    logger.info("Target allocations:")
    for symbol, weight in target_allocations.items():
        logger.info(f"  {symbol}: {weight:.0%}")
    logger.info("="*80 + "\n")

    # Create rebalancer with custom allocations
    rebalancer = PortfolioRebalancer(
        broker=broker,
        target_allocations=target_allocations,
        rebalance_threshold=0.03,  # Tighter 3% threshold
        rebalance_frequency='daily',
        min_trade_size=100.0,
        dry_run=False
    )

    # Check and rebalance
    report = await rebalancer.get_rebalance_report()
    print(report)

    if await rebalancer.needs_rebalancing():
        orders = await rebalancer.generate_rebalance_orders()
        result = await rebalancer.execute_rebalancing(orders)
        logger.info(f"Result: {result}")


async def example_continuous_rebalancing():
    """
    Example 3: Continuous monitoring and automatic rebalancing.

    Monitors portfolio every hour and rebalances when needed.
    """
    from brokers.alpaca_broker import AlpacaBroker

    # Initialize broker
    broker = AlpacaBroker(paper=True)

    # Equal weight portfolio
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    logger.info("="*80)
    logger.info("EXAMPLE 3: Continuous Automatic Rebalancing")
    logger.info("="*80)
    logger.info(f"Monitoring {', '.join(symbols)}")
    logger.info("Checking every hour, rebalancing daily if drift > 5%")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*80 + "\n")

    # Create rebalancer
    rebalancer = PortfolioRebalancer(
        broker=broker,
        equal_weight_symbols=symbols,
        rebalance_threshold=0.05,
        rebalance_frequency='daily',
        min_trade_size=50.0,
        dry_run=False
    )

    # Continuous monitoring loop
    try:
        while True:
            logger.info(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking portfolio...")

            # Get current state
            report = await rebalancer.get_rebalance_report()
            print(report)

            # Rebalance if needed
            if await rebalancer.needs_rebalancing():
                logger.info("\n⚠️  Rebalancing triggered!")
                orders = await rebalancer.generate_rebalance_orders()
                result = await rebalancer.execute_rebalancing(orders)
                logger.info(f"Rebalancing complete: {result}")
            else:
                logger.info("✅ Portfolio balanced - no action needed")

            # Wait 1 hour before next check
            logger.info("\nNext check in 1 hour...")
            await asyncio.sleep(3600)  # 1 hour

    except KeyboardInterrupt:
        logger.info("\n\nStopping continuous rebalancing...")


async def example_backtest_rebalancing_benefit():
    """
    Example 4: Demonstrate the benefit of rebalancing vs buy-and-hold.

    Compares:
    - Portfolio WITH monthly rebalancing
    - Portfolio WITHOUT rebalancing (buy and hold)
    """
    logger.info("="*80)
    logger.info("EXAMPLE 4: Rebalancing Benefit Analysis")
    logger.info("="*80)
    logger.info("Comparing rebalanced portfolio vs buy-and-hold")
    logger.info("="*80 + "\n")

    # This would require historical data and backtesting framework
    # Placeholder for future implementation
    logger.info("TODO: Implement backtest comparison")
    logger.info("Expected benefits of rebalancing:")
    logger.info("  1. Lower volatility (smoother returns)")
    logger.info("  2. Enforced discipline (sell high, buy low)")
    logger.info("  3. Prevents over-concentration in winners")
    logger.info("  4. Captures mean reversion premium")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run examples
    print("\nPortfolio Rebalancing Examples")
    print("="*80)
    print("Choose an example:")
    print("1. Equal-weight portfolio (one-time rebalance)")
    print("2. Custom allocation portfolio")
    print("3. Continuous automatic rebalancing")
    print("4. Backtest rebalancing benefit (TODO)")
    print("="*80)

    choice = input("\nEnter choice (1-4): ")

    if choice == '1':
        asyncio.run(example_equal_weight_rebalancing())
    elif choice == '2':
        asyncio.run(example_custom_allocation_rebalancing())
    elif choice == '3':
        asyncio.run(example_continuous_rebalancing())
    elif choice == '4':
        asyncio.run(example_backtest_rebalancing_benefit())
    else:
        print("Invalid choice")
