#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing Example

Demonstrates how to use Kelly Criterion for optimal position sizing.

The Kelly Criterion answers: "What's the optimal bet size to maximize growth?"

Benefits:
1. Mathematically optimal for long-term growth
2. Automatically adjusts size based on edge
3. Prevents over-betting (protects capital)
4. Adapts to changing performance

Risk:
- Full Kelly can be aggressive (large drawdowns)
- Solution: Use Half Kelly or Quarter Kelly

This example shows:
1. Basic Kelly calculation
2. Integration with trading strategy
3. Performance comparison (Fixed size vs Kelly)
"""

import logging
from datetime import datetime, timedelta

from utils.kelly_criterion import KellyCriterion, Trade

logger = logging.getLogger(__name__)


def example_1_basic_kelly_calculation():
    """
    Example 1: Basic Kelly Criterion calculation.

    Scenario: You have a trading system with:
    - 60% win rate
    - Average win: $300
    - Average loss: $100
    - Current capital: $100,000

    What's the optimal position size?
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Kelly Calculation")
    logger.info("=" * 80 + "\n")

    # Create Kelly calculator with Half Kelly (conservative)
    kelly = KellyCriterion(
        kelly_fraction=0.5,  # Half Kelly
        min_trades_required=1,  # Allow calculation immediately for example
        max_position_size=0.25,  # Cap at 25%
    )

    # Simulate some historical trades
    trades = [
        (True, 0.03),  # Win, +3%
        (True, 0.025),  # Win, +2.5%
        (False, -0.01),  # Loss, -1%
        (True, 0.04),  # Win, +4%
        (False, -0.01),  # Loss, -1%
        (True, 0.02),  # Win, +2%
        (False, -0.015),  # Loss, -1.5%
        (True, 0.035),  # Win, +3.5%
        (True, 0.03),  # Win, +3%
        (True, 0.02),  # Win, +2%
    ]

    # Add trades to Kelly calculator
    for i, (is_winner, pnl_pct) in enumerate(trades):
        trade = Trade(
            symbol="TEST",
            entry_time=datetime.now() - timedelta(days=len(trades) - i),
            exit_time=datetime.now() - timedelta(days=len(trades) - i - 1),
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct),
            quantity=10,
            pnl=1000 * pnl_pct,
            pnl_pct=pnl_pct,
            is_winner=is_winner,
        )
        kelly.add_trade(trade)

    # Get performance summary
    summary = kelly.get_performance_summary()

    logger.info("Performance Metrics:")
    logger.info(f"  Total trades: {summary['total_trades']}")
    logger.info(f"  Win rate: {summary['win_rate']:.1%}")
    logger.info(f"  Average win: {summary['avg_win']:.2%}")
    logger.info(f"  Average loss: {summary['avg_loss']:.2%}")
    logger.info(f"  Profit factor: {summary['profit_factor']:.2f}")
    logger.info(f"  Kelly fraction: {summary['kelly_fraction']:.1%}")
    logger.info(f"  Recommended position: {summary['recommended_position']:.1%}\n")

    # Calculate position size for $100k account
    capital = 100000
    position_value, position_fraction = kelly.calculate_position_size(
        current_capital=capital, current_price=150.0  # Assume $150/share
    )

    logger.info(f"\nFor ${capital:,} account:")
    logger.info(f"  Recommended position size: ${position_value:,.2f} ({position_fraction:.1%})")
    logger.info(f"  At $150/share: {position_value/150:.2f} shares\n")

    # Show sizing table
    print(kelly.get_recommended_sizes_table([50000, 100000, 200000, 500000]))


def example_2_compare_kelly_fractions():
    """
    Example 2: Compare Full Kelly vs Half Kelly vs Quarter Kelly.

    Shows how different Kelly fractions affect position sizing and risk.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Comparing Kelly Fractions")
    logger.info("=" * 80 + "\n")

    # Same trading performance for all
    trades = [
        (True, 0.04),
        (True, 0.03),
        (False, -0.02),
        (True, 0.05),
        (False, -0.01),
        (True, 0.03),
        (True, 0.04),
        (False, -0.02),
        (True, 0.06),
        (True, 0.02),
        (True, 0.04),
        (False, -0.015),
    ]

    # Test different Kelly fractions
    fractions = [
        (1.0, "Full Kelly (Aggressive)"),
        (0.5, "Half Kelly (Moderate)"),
        (0.25, "Quarter Kelly (Conservative)"),
    ]

    capital = 100000

    results = []
    for fraction, name in fractions:
        kelly = KellyCriterion(
            kelly_fraction=fraction,
            min_trades_required=1,
            max_position_size=0.50,  # Allow up to 50% for comparison
        )

        # Add trades
        for i, (is_winner, pnl_pct) in enumerate(trades):
            trade = Trade(
                symbol="TEST",
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100.0,
                exit_price=100.0 * (1 + pnl_pct),
                quantity=10,
                pnl=1000 * pnl_pct,
                pnl_pct=pnl_pct,
                is_winner=is_winner,
            )
            kelly.add_trade(trade)

        # Calculate position
        position_value, position_fraction = kelly.calculate_position_size(capital)

        results.append(
            {
                "name": name,
                "fraction": fraction,
                "position_pct": position_fraction,
                "position_value": position_value,
                "max_loss_2pct": position_value * 0.02,  # 2% stop loss
            }
        )

    # Display comparison
    logger.info("Comparison of Kelly Fractions:\n")
    logger.info(f"{'Strategy':<30} {'Position %':<15} {'Position $':<15} {'Max Loss*':<15}")
    logger.info("-" * 75)

    for r in results:
        logger.info(
            f"{r['name']:<30} "
            f"{r['position_pct']:>8.1%}       "
            f"${r['position_value']:>12,.2f}  "
            f"${r['max_loss_2pct']:>12,.2f}"
        )

    logger.info("\n* Assuming 2% stop loss\n")

    logger.info("Analysis:")
    logger.info("  - Full Kelly: Maximum growth potential, but highest volatility")
    logger.info("  - Half Kelly: Good balance of growth and risk")
    logger.info("  - Quarter Kelly: Most conservative, smoothest equity curve")
    logger.info("\nRecommendation: Start with Half Kelly or Quarter Kelly\n")


def example_3_adaptive_position_sizing():
    """
    Example 3: Adaptive position sizing based on recent performance.

    Demonstrates how Kelly automatically reduces size after losses
    and increases after wins.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Adaptive Position Sizing")
    logger.info("=" * 80 + "\n")

    kelly = KellyCriterion(
        kelly_fraction=0.5, min_trades_required=5, lookback_trades=10  # Only look at last 10 trades
    )

    capital = 100000

    # Scenario 1: Hot streak (60% win rate)
    logger.info("SCENARIO 1: Hot Streak (recent wins)")
    logger.info("-" * 40)

    trades_hot = [
        (True, 0.03),
        (True, 0.04),
        (False, -0.01),
        (True, 0.03),
        (True, 0.05),
        (True, 0.02),
    ]

    for i, (is_winner, pnl_pct) in enumerate(trades_hot):
        trade = Trade(
            symbol="TEST",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct),
            quantity=10,
            pnl=1000 * pnl_pct,
            pnl_pct=pnl_pct,
            is_winner=is_winner,
        )
        kelly.add_trade(trade)

    pos_value_hot, pos_frac_hot = kelly.calculate_position_size(capital)
    logger.info(f"Position size after hot streak: ${pos_value_hot:,.2f} ({pos_frac_hot:.1%})\n")

    # Scenario 2: Cold streak (adding losses)
    logger.info("SCENARIO 2: Cold Streak (recent losses)")
    logger.info("-" * 40)

    trades_cold = [
        (False, -0.02),
        (False, -0.01),
        (False, -0.015),
        (False, -0.02),
        (True, 0.01),
        (False, -0.01),
    ]

    for i, (is_winner, pnl_pct) in enumerate(trades_cold):
        trade = Trade(
            symbol="TEST",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct),
            quantity=10,
            pnl=1000 * pnl_pct,
            pnl_pct=pnl_pct,
            is_winner=is_winner,
        )
        kelly.add_trade(trade)

    pos_value_cold, pos_frac_cold = kelly.calculate_position_size(capital)
    logger.info(f"Position size after cold streak: ${pos_value_cold:,.2f} ({pos_frac_cold:.1%})\n")

    # Analysis
    reduction = ((pos_value_hot - pos_value_cold) / pos_value_hot) * 100

    logger.info("Analysis:")
    logger.info(f"  Kelly REDUCED position size by {reduction:.1f}% after losses")
    logger.info("  This protects capital during drawdowns")
    logger.info("  Position size will increase again as performance improves\n")


def example_4_negative_kelly_warning():
    """
    Example 4: What happens when you have no edge (negative Kelly).

    Demonstrates Kelly Criterion's warning system for losing strategies.
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Negative Kelly Warning (No Edge)")
    logger.info("=" * 80 + "\n")

    kelly = KellyCriterion(kelly_fraction=0.5, min_trades_required=1)

    # Simulate losing strategy (40% win rate, poor profit factor)
    trades_losing = [
        (True, 0.02),
        (False, -0.03),
        (False, -0.02),
        (False, -0.025),
        (True, 0.015),
        (False, -0.03),
        (False, -0.02),
        (True, 0.01),
        (False, -0.025),
        (True, 0.02),
        (False, -0.03),
        (False, -0.02),
    ]

    for i, (is_winner, pnl_pct) in enumerate(trades_losing):
        trade = Trade(
            symbol="TEST",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=100.0,
            exit_price=100.0 * (1 + pnl_pct),
            quantity=10,
            pnl=1000 * pnl_pct,
            pnl_pct=pnl_pct,
            is_winner=is_winner,
        )
        kelly.add_trade(trade)

    summary = kelly.get_performance_summary()

    logger.info("Losing Strategy Performance:")
    logger.info(f"  Win rate: {summary['win_rate']:.1%}")
    logger.info(f"  Profit factor: {summary['profit_factor']:.2f}")
    logger.info(f"  Kelly fraction: {summary['kelly_fraction']:.1%}\n")

    position_value, position_fraction = kelly.calculate_position_size(100000)

    logger.info(f"Recommended position: ${position_value:,.2f} ({position_fraction:.1%})\n")

    logger.info("⚠️  WARNING: Negative or very small Kelly indicates NO EDGE")
    logger.info("This strategy should NOT be traded!")
    logger.info("Fix the strategy before risking real capital.\n")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("\nKelly Criterion Position Sizing Examples")
    logger.info("=" * 80 + "\n")

    # Run all examples
    example_1_basic_kelly_calculation()
    print("\n" + "=" * 80 + "\n")

    example_2_compare_kelly_fractions()
    print("\n" + "=" * 80 + "\n")

    example_3_adaptive_position_sizing()
    print("\n" + "=" * 80 + "\n")

    example_4_negative_kelly_warning()

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("\nKey Takeaways:")
    logger.info("1. Kelly Criterion maximizes long-term growth mathematically")
    logger.info("2. Use Half Kelly (0.5) or Quarter Kelly (0.25) for safety")
    logger.info("3. Kelly automatically adapts to changing performance")
    logger.info("4. Negative Kelly warns when strategy has no edge")
    logger.info("5. Combine Kelly with stop losses and position limits")
    logger.info("\nRecommendation: Start with Quarter Kelly, increase as confidence builds\n")
