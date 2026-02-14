#!/usr/bin/env python3
"""
Backtest all strategies to compare performance
"""

import asyncio

from brokers.alpaca_broker import AlpacaBroker
from engine.backtest_engine import BacktestEngine
from strategies.bracket_momentum_strategy import BracketMomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy

# Test configuration
START_DATE = "2024-08-01"
END_DATE = "2024-10-31"
INITIAL_CAPITAL = 100000
SYMBOLS = ["AAPL", "MSFT", "AMZN", "META", "TSLA"]


async def backtest_strategy(strategy_class, name):
    """Backtest a single strategy"""
    print(f"\n{'='*80}")
    print(f"Backtesting: {name}")
    print(f"{'='*80}")

    try:
        # Create broker and strategy
        broker = AlpacaBroker(paper=True)
        strategy = strategy_class(broker=broker)
        await strategy.initialize(symbols=SYMBOLS)

        # Create backtest engine
        engine = BacktestEngine(broker=broker)

        # Convert date strings to datetime objects
        from datetime import datetime as dt

        start = dt.strptime(START_DATE, "%Y-%m-%d")
        end = dt.strptime(END_DATE, "%Y-%m-%d")

        # Run backtest
        results_list = await engine.run(strategies=[strategy], start_date=start, end_date=end)

        # Extract metrics from the result DataFrame
        if results_list and len(results_list) > 0:
            result_df = results_list[0]

            # Get metrics from DataFrame attributes
            sharpe_ratio = result_df.attrs.get("sharpe_ratio", 0)
            max_drawdown = result_df.attrs.get("max_drawdown", 0)
            annualized_return = result_df.attrs.get("annualized_return", 0)

            # Calculate additional metrics from the DataFrame
            total_return = result_df["cum_returns"].iloc[-1] if len(result_df) > 0 else 0
            final_equity = result_df["equity"].iloc[-1] if len(result_df) > 0 else INITIAL_CAPITAL
            total_trades = result_df["trades"].sum() if "trades" in result_df.columns else 0

            # Calculate win rate from returns
            winning_days = (result_df["returns"] > 0).sum() if "returns" in result_df.columns else 0
            total_days = (result_df["returns"] != 0).sum() if "returns" in result_df.columns else 1
            win_rate = winning_days / total_days if total_days > 0 else 0

            print(f"\n{name} Results:")
            print(f"  Total Return: {total_return:.2%}")
            print(f"  Annualized Return: {annualized_return:.2%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {max_drawdown:.2%}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Total Trades: {int(total_trades)}")
            print(f"  Final Equity: ${final_equity:,.2f}")

            # Return metrics as dict
            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": int(total_trades),
                "final_equity": final_equity,
            }

        return None

    except Exception as e:
        print(f"  ❌ Error backtesting {name}: {e}")
        import traceback

        traceback.print_exc()
        return None


async def main():
    """Run backtests on all strategies"""
    print("\n" + "=" * 80)
    print("BACKTESTING ALL STRATEGIES")
    print("=" * 80)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print("=" * 80)

    strategies_to_test = [
        (MomentumStrategy, "Momentum Strategy"),
        (MeanReversionStrategy, "Mean Reversion Strategy"),
        (BracketMomentumStrategy, "Bracket Momentum Strategy"),
    ]

    results = {}

    for strategy_class, name in strategies_to_test:
        result = await backtest_strategy(strategy_class, name)
        if result:
            results[name] = result
        await asyncio.sleep(1)  # Brief pause between tests

    # Summary comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(
        f"{'Strategy':<30} {'Return':<12} {'Sharpe':<10} {'Drawdown':<12} {'Win Rate':<10} {'Trades':<8}"
    )
    print("-" * 80)

    for name, result in results.items():
        print(
            f"{name:<30} {result.get('total_return', 0):>10.2%}  "
            f"{result.get('sharpe_ratio', 0):>8.2f}  "
            f"{result.get('max_drawdown', 0):>10.2%}  "
            f"{result.get('win_rate', 0):>8.1%}  "
            f"{result.get('total_trades', 0):>6}"
        )

    print("=" * 80)

    # Rank by Sharpe ratio
    if results:
        ranked = sorted(results.items(), key=lambda x: x[1].get("sharpe_ratio", 0), reverse=True)
        print("\nRanking by Sharpe Ratio:")
        for i, (name, result) in enumerate(ranked, 1):
            print(f"  {i}. {name}: {result.get('sharpe_ratio', 0):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
