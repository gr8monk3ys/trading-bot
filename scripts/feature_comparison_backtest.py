#!/usr/bin/env python3
"""
Feature Comparison Backtest Script

Compares different feature configurations for MomentumStrategy:
1. Baseline (no advanced features)
2. RSI-2 aggressive mode
3. Kelly Criterion position sizing
4. Multi-Timeframe Analysis
5. Volatility Regime Detection
6. All features combined

Generates a comparison table and recommendations.
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_backtest import simple_backtest

# Feature configurations to test
CONFIGURATIONS = [
    {
        "name": "Baseline",
        "params": {},  # All advanced features OFF by default
        "description": "No advanced features (current production)",
    },
    {
        "name": "RSI-2 Aggressive",
        "params": {
            "rsi_mode": "aggressive",
        },
        "description": "RSI-2 with extreme thresholds (10/90)",
    },
    {
        "name": "Kelly Criterion",
        "params": {
            "use_kelly_criterion": True,
            "kelly_fraction": 0.5,
            "kelly_min_trades": 30,
        },
        "description": "Half-Kelly optimal position sizing",
    },
    {
        "name": "Multi-Timeframe",
        "params": {
            "use_multi_timeframe": True,
            "mtf_timeframes": ["5Min", "15Min", "1Hour"],
        },
        "description": "Signal alignment across timeframes",
    },
    {
        "name": "Volatility Regime",
        "params": {
            "use_volatility_regime": True,
        },
        "description": "VIX-based position/stop adjustment",
    },
    {
        "name": "All Features",
        "params": {
            "rsi_mode": "aggressive",
            "use_kelly_criterion": True,
            "kelly_fraction": 0.5,
            "use_multi_timeframe": True,
            "use_volatility_regime": True,
        },
        "description": "All advanced features enabled",
    },
]


async def run_comparison(
    symbols: list = None,
    start_date: str = "2024-08-01",
    end_date: str = "2024-11-01",
    initial_capital: int = 100000,
):
    """
    Run backtests for all configurations and compare results.

    Args:
        symbols: List of stock symbols (default: 5 tech stocks)
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]

    print("\n" + "=" * 100)
    print("FEATURE COMPARISON BACKTEST")
    print("=" * 100)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Configurations to test: {len(CONFIGURATIONS)}")
    print("=" * 100 + "\n")

    results = []

    for i, config in enumerate(CONFIGURATIONS):
        print(f"\n[{i+1}/{len(CONFIGURATIONS)}] Testing: {config['name']}")
        print(f"    Description: {config['description']}")
        print("-" * 60)

        try:
            result = await simple_backtest(
                symbols=symbols,
                start_date_str=start_date,
                end_date_str=end_date,
                initial_capital=initial_capital,
                use_slippage=True,
                strategy_params=config["params"],
                quiet=True,  # Suppress detailed output
            )

            if result:
                result["config"] = config
                results.append(result)
                print(f"    Return: {result['total_return']:+.2%}")
                print(f"    Sharpe: {result['sharpe_ratio']:.2f}")
                print(f"    Win Rate: {result['win_rate']:.1%}")
                print(f"    Trades: {result['num_trades']}")
            else:
                print("    ERROR: Backtest returned None")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Generate comparison table
    print("\n\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)

    if not results:
        print("\nNo results to compare!")
        return

    # Sort by Sharpe ratio
    results_sorted = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)

    # Header
    print(
        f"\n{'Configuration':<20} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10} {'Drawdown':>10} {'Trades':>8} {'Significant':>12}"
    )
    print("-" * 88)

    baseline_return = None
    for r in results_sorted:
        if r["config"]["name"] == "Baseline":
            baseline_return = r["total_return"]
            break

    for r in results_sorted:
        sig = "YES" if r["statistically_significant"] else "NO"

        # Calculate improvement vs baseline
        if baseline_return is not None and r["config"]["name"] != "Baseline":
            r["total_return"] - baseline_return
        else:
            pass

        print(
            f"{r['config']['name']:<20} {r['total_return']:>+9.2%} {r['sharpe_ratio']:>8.2f} {r['win_rate']:>9.1%} {r['max_drawdown']:>9.2%} {r['num_trades']:>8} {sig:>12}"
        )

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    # Find baseline
    baseline = next((r for r in results if r["config"]["name"] == "Baseline"), None)

    if baseline:
        print("\nBaseline Performance:")
        print(f"  Return: {baseline['total_return']:+.2%}")
        print(f"  Sharpe: {baseline['sharpe_ratio']:.2f}")

        improvements = []
        regressions = []

        for r in results:
            if r["config"]["name"] == "Baseline":
                continue

            # Compare to baseline
            return_diff = r["total_return"] - baseline["total_return"]
            sharpe_diff = r["sharpe_ratio"] - baseline["sharpe_ratio"]

            if sharpe_diff > 0.1 or return_diff > 0.01:  # Meaningful improvement
                improvements.append(
                    {
                        "name": r["config"]["name"],
                        "return_diff": return_diff,
                        "sharpe_diff": sharpe_diff,
                        "config": r["config"],
                    }
                )
            elif sharpe_diff < -0.1 or return_diff < -0.01:  # Regression
                regressions.append(
                    {
                        "name": r["config"]["name"],
                        "return_diff": return_diff,
                        "sharpe_diff": sharpe_diff,
                    }
                )

        if improvements:
            print("\n✅ ENABLE these features (improved performance):")
            for imp in sorted(improvements, key=lambda x: x["sharpe_diff"], reverse=True):
                print(
                    f"   - {imp['name']}: Return {imp['return_diff']:+.2%}, Sharpe {imp['sharpe_diff']:+.2f}"
                )
        else:
            print("\n⚠️  No features showed significant improvement")

        if regressions:
            print("\n❌ SKIP these features (hurt performance):")
            for reg in regressions:
                print(
                    f"   - {reg['name']}: Return {reg['return_diff']:+.2%}, Sharpe {reg['sharpe_diff']:+.2f}"
                )

    # Best configuration
    best = results_sorted[0]
    print(f"\n🏆 BEST CONFIGURATION: {best['config']['name']}")
    print(f"   Return: {best['total_return']:+.2%}")
    print(f"   Sharpe: {best['sharpe_ratio']:.2f}")
    print(f"   Win Rate: {best['win_rate']:.1%}")

    print("\n" + "=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print(
        """
1. If improvements found:
   - Update strategies/momentum_strategy.py default_parameters()
   - Enable the recommended features
   - Re-run backtest to confirm

2. If no improvements:
   - Keep current baseline configuration
   - Consider testing with different symbols or date ranges

3. Monitor paper trading with new configuration for 7+ days
"""
    )

    return results


if __name__ == "__main__":
    # Default configuration
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
    start = "2024-08-01"
    end = "2024-11-01"

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print(
                """
Feature Comparison Backtest

Usage:
    python scripts/feature_comparison_backtest.py [--symbols SYMS] [--start DATE] [--end DATE]

Examples:
    python scripts/feature_comparison_backtest.py
    python scripts/feature_comparison_backtest.py --symbols AAPL,MSFT,GOOGL
    python scripts/feature_comparison_backtest.py --start 2024-01-01 --end 2024-06-01
            """
            )
            sys.exit(0)

        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--symbols" and i + 1 < len(sys.argv):
                symbols = sys.argv[i + 1].split(",")
                i += 2
            elif sys.argv[i] == "--start" and i + 1 < len(sys.argv):
                start = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--end" and i + 1 < len(sys.argv):
                end = sys.argv[i + 1]
                i += 2
            else:
                i += 1

    # Run comparison
    asyncio.run(run_comparison(symbols=symbols, start_date=start, end_date=end))
