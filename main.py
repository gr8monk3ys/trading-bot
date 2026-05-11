#!/usr/bin/env python3
"""
Alpaca Trading Bot - Main Entry Point

Three modes:
- live:     start live (paper-by-default) trading
- backtest: run historical backtest
- optimize: grid-search strategy parameters

This is an experimental paper-trading sandbox. Real-money use is strongly
discouraged. See CLAUDE.md and README.md for the honest status.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path to allow imports from project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brokers.alpaca_broker import AlpacaBroker
from config import SYMBOLS
from engine.strategy_manager import StrategyManager
from utils.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """Configure application logging exactly once per process."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    root_logger = logging.getLogger()
    if root_logger.handlers:
        _LOGGING_CONFIGURED = True
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"),
        ],
    )
    _LOGGING_CONFIGURED = True


async def run_backtest(args) -> None:
    """Run backtest mode for one or more strategies."""
    strategy_manager = None
    try:
        logger.info(f"Starting backtest from {args.start_date} to {args.end_date}")

        broker = AlpacaBroker(paper=True)
        strategy_manager = StrategyManager(broker=broker)

        available_strategies = strategy_manager.get_available_strategy_names()
        logger.info(f"Available strategies: {available_strategies}")

        if args.strategy == "all":
            strategies_to_test = available_strategies
        elif args.strategy in available_strategies:
            strategies_to_test = [args.strategy]
        else:
            logger.error(f"Strategy '{args.strategy}' not found")
            return

        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        symbols = args.symbols.split(",") if args.symbols else SYMBOLS

        results: dict = {}
        metrics: dict = {}

        for strategy_name in strategies_to_test:
            try:
                logger.info(f"Backtesting strategy: {strategy_name}")
                strategy_class = strategy_manager.available_strategies[strategy_name]

                result = await strategy_manager.backtest_engine.run_backtest(
                    strategy_class=strategy_class,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=args.capital,
                    execution_profile=args.execution_profile,
                )

                strategy_metrics = strategy_manager.perf_metrics.calculate_metrics(result)
                results[strategy_name] = result
                metrics[strategy_name] = strategy_metrics

                print(f"\n--- {strategy_name} Performance Summary ---")
                print(f"Total Return:      {strategy_metrics['total_return']:.2%}")
                print(f"Annualized Return: {strategy_metrics['annualized_return']:.2%}")
                print(f"Sharpe Ratio:      {strategy_metrics['sharpe_ratio']:.2f}")
                print(f"Max Drawdown:      {strategy_metrics['max_drawdown']:.2%}")
                print(f"Win Rate:          {strategy_metrics['win_rate']:.2%}")
                print(f"Average Win:       {strategy_metrics['avg_win']:.2%}")
                print(f"Average Loss:      {strategy_metrics['avg_loss']:.2%}")
                print(f"Profit Factor:     {strategy_metrics['profit_factor']:.2f}")
                print(f"Number of Trades:  {strategy_metrics['num_trades']}")

            except Exception as e:
                logger.error(f"Error backtesting {strategy_name}: {e}", exc_info=True)

        if len(metrics) > 1:
            print("\n--- Strategy Comparison ---")
            comparison = pd.DataFrame(
                {
                    k: {
                        "Total Return": v["total_return"],
                        "Annualized Return": v["annualized_return"],
                        "Sharpe Ratio": v["sharpe_ratio"],
                        "Max Drawdown": v["max_drawdown"],
                        "Win Rate": v["win_rate"],
                        "Profit Factor": v["profit_factor"],
                        "Number of Trades": v["num_trades"],
                    }
                    for k, v in metrics.items()
                }
            ).T

            pd.set_option("display.float_format", "{:.2%}".format)
            for col in ["Sharpe Ratio", "Profit Factor", "Number of Trades"]:
                comparison[col] = pd.to_numeric(comparison[col])
                pd.set_option("display.float_format", "{:.2f}".format)
            print(comparison)

            if args.plot:
                try:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(12, 6))
                    for name, result in results.items():
                        curve = result.get("equity_curve") or result.get("portfolio_value")
                        if curve is None:
                            continue
                        plt.plot(curve, label=name)
                    plt.title("Equity Curves")
                    plt.xlabel("Date")
                    plt.ylabel("Portfolio Value")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig("backtest_equity_curves.png")
                    plt.close()
                    logger.info("Wrote backtest_equity_curves.png")
                except Exception as e:
                    logger.error(f"Error generating plots: {e}")

        logger.info("Backtest completed")

    except Exception as e:
        logger.error(f"Error in backtest mode: {e}", exc_info=True)
    finally:
        if strategy_manager is not None:
            strategy_manager.close()


async def run_live(args) -> None:
    """Run live (or paper) trading mode."""
    strategy_manager = None
    try:
        paper = not args.real

        if not paper:
            print("=" * 60)
            print("WARNING: LIVE TRADING WITH REAL MONEY")
            print("=" * 60)
            print("\nYou are about to start trading with REAL MONEY.")
            print("This will execute actual trades on your Alpaca account.")
            print("This bot has no proven edge and is not recommended for live capital.")
            confirmation = input("\nType 'CONFIRM LIVE TRADING' to proceed: ")
            if confirmation != "CONFIRM LIVE TRADING":
                print("\nLive trading cancelled.")
                return
            logger.warning("LIVE TRADING MODE ACTIVATED - REAL MONEY AT RISK")
        else:
            logger.info("Paper trading mode - no real money at risk")

        broker = AlpacaBroker(paper=paper)

        await broker.start_websocket(SYMBOLS if not args.symbols else args.symbols.split(","))
        logger.info("Websocket started")

        circuit_breaker = CircuitBreaker(
            max_daily_loss=0.03,
            auto_close_positions=True,
        )
        await circuit_breaker.initialize(broker)
        logger.info("Circuit breaker armed (3% daily loss limit)")

        market_status = await broker.get_market_status()
        logger.info(f"Market status: {market_status}")
        if not market_status.get("is_open", False) and not args.force:
            logger.warning("Market is closed. Use --force to run anyway.")
            print("Market is closed. Use --force to run anyway.")
            return

        strategy_manager = StrategyManager(
            broker=broker,
            max_strategies=args.max_strategies,
            max_allocation=args.max_allocation,
            circuit_breaker=circuit_breaker,
        )

        available_strategies = strategy_manager.get_available_strategy_names()
        logger.info(f"Available strategies: {available_strategies}")

        if args.strategy == "auto":
            await strategy_manager.evaluate_all_strategies()
            strategies_to_run = await strategy_manager.select_top_strategies(
                n=args.max_strategies, min_score=args.min_score
            )
        elif args.strategy == "all":
            strategies_to_run = available_strategies
        elif args.strategy in available_strategies:
            strategies_to_run = [args.strategy]
        else:
            logger.error(f"Strategy '{args.strategy}' not found")
            return

        logger.info(f"Selected strategies: {strategies_to_run}")

        if args.auto_allocate:
            allocations = await strategy_manager.optimize_allocations(strategies_to_run)
        else:
            equal_alloc = args.max_allocation / max(1, len(strategies_to_run))
            allocations = dict.fromkeys(strategies_to_run, equal_alloc)
        logger.info(f"Allocations: {allocations}")

        started = []
        for strategy_name in strategies_to_run:
            allocation = allocations.get(strategy_name, 0.1)
            symbols_to_use = args.symbols.split(",") if args.symbols else SYMBOLS
            success = await strategy_manager.start_strategy(
                strategy_name=strategy_name, allocation=allocation, symbols=symbols_to_use
            )
            if success:
                started.append(strategy_name)

        if not started:
            logger.error("Failed to start any strategies")
            return

        logger.info(f"Started {len(started)} strategies: {started}")

        try:
            logger.info("Trading bot running. Press Ctrl+C to stop.")
            check_counter = 0
            last_rebalance_hour = datetime.now().hour

            while True:
                await asyncio.sleep(1)
                check_counter += 1
                if check_counter >= 60:
                    check_counter = 0
                    if await circuit_breaker.check_and_halt():
                        logger.critical("CIRCUIT BREAKER TRIGGERED - daily loss limit hit")
                        await strategy_manager.stop_all_strategies(liquidate=True)
                        break

                current_hour = datetime.now().hour
                if (
                    len(started) > 1
                    and current_hour % 4 == 0
                    and current_hour != last_rebalance_hour
                ):
                    last_rebalance_hour = current_hour
                    try:
                        logger.info("Running portfolio rebalancing")
                        await strategy_manager.rebalance_strategies()
                    except Exception as e:
                        logger.error(f"Rebalance error: {e}")

        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        finally:
            await strategy_manager.stop_all_strategies(liquidate=args.liquidate_on_exit)
            try:
                await broker.stop_websocket()
            except Exception as e:
                logger.warning(f"Error stopping websocket: {e}")

    except Exception as e:
        logger.error(f"Error in live mode: {e}", exc_info=True)
    finally:
        if strategy_manager is not None:
            strategy_manager.close()


async def optimize_parameters(args) -> None:
    """Grid-search strategy parameters."""
    strategy_manager = None
    try:
        logger.info(f"Starting parameter optimization for {args.strategy}")
        broker = AlpacaBroker(paper=True)
        strategy_manager = StrategyManager(broker=broker)

        available_strategies = strategy_manager.get_available_strategy_names()
        if args.strategy not in available_strategies:
            logger.error(f"Strategy '{args.strategy}' not found")
            return

        strategy_class = strategy_manager.available_strategies[args.strategy]
        temp_strategy = strategy_class(broker=broker, symbols=[])
        default_params = temp_strategy.default_parameters()

        param_ranges: dict = {}
        if args.param_ranges:
            try:
                param_ranges = json.loads(args.param_ranges)
            except json.JSONDecodeError:
                logger.error("Invalid JSON for parameter ranges")
                return

        if not param_ranges:
            for param, value in default_params.items():
                if isinstance(value, (int, float)) and param != "allocation":
                    if isinstance(value, int):
                        param_ranges[param] = {
                            "min": max(1, int(value * 0.5)),
                            "max": int(value * 1.5),
                            "step": 1,
                        }
                    else:
                        param_ranges[param] = {
                            "min": value * 0.5,
                            "max": value * 1.5,
                            "step": value * 0.1,
                        }

        logger.info(f"Parameter ranges: {param_ranges}")

        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        symbols = args.symbols.split(",") if args.symbols else SYMBOLS

        import itertools

        param_values: dict = {}
        for param, range_info in param_ranges.items():
            min_val = range_info["min"]
            max_val = range_info["max"]
            step = range_info["step"]
            if isinstance(min_val, int) and isinstance(max_val, int):
                values = list(range(min_val, max_val + 1, step))
            else:
                values: list = []
                val = min_val
                while val <= max_val:
                    values.append(val)
                    val += step
            param_values[param] = values

        param_names = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[p] for p in param_names]))
        logger.info(f"Testing {len(combinations)} parameter combinations")

        results: list = []
        for i, combo in enumerate(combinations):
            params = default_params.copy()
            for j, param in enumerate(param_names):
                params[param] = combo[j]
            logger.info(f"Combination {i + 1}/{len(combinations)}: {params}")

            result = await strategy_manager.backtest_engine.run_backtest(
                strategy_class=strategy_class,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=args.capital,
                strategy_params=params,
                execution_profile=args.execution_profile,
            )
            metrics = strategy_manager.perf_metrics.calculate_metrics(result)
            results.append({"params": params, "metrics": metrics})

        if args.optimize_for == "return":
            best_result = max(results, key=lambda x: x["metrics"]["total_return"])
        elif args.optimize_for == "drawdown":
            best_result = min(results, key=lambda x: x["metrics"]["max_drawdown"])
        else:
            best_result = max(results, key=lambda x: x["metrics"]["sharpe_ratio"])

        print("\n--- Parameter Optimization Results ---")
        print(f"Best parameters for {args.strategy} optimized for {args.optimize_for}:")
        for param, value in best_result["params"].items():
            if param in param_ranges:
                print(f"  {param}: {value}")

        print("\nPerformance:")
        print(f"  Total Return: {best_result['metrics']['total_return']:.2%}")
        print(f"  Sharpe Ratio: {best_result['metrics']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {best_result['metrics']['max_drawdown']:.2%}")
        print(f"  Win Rate:     {best_result['metrics']['win_rate']:.2%}")

        output_file = f"{args.strategy}_optimized_params.json"
        with open(output_file, "w") as f:
            json.dump(
                {"optimized_params": best_result["params"], "performance": best_result["metrics"]},
                f,
                indent=4,
                default=str,
            )
        logger.info(f"Wrote optimized parameters to {output_file}")

    except Exception as e:
        logger.error(f"Error in optimize mode: {e}", exc_info=True)
    finally:
        if strategy_manager is not None:
            strategy_manager.close()


def main() -> None:
    """Entry point for the trading bot."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Alpaca Trading Bot")

    parser.add_argument(
        "mode",
        choices=["live", "backtest", "optimize"],
        help="Operation mode",
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        help='Strategy to use (name, "all", or "auto")',
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols")

    # Live options
    parser.add_argument("--real", action="store_true", help="Use real trading (NOT recommended)")
    parser.add_argument("--force", action="store_true", help="Force execution if market closed")
    parser.add_argument("--max-strategies", type=int, default=3)
    parser.add_argument("--max-allocation", type=float, default=0.9)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--auto-allocate", action="store_true")
    parser.add_argument("--liquidate-on-exit", action="store_true")

    # Backtest options
    parser.add_argument(
        "--start-date",
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--execution-profile",
        choices=["idealistic", "realistic", "stressed"],
        default="realistic",
    )

    # Optimize options
    parser.add_argument("--param-ranges", default=None, help="JSON object of parameter ranges")
    parser.add_argument(
        "--optimize-for",
        choices=["sharpe", "return", "drawdown"],
        default="sharpe",
    )

    args = parser.parse_args()

    try:
        if args.mode == "live":
            asyncio.run(run_live(args))
        elif args.mode == "backtest":
            asyncio.run(run_backtest(args))
        elif args.mode == "optimize":
            asyncio.run(optimize_parameters(args))
    except Exception as e:
        logger.error(f"Error running {args.mode} mode: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
