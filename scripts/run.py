#!/usr/bin/env python3
"""
Trading Bot CLI Runner

A command-line tool to run the trading bot with various options
including live trading, backtesting, and performance visualization.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.sentiment_strategy import SentimentStockStrategy as SentimentStrategy

from brokers.backtest_broker import BacktestBroker
from config import ALPACA_CREDS, SYMBOLS, TRADING_PARAMS
from engine.backtest_engine import BacktestEngine
from engine.strategy_manager import StrategyManager
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from utils.visualization import create_performance_report

# Define available strategies
STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "sentiment": SentimentStrategy,
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"data/logs/trading_bot_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


async def run_live_trading(strategy_names, symbols=None, use_paper=True):
    """Run live trading with selected strategies"""
    logger.info(f"Starting live trading with strategies: {', '.join(strategy_names)}")

    # Create broker
    broker = BacktestBroker(
        api_key=ALPACA_CREDS["API_KEY"], api_secret=ALPACA_CREDS["API_SECRET"], paper=use_paper
    )

    # Create strategy instances
    strategies = []
    for name in strategy_names:
        if name in STRATEGIES:
            strategy_class = STRATEGIES[name]
            strategy = strategy_class(broker=broker)
            strategies.append(strategy)
            logger.info(f"Initialized {name} strategy")
        else:
            logger.warning(f"Unknown strategy: {name}, skipping")

    if not strategies:
        logger.error("No valid strategies specified. Exiting.")
        return

    # Use specified symbols or default
    trading_symbols = symbols if symbols else SYMBOLS
    logger.info(f"Trading on symbols: {', '.join(trading_symbols)}")

    # Create strategy manager
    manager = StrategyManager(broker=broker, strategies=strategies, symbols=trading_symbols)

    # Initialize and run
    try:
        await manager.initialize()
        logger.info(f"Running in {'paper' if use_paper else 'live'} trading mode")

        # Trading loop with clean shutdown handling
        check_interval = TRADING_PARAMS["INTERVAL"]
        running = True

        def signal_handler(sig, frame):
            nonlocal running
            logger.info("Shutdown signal received. Closing positions and exiting...")
            running = False

        # Register signal handlers
        import signal

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while running:
            try:
                logger.info(f"Running strategy update at {datetime.now().strftime('%H:%M:%S')}")
                await manager.update()
                logger.info(f"Sleeping for {check_interval} seconds")
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(check_interval)

        # Clean shutdown
        logger.info("Closing all positions...")
        await manager.close_all_positions()
        logger.info("Trading bot shutdown complete")

    except Exception as e:
        logger.error(f"Error starting trading: {e}", exc_info=True)


async def run_backtest(strategy_names, days=30, symbols=None, output_dir=None):
    """
    Run backtest with selected strategies.

    Args:
        strategy_names: List of strategy names to run
        days: Number of days to backtest
        symbols: List of symbols to trade
        output_dir: Directory to save results

    Returns:
        Directory where results are saved
    """
    # Parse strategy names
    if isinstance(strategy_names, str):
        strategy_names = [s.strip() for s in strategy_names.split(",")]

    logger.info(f"Starting backtest with strategies: {', '.join(strategy_names)}")

    # Set dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()} ({days} days)")

    # Create broker
    broker = BacktestBroker(initial_balance=100000)

    # Set symbols
    if symbols is None:
        symbols = SYMBOLS
    elif isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]

    # Create and run strategies
    strategies = []
    for strategy_name in strategy_names:
        strategy_class = _get_strategy_class(strategy_name)
        if not strategy_class:
            logger.error(f"Strategy '{strategy_name}' not found")
            continue

        strategy = strategy_class()
        strategy._legacy_initialize(symbols=symbols)
        strategies.append(strategy)

    # Create engine
    engine = BacktestEngine(broker=broker)

    # Load historical data for each symbol
    for symbol in symbols:
        # Get historical data for the symbol
        historical_data = await _get_historical_data(symbol, start_date, end_date)
        broker.set_price_data(symbol, historical_data)

    # Run backtest
    results = await engine.run(strategies=strategies, start_date=start_date, end_date=end_date)

    # Generate output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_slug = "_".join(strategy_names)
    results_dir = output_dir or f"./results/backtest_{strategy_slug}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save results
    for i, result in enumerate(results):
        strategy_name = strategy_names[i]
        result_file = os.path.join(results_dir, f"{strategy_name}_result.csv")
        result.to_csv(result_file)
        logger.info(f"Saved results for {strategy_name} to {result_file}")

    # Generate performance report
    report_file = os.path.join(results_dir, "performance_report.html")
    create_performance_report(results, strategy_names, report_file)
    logger.info(f"Generated performance report at {report_file}")

    return results_dir


async def _get_historical_data(symbol, start_date, end_date):
    """Get historical data for a symbol."""
    # For simplicity, generate some mock data
    days = (end_date - start_date).days
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    # Generate random prices with upward trend
    np.random.seed(42 + hash(symbol) % 100)  # Consistent but different for each symbol
    price = 100 + np.random.rand() * 100  # Random start price between 100-200
    daily_returns = np.random.normal(0.0005, 0.015, len(dates))  # Slight upward bias
    prices = price * (1 + pd.Series(daily_returns)).cumprod()

    # Create OHLC data
    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "high": prices * (1 + np.random.uniform(0.001, 0.02, len(dates))),
            "low": prices * (1 - np.random.uniform(0.001, 0.02, len(dates))),
            "close": prices,
            "volume": np.random.randint(100000, 10000000, len(dates)),
        },
        index=dates,
    )

    # Ensure high is always highest and low is always lowest
    data["high"] = data[["open", "close", "high"]].max(axis=1)
    data["low"] = data[["open", "close", "low"]].min(axis=1)

    return data


def _get_strategy_class(strategy_name):
    return STRATEGIES.get(strategy_name)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Alpaca Trading Bot CLI")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--live", action="store_true", help="Run in live trading mode")
    mode_group.add_argument("--backtest", action="store_true", help="Run in backtest mode")

    # Strategy selection
    parser.add_argument(
        "--strategies",
        type=str,
        default="momentum",
        help="Comma-separated list of strategies to use (momentum,mean_reversion,sentiment)",
    )

    # Symbol selection
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to trade (defaults to config.py symbols)",
    )

    # Backtest options
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days to backtest (default: 30)"
    )
    parser.add_argument("--output", type=str, help="Output directory for backtest results")

    # Live trading options
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real trading instead of paper trading (USE WITH CAUTION)",
    )

    args = parser.parse_args()

    # Process arguments
    strategy_names = [s.strip() for s in args.strategies.split(",")]
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    # Run in the selected mode
    try:
        if args.live:
            asyncio.run(
                run_live_trading(
                    strategy_names=strategy_names, symbols=symbols, use_paper=not args.real
                )
            )
        elif args.backtest:
            results_dir = asyncio.run(
                run_backtest(
                    strategy_names=strategy_names,
                    days=args.days,
                    symbols=symbols,
                    output_dir=args.output,
                )
            )
            print(f"\nBacktest complete! Results saved to: {results_dir}")
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user.")
    except Exception as e:
        logger.error(f"Error running trading bot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
