#!/usr/bin/env python3
"""
Trading Bot Dashboard Demo

This script demonstrates the visualization capabilities for strategy backtesting results.
"""

import os
import sys
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from brokers.alpaca_broker import AlpacaBroker
from engine.backtest_engine import BacktestEngine
from utils.visualization import create_performance_report
from config import ALPACA_CREDS, SYMBOLS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_backtest_dashboard():
    """Run backtests and create performance dashboard"""
    
    # Set test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 90-day backtest
    
    # Create Alpaca broker
    broker = AlpacaBroker(
        api_key=ALPACA_CREDS['API_KEY'],
        api_secret=ALPACA_CREDS['API_SECRET'],
        paper=ALPACA_CREDS['PAPER']
    )
    
    # Create strategies
    momentum = MomentumStrategy(broker=broker)
    mean_reversion = MeanReversionStrategy(broker=broker)
    
    # Use a subset of symbols for faster testing
    test_symbols = SYMBOLS[:3]  # First 3 symbols
    
    # Create results directory
    os.makedirs("data/results/dashboard_demo", exist_ok=True)
    
    # Run momentum strategy backtest
    logger.info(f"Running momentum strategy backtest from {start_date.date()} to {end_date.date()}...")
    momentum_backtest = BacktestEngine(
        strategy=momentum,
        start_date=start_date,
        end_date=end_date,
        symbols=test_symbols
    )
    momentum_results = await momentum_backtest.run()
    
    # Run mean reversion strategy backtest
    logger.info(f"Running mean reversion strategy backtest from {start_date.date()} to {end_date.date()}...")
    mean_reversion_backtest = BacktestEngine(
        strategy=mean_reversion,
        start_date=start_date,
        end_date=end_date,
        symbols=test_symbols
    )
    mean_reversion_results = await mean_reversion_backtest.run()
    
    # Create S&P 500 benchmark data (simulated for demo)
    logger.info("Creating benchmark data...")
    benchmark_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)  # For reproducibility
    
    # Generate random returns with a slight upward bias
    daily_returns = np.random.normal(0.0005, 0.01, len(benchmark_dates))
    benchmark_equity = 10000 * (1 + pd.Series(daily_returns)).cumprod()
    
    benchmark_data = pd.DataFrame({
        'equity': benchmark_equity,
        'returns': pd.Series(daily_returns)
    }, index=benchmark_dates)
    
    # Generate performance reports
    logger.info("Generating performance reports...")
    create_performance_report(
        momentum_results, 
        benchmark_data=benchmark_data,
        output_path="data/results/dashboard_demo/momentum"
    )
    
    create_performance_report(
        mean_reversion_results, 
        benchmark_data=benchmark_data,
        output_path="data/results/dashboard_demo/mean_reversion"
    )
    
    # Generate combined performance comparison
    logger.info("Generating strategy comparison...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot equity curves
    momentum_results['equity_curve']['equity'].plot(
        ax=ax, linewidth=2, label='Momentum Strategy'
    )
    mean_reversion_results['equity_curve']['equity'].plot(
        ax=ax, linewidth=2, label='Mean Reversion Strategy'
    )
    benchmark_data['equity'].plot(
        ax=ax, linewidth=2, linestyle='--', label='Benchmark (S&P 500)'
    )
    
    # Format plot
    ax.set_title('Strategy Comparison', fontsize=16)
    ax.set_ylabel('Equity ($)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    # Save comparison plot
    fig.savefig("data/results/dashboard_demo/strategy_comparison.png", dpi=150, bbox_inches='tight')
    
    logger.info("Dashboard generation complete!")
    logger.info("Results saved to data/results/dashboard_demo/")
    
    # Return paths to the generated reports
    return {
        "momentum": "data/results/dashboard_demo/momentum",
        "mean_reversion": "data/results/dashboard_demo/mean_reversion",
        "comparison": "data/results/dashboard_demo/strategy_comparison.png"
    }

if __name__ == "__main__":
    try:
        report_paths = asyncio.run(run_backtest_dashboard())
        
        # Print results and next steps
        print("\n===== Dashboard Demo Complete =====")
        print(f"Momentum strategy report: {report_paths['momentum']}")
        print(f"Mean Reversion strategy report: {report_paths['mean_reversion']}")
        print(f"Strategy comparison: {report_paths['comparison']}")
        print("\nView the generated reports to analyze strategy performance.")
    except KeyboardInterrupt:
        print("\nDashboard generation interrupted.")
    except Exception as e:
        logging.error(f"Error during dashboard generation: {e}", exc_info=True)
