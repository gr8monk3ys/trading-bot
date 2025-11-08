#!/usr/bin/env python3
"""
Strategy Demo Example

This script demonstrates how to use the trading bot with a simple momentum strategy.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.momentum_strategy import MomentumStrategy
from brokers.alpaca_broker import AlpacaBroker
from engine.backtest_engine import BacktestEngine
from engine.performance_metrics import PerformanceMetrics
from config import ALPACA_CREDS, SYMBOLS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest():
    """Run a simple backtest demonstration with the momentum strategy."""
    
    # Create Alpaca broker
    broker = AlpacaBroker(
        api_key=ALPACA_CREDS['API_KEY'],
        api_secret=ALPACA_CREDS['API_SECRET'],
        paper=ALPACA_CREDS['PAPER']
    )
    
    # Create momentum strategy with custom parameters
    strategy = MomentumStrategy(broker=broker)
    strategy.set_parameters({
        'position_size': 0.1,
        'max_positions': 3,
        'stop_loss': 0.02,
        'take_profit': 0.05,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30
    })
    
    # Configure backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 30-day backtest
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        symbols=SYMBOLS[:3]  # Use first 3 symbols for demo
    )
    
    # Run backtest
    results = backtest_engine.run()
    
    # Calculate performance metrics
    performance = PerformanceMetrics(results)
    metrics = performance.calculate_metrics()
    
    # Print results
    logger.info("======= Backtest Results =======")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Generate plot if available
    performance.plot_equity_curve()

if __name__ == "__main__":
    run_backtest()
