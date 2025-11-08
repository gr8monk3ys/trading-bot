#!/usr/bin/env python3
"""
Live Trading Example

This script demonstrates how to run the trading bot in live mode with the Alpaca broker.
"""

import os
import sys
import logging
import asyncio
import signal
from datetime import datetime

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.sentiment_strategy import SentimentStrategy
from brokers.alpaca_broker import AlpacaBroker
from engine.strategy_manager import StrategyManager
from config import ALPACA_CREDS, SYMBOLS, TRADING_PARAMS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for clean shutdown
running = True
strategy_manager = None

def signal_handler(sig, frame):
    """Handle interrupt signals to gracefully shutdown the bot."""
    global running
    logger.info("Shutdown signal received. Closing positions and exiting...")
    running = False

async def run_trading_bot():
    """Run the trading bot with multiple strategies in live mode."""
    global strategy_manager
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create Alpaca broker
    broker = AlpacaBroker(
        api_key=ALPACA_CREDS['API_KEY'],
        api_secret=ALPACA_CREDS['API_SECRET'],
        paper=ALPACA_CREDS['PAPER']
    )
    
    # Create strategies
    momentum = MomentumStrategy(broker=broker)
    mean_reversion = MeanReversionStrategy(broker=broker)
    sentiment = SentimentStrategy(broker=broker)
    
    # Update with custom parameters
    momentum.set_parameters({
        'position_size': TRADING_PARAMS['POSITION_SIZE'],
        'stop_loss': TRADING_PARAMS['STOP_LOSS'],
        'take_profit': TRADING_PARAMS['TAKE_PROFIT']
    })
    
    mean_reversion.set_parameters({
        'position_size': TRADING_PARAMS['POSITION_SIZE'],
        'stop_loss': TRADING_PARAMS['STOP_LOSS'],
        'take_profit': TRADING_PARAMS['TAKE_PROFIT']
    })
    
    sentiment.set_parameters({
        'position_size': TRADING_PARAMS['POSITION_SIZE'],
        'sentiment_threshold': TRADING_PARAMS['SENTIMENT_THRESHOLD'],
        'stop_loss': TRADING_PARAMS['STOP_LOSS'],
        'take_profit': TRADING_PARAMS['TAKE_PROFIT']
    })
    
    # Create strategy manager with all strategies
    strategy_manager = StrategyManager(
        broker=broker,
        strategies=[momentum, mean_reversion, sentiment],
        symbols=SYMBOLS
    )
    
    # Initialize and run the strategy manager
    await strategy_manager.initialize()
    
    logger.info(f"Trading bot started with {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}")
    logger.info(f"Running in {'paper' if ALPACA_CREDS['PAPER'] else 'live'} trading mode")
    
    # Trading loop
    check_interval = TRADING_PARAMS['INTERVAL']
    while running:
        try:
            logger.info(f"Running strategy update at {datetime.now().strftime('%H:%M:%S')}")
            await strategy_manager.update()
            logger.info(f"Sleeping for {check_interval} seconds")
            await asyncio.sleep(check_interval)
        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)
            await asyncio.sleep(check_interval)
    
    # Clean shutdown
    logger.info("Closing all positions...")
    await strategy_manager.close_all_positions()
    logger.info("Trading bot shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_bot())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
