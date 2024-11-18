import asyncio
import signal
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Optional, Set, List
from config import ALPACA_CREDS, TRADING_PARAMS, SYMBOLS, RISK_PARAMS, TECHNICAL_PARAMS
from brokers.alpaca_broker import AlpacaBroker
from strategies.sentiment_stock_strategy import SentimentStockStrategy

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

if not ALPACA_CREDS["API_KEY"] or not ALPACA_CREDS["API_SECRET"]:
    raise ValueError(
        "Alpaca API credentials not found. Please ensure API_KEY and API_SECRET "
        "are set in your .env file."
    )

class TradingBot:
    def __init__(self):
        """Initialize the trading bot."""
        logger.info("Initializing TradingBot...")
        self.broker = None
        self.strategies = []
        self._shutdown_event = asyncio.Event()
        self._loop = asyncio.get_event_loop()
        self._tasks = set()
        self.api_key = ALPACA_CREDS["API_KEY"]
        self.api_secret = ALPACA_CREDS["API_SECRET"]
        self.paper = ALPACA_CREDS["PAPER"]
        self._ws_task = None
        logger.debug(f"Trading parameters loaded: {TRADING_PARAMS}")
        
    def _signal_handler(self, sig):
        """Handle system signals."""
        logger.info(f"Received signal {sig}")
        self._shutdown_event.set()

    def _configure_signal_handlers(self):
        """Configure signal handlers for graceful shutdown."""
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                self._loop.add_signal_handler(
                    sig, 
                    lambda s=sig: asyncio.create_task(self._handle_shutdown(s))
                )
            logger.debug("Signal handlers configured")
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning("Signal handlers not supported on this platform")
        except Exception as e:
            logger.error(f"Error configuring signal handlers: {e}", exc_info=True)

    async def _handle_shutdown(self, sig):
        """Handle shutdown signal."""
        logger.info(f"Handling shutdown signal {sig}")
        self._shutdown_event.set()
        await self.shutdown()

    async def _initialize_broker(self):
        """Initialize the broker and establish connections."""
        try:
            logger.info(f"Initializing broker with paper={self.paper} (type: {type(self.paper)})")
            
            # Initialize broker
            self.broker = AlpacaBroker(paper=self.paper)
            
            # Connect to websocket
            logger.info("Connecting to websocket...")
            self._ws_task = await self.broker.connect()
            if not self._ws_task:
                raise RuntimeError("Failed to establish websocket connection")
            
            logger.info("Broker initialization complete")
            return self._ws_task
            
        except Exception as e:
            logger.error(f"Failed to initialize broker: {e}", exc_info=True)
            raise

    async def _initialize_strategies(self):
        """Initialize trading strategies."""
        try:
            logger.info("Initializing trading strategies...")
            
            # Load strategy parameters
            parameters = {
                'interval': TRADING_PARAMS.get('INTERVAL', 60),
                'sentiment_threshold': TRADING_PARAMS.get('SENTIMENT_THRESHOLD', 0.6),
                'position_size': TRADING_PARAMS.get('POSITION_SIZE', 0.1),
                'max_position_size': TRADING_PARAMS.get('MAX_POSITION_SIZE', 0.25),
                'stop_loss': TRADING_PARAMS.get('STOP_LOSS', 0.02),
                'take_profit': TRADING_PARAMS.get('TAKE_PROFIT', 0.05),
                'symbols': SYMBOLS,
                # Risk parameters
                'max_portfolio_risk': RISK_PARAMS.get('MAX_PORTFOLIO_RISK', 0.02),
                'max_position_risk': RISK_PARAMS.get('MAX_POSITION_RISK', 0.01),
                'max_correlation': RISK_PARAMS.get('MAX_CORRELATION', 0.7),
                # Technical parameters
                'sentiment_window': TECHNICAL_PARAMS.get('SENTIMENT_WINDOW', 5),
                'price_history_window': TECHNICAL_PARAMS.get('PRICE_HISTORY_WINDOW', 30)
            }
            
            logger.debug(f"Strategy parameters: {parameters}")
            logger.debug(f"Trading symbols: {SYMBOLS}")
            
            # Initialize the sentiment strategy
            strategy = SentimentStockStrategy(
                name="SentimentStockStrategy",
                broker=self.broker,
                parameters=parameters
            )
            
            # Initialize the strategy
            logger.debug("Initializing sentiment strategy...")
            success = await strategy.initialize()
            if not success:
                raise RuntimeError("Failed to initialize strategy")
            
            # Start the strategy
            logger.debug("Starting strategy execution...")
            self._tasks.add(asyncio.create_task(strategy.run()))
            
            self.strategies.append(strategy)
            logger.info("Successfully initialized and started trading strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}", exc_info=True)
            raise

    async def _monitor_positions(self):
        """Monitor open positions and update stop losses."""
        logger.debug("Starting position monitoring...")
        try:
            while not self._shutdown_event.is_set():
                logger.debug("Checking positions...")
                for strategy in self.strategies:
                    positions = await strategy.get_positions()
                    if positions:
                        logger.info(f"Current positions: {positions}")
                    else:
                        logger.debug("No open positions")
                await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in position monitoring: {e}", exc_info=True)
        finally:
            logger.info("Position monitoring stopped")

    async def _process_market_data(self):
        """Process incoming market data."""
        logger.debug("Starting market data processing...")
        try:
            while not self._shutdown_event.is_set():
                for strategy in self.strategies:
                    for symbol in strategy.symbols:
                        logger.debug(f"Processing market data for {symbol}...")
                        signal = await strategy.analyze_symbol(symbol)
                        if signal:
                            logger.info(f"Trading signal for {symbol}: {signal}")
                            await strategy.execute_trade(symbol, signal)
                        else:
                            logger.debug(f"No trading signal for {symbol}")
                await asyncio.sleep(60)  # Process every minute
        except Exception as e:
            logger.error(f"Error in market data processing: {e}", exc_info=True)
        finally:
            logger.info("Market data processing stopped")

    async def start(self):
        """Start the trading bot."""
        logger.info("Starting trading bot...")
        try:
            # Configure signal handlers
            self._configure_signal_handlers()
            logger.info("Signal handlers configured successfully")
            
            # Initialize broker and connect
            self._ws_task = await self._initialize_broker()
            if not self._ws_task:
                raise RuntimeError("Failed to initialize broker")
            
            # Initialize strategies
            logger.info("Initializing strategies...")
            await self._initialize_strategies()
            
            # Create tasks for monitoring and processing
            logger.info("Creating monitoring tasks...")
            position_task = asyncio.create_task(self._monitor_positions())
            market_task = asyncio.create_task(self._process_market_data())
            self._tasks.update({position_task, market_task, self._ws_task})
            
            # Wait for shutdown signal
            logger.info("Bot is running. Press Ctrl+C to stop.")
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error in trading bot: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the trading bot gracefully."""
        if hasattr(self, '_shutdown_in_progress'):
            return
        self._shutdown_in_progress = True
        
        logger.info("Initiating trading bot shutdown...")
        try:
            # Cancel all running tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            if self._tasks:
                # Wait for all tasks to complete
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Close broker connection
            if self.broker:
                await self.broker.disconnect()
            
            logger.info("Trading bot shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            self._tasks.clear()

async def main():
    """Main entry point for the trading bot."""
    bot = TradingBot()
    try:
        logger.info("Starting main trading bot process...")
        await bot.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
