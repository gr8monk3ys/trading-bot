#!/usr/bin/env python3
"""
Real-Time Streaming Example for Alpaca Trading Bot.

This example demonstrates how to use the WebSocket streaming functionality
to receive real-time market data (bars, quotes, trades) from Alpaca.

Usage:
    # Basic usage with default symbols
    python examples/run_streaming.py

    # Custom symbols
    python examples/run_streaming.py --symbols AAPL,MSFT,GOOGL

    # Subscribe to all data types
    python examples/run_streaming.py --symbols AAPL --quotes --trades

    # Run for specific duration
    python examples/run_streaming.py --duration 60  # 60 seconds

Features demonstrated:
    - Subscribing to real-time bar data
    - Subscribing to real-time quotes (bid/ask)
    - Subscribing to real-time trade data
    - Integrating with trading strategies
    - Graceful shutdown handling
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from brokers.alpaca_broker import AlpacaBroker
from config import SYMBOLS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingDemo:
    """
    Demonstrates real-time WebSocket streaming with Alpaca.

    This class shows how to:
    1. Set up WebSocket connections
    2. Register handlers for different data types
    3. Process real-time market data
    4. Integrate with trading strategies
    """

    def __init__(self, symbols: list, subscribe_quotes: bool = False, subscribe_trades: bool = False):
        """
        Initialize the streaming demo.

        Args:
            symbols: List of stock symbols to stream
            subscribe_quotes: Whether to subscribe to quote data
            subscribe_trades: Whether to subscribe to trade data
        """
        self.symbols = symbols
        self.subscribe_quotes = subscribe_quotes
        self.subscribe_trades = subscribe_trades

        # Initialize broker
        self.broker = AlpacaBroker(paper=True)

        # Statistics tracking
        self.bar_count = 0
        self.quote_count = 0
        self.trade_count = 0
        self.start_time = None

        # Price tracking for simple analysis
        self.latest_prices = {}

        # Shutdown flag
        self._shutdown = False

    async def on_bar(self, bar) -> None:
        """
        Handler for real-time bar data.

        This is called whenever a new bar (candlestick) is received.
        Bars are typically 1-minute intervals during market hours.

        Args:
            bar: Bar object with open, high, low, close, volume, timestamp
        """
        self.bar_count += 1

        symbol = bar.symbol
        self.latest_prices[symbol] = bar.close

        logger.info(
            f"BAR | {symbol:6s} | "
            f"O:{bar.open:8.2f} H:{bar.high:8.2f} L:{bar.low:8.2f} C:{bar.close:8.2f} | "
            f"Vol:{bar.volume:10,d} | "
            f"Time: {bar.timestamp}"
        )

        # Example: Simple price change detection
        # In a real strategy, you would update indicators and check for signals here

    async def on_quote(self, quote) -> None:
        """
        Handler for real-time quote data.

        This is called on every bid/ask update - can be very frequent!
        Useful for tracking spreads and order book dynamics.

        Args:
            quote: Quote object with bid_price, ask_price, bid_size, ask_size, timestamp
        """
        self.quote_count += 1

        symbol = quote.symbol
        spread = quote.ask_price - quote.bid_price
        spread_pct = (spread / quote.bid_price) * 100 if quote.bid_price > 0 else 0

        # Only log every 10th quote to reduce noise
        if self.quote_count % 10 == 0:
            logger.info(
                f"QUOTE | {symbol:6s} | "
                f"Bid:{quote.bid_price:8.2f} ({quote.bid_size:5d}) | "
                f"Ask:{quote.ask_price:8.2f} ({quote.ask_size:5d}) | "
                f"Spread: ${spread:.4f} ({spread_pct:.3f}%)"
            )

    async def on_trade(self, trade) -> None:
        """
        Handler for real-time trade data.

        This is called on every individual trade - can be very high frequency!
        Useful for tracking volume and price momentum in real-time.

        Args:
            trade: Trade object with price, size, timestamp, exchange, conditions
        """
        self.trade_count += 1

        symbol = trade.symbol
        self.latest_prices[symbol] = trade.price

        # Only log significant trades (larger size)
        if trade.size >= 100 or self.trade_count % 50 == 0:
            logger.info(
                f"TRADE | {symbol:6s} | "
                f"Price:{trade.price:8.2f} | "
                f"Size:{trade.size:7d} | "
                f"Exchange: {trade.exchange}"
            )

    async def run(self, duration: int = None) -> None:
        """
        Run the streaming demo.

        Args:
            duration: Optional duration in seconds. If None, runs until interrupted.
        """
        self.start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("Starting Real-Time Streaming Demo")
        logger.info("=" * 60)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info("Subscribe to bars: True")
        logger.info(f"Subscribe to quotes: {self.subscribe_quotes}")
        logger.info(f"Subscribe to trades: {self.subscribe_trades}")
        logger.info(f"Duration: {duration} seconds" if duration else "Duration: Until interrupted (Ctrl+C)")
        logger.info("=" * 60)

        try:
            # Setup WebSocket with IEX feed (free tier)
            self.broker.setup_websocket(feed="iex")

            # Register handlers
            self.broker.register_bar_handler(self.on_bar, self.symbols)

            if self.subscribe_quotes:
                self.broker.register_quote_handler(self.on_quote, self.symbols)

            if self.subscribe_trades:
                self.broker.register_trade_handler(self.on_trade, self.symbols)

            # Start streaming
            connected = await self.broker.start_streaming(
                symbols=self.symbols,
                subscribe_bars=True,
                subscribe_quotes=self.subscribe_quotes,
                subscribe_trades=self.subscribe_trades
            )

            if not connected:
                logger.error("Failed to connect to WebSocket stream")
                return

            logger.info("WebSocket connected! Waiting for market data...")
            logger.info("(Note: Data only flows during market hours)")

            # Run for specified duration or until interrupted
            if duration:
                await asyncio.sleep(duration)
            else:
                # Run until shutdown signal
                while not self._shutdown:
                    await asyncio.sleep(1)

                    # Print stats every 30 seconds
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                        self._print_stats()

        except asyncio.CancelledError:
            logger.info("Streaming cancelled")

        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown the streaming connection."""
        self._shutdown = True

        logger.info("\nShutting down...")

        # Stop streaming
        await self.broker.stop_streaming()

        # Print final statistics
        self._print_stats()

    def _print_stats(self) -> None:
        """Print streaming statistics."""
        if not self.start_time:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds()

        logger.info("")
        logger.info("=" * 40)
        logger.info("Streaming Statistics")
        logger.info("=" * 40)
        logger.info(f"Runtime: {elapsed:.1f} seconds")
        logger.info(f"Bars received: {self.bar_count}")
        logger.info(f"Quotes received: {self.quote_count}")
        logger.info(f"Trades received: {self.trade_count}")

        if self.bar_count > 0:
            logger.info(f"Bar rate: {self.bar_count / elapsed:.2f} bars/sec")

        if self.quote_count > 0:
            logger.info(f"Quote rate: {self.quote_count / elapsed:.2f} quotes/sec")

        if self.trade_count > 0:
            logger.info(f"Trade rate: {self.trade_count / elapsed:.2f} trades/sec")

        if self.latest_prices:
            logger.info("\nLatest Prices:")
            for symbol, price in sorted(self.latest_prices.items()):
                logger.info(f"  {symbol}: ${price:.2f}")

        logger.info("=" * 40)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time streaming demo for Alpaca Trading Bot"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: from config)"
    )
    parser.add_argument(
        "--quotes",
        action="store_true",
        help="Subscribe to real-time quote data"
    )
    parser.add_argument(
        "--trades",
        action="store_true",
        help="Subscribe to real-time trade data"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration to run in seconds (default: until Ctrl+C)"
    )

    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = SYMBOLS[:5]  # Use first 5 default symbols

    # Create and run demo
    demo = StreamingDemo(
        symbols=symbols,
        subscribe_quotes=args.quotes,
        subscribe_trades=args.trades
    )

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("\nReceived interrupt signal...")
        demo._shutdown = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await demo.run(duration=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
