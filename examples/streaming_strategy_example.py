#!/usr/bin/env python3
"""
Example: Using WebSocket Streaming with Trading Strategies.

This example demonstrates how to integrate real-time WebSocket streaming
with the trading bot's strategy framework. It shows a pattern where
strategies can receive real-time data instead of polling.

Key concepts:
    - Strategies receive real-time bar data via callbacks
    - Price history is updated in real-time
    - Signals can be generated immediately when new data arrives
    - Polling is kept as a fallback when streaming is unavailable

Usage:
    python examples/streaming_strategy_example.py --symbols AAPL,MSFT --duration 300
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca.data.models import Bar

from brokers.alpaca_broker import AlpacaBroker
from config import SYMBOLS
from strategies.base_strategy import BaseStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingMomentumStrategy(BaseStrategy):
    """
    A simple momentum strategy that uses real-time WebSocket streaming.

    This demonstrates the pattern for receiving real-time data in strategies.
    Instead of polling for prices, the strategy receives bar data as it arrives.

    Features:
        - Real-time price history updates
        - Immediate signal generation on new bars
        - Simple momentum calculation (price vs moving average)
        - Integration with broker's streaming infrastructure
    """

    NAME = "StreamingMomentum"

    def __init__(self, broker=None, parameters=None):
        """Initialize the streaming momentum strategy."""
        parameters = parameters or {}
        parameters.setdefault("sma_period", 20)
        parameters.setdefault("momentum_threshold", 0.02)  # 2% above SMA to trigger

        super().__init__(name=self.NAME, broker=broker, parameters=parameters)

        # Strategy-specific state
        self.sma_period = parameters.get("sma_period", 20)
        self.momentum_threshold = parameters.get("momentum_threshold", 0.02)

        # Real-time price tracking (updated by streaming)
        self.realtime_prices: Dict[str, List[float]] = {}
        self.last_signals: Dict[str, str] = {}

        # Statistics
        self.bars_received = 0
        self.signals_generated = 0

    async def on_bar(self, bar: Bar) -> None:
        """
        Handler for real-time bar data from WebSocket.

        This is called automatically when new bar data arrives via streaming.
        The strategy should update its state and potentially generate signals.

        Args:
            bar: Real-time bar data from Alpaca stream
        """
        symbol = bar.symbol
        close_price = float(bar.close)

        self.bars_received += 1

        # Update price history
        if symbol not in self.realtime_prices:
            self.realtime_prices[symbol] = []

        self.realtime_prices[symbol].append(close_price)

        # Keep only recent prices (for SMA calculation)
        max_history = self.sma_period * 2
        if len(self.realtime_prices[symbol]) > max_history:
            self.realtime_prices[symbol] = self.realtime_prices[symbol][-max_history:]

        # Also update base strategy's price_history for compatibility
        self.price_history[symbol] = self.realtime_prices[symbol]

        # Generate signal if we have enough data
        if len(self.realtime_prices[symbol]) >= self.sma_period:
            signal = await self._analyze_realtime(symbol, close_price)

            if signal and signal != self.last_signals.get(symbol):
                self.last_signals[symbol] = signal
                self.signals_generated += 1

                self.logger.info(
                    f"SIGNAL | {symbol:6s} | {signal.upper():4s} | "
                    f"Price: ${close_price:.2f} | "
                    f"SMA({self.sma_period}): ${self._calculate_sma(symbol):.2f}"
                )

                # In a real strategy, you would execute trades here
                # await self.execute_trade(symbol, {"action": signal})

    async def _analyze_realtime(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Analyze real-time data and generate a signal.

        Args:
            symbol: Stock symbol
            current_price: Current price from latest bar

        Returns:
            'buy', 'sell', or None
        """
        sma = self._calculate_sma(symbol)
        if sma is None:
            return None

        momentum = (current_price - sma) / sma

        # Simple momentum signal
        if momentum > self.momentum_threshold:
            return "buy"  # Price is above SMA by threshold - bullish momentum
        elif momentum < -self.momentum_threshold:
            return "sell"  # Price is below SMA by threshold - bearish momentum

        return None  # No signal (in neutral zone)

    def _calculate_sma(self, symbol: str) -> Optional[float]:
        """Calculate simple moving average."""
        prices = self.realtime_prices.get(symbol, [])
        if len(prices) < self.sma_period:
            return None
        return sum(prices[-self.sma_period:]) / self.sma_period

    async def analyze_symbol(self, symbol: str) -> dict:
        """
        Analyze a symbol (polling fallback).

        This is the traditional polling-based analysis. When using streaming,
        signals are generated in on_bar() instead, but this provides a fallback.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Signal dict with action, confidence, etc.
        """
        # Get current price via API (fallback when streaming not available)
        current_price = await self.broker.get_last_price(symbol)
        if current_price is None:
            return {"action": "hold", "reason": "no_price_data"}

        # Use same analysis logic
        if symbol in self.realtime_prices and len(self.realtime_prices[symbol]) >= self.sma_period:
            signal = await self._analyze_realtime(symbol, current_price)
            return {
                "action": signal or "hold",
                "price": current_price,
                "sma": self._calculate_sma(symbol),
                "source": "realtime"
            }

        return {"action": "hold", "reason": "insufficient_data", "source": "polling"}

    async def execute_trade(self, symbol: str, signal: dict) -> None:
        """
        Execute a trade based on signal.

        In this demo, we just log the trade. A real strategy would
        submit orders via the broker.

        Args:
            symbol: Stock symbol
            signal: Signal dict with action
        """
        action = signal.get("action")
        if action in ("buy", "sell"):
            self.logger.info(f"Would execute {action.upper()} for {symbol}")
            # Real implementation:
            # order = OrderBuilder(symbol, action, qty).market().build()
            # await self.broker.submit_order_advanced(order)

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        return {
            "bars_received": self.bars_received,
            "signals_generated": self.signals_generated,
            "symbols_tracking": len(self.realtime_prices),
            "last_signals": dict(self.last_signals)
        }


async def run_streaming_strategy(
    symbols: List[str],
    duration: int = None,
    subscribe_quotes: bool = False
) -> None:
    """
    Run a strategy with real-time WebSocket streaming.

    Args:
        symbols: List of symbols to trade
        duration: Optional duration in seconds
        subscribe_quotes: Whether to also subscribe to quotes
    """
    logger.info("=" * 60)
    logger.info("Streaming Strategy Example")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Duration: {duration}s" if duration else "Duration: Until Ctrl+C")
    logger.info("=" * 60)

    # Initialize broker
    broker = AlpacaBroker(paper=True)

    # Initialize strategy
    strategy = StreamingMomentumStrategy(
        broker=broker,
        parameters={
            "symbols": symbols,
            "sma_period": 20,
            "momentum_threshold": 0.015,  # 1.5% momentum threshold
        }
    )

    await strategy.initialize()

    # Setup WebSocket streaming
    broker.setup_websocket(feed="iex")

    # Register the strategy's on_bar handler
    # This is the key integration - strategy receives data via callback
    broker.register_bar_handler(strategy.on_bar, symbols)

    if subscribe_quotes:
        # Could also register quote handler if strategy uses quotes
        pass

    shutdown_event = asyncio.Event()

    # Handle shutdown
    def signal_handler():
        logger.info("\nShutdown requested...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Start streaming
        connected = await broker.start_streaming(
            symbols=symbols,
            subscribe_bars=True,
            subscribe_quotes=subscribe_quotes
        )

        if not connected:
            logger.error("Failed to connect to WebSocket")
            return

        logger.info("Streaming connected! Strategy is now receiving real-time data.")
        logger.info("(Bars flow during market hours; signals generated on momentum)")

        # Run until duration expires or shutdown requested
        start_time = datetime.now()

        while not shutdown_event.is_set():
            await asyncio.sleep(1)

            # Check duration
            if duration:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration:
                    logger.info(f"Duration of {duration}s reached")
                    break

            # Print stats every 60 seconds
            elapsed = (datetime.now() - start_time).total_seconds()
            if int(elapsed) % 60 == 0 and int(elapsed) > 0:
                stats = strategy.get_stats()
                logger.info(
                    f"Stats: {stats['bars_received']} bars | "
                    f"{stats['signals_generated']} signals | "
                    f"Tracking {stats['symbols_tracking']} symbols"
                )

    finally:
        # Cleanup
        await broker.stop_streaming()

        # Print final stats
        stats = strategy.get_stats()
        logger.info("")
        logger.info("=" * 40)
        logger.info("Final Strategy Statistics")
        logger.info("=" * 40)
        logger.info(f"Bars received: {stats['bars_received']}")
        logger.info(f"Signals generated: {stats['signals_generated']}")
        logger.info(f"Symbols tracked: {stats['symbols_tracking']}")

        if stats['last_signals']:
            logger.info("\nLast signals by symbol:")
            for sym, sig in sorted(stats['last_signals'].items()):
                logger.info(f"  {sym}: {sig.upper()}")

        # Show price history summary
        for sym, prices in strategy.realtime_prices.items():
            if prices:
                logger.info(f"\n{sym} price range: ${min(prices):.2f} - ${max(prices):.2f}")

        logger.info("=" * 40)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run a trading strategy with real-time WebSocket streaming"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (default: from config)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds (default: until Ctrl+C)"
    )
    parser.add_argument(
        "--quotes",
        action="store_true",
        help="Also subscribe to quote data"
    )

    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = SYMBOLS[:3]  # Use first 3 default symbols

    await run_streaming_strategy(
        symbols=symbols,
        duration=args.duration,
        subscribe_quotes=args.quotes
    )


if __name__ == "__main__":
    asyncio.run(main())
