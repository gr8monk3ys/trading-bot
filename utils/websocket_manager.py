"""
WebSocket Manager for Alpaca Real-Time Data Streaming.

This module provides a robust WebSocket manager for receiving real-time
market data (bars, quotes, trades) from Alpaca's data stream API.

Features:
- Auto-reconnection with exponential backoff
- Thread-safe subscription management
- Multiple handler registration per symbol
- Graceful shutdown handling
- Connection health monitoring

Usage:
    from utils.websocket_manager import WebSocketManager

    manager = WebSocketManager(api_key, secret_key, feed="iex")
    manager.subscribe_bars(["AAPL", "MSFT"], my_bar_handler)
    await manager.start()
    # ... later ...
    await manager.stop()
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar, Quote, Trade

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections to Alpaca's real-time data stream.

    Provides a clean interface for subscribing to real-time bars, quotes,
    and trades with automatic reconnection handling.

    Attributes:
        feed: Data feed type ("iex" for free, "sip" for paid)
        is_running: Whether the WebSocket is currently connected and running
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        feed: str = "iex",
        raw_data: bool = False
    ):
        """
        Initialize WebSocket manager.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            feed: Data feed ("iex" for free tier, "sip" for paid premium)
            raw_data: If True, receive raw data dicts instead of model objects
        """
        if not api_key or not secret_key:
            raise ValueError("API key and secret key are required")

        self._api_key = api_key
        self._secret_key = secret_key
        self._feed = feed
        self._raw_data = raw_data

        # Initialize the Alpaca stream client
        self._stream: Optional[StockDataStream] = None

        # Handler registrations (symbol -> set of callbacks)
        self._bar_handlers: Dict[str, Set[Callable]] = defaultdict(set)
        self._quote_handlers: Dict[str, Set[Callable]] = defaultdict(set)
        self._trade_handlers: Dict[str, Set[Callable]] = defaultdict(set)

        # Global handlers (called for all symbols)
        self._global_bar_handlers: Set[Callable] = set()
        self._global_quote_handlers: Set[Callable] = set()
        self._global_trade_handlers: Set[Callable] = set()

        # Connection state
        self._running = False
        self._connected = False
        self._stream_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Reconnection settings
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._base_reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

        # Subscribed symbols tracking
        self._subscribed_bars: Set[str] = set()
        self._subscribed_quotes: Set[str] = set()
        self._subscribed_trades: Set[str] = set()

        # Health monitoring
        self._last_message_time: Optional[datetime] = None
        self._message_count = 0

        logger.info(f"WebSocketManager initialized with feed={feed}")

    @property
    def is_running(self) -> bool:
        """Check if the WebSocket is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is connected."""
        return self._connected

    @property
    def message_count(self) -> int:
        """Get total number of messages received."""
        return self._message_count

    @property
    def last_message_time(self) -> Optional[datetime]:
        """Get timestamp of last received message."""
        return self._last_message_time

    def _create_stream(self) -> StockDataStream:
        """Create a new StockDataStream instance."""
        return StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed=self._feed,
            raw_data=self._raw_data
        )

    # =========================================================================
    # Generic Subscription/Unsubscription Helpers
    # =========================================================================

    def _subscribe(
        self,
        symbols: List[str],
        handler: Optional[Callable],
        handlers_dict: Dict[str, Set[Callable]],
        subscribed_set: Set[str],
        data_type: str,
    ) -> None:
        """
        Generic subscribe method for any data type.

        Args:
            symbols: List of stock symbols to subscribe to
            handler: Optional callback function for data updates
            handlers_dict: Symbol->handlers mapping for this data type
            subscribed_set: Set tracking subscribed symbols for this data type
            data_type: Human-readable type name for logging ("bar", "quote", "trade")
        """
        symbols = [s.upper() for s in symbols]

        if handler:
            for symbol in symbols:
                handlers_dict[symbol].add(handler)
                logger.debug(f"Registered {data_type} handler for {symbol}")

        subscribed_set.update(symbols)

        logger.info(f"Subscribed to {data_type}s for: {', '.join(symbols)}")

    def _unsubscribe(
        self,
        symbols: List[str],
        handlers_dict: Dict[str, Set[Callable]],
        subscribed_set: Set[str],
        data_type: str,
    ) -> None:
        """
        Generic unsubscribe method for any data type.

        Args:
            symbols: List of stock symbols to unsubscribe from
            handlers_dict: Symbol->handlers mapping for this data type
            subscribed_set: Set tracking subscribed symbols for this data type
            data_type: Human-readable type name for logging ("bar", "quote", "trade")
        """
        symbols = [s.upper() for s in symbols]
        for symbol in symbols:
            subscribed_set.discard(symbol)
            handlers_dict.pop(symbol, None)
        logger.info(f"Unsubscribed from {data_type}s for: {', '.join(symbols)}")

    # =========================================================================
    # Subscription Methods
    # =========================================================================

    def subscribe_bars(
        self,
        symbols: List[str],
        handler: Optional[Callable] = None
    ) -> None:
        """
        Subscribe to real-time bar data for specified symbols.

        Args:
            symbols: List of stock symbols to subscribe to
            handler: Optional callback function for bar updates.
                     Signature: async def handler(bar: Bar) -> None
                     If None, uses registered global handlers.

        Example:
            async def on_bar(bar):
                print(f"{bar.symbol}: {bar.close}")

            manager.subscribe_bars(["AAPL", "MSFT"], on_bar)
        """
        self._subscribe(symbols, handler, self._bar_handlers, self._subscribed_bars, "bar")

    def subscribe_quotes(
        self,
        symbols: List[str],
        handler: Optional[Callable] = None
    ) -> None:
        """
        Subscribe to real-time quote data for specified symbols.

        Args:
            symbols: List of stock symbols to subscribe to
            handler: Optional callback function for quote updates.
                     Signature: async def handler(quote: Quote) -> None

        Example:
            async def on_quote(quote):
                spread = quote.ask_price - quote.bid_price
                print(f"{quote.symbol} spread: ${spread:.2f}")

            manager.subscribe_quotes(["AAPL"], on_quote)
        """
        self._subscribe(symbols, handler, self._quote_handlers, self._subscribed_quotes, "quote")

    def subscribe_trades(
        self,
        symbols: List[str],
        handler: Optional[Callable] = None
    ) -> None:
        """
        Subscribe to real-time trade data for specified symbols.

        Args:
            symbols: List of stock symbols to subscribe to
            handler: Optional callback function for trade updates.
                     Signature: async def handler(trade: Trade) -> None

        Example:
            async def on_trade(trade):
                print(f"{trade.symbol} traded at ${trade.price}")

            manager.subscribe_trades(["AAPL"], on_trade)
        """
        self._subscribe(symbols, handler, self._trade_handlers, self._subscribed_trades, "trade")

    def add_global_bar_handler(self, handler: Callable) -> None:
        """
        Add a global handler that receives bars for all subscribed symbols.

        Args:
            handler: Callback function for all bar updates.
        """
        self._global_bar_handlers.add(handler)
        logger.debug("Added global bar handler")

    def add_global_quote_handler(self, handler: Callable) -> None:
        """
        Add a global handler that receives quotes for all subscribed symbols.

        Args:
            handler: Callback function for all quote updates.
        """
        self._global_quote_handlers.add(handler)
        logger.debug("Added global quote handler")

    def add_global_trade_handler(self, handler: Callable) -> None:
        """
        Add a global handler that receives trades for all subscribed symbols.

        Args:
            handler: Callback function for all trade updates.
        """
        self._global_trade_handlers.add(handler)
        logger.debug("Added global trade handler")

    def unsubscribe_bars(self, symbols: List[str]) -> None:
        """Remove bar subscriptions for specified symbols."""
        self._unsubscribe(symbols, self._bar_handlers, self._subscribed_bars, "bar")

    def unsubscribe_quotes(self, symbols: List[str]) -> None:
        """Remove quote subscriptions for specified symbols."""
        self._unsubscribe(symbols, self._quote_handlers, self._subscribed_quotes, "quote")

    def unsubscribe_trades(self, symbols: List[str]) -> None:
        """Remove trade subscriptions for specified symbols."""
        self._unsubscribe(symbols, self._trade_handlers, self._subscribed_trades, "trade")

    # =========================================================================
    # Internal Handlers
    # =========================================================================

    async def _dispatch_to_handlers(
        self,
        data,
        symbol_handlers: Dict[str, Set[Callable]],
        global_handlers: Set[Callable],
        data_type: str,
    ) -> None:
        """
        Generic handler dispatch for incoming stream data.

        Routes incoming data to both symbol-specific and global handlers,
        supporting both async and sync callback functions.

        Args:
            data: The incoming data object (Bar, Quote, or Trade).
                  Must have a `symbol` attribute.
            symbol_handlers: Dict mapping symbol -> set of handler callables
            global_handlers: Set of global handler callables for all symbols
            data_type: Human-readable type name for logging ("bar", "quote", "trade")
        """
        try:
            self._last_message_time = datetime.now()
            self._message_count += 1

            symbol = data.symbol

            # Call symbol-specific handlers
            if symbol in symbol_handlers:
                for handler in symbol_handlers[symbol]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data)
                        else:
                            handler(data)
                    except Exception as e:
                        logger.error(f"Error in {data_type} handler for {symbol}: {e}")

            # Call global handlers
            for handler in global_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in global {data_type} handler: {e}")

        except Exception as e:
            logger.error(f"Error handling {data_type} data: {e}")

    async def _handle_bar(self, bar: Bar) -> None:
        """
        Internal handler that routes bar data to registered callbacks.

        Args:
            bar: Bar data from Alpaca stream
        """
        await self._dispatch_to_handlers(
            bar, self._bar_handlers, self._global_bar_handlers, "bar"
        )

    async def _handle_quote(self, quote: Quote) -> None:
        """
        Internal handler that routes quote data to registered callbacks.

        Args:
            quote: Quote data from Alpaca stream
        """
        await self._dispatch_to_handlers(
            quote, self._quote_handlers, self._global_quote_handlers, "quote"
        )

    async def _handle_trade(self, trade: Trade) -> None:
        """
        Internal handler that routes trade data to registered callbacks.

        Args:
            trade: Trade data from Alpaca stream
        """
        await self._dispatch_to_handlers(
            trade, self._trade_handlers, self._global_trade_handlers, "trade"
        )

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def start(self) -> None:
        """
        Start the WebSocket connection.

        This method initiates the connection and begins receiving data.
        It handles reconnection automatically if the connection drops.

        Raises:
            RuntimeError: If already running
        """
        async with self._lock:
            if self._running:
                logger.warning("WebSocketManager is already running")
                return

            self._running = True

        logger.info("Starting WebSocket connection...")

        # Create the stream task
        self._stream_task = asyncio.create_task(self._run_stream())

        # Wait for initial connection
        for _ in range(50):  # Wait up to 5 seconds for connection
            await asyncio.sleep(0.1)
            if self._connected:
                logger.info("WebSocket connected successfully")
                return

        logger.warning("WebSocket connection still pending after 5 seconds")

    async def stop(self) -> None:
        """
        Stop the WebSocket connection gracefully.

        This method cleanly shuts down the connection and cancels
        any pending tasks.
        """
        async with self._lock:
            if not self._running:
                logger.warning("WebSocketManager is not running")
                return

            self._running = False
            self._connected = False

        logger.info("Stopping WebSocket connection...")

        # Cancel the stream task
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        # Close the stream
        if self._stream:
            try:
                self._stream.stop()
            except Exception as e:
                logger.warning(f"Error stopping stream: {e}")
            self._stream = None

        self._stream_task = None
        logger.info("WebSocket connection stopped")

    async def _run_stream(self) -> None:
        """
        Main stream runner with automatic reconnection.

        This is the core loop that maintains the WebSocket connection
        and handles reconnection with exponential backoff.
        """
        while self._running:
            try:
                # Create new stream instance
                self._stream = self._create_stream()

                # Register internal handlers
                if self._subscribed_bars:
                    self._stream.subscribe_bars(
                        self._handle_bar,
                        *self._subscribed_bars
                    )

                if self._subscribed_quotes:
                    self._stream.subscribe_quotes(
                        self._handle_quote,
                        *self._subscribed_quotes
                    )

                if self._subscribed_trades:
                    self._stream.subscribe_trades(
                        self._handle_trade,
                        *self._subscribed_trades
                    )

                logger.info(
                    f"Connecting to Alpaca stream (feed={self._feed}, "
                    f"bars={len(self._subscribed_bars)}, "
                    f"quotes={len(self._subscribed_quotes)}, "
                    f"trades={len(self._subscribed_trades)})"
                )

                # NOTE: _connected is set here before self._stream.run()
                # completes the actual WebSocket handshake. The Alpaca
                # StockDataStream API does not expose a post-connection
                # callback, so there is a brief window where _connected
                # is True but the underlying socket may not yet be fully
                # established. Consumers should rely on receiving data
                # messages (via _last_message_time) to confirm the stream
                # is truly active.
                self._connected = True
                self._reconnect_attempts = 0

                # Run the stream (blocking)
                await asyncio.to_thread(self._stream.run)

            except asyncio.CancelledError:
                logger.info("Stream task cancelled")
                break

            except Exception as e:
                self._connected = False
                self._reconnect_attempts += 1

                if self._reconnect_attempts > self._max_reconnect_attempts:
                    logger.error(
                        f"Max reconnection attempts ({self._max_reconnect_attempts}) "
                        f"exceeded. Stopping WebSocket manager."
                    )
                    self._running = False
                    break

                # Calculate backoff delay
                delay = min(
                    self._base_reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
                    self._max_reconnect_delay
                )

                logger.warning(
                    f"WebSocket error: {e}. "
                    f"Reconnecting in {delay:.1f}s "
                    f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})"
                )

                await asyncio.sleep(delay)

        self._connected = False
        logger.info("Stream runner exited")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_subscription_info(self) -> Dict:
        """
        Get current subscription information.

        Returns:
            Dict with subscription details
        """
        return {
            "bars": list(self._subscribed_bars),
            "quotes": list(self._subscribed_quotes),
            "trades": list(self._subscribed_trades),
            "bar_handlers": {s: len(h) for s, h in self._bar_handlers.items()},
            "quote_handlers": {s: len(h) for s, h in self._quote_handlers.items()},
            "trade_handlers": {s: len(h) for s, h in self._trade_handlers.items()},
            "global_bar_handlers": len(self._global_bar_handlers),
            "global_quote_handlers": len(self._global_quote_handlers),
            "global_trade_handlers": len(self._global_trade_handlers),
        }

    def get_connection_stats(self) -> Dict:
        """
        Get connection statistics.

        Returns:
            Dict with connection stats
        """
        return {
            "is_running": self._running,
            "is_connected": self._connected,
            "reconnect_attempts": self._reconnect_attempts,
            "message_count": self._message_count,
            "last_message_time": self._last_message_time,
            "feed": self._feed,
        }

    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for the WebSocket to connect.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected within timeout, False otherwise
        """
        start = datetime.now()
        while (datetime.now() - start).total_seconds() < timeout:
            if self._connected:
                return True
            await asyncio.sleep(0.1)
        return False


# Convenience function for creating a manager from config
def create_websocket_manager_from_config() -> WebSocketManager:
    """
    Create a WebSocketManager using credentials from config.

    Returns:
        Configured WebSocketManager instance

    Raises:
        ValueError: If credentials are not configured
    """
    from config import ALPACA_CREDS

    api_key = ALPACA_CREDS.get("API_KEY")
    api_secret = ALPACA_CREDS.get("API_SECRET")

    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca API credentials not found in config. "
            "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
        )

    # Use IEX feed for free tier (most common)
    # Change to "sip" if you have a paid subscription
    return WebSocketManager(
        api_key=api_key,
        secret_key=api_secret,
        feed="iex"
    )
