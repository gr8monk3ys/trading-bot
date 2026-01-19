import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
)

# NOTE: Removed lumibot imports - they crash at import time due to
# lumibot.credentials.py trying to instantiate Alpaca broker before config is ready
# We don't actually need lumibot's Broker class - we built our own implementation
from config import ALPACA_CREDS, SYMBOLS

# P2 FIX: Environment-aware logging - only show full tracebacks in debug mode
# This prevents sensitive information from leaking in production logs
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")

# Set up logging
logger = logging.getLogger(__name__)


# P2 FIX: Custom exception for broker errors
class BrokerError(Exception):
    """
    Exception raised for broker operation failures.

    Use this for critical errors where the caller MUST handle the failure
    (e.g., order submission failures, authentication errors).
    """

    pass


class BrokerConnectionError(BrokerError):
    """Raised when broker connection fails."""

    pass


class OrderError(BrokerError):
    """Raised when order operations fail."""

    pass


# ERROR HANDLING CONVENTIONS:
# This module uses the following patterns for error handling:
#
# 1. QUERY methods (get_position, get_last_price, get_bars):
#    - Return None or [] on error
#    - Log the error
#    - Caller should check for None/empty before using
#
# 2. ACTION methods (submit_order, cancel_order):
#    - Raise OrderError on critical failures
#    - Return False for non-critical failures (already cancelled, etc.)
#    - Log the error
#
# 3. CONNECTION methods (get_account, get_positions):
#    - Raise BrokerConnectionError if broker is unreachable
#    - These are critical - trading cannot proceed without them


def retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10, jitter=0.1):
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        jitter: Jitter factor (0.0-1.0) to randomize delay and prevent thundering herd

    The jitter adds randomness to prevent many clients from retrying simultaneously.
    For example, with jitter=0.1 and base delay of 2s, actual delay will be 1.8s-2.2s.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (ConnectionError, TimeoutError, OSError) as e:
                    # Network-related errors are retryable
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed (network error): {e}"
                    )
                except Exception as e:
                    # For other exceptions, check if they seem transient
                    error_str = str(e).lower()
                    is_transient = any(
                        term in error_str
                        for term in [
                            "timeout",
                            "connection",
                            "rate limit",
                            "429",
                            "503",
                            "502",
                            "504",
                        ]
                    )

                    if is_transient:
                        last_exception = e
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed (transient): {e}"
                        )
                    else:
                        # Non-transient error, don't retry
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                if attempt < max_retries - 1:
                    # Calculate base delay with exponential backoff
                    base_delay = min(initial_delay * (2**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    # Jitter range: [base * (1 - jitter), base * (1 + jitter)]
                    jitter_range = base_delay * jitter
                    sleep_time = base_delay + random.uniform(-jitter_range, jitter_range)
                    sleep_time = max(0.1, sleep_time)  # Ensure minimum delay

                    logger.info(f"Retrying {func.__name__} in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)

            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception

        return wrapper

    return decorator


class AlpacaBroker:
    """
    Alpaca broker implementation.

    Direct implementation without lumibot dependency to avoid import-time crashes.
    Provides all necessary broker functionality for live and paper trading.
    """

    NAME = "alpaca"
    IS_BACKTESTING_BROKER = False

    def __init__(self, paper=True):
        """Initialize the AlpacaBroker."""
        try:
            # Ensure paper is a boolean
            if isinstance(paper, str):
                paper = paper.lower() == "true"
            self.paper = bool(paper)
            logger.info(
                f"AlpacaBroker initialized with paper={self.paper} (type: {type(self.paper)})"
            )

            # Initialize position tracking
            self._filled_positions = []
            self._subscribers = set()
            self._ws_lock = asyncio.Lock()
            self._ws_task = None
            self._connected = False
            self._reconnect_attempts = 0
            self._reconnect_delay = 1  # Initial reconnect delay in seconds
            self._max_reconnect_delay = 60  # Max reconnect delay in seconds

            # P0 FIX: Use local variables for credentials instead of storing as attributes
            # This prevents accidental exposure through logging, serialization, or debugging
            _api_key = ALPACA_CREDS["API_KEY"]
            _api_secret = ALPACA_CREDS["API_SECRET"]

            if not _api_key or not _api_secret:
                raise ValueError(
                    "Alpaca API credentials not found. Please set them in your environment variables."
                )

            # Initialize the trading client
            self.trading_client = TradingClient(
                api_key=_api_key,
                secret_key=_api_secret,
                paper=self.paper,
                url_override="https://paper-api.alpaca.markets" if self.paper else None,
            )

            # Initialize the data client
            self.data_client = StockHistoricalDataClient(api_key=_api_key, secret_key=_api_secret)

            # Initialize the stream for WebSockets
            self.stream = StockDataStream(
                api_key=_api_key,
                secret_key=_api_secret,
                url_override="https://paper-api.alpaca.markets/stream" if self.paper else None,
            )

            self._subscribed_symbols = set()  # Keep track of subscribed symbols

            # P0 FIX: Removed unused config dict that stored credentials in memory

        except Exception as e:
            logger.error(f"Error initializing AlpacaBroker: {e}", exc_info=DEBUG_MODE)
            raise

    async def is_connected(self) -> bool:
        """Thread-safe check if websocket is connected."""
        async with self._ws_lock:
            return self._connected

    @staticmethod
    def _validate_symbol(symbol: str) -> str:
        """
        P2 FIX: Validate and sanitize stock symbol.

        Args:
            symbol: Stock symbol to validate

        Returns:
            Sanitized uppercase symbol

        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if not isinstance(symbol, str):
            raise ValueError(f"Symbol must be a string, got {type(symbol)}")

        symbol = symbol.upper().strip()

        # Valid stock symbols: 1-5 uppercase letters (some ETFs have numbers)
        if not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid symbol format: {symbol}")
        if len(symbol) > 10:  # Allow for options symbols which are longer
            raise ValueError(f"Symbol too long: {symbol}")

        return symbol

    def _add_subscriber(self, subscriber):
        """Add a subscriber for market data updates."""
        if subscriber not in self._subscribers:
            self._subscribers.add(subscriber)
            logger.debug(f"Added subscriber: {subscriber}")

    def _remove_subscriber(self, subscriber):
        """Remove a subscriber from market data updates."""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            logger.debug(f"Removed subscriber: {subscriber}")

    async def _handle_trade_updates(self, data):
        """Handle trade update events from websocket."""
        try:
            # Process trade update from websocket
            logger.debug(f"Trade update received: {data}")

            # Extract trade event details
            event_type = data.get("event")
            order = data.get("order", {})
            order_id = order.get("id")

            # Handle different trade events
            if event_type == "fill":
                logger.info(f"Order {order_id} filled")
                # Notify subscribers
                for subscriber in self._subscribers:
                    if hasattr(subscriber, "on_trade_update"):
                        await subscriber.on_trade_update(data)
            elif event_type == "partial_fill":
                logger.info(f"Order {order_id} partially filled")
                # Notify subscribers
                for subscriber in self._subscribers:
                    if hasattr(subscriber, "on_trade_update"):
                        await subscriber.on_trade_update(data)
            elif event_type == "canceled":
                logger.info(f"Order {order_id} canceled")
            elif event_type == "rejected":
                logger.warning(f"Order {order_id} rejected: {order.get('reject_reason')}")

        except Exception as e:
            logger.error(f"Error handling trade update: {e}", exc_info=DEBUG_MODE)

    async def _handle_bars(self, data):
        """Handle bar data from websocket."""
        try:
            # Extract bar data
            symbol = data.get("S")
            open_price = float(data.get("o", 0))
            high_price = float(data.get("h", 0))
            low_price = float(data.get("l", 0))
            close_price = float(data.get("c", 0))
            volume = int(data.get("v", 0))
            timestamp = datetime.fromtimestamp(data.get("t", 0) / 1000.0)

            # Notify subscribers
            for subscriber in self._subscribers:
                if hasattr(subscriber, "on_bar"):
                    await subscriber.on_bar(
                        symbol, open_price, high_price, low_price, close_price, volume, timestamp
                    )

        except Exception as e:
            logger.error(f"Error handling bar data: {e}", exc_info=DEBUG_MODE)

    async def _handle_quotes(self, data):
        """Handle quote data from websocket."""
        try:
            # Extract quote data
            symbol = data.get("S")
            bid_price = float(data.get("bp", 0))
            ask_price = float(data.get("ap", 0))
            bid_size = int(data.get("bs", 0))
            ask_size = int(data.get("as", 0))
            timestamp = datetime.fromtimestamp(data.get("t", 0) / 1000.0)

            # Notify subscribers
            for subscriber in self._subscribers:
                if hasattr(subscriber, "on_quote"):
                    await subscriber.on_quote(
                        symbol, bid_price, ask_price, bid_size, ask_size, timestamp
                    )

        except Exception as e:
            logger.error(f"Error handling quote data: {e}", exc_info=DEBUG_MODE)

    async def _handle_trades(self, data):
        """Handle trade data from websocket."""
        try:
            # Extract trade data
            symbol = data.get("S")
            price = float(data.get("p", 0))
            size = int(data.get("s", 0))
            timestamp = datetime.fromtimestamp(data.get("t", 0) / 1000.0)

            # Notify subscribers
            for subscriber in self._subscribers:
                if hasattr(subscriber, "on_trade"):
                    await subscriber.on_trade(symbol, price, size, timestamp)

        except Exception as e:
            logger.error(f"Error handling trade data: {e}", exc_info=DEBUG_MODE)

    async def _websocket_handler(self):
        """Main websocket handler with proper locking for thread safety."""
        while True:
            try:
                logger.info("Starting websocket connection...")

                # Reset for new connection (with lock for thread safety)
                async with self._ws_lock:
                    self._connected = False
                    self._subscribed_symbols.clear()

                # Initialize and connect
                self.stream.connect()

                # Register handlers
                self.stream.add_trade_update_handler(self._handle_trade_updates)
                self.stream.add_bars_handler(self._handle_bars)
                self.stream.add_quotes_handler(self._handle_quotes)
                self.stream.add_trades_handler(self._handle_trades)

                # Subscribe to trade updates (account activity)
                self.stream.subscribe_trade_updates()

                # Mark as connected (with lock for thread safety)
                async with self._ws_lock:
                    self._connected = True
                    self._reconnect_attempts = 0
                    self._reconnect_delay = 1  # Reset delay

                logger.info("Websocket connected successfully")

                # Subscribe to market data for all tracked symbols
                await self._subscribe_to_symbols(SYMBOLS)

                # Keep connection alive
                while True:
                    async with self._ws_lock:
                        if not self._connected:
                            break
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Websocket handler cancelled")
                async with self._ws_lock:
                    self._connected = False
                raise
            except Exception as e:
                logger.error(f"Websocket error: {e}", exc_info=DEBUG_MODE)

                async with self._ws_lock:
                    self._connected = False
                    # Implement exponential backoff for reconnection
                    sleep_time = min(
                        self._reconnect_delay * (2**self._reconnect_attempts),
                        self._max_reconnect_delay,
                    )
                    self._reconnect_attempts += 1

                logger.info(
                    f"Attempting to reconnect in {sleep_time} seconds (attempt {self._reconnect_attempts})..."
                )
                await asyncio.sleep(sleep_time)

    async def _subscribe_to_symbols(self, symbols):
        """Subscribe to market data for multiple symbols with proper locking."""
        # P1 FIX: Keep all operations within a single lock to prevent race conditions
        async with self._ws_lock:
            if not self._connected:
                logger.warning("Cannot subscribe to symbols: websocket not connected")
                return False

            try:
                # Subscribe to bars (1-minute timeframe)
                self.stream.subscribe_bars(symbols)

                # Subscribe to quotes
                self.stream.subscribe_quotes(symbols)

                # Subscribe to trades
                self.stream.subscribe_trades(symbols)

                # Add symbols to subscribed set (still within the same lock)
                for symbol in symbols:
                    self._subscribed_symbols.add(symbol)

                logger.info(f"Subscribed to market data for: {', '.join(symbols)}")
                return True

            except Exception as e:
                logger.error(f"Error subscribing to symbols: {e}", exc_info=DEBUG_MODE)
                return False

    async def start_websocket(self):
        """Start the websocket connection with proper locking."""
        async with self._ws_lock:
            if self._ws_task is not None:
                logger.info("Websocket already running")
                return

            self._ws_task = asyncio.create_task(self._websocket_handler())
            logger.info("Started websocket handler task")

    async def stop_websocket(self):
        """Stop the websocket connection with proper locking."""
        async with self._ws_lock:
            if self._ws_task is None:
                logger.info("No websocket running")
                return

            if not self._ws_task.done():
                self._connected = False
                task = self._ws_task
                self._ws_task = None
            else:
                self._ws_task = None
                logger.info("Websocket task already completed")
                return

        # Cancel outside the lock to avoid deadlock
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        logger.info("Stopped websocket handler task")

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_account(self):
        """Get account information."""
        try:
            account = self.trading_client.get_account()
            return account
        except Exception as e:
            logger.error(f"Error getting account info: {e}", exc_info=DEBUG_MODE)
            raise

    async def get_market_status(self):
        """Get current market status."""
        try:
            clock = self.trading_client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
                "timestamp": clock.timestamp,
            }
        except Exception as e:
            logger.error(f"Error getting market status: {e}", exc_info=DEBUG_MODE)
            # Return safe default if error
            return {"is_open": False}

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_positions(self):
        """Get current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_position(self, symbol):
        """Get position for a specific symbol."""
        try:
            # P2 FIX: Validate symbol before API call
            symbol = self._validate_symbol(symbol)
            position = self.trading_client.get_position(symbol)
            return position
        except ValueError as e:
            logger.error(f"Invalid symbol: {e}")
            return None
        except Exception:
            # Position not found, return None
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_tracked_positions(self, strategy):
        """Get positions for a specific strategy."""
        try:
            all_positions = await self.get_positions()
            # For now, return all positions. In the future, we can filter by strategy
            return all_positions
        except Exception as e:
            logger.error(f"Error getting tracked positions: {e}", exc_info=DEBUG_MODE)
            return []

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def submit_order(self, order):
        """Submit an order."""
        try:
            # Convert order to alpaca-py format
            side = OrderSide.BUY if order["side"].lower() == "buy" else OrderSide.SELL

            # Determine order type and create appropriate request
            if order.get("type", "market").lower() == "market":
                order_request = MarketOrderRequest(
                    symbol=order["symbol"],
                    qty=float(order["quantity"]),
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.get("type", "").lower() == "limit":
                order_request = LimitOrderRequest(
                    symbol=order["symbol"],
                    limit_price=float(order["limit_price"]),
                    qty=float(order["quantity"]),
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                raise ValueError(f"Unsupported order type: {order.get('type')}")

            # Submit the order
            result = self.trading_client.submit_order(order_request)
            logger.info(f"Order submitted: {result.id} for {result.symbol} ({result.qty} shares)")
            return result

        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def submit_order_advanced(self, order_request):
        """
        Submit an advanced order using OrderBuilder or direct request object.

        Args:
            order_request: Either an OrderBuilder instance or Alpaca order request object

        Returns:
            Order confirmation from Alpaca
        """
        try:
            # Import OrderBuilder inside method to avoid circular import
            from brokers.order_builder import OrderBuilder

            # If OrderBuilder, build it first
            if isinstance(order_request, OrderBuilder):
                order_request = order_request.build()

            # Submit the order
            result = self.trading_client.submit_order(order_request)
            logger.info(
                f"Advanced order submitted: {result.id} for {result.symbol} "
                f"({result.qty} shares, type={result.type}, class={result.order_class})"
            )
            return result

        except Exception as e:
            logger.error(f"Error submitting advanced order: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def cancel_order(self, order_id: str):
        """
        Cancel an open order by ID.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Canceled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}", exc_info=DEBUG_MODE)
            return False

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            result = self.trading_client.cancel_orders()
            logger.info("Canceled all open orders")
            return result
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}", exc_info=DEBUG_MODE)
            return []

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def replace_order(
        self,
        order_id: str,
        qty: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        client_order_id: Optional[str] = None,
    ):
        """
        Replace an existing order (PATCH endpoint).

        Args:
            order_id: Order ID to replace
            qty: New quantity
            limit_price: New limit price
            stop_price: New stop price
            trail: New trail amount
            time_in_force: New time in force
            client_order_id: New client order ID

        Returns:
            Updated order object
        """
        try:
            # Build replacement request with provided parameters
            replace_params = {}
            if qty is not None:
                replace_params["qty"] = float(qty)
            if limit_price is not None:
                replace_params["limit_price"] = float(limit_price)
            if stop_price is not None:
                replace_params["stop_price"] = float(stop_price)
            if trail is not None:
                replace_params["trail"] = float(trail)
            if time_in_force is not None:
                replace_params["time_in_force"] = time_in_force
            if client_order_id is not None:
                replace_params["client_order_id"] = client_order_id

            replace_request = ReplaceOrderRequest(**replace_params)
            result = self.trading_client.replace_order_by_id(order_id, replace_request)

            logger.info(f"Replaced order: {order_id}")
            return result

        except Exception as e:
            logger.error(f"Error replacing order {order_id}: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_order_by_id(self, order_id: str):
        """Get order by ID."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return order
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}", exc_info=DEBUG_MODE)
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_order_by_client_id(self, client_order_id: str):
        """Get order by client order ID."""
        try:
            order = self.trading_client.get_order_by_client_id(client_order_id)
            return order
        except Exception as e:
            logger.error(
                f"Error getting order by client ID {client_order_id}: {e}", exc_info=DEBUG_MODE
            )
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_orders(self, status: Optional[QueryOrderStatus] = None, limit: int = 100):
        """
        Get orders with specified status.

        Args:
            status: Filter by order status (OPEN, CLOSED, ALL)
            limit: Maximum number of orders to return
        """
        try:
            request_params = GetOrdersRequest(status=status or QueryOrderStatus.OPEN, limit=limit)
            orders = self.trading_client.get_orders(request_params)
            return orders
        except Exception as e:
            logger.error(f"Error getting orders: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_last_price(self, symbol):
        """Get last price for a symbol."""
        try:
            # P2 FIX: Validate symbol before API call
            symbol = self._validate_symbol(symbol)

            request_params = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            response = self.data_client.get_stock_latest_trade(request_params)

            if symbol in response:
                return float(response[symbol].price)
            else:
                logger.warning(f"No price data found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error getting last price for {symbol}: {e}", exc_info=DEBUG_MODE)
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_bars(self, symbol, timeframe=TimeFrame.Day, limit=100, start=None, end=None):
        """Get historical bars for a symbol."""
        try:
            # P2 FIX: Validate symbol before API call
            symbol = self._validate_symbol(symbol)

            # Convert string timeframe to TimeFrame object if needed
            if isinstance(timeframe, str):
                timeframe_map = {
                    "1Min": TimeFrame.Minute,
                    "5Min": TimeFrame(5, "Min"),
                    "15Min": TimeFrame(15, "Min"),
                    "1Hour": TimeFrame.Hour,
                    "1Day": TimeFrame.Day,
                    "Day": TimeFrame.Day,
                    "Hour": TimeFrame.Hour,
                    "Minute": TimeFrame.Minute,
                }
                timeframe = timeframe_map.get(timeframe, TimeFrame.Day)
                logger.debug(f"Converted timeframe string to TimeFrame object: {timeframe}")

            # Set default dates if not provided
            if end is None:
                end = datetime.now().date()
            if start is None:
                start = (datetime.now() - timedelta(days=limit)).date()

            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol], timeframe=timeframe, start=start, end=end
            )

            bars = self.data_client.get_stock_bars(request_params)

            if symbol in bars.data:
                return bars.data[symbol]
            else:
                logger.warning(f"No bar data found for {symbol}")
                return []

        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}", exc_info=DEBUG_MODE)
            return []

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_news(self, symbol, start, end):
        """
        Get news for a symbol.

        ⚠️  WARNING: News API not yet implemented!

        The alpaca-py library's news API is not yet integrated.
        This method returns an empty list to prevent fake data from being used.

        DO NOT use SentimentStockStrategy until this is properly implemented.

        To implement:
        1. Use Alpaca News API when available in alpaca-py
        2. Alternative: Integrate NewsAPI.org or Alpha Vantage News
        3. Validate news data before returning
        """
        try:
            # Log warning about missing news API
            logger.warning(f"⚠️  News API not implemented - returning empty news for {symbol}")
            logger.warning("   SentimentStockStrategy will not work without real news data!")

            # Return empty list instead of fake data
            # This prevents sentiment strategy from making trades on fabricated information
            return []

        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}", exc_info=DEBUG_MODE)
            return []
