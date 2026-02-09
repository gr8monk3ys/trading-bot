import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional

import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    CryptoLatestTradeRequest,
    StockBarsRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
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
from utils.audit_log import AuditEventType, AuditLog, log_order_event
from utils.crypto_utils import (
    is_crypto_symbol,
    normalize_crypto_symbol,
)
from utils.order_lifecycle import OrderLifecycleTracker, OrderState

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


class GatewayBypassError(BrokerError):
    """
    Raised when attempting to submit orders without using OrderGateway.

    CRITICAL SAFETY: All orders MUST route through OrderGateway to ensure:
    - Circuit breaker checks
    - Position conflict detection
    - Risk manager limits enforcement
    - Audit trail maintenance

    To fix this error, use order_gateway.submit_order() instead of
    broker.submit_order_advanced() directly.
    """

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
    Supports both stocks and 24/7 cryptocurrency trading.
    """

    NAME = "alpaca"
    IS_BACKTESTING_BROKER = False

    # CRYPTO_PAIRS is now imported from utils.crypto_utils for consistency

    # Default timeout for API calls (in seconds)
    DEFAULT_API_TIMEOUT = 30.0
    # Timeout for data-heavy operations (bars, portfolio history)
    DATA_API_TIMEOUT = 60.0
    # Timeout for order operations (more critical, shorter timeout)
    ORDER_API_TIMEOUT = 15.0

    async def _async_call_with_timeout(
        self,
        func,
        *args,
        timeout: float = None,
        operation_name: str = "API call",
        **kwargs
    ):
        """
        Execute a sync function in a thread pool with timeout protection.

        This wrapper prevents broker API calls from hanging indefinitely,
        which could freeze the entire trading system.

        Args:
            func: Synchronous function to call
            *args: Positional arguments for func
            timeout: Timeout in seconds (defaults to DEFAULT_API_TIMEOUT)
            operation_name: Description for logging on timeout
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)

        Raises:
            asyncio.TimeoutError: If call exceeds timeout
            Exception: Any exception from the underlying function
        """
        if timeout is None:
            timeout = self.DEFAULT_API_TIMEOUT

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"TIMEOUT: {operation_name} exceeded {timeout}s limit. "
                "This may indicate network issues or API problems."
            )
            raise

    def __init__(self, paper=True, audit_log: Optional[AuditLog] = None):
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

            # Store credentials for crypto client initialization (lazy loaded)
            self._api_key = _api_key
            self._api_secret = _api_secret

            # Initialize the trading client
            self.trading_client = TradingClient(
                api_key=_api_key,
                secret_key=_api_secret,
                paper=self.paper,
                url_override="https://paper-api.alpaca.markets" if self.paper else None,
            )

            # Initialize the data client for stocks
            self.data_client = StockHistoricalDataClient(api_key=_api_key, secret_key=_api_secret)

            # Crypto clients (lazy initialized to avoid unnecessary connections)
            self._crypto_data_client = None
            self._crypto_stream = None

            # Initialize the stream for WebSockets (stocks)
            self.stream = StockDataStream(
                api_key=_api_key,
                secret_key=_api_secret,
                url_override="https://paper-api.alpaca.markets/stream" if self.paper else None,
            )

            self._subscribed_symbols = set()  # Keep track of subscribed symbols

            # Performance optimization: TTL-based price cache to reduce API calls
            self._price_cache = {}  # {symbol: (price, timestamp)}
            self._price_cache_ttl = timedelta(seconds=5)  # Cache prices for 5 seconds

            # INSTITUTIONAL SAFETY: Gateway enforcement flag
            # When True, direct calls to submit_order_advanced() will raise GatewayBypassError
            # All orders must route through OrderGateway for safety checks
            self._gateway_required = False  # Set to True after OrderGateway is initialized
            self._gateway_caller_token = None  # Token for authorized gateway calls

            # INSTITUTIONAL SAFETY: Partial fill tracking
            # Tracks order fills and handles unfilled quantities
            from utils.partial_fill_tracker import PartialFillPolicy, PartialFillTracker
            self._partial_fill_tracker = PartialFillTracker(
                policy=PartialFillPolicy.ALERT_ONLY,  # Default to alerting
            )

            # Audit log (optional)
            self._audit_log = audit_log
            self._order_metadata: Dict[str, Dict] = {}
            self._lifecycle_tracker: Optional[OrderLifecycleTracker] = None
            self._position_manager = None

            # P0 FIX: Removed unused config dict that stored credentials in memory

        except Exception as e:
            logger.error(f"Error initializing AlpacaBroker: {e}", exc_info=DEBUG_MODE)
            raise

    def set_audit_log(self, audit_log: Optional[AuditLog]) -> None:
        """Attach an audit log for order lifecycle events."""
        self._audit_log = audit_log

    def set_position_manager(self, position_manager) -> None:
        """Attach a position manager for fill-driven updates."""
        self._position_manager = position_manager

    def set_lifecycle_tracker(self, tracker: Optional[OrderLifecycleTracker]) -> None:
        """Attach an order lifecycle tracker."""
        self._lifecycle_tracker = tracker

    def register_order_metadata(self, order_id: str, metadata: Dict) -> None:
        """Store order metadata for lifecycle updates."""
        self._order_metadata[order_id] = metadata

    def track_order_for_fills(self, order_id: str, symbol: str, side: str, qty: float) -> None:
        """Register an order with the partial fill tracker."""
        self._partial_fill_tracker.track_order(order_id, symbol, side, qty)

    # =========================================================================
    # CRYPTO HELPER METHODS
    # =========================================================================

    def _get_crypto_data_client(self) -> CryptoHistoricalDataClient:
        """
        Lazy load crypto data client.

        Returns:
            CryptoHistoricalDataClient instance
        """
        if self._crypto_data_client is None:
            # Crypto data client does not require authentication for public data
            self._crypto_data_client = CryptoHistoricalDataClient()
            logger.info("Initialized crypto data client")
        return self._crypto_data_client

    def is_crypto(self, symbol: str) -> bool:
        """
        Check if symbol is a cryptocurrency pair.

        Delegates to utils.crypto_utils.is_crypto_symbol for consistency.

        Args:
            symbol: Symbol to check (e.g., "BTC/USD", "BTCUSD", "BTC-USD")

        Returns:
            True if the symbol is a crypto pair, False otherwise
        """
        return is_crypto_symbol(symbol)

    def normalize_crypto_symbol(self, symbol: str) -> str:
        """
        Normalize crypto symbol to Alpaca format (e.g., BTCUSD -> BTC/USD).

        Delegates to utils.crypto_utils.normalize_crypto_symbol for consistency.

        Args:
            symbol: Crypto symbol in any format

        Returns:
            Normalized symbol in Alpaca format (e.g., "BTC/USD")

        Raises:
            ValueError: If symbol is not a recognized crypto pair
        """
        return normalize_crypto_symbol(symbol)

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

            # Extract fill info for tracking
            filled_qty = float(order.get("filled_qty", 0))
            filled_avg_price = float(order.get("filled_avg_price", 0)) if order.get("filled_avg_price") else 0.0
            symbol = order.get("symbol", "")
            side = order.get("side", "")
            meta = self._order_metadata.get(str(order_id), {})
            strategy_name = meta.get("strategy_name", "unknown")

            # Handle different trade events
            if event_type == "fill":
                logger.info(f"Order {order_id} filled")

                # INSTITUTIONAL: Record fill in tracker
                event = await self._partial_fill_tracker.record_fill(
                    order_id=str(order_id),
                    filled_qty=filled_qty,
                    fill_price=filled_avg_price,
                    is_final=True,
                    status="filled",
                )
                if self._lifecycle_tracker:
                    self._lifecycle_tracker.update_state(str(order_id), OrderState.FILLED)
                if self._position_manager and event:
                    await self._position_manager.apply_fill(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        side=side,
                        filled_qty=filled_qty,
                        fill_price=filled_avg_price,
                        delta_qty=event.delta_qty,
                    )

                if self._audit_log:
                    log_order_event(
                        self._audit_log,
                        AuditEventType.ORDER_FILLED,
                        order_id=str(order_id),
                        symbol=symbol,
                        side=side,
                        quantity=filled_qty,
                        price=filled_avg_price,
                        status="filled",
                    )

                # Notify subscribers
                for subscriber in self._subscribers:
                    if hasattr(subscriber, "on_trade_update"):
                        await subscriber.on_trade_update(data)

            elif event_type == "partial_fill":
                logger.info(f"Order {order_id} partially filled: {filled_qty} @ ${filled_avg_price:.2f}")

                # INSTITUTIONAL: Record partial fill in tracker
                event = await self._partial_fill_tracker.record_fill(
                    order_id=str(order_id),
                    filled_qty=filled_qty,
                    fill_price=filled_avg_price,
                    is_final=False,
                    status="partial",
                )
                if self._lifecycle_tracker:
                    self._lifecycle_tracker.update_state(str(order_id), OrderState.PARTIAL)
                if self._position_manager and event:
                    await self._position_manager.apply_fill(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        side=side,
                        filled_qty=filled_qty,
                        fill_price=filled_avg_price,
                        delta_qty=event.delta_qty,
                    )

                if self._audit_log:
                    log_order_event(
                        self._audit_log,
                        AuditEventType.ORDER_PARTIAL_FILL,
                        order_id=str(order_id),
                        symbol=symbol,
                        side=side,
                        quantity=filled_qty,
                        price=filled_avg_price,
                        status="partial",
                    )

                # Notify subscribers
                for subscriber in self._subscribers:
                    if hasattr(subscriber, "on_trade_update"):
                        await subscriber.on_trade_update(data)

            elif event_type == "canceled":
                logger.info(f"Order {order_id} canceled")

                # INSTITUTIONAL: Record cancellation
                await self._partial_fill_tracker.record_fill(
                    order_id=str(order_id),
                    filled_qty=filled_qty,
                    fill_price=filled_avg_price,
                    is_final=True,
                    status="canceled",
                )
                if self._lifecycle_tracker:
                    self._lifecycle_tracker.update_state(str(order_id), OrderState.CANCELED)

                if self._audit_log:
                    log_order_event(
                        self._audit_log,
                        AuditEventType.ORDER_CANCELED,
                        order_id=str(order_id),
                        symbol=symbol,
                        side=side,
                        quantity=filled_qty,
                        price=filled_avg_price,
                        status="canceled",
                    )

            elif event_type == "rejected":
                logger.warning(f"Order {order_id} rejected: {order.get('reject_reason')}")

                # INSTITUTIONAL: Record rejection
                await self._partial_fill_tracker.record_fill(
                    order_id=str(order_id),
                    filled_qty=0,
                    fill_price=0,
                    is_final=True,
                    status="rejected",
                )
                if self._lifecycle_tracker:
                    self._lifecycle_tracker.update_state(str(order_id), OrderState.REJECTED)

                if self._audit_log:
                    log_order_event(
                        self._audit_log,
                        AuditEventType.ORDER_REJECTED,
                        order_id=str(order_id),
                        symbol=symbol,
                        side=side,
                        quantity=0,
                        price=0,
                        rejection_reason=order.get("reject_reason"),
                        status="rejected",
                    )

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

    # =========================================================================
    # PARTIAL FILL TRACKING METHODS
    # =========================================================================

    def set_partial_fill_policy(self, policy: str) -> None:
        """
        Set the policy for handling partial fills.

        Args:
            policy: One of 'alert_only', 'auto_resubmit', 'cancel_remainder', 'track_only'
        """
        from utils.partial_fill_tracker import PartialFillPolicy
        self._partial_fill_tracker.set_policy(PartialFillPolicy(policy))

    def register_partial_fill_callback(self, callback) -> None:
        """
        Register a callback for partial fill events.

        The callback will be called with a PartialFillEvent when a partial
        fill is detected.

        Args:
            callback: Async function taking PartialFillEvent
        """
        self._partial_fill_tracker.register_callback(callback)

    def set_partial_fill_resubmit_callback(self, callback) -> None:
        """
        Set the callback for auto-resubmitting partial fills.

        Only used when policy is AUTO_RESUBMIT.

        Args:
            callback: Async function taking (symbol, side, qty) returning new order_id
        """
        self._partial_fill_tracker.set_resubmit_callback(callback)

    def get_partial_fill_statistics(self):
        """Get aggregate statistics on partial fills."""
        return self._partial_fill_tracker.get_statistics()

    def get_order_fill_status(self, order_id: str):
        """Get the current fill status of an order."""
        return self._partial_fill_tracker.get_order_status(order_id)

    def get_pending_partial_fills(self):
        """Get all orders with unfilled quantities."""
        return self._partial_fill_tracker.get_pending_orders()

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_account(self):
        """Get account information."""
        try:
            # Use timeout-protected async call to prevent hanging
            account = await self._async_call_with_timeout(
                self.trading_client.get_account,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name="get_account"
            )
            return account
        except asyncio.TimeoutError:
            raise BrokerConnectionError("Account fetch timed out - broker may be unreachable") from None
        except Exception as e:
            logger.error(f"Error getting account info: {e}", exc_info=DEBUG_MODE)
            raise

    async def get_market_status(self):
        """Get current market status."""
        try:
            # Use timeout-protected async call
            clock = await self._async_call_with_timeout(
                self.trading_client.get_clock,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name="get_market_status"
            )
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
                "timestamp": clock.timestamp,
            }
        except asyncio.TimeoutError:
            logger.warning("Market status check timed out, assuming closed")
            return {"is_open": False}
        except Exception as e:
            logger.error(f"Error getting market status: {e}", exc_info=DEBUG_MODE)
            # Return safe default if error
            return {"is_open": False}

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_positions(self):
        """Get current positions."""
        try:
            # Use timeout-protected async call to prevent hanging
            positions = await self._async_call_with_timeout(
                self.trading_client.get_all_positions,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name="get_positions"
            )
            return positions
        except asyncio.TimeoutError:
            raise BrokerConnectionError("Position fetch timed out - broker may be unreachable") from None
        except Exception as e:
            logger.error(f"Error getting positions: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_position(self, symbol):
        """Get position for a specific symbol."""
        try:
            # P2 FIX: Validate symbol before API call
            symbol = self._validate_symbol(symbol)
            # Use timeout-protected async call to prevent hanging
            position = await self._async_call_with_timeout(
                self.trading_client.get_position,
                symbol,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name=f"get_position({symbol})"
            )
            return position
        except ValueError as e:
            logger.error(f"Invalid symbol: {e}")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Position fetch for {symbol} timed out")
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
    async def get_asset(self, symbol: str) -> Optional[dict]:
        """
        Get asset information including extended hours and overnight trading status.

        This method retrieves comprehensive asset details from Alpaca, including
        whether the symbol supports overnight trading via Blue Ocean ATS.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Dict with asset attributes:
            - symbol: Stock symbol
            - name: Company name
            - exchange: Primary exchange
            - tradeable: Whether the asset can be traded
            - marginable: Whether margin trading is allowed
            - shortable: Whether short selling is allowed
            - fractionable: Whether fractional shares are allowed
            - overnight_tradeable: Whether overnight trading is available (Blue Ocean ATS)
            - overnight_halted: Whether overnight trading is currently halted
            - easy_to_borrow: Whether shares are easy to borrow for shorting

            Returns None on error or if asset not found.

        Example:
            asset = await broker.get_asset("AAPL")
            if asset and asset["overnight_tradeable"]:
                # Can trade overnight via Blue Ocean ATS
                pass
        """
        try:
            # Validate symbol
            symbol = self._validate_symbol(symbol)

            # Use timeout-protected async call
            asset = await self._async_call_with_timeout(
                self.trading_client.get_asset,
                symbol,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name=f"get_asset({symbol})"
            )

            return {
                "symbol": asset.symbol,
                "name": getattr(asset, "name", None),
                "exchange": getattr(asset, "exchange", None),
                "asset_class": getattr(asset, "asset_class", None),
                "tradeable": getattr(asset, "tradable", False),
                "marginable": getattr(asset, "marginable", False),
                "shortable": getattr(asset, "shortable", False),
                "fractionable": getattr(asset, "fractionable", False),
                "easy_to_borrow": getattr(asset, "easy_to_borrow", False),
                # Overnight trading attributes (Blue Ocean ATS)
                "overnight_tradeable": getattr(asset, "overnight_tradeable", False),
                "overnight_halted": getattr(asset, "overnight_halted", False),
                # Extended hours status
                "status": getattr(asset, "status", None),
            }

        except ValueError as e:
            logger.error(f"Invalid symbol: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching asset {symbol}: {e}", exc_info=DEBUG_MODE)
            return None

    async def is_overnight_tradeable(self, symbol: str) -> bool:
        """
        Check if a symbol supports overnight trading via Blue Ocean ATS.

        This is a convenience method that wraps get_asset() to quickly check
        overnight trading eligibility.

        Args:
            symbol: Stock symbol

        Returns:
            True if overnight trading is available and not halted,
            False otherwise
        """
        try:
            asset = await self.get_asset(symbol)
            if asset:
                return (
                    asset.get("overnight_tradeable", False) and
                    not asset.get("overnight_halted", True)
                )
            return False
        except Exception as e:
            logger.warning(f"Error checking overnight status for {symbol}: {e}")
            return False

    # =========================================================================
    # ALMGREN-CHRISS MARKET IMPACT MODEL
    # Ported from BacktestBroker for live trading execution awareness
    # =========================================================================

    # Maximum participation rate (never trade > 10% of ADV)
    MAX_PARTICIPATION_RATE = 0.10

    async def _calculate_market_impact(
        self, symbol: str, qty: float, side: str
    ) -> Dict:
        """
        Calculate expected slippage using Almgren-Chriss market impact model.

        This model estimates the cost of executing an order based on:
        - Temporary impact: Short-term price pressure from order flow
        - Permanent impact: Information content revealed by the trade

        Args:
            symbol: Stock symbol
            qty: Order quantity (shares)
            side: 'buy' or 'sell'

        Returns:
            Dict with impact metrics:
                - expected_slippage_pct: Total expected slippage as percentage
                - participation_rate: Order size as fraction of ADV
                - temporary_impact: Short-term price pressure
                - permanent_impact: Long-term price impact
                - safe_to_trade: Whether order passes liquidity check
        """
        try:
            # Get historical bars for volume and volatility calculation
            bars = await self.get_bars(symbol, timeframe="1Day", limit=20)

            if not bars or len(bars) < 5:
                logger.warning(f"Insufficient data for {symbol}, using conservative defaults")
                return {
                    "expected_slippage_pct": 0.005,  # Default 0.5%
                    "participation_rate": 0.0,
                    "temporary_impact": 0.0,
                    "permanent_impact": 0.0,
                    "safe_to_trade": True,
                    "avg_daily_volume": None,
                }

            # Calculate average daily volume
            volumes = [float(b.volume) for b in bars if hasattr(b, 'volume')]
            avg_daily_volume = np.mean(volumes) if volumes else 1000000.0
            avg_daily_volume = max(avg_daily_volume, 100000.0)  # Floor at 100K

            # Calculate volatility (annualized)
            closes = [float(b.close) for b in bars if hasattr(b, 'close')]
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                volatility = np.std(returns) * np.sqrt(252)
            else:
                volatility = 0.30  # Default 30% volatility

            # Participation rate
            participation_rate = qty / avg_daily_volume

            # Almgren-Chriss coefficients
            c_temp = 0.6   # Temporary impact coefficient
            d_perm = 0.15  # Permanent impact coefficient

            # Calculate impacts
            temporary_impact = c_temp * volatility * np.sqrt(participation_rate)
            permanent_impact = d_perm * volatility * participation_rate

            # Total impact (capped at 10%)
            total_impact = min(temporary_impact + permanent_impact, 0.10)

            # Check if order is safe to trade
            safe_to_trade = participation_rate <= self.MAX_PARTICIPATION_RATE

            if not safe_to_trade:
                logger.warning(
                    f"Order for {qty:.0f} shares of {symbol} exceeds "
                    f"{self.MAX_PARTICIPATION_RATE*100:.0f}% of ADV "
                    f"({avg_daily_volume:.0f}). Participation: {participation_rate*100:.1f}%"
                )

            return {
                "expected_slippage_pct": total_impact,
                "participation_rate": participation_rate,
                "temporary_impact": temporary_impact,
                "permanent_impact": permanent_impact,
                "safe_to_trade": safe_to_trade,
                "avg_daily_volume": avg_daily_volume,
                "volatility": volatility,
            }

        except Exception as e:
            logger.warning(f"Error calculating market impact for {symbol}: {e}")
            return {
                "expected_slippage_pct": 0.005,
                "participation_rate": 0.0,
                "temporary_impact": 0.0,
                "permanent_impact": 0.0,
                "safe_to_trade": True,
                "avg_daily_volume": None,
            }

    async def check_liquidity(self, symbol: str, qty: float) -> bool:
        """
        Check if order size is safe relative to average daily volume.

        Args:
            symbol: Stock symbol
            qty: Order quantity

        Returns:
            True if order passes liquidity check
        """
        impact = await self._calculate_market_impact(symbol, qty, "buy")
        return impact["safe_to_trade"]

    async def get_expected_slippage(self, symbol: str, qty: float, side: str) -> float:
        """
        Get expected slippage for an order.

        Args:
            symbol: Stock symbol
            qty: Order quantity
            side: 'buy' or 'sell'

        Returns:
            Expected slippage as decimal (e.g., 0.005 = 0.5%)
        """
        impact = await self._calculate_market_impact(symbol, qty, side)
        return impact["expected_slippage_pct"]

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

            # Submit the order with timeout protection (critical operation)
            result = await self._async_call_with_timeout(
                self.trading_client.submit_order,
                order_request,
                timeout=self.ORDER_API_TIMEOUT,
                operation_name=f"submit_order({order['symbol']})"
            )
            logger.info(f"Order submitted: {result.id} for {result.symbol} ({result.qty} shares)")
            return result

        except asyncio.TimeoutError:
            raise OrderError(f"Order submission timed out for {order['symbol']} - check order status manually") from None
        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=DEBUG_MODE)
            raise

    def enable_gateway_requirement(self) -> str:
        """
        Enable mandatory OrderGateway routing for all orders.

        CRITICAL SAFETY: Once enabled, direct calls to submit_order_advanced()
        will raise GatewayBypassError. Only the OrderGateway can submit orders
        using the returned authorization token.

        Returns:
            Authorization token that must be passed to _internal_submit_order

        Usage:
            gateway_token = broker.enable_gateway_requirement()
            # Store token in OrderGateway
            # Now all orders MUST go through OrderGateway
        """
        import secrets
        self._gateway_caller_token = secrets.token_hex(16)
        self._gateway_required = True
        logger.info(
            "🔒 GATEWAY ENFORCEMENT ENABLED: All orders must route through OrderGateway"
        )
        return self._gateway_caller_token

    def disable_gateway_requirement(self):
        """
        Disable gateway requirement (for testing only).

        WARNING: This should NEVER be called in production.
        """
        self._gateway_required = False
        self._gateway_caller_token = None
        logger.warning(
            "⚠️ GATEWAY ENFORCEMENT DISABLED - Direct order submission allowed"
        )

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def submit_order_advanced(self, order_request, check_impact: bool = True):
        """
        Submit an advanced order using OrderBuilder or direct request object.

        IMPORTANT: When gateway enforcement is enabled, this method will raise
        GatewayBypassError. Use OrderGateway.submit_order() instead.

        Now includes Almgren-Chriss market impact calculation for execution awareness.

        Args:
            order_request: Either an OrderBuilder instance or Alpaca order request object
            check_impact: If True, calculate and log expected market impact

        Returns:
            Order confirmation from Alpaca

        Raises:
            GatewayBypassError: If gateway enforcement is enabled (use OrderGateway instead)
        """
        # INSTITUTIONAL SAFETY: Enforce gateway requirement
        if self._gateway_required:
            raise GatewayBypassError(
                "Direct order submission is disabled. "
                "All orders must route through OrderGateway for safety checks. "
                "Use order_gateway.submit_order() instead of broker.submit_order_advanced()."
            )

        try:
            # Import OrderBuilder inside method to avoid circular import
            from brokers.order_builder import OrderBuilder

            # If OrderBuilder, build it first
            if isinstance(order_request, OrderBuilder):
                order_request = order_request.build()

            # Calculate market impact before submission (if enabled)
            impact_info = None
            if check_impact:
                try:
                    symbol = order_request.symbol
                    qty = float(order_request.qty) if order_request.qty else 0
                    side = str(order_request.side).lower() if hasattr(order_request, 'side') else 'buy'

                    if qty > 0:
                        impact_info = await self._calculate_market_impact(symbol, qty, side)

                        # Log impact metrics
                        if impact_info["participation_rate"] > 0.01:  # Only log if meaningful
                            logger.info(
                                f"📊 Market Impact Analysis for {symbol}: "
                                f"Expected slippage: {impact_info['expected_slippage_pct']*100:.2f}%, "
                                f"Participation: {impact_info['participation_rate']*100:.1f}% of ADV"
                            )

                        # Warn if order is large relative to volume
                        if not impact_info["safe_to_trade"]:
                            logger.warning(
                                f"⚠️ LARGE ORDER WARNING: {symbol} order of {qty:.0f} shares "
                                f"exceeds {self.MAX_PARTICIPATION_RATE*100:.0f}% of ADV. "
                                f"Consider splitting or using VWAP execution."
                            )
                except Exception as e:
                    logger.debug(f"Could not calculate market impact: {e}")

            # Submit the order with timeout protection (critical operation)
            result = await self._async_call_with_timeout(
                self.trading_client.submit_order,
                order_request,
                timeout=self.ORDER_API_TIMEOUT,
                operation_name=f"submit_order_advanced({order_request.symbol})"
            )

            # Build size info for logging (qty or notional)
            if hasattr(result, 'notional') and result.notional is not None:
                size_info = f"${float(result.notional):.2f} notional"
            else:
                size_info = f"{result.qty} shares"

            logger.info(
                f"Advanced order submitted: {result.id} for {result.symbol} "
                f"({size_info}, type={result.type}, class={result.order_class})"
            )

            # INSTITUTIONAL: Track order for partial fill monitoring
            if result.qty:
                qty = float(result.qty)
                side = str(result.side).lower() if hasattr(result, 'side') else 'buy'
                self._partial_fill_tracker.track_order(
                    order_id=str(result.id),
                    symbol=result.symbol,
                    side=side,
                    requested_qty=qty,
                )

            return result

        except asyncio.TimeoutError:
            raise OrderError("Advanced order submission timed out - check order status manually") from None
        except Exception as e:
            logger.error(f"Error submitting advanced order: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def _internal_submit_order(self, order_request, gateway_token: str, check_impact: bool = True):
        """
        Internal order submission method for authorized callers (OrderGateway only).

        PRIVATE API: This method should ONLY be called by OrderGateway with
        the authorization token obtained from enable_gateway_requirement().

        Args:
            order_request: Either an OrderBuilder instance or Alpaca order request object
            gateway_token: Authorization token from enable_gateway_requirement()
            check_impact: If True, calculate and log expected market impact

        Returns:
            Order confirmation from Alpaca

        Raises:
            GatewayBypassError: If invalid or missing gateway token
        """
        # Verify authorization token
        if self._gateway_required:
            if not gateway_token or gateway_token != self._gateway_caller_token:
                raise GatewayBypassError(
                    "Invalid gateway authorization token. "
                    "This method is reserved for OrderGateway internal use only."
                )

        try:
            # Import OrderBuilder inside method to avoid circular import
            from brokers.order_builder import OrderBuilder

            # If OrderBuilder, build it first
            if isinstance(order_request, OrderBuilder):
                order_request = order_request.build()

            # Calculate market impact before submission (if enabled)
            if check_impact:
                try:
                    symbol = order_request.symbol
                    qty = float(order_request.qty) if order_request.qty else 0
                    side = str(order_request.side).lower() if hasattr(order_request, 'side') else 'buy'

                    if qty > 0:
                        impact_info = await self._calculate_market_impact(symbol, qty, side)

                        # Log impact metrics
                        if impact_info["participation_rate"] > 0.01:
                            logger.info(
                                f"📊 Market Impact Analysis for {symbol}: "
                                f"Expected slippage: {impact_info['expected_slippage_pct']*100:.2f}%, "
                                f"Participation: {impact_info['participation_rate']*100:.1f}% of ADV"
                            )

                        # Warn if order is large relative to volume
                        if not impact_info["safe_to_trade"]:
                            logger.warning(
                                f"⚠️ LARGE ORDER WARNING: {symbol} order of {qty:.0f} shares "
                                f"exceeds {self.MAX_PARTICIPATION_RATE*100:.0f}% of ADV. "
                                f"Consider splitting or using VWAP execution."
                            )
                except Exception as e:
                    logger.debug(f"Could not calculate market impact: {e}")

            # Submit the order with timeout protection
            result = await self._async_call_with_timeout(
                self.trading_client.submit_order,
                order_request,
                timeout=self.ORDER_API_TIMEOUT,
                operation_name=f"_internal_submit_order({order_request.symbol})"
            )

            # Build size info for logging
            if hasattr(result, 'notional') and result.notional is not None:
                size_info = f"${float(result.notional):.2f} notional"
            else:
                size_info = f"{result.qty} shares"

            logger.info(
                f"[GATEWAY] Order submitted: {result.id} for {result.symbol} "
                f"({size_info}, type={result.type}, class={result.order_class})"
            )

            # INSTITUTIONAL: Track order for partial fill monitoring
            if result.qty:
                qty = float(result.qty)
                side = str(result.side).lower() if hasattr(result, 'side') else 'buy'
                self._partial_fill_tracker.track_order(
                    order_id=str(result.id),
                    symbol=result.symbol,
                    side=side,
                    requested_qty=qty,
                )

            return result

        except asyncio.TimeoutError:
            raise OrderError("Order submission timed out - check order status manually") from None
        except GatewayBypassError:
            raise  # Re-raise gateway errors
        except Exception as e:
            logger.error(f"Error in _internal_submit_order: {e}", exc_info=DEBUG_MODE)
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
            # Use timeout-protected async call (critical operation)
            await self._async_call_with_timeout(
                self.trading_client.cancel_order_by_id,
                order_id,
                timeout=self.ORDER_API_TIMEOUT,
                operation_name=f"cancel_order({order_id})"
            )
            logger.info(f"Canceled order: {order_id}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Cancel order {order_id} timed out - check order status manually")
            return False
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}", exc_info=DEBUG_MODE)
            return False

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            # Use timeout-protected async call (critical operation)
            result = await self._async_call_with_timeout(
                self.trading_client.cancel_orders,
                timeout=self.ORDER_API_TIMEOUT,
                operation_name="cancel_all_orders"
            )
            logger.info("Canceled all open orders")
            return result
        except asyncio.TimeoutError:
            logger.error("Cancel all orders timed out - check order status manually")
            return []
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
            # Use asyncio.to_thread to avoid blocking the event loop
            result = await asyncio.to_thread(
                self.trading_client.replace_order_by_id, order_id, replace_request
            )

            logger.info(f"Replaced order: {order_id}")
            if self._audit_log:
                self._audit_log.log(
                    AuditEventType.ORDER_MODIFIED,
                    {
                        "order_id": order_id,
                        "qty": qty,
                        "limit_price": limit_price,
                        "stop_price": stop_price,
                        "trail": trail,
                        "time_in_force": str(time_in_force) if time_in_force else None,
                        "client_order_id": client_order_id,
                    },
                )
            return result

        except Exception as e:
            logger.error(f"Error replacing order {order_id}: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_order_by_id(self, order_id: str):
        """Get order by ID."""
        try:
            # Use asyncio.to_thread to avoid blocking the event loop
            order = await asyncio.to_thread(self.trading_client.get_order_by_id, order_id)
            return order
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}", exc_info=DEBUG_MODE)
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_order_by_client_id(self, client_order_id: str):
        """Get order by client order ID."""
        try:
            # Use asyncio.to_thread to avoid blocking the event loop
            order = await asyncio.to_thread(
                self.trading_client.get_order_by_client_id, client_order_id
            )
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
            # Use timeout-protected async call
            orders = await self._async_call_with_timeout(
                self.trading_client.get_orders,
                request_params,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name="get_orders"
            )
            return orders
        except asyncio.TimeoutError:
            raise BrokerConnectionError("Get orders timed out - broker may be unreachable") from None
        except Exception as e:
            logger.error(f"Error getting orders: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_last_price(self, symbol):
        """Get last price for a symbol with TTL-based caching."""
        try:
            # P2 FIX: Validate symbol before API call
            symbol = self._validate_symbol(symbol)

            # Performance optimization: Check cache first
            now = datetime.now()
            if symbol in self._price_cache:
                cached_price, cached_time = self._price_cache[symbol]
                if now - cached_time < self._price_cache_ttl:
                    logger.debug(f"Price cache hit for {symbol}: ${cached_price:.2f}")
                    return cached_price

            # Cache miss - fetch from API with timeout protection
            request_params = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            response = await self._async_call_with_timeout(
                self.data_client.get_stock_latest_trade,
                request_params,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name=f"get_last_price({symbol})"
            )

            if symbol in response:
                price = float(response[symbol].price)
                # Cache the result before returning
                self._price_cache[symbol] = (price, now)
                return price
            else:
                logger.warning(f"No price data found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error getting last price for {symbol}: {e}", exc_info=DEBUG_MODE)
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_last_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Get last trade prices for multiple symbols in a single API call.

        More efficient than calling get_last_price() multiple times.

        Args:
            symbols: List of stock symbols to get prices for

        Returns:
            Dict mapping symbol to price (None if not available)
        """
        if not symbols:
            return {}

        try:
            # Validate and normalize symbols
            validated_symbols = [self._validate_symbol(s) for s in symbols]

            request_params = StockLatestTradeRequest(symbol_or_symbols=validated_symbols)
            response = await self._async_call_with_timeout(
                self.data_client.get_stock_latest_trade,
                request_params,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name=f"get_last_prices({len(validated_symbols)} symbols)"
            )

            result = {}
            now = datetime.now()
            for symbol in validated_symbols:
                if symbol in response:
                    price = float(response[symbol].price)
                    result[symbol] = price
                    # Update cache for individual symbol lookups
                    self._price_cache[symbol] = (price, now)
                else:
                    result[symbol] = None

            logger.debug(f"Fetched batch prices for {len(result)} symbols")
            return result

        except ValueError as e:
            logger.error(f"Invalid symbol in batch: {e}")
            return dict.fromkeys(symbols)
        except Exception as e:
            logger.error(f"Error fetching batch prices for {symbols}: {e}", exc_info=DEBUG_MODE)
            return dict.fromkeys(symbols)

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

            # Use timeout-protected async call (longer timeout for data-heavy operations)
            bars = await self._async_call_with_timeout(
                self.data_client.get_stock_bars,
                request_params,
                timeout=self.DATA_API_TIMEOUT,
                operation_name=f"get_bars({symbol})"
            )

            if symbol in bars.data:
                return bars.data[symbol]
            else:
                logger.warning(f"No bar data found for {symbol}")
                return []

        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}", exc_info=DEBUG_MODE)
            return []

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_news(self, symbols, start=None, end=None, limit=50):
        """
        Get news for symbols using Alpaca News API.

        Args:
            symbols: Single symbol string or list of symbols
            start: Start datetime (default: 24 hours ago)
            end: End datetime (default: now)
            limit: Maximum number of articles to return

        Returns:
            List of news articles or empty list on error
        """
        try:
            from alpaca.data.historical.news import NewsClient
            from alpaca.data.requests import NewsRequest

            # Normalize symbols to list
            if isinstance(symbols, str):
                symbols = [symbols]

            # P2 FIX: Validate symbols before API call
            symbols = [self._validate_symbol(s) for s in symbols]

            # Set default dates if not provided
            if end is None:
                end = datetime.now()
            if start is None:
                start = end - timedelta(hours=24)
            # Initialize news client (lazy load)
            if not hasattr(self, "_news_client") or self._news_client is None:
                _api_key = ALPACA_CREDS["API_KEY"]
                _api_secret = ALPACA_CREDS["API_SECRET"]
                self._news_client = NewsClient(
                    api_key=_api_key,
                    secret_key=_api_secret,
                )

            request = NewsRequest(
                symbols=symbols,
                start=start,
                end=end,
                limit=limit,
            )

            # Use timeout-protected async call
            news_response = await self._async_call_with_timeout(
                self._news_client.get_news,
                request,
                timeout=self.DATA_API_TIMEOUT,
                operation_name=f"get_news({symbols})"
            )

            # Convert to list of dicts for easier consumption
            articles = []
            for item in news_response.news:
                articles.append({
                    "id": str(item.id),
                    "headline": item.headline or "",
                    "summary": item.summary or "",
                    "author": item.author or "",
                    "source": item.source or "",
                    "url": item.url or "",
                    "symbols": list(item.symbols) if item.symbols else [],
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                })

            logger.debug(f"Fetched {len(articles)} news articles for {symbols}")
            return articles

        except ImportError as e:
            logger.error(
                f"News API import error: {e}. "
                "Ensure alpaca-py is installed with news support."
            )
            return []
        except ValueError as e:
            logger.error(f"Invalid symbol: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting news for {symbols}: {e}", exc_info=DEBUG_MODE)
            return []

    # =========================================================================
    # CRYPTO TRADING METHODS
    # =========================================================================

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_crypto_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: datetime = None,
        end: datetime = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        Get historical crypto bars.

        Crypto trading is available 24/7, unlike stocks which are limited to market hours.

        Args:
            symbol: Crypto pair (e.g., "BTC/USD" or "BTCUSD")
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
            start: Start datetime (defaults to 1 day ago)
            end: End datetime (defaults to now)
            limit: Maximum bars to return

        Returns:
            List of bar dicts with open, high, low, close, volume, vwap, timestamp
        """
        try:
            symbol = self.normalize_crypto_symbol(symbol)

            if start is None:
                start = datetime.now() - timedelta(days=1)
            if end is None:
                end = datetime.now()

            # Map timeframe strings to TimeFrame objects
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
                "1Day": TimeFrame.Day,
                "Day": TimeFrame.Day,
                "Hour": TimeFrame.Hour,
                "Minute": TimeFrame.Minute,
            }

            tf = tf_map.get(timeframe, TimeFrame.Minute)

            client = self._get_crypto_data_client()
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )

            bars = await self._async_call_with_timeout(
                client.get_crypto_bars,
                request,
                timeout=self.DATA_API_TIMEOUT,
                operation_name=f"get_crypto_bars({symbol})"
            )

            result = []
            if symbol in bars:
                for bar in bars[symbol]:
                    result.append({
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "vwap": float(bar.vwap) if bar.vwap else None,
                        "trade_count": int(bar.trade_count) if bar.trade_count else None,
                    })

            logger.debug(f"Fetched {len(result)} crypto bars for {symbol}")
            return result

        except ValueError as e:
            logger.error(f"Invalid crypto symbol: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching crypto bars for {symbol}: {e}", exc_info=DEBUG_MODE)
            return []

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_crypto_quote(self, symbol: str) -> Optional[dict]:
        """
        Get latest crypto quote (bid/ask).

        Args:
            symbol: Crypto pair (e.g., "BTC/USD" or "BTCUSD")

        Returns:
            Dict with bid, ask, bid_size, ask_size, timestamp or None on error
        """
        try:
            symbol = self.normalize_crypto_symbol(symbol)

            client = self._get_crypto_data_client()
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)

            quote = await self._async_call_with_timeout(
                client.get_crypto_latest_quote,
                request,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name=f"get_crypto_quote({symbol})"
            )

            if symbol in quote:
                q = quote[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(q.bid_price),
                    "ask": float(q.ask_price),
                    "bid_size": float(q.bid_size),
                    "ask_size": float(q.ask_size),
                    "timestamp": q.timestamp,
                }

            logger.warning(f"No quote data found for {symbol}")
            return None

        except ValueError as e:
            logger.error(f"Invalid crypto symbol: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching crypto quote for {symbol}: {e}", exc_info=DEBUG_MODE)
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_crypto_last_price(self, symbol: str) -> Optional[float]:
        """
        Get last trade price for crypto pair.

        Args:
            symbol: Crypto pair (e.g., "BTC/USD" or "BTCUSD")

        Returns:
            Last trade price as float, or None on error
        """
        try:
            symbol = self.normalize_crypto_symbol(symbol)

            # Check cache first
            now = datetime.now()
            cache_key = f"crypto:{symbol}"
            if cache_key in self._price_cache:
                cached_price, cached_time = self._price_cache[cache_key]
                if now - cached_time < self._price_cache_ttl:
                    logger.debug(f"Crypto price cache hit for {symbol}: ${cached_price:.2f}")
                    return cached_price

            client = self._get_crypto_data_client()
            request = CryptoLatestTradeRequest(symbol_or_symbols=symbol)

            trade = await self._async_call_with_timeout(
                client.get_crypto_latest_trade,
                request,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name=f"get_crypto_last_price({symbol})"
            )

            if symbol in trade:
                price = float(trade[symbol].price)
                # Cache the result
                self._price_cache[cache_key] = (price, now)
                return price

            logger.warning(f"No trade data found for {symbol}")
            return None

        except ValueError as e:
            logger.error(f"Invalid crypto symbol: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching crypto price for {symbol}: {e}", exc_info=DEBUG_MODE)
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def submit_crypto_order(
        self,
        symbol: str,
        side: str,
        qty: float = None,
        notional: float = None,
        order_type: str = "market",
        limit_price: float = None,
        time_in_force: str = "gtc",
    ) -> Optional[dict]:
        """
        Submit a cryptocurrency order.

        Crypto orders can be placed 24/7, unlike stock orders which are limited to market hours.
        Supports both quantity-based and notional (dollar amount) orders.

        Args:
            symbol: Crypto pair (e.g., "BTC/USD" or "BTCUSD")
            side: "buy" or "sell"
            qty: Quantity in base currency (e.g., 0.5 BTC). Mutually exclusive with notional.
            notional: Dollar amount to buy/sell (e.g., 1000 for $1000). Mutually exclusive with qty.
            order_type: "market" or "limit"
            limit_price: Price for limit orders (required if order_type is "limit")
            time_in_force: "gtc" (good-till-canceled), "ioc" (immediate-or-cancel), "fok" (fill-or-kill)

        Returns:
            Order dict with id, symbol, side, qty, notional, type, status, created_at
            or None on error

        Raises:
            ValueError: If neither qty nor notional is specified, or if both are specified
        """
        try:
            symbol = self.normalize_crypto_symbol(symbol)

            # Validate qty/notional
            if qty is None and notional is None:
                raise ValueError("Either qty or notional must be specified")
            if qty is not None and notional is not None:
                raise ValueError("Specify either qty or notional, not both")

            # Validate side
            if side.lower() not in ("buy", "sell"):
                raise ValueError(f"Side must be 'buy' or 'sell', got: {side}")
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Map time_in_force
            tif_map = {
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }
            tif = tif_map.get(time_in_force.lower(), TimeInForce.GTC)

            # Build order request
            if order_type.lower() == "market":
                if notional is not None:
                    # Notional order (dollar amount)
                    request = MarketOrderRequest(
                        symbol=symbol,
                        notional=float(notional),
                        side=order_side,
                        time_in_force=tif,
                    )
                else:
                    # Quantity order
                    request = MarketOrderRequest(
                        symbol=symbol,
                        qty=float(qty),
                        side=order_side,
                        time_in_force=tif,
                    )

            elif order_type.lower() == "limit":
                if limit_price is None:
                    raise ValueError("limit_price required for limit orders")
                if notional is not None:
                    raise ValueError("Limit orders do not support notional, use qty instead")

                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=float(qty),
                    side=order_side,
                    limit_price=float(limit_price),
                    time_in_force=tif,
                )

            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            # Submit the order with timeout protection
            order = await self._async_call_with_timeout(
                self.trading_client.submit_order,
                request,
                timeout=self.ORDER_API_TIMEOUT,
                operation_name=f"submit_crypto_order({symbol})"
            )

            result = {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": str(order.qty) if order.qty else None,
                "notional": str(order.notional) if order.notional else None,
                "type": order.type.value,
                "status": order.status.value,
                "created_at": order.created_at,
            }

            logger.info(
                f"Crypto order submitted: {result['id']} - {result['side']} "
                f"{result['qty'] or '$' + result['notional']} {symbol}"
            )
            return result

        except asyncio.TimeoutError:
            raise OrderError(f"Crypto order submission timed out for {symbol}") from None
        except ValueError as e:
            logger.error(f"Invalid crypto order parameters: {e}")
            raise
        except Exception as e:
            logger.error(f"Error submitting crypto order: {e}", exc_info=DEBUG_MODE)
            return None

    def setup_crypto_stream(self, symbols: List[str]) -> CryptoDataStream:
        """
        Setup crypto WebSocket streaming for real-time data.

        Unlike stock streams which are only available during market hours,
        crypto streams are available 24/7.

        Args:
            symbols: List of crypto pairs to stream (e.g., ["BTC/USD", "ETH/USD"])

        Returns:
            CryptoDataStream instance for subscribing to real-time data
        """
        # Normalize all symbols
        normalized_symbols = [self.normalize_crypto_symbol(s) for s in symbols]

        self._crypto_stream = CryptoDataStream(
            api_key=self._api_key,
            secret_key=self._api_secret,
        )

        logger.info(f"Setup crypto stream for symbols: {normalized_symbols}")
        return self._crypto_stream

    async def get_crypto_positions(self) -> List[dict]:
        """
        Get all crypto positions.

        Returns:
            List of crypto position dicts with symbol, qty, market_value, cost_basis, etc.
        """
        try:
            positions = await self.get_positions()

            crypto_positions = []
            for pos in positions:
                # Check if the position symbol is a crypto pair
                if self.is_crypto(pos.symbol):
                    crypto_positions.append({
                        "symbol": pos.symbol,
                        "qty": float(pos.qty),
                        "market_value": float(pos.market_value),
                        "cost_basis": float(pos.cost_basis),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc),
                        "current_price": float(pos.current_price),
                        "avg_entry_price": float(pos.avg_entry_price),
                        "side": pos.side,
                    })

            return crypto_positions

        except Exception as e:
            logger.error(f"Error getting crypto positions: {e}", exc_info=DEBUG_MODE)
            return []

    async def is_crypto_tradeable(self, symbol: str) -> bool:
        """
        Check if a crypto pair is tradeable.

        Crypto is tradeable 24/7, but we still validate the symbol is supported.

        Args:
            symbol: Crypto symbol to check

        Returns:
            True if the symbol is a supported crypto pair
        """
        try:
            self.normalize_crypto_symbol(symbol)
            return True
        except ValueError:
            return False

    # =========================================================================
    # WebSocket Real-Time Streaming Methods
    # =========================================================================

    def setup_websocket(self, feed: str = "iex") -> None:
        """
        Initialize the WebSocket manager for real-time data streaming.

        This method sets up the WebSocketManager instance that can be used
        to subscribe to real-time bars, quotes, and trades.

        Args:
            feed: Data feed type ("iex" for free tier, "sip" for paid premium)

        Example:
            broker = AlpacaBroker(paper=True)
            broker.setup_websocket(feed="iex")
            broker.register_bar_handler(my_strategy.on_bar)
            await broker.start_streaming(["AAPL", "MSFT"])
        """
        from utils.websocket_manager import WebSocketManager

        # Get credentials (use same pattern as __init__)
        _api_key = ALPACA_CREDS["API_KEY"]
        _api_secret = ALPACA_CREDS["API_SECRET"]

        self._websocket_manager = WebSocketManager(
            api_key=_api_key,
            secret_key=_api_secret,
            feed=feed
        )

        logger.info(f"WebSocket manager initialized with feed={feed}")

    async def start_streaming(
        self,
        symbols: list,
        subscribe_bars: bool = True,
        subscribe_quotes: bool = False,
        subscribe_trades: bool = False
    ) -> bool:
        """
        Start streaming real-time data for specified symbols.

        Args:
            symbols: List of stock symbols to stream
            subscribe_bars: Subscribe to real-time bars (default: True)
            subscribe_quotes: Subscribe to real-time quotes (default: False)
            subscribe_trades: Subscribe to real-time trades (default: False)

        Returns:
            True if streaming started successfully

        Raises:
            RuntimeError: If WebSocket manager not initialized

        Example:
            await broker.start_streaming(
                symbols=["AAPL", "MSFT", "GOOGL"],
                subscribe_bars=True,
                subscribe_quotes=True
            )
        """
        if not hasattr(self, '_websocket_manager') or self._websocket_manager is None:
            # Auto-initialize if not done
            self.setup_websocket()

        # Subscribe to requested data types
        if subscribe_bars:
            self._websocket_manager.subscribe_bars(symbols)

        if subscribe_quotes:
            self._websocket_manager.subscribe_quotes(symbols)

        if subscribe_trades:
            self._websocket_manager.subscribe_trades(symbols)

        # Start the connection
        await self._websocket_manager.start()

        # Wait for connection
        connected = await self._websocket_manager.wait_for_connection(timeout=10.0)

        if connected:
            logger.info(f"Streaming started for {len(symbols)} symbols")
        else:
            logger.warning("Streaming started but connection not confirmed")

        return connected

    async def stop_streaming(self) -> None:
        """
        Stop the WebSocket streaming connection.

        Gracefully shuts down the WebSocket connection and cleans up resources.
        """
        if hasattr(self, '_websocket_manager') and self._websocket_manager:
            await self._websocket_manager.stop()
            logger.info("Streaming stopped")
        else:
            logger.warning("No streaming connection to stop")

    def register_bar_handler(self, handler, symbols: Optional[list] = None) -> None:
        """
        Register a callback handler for real-time bar data.

        Args:
            handler: Async callback function with signature:
                     async def handler(bar: Bar) -> None
                     Bar has attributes: symbol, open, high, low, close, volume, timestamp
            symbols: Optional list of symbols to receive bars for.
                     If None, handler receives all bars (global handler).

        Example:
            async def on_bar(bar):
                print(f"{bar.symbol} closed at ${bar.close}")

            broker.register_bar_handler(on_bar, ["AAPL", "MSFT"])
        """
        if not hasattr(self, '_websocket_manager') or self._websocket_manager is None:
            self.setup_websocket()

        if symbols:
            self._websocket_manager.subscribe_bars(symbols, handler)
        else:
            self._websocket_manager.add_global_bar_handler(handler)

        logger.debug(f"Registered bar handler for: {symbols or 'all symbols'}")

    def register_quote_handler(self, handler, symbols: Optional[list] = None) -> None:
        """
        Register a callback handler for real-time quote data.

        Args:
            handler: Async callback function with signature:
                     async def handler(quote: Quote) -> None
                     Quote has: symbol, bid_price, ask_price, bid_size, ask_size, timestamp
            symbols: Optional list of symbols. If None, receives all quotes.

        Example:
            async def on_quote(quote):
                spread = quote.ask_price - quote.bid_price
                print(f"{quote.symbol} spread: ${spread:.4f}")

            broker.register_quote_handler(on_quote, ["AAPL"])
        """
        if not hasattr(self, '_websocket_manager') or self._websocket_manager is None:
            self.setup_websocket()

        if symbols:
            self._websocket_manager.subscribe_quotes(symbols, handler)
        else:
            self._websocket_manager.add_global_quote_handler(handler)

        logger.debug(f"Registered quote handler for: {symbols or 'all symbols'}")

    def register_trade_handler(self, handler, symbols: Optional[list] = None) -> None:
        """
        Register a callback handler for real-time trade data.

        Args:
            handler: Async callback function with signature:
                     async def handler(trade: Trade) -> None
                     Trade has: symbol, price, size, timestamp, exchange, conditions
            symbols: Optional list of symbols. If None, receives all trades.

        Example:
            async def on_trade(trade):
                print(f"{trade.symbol} traded {trade.size} @ ${trade.price}")

            broker.register_trade_handler(on_trade)
        """
        if not hasattr(self, '_websocket_manager') or self._websocket_manager is None:
            self.setup_websocket()

        if symbols:
            self._websocket_manager.subscribe_trades(symbols, handler)
        else:
            self._websocket_manager.add_global_trade_handler(handler)

        logger.debug(f"Registered trade handler for: {symbols or 'all symbols'}")

    def get_streaming_status(self) -> dict:
        """
        Get current streaming connection status.

        Returns:
            Dict with connection status, subscriptions, and statistics

        Example:
            status = broker.get_streaming_status()
            print(f"Connected: {status['is_connected']}")
            print(f"Messages received: {status['message_count']}")
        """
        if not hasattr(self, '_websocket_manager') or self._websocket_manager is None:
            return {
                "initialized": False,
                "is_running": False,
                "is_connected": False,
                "subscriptions": {},
                "message_count": 0
            }

        return {
            "initialized": True,
            **self._websocket_manager.get_connection_stats(),
            "subscriptions": self._websocket_manager.get_subscription_info()
        }

    # =========================================================================
    # Portfolio History API Methods
    # =========================================================================

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_portfolio_history(
        self,
        period: str = "1M",
        timeframe: str = "1D",
        extended_hours: bool = True,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
    ) -> Optional[dict]:
        """
        Get portfolio history from Alpaca.

        This method retrieves historical portfolio data including equity values,
        profit/loss, and timestamps for performance tracking and analysis.

        Args:
            period: Time period - "1D", "1W", "1M", "3M", "6M", "1A", "all"
                   Ignored if date_start is provided.
            timeframe: Resolution - "1Min", "5Min", "15Min", "1H", "1D"
            extended_hours: Include extended hours data in the results
            date_start: Custom start date (overrides period if provided)
            date_end: Custom end date (defaults to now)

        Returns:
            Dict with:
            - timestamp: List of Unix timestamps
            - equity: List of equity values
            - profit_loss: List of daily P&L values
            - profit_loss_pct: List of daily P&L percentages
            - base_value: Starting portfolio value
            - timeframe: Timeframe used
            Returns None on error.

        Example:
            history = await broker.get_portfolio_history(period="1M", timeframe="1D")
            if history:
                for ts, eq in zip(history["timestamp"], history["equity"]):
                    print(f"{datetime.fromtimestamp(ts)}: ${eq:,.2f}")
        """
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest

            # Build request parameters
            request_params = {
                "timeframe": timeframe,
                "extended_hours": extended_hours,
            }

            # Use date_start/date_end if provided, otherwise use period
            if date_start:
                request_params["date_start"] = date_start.strftime("%Y-%m-%d")
                if date_end:
                    request_params["date_end"] = date_end.strftime("%Y-%m-%d")
            else:
                request_params["period"] = period

            request = GetPortfolioHistoryRequest(**request_params)

            # Execute API call with timeout protection (data-heavy operation)
            history = await self._async_call_with_timeout(
                self.trading_client.get_portfolio_history,
                request,
                timeout=self.DATA_API_TIMEOUT,
                operation_name="get_portfolio_history"
            )

            # Convert to serializable dict
            # Handle potential None values in the response
            return {
                "timestamp": list(history.timestamp) if history.timestamp else [],
                "equity": list(history.equity) if history.equity else [],
                "profit_loss": list(history.profit_loss) if history.profit_loss else [],
                "profit_loss_pct": (
                    list(history.profit_loss_pct) if history.profit_loss_pct else []
                ),
                "base_value": history.base_value,
                "timeframe": str(history.timeframe) if history.timeframe else timeframe,
            }

        except ImportError as e:
            logger.error(
                f"Portfolio History API import error: {e}. "
                "Ensure alpaca-py is up to date."
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching portfolio history: {e}", exc_info=DEBUG_MODE)
            return None

    async def get_equity_curve(self, days: int = 30) -> list:
        """
        Get equity curve for the last N days.

        Convenience method that returns a list of (timestamp, equity) tuples
        for easy plotting and analysis.

        Args:
            days: Number of days of history to retrieve (default: 30)

        Returns:
            List of (timestamp, equity) tuples sorted chronologically.
            Returns empty list on error.

        Example:
            curve = await broker.get_equity_curve(days=30)
            for timestamp, equity in curve:
                date = datetime.fromtimestamp(timestamp)
                print(f"{date.strftime('%Y-%m-%d')}: ${equity:,.2f}")
        """
        # Map days to Alpaca period strings
        period_map = {
            1: "1D",
            7: "1W",
            30: "1M",
            90: "3M",
            180: "6M",
            365: "1A",
        }

        # Find the smallest period that covers the requested days
        period = "all"
        for threshold, period_str in sorted(period_map.items()):
            if days <= threshold:
                period = period_str
                break

        history = await self.get_portfolio_history(period=period, timeframe="1D")
        if not history or not history.get("equity"):
            return []

        # Combine timestamps and equity into tuples
        timestamps = history.get("timestamp", [])
        equity_values = history.get("equity", [])

        # Ensure both lists have the same length
        min_len = min(len(timestamps), len(equity_values))
        return list(zip(timestamps[:min_len], equity_values[:min_len], strict=False))

    async def get_performance_summary(self, period: str = "1M") -> Optional[dict]:
        """
        Get performance summary for a period.

        Calculates key performance metrics from portfolio history including
        total return, max drawdown, and equity range.

        Args:
            period: Time period - "1D", "1W", "1M", "3M", "6M", "1A", "all"

        Returns:
            Dict with:
            - period: The requested period
            - start_equity: Equity at start of period
            - end_equity: Equity at end of period
            - total_return: Absolute return in dollars
            - total_return_pct: Return as percentage
            - max_equity: Highest equity value in period
            - min_equity: Lowest equity value in period
            - max_drawdown: Maximum drawdown as percentage
            - data_points: Number of data points in the period
            Returns None on error or if no data available.

        Example:
            summary = await broker.get_performance_summary(period="1M")
            if summary:
                print(f"1-Month Return: {summary['total_return_pct']:.2f}%")
                print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
        """
        history = await self.get_portfolio_history(period=period)
        if not history or not history.get("equity"):
            return None

        equity = history["equity"]
        pnl = history.get("profit_loss", [])

        # Filter out None values from equity list
        equity = [e for e in equity if e is not None]
        if not equity:
            return None

        start_equity = equity[0]
        end_equity = equity[-1]

        # Calculate total return
        if pnl:
            # Filter None values
            pnl = [p for p in pnl if p is not None]
            total_return = sum(pnl) if pnl else 0
        else:
            total_return = end_equity - start_equity

        # Calculate percentage return
        if start_equity > 0:
            total_return_pct = ((end_equity / start_equity) - 1) * 100
        else:
            total_return_pct = 0.0

        return {
            "period": period,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "max_equity": max(equity),
            "min_equity": min(equity),
            "max_drawdown": self._calculate_max_drawdown(equity),
            "data_points": len(equity),
        }

    def _calculate_max_drawdown(self, equity: list) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Maximum drawdown is the largest peak-to-trough decline in portfolio
        value, expressed as a percentage.

        Args:
            equity: List of equity values (chronological order)

        Returns:
            Maximum drawdown as a percentage (e.g., 5.5 for 5.5% drawdown).
            Returns 0.0 if equity list is empty or all values are None.
        """
        if not equity:
            return 0.0

        # Filter out None values
        equity = [e for e in equity if e is not None]
        if not equity:
            return 0.0

        peak = equity[0]
        max_dd = 0.0

        for value in equity:
            if value > peak:
                peak = value
            if peak > 0:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)

        return max_dd * 100  # Return as percentage

    async def get_intraday_equity(
        self, timeframe: str = "1H"
    ) -> Optional[dict]:
        """
        Get intraday equity data for today.

        Useful for monitoring real-time portfolio performance throughout
        the trading day.

        Args:
            timeframe: Resolution - "1Min", "5Min", "15Min", "1H"

        Returns:
            Dict with timestamp, equity, profit_loss, profit_loss_pct
            for today's trading session. Returns None on error.

        Example:
            intraday = await broker.get_intraday_equity(timeframe="15Min")
            if intraday:
                print(f"Current P&L: ${intraday['profit_loss'][-1]:,.2f}")
        """
        return await self.get_portfolio_history(
            period="1D",
            timeframe=timeframe,
            extended_hours=True,
        )

    async def get_historical_performance(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        timeframe: str = "1D",
    ) -> Optional[dict]:
        """
        Get historical portfolio performance for a custom date range.

        Args:
            start_date: Start of the date range
            end_date: End of the date range (defaults to now)
            timeframe: Resolution - "1Min", "5Min", "15Min", "1H", "1D"

        Returns:
            Dict with portfolio history data for the specified range.
            Returns None on error.

        Example:
            from datetime import datetime
            start = datetime(2024, 1, 1)
            end = datetime(2024, 6, 30)
            history = await broker.get_historical_performance(start, end)
            if history:
                print(f"6-month equity data: {len(history['equity'])} points")
        """
        return await self.get_portfolio_history(
            timeframe=timeframe,
            extended_hours=True,
            date_start=start_date,
            date_end=end_date or datetime.now(),
        )
