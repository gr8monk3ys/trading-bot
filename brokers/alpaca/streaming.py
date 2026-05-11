"""
AlpacaBroker streaming mixin.

Contains:
    - Trade update / bar / quote / trade websocket handlers
    - Stock + crypto websocket subscription orchestration
    - start_websocket / stop_websocket lifecycle
    - WebSocketManager (utils/websocket_manager.py) handler registration
      (setup_websocket, start_streaming, stop_streaming, register_*_handler)

The trade-update handler interacts with PartialFillTracker, OrderLifecycleTracker,
PositionManager, and AuditLog (all attached to the broker instance).

Portfolio history / equity-curve analytics live in `brokers/alpaca/portfolio.py`.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from alpaca.data.live import CryptoDataStream

from utils.audit_log import AuditEventType, log_order_event
from utils.order_lifecycle import OrderState

from brokers.alpaca._retry import DEBUG_MODE

logger = logging.getLogger(__name__)


class AlpacaStreamingMixin:
    """Websocket subscriptions, streaming handlers, and portfolio history."""

    # =========================================================================
    # WEBSOCKET SHARED HELPERS
    # =========================================================================

    def _get_or_create_crypto_stream(self) -> CryptoDataStream:
        """Return a live crypto stream, creating one lazily when needed.

        Looks up `CryptoDataStream` through `brokers.alpaca_broker` so that
        test code patching `brokers.alpaca_broker.CryptoDataStream` works.
        """
        if self._crypto_stream is None:
            # Resolve the class through the alpaca_broker module namespace
            # so unittest.mock.patch("brokers.alpaca_broker.CryptoDataStream")
            # still intercepts construction here.
            import brokers.alpaca_broker as _broker_mod

            self._crypto_stream = _broker_mod.CryptoDataStream(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
        return self._crypto_stream

    def _resolve_websocket_configuration(
        self, symbols: Optional[List[str]]
    ) -> tuple[str, List[str]]:
        """
        Resolve websocket stream mode and normalized symbol list.

        Returns:
            Tuple of (asset_class, normalized_symbols) where asset_class is
            either "stock" or "crypto".

        Raises:
            ValueError: When symbols are empty or mix stock + crypto assets.
        """
        requested_symbols = symbols if symbols is not None else self._ws_symbols
        cleaned = [str(symbol).strip() for symbol in requested_symbols if str(symbol).strip()]
        if not cleaned:
            raise ValueError("At least one symbol is required to start websocket streaming")

        stock_symbols: List[str] = []
        crypto_symbols: List[str] = []

        for symbol in cleaned:
            if self.is_crypto(symbol):
                crypto_symbols.append(self.normalize_crypto_symbol(symbol))
            else:
                stock_symbols.append(self._validate_symbol(symbol))

        if stock_symbols and crypto_symbols:
            raise ValueError(
                "Mixed stock and crypto symbols are not supported in a single websocket session. "
                "Run separate sessions for each asset class."
            )

        # Keep order while removing duplicates.
        normalized = list(dict.fromkeys(crypto_symbols if crypto_symbols else stock_symbols))
        asset_class = "crypto" if crypto_symbols else "stock"
        return asset_class, normalized

    # =========================================================================
    # WEBSOCKET HANDLERS
    # =========================================================================

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
            filled_avg_price = (
                float(order.get("filled_avg_price", 0)) if order.get("filled_avg_price") else 0.0
            )
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
                logger.info(
                    f"Order {order_id} partially filled: {filled_qty} @ ${filled_avg_price:.2f}"
                )

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
            # Extract bar data (supports both dict payloads and alpaca-py model objects)
            if isinstance(data, dict):
                symbol = data.get("S") or data.get("symbol")
                open_price = float(data.get("o", data.get("open", 0)) or 0)
                high_price = float(data.get("h", data.get("high", 0)) or 0)
                low_price = float(data.get("l", data.get("low", 0)) or 0)
                close_price = float(data.get("c", data.get("close", 0)) or 0)
                volume = int(data.get("v", data.get("volume", 0)) or 0)
                ts_raw = data.get("t", data.get("timestamp"))
            else:
                symbol = getattr(data, "symbol", None)
                open_price = float(getattr(data, "open", 0) or 0)
                high_price = float(getattr(data, "high", 0) or 0)
                low_price = float(getattr(data, "low", 0) or 0)
                close_price = float(getattr(data, "close", 0) or 0)
                volume = int(getattr(data, "volume", 0) or 0)
                ts_raw = getattr(data, "timestamp", None)

            if isinstance(ts_raw, datetime):
                timestamp = ts_raw
            elif isinstance(ts_raw, (int, float)):
                ts_float = float(ts_raw)
                if ts_float > 1e14:  # nanoseconds
                    timestamp = datetime.fromtimestamp(ts_float / 1_000_000_000.0)
                elif ts_float > 1e11:  # milliseconds
                    timestamp = datetime.fromtimestamp(ts_float / 1000.0)
                else:  # seconds
                    timestamp = datetime.fromtimestamp(ts_float)
            else:
                timestamp = datetime.utcnow()

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
            # Extract quote data (supports both dict payloads and alpaca-py model objects)
            if isinstance(data, dict):
                symbol = data.get("S") or data.get("symbol")
                bid_price = float(data.get("bp", data.get("bid_price", 0)) or 0)
                ask_price = float(data.get("ap", data.get("ask_price", 0)) or 0)
                bid_size = int(data.get("bs", data.get("bid_size", 0)) or 0)
                ask_size = int(data.get("as", data.get("ask_size", 0)) or 0)
                ts_raw = data.get("t", data.get("timestamp"))
            else:
                symbol = getattr(data, "symbol", None)
                bid_price = float(getattr(data, "bid_price", 0) or 0)
                ask_price = float(getattr(data, "ask_price", 0) or 0)
                bid_size = int(getattr(data, "bid_size", 0) or 0)
                ask_size = int(getattr(data, "ask_size", 0) or 0)
                ts_raw = getattr(data, "timestamp", None)

            if isinstance(ts_raw, datetime):
                timestamp = ts_raw
            elif isinstance(ts_raw, (int, float)):
                ts_float = float(ts_raw)
                if ts_float > 1e14:
                    timestamp = datetime.fromtimestamp(ts_float / 1_000_000_000.0)
                elif ts_float > 1e11:
                    timestamp = datetime.fromtimestamp(ts_float / 1000.0)
                else:
                    timestamp = datetime.fromtimestamp(ts_float)
            else:
                timestamp = datetime.utcnow()

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
            # Extract trade data (supports both dict payloads and alpaca-py model objects)
            if isinstance(data, dict):
                symbol = data.get("S") or data.get("symbol")
                price = float(data.get("p", data.get("price", 0)) or 0)
                size = int(data.get("s", data.get("size", 0)) or 0)
                ts_raw = data.get("t", data.get("timestamp"))
            else:
                symbol = getattr(data, "symbol", None)
                price = float(getattr(data, "price", 0) or 0)
                size = int(getattr(data, "size", 0) or 0)
                ts_raw = getattr(data, "timestamp", None)

            if isinstance(ts_raw, datetime):
                timestamp = ts_raw
            elif isinstance(ts_raw, (int, float)):
                ts_float = float(ts_raw)
                if ts_float > 1e14:
                    timestamp = datetime.fromtimestamp(ts_float / 1_000_000_000.0)
                elif ts_float > 1e11:
                    timestamp = datetime.fromtimestamp(ts_float / 1000.0)
                else:
                    timestamp = datetime.fromtimestamp(ts_float)
            else:
                timestamp = datetime.utcnow()

            # Notify subscribers
            for subscriber in self._subscribers:
                if hasattr(subscriber, "on_trade"):
                    await subscriber.on_trade(symbol, price, size, timestamp)

        except Exception as e:
            logger.error(f"Error handling trade data: {e}", exc_info=DEBUG_MODE)

    # =========================================================================
    # WEBSOCKET LIFECYCLE
    # =========================================================================

    async def _websocket_handler(self):
        """Main websocket handler with proper locking for thread safety."""
        while True:
            active_stream = None
            stream_mode = "stock"
            symbols_to_subscribe: List[str] = []
            try:
                logger.info("Starting websocket connection...")

                # Reset for new connection (with lock for thread safety)
                async with self._ws_lock:
                    self._connected = False
                    self._subscribed_symbols.clear()
                    stream_mode = self._ws_asset_class
                    symbols_to_subscribe = list(self._ws_symbols)
                    if stream_mode == "crypto":
                        active_stream = self._get_or_create_crypto_stream()
                    else:
                        active_stream = self.stream
                    self._active_stream = active_stream

                # Mark as connected before subscriptions; alpaca-py allows
                # subscriptions to be registered before run().
                async with self._ws_lock:
                    self._connected = True
                    self._reconnect_attempts = 0
                    self._reconnect_delay = 1  # Reset delay

                logger.info("Websocket connected successfully (%s mode)", stream_mode)

                # Subscribe to market data for all tracked symbols
                if stream_mode == "crypto":
                    subscribed = await self._subscribe_to_crypto_symbols(symbols_to_subscribe)
                else:
                    subscribed = await self._subscribe_to_symbols(symbols_to_subscribe)
                if not subscribed:
                    raise RuntimeError(f"Failed to subscribe symbols for {stream_mode} stream")

                # alpaca-py stream run() is blocking; execute in worker thread.
                await asyncio.to_thread(active_stream.run)
                raise RuntimeError(f"{stream_mode} stream.run() exited unexpectedly")

            except asyncio.CancelledError:
                logger.info("Websocket handler cancelled")
                async with self._ws_lock:
                    self._connected = False
                try:
                    (active_stream or self._active_stream).stop()
                except Exception:
                    pass
                raise
            except Exception as e:
                logger.error(f"Websocket error: {e}", exc_info=DEBUG_MODE)
                try:
                    if active_stream is not None:
                        active_stream.stop()
                except Exception:
                    pass

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
                symbol_list = [
                    str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
                ]
                if not symbol_list:
                    logger.warning("No symbols provided for market-data subscription")
                    return False

                # Subscribe to bars (1-minute timeframe)
                self.stream.subscribe_bars(self._handle_bars, *symbol_list)

                # Subscribe to quotes
                self.stream.subscribe_quotes(self._handle_quotes, *symbol_list)

                # Subscribe to trades
                self.stream.subscribe_trades(self._handle_trades, *symbol_list)

                # Add symbols to subscribed set (still within the same lock)
                for symbol in symbol_list:
                    self._subscribed_symbols.add(symbol)

                logger.info(f"Subscribed to market data for: {', '.join(symbol_list)}")
                return True

            except Exception as e:
                logger.error(f"Error subscribing to symbols: {e}", exc_info=DEBUG_MODE)
                return False

    async def _subscribe_to_crypto_symbols(self, symbols: List[str]):
        """Subscribe to crypto market data for multiple symbols."""
        async with self._ws_lock:
            if not self._connected:
                logger.warning("Cannot subscribe to crypto symbols: websocket not connected")
                return False

            try:
                stream = self._get_or_create_crypto_stream()
                symbol_list = [
                    self.normalize_crypto_symbol(symbol)
                    for symbol in symbols
                    if str(symbol).strip()
                ]
                symbol_list = list(dict.fromkeys(symbol_list))
                if not symbol_list:
                    logger.warning("No crypto symbols provided for market-data subscription")
                    return False

                stream.subscribe_bars(self._handle_bars, *symbol_list)
                stream.subscribe_quotes(self._handle_quotes, *symbol_list)
                stream.subscribe_trades(self._handle_trades, *symbol_list)

                for symbol in symbol_list:
                    self._subscribed_symbols.add(symbol)

                logger.info(f"Subscribed to crypto market data for: {', '.join(symbol_list)}")
                return True

            except Exception as e:
                logger.error(f"Error subscribing to crypto symbols: {e}", exc_info=DEBUG_MODE)
                return False

    async def start_websocket(self, symbols: Optional[List[str]] = None):
        """Start the websocket connection with proper locking."""
        stream_mode, normalized_symbols = self._resolve_websocket_configuration(symbols)

        async with self._ws_lock:
            if self._ws_task is not None:
                logger.info("Websocket already running")
                return

            self._ws_asset_class = stream_mode
            self._ws_symbols = normalized_symbols
            if stream_mode == "crypto":
                self._active_stream = self._get_or_create_crypto_stream()
            else:
                self._active_stream = self.stream
            self._ws_task = asyncio.create_task(self._websocket_handler())
            logger.info(
                "Started websocket handler task (%s mode): %s",
                stream_mode,
                ", ".join(normalized_symbols),
            )

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
                stream_to_stop = self._active_stream
            else:
                self._ws_task = None
                logger.info("Websocket task already completed")
                return

        try:
            if stream_to_stop is not None:
                stream_to_stop.stop()
        except Exception:
            pass

        # Cancel outside the lock to avoid deadlock
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        logger.info("Stopped websocket handler task")

    # =========================================================================
    # WebSocketManager-backed real-time streaming
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
        from config import ALPACA_CREDS
        from utils.websocket_manager import WebSocketManager

        # Get credentials (use same pattern as __init__)
        _api_key = ALPACA_CREDS["API_KEY"]
        _api_secret = ALPACA_CREDS["API_SECRET"]

        self._websocket_manager = WebSocketManager(
            api_key=_api_key, secret_key=_api_secret, feed=feed
        )

        logger.info(f"WebSocket manager initialized with feed={feed}")

    async def start_streaming(
        self,
        symbols: list,
        subscribe_bars: bool = True,
        subscribe_quotes: bool = False,
        subscribe_trades: bool = False,
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
        if not hasattr(self, "_websocket_manager") or self._websocket_manager is None:
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
        if hasattr(self, "_websocket_manager") and self._websocket_manager:
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
        if not hasattr(self, "_websocket_manager") or self._websocket_manager is None:
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
        if not hasattr(self, "_websocket_manager") or self._websocket_manager is None:
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
        if not hasattr(self, "_websocket_manager") or self._websocket_manager is None:
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
        if not hasattr(self, "_websocket_manager") or self._websocket_manager is None:
            return {
                "initialized": False,
                "is_running": False,
                "is_connected": False,
                "subscriptions": {},
                "message_count": 0,
            }

        return {
            "initialized": True,
            **self._websocket_manager.get_connection_stats(),
            "subscriptions": self._websocket_manager.get_subscription_info(),
        }
