import os
import logging
import asyncio
import signal
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from alpaca.trading import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

from lumibot.brokers import Broker
from lumibot.data_sources import AlpacaData
from lumibot.entities import Position

from config import ALPACA_CREDS, SYMBOLS

# Set up logging
logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < max_retries - 1:
                        sleep_time = min(delay * (2 ** attempt), max_delay)
                        logger.info(f"Retrying in {sleep_time} seconds...")
                        await asyncio.sleep(sleep_time)
            
            logger.error(f"All {max_retries} attempts failed")
            raise last_exception
            
        return wrapper
    return decorator

class AlpacaBroker(Broker):
    """Alpaca broker implementation."""
    
    NAME = "alpaca"
    IS_BACKTESTING_BROKER = False

    def __init__(self, paper=True):
        """Initialize the AlpacaBroker."""
        try:
            paper_str = str(paper).lower()
            self.paper = paper_str == "true"
            logger.info(f"AlpacaBroker initialized with paper={self.paper} (type: {type(self.paper)})")
            
            # Initialize position tracking
            self._filled_positions = []
            self._subscribers = []
            self._ws_lock = asyncio.Lock()
            self._ws_task = None
            
            # Store API credentials as attributes
            self.api_key = ALPACA_CREDS["API_KEY"]
            self.api_secret = ALPACA_CREDS["API_SECRET"]
            
            # Initialize the trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper
            )
            
            # Initialize the data client
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            
            # Initialize the stream
            self.stream = StockDataStream(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            
            # Initialize data source for Lumibot
            config = {
                "API_KEY": self.api_key,
                "API_SECRET": self.api_secret,
                "PAPER": self.paper
            }
            self.data_source = AlpacaData(config)
            
            logger.info("Successfully initialized AlpacaData source")
            
        except Exception as e:
            logger.error(f"Error initializing AlpacaBroker: {e}", exc_info=True)
            raise

    async def initialize_stream(self):
        """Initialize and connect the websocket stream."""
        try:
            # Create coroutine handlers
            async def trade_handler(data):
                await self._handle_trade(data)
            
            async def quote_handler(data):
                await self._handle_quote(data)
            
            async def bar_handler(data):
                await self._handle_bar(data)
            
            # Subscribe to market data for each symbol individually
            logger.info(f"Subscribing to symbols: {SYMBOLS}")
            for symbol in SYMBOLS:
                self.stream.subscribe_bars(bar_handler, symbol)
                self.stream.subscribe_trades(trade_handler, symbol)
                self.stream.subscribe_quotes(quote_handler, symbol)
            
            logger.info("Starting message processing...")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing stream: {e}", exc_info=True)
            return False

    async def connect(self):
        """Connect to the websocket stream."""
        logger.info("Connecting to websocket stream...")
        try:
            # Initialize stream first
            success = await self.initialize_stream()
            if not success:
                logger.error("Failed to initialize stream")
                return None

            # Now connect to the websocket
            await self.stream._connect()
            
            # Create task to process messages
            task = asyncio.create_task(self.stream._run_forever())
            return task
            
        except Exception as e:
            logger.error(f"Error connecting to stream: {e}", exc_info=True)
            return None

    async def disconnect(self):
        """Disconnect from the websocket stream."""
        try:
            logger.info("Disconnecting from websocket stream...")
            
            # Unsubscribe from all symbols
            for symbol in SYMBOLS:
                try:
                    self.stream.unsubscribe_bars(symbol)
                    self.stream.unsubscribe_trades(symbol)
                    self.stream.unsubscribe_quotes(symbol)
                except Exception as e:
                    logger.warning(f"Error unsubscribing from {symbol}: {e}")
            
            # Close the websocket connection
            if hasattr(self.stream, '_ws') and self.stream._ws:
                await self.stream._ws.close()
                
            logger.info("Successfully disconnected from stream")
            
        except Exception as e:
            logger.error(f"Error disconnecting from stream: {e}", exc_info=True)
            raise

    async def _handle_trade(self, data):
        """Handle trade updates."""
        try:
            symbol = data.symbol
            price = float(data.price)
            size = float(data.size)
            timestamp = data.timestamp
            
            # Notify subscribers
            for subscriber in self._subscribers:
                if hasattr(subscriber, 'on_trade'):
                    await subscriber.on_trade(symbol, price, size, timestamp)
                    
        except Exception as e:
            logger.error(f"Error handling trade: {e}", exc_info=True)

    async def _handle_quote(self, data):
        """Handle quote updates."""
        try:
            symbol = data.symbol
            bid_price = float(data.bid_price)
            bid_size = float(data.bid_size)
            ask_price = float(data.ask_price)
            ask_size = float(data.ask_size)
            timestamp = data.timestamp
            
            # Notify subscribers
            for subscriber in self._subscribers:
                if hasattr(subscriber, 'on_quote'):
                    await subscriber.on_quote(symbol, bid_price, bid_size, ask_price, ask_size, timestamp)
                    
        except Exception as e:
            logger.error(f"Error handling quote: {e}", exc_info=True)

    async def _handle_bar(self, data):
        """Handle bar updates."""
        try:
            symbol = data.symbol
            open_price = float(data.open)
            high_price = float(data.high)
            low_price = float(data.low)
            close_price = float(data.close)
            volume = float(data.volume)
            timestamp = data.timestamp
            
            # Notify subscribers
            for subscriber in self._subscribers:
                if hasattr(subscriber, 'on_bar'):
                    await subscriber.on_bar(symbol, open_price, high_price, low_price, close_price, volume, timestamp)
                    
        except Exception as e:
            logger.error(f"Error handling bar: {e}", exc_info=True)

    async def get_latest_volume(self, symbol: str) -> float:
        """Get the latest daily trading volume for a symbol."""
        logger.debug(f"Fetching latest volume for {symbol}")
        try:
            # Get the latest bar data
            from datetime import datetime, timedelta
            
            end = datetime.now()
            start = end - timedelta(days=1)
            
            bars = self.data_client.get_stock_bars(
                symbol=symbol,
                timeframe=TimeFrame.Day,
                start=start.isoformat(),
                end=end.isoformat()
            )
            
            if not bars or not bars.data:
                logger.warning(f"No volume data available for {symbol}")
                return 0
            
            # Return the latest volume
            latest_bar = bars.data[-1]
            logger.debug(f"Latest volume for {symbol}: {latest_bar.volume}")
            return latest_bar.volume
            
        except Exception as e:
            logger.error(f"Error fetching volume for {symbol}: {e}", exc_info=True)
            return 0

    async def subscribe_to_market_data(self, symbols: List[str]):
        """Subscribe to market data for specified symbols."""
        logger.info(f"Subscribing to market data for {len(symbols)} symbols...")
        try:
            async with self._ws_lock:
                new_symbols = set(symbols) - self._subscribed_symbols
                if new_symbols:
                    logger.debug(f"New symbols to subscribe: {new_symbols}")
                    await self.stream.subscribe_bars(new_symbols)
                    await self.stream.subscribe_trades(new_symbols)
                    await self.stream.subscribe_quotes(new_symbols)
                    self._subscribed_symbols.update(new_symbols)
                    logger.info(f"Successfully subscribed to {len(new_symbols)} new symbols")
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}", exc_info=True)
            raise
    
    async def unsubscribe_from_market_data(self, symbols: List[str]):
        """Unsubscribe from market data for specified symbols."""
        logger.info(f"Unsubscribing from market data for {len(symbols)} symbols...")
        try:
            async with self._ws_lock:
                symbols_to_remove = set(symbols) & self._subscribed_symbols
                if symbols_to_remove:
                    logger.debug(f"Symbols to unsubscribe: {symbols_to_remove}")
                    await self.stream.unsubscribe_bars(symbols_to_remove)
                    await self.stream.unsubscribe_trades(symbols_to_remove)
                    await self.stream.unsubscribe_quotes(symbols_to_remove)
                    self._subscribed_symbols -= symbols_to_remove
                    logger.info(f"Successfully unsubscribed from {len(symbols_to_remove)} symbols")
        except Exception as e:
            logger.error(f"Error unsubscribing from market data: {e}", exc_info=True)
            raise
    
    async def close(self):
        """Close the broker connection."""
        logger.info("Closing the broker connection...")
        try:
            if self._ws_task and not self._ws_task.done():
                self._ws_task.cancel()
                try:
                    await self._ws_task
                except asyncio.CancelledError:
                    pass
            
            if self.stream and self.stream.is_running():
                await self.stream.close()
            
            logger.info("Successfully closed Alpaca broker connection")
        except Exception as e:
            logger.error(f"Error closing Alpaca broker connection: {e}", exc_info=True)
            raise
    
    async def get_account(self):
        """Get account information."""
        try:
            # get_account() is synchronous
            return self.trading_client.get_account()
        except Exception as e:
            logger.error(f"Failed to get account: {e}", exc_info=True)
            raise

    async def get_position(self, symbol: str):
        """Get position information for a symbol."""
        logger.debug(f"Fetching position for {symbol}...")
        try:
            return self.trading_client.get_position(symbol)
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}", exc_info=True)
            raise
    
    async def get_positions(self):
        """Get all positions."""
        logger.debug("Fetching all positions...")
        try:
            # get_all_positions() is synchronous
            positions = self.trading_client.get_all_positions()
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}", exc_info=True)
            raise
    
    async def submit_order(self, order):
        """Submit a trading order."""
        logger.debug(f"Submitting order: {order}...")
        try:
            # Convert order side to Alpaca enum
            side = OrderSide.BUY if order.side.lower() == 'buy' else OrderSide.SELL
            
            # Base order parameters
            base_params = {
                "symbol": order.symbol,
                "qty": float(order.quantity),
                "side": side,
                "time_in_force": TimeInForce.DAY,
            }

            # Create appropriate order request based on type
            if order.type == "market":
                order_request = MarketOrderRequest(**base_params)
            else:
                base_params["limit_price"] = float(order.limit_price)
                order_request = LimitOrderRequest(**base_params)

            logger.debug(f"Submitting order request: {order_request}")
            # submit_order() is synchronous
            return self.trading_client.submit_order(order_request)
        except Exception as e:
            logger.error(f"Failed to submit order: {e}", exc_info=True)
            raise
    
    async def cancel_order(self, order_id: str):
        """Cancel a trading order."""
        logger.debug(f"Cancelling order {order_id}...")
        try:
            return self.trading_client.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
            raise

    async def _pull_positions(self, strategy=None):
        """Pull positions from broker."""
        try:
            # Get positions synchronously since Alpaca SDK doesn't support async
            positions = self.trading_client.get_all_positions()
            
            # Convert to our Position objects
            self._filled_positions = [
                Position(
                    symbol=position.symbol,
                    quantity=float(position.qty),
                    entry_price=float(position.avg_entry_price),
                    current_price=float(position.current_price),
                    strategy=strategy
                )
                for position in positions
            ]
            return self._filled_positions
        except Exception as e:
            logger.error(f"Error pulling positions: {e}", exc_info=True)
            return []

    async def get_tracked_positions(self, strategy=None):
        """Get tracked positions, optionally filtered by strategy."""
        try:
            # Since _pull_positions is synchronous under the hood, we don't need to await it
            positions = self._pull_positions(strategy)
            return [pos for pos in positions if strategy is None or pos.strategy == strategy]
        except Exception as e:
            logger.error(f"Error getting tracked positions: {e}", exc_info=True)
            return []

    async def get_last_price(self, symbol):
        """Get the last price for a symbol."""
        try:
            # Get latest trade using Alpaca SDK
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self.data_client.get_stock_latest_trade(request)
            
            if trade and symbol in trade:
                return float(trade[symbol].price)
            
            # Fallback to getting latest bar if trade not available
            bars_request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=1
            )
            bars = self.data_client.get_stock_bars(bars_request)
            
            if bars and symbol in bars:
                return float(bars[symbol][0].close)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting last price for {symbol}: {e}", exc_info=True)
            return None

    def get_datetime(self):
        """Get current datetime."""
        return datetime.now()

    def get_timestamp(self):
        """Get current timestamp."""
        return int(datetime.now().timestamp())

    def get_time_to_close(self):
        """Get time remaining until market close."""
        clock = self.trading_client.get_clock()
        return (clock.next_close - clock.timestamp).total_seconds()

    def is_market_open(self):
        """Check if market is open."""
        return self.trading_client.get_clock().is_open

    def get_tradable_assets(self):
        """Get list of tradable assets."""
        try:
            assets = self.trading_client.get_all_assets()
            return [asset.symbol for asset in assets if asset.tradable]
        except Exception as e:
            logger.error(f"Error getting tradable assets: {e}", exc_info=True)
            return []

    @property
    def is_backtesting(self):
        return False

    @property
    def orders(self):
        return []  # Implement order tracking if needed

    @property
    def positions(self):
        return self._filled_positions

    @property
    def portfolio_value(self):
        """Get the current portfolio value."""
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0

    @property
    def cash(self):
        """Get the current cash balance."""
        try:
            account = self.trading_client.get_account()
            return float(account.cash)
        except Exception as e:
            logger.error(f"Error getting cash balance: {e}")
            return 0.0

    def _get_stream_object(self):
        """Get the stream object for real-time data."""
        return self.stream

    async def _register_stream_events(self):
        """Register stream events for real-time data."""
        try:
            # Events are registered during stream initialization
            pass
        except Exception as e:
            logger.error(f"Error registering stream events: {e}", exc_info=True)
            raise

    async def _run_stream(self):
        """Run the data stream."""
        try:
            if self._ws_task is None:
                async with self._ws_lock:
                    self._ws_task = asyncio.create_task(self.stream._run_forever())
                    await asyncio.sleep(0)  # Let other tasks run
        except Exception as e:
            logger.error(f"Error running stream: {e}", exc_info=True)
            raise

    def _add_subscriber(self, subscriber):
        """Add a subscriber to the broker."""
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)
            logger.info(f"Added subscriber: {subscriber}")

    def _get_balances_at_broker(self, quote_asset=None, strategy=None):
        """Get account balances from broker."""
        try:
            account = self.trading_client.get_account()
            cash = float(account.cash)
            positions_value = float(account.long_market_value) + float(account.short_market_value)
            total_value = float(account.equity)
            return cash, positions_value, total_value
        except Exception as e:
            logger.error(f"Error getting balances: {e}", exc_info=True)
            return 0.0, 0.0, 0.0

    def _pull_positions(self, strategy=None):
        """Pull positions from broker."""
        try:
            positions = []
            raw_positions = self.trading_client.get_all_positions()
            
            for pos in raw_positions:
                position = {
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "avg_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "asset_id": pos.asset_id,
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "unrealized_intraday_pl": float(pos.unrealized_intraday_pl),
                    "unrealized_intraday_plpc": float(pos.unrealized_intraday_plpc),
                    "side": "long" if float(pos.qty) > 0 else "short"
                }
                positions.append(position)
            
            return positions
        except Exception as e:
            logger.error(f"Error pulling positions: {e}", exc_info=True)
            return []

    def _pull_position(self, symbol, strategy=None):
        """Pull single position from broker."""
        try:
            position = self.trading_client.get_position(symbol)
            return {
                "symbol": position.symbol,
                "quantity": float(position.qty),
                "avg_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "asset_id": position.asset_id,
                "market_value": float(position.market_value),
                "cost_basis": float(position.cost_basis),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "unrealized_intraday_pl": float(position.unrealized_intraday_pl),
                "unrealized_intraday_plpc": float(position.unrealized_intraday_plpc),
                "side": "long" if float(position.qty) > 0 else "short"
            }
        except Exception as e:
            if "position does not exist" not in str(e).lower():
                logger.error(f"Error pulling position for {symbol}: {e}", exc_info=True)
            return None

    def _submit_order(self, order):
        """Submit order to broker."""
        try:
            # Convert order side to Alpaca enum
            side = OrderSide.BUY if order["side"].lower() == 'buy' else OrderSide.SELL
            
            # Base order parameters
            base_params = {
                "symbol": order["symbol"],
                "qty": float(order["quantity"]),
                "side": side,
                "time_in_force": TimeInForce.DAY,
            }

            # Create appropriate order request based on type
            if order["type"] == "market":
                order_request = MarketOrderRequest(**base_params)
            else:
                base_params["limit_price"] = float(order["limit_price"])
                order_request = LimitOrderRequest(**base_params)

            logger.debug(f"Submitting order request: {order_request}")
            return self.trading_client.submit_order(order_request)
        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=True)
            raise

    def _parse_broker_order(self, order):
        """Parse broker order object into standardized format."""
        try:
            return {
                "id": order.id,
                "symbol": order.symbol,
                "quantity": float(order.qty),
                "side": order.side,
                "type": order.type,
                "status": order.status,
                "filled_quantity": float(order.filled_qty) if order.filled_qty else 0,
                "filled_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "created_at": order.created_at,
                "updated_at": order.updated_at
            }
        except Exception as e:
            logger.error(f"Error parsing broker order: {e}", exc_info=True)
            raise

    def _pull_broker_all_orders(self, symbols=None, status=None):
        """Pull all orders from broker."""
        try:
            orders = self.trading_client.get_orders(status=status, symbols=symbols)
            parsed_orders = []
            for order in orders:
                parsed_order = self._parse_broker_order(order)
                parsed_orders.append(parsed_order)
            return parsed_orders
        except Exception as e:
            logger.error(f"Error pulling all orders: {e}", exc_info=True)
            return []

    def _pull_broker_order(self, order_id):
        """Pull specific order from broker."""
        try:
            order = self.trading_client.get_order(order_id)
            return self._parse_broker_order(order)
        except Exception as e:
            logger.error(f"Error pulling order {order_id}: {e}", exc_info=True)
            return None

    async def get_historical_account_value(self, timeframe="1D", start_date=None, end_date=None):
        """Get historical account value."""
        try:
            portfolio_history = self.trading_client.get_portfolio_history(
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            return {
                "timestamps": portfolio_history.timestamp,
                "equity": portfolio_history.equity,
                "profit_loss": portfolio_history.profit_loss,
                "profit_loss_pct": portfolio_history.profit_loss_pct
            }
        except Exception as e:
            logger.error(f"Error getting historical account value: {e}", exc_info=True)
            return {
                "timestamps": [],
                "equity": [],
                "profit_loss": [],
                "profit_loss_pct": []
            }
