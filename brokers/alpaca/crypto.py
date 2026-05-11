"""
AlpacaBroker crypto mixin.

Contains:
    - get_crypto_bars / get_crypto_quote / get_crypto_last_price
    - submit_crypto_order (qty or notional, market or limit)
    - setup_crypto_stream
    - get_crypto_positions / is_crypto_tradeable

Crypto trading is available 24/7 and uses separate Alpaca data clients
(loaded lazily by the account mixin via `_get_crypto_data_client`).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, List, Optional

from alpaca.data.live import CryptoDataStream
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    CryptoLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from brokers.alpaca._retry import DEBUG_MODE, OrderError, retry_with_backoff

logger = logging.getLogger(__name__)


class AlpacaCryptoMixin:
    """24/7 cryptocurrency bars/quotes/orders/positions."""

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_crypto_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
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
                operation_name=f"get_crypto_bars({symbol})",
            )

            result: List[dict] = []

            # alpaca-py returns a BarSet where `symbol in bars` can be False even when
            # the symbol exists in bars.data. Always resolve from .data first.
            bars_by_symbol: dict = {}
            if hasattr(bars, "data") and isinstance(bars.data, dict):
                bars_by_symbol = bars.data
            elif isinstance(bars, dict):
                bars_by_symbol = bars

            symbol_compact = symbol.replace("/", "").upper()
            bars_for_symbol = []
            for key, value in bars_by_symbol.items():
                if str(key).replace("/", "").upper() == symbol_compact:
                    bars_for_symbol = list(value or [])
                    break

            for bar in bars_for_symbol:
                result.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "vwap": float(bar.vwap) if bar.vwap else None,
                        "trade_count": int(bar.trade_count) if bar.trade_count else None,
                    }
                )

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
                operation_name=f"get_crypto_quote({symbol})",
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
                operation_name=f"get_crypto_last_price({symbol})",
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
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        limit_price: Optional[float] = None,
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
            request: Any
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
                    if qty is None:
                        raise ValueError("qty required for quantity-based market orders")
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
                if qty is None:
                    raise ValueError("qty required for limit orders")

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
                operation_name=f"submit_crypto_order({symbol})",
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

        # Resolve through brokers.alpaca_broker so test patches on
        # brokers.alpaca_broker.CryptoDataStream still intercept construction.
        import brokers.alpaca_broker as _broker_mod

        stream = _broker_mod.CryptoDataStream(
            api_key=self._api_key,
            secret_key=self._api_secret,
        )
        self._crypto_stream = stream

        logger.info(f"Setup crypto stream for symbols: {normalized_symbols}")
        return stream

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
                    crypto_positions.append(
                        {
                            "symbol": pos.symbol,
                            "qty": float(pos.qty),
                            "market_value": float(pos.market_value),
                            "cost_basis": float(pos.cost_basis),
                            "unrealized_pl": float(pos.unrealized_pl),
                            "unrealized_plpc": float(pos.unrealized_plpc),
                            "current_price": float(pos.current_price),
                            "avg_entry_price": float(pos.avg_entry_price),
                            "side": pos.side,
                        }
                    )

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
