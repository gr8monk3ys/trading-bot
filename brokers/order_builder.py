#!/usr/bin/env python3
"""
Advanced Order Builder for Alpaca Trading API

This module provides a fluent interface for creating complex orders including
bracket orders, OCO (One-Cancels-Other), OTO (One-Triggers-Other), and trailing stops.
"""

import logging
from typing import Optional, Dict, Any, Literal
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType

logger = logging.getLogger(__name__)


class OrderBuilder:
    """
    Fluent interface for building Alpaca orders with all supported features.

    Examples:
        # Simple market order
        order = OrderBuilder('AAPL', 'buy', 100).market().day().build()

        # Limit order with GTC
        order = OrderBuilder('TSLA', 'sell', 50).limit(250.00).gtc().build()

        # Trailing stop order
        order = OrderBuilder('SPY', 'sell', 100).trailing_stop(trail_percent=2.5).gtc().build()

        # Bracket order (entry + take-profit + stop-loss)
        order = (OrderBuilder('NVDA', 'buy', 10)
                 .market()
                 .bracket(take_profit=120.00, stop_loss=95.00, stop_limit=94.50)
                 .build())
    """

    # P1 FIX: Maximum allowed quantity to prevent accidental large orders
    MAX_QUANTITY = 1_000_000

    def __init__(self, symbol: str, side: Literal['buy', 'sell'], qty: float):
        """
        Initialize order builder.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            side: 'buy' or 'sell'
            qty: Quantity of shares (can be fractional)

        Raises:
            ValueError: If symbol, side, or quantity is invalid
        """
        # P1 FIX: Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        symbol = symbol.upper().strip()
        if not symbol.isalpha() or len(symbol) > 5:
            raise ValueError(f"Invalid symbol format: {symbol}. Must be 1-5 letters.")
        self.symbol = symbol

        # P1 FIX: Validate side
        if not side or side.lower() not in ('buy', 'sell'):
            raise ValueError(f"Side must be 'buy' or 'sell', got: {side}")
        self.side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        # P1 FIX: Validate quantity
        try:
            self.qty = float(qty)
        except (TypeError, ValueError):
            raise ValueError(f"Quantity must be numeric, got: {qty}")

        if self.qty <= 0:
            raise ValueError(f"Quantity must be positive, got: {self.qty}")
        if self.qty > self.MAX_QUANTITY:
            raise ValueError(f"Quantity {self.qty} exceeds maximum allowed ({self.MAX_QUANTITY})")

        # Order parameters
        self._order_type: Optional[OrderType] = None
        self._time_in_force: TimeInForce = TimeInForce.DAY  # Default
        self._order_class: OrderClass = OrderClass.SIMPLE  # Default

        # Price parameters
        self._limit_price: Optional[float] = None
        self._stop_price: Optional[float] = None
        self._trail_price: Optional[float] = None
        self._trail_percent: Optional[float] = None

        # Bracket/OCO/OTO parameters
        self._take_profit: Optional[Dict[str, float]] = None
        self._stop_loss: Optional[Dict[str, float]] = None

        # Extended hours
        self._extended_hours: bool = False

        # Client order ID (for tracking)
        self._client_order_id: Optional[str] = None

    # =========================================================================
    # ORDER TYPE METHODS
    # =========================================================================

    def market(self) -> 'OrderBuilder':
        """Create a market order (executes at current price)."""
        self._order_type = OrderType.MARKET
        return self

    def limit(self, limit_price: float) -> 'OrderBuilder':
        """
        Create a limit order.

        Args:
            limit_price: Maximum price for buy, minimum for sell
        """
        self._order_type = OrderType.LIMIT
        self._limit_price = self._validate_price(limit_price)
        return self

    def stop(self, stop_price: float) -> 'OrderBuilder':
        """
        Create a stop order (converts to market when stop price hit).

        Args:
            stop_price: Trigger price
        """
        self._order_type = OrderType.STOP
        self._stop_price = self._validate_price(stop_price)
        return self

    def stop_limit(self, stop_price: float, limit_price: float) -> 'OrderBuilder':
        """
        Create a stop-limit order.

        Args:
            stop_price: Trigger price
            limit_price: Limit price after trigger
        """
        self._order_type = OrderType.STOP_LIMIT
        self._stop_price = self._validate_price(stop_price)
        self._limit_price = self._validate_price(limit_price)
        return self

    def trailing_stop(
        self,
        trail_price: Optional[float] = None,
        trail_percent: Optional[float] = None
    ) -> 'OrderBuilder':
        """
        Create a trailing stop order.

        Args:
            trail_price: Dollar amount to trail (e.g., 5.00)
            trail_percent: Percentage to trail (e.g., 2.5 for 2.5%)

        Note: Provide either trail_price OR trail_percent, not both.
        """
        if trail_price and trail_percent:
            raise ValueError("Provide either trail_price OR trail_percent, not both")
        if not trail_price and not trail_percent:
            raise ValueError("Must provide either trail_price or trail_percent")

        self._order_type = OrderType.TRAILING_STOP
        self._trail_price = float(trail_price) if trail_price else None
        self._trail_percent = float(trail_percent) if trail_percent else None
        return self

    # =========================================================================
    # TIME IN FORCE METHODS
    # =========================================================================

    def day(self) -> 'OrderBuilder':
        """Valid only during trading day (cancels at market close)."""
        self._time_in_force = TimeInForce.DAY
        return self

    def gtc(self) -> 'OrderBuilder':
        """Good-Till-Canceled (expires after 90 days)."""
        self._time_in_force = TimeInForce.GTC
        return self

    def ioc(self) -> 'OrderBuilder':
        """Immediate-Or-Cancel (fills immediately or cancels)."""
        self._time_in_force = TimeInForce.IOC
        return self

    def fok(self) -> 'OrderBuilder':
        """Fill-Or-Kill (entire order fills or cancels)."""
        self._time_in_force = TimeInForce.FOK
        return self

    def opg(self) -> 'OrderBuilder':
        """Market/Limit on Open (executes in opening auction)."""
        self._time_in_force = TimeInForce.OPG
        return self

    def cls(self) -> 'OrderBuilder':
        """Market/Limit on Close (executes in closing auction)."""
        self._time_in_force = TimeInForce.CLS
        return self

    # =========================================================================
    # ADVANCED ORDER CLASSES
    # =========================================================================

    def bracket(
        self,
        take_profit: float,
        stop_loss: float,
        stop_limit: Optional[float] = None
    ) -> 'OrderBuilder':
        """
        Create a bracket order (entry + take-profit + stop-loss).

        Args:
            take_profit: Limit price for profit target
            stop_loss: Stop price for loss protection
            stop_limit: Optional limit price for stop-loss (makes it stop-limit)

        Note: Entry order can be market or limit type.
        """
        self._order_class = OrderClass.BRACKET

        self._take_profit = {
            'limit_price': self._validate_price(take_profit)
        }

        self._stop_loss = {
            'stop_price': self._validate_price(stop_loss)
        }

        if stop_limit:
            self._stop_loss['limit_price'] = self._validate_price(stop_limit)

        # Bracket orders only support day or gtc
        if self._time_in_force not in [TimeInForce.DAY, TimeInForce.GTC]:
            logger.warning(f"Bracket orders only support DAY or GTC, changing from {self._time_in_force}")
            self._time_in_force = TimeInForce.GTC

        return self

    def oco(
        self,
        take_profit: float,
        stop_loss: float,
        stop_limit: Optional[float] = None
    ) -> 'OrderBuilder':
        """
        Create an OCO (One-Cancels-Other) order - for exiting existing positions.

        Args:
            take_profit: Limit price for profit target
            stop_loss: Stop price for loss protection
            stop_limit: Optional limit price for stop-loss

        Note: Both orders must have same side (used to exit positions).
        """
        self._order_class = OrderClass.OCO

        # OCO requires limit order type
        if self._order_type != OrderType.LIMIT:
            logger.warning("OCO orders require LIMIT type, setting order type to LIMIT")
            self._order_type = OrderType.LIMIT

        self._take_profit = {
            'limit_price': self._validate_price(take_profit)
        }

        self._stop_loss = {
            'stop_price': self._validate_price(stop_loss)
        }

        if stop_limit:
            self._stop_loss['limit_price'] = self._validate_price(stop_limit)

        return self

    def oto(
        self,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        stop_limit: Optional[float] = None
    ) -> 'OrderBuilder':
        """
        Create an OTO (One-Triggers-Other) order - entry with one exit.

        Args:
            take_profit: Optional profit target limit price
            stop_loss: Optional stop-loss price
            stop_limit: Optional stop-limit price

        Note: Provide either take_profit OR stop_loss, not both.
        """
        if take_profit and stop_loss:
            raise ValueError("OTO orders accept either take_profit OR stop_loss, not both")
        if not take_profit and not stop_loss:
            raise ValueError("OTO requires either take_profit or stop_loss")

        self._order_class = OrderClass.OTO

        if take_profit:
            self._take_profit = {
                'limit_price': self._validate_price(take_profit)
            }

        if stop_loss:
            self._stop_loss = {
                'stop_price': self._validate_price(stop_loss)
            }
            if stop_limit:
                self._stop_loss['limit_price'] = self._validate_price(stop_limit)

        return self

    # =========================================================================
    # ADDITIONAL OPTIONS
    # =========================================================================

    def extended_hours(self, enabled: bool = True) -> 'OrderBuilder':
        """
        Enable extended hours trading (pre-market and after-hours).

        Requirements:
            - Order type must be LIMIT
            - Time in force must be DAY
        """
        self._extended_hours = enabled

        if enabled:
            if self._order_type != OrderType.LIMIT:
                logger.warning("Extended hours requires LIMIT order type")
            if self._time_in_force != TimeInForce.DAY:
                logger.warning("Extended hours requires DAY time in force")

        return self

    def client_order_id(self, order_id: str) -> 'OrderBuilder':
        """
        Set a client order ID for tracking.

        Args:
            order_id: Unique identifier (up to 48 characters)
        """
        if len(order_id) > 48:
            raise ValueError("Client order ID must be 48 characters or less")
        self._client_order_id = order_id
        return self

    # =========================================================================
    # BUILD METHOD
    # =========================================================================

    def build(self) -> Any:
        """
        Build and return the appropriate Alpaca order request object.

        Returns:
            Alpaca order request (MarketOrderRequest, LimitOrderRequest, etc.)
        """
        if not self._order_type:
            raise ValueError("Order type not set. Call .market(), .limit(), etc.")

        # Validate order configuration
        self._validate_order()

        # Build base kwargs common to all order types
        base_kwargs = {
            'symbol': self.symbol,
            'qty': self.qty,
            'side': self.side,
            'time_in_force': self._time_in_force,
        }

        # Add optional fields
        if self._client_order_id:
            base_kwargs['client_order_id'] = self._client_order_id
        if self._extended_hours:
            base_kwargs['extended_hours'] = True

        # Add order class and legs for bracket/OCO/OTO
        if self._order_class != OrderClass.SIMPLE:
            base_kwargs['order_class'] = self._order_class

            if self._take_profit:
                base_kwargs['take_profit'] = self._take_profit
            if self._stop_loss:
                base_kwargs['stop_loss'] = self._stop_loss

        # Build specific order type
        if self._order_type == OrderType.MARKET:
            return MarketOrderRequest(**base_kwargs)

        elif self._order_type == OrderType.LIMIT:
            return LimitOrderRequest(
                limit_price=self._limit_price,
                **base_kwargs
            )

        elif self._order_type == OrderType.STOP:
            return StopOrderRequest(
                stop_price=self._stop_price,
                **base_kwargs
            )

        elif self._order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                stop_price=self._stop_price,
                limit_price=self._limit_price,
                **base_kwargs
            )

        elif self._order_type == OrderType.TRAILING_STOP:
            trailing_kwargs = base_kwargs.copy()

            if self._trail_price:
                trailing_kwargs['trail_price'] = self._trail_price
            else:
                trailing_kwargs['trail_percent'] = self._trail_percent

            # Trailing stops only support DAY or GTC
            if self._time_in_force not in [TimeInForce.DAY, TimeInForce.GTC]:
                logger.warning("Trailing stop only supports DAY/GTC, using GTC")
                trailing_kwargs['time_in_force'] = TimeInForce.GTC

            return TrailingStopOrderRequest(**trailing_kwargs)

        else:
            raise ValueError(f"Unsupported order type: {self._order_type}")

    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================

    def _validate_price(self, price: float) -> float:
        """Validate and format price according to Alpaca rules."""
        price = float(price)

        if price <= 0:
            raise ValueError(f"Price must be positive: {price}")

        # Alpaca price rules:
        # >= $1.00: max 2 decimal places
        # < $1.00: max 4 decimal places
        if price >= 1.0:
            # Round to 2 decimals
            return round(price, 2)
        else:
            # Round to 4 decimals
            return round(price, 4)

    def _validate_order(self):
        """Validate order configuration before building."""
        # Extended hours validation
        if self._extended_hours:
            if self._order_type != OrderType.LIMIT:
                raise ValueError("Extended hours requires LIMIT order type")
            if self._time_in_force != TimeInForce.DAY:
                raise ValueError("Extended hours requires DAY time in force")

        # Bracket order validation
        if self._order_class == OrderClass.BRACKET:
            if self._time_in_force not in [TimeInForce.DAY, TimeInForce.GTC]:
                raise ValueError("Bracket orders only support DAY or GTC")

        # OCO validation
        if self._order_class == OrderClass.OCO:
            if self._order_type != OrderType.LIMIT:
                raise ValueError("OCO orders must be LIMIT type")

        # Trailing stop validation
        if self._order_type == OrderType.TRAILING_STOP:
            if self._time_in_force not in [TimeInForce.DAY, TimeInForce.GTC]:
                raise ValueError("Trailing stops only support DAY or GTC")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"OrderBuilder(symbol={self.symbol}, side={self.side}, qty={self.qty}, "
                f"type={self._order_type}, tif={self._time_in_force}, "
                f"class={self._order_class})")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def market_order(symbol: str, side: str, qty: float, **kwargs) -> Any:
    """Quick market order creation."""
    builder = OrderBuilder(symbol, side, qty).market()

    if 'gtc' in kwargs and kwargs['gtc']:
        builder.gtc()

    return builder.build()


def limit_order(symbol: str, side: str, qty: float, limit_price: float, **kwargs) -> Any:
    """Quick limit order creation."""
    builder = OrderBuilder(symbol, side, qty).limit(limit_price)

    if 'gtc' in kwargs and kwargs['gtc']:
        builder.gtc()
    if 'extended_hours' in kwargs and kwargs['extended_hours']:
        builder.extended_hours().day()

    return builder.build()


def bracket_order(
    symbol: str,
    side: str,
    qty: float,
    entry_price: Optional[float] = None,
    take_profit: float = None,
    stop_loss: float = None,
    stop_limit: Optional[float] = None
) -> Any:
    """Quick bracket order creation."""
    builder = OrderBuilder(symbol, side, qty)

    if entry_price:
        builder.limit(entry_price)
    else:
        builder.market()

    builder.bracket(
        take_profit=take_profit,
        stop_loss=stop_loss,
        stop_limit=stop_limit
    ).gtc()

    return builder.build()
