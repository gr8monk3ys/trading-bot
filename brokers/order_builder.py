#!/usr/bin/env python3
"""
Advanced Order Builder for Alpaca Trading API

This module provides a fluent interface for creating complex orders including
bracket orders, OCO (One-Cancels-Other), OTO (One-Triggers-Other), and trailing stops.
"""

import logging
import re
from typing import Any, Dict, Literal, Optional

from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)

from utils.crypto_utils import is_crypto_symbol, normalize_crypto_symbol

logger = logging.getLogger(__name__)


class OrderBuilder:
    """
    Fluent interface for building Alpaca orders with all supported features.

    Supports both stocks and cryptocurrencies. Crypto symbols are automatically
    detected and use appropriate defaults (GTC time-in-force for 24/7 trading).

    Examples:
        # Simple market order (stock)
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

        # Notional (dollar-based) order
        order = (OrderBuilder('AAPL', 'buy')
                 .notional(1500.00)  # Buy $1500 worth of AAPL
                 .market()
                 .day()
                 .build())

        # Crypto market order (auto-detects crypto, defaults to GTC)
        order = OrderBuilder('BTC/USD', 'buy', 0.5).market().build()

        # Crypto notional order (buy $1000 worth of ETH)
        order = OrderBuilder('ETHUSD', 'buy').notional(1000.00).market().build()
    """

    # P1 FIX: Maximum allowed quantity to prevent accidental large orders
    MAX_QUANTITY = 1_000_000

    # Maximum notional order amount
    MAX_NOTIONAL = 1_000_000

    # Minimum notional order amount (Alpaca minimum)
    MIN_NOTIONAL = 1.0

    # CRYPTO_PAIRS is now imported from utils.crypto_utils for consistency

    def __init__(
        self, symbol: str, side: Literal["buy", "sell"], qty: Optional[float] = None
    ):
        """
        Initialize order builder.

        Args:
            symbol: Stock or crypto symbol (e.g., 'AAPL', 'BTC/USD', 'BTCUSD')
            side: 'buy' or 'sell'
            qty: Quantity of shares/coins (can be fractional). Optional if using notional().

        Raises:
            ValueError: If symbol or side is invalid, or if qty is invalid when provided
        """
        # P1 FIX: Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        symbol = symbol.upper().strip()

        # Check if this is a crypto symbol using centralized utility
        self._is_crypto = is_crypto_symbol(symbol)

        if self._is_crypto:
            # Normalize crypto symbol to Alpaca format (e.g., BTCUSD -> BTC/USD)
            self.symbol = normalize_crypto_symbol(symbol)
        else:
            # Validate stock symbol format
            # Allow 1-10 chars: letters, numbers, dots, hyphens (valid for ETFs like BRK.B, SPY1)
            if not re.match(r'^[A-Z0-9.\-]{1,10}$', symbol):
                raise ValueError(f"Invalid symbol format: {symbol}. Must be 1-10 alphanumeric characters, dots, or hyphens.")
            self.symbol = symbol

        # P1 FIX: Validate side
        if not side or side.lower() not in ("buy", "sell"):
            raise ValueError(f"Side must be 'buy' or 'sell', got: {side}")
        self.side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # P1 FIX: Validate quantity (now optional for notional orders)
        self.qty: Optional[float] = None
        if qty is not None:
            try:
                self.qty = float(qty)
            except (TypeError, ValueError):
                raise ValueError(f"Quantity must be numeric, got: {qty}") from None

            if self.qty <= 0:
                raise ValueError(f"Quantity must be positive, got: {self.qty}")
            if self.qty > self.MAX_QUANTITY:
                raise ValueError(f"Quantity {self.qty} exceeds maximum allowed ({self.MAX_QUANTITY})")

        # Notional (dollar amount) parameter - mutually exclusive with qty
        self._notional: Optional[float] = None

        # Order parameters
        self._order_type: Optional[OrderType] = None
        # Crypto defaults to GTC (24/7 trading), stocks default to DAY
        self._time_in_force: TimeInForce = TimeInForce.GTC if self._is_crypto else TimeInForce.DAY
        self._order_class: OrderClass = OrderClass.SIMPLE  # Default

        # Price parameters
        self._limit_price: Optional[float] = None
        self._stop_price: Optional[float] = None
        self._trail_price: Optional[float] = None
        self._trail_percent: Optional[float] = None

        # Bracket/OCO/OTO parameters
        self._take_profit: Optional[Dict[str, float]] = None
        self._stop_loss: Optional[Dict[str, float]] = None

        # Extended hours (not applicable to crypto - always 24/7)
        self._extended_hours: bool = False

        # Client order ID (for tracking)
        self._client_order_id: Optional[str] = None

    # _detect_crypto and _normalize_crypto_symbol methods have been removed.
    # Use utils.crypto_utils.is_crypto_symbol and normalize_crypto_symbol instead.

    @property
    def is_crypto(self) -> bool:
        """Check if this order is for a cryptocurrency."""
        return self._is_crypto

    # =========================================================================
    # QUANTITY METHODS
    # =========================================================================

    def notional(self, amount: float) -> "OrderBuilder":
        """
        Set dollar amount for the order instead of quantity.

        Notional orders allow you to specify a dollar amount to invest
        rather than a number of shares. This is useful for:
        - Percentage-based position sizing
        - Fractional share trading
        - Portfolio rebalancing

        Args:
            amount: Dollar amount to invest (minimum $1.00, maximum $1,000,000)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If amount is less than $1.00 or greater than $1,000,000

        Example:
            order = (OrderBuilder("AAPL", "buy")
                     .notional(1500.00)  # Buy $1500 worth of AAPL
                     .market()
                     .day()
                     .build())
        """
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise ValueError(f"Notional amount must be numeric, got: {amount}") from None

        if amount < self.MIN_NOTIONAL:
            raise ValueError(f"Minimum notional order is ${self.MIN_NOTIONAL:.2f}")
        if amount > self.MAX_NOTIONAL:
            raise ValueError(f"Maximum notional order is ${self.MAX_NOTIONAL:,.2f}")

        self._notional = round(amount, 2)
        self.qty = None  # Mutually exclusive with qty
        return self

    # =========================================================================
    # ORDER TYPE METHODS
    # =========================================================================

    def market(self) -> "OrderBuilder":
        """Create a market order (executes at current price)."""
        self._order_type = OrderType.MARKET
        return self

    def limit(self, limit_price: float) -> "OrderBuilder":
        """
        Create a limit order.

        Args:
            limit_price: Maximum price for buy, minimum for sell
        """
        self._order_type = OrderType.LIMIT
        self._limit_price = self._validate_price(limit_price)
        return self

    def stop(self, stop_price: float) -> "OrderBuilder":
        """
        Create a stop order (converts to market when stop price hit).

        Args:
            stop_price: Trigger price
        """
        self._order_type = OrderType.STOP
        self._stop_price = self._validate_price(stop_price)
        return self

    def stop_limit(self, stop_price: float, limit_price: float) -> "OrderBuilder":
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
        self, trail_price: Optional[float] = None, trail_percent: Optional[float] = None
    ) -> "OrderBuilder":
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

    def day(self) -> "OrderBuilder":
        """Valid only during trading day (cancels at market close)."""
        self._time_in_force = TimeInForce.DAY
        return self

    def gtc(self) -> "OrderBuilder":
        """Good-Till-Canceled (expires after 90 days)."""
        self._time_in_force = TimeInForce.GTC
        return self

    def ioc(self) -> "OrderBuilder":
        """Immediate-Or-Cancel (fills immediately or cancels)."""
        self._time_in_force = TimeInForce.IOC
        return self

    def fok(self) -> "OrderBuilder":
        """Fill-Or-Kill (entire order fills or cancels)."""
        self._time_in_force = TimeInForce.FOK
        return self

    def opg(self) -> "OrderBuilder":
        """Market/Limit on Open (executes in opening auction)."""
        self._time_in_force = TimeInForce.OPG
        return self

    def cls(self) -> "OrderBuilder":
        """Market/Limit on Close (executes in closing auction)."""
        self._time_in_force = TimeInForce.CLS
        return self

    # =========================================================================
    # ADVANCED ORDER CLASSES
    # =========================================================================

    def bracket(
        self, take_profit: float, stop_loss: float, stop_limit: Optional[float] = None
    ) -> "OrderBuilder":
        """
        Create a bracket order (entry + take-profit + stop-loss).

        Args:
            take_profit: Limit price for profit target
            stop_loss: Stop price for loss protection
            stop_limit: Optional limit price for stop-loss (makes it stop-limit)

        Note: Entry order can be market or limit type.
        """
        self._order_class = OrderClass.BRACKET

        self._take_profit = {"limit_price": self._validate_price(take_profit)}

        self._stop_loss = {"stop_price": self._validate_price(stop_loss)}

        if stop_limit:
            self._stop_loss["limit_price"] = self._validate_price(stop_limit)

        # Bracket orders only support day or gtc
        if self._time_in_force not in [TimeInForce.DAY, TimeInForce.GTC]:
            logger.warning(
                f"Bracket orders only support DAY or GTC, changing from {self._time_in_force}"
            )
            self._time_in_force = TimeInForce.GTC

        return self

    def oco(
        self, take_profit: float, stop_loss: float, stop_limit: Optional[float] = None
    ) -> "OrderBuilder":
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

        self._take_profit = {"limit_price": self._validate_price(take_profit)}

        self._stop_loss = {"stop_price": self._validate_price(stop_loss)}

        if stop_limit:
            self._stop_loss["limit_price"] = self._validate_price(stop_limit)

        return self

    def oto(
        self,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        stop_limit: Optional[float] = None,
    ) -> "OrderBuilder":
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
            self._take_profit = {"limit_price": self._validate_price(take_profit)}

        if stop_loss:
            self._stop_loss = {"stop_price": self._validate_price(stop_loss)}
            if stop_limit:
                self._stop_loss["limit_price"] = self._validate_price(stop_limit)

        return self

    # =========================================================================
    # ADDITIONAL OPTIONS
    # =========================================================================

    def extended_hours(self, enabled: bool = True) -> "OrderBuilder":
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

    def client_order_id(self, order_id: str) -> "OrderBuilder":
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
            "symbol": self.symbol,
            "side": self.side,
            "time_in_force": self._time_in_force,
        }

        # Add qty or notional (mutually exclusive)
        if self._notional is not None:
            base_kwargs["notional"] = self._notional
        else:
            base_kwargs["qty"] = self.qty

        # Add optional fields
        if self._client_order_id:
            base_kwargs["client_order_id"] = self._client_order_id
        if self._extended_hours:
            base_kwargs["extended_hours"] = True

        # Add order class and legs for bracket/OCO/OTO
        if self._order_class != OrderClass.SIMPLE:
            base_kwargs["order_class"] = self._order_class

            if self._take_profit:
                base_kwargs["take_profit"] = self._take_profit
            if self._stop_loss:
                base_kwargs["stop_loss"] = self._stop_loss

        # Build specific order type
        if self._order_type == OrderType.MARKET:
            return MarketOrderRequest(**base_kwargs)

        elif self._order_type == OrderType.LIMIT:
            return LimitOrderRequest(limit_price=self._limit_price, **base_kwargs)

        elif self._order_type == OrderType.STOP:
            return StopOrderRequest(stop_price=self._stop_price, **base_kwargs)

        elif self._order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                stop_price=self._stop_price, limit_price=self._limit_price, **base_kwargs
            )

        elif self._order_type == OrderType.TRAILING_STOP:
            trailing_kwargs = base_kwargs.copy()

            if self._trail_price:
                trailing_kwargs["trail_price"] = self._trail_price
            else:
                trailing_kwargs["trail_percent"] = self._trail_percent

            # Trailing stops only support DAY or GTC
            if self._time_in_force not in [TimeInForce.DAY, TimeInForce.GTC]:
                logger.warning("Trailing stop only supports DAY/GTC, using GTC")
                trailing_kwargs["time_in_force"] = TimeInForce.GTC

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
        # Quantity or notional must be specified
        if self.qty is None and self._notional is None:
            raise ValueError(
                "Either quantity or notional amount must be specified. "
                "Use qty parameter in constructor or call .notional() method."
            )

        # Notional order restrictions (Alpaca API constraints)
        if self._notional is not None:
            # Notional orders only work with market and limit orders
            if self._order_type not in [OrderType.MARKET, OrderType.LIMIT]:
                raise ValueError(
                    f"Notional orders only support MARKET or LIMIT order types, "
                    f"got: {self._order_type}"
                )

            # Notional orders cannot be used with advanced order classes
            if self._order_class != OrderClass.SIMPLE:
                raise ValueError(
                    f"Notional orders cannot be used with {self._order_class} order class. "
                    "Use quantity-based orders for bracket, OCO, or OTO orders."
                )

            # Notional orders cannot use extended hours
            if self._extended_hours:
                raise ValueError(
                    "Notional orders cannot be used with extended hours trading"
                )

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
        if self._notional is not None:
            size_info = f"notional=${self._notional:.2f}"
        else:
            size_info = f"qty={self.qty}"
        return (
            f"OrderBuilder(symbol={self.symbol}, side={self.side}, {size_info}, "
            f"type={self._order_type}, tif={self._time_in_force}, "
            f"class={self._order_class})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def market_order(symbol: str, side: str, qty: float, **kwargs) -> Any:
    """Quick market order creation."""
    builder = OrderBuilder(symbol, side, qty).market()

    if "gtc" in kwargs and kwargs["gtc"]:
        builder.gtc()

    return builder.build()


def limit_order(symbol: str, side: str, qty: float, limit_price: float, **kwargs) -> Any:
    """Quick limit order creation."""
    builder = OrderBuilder(symbol, side, qty).limit(limit_price)

    if "gtc" in kwargs and kwargs["gtc"]:
        builder.gtc()
    if "extended_hours" in kwargs and kwargs["extended_hours"]:
        builder.extended_hours().day()

    return builder.build()


def bracket_order(
    symbol: str,
    side: str,
    qty: float,
    entry_price: Optional[float] = None,
    take_profit: float = None,
    stop_loss: float = None,
    stop_limit: Optional[float] = None,
) -> Any:
    """Quick bracket order creation."""
    builder = OrderBuilder(symbol, side, qty)

    if entry_price:
        builder.limit(entry_price)
    else:
        builder.market()

    builder.bracket(take_profit=take_profit, stop_loss=stop_loss, stop_limit=stop_limit).gtc()

    return builder.build()


def notional_market_order(symbol: str, side: str, amount: float, **kwargs) -> Any:
    """
    Quick notional (dollar-based) market order creation.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        side: 'buy' or 'sell'
        amount: Dollar amount to invest (minimum $1.00)
        **kwargs: Optional arguments (gtc=True for Good-Till-Canceled)

    Returns:
        Alpaca MarketOrderRequest with notional amount

    Example:
        # Buy $1500 worth of AAPL
        order = notional_market_order("AAPL", "buy", 1500.00)
    """
    builder = OrderBuilder(symbol, side).notional(amount).market()

    if kwargs.get("gtc"):
        builder.gtc()

    return builder.build()


def notional_limit_order(
    symbol: str, side: str, amount: float, limit_price: float, **kwargs
) -> Any:
    """
    Quick notional (dollar-based) limit order creation.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        side: 'buy' or 'sell'
        amount: Dollar amount to invest (minimum $1.00)
        limit_price: Maximum price for buy, minimum for sell
        **kwargs: Optional arguments (gtc=True for Good-Till-Canceled)

    Returns:
        Alpaca LimitOrderRequest with notional amount

    Example:
        # Buy $1500 worth of AAPL at limit price of $150
        order = notional_limit_order("AAPL", "buy", 1500.00, 150.00, gtc=True)
    """
    builder = OrderBuilder(symbol, side).notional(amount).limit(limit_price)

    if kwargs.get("gtc"):
        builder.gtc()

    return builder.build()


# =============================================================================
# CRYPTO CONVENIENCE FUNCTIONS
# =============================================================================


def crypto_market_order(symbol: str, side: str, qty: float = None, notional: float = None) -> Any:
    """
    Quick crypto market order creation.

    Crypto orders are available 24/7 and default to GTC (Good-Till-Canceled).

    Args:
        symbol: Crypto pair (e.g., 'BTC/USD', 'BTCUSD', 'ETH/USD')
        side: 'buy' or 'sell'
        qty: Quantity in base currency (e.g., 0.5 BTC). Mutually exclusive with notional.
        notional: Dollar amount (e.g., 1000.00 for $1000). Mutually exclusive with qty.

    Returns:
        Alpaca MarketOrderRequest for crypto

    Examples:
        # Buy 0.5 BTC
        order = crypto_market_order("BTC/USD", "buy", qty=0.5)

        # Buy $1000 worth of ETH
        order = crypto_market_order("ETHUSD", "buy", notional=1000.00)
    """
    if qty is None and notional is None:
        raise ValueError("Either qty or notional must be specified")
    if qty is not None and notional is not None:
        raise ValueError("Specify either qty or notional, not both")

    builder = OrderBuilder(symbol, side, qty)

    if notional is not None:
        builder.notional(notional)

    return builder.market().build()


def crypto_limit_order(
    symbol: str, side: str, qty: float, limit_price: float
) -> Any:
    """
    Quick crypto limit order creation.

    Crypto orders are available 24/7 and default to GTC (Good-Till-Canceled).

    Args:
        symbol: Crypto pair (e.g., 'BTC/USD', 'BTCUSD', 'ETH/USD')
        side: 'buy' or 'sell'
        qty: Quantity in base currency (e.g., 0.5 BTC)
        limit_price: Limit price for the order

    Returns:
        Alpaca LimitOrderRequest for crypto

    Example:
        # Buy 0.5 BTC at $40,000
        order = crypto_limit_order("BTC/USD", "buy", 0.5, 40000.00)
    """
    return OrderBuilder(symbol, side, qty).limit(limit_price).build()


# Re-export is_crypto_symbol from crypto_utils for backward compatibility
# The function is already imported at module level:
# from utils.crypto_utils import is_crypto_symbol, normalize_crypto_symbol
#
# External code can use either:
#   from brokers.order_builder import is_crypto_symbol
#   from utils.crypto_utils import is_crypto_symbol
