#!/usr/bin/env python3
"""
Options trading support for Alpaca API.

This module provides options trading functionality including:
- OCC symbol building and parsing
- Option chain retrieval with filtering
- Single-leg options orders (buy/sell calls/puts)
- Strategy helpers (covered calls, cash-secured puts)
- Greeks and quote data when available

Note: Requires options trading approval on your Alpaca account.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from config import ALPACA_CREDS

# Environment-aware logging
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type (Call or Put)."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """
    Represents an option contract with quote and Greeks data.

    Attributes:
        symbol: OCC option symbol (e.g., AAPL230120C00150000)
        underlying: Underlying stock symbol (e.g., AAPL)
        expiration: Expiration date
        strike: Strike price
        option_type: Call or Put
        delta/gamma/theta/vega: Option Greeks (optional)
        implied_volatility: Implied volatility (optional)
        bid/ask/last: Quote prices (optional)
        volume/open_interest: Trading activity (optional)
    """
    symbol: str
    underlying: str
    expiration: date
    strike: float
    option_type: OptionType

    # Greeks (populated from quotes when available)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None

    # Quote data
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask spread."""
        if self.bid is not None and self.ask is not None:
            return round((self.bid + self.ask) / 2, 2)
        return self.last

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return round(self.ask - self.bid, 2)
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        spread = self.spread
        if mid and spread and mid > 0:
            return round((spread / mid) * 100, 2)
        return None

    @property
    def days_to_expiration(self) -> int:
        """Calculate days until expiration."""
        today = date.today()
        return (self.expiration - today).days

    def is_itm(self, underlying_price: float) -> bool:
        """
        Check if option is in-the-money.

        Args:
            underlying_price: Current price of the underlying asset.

        Returns:
            True if the option is in-the-money, False otherwise.
        """
        if self.option_type == OptionType.CALL:
            return underlying_price > self.strike
        else:  # PUT
            return underlying_price < self.strike

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptionContract({self.underlying} {self.expiration} "
            f"{self.strike} {self.option_type.value.upper()}, "
            f"bid={self.bid}, ask={self.ask})"
        )


@dataclass
class OptionChain:
    """
    Represents an option chain for a specific underlying and expiration.

    Attributes:
        underlying: Underlying stock symbol
        expiration: Expiration date
        calls: List of call option contracts
        puts: List of put option contracts
    """
    underlying: str
    expiration: date
    calls: List[OptionContract] = field(default_factory=list)
    puts: List[OptionContract] = field(default_factory=list)

    @property
    def num_strikes(self) -> int:
        """Number of unique strike prices."""
        strikes = set()
        for c in self.calls:
            strikes.add(c.strike)
        for p in self.puts:
            strikes.add(p.strike)
        return len(strikes)

    def get_call_at_strike(self, strike: float) -> Optional[OptionContract]:
        """Get call option at specific strike."""
        for c in self.calls:
            if abs(c.strike - strike) < 0.01:
                return c
        return None

    def get_put_at_strike(self, strike: float) -> Optional[OptionContract]:
        """Get put option at specific strike."""
        for p in self.puts:
            if abs(p.strike - strike) < 0.01:
                return p
        return None

    def get_atm_strike(self, underlying_price: float) -> float:
        """Get the at-the-money strike closest to underlying price."""
        all_strikes = sorted(set(c.strike for c in self.calls))
        if not all_strikes:
            return underlying_price

        closest = min(all_strikes, key=lambda x: abs(x - underlying_price))
        return closest

    def filter_by_delta(
        self,
        min_delta: float = None,
        max_delta: float = None
    ) -> "OptionChain":
        """
        Filter contracts by delta range.

        Args:
            min_delta: Minimum absolute delta (e.g., 0.20)
            max_delta: Maximum absolute delta (e.g., 0.40)

        Returns:
            New OptionChain with filtered contracts
        """
        def delta_in_range(contract: OptionContract) -> bool:
            if contract.delta is None:
                return True  # Include if no delta available
            abs_delta = abs(contract.delta)
            if min_delta is not None and abs_delta < min_delta:
                return False
            if max_delta is not None and abs_delta > max_delta:
                return False
            return True

        return OptionChain(
            underlying=self.underlying,
            expiration=self.expiration,
            calls=[c for c in self.calls if delta_in_range(c)],
            puts=[p for p in self.puts if delta_in_range(p)]
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptionChain({self.underlying} {self.expiration}, "
            f"{len(self.calls)} calls, {len(self.puts)} puts)"
        )


class OptionsError(Exception):
    """Base exception for options trading errors."""
    pass


class OptionsNotEnabledError(OptionsError):
    """Raised when options trading is not enabled on account."""
    pass


class InvalidContractError(OptionsError):
    """Raised when an invalid option contract is specified."""
    pass


class OptionsBroker:
    """
    Options trading functionality for Alpaca.

    Provides methods for:
    - Building and parsing OCC option symbols
    - Retrieving option chains with filtering
    - Getting option quotes
    - Submitting single-leg option orders
    - Strategy helpers (covered calls, cash-secured puts)

    Note: Requires options trading approval on your Alpaca account.
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        paper: bool = True
    ):
        """
        Initialize OptionsBroker.

        Args:
            api_key: Alpaca API key (defaults to env var)
            secret_key: Alpaca secret key (defaults to env var)
            paper: Use paper trading environment (default True)
        """
        self._api_key = api_key or ALPACA_CREDS.get("API_KEY", "")
        self._secret_key = secret_key or ALPACA_CREDS.get("API_SECRET", "")

        # Handle paper parameter - can be string or bool
        if isinstance(paper, str):
            self.paper = paper.lower() == "true"
        else:
            self.paper = bool(paper)

        self.logger = logging.getLogger(__name__)

        # Lazy-loaded clients
        self._trading_client = None
        self._options_client = None

        # Cache for option chains (TTL-based)
        self._chain_cache: Dict[str, tuple] = {}  # {key: (chain, timestamp)}
        self._chain_cache_ttl = timedelta(minutes=5)

    def _get_trading_client(self):
        """Lazy load trading client."""
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient
                self._trading_client = TradingClient(
                    self._api_key,
                    self._secret_key,
                    paper=self.paper
                )
                self.logger.debug("Initialized options trading client")
            except ImportError as e:
                self.logger.error(f"Failed to import TradingClient: {e}")
                raise OptionsError("alpaca-py package not installed or import error")
        return self._trading_client

    def _get_options_client(self):
        """Lazy load options data client."""
        if self._options_client is None:
            try:
                from alpaca.data.historical.option import OptionHistoricalDataClient
                self._options_client = OptionHistoricalDataClient(
                    self._api_key,
                    self._secret_key
                )
                self.logger.debug("Initialized options data client")
            except ImportError:
                self.logger.warning(
                    "OptionHistoricalDataClient not available. "
                    "Options data features will be limited. "
                    "Ensure alpaca-py >= 0.14.0 is installed."
                )
                return None
            except Exception as e:
                self.logger.error(f"Error initializing options data client: {e}")
                return None
        return self._options_client

    # =========================================================================
    # OCC SYMBOL METHODS
    # =========================================================================

    @staticmethod
    def build_occ_symbol(
        underlying: str,
        expiration: date,
        option_type: OptionType,
        strike: float
    ) -> str:
        """
        Build OCC option symbol from components.

        OCC Format: AAPL  230120C00150000
        - Underlying: 6 characters, left-justified, space-padded
        - Expiration: YYMMDD
        - Option Type: C (call) or P (put)
        - Strike: Strike * 1000, 8 digits, zero-padded

        Args:
            underlying: Stock symbol (e.g., "AAPL")
            expiration: Expiration date
            option_type: OptionType.CALL or OptionType.PUT
            strike: Strike price (e.g., 150.00)

        Returns:
            OCC symbol string (e.g., "AAPL  230120C00150000")

        Example:
            >>> OptionsBroker.build_occ_symbol("AAPL", date(2023, 1, 20), OptionType.CALL, 150)
            "AAPL  230120C00150000"
        """
        # Validate inputs
        if not underlying or not isinstance(underlying, str):
            raise ValueError("Underlying must be a non-empty string")

        underlying = underlying.upper().strip()
        if len(underlying) > 6:
            raise ValueError(f"Underlying symbol too long: {underlying}")

        # Pad underlying to 6 characters
        underlying_padded = underlying.ljust(6)

        # Format expiration as YYMMDD
        exp_str = expiration.strftime("%y%m%d")

        # Option type indicator
        cp = "C" if option_type == OptionType.CALL else "P"

        # Strike price: multiply by 1000 and format as 8 digits
        # This handles strikes like 150.00 -> 150000 and 150.50 -> 150500
        strike_int = int(round(strike * 1000))
        strike_str = f"{strike_int:08d}"

        return f"{underlying_padded}{exp_str}{cp}{strike_str}"

    @staticmethod
    def parse_occ_symbol(occ_symbol: str) -> dict:
        """
        Parse OCC option symbol into components.

        Args:
            occ_symbol: OCC symbol (e.g., "AAPL  230120C00150000")

        Returns:
            Dict with keys: underlying, expiration, option_type, strike

        Example:
            >>> OptionsBroker.parse_occ_symbol("AAPL  230120C00150000")
            {"underlying": "AAPL", "expiration": date(2023, 1, 20),
             "option_type": OptionType.CALL, "strike": 150.0}
        """
        if not occ_symbol or len(occ_symbol) < 15:
            raise InvalidContractError(f"Invalid OCC symbol format: {occ_symbol}")

        try:
            # Extract components
            underlying = occ_symbol[:6].strip()
            exp_str = occ_symbol[6:12]
            option_char = occ_symbol[12]
            strike_str = occ_symbol[13:21]

            # Parse expiration
            expiration = datetime.strptime(exp_str, "%y%m%d").date()

            # Parse option type
            if option_char.upper() == "C":
                option_type = OptionType.CALL
            elif option_char.upper() == "P":
                option_type = OptionType.PUT
            else:
                raise InvalidContractError(f"Invalid option type: {option_char}")

            # Parse strike (divide by 1000)
            strike = int(strike_str) / 1000.0

            return {
                "underlying": underlying,
                "expiration": expiration,
                "option_type": option_type,
                "strike": strike
            }
        except ValueError as e:
            raise InvalidContractError(f"Error parsing OCC symbol '{occ_symbol}': {e}")

    @staticmethod
    def is_option_symbol(symbol: str) -> bool:
        """
        Check if a symbol is an option symbol (OCC format).

        Args:
            symbol: Symbol to check

        Returns:
            True if symbol appears to be an OCC option symbol
        """
        if not symbol or len(symbol) < 15:
            return False

        # Check if it has the structure of an OCC symbol
        # 6 char underlying + 6 char date + 1 char C/P + 8 char strike = 21
        if len(symbol) < 15 or len(symbol) > 21:
            return False

        # Check for C or P at position 12
        if len(symbol) >= 13:
            option_char = symbol[12].upper()
            if option_char not in ("C", "P"):
                return False

        return True

    # =========================================================================
    # OPTION CHAIN METHODS
    # =========================================================================

    async def get_option_chain(
        self,
        underlying: str,
        expiration: date = None,
        min_strike: float = None,
        max_strike: float = None,
        min_dte: int = None,
        max_dte: int = None,
        use_cache: bool = True
    ) -> Optional[OptionChain]:
        """
        Get option chain for underlying symbol.

        Args:
            underlying: Stock symbol (e.g., "AAPL")
            expiration: Specific expiration date (default: nearest monthly)
            min_strike: Minimum strike price filter
            max_strike: Maximum strike price filter
            min_dte: Minimum days to expiration filter
            max_dte: Maximum days to expiration filter
            use_cache: Use cached data if available (default True)

        Returns:
            OptionChain or None if unavailable
        """
        underlying = underlying.upper().strip()

        # Build cache key
        cache_key = f"{underlying}:{expiration}:{min_strike}:{max_strike}"

        # Check cache
        if use_cache and cache_key in self._chain_cache:
            cached_chain, cached_time = self._chain_cache[cache_key]
            if datetime.now() - cached_time < self._chain_cache_ttl:
                self.logger.debug(f"Cache hit for option chain: {cache_key}")
                return cached_chain

        try:
            client = self._get_options_client()
            if client is None:
                self.logger.warning("Options data client not available")
                return None

            # Try to use Alpaca's option chain endpoint
            from alpaca.data.requests import OptionChainRequest

            request_params = {"underlying_symbol": underlying}

            if expiration:
                request_params["expiration_date"] = expiration

            request = OptionChainRequest(**request_params)

            chain_data = await asyncio.to_thread(
                client.get_option_chain,
                request
            )

            calls = []
            puts = []
            actual_expiration = expiration or date.today()

            for contract in chain_data:
                try:
                    parsed = self.parse_occ_symbol(contract.symbol)

                    # Apply strike filters
                    if min_strike is not None and parsed["strike"] < min_strike:
                        continue
                    if max_strike is not None and parsed["strike"] > max_strike:
                        continue

                    # Apply DTE filters
                    dte = (parsed["expiration"] - date.today()).days
                    if min_dte is not None and dte < min_dte:
                        continue
                    if max_dte is not None and dte > max_dte:
                        continue

                    opt = OptionContract(
                        symbol=contract.symbol,
                        underlying=underlying,
                        expiration=parsed["expiration"],
                        strike=parsed["strike"],
                        option_type=parsed["option_type"]
                    )

                    # Update actual expiration if not specified
                    if expiration is None:
                        actual_expiration = parsed["expiration"]

                    if parsed["option_type"] == OptionType.CALL:
                        calls.append(opt)
                    else:
                        puts.append(opt)

                except Exception as e:
                    self.logger.debug(f"Error parsing contract: {e}")
                    continue

            # Sort by strike
            calls.sort(key=lambda x: x.strike)
            puts.sort(key=lambda x: x.strike)

            result = OptionChain(
                underlying=underlying,
                expiration=actual_expiration,
                calls=calls,
                puts=puts
            )

            # Cache result
            self._chain_cache[cache_key] = (result, datetime.now())

            self.logger.info(
                f"Retrieved option chain for {underlying}: "
                f"{len(calls)} calls, {len(puts)} puts"
            )

            return result

        except ImportError:
            self.logger.warning("OptionChainRequest not available in this alpaca-py version")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching option chain: {e}", exc_info=DEBUG_MODE)
            return None

    async def get_expirations(self, underlying: str) -> List[date]:
        """
        Get available expiration dates for underlying.

        Args:
            underlying: Stock symbol

        Returns:
            List of available expiration dates, sorted chronologically
        """
        try:
            client = self._get_options_client()
            if client is None:
                return []

            # This would require an API endpoint that lists expirations
            # For now, we'll return common monthly expirations
            today = date.today()
            expirations = []

            # Generate next 6 monthly expirations (3rd Friday of each month)
            for i in range(6):
                month = today.month + i
                year = today.year + (month - 1) // 12
                month = ((month - 1) % 12) + 1

                # Find 3rd Friday
                first_day = date(year, month, 1)
                first_friday = first_day + timedelta(
                    days=(4 - first_day.weekday() + 7) % 7
                )
                third_friday = first_friday + timedelta(days=14)

                if third_friday > today:
                    expirations.append(third_friday)

            return expirations

        except Exception as e:
            self.logger.error(f"Error getting expirations: {e}", exc_info=DEBUG_MODE)
            return []

    # =========================================================================
    # OPTION QUOTE METHODS
    # =========================================================================

    async def get_option_quote(self, occ_symbol: str) -> Optional[OptionContract]:
        """
        Get latest quote for a specific option contract.

        Args:
            occ_symbol: OCC option symbol

        Returns:
            OptionContract with quote data, or None
        """
        try:
            client = self._get_options_client()
            if client is None:
                return None

            from alpaca.data.requests import OptionLatestQuoteRequest

            request = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol)
            quote_data = await asyncio.to_thread(
                client.get_option_latest_quote,
                request
            )

            if occ_symbol in quote_data:
                q = quote_data[occ_symbol]
                parsed = self.parse_occ_symbol(occ_symbol)

                return OptionContract(
                    symbol=occ_symbol,
                    underlying=parsed["underlying"],
                    expiration=parsed["expiration"],
                    strike=parsed["strike"],
                    option_type=parsed["option_type"],
                    bid=float(q.bid_price) if q.bid_price else None,
                    ask=float(q.ask_price) if q.ask_price else None,
                    volume=int(q.bid_size + q.ask_size) if q.bid_size and q.ask_size else None
                )

            return None

        except ImportError:
            self.logger.warning("OptionLatestQuoteRequest not available")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching option quote: {e}", exc_info=DEBUG_MODE)
            return None

    async def get_option_quotes(
        self,
        occ_symbols: List[str]
    ) -> Dict[str, OptionContract]:
        """
        Get quotes for multiple option contracts.

        Args:
            occ_symbols: List of OCC option symbols

        Returns:
            Dict mapping symbol to OptionContract with quote data
        """
        results = {}

        # Batch quotes in groups of 100 (API limit)
        batch_size = 100
        for i in range(0, len(occ_symbols), batch_size):
            batch = occ_symbols[i:i + batch_size]

            try:
                client = self._get_options_client()
                if client is None:
                    break

                from alpaca.data.requests import OptionLatestQuoteRequest

                request = OptionLatestQuoteRequest(symbol_or_symbols=batch)
                quote_data = await asyncio.to_thread(
                    client.get_option_latest_quote,
                    request
                )

                for symbol, q in quote_data.items():
                    try:
                        parsed = self.parse_occ_symbol(symbol)
                        results[symbol] = OptionContract(
                            symbol=symbol,
                            underlying=parsed["underlying"],
                            expiration=parsed["expiration"],
                            strike=parsed["strike"],
                            option_type=parsed["option_type"],
                            bid=float(q.bid_price) if q.bid_price else None,
                            ask=float(q.ask_price) if q.ask_price else None
                        )
                    except Exception as e:
                        self.logger.debug(f"Error parsing quote for {symbol}: {e}")

            except Exception as e:
                self.logger.error(f"Error fetching batch quotes: {e}", exc_info=DEBUG_MODE)

        return results

    # =========================================================================
    # ORDER SUBMISSION METHODS
    # =========================================================================

    async def submit_option_order(
        self,
        occ_symbol: str,
        side: str,
        qty: int,
        order_type: str = "limit",
        limit_price: float = None,
        time_in_force: str = "day"
    ) -> Optional[dict]:
        """
        Submit an option order.

        Args:
            occ_symbol: OCC option symbol
            side: "buy" or "sell"
            qty: Number of contracts
            order_type: "market" or "limit" (default: "limit")
            limit_price: Price for limit orders (required if limit)
            time_in_force: "day", "gtc", "ioc", "fok" (default: "day")

        Returns:
            Order dict with id, symbol, side, qty, type, status, created_at
            or None on error

        Note:
            Options are typically traded with limit orders to control
            execution price due to wide bid-ask spreads.
        """
        # Validate inputs
        if not self.is_option_symbol(occ_symbol):
            raise InvalidContractError(f"Invalid OCC symbol: {occ_symbol}")

        if side.lower() not in ("buy", "sell"):
            raise ValueError(f"Side must be 'buy' or 'sell', got: {side}")

        if qty <= 0:
            raise ValueError(f"Quantity must be positive, got: {qty}")

        if order_type.lower() == "limit" and limit_price is None:
            raise ValueError("limit_price required for limit orders")

        try:
            client = self._get_trading_client()

            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                OrderSide,
                TimeInForce as TIF
            )

            # Map side
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Map time in force
            tif_map = {
                "day": TIF.DAY,
                "gtc": TIF.GTC,
                "ioc": TIF.IOC,
                "fok": TIF.FOK
            }
            tif = tif_map.get(time_in_force.lower(), TIF.DAY)

            # Build order request
            if order_type.lower() == "market":
                request = MarketOrderRequest(
                    symbol=occ_symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )
            else:
                request = LimitOrderRequest(
                    symbol=occ_symbol,
                    qty=qty,
                    side=order_side,
                    limit_price=round(limit_price, 2),
                    time_in_force=tif
                )

            # Submit order
            order = await asyncio.to_thread(client.submit_order, request)

            result = {
                "id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": str(order.qty),
                "type": order.type.value,
                "status": order.status.value,
                "created_at": order.created_at,
                "limit_price": str(order.limit_price) if order.limit_price else None
            }

            self.logger.info(
                f"Option order submitted: {result['id']} - "
                f"{side.upper()} {qty} {occ_symbol} @ "
                f"{'MKT' if order_type == 'market' else f'${limit_price:.2f}'}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error submitting option order: {e}", exc_info=DEBUG_MODE)
            return None

    async def cancel_option_order(self, order_id: str) -> bool:
        """
        Cancel an open option order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self._get_trading_client()
            await asyncio.to_thread(client.cancel_order_by_id, order_id)
            self.logger.info(f"Canceled option order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}", exc_info=DEBUG_MODE)
            return False

    # =========================================================================
    # POSITION METHODS
    # =========================================================================

    async def get_option_positions(self) -> List[dict]:
        """
        Get all open option positions.

        Returns:
            List of position dicts with symbol, underlying, expiration,
            strike, option_type, qty, avg_entry_price, market_value,
            unrealized_pl, unrealized_plpc
        """
        try:
            client = self._get_trading_client()
            positions = await asyncio.to_thread(client.get_all_positions)

            option_positions = []
            for pos in positions:
                # Check if this is an option position
                if self.is_option_symbol(pos.symbol):
                    try:
                        parsed = self.parse_occ_symbol(pos.symbol)
                        option_positions.append({
                            "symbol": pos.symbol,
                            "underlying": parsed["underlying"],
                            "expiration": parsed["expiration"],
                            "strike": parsed["strike"],
                            "option_type": parsed["option_type"].value,
                            "qty": int(pos.qty),
                            "avg_entry_price": float(pos.avg_entry_price),
                            "market_value": float(pos.market_value),
                            "unrealized_pl": float(pos.unrealized_pl),
                            "unrealized_plpc": float(pos.unrealized_plpc),
                            "cost_basis": float(pos.cost_basis)
                        })
                    except Exception as e:
                        self.logger.debug(f"Error parsing position {pos.symbol}: {e}")

            return option_positions

        except Exception as e:
            self.logger.error(f"Error fetching option positions: {e}", exc_info=DEBUG_MODE)
            return []

    async def close_option_position(
        self,
        occ_symbol: str,
        order_type: str = "market"
    ) -> Optional[dict]:
        """
        Close an option position.

        Args:
            occ_symbol: OCC symbol of position to close
            order_type: "market" or "limit"

        Returns:
            Order dict or None
        """
        try:
            # Get current position
            positions = await self.get_option_positions()
            pos = next((p for p in positions if p["symbol"] == occ_symbol), None)

            if pos is None:
                self.logger.warning(f"No position found for {occ_symbol}")
                return None

            qty = abs(pos["qty"])
            side = "sell" if pos["qty"] > 0 else "buy"

            # Get current quote for limit price
            limit_price = None
            if order_type.lower() == "limit":
                quote = await self.get_option_quote(occ_symbol)
                if quote:
                    # Use bid for sells, ask for buys
                    limit_price = quote.bid if side == "sell" else quote.ask

            return await self.submit_option_order(
                occ_symbol=occ_symbol,
                side=side,
                qty=qty,
                order_type=order_type,
                limit_price=limit_price
            )

        except Exception as e:
            self.logger.error(f"Error closing position {occ_symbol}: {e}", exc_info=DEBUG_MODE)
            return None

    # =========================================================================
    # STRATEGY HELPERS
    # =========================================================================

    async def buy_call(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        qty: int,
        limit_price: float = None
    ) -> Optional[dict]:
        """
        Buy a call option (bullish strategy).

        Args:
            underlying: Stock symbol
            expiration: Expiration date
            strike: Strike price
            qty: Number of contracts
            limit_price: Limit price (recommended)

        Returns:
            Order dict or None
        """
        symbol = self.build_occ_symbol(underlying, expiration, OptionType.CALL, strike)
        return await self.submit_option_order(
            occ_symbol=symbol,
            side="buy",
            qty=qty,
            order_type="limit" if limit_price else "market",
            limit_price=limit_price
        )

    async def buy_put(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        qty: int,
        limit_price: float = None
    ) -> Optional[dict]:
        """
        Buy a put option (bearish/hedging strategy).

        Args:
            underlying: Stock symbol
            expiration: Expiration date
            strike: Strike price
            qty: Number of contracts
            limit_price: Limit price (recommended)

        Returns:
            Order dict or None
        """
        symbol = self.build_occ_symbol(underlying, expiration, OptionType.PUT, strike)
        return await self.submit_option_order(
            occ_symbol=symbol,
            side="buy",
            qty=qty,
            order_type="limit" if limit_price else "market",
            limit_price=limit_price
        )

    async def sell_covered_call(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        qty: int,
        limit_price: float = None
    ) -> Optional[dict]:
        """
        Sell a covered call (income strategy).

        Requires owning 100 shares of underlying per contract.

        Args:
            underlying: Stock symbol (must own shares)
            expiration: Expiration date
            strike: Strike price (above current price for OTM)
            qty: Number of contracts (need 100 shares per contract)
            limit_price: Limit price (recommended)

        Returns:
            Order dict or None
        """
        symbol = self.build_occ_symbol(underlying, expiration, OptionType.CALL, strike)
        return await self.submit_option_order(
            occ_symbol=symbol,
            side="sell",
            qty=qty,
            order_type="limit" if limit_price else "market",
            limit_price=limit_price
        )

    async def sell_cash_secured_put(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        qty: int,
        limit_price: float = None
    ) -> Optional[dict]:
        """
        Sell a cash-secured put (income/acquisition strategy).

        Requires cash to cover assignment (strike * 100 * qty).

        Args:
            underlying: Stock symbol
            expiration: Expiration date
            strike: Strike price (below current price for OTM)
            qty: Number of contracts
            limit_price: Limit price (recommended)

        Returns:
            Order dict or None
        """
        symbol = self.build_occ_symbol(underlying, expiration, OptionType.PUT, strike)
        return await self.submit_option_order(
            occ_symbol=symbol,
            side="sell",
            qty=qty,
            order_type="limit" if limit_price else "market",
            limit_price=limit_price
        )

    async def buy_to_close(
        self,
        occ_symbol: str,
        qty: int,
        limit_price: float = None
    ) -> Optional[dict]:
        """
        Buy to close a short option position.

        Args:
            occ_symbol: OCC symbol of short position
            qty: Number of contracts to close
            limit_price: Limit price (recommended)

        Returns:
            Order dict or None
        """
        return await self.submit_option_order(
            occ_symbol=occ_symbol,
            side="buy",
            qty=qty,
            order_type="limit" if limit_price else "market",
            limit_price=limit_price
        )

    async def sell_to_close(
        self,
        occ_symbol: str,
        qty: int,
        limit_price: float = None
    ) -> Optional[dict]:
        """
        Sell to close a long option position.

        Args:
            occ_symbol: OCC symbol of long position
            qty: Number of contracts to close
            limit_price: Limit price (recommended)

        Returns:
            Order dict or None
        """
        return await self.submit_option_order(
            occ_symbol=occ_symbol,
            side="sell",
            qty=qty,
            order_type="limit" if limit_price else "market",
            limit_price=limit_price
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def calculate_max_loss(
        self,
        option_type: OptionType,
        side: str,
        strike: float,
        premium: float,
        qty: int = 1
    ) -> float:
        """
        Calculate maximum loss for an option position.

        Args:
            option_type: CALL or PUT
            side: "buy" or "sell"
            strike: Strike price
            premium: Option premium paid/received
            qty: Number of contracts

        Returns:
            Maximum potential loss (positive number)
        """
        if side.lower() == "buy":
            # Long options: max loss is premium paid
            return premium * 100 * qty
        else:
            # Short options
            if option_type == OptionType.CALL:
                # Short call: unlimited loss (return large number)
                return float('inf')
            else:
                # Short put: max loss is strike - premium
                return max(0, (strike - premium) * 100 * qty)

    def calculate_breakeven(
        self,
        option_type: OptionType,
        strike: float,
        premium: float,
        **kwargs,
    ) -> float:
        """
        Calculate breakeven price for an option position.

        Args:
            option_type: CALL or PUT
            strike: Strike price
            premium: Option premium

        Returns:
            Breakeven underlying price
        """
        if option_type == OptionType.CALL:
            return strike + premium
        else:  # PUT
            return strike - premium

    def clear_cache(self) -> None:
        """Clear option chain cache."""
        self._chain_cache.clear()
        self.logger.debug("Option chain cache cleared")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_contract_value(
    strike: float,
    premium: float,
    qty: int = 1
) -> dict:
    """
    Calculate contract values and costs.

    Args:
        strike: Strike price
        premium: Option premium per share
        qty: Number of contracts

    Returns:
        Dict with total_premium, notional_value, margin_required (approx)
    """
    return {
        "total_premium": premium * 100 * qty,
        "notional_value": strike * 100 * qty,
        "shares_controlled": 100 * qty,
        "cost_to_buy": premium * 100 * qty,
        "cash_for_csp": strike * 100 * qty  # Cash needed for cash-secured put
    }


def get_monthly_expiration(months_out: int = 1) -> date:
    """
    Get the monthly expiration date (3rd Friday) N months out.

    Args:
        months_out: Number of months from now (1 = next month)

    Returns:
        Date of the 3rd Friday
    """
    today = date.today()
    target_month = today.month + months_out
    target_year = today.year + (target_month - 1) // 12
    target_month = ((target_month - 1) % 12) + 1

    # Find first day of month
    first_day = date(target_year, target_month, 1)

    # Find first Friday
    days_until_friday = (4 - first_day.weekday() + 7) % 7
    first_friday = first_day + timedelta(days=days_until_friday)

    # Third Friday is 14 days after first Friday
    third_friday = first_friday + timedelta(days=14)

    return third_friday


def get_weekly_expiration(weeks_out: int = 1) -> date:
    """
    Get the weekly expiration date (Friday) N weeks out.

    Args:
        weeks_out: Number of weeks from now (1 = next Friday)

    Returns:
        Date of the target Friday
    """
    today = date.today()
    days_until_friday = (4 - today.weekday() + 7) % 7
    if days_until_friday == 0:
        days_until_friday = 7  # If today is Friday, go to next Friday

    return today + timedelta(days=days_until_friday + (weeks_out - 1) * 7)
