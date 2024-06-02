"""
AlpacaBroker account / connection mixin.

Contains:
    - Symbol validation
    - Subscriber registry helpers (used by streaming code)
    - Account / clock / market status queries
    - Position queries (positions, get_position, tracked positions)
    - Asset metadata + overnight-tradeable check
    - Partial fill / lifecycle / audit log attachment helpers
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from alpaca.data.historical import CryptoHistoricalDataClient

from utils.audit_log import AuditLog
from utils.crypto_utils import is_crypto_symbol, normalize_crypto_symbol
from utils.order_lifecycle import OrderLifecycleTracker

from brokers.alpaca._retry import (
    DEBUG_MODE,
    BrokerConnectionError,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)


class AlpacaAccountMixin:
    """Account, position, asset, and connection-state helpers for AlpacaBroker."""

    # --- Crypto helpers (shared shim, used by other mixins too) ----------

    def _get_crypto_data_client(self) -> CryptoHistoricalDataClient:
        """
        Lazy load crypto data client.

        Returns:
            CryptoHistoricalDataClient instance
        """
        client = self._crypto_data_client
        if client is None:
            # Crypto data client does not require authentication for public data
            client = CryptoHistoricalDataClient()
            self._crypto_data_client = client
            logger.info("Initialized crypto data client")
        return client

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

    # --- Attachment helpers ----------------------------------------------

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

    # --- Partial fill helpers (registry surface) -------------------------

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

    # --- Account queries -------------------------------------------------

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_account(self):
        """Get account information."""
        try:
            # Use timeout-protected async call to prevent hanging
            account = await self._async_call_with_timeout(
                self.trading_client.get_account,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name="get_account",
            )
            return account
        except asyncio.TimeoutError:
            raise BrokerConnectionError(
                "Account fetch timed out - broker may be unreachable"
            ) from None
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
                operation_name="get_market_status",
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
    async def get_clock(self):
        """
        Return Alpaca market clock object.

        LiveTrader expects this interface to expose `is_open`, `next_open`,
        and `next_close` attributes.
        """
        try:
            return await self._async_call_with_timeout(
                self.trading_client.get_clock,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name="get_clock",
            )
        except asyncio.TimeoutError:
            raise BrokerConnectionError(
                "Clock fetch timed out - broker may be unreachable"
            ) from None
        except Exception as e:
            logger.error(f"Error getting clock: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_positions(self):
        """Get current positions."""
        try:
            # Use timeout-protected async call to prevent hanging
            positions = await self._async_call_with_timeout(
                self.trading_client.get_all_positions,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name="get_positions",
            )
            return positions
        except asyncio.TimeoutError:
            raise BrokerConnectionError(
                "Position fetch timed out - broker may be unreachable"
            ) from None
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
                operation_name=f"get_position({symbol})",
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
                operation_name=f"get_asset({symbol})",
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
                return asset.get("overnight_tradeable", False) and not asset.get(
                    "overnight_halted", True
                )
            return False
        except Exception as e:
            logger.warning(f"Error checking overnight status for {symbol}: {e}")
            return False

    # --- Gateway enforcement (lives with account/connection state) -------

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

        token = secrets.token_hex(16)
        self._gateway_caller_token = token
        self._gateway_required = True
        logger.info("🔒 GATEWAY ENFORCEMENT ENABLED: All orders must route through OrderGateway")
        return token

    def disable_gateway_requirement(self):
        """
        Disable gateway requirement (for testing only).

        WARNING: This should NEVER be called in production.
        """
        self._gateway_required = False
        self._gateway_caller_token = None
        logger.warning("⚠️ GATEWAY ENFORCEMENT DISABLED - Direct order submission allowed")

    # --- Timeout helper (placed here so all mixins can rely on it) -------

    async def _async_call_with_timeout(
        self,
        func,
        *args,
        timeout: Optional[float] = None,
        operation_name: str = "API call",
        **kwargs,
    ) -> Any:
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
            return await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"TIMEOUT: {operation_name} exceeded {timeout}s limit. "
                "This may indicate network issues or API problems."
            )
            raise
