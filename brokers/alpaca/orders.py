"""
AlpacaBroker orders mixin.

Contains:
    - Almgren-Chriss market impact + liquidity check
    - submit_order (legacy dict API)
    - submit_order_advanced (OrderBuilder + alpaca-py order requests)
    - _internal_submit_order (gateway-authorized path)
    - cancel_order, cancel_all_orders, replace_order
    - get_order_by_id, get_order_by_client_id, get_orders
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import numpy as np
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
)

from utils.audit_log import AuditEventType

from brokers.alpaca._retry import (
    DEBUG_MODE,
    BrokerConnectionError,
    GatewayBypassError,
    OrderError,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)


class AlpacaOrdersMixin:
    """Order submission, modification, query, and execution-cost helpers."""

    # =========================================================================
    # ALMGREN-CHRISS MARKET IMPACT MODEL
    # Ported from BacktestBroker for live trading execution awareness
    # =========================================================================

    # Maximum participation rate (never trade > 10% of ADV)
    MAX_PARTICIPATION_RATE = 0.10

    async def _calculate_market_impact(self, symbol: str, qty: float, side: str) -> Dict:
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
            if self.is_crypto(symbol):
                bars = await self.get_crypto_bars(symbol, timeframe="1Day", limit=20)
            else:
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
            volumes = [float(b.volume) for b in bars if hasattr(b, "volume")]
            avg_daily_volume = np.mean(volumes) if volumes else 1000000.0
            avg_daily_volume = max(avg_daily_volume, 100000.0)  # Floor at 100K

            # Calculate volatility (annualized)
            closes = [float(b.close) for b in bars if hasattr(b, "close")]
            if len(closes) >= 2:
                returns = np.diff(np.log(closes))
                volatility = np.std(returns) * np.sqrt(252)
            else:
                volatility = 0.30  # Default 30% volatility

            # Participation rate
            participation_rate = qty / avg_daily_volume

            # Almgren-Chriss coefficients
            c_temp = 0.6  # Temporary impact coefficient
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
        return bool(impact["safe_to_trade"])

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
        return float(impact["expected_slippage_pct"])

    # =========================================================================
    # ORDER SUBMISSION
    # =========================================================================

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def submit_order(self, order):
        """Submit an order."""
        if self._gateway_required:
            raise GatewayBypassError(
                "Direct order submission is disabled. "
                "All orders must route through OrderGateway for safety checks. "
                "Use order_gateway.submit_order() instead of broker.submit_order()."
            )
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
                operation_name=f"submit_order({order['symbol']})",
            )
            logger.info(f"Order submitted: {result.id} for {result.symbol} ({result.qty} shares)")
            return result

        except asyncio.TimeoutError:
            raise OrderError(
                f"Order submission timed out for {order['symbol']} - check order status manually"
            ) from None
        except Exception as e:
            logger.error(f"Error submitting order: {e}", exc_info=DEBUG_MODE)
            raise

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
                    side = (
                        str(order_request.side).lower() if hasattr(order_request, "side") else "buy"
                    )

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
                operation_name=f"submit_order_advanced({order_request.symbol})",
            )

            # Build size info for logging (qty or notional)
            if hasattr(result, "notional") and result.notional is not None:
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
                side = str(result.side).lower() if hasattr(result, "side") else "buy"
                self._partial_fill_tracker.track_order(
                    order_id=str(result.id),
                    symbol=result.symbol,
                    side=side,
                    requested_qty=qty,
                )

            return result

        except asyncio.TimeoutError:
            raise OrderError(
                "Advanced order submission timed out - check order status manually"
            ) from None
        except Exception as e:
            logger.error(f"Error submitting advanced order: {e}", exc_info=DEBUG_MODE)
            raise

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def _internal_submit_order(
        self, order_request, gateway_token: str, check_impact: bool = True
    ):
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
                    side = (
                        str(order_request.side).lower() if hasattr(order_request, "side") else "buy"
                    )

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
                operation_name=f"_internal_submit_order({order_request.symbol})",
            )

            # Build size info for logging
            if hasattr(result, "notional") and result.notional is not None:
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
                side = str(result.side).lower() if hasattr(result, "side") else "buy"
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

    # =========================================================================
    # ORDER MODIFICATION / CANCELLATION
    # =========================================================================

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
                operation_name=f"cancel_order({order_id})",
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
                operation_name="cancel_all_orders",
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
            replace_params: Dict[str, Any] = {}
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

    # =========================================================================
    # ORDER QUERY
    # =========================================================================

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
                operation_name="get_orders",
            )
            return orders
        except asyncio.TimeoutError:
            raise BrokerConnectionError(
                "Get orders timed out - broker may be unreachable"
            ) from None
        except Exception as e:
            logger.error(f"Error getting orders: {e}", exc_info=DEBUG_MODE)
            raise
