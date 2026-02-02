"""
OrderGateway - Centralized order submission with safety checks.

ALL orders must pass through this gateway to ensure:
1. Circuit breaker is checked before every order
2. Position conflicts are detected
3. Risk manager limits are enforced
4. Pre-submit verification is performed
5. Audit trail is maintained

This eliminates the gaps where orders could bypass safety checks.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order submission attempt."""

    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0
    filled_price: Optional[float] = None
    rejection_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class OrderGateway:
    """
    Centralized gateway for all order submissions.

    Ensures ALL orders pass through safety checks before submission.

    Usage:
        gateway = OrderGateway(broker, circuit_breaker, position_manager, risk_manager)

        result = await gateway.submit_order(
            order_request=order,
            strategy_name="MomentumStrategy",
            max_positions=5,
        )

        if not result.success:
            logger.warning(f"Order rejected: {result.rejection_reason}")
    """

    def __init__(
        self,
        broker,
        circuit_breaker=None,
        position_manager=None,
        risk_manager=None,
        enforce_gateway: bool = True,
    ):
        """
        Initialize the order gateway.

        Args:
            broker: Broker instance for order submission
            circuit_breaker: CircuitBreaker instance (optional but recommended)
            position_manager: PositionManager instance (optional but recommended)
            risk_manager: RiskManager instance (optional)
            enforce_gateway: If True, enables mandatory gateway routing (default: True)
        """
        self.broker = broker
        self.circuit_breaker = circuit_breaker
        self.position_manager = position_manager
        self.risk_manager = risk_manager

        # INSTITUTIONAL SAFETY: Enable gateway enforcement
        # This prevents any code from bypassing safety checks by calling
        # broker.submit_order_advanced() directly
        self._gateway_token = None
        if enforce_gateway and hasattr(broker, 'enable_gateway_requirement'):
            self._gateway_token = broker.enable_gateway_requirement()
            logger.info(
                "🔒 OrderGateway initialized with mandatory routing - "
                "direct broker access is now blocked"
            )

        # Statistics
        self._orders_submitted = 0
        self._orders_rejected = 0
        self._rejection_reasons: Dict[str, int] = {}

    async def submit_order(
        self,
        order_request,
        strategy_name: str,
        max_positions: Optional[int] = None,
        price_history: Optional[list] = None,
        current_positions: Optional[dict] = None,
        is_exit_order: bool = False,
    ) -> OrderResult:
        """
        Submit an order with full safety verification.

        Args:
            order_request: Order request object (from OrderBuilder.build())
            strategy_name: Name of the strategy submitting the order
            max_positions: Maximum allowed positions (optional)
            price_history: Price history for risk calculations (optional)
            current_positions: Current positions dict for risk calculations (optional)
            is_exit_order: True if this is an exit/close order (relaxed checks)

        Returns:
            OrderResult with success status and details
        """
        # Extract order details
        try:
            symbol = getattr(order_request, "symbol", None)
            qty = getattr(order_request, "qty", None) or getattr(
                order_request, "quantity", None
            )
            side = getattr(order_request, "side", "buy")
            if hasattr(side, "value"):
                side = side.value
        except Exception as e:
            return self._reject_order("", "buy", 0, f"Invalid order request: {e}")

        if not symbol or not qty:
            return self._reject_order(
                symbol or "", side, 0, "Missing symbol or quantity"
            )

        qty = float(qty)

        # === PRE-FLIGHT SAFETY CHECKS ===

        # 1. Circuit breaker check (ATOMIC - uses enforce_before_order)
        if self.circuit_breaker and not is_exit_order:
            try:
                # Import TradingHaltedException for atomic enforcement
                from utils.circuit_breaker import TradingHaltedException

                # Use enforce_before_order() for atomic check (raises exception on halt)
                await self.circuit_breaker.enforce_before_order(is_exit_order=is_exit_order)

            except TradingHaltedException as e:
                # ATOMIC REJECTION: Circuit breaker raised exception
                return self._reject_order(
                    symbol, side, qty,
                    f"Circuit breaker: {e.reason} ({e.loss_pct:.2%})"
                    if e.loss_pct else f"Circuit breaker: {e.reason}"
                )
            except Exception as e:
                logger.error(f"Circuit breaker check failed: {e}")
                # Fail-safe: reject order if circuit breaker check fails
                return self._reject_order(
                    symbol, side, qty, f"Circuit breaker check error: {e}"
                )

        # 2. Position conflict check
        if self.position_manager and not is_exit_order:
            is_available = await self.position_manager.is_position_available(
                symbol, strategy_name
            )
            if not is_available:
                return self._reject_order(
                    symbol,
                    side,
                    qty,
                    f"Position conflict: another strategy owns or reserved {symbol}",
                )

        # 3. Max positions check
        if max_positions is not None and not is_exit_order:
            try:
                positions = await self.broker.get_positions()
                if hasattr(positions, "__len__") and len(positions) >= max_positions:
                    return self._reject_order(
                        symbol,
                        side,
                        qty,
                        f"Max positions reached ({len(positions)}/{max_positions})",
                    )
            except Exception as e:
                logger.warning(f"Could not check positions: {e}")

        # 4. Risk manager enforcement (if provided and not exit)
        if self.risk_manager and price_history and not is_exit_order:
            try:
                violations = self._check_risk_limits(
                    symbol, qty, price_history, current_positions or {}
                )
                if violations:
                    return self._reject_order(
                        symbol,
                        side,
                        qty,
                        f"Risk limit violated: {', '.join(violations)}",
                    )
            except Exception as e:
                logger.warning(f"Risk check failed: {e}")

        # 5. Reserve position (if position manager available)
        if self.position_manager and not is_exit_order:
            reservation = await self.position_manager.reserve_position(
                symbol, strategy_name, qty, side
            )
            if reservation is None:
                return self._reject_order(
                    symbol, side, qty, "Failed to reserve position"
                )

        # === SUBMIT ORDER ===
        try:
            result = await self._submit_to_broker(order_request)

            if result is None:
                # Order submission failed
                if self.position_manager:
                    await self.position_manager.release_reservation(
                        symbol, strategy_name
                    )
                return self._reject_order(
                    symbol, side, qty, "Broker rejected order"
                )

            # Order submitted successfully
            self._orders_submitted += 1

            # Confirm with position manager
            if self.position_manager and not is_exit_order:
                await self.position_manager.confirm_order_submitted(
                    symbol, strategy_name, str(result.id)
                )

            logger.info(
                f"ORDER SUBMITTED: {side.upper()} {qty} {symbol} "
                f"by {strategy_name} (ID: {result.id})"
            )

            return OrderResult(
                success=True,
                order_id=str(result.id),
                symbol=symbol,
                side=side,
                quantity=qty,
                filled_price=float(result.filled_avg_price)
                if hasattr(result, "filled_avg_price")
                else None,
            )

        except Exception as e:
            # Release reservation on failure
            if self.position_manager:
                await self.position_manager.release_reservation(symbol, strategy_name)

            logger.error(f"Order submission failed: {e}")
            return self._reject_order(symbol, side, qty, f"Submission error: {e}")

    async def submit_exit_order(
        self,
        symbol: str,
        quantity: float,
        strategy_name: str,
        side: str = "sell",
        reason: str = "exit",
    ) -> OrderResult:
        """
        Submit an exit order with relaxed safety checks.

        Exit orders have relaxed checks because:
        - They REDUCE risk (closing positions)
        - Blocking exits during drawdown could make things WORSE

        However, we still:
        - Verify the position exists
        - Log for audit trail
        - Update position manager

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            strategy_name: Name of the strategy
            side: 'sell' for long exit, 'buy' for short exit
            reason: Reason for exit (for logging)

        Returns:
            OrderResult
        """
        from brokers.order_builder import OrderBuilder

        # Verify position exists
        try:
            positions = await self.broker.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)

            if not position:
                logger.warning(f"EXIT REJECTED: No position found for {symbol}")
                return self._reject_order(
                    symbol, side, quantity, "No position to exit"
                )

            actual_qty = abs(float(position.qty))
            if quantity > actual_qty * 1.01:  # 1% tolerance for fractional shares
                logger.warning(
                    f"EXIT ADJUSTED: Requested {quantity} but only have {actual_qty}"
                )
                quantity = actual_qty

        except Exception as e:
            logger.error(f"Could not verify position: {e}")

        # Build and submit order
        order = OrderBuilder(symbol, side, quantity).market().day().build()

        result = await self.submit_order(
            order_request=order,
            strategy_name=strategy_name,
            is_exit_order=True,
        )

        if result.success:
            # Release position in position manager
            if self.position_manager:
                await self.position_manager.release_position(symbol, strategy_name)

            logger.info(
                f"EXIT ORDER: {reason} - {side.upper()} {quantity:.4f} {symbol} "
                f"by {strategy_name}"
            )

        return result

    async def _submit_to_broker(self, order_request) -> Optional[Any]:
        """
        Submit order to broker via authorized internal method.

        Uses _internal_submit_order with gateway token to bypass the
        gateway enforcement check (since we ARE the gateway).

        Args:
            order_request: Order request object

        Returns:
            Order result from broker, or None on failure
        """
        try:
            # Handle both OrderBuilder objects and raw order requests
            if hasattr(order_request, "build"):
                order_request = order_request.build()

            # Use internal method with gateway token if available
            if self._gateway_token and hasattr(self.broker, '_internal_submit_order'):
                result = await self.broker._internal_submit_order(
                    order_request,
                    gateway_token=self._gateway_token
                )
            else:
                # Fallback for brokers without gateway enforcement (e.g., BacktestBroker)
                result = await self.broker.submit_order_advanced(order_request)

            return result

        except Exception as e:
            logger.error(f"Broker submission error: {e}")
            return None

    def _check_risk_limits(
        self,
        symbol: str,
        quantity: float,
        price_history: list,
        current_positions: dict,
    ) -> list:
        """
        Check risk manager limits.

        Returns:
            List of violation reasons (empty if all checks pass)
        """
        violations = []

        if not self.risk_manager:
            return violations

        try:
            # Check if risk manager has enforce_limits method
            if hasattr(self.risk_manager, "enforce_limits"):
                _, violations_dict = self.risk_manager.enforce_limits(
                    symbol, quantity, price_history, current_positions
                )
                violations = list(violations_dict.values())

            # Fallback: check individual limits
            elif hasattr(self.risk_manager, "calculate_position_risk"):
                risk = self.risk_manager.calculate_position_risk(symbol, price_history)
                if risk > 0.8:  # High risk threshold
                    violations.append(f"Position risk too high: {risk:.2f}")

        except Exception as e:
            logger.warning(f"Risk limit check failed: {e}")

        return violations

    def _reject_order(
        self, symbol: str, side: str, quantity: float, reason: str
    ) -> OrderResult:
        """
        Record and return order rejection.

        Args:
            symbol: Stock symbol
            side: Order side
            quantity: Order quantity
            reason: Rejection reason

        Returns:
            OrderResult with success=False
        """
        self._orders_rejected += 1

        # Track rejection reasons
        reason_key = reason.split(":")[0]  # Get base reason
        self._rejection_reasons[reason_key] = (
            self._rejection_reasons.get(reason_key, 0) + 1
        )

        logger.warning(f"ORDER REJECTED: {side.upper()} {quantity} {symbol} - {reason}")

        return OrderResult(
            success=False,
            symbol=symbol,
            side=side,
            quantity=quantity,
            rejection_reason=reason,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get gateway statistics.

        Returns:
            Dict with order counts and rejection breakdown
        """
        total = self._orders_submitted + self._orders_rejected
        return {
            "total_orders": total,
            "orders_submitted": self._orders_submitted,
            "orders_rejected": self._orders_rejected,
            "rejection_rate": self._orders_rejected / total if total > 0 else 0,
            "rejection_reasons": self._rejection_reasons.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"OrderGateway("
            f"submitted={self._orders_submitted}, "
            f"rejected={self._orders_rejected})"
        )
