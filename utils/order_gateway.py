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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from utils.audit_log import AuditEventType, AuditLog, log_order_event
from utils.order_lifecycle import OrderLifecycleTracker

logger = logging.getLogger(__name__)


def _has_real_method(obj: Any, method_name: str) -> bool:
    """
    Check whether a method exists on the object's class.

    This avoids AsyncMock auto-creating arbitrary attributes during hasattr checks.
    """
    return callable(getattr(type(obj), method_name, None))


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
        audit_log: AuditLog | None = None,
        enforce_gateway: bool = True,
        max_intraday_drawdown_pct: Optional[float] = None,
        kill_switch_cooldown_minutes: int = 60,
    ):
        """
        Initialize the order gateway.

        Args:
            broker: Broker instance for order submission
            circuit_breaker: CircuitBreaker instance (optional but recommended)
            position_manager: PositionManager instance (optional but recommended)
            risk_manager: RiskManager instance (optional)
            enforce_gateway: If True, enables mandatory gateway routing (default: True)
            max_intraday_drawdown_pct: Optional drawdown kill-switch threshold
            kill_switch_cooldown_minutes: Halt window after kill switch triggers
        """
        self.broker = broker
        self.circuit_breaker = circuit_breaker
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.audit_log = audit_log
        self.lifecycle_tracker = OrderLifecycleTracker()
        self.max_intraday_drawdown_pct = max_intraday_drawdown_pct
        self.kill_switch_cooldown_minutes = kill_switch_cooldown_minutes

        # Portfolio guardrail state
        self._session_date = None
        self._session_start_equity: Optional[float] = None
        self._session_peak_equity: Optional[float] = None
        self._last_equity: Optional[float] = None
        self._trading_halted_until: Optional[datetime] = None
        self._halt_reason: Optional[str] = None

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
        if _has_real_method(broker, "set_lifecycle_tracker"):
            broker.set_lifecycle_tracker(self.lifecycle_tracker)
        if _has_real_method(broker, "set_position_manager") and self.position_manager:
            broker.set_position_manager(self.position_manager)

        # Statistics
        self._orders_submitted = 0
        self._orders_rejected = 0
        self._duplicate_orders_suppressed = 0
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
            client_order_id = str(
                getattr(order_request, "client_order_id", "") or ""
            ).strip()
            if hasattr(side, "value"):
                side = side.value
        except Exception as e:
            return self._reject_order(
                "", "buy", 0, f"Invalid order request: {e}", strategy_name=strategy_name
            )

        if not symbol or not qty:
            return self._reject_order(
                symbol or "",
                side,
                0,
                "Missing symbol or quantity",
                strategy_name=strategy_name,
            )

        qty = float(qty)

        existing_order_id = None
        if client_order_id:
            existing_order_id = await self._find_existing_order_for_client_id(client_order_id)
            if existing_order_id:
                self._duplicate_orders_suppressed += 1
                logger.warning(
                    "IDEMPOTENT DUPLICATE SUPPRESSED: strategy=%s symbol=%s "
                    "client_order_id=%s existing_order_id=%s",
                    strategy_name,
                    symbol,
                    client_order_id,
                    existing_order_id,
                )
                if self.lifecycle_tracker.get_state(existing_order_id) is None:
                    self.lifecycle_tracker.register_order(
                        order_id=existing_order_id,
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        strategy_name=strategy_name,
                        client_order_id=client_order_id,
                    )
                return OrderResult(
                    success=True,
                    order_id=existing_order_id,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                )

        # === PRE-FLIGHT SAFETY CHECKS ===

        # 1. Circuit breaker check (ATOMIC - uses enforce_before_order)
        guardrail_reason = await self._check_portfolio_guardrails(is_exit_order=is_exit_order)
        if guardrail_reason:
            return self._reject_order(
                symbol,
                side,
                qty,
                guardrail_reason,
                strategy_name=strategy_name,
            )

        # 2. Circuit breaker check (ATOMIC - uses enforce_before_order)
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
                    if e.loss_pct else f"Circuit breaker: {e.reason}",
                    strategy_name=strategy_name,
                )
            except Exception as e:
                logger.error(f"Circuit breaker check failed: {e}")
                # Fail-safe: reject order if circuit breaker check fails
                return self._reject_order(
                    symbol, side, qty, f"Circuit breaker check error: {e}", strategy_name=strategy_name
                )

        # 3. Position conflict check
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
                    strategy_name=strategy_name,
                )

        # 4. Max positions check
        if max_positions is not None and not is_exit_order:
            try:
                positions = await self.broker.get_positions()
                if hasattr(positions, "__len__") and len(positions) >= max_positions:
                    return self._reject_order(
                        symbol,
                        side,
                        qty,
                        f"Max positions reached ({len(positions)}/{max_positions})",
                        strategy_name=strategy_name,
                    )
            except Exception as e:
                logger.warning(f"Could not check positions: {e}")

        # 5. Risk manager enforcement (if provided and not exit)
        if self.risk_manager and price_history and not is_exit_order:
            try:
                violations = self._check_risk_limits(
                    symbol, qty, price_history, current_positions or {}
                )
                if violations:
                    if self.audit_log:
                        self.audit_log.log(
                            AuditEventType.RISK_LIMIT_BREACH,
                            {
                                "symbol": symbol,
                                "side": side,
                                "quantity": qty,
                                "violations": violations,
                                "strategy_name": strategy_name,
                            },
                        )
                    return self._reject_order(
                        symbol,
                        side,
                        qty,
                        f"Risk limit violated: {', '.join(violations)}",
                        strategy_name=strategy_name,
                    )
            except Exception as e:
                logger.warning(f"Risk check failed: {e}")

        # 6. Reserve position (if position manager available)
        if self.position_manager and not is_exit_order:
            reservation = await self.position_manager.reserve_position(
                symbol, strategy_name, qty, side
            )
            if reservation is None:
                return self._reject_order(
                    symbol, side, qty, "Failed to reserve position", strategy_name=strategy_name
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
                    symbol, side, qty, "Broker rejected order", strategy_name=strategy_name
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
            self.lifecycle_tracker.register_order(
                order_id=str(result.id),
                symbol=symbol,
                side=side,
                quantity=qty,
                strategy_name=strategy_name,
                client_order_id=client_order_id or None,
            )
            if _has_real_method(self.broker, "register_order_metadata"):
                self.broker.register_order_metadata(
                    str(result.id),
                    {
                        "strategy_name": strategy_name,
                        "symbol": symbol,
                        "side": side,
                        "quantity": qty,
                    },
                )
            if _has_real_method(self.broker, "track_order_for_fills"):
                self.broker.track_order_for_fills(str(result.id), symbol, side, qty)
            if self.audit_log:
                log_order_event(
                    self.audit_log,
                    AuditEventType.ORDER_SUBMITTED,
                    order_id=str(result.id),
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    price=float(result.filled_avg_price)
                    if hasattr(result, "filled_avg_price")
                    else None,
                    strategy_name=strategy_name,
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
            return self._reject_order(
                symbol, side, qty, f"Submission error: {e}", strategy_name=strategy_name
            )

    async def _find_existing_order_for_client_id(self, client_order_id: str) -> Optional[str]:
        """
        Resolve existing order ID for a client order ID.

        Checks persisted lifecycle state first, then broker lookup (if supported).
        """
        normalized = str(client_order_id or "").strip()
        if not normalized:
            return None

        lifecycle_state = self.lifecycle_tracker.export_state()
        for order_id, record in lifecycle_state.items():
            if str(record.get("client_order_id", "") or "").strip() != normalized:
                continue
            return str(record.get("order_id") or order_id)

        if not _has_real_method(self.broker, "get_order_by_client_id"):
            return None

        try:
            existing = await self.broker.get_order_by_client_id(normalized)
        except Exception as e:
            logger.debug(
                "Client-order lookup failed for %s: %s",
                normalized,
                e,
            )
            return None

        if existing is None:
            return None

        existing_order_id = getattr(existing, "id", None) or getattr(existing, "order_id", None)
        if not existing_order_id:
            return None
        return str(existing_order_id)

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
            if self.audit_log:
                self.audit_log.log(
                    AuditEventType.POSITION_CLOSED,
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "reason": reason,
                        "strategy_name": strategy_name,
                    },
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

    async def _check_portfolio_guardrails(self, is_exit_order: bool = False) -> Optional[str]:
        """
        Apply portfolio-level kill switch based on intraday drawdown.

        Exit orders bypass this check by design.
        """
        if is_exit_order:
            return None

        now = datetime.now()
        if self._trading_halted_until and now < self._trading_halted_until:
            return (
                f"Portfolio kill switch active until {self._trading_halted_until.isoformat()} "
                f"({self._halt_reason or 'drawdown breach'})"
            )

        if self.max_intraday_drawdown_pct is None:
            return None

        equity = await self._get_current_equity()
        if equity is None or equity <= 0:
            return None

        if self._session_date != now.date():
            self._session_date = now.date()
            self._session_start_equity = equity
            self._session_peak_equity = equity
            self._trading_halted_until = None
            self._halt_reason = None
        else:
            self._session_peak_equity = (
                max(self._session_peak_equity, equity)
                if self._session_peak_equity is not None
                else equity
            )

        self._last_equity = equity

        if not self._session_peak_equity or self._session_peak_equity <= 0:
            return None

        drawdown = (self._session_peak_equity - equity) / self._session_peak_equity
        if drawdown >= self.max_intraday_drawdown_pct:
            self._trading_halted_until = now + timedelta(minutes=self.kill_switch_cooldown_minutes)
            self._halt_reason = (
                f"intraday drawdown {drawdown:.2%} exceeded threshold "
                f"{self.max_intraday_drawdown_pct:.2%}"
            )
            logger.error(f"KILL SWITCH TRIGGERED: {self._halt_reason}")
            if self.audit_log:
                self.audit_log.log(
                    AuditEventType.RISK_LIMIT_BREACH,
                    {
                        "type": "portfolio_kill_switch",
                        "drawdown": drawdown,
                        "threshold": self.max_intraday_drawdown_pct,
                        "halted_until": self._trading_halted_until.isoformat(),
                    },
                )
            return (
                f"Portfolio kill switch triggered: {self._halt_reason}. "
                f"Halted until {self._trading_halted_until.isoformat()}"
            )

        return None

    def activate_kill_switch(
        self,
        reason: str,
        cooldown_minutes: Optional[int] = None,
        source: str = "external",
    ) -> None:
        """
        Trigger or extend the portfolio kill switch from external safety systems.

        Args:
            reason: Human-readable halt reason
            cooldown_minutes: Halt duration override (defaults to configured cooldown)
            source: Origin of trigger (e.g. order_reconciliation)
        """
        now = datetime.now()
        halt_minutes = cooldown_minutes or self.kill_switch_cooldown_minutes
        requested_until = now + timedelta(minutes=halt_minutes)

        if (
            self._trading_halted_until
            and self._trading_halted_until > now
            and self._halt_reason == reason
        ):
            return

        if not (self._trading_halted_until and self._trading_halted_until > requested_until):
            self._trading_halted_until = requested_until

        self._halt_reason = reason
        logger.error(
            "KILL SWITCH ACTIVATED (%s): %s. Halted until %s",
            source,
            reason,
            self._trading_halted_until.isoformat(),
        )

        if self.audit_log:
            self.audit_log.log(
                AuditEventType.RISK_LIMIT_BREACH,
                {
                    "type": "portfolio_kill_switch",
                    "source": source,
                    "reason": reason,
                    "halted_until": self._trading_halted_until.isoformat(),
                },
            )

    async def _get_current_equity(self) -> Optional[float]:
        """Best-effort retrieval of current equity for guardrails."""
        try:
            if hasattr(self.broker, "get_account"):
                account = await self.broker.get_account()
                for attr in ("equity", "portfolio_value", "cash"):
                    val = getattr(account, attr, None)
                    if val is not None:
                        try:
                            return float(val)
                        except (TypeError, ValueError):
                            continue
            if hasattr(self.broker, "get_portfolio_value"):
                value = self.broker.get_portfolio_value()
                return float(value)
        except Exception as e:
            logger.debug(f"Failed to retrieve equity for guardrails: {e}")
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
        self,
        symbol: str,
        side: str,
        quantity: float,
        reason: str,
        strategy_name: str | None = None,
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
        if self.audit_log:
            log_order_event(
                self.audit_log,
                AuditEventType.ORDER_REJECTED,
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                rejection_reason=reason,
                strategy_name=strategy_name,
            )

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
            "duplicate_orders_suppressed": self._duplicate_orders_suppressed,
            "rejection_rate": self._orders_rejected / total if total > 0 else 0,
            "rejection_reasons": self._rejection_reasons.copy(),
            "guardrails": {
                "max_intraday_drawdown_pct": self.max_intraday_drawdown_pct,
                "kill_switch_cooldown_minutes": self.kill_switch_cooldown_minutes,
                "session_start_equity": self._session_start_equity,
                "session_peak_equity": self._session_peak_equity,
                "last_equity": self._last_equity,
                "trading_halted_until": (
                    self._trading_halted_until.isoformat() if self._trading_halted_until else None
                ),
                "halt_reason": self._halt_reason,
            },
        }

    def export_runtime_state(self) -> Dict[str, Any]:
        """
        Export gateway state required for restart-safe recovery.
        """
        return {
            "session_date": self._session_date.isoformat() if self._session_date else None,
            "session_start_equity": self._session_start_equity,
            "session_peak_equity": self._session_peak_equity,
            "last_equity": self._last_equity,
            "trading_halted_until": (
                self._trading_halted_until.isoformat() if self._trading_halted_until else None
            ),
            "halt_reason": self._halt_reason,
            "orders_submitted": self._orders_submitted,
            "orders_rejected": self._orders_rejected,
            "duplicate_orders_suppressed": self._duplicate_orders_suppressed,
            "rejection_reasons": dict(self._rejection_reasons),
        }

    def import_runtime_state(self, state: Dict[str, Any] | None) -> None:
        """
        Restore gateway runtime state across process restarts.
        """
        if not isinstance(state, dict):
            return

        session_date = state.get("session_date")
        if isinstance(session_date, str):
            try:
                self._session_date = datetime.fromisoformat(session_date).date()
            except ValueError:
                self._session_date = None
        else:
            self._session_date = None

        self._session_start_equity = state.get("session_start_equity")
        self._session_peak_equity = state.get("session_peak_equity")
        self._last_equity = state.get("last_equity")
        self._halt_reason = state.get("halt_reason")

        halted_until = state.get("trading_halted_until")
        if isinstance(halted_until, str):
            try:
                self._trading_halted_until = datetime.fromisoformat(halted_until)
            except ValueError:
                self._trading_halted_until = None
        else:
            self._trading_halted_until = None

        try:
            self._orders_submitted = int(state.get("orders_submitted", self._orders_submitted))
        except (TypeError, ValueError):
            pass
        try:
            self._orders_rejected = int(state.get("orders_rejected", self._orders_rejected))
        except (TypeError, ValueError):
            pass
        try:
            self._duplicate_orders_suppressed = int(
                state.get("duplicate_orders_suppressed", self._duplicate_orders_suppressed)
            )
        except (TypeError, ValueError):
            pass

        rejection_reasons = state.get("rejection_reasons")
        if isinstance(rejection_reasons, dict):
            normalized: Dict[str, int] = {}
            for reason, count in rejection_reasons.items():
                try:
                    normalized[str(reason)] = int(count)
                except (TypeError, ValueError):
                    continue
            self._rejection_reasons = normalized

    def __repr__(self) -> str:
        return (
            f"OrderGateway("
            f"submitted={self._orders_submitted}, "
            f"rejected={self._orders_rejected})"
        )
