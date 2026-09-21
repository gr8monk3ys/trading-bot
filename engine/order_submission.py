"""Order submission: the one act of turning an order intent into an order outcome.

There is exactly one of these. Live and backtest differ only in the broker
adapter behind it. It owns the sequence a strategy used to orchestrate by
hand across six modules: halt gate → build → dispatch → normalise the reply
→ register protective levels → record. See docs/adr/0004-rejected-is-an-outcome.md.

Sizing still happens in the strategy for now (arch 6 moves it here).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from brokers.order_builder import OrderBuilder
from utils.circuit_breaker import TradingHaltedException

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    ACCEPTED = "accepted"  # resting at the broker (live limit orders, etc.)
    REJECTED = "rejected"
    HALTED = "halted"  # refused here by the circuit breaker; never reached the broker


@dataclass(frozen=True)
class OrderIntent:
    """What a strategy asks for. Not an order until submitted."""

    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    is_exit: bool = False
    order_type: str = "market"  # "market" | "limit"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: str = "day"  # "day" | "gtc"
    strategy_name: str = ""
    reason: str = ""


@dataclass(frozen=True)
class OrderOutcome:
    """What submission returned. Rejected and halted are outcomes, not fills of zero."""

    status: OrderStatus
    symbol: str
    side: str
    qty_requested: float
    qty_filled: float = 0.0
    order_id: str = ""
    fill_price: Optional[float] = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIAL, OrderStatus.ACCEPTED)


class OrderSubmission:
    """Submit order intents to a broker under circuit-breaker control.

    ``broker`` satisfies ``brokers.protocol.Broker``. On construction it
    claims the Alpaca gateway token when the broker offers one, which locks
    the broker's public submit methods against un-gated use.
    """

    def __init__(self, broker: Any, *, circuit_breaker: Any = None, audit_log: Any = None):
        self.broker = broker
        self.circuit_breaker = circuit_breaker
        self.audit_log = audit_log
        self.history: List[OrderOutcome] = []
        enable = getattr(broker, "enable_gateway_requirement", None)
        self._gateway_token: Optional[str] = enable() if callable(enable) else None

    # -- the interface ---------------------------------------------------------

    async def submit(self, intent: OrderIntent) -> OrderOutcome:
        outcome = await self._submit(intent)
        self.history.append(outcome)
        return outcome

    # -- implementation --------------------------------------------------------

    async def _submit(self, intent: OrderIntent) -> OrderOutcome:
        if intent.qty is None or intent.qty <= 0:
            return self._refuse(intent, OrderStatus.REJECTED, "zero_quantity")

        halt = await self._halt_reason(is_exit_order=intent.is_exit)
        if halt is not None:
            kind = "Exit" if intent.is_exit else "Entry"
            logger.warning(f"{kind} order for {intent.symbol} halted by circuit breaker: {halt}")
            return self._refuse(intent, OrderStatus.HALTED, halt)

        try:
            request = self._build_request(intent)
            reply = await self._dispatch(request)
        except Exception as e:  # broker/network/validation errors are outcomes, not crashes
            logger.error(f"Order submission failed for {intent.symbol}: {e}")
            return self._refuse(intent, OrderStatus.REJECTED, str(e))

        outcome = self._normalise(intent, reply)
        if outcome.ok and outcome.qty_filled > 0:
            self._register_protective_levels(intent, outcome)
        if outcome.status is OrderStatus.REJECTED:
            logger.warning(f"Order for {intent.symbol} rejected: {outcome.reason}")
        return outcome

    async def _halt_reason(self, *, is_exit_order: bool) -> Optional[str]:
        if self.circuit_breaker is None:
            return None
        try:
            await self.circuit_breaker.enforce_before_order(is_exit_order=is_exit_order)
        except TradingHaltedException as e:
            return str(e)
        return None

    def _build_request(self, intent: OrderIntent) -> Any:
        builder = OrderBuilder(intent.symbol, intent.side, intent.qty)
        if intent.order_type == "limit":
            if intent.limit_price is None:
                raise ValueError("limit order needs a limit_price")
            builder = builder.limit(intent.limit_price)
        else:
            builder = builder.market()
        # Alpaca crypto has no bracket order class; the strategy manages those exits.
        if (
            not intent.is_exit
            and intent.stop_loss is not None
            and intent.take_profit is not None
            and not _is_crypto(builder)
        ):
            builder = builder.bracket(take_profit=intent.take_profit, stop_loss=intent.stop_loss)
        builder = builder.gtc() if intent.time_in_force == "gtc" else builder.day()
        return builder.build()

    async def _dispatch(self, request: Any) -> Any:
        internal = getattr(self.broker, "_internal_submit_order", None)
        if self._gateway_token is not None and callable(internal):
            return await internal(request, gateway_token=self._gateway_token)
        return await self.broker.submit_order_advanced(request)

    def _normalise(self, intent: OrderIntent, reply: Any) -> OrderOutcome:
        if reply is None:
            return self._refuse(intent, OrderStatus.REJECTED, "broker_returned_none")
        status_text = str(getattr(reply, "status", "") or "").lower()
        order_id = str(getattr(reply, "id", "") or "")
        filled = _to_float(getattr(reply, "filled_qty", 0))
        requested = _to_float(getattr(reply, "qty", intent.qty)) or intent.qty
        price = getattr(reply, "filled_avg_price", None)
        fill_price = _to_float(price) if price not in (None, "", "None") else None
        side = str(getattr(reply, "side", intent.side) or intent.side)
        if "reject" in status_text or "cancel" in status_text or "expired" in status_text:
            reason = str(getattr(reply, "rejection_reason", "") or status_text)
            return OrderOutcome(
                OrderStatus.REJECTED, intent.symbol, side, requested, 0.0, order_id, None, reason
            )
        if filled > 0 and filled + 1e-9 >= requested:
            status = OrderStatus.FILLED
        elif filled > 0:
            status = OrderStatus.PARTIAL
        else:
            status = OrderStatus.ACCEPTED
        return OrderOutcome(status, intent.symbol, side, requested, filled, order_id, fill_price)

    def _register_protective_levels(self, intent: OrderIntent, outcome: OrderOutcome) -> None:
        """Backtest broker: the bracket legs are simulated as a stop order.

        Alpaca carries the legs on the bracket itself, so this is a no-op there.
        """
        if intent.is_exit or intent.stop_loss is None:
            return
        register = getattr(self.broker, "set_stop_order", None)
        if not callable(register):
            return
        exit_side = "sell" if intent.side.lower().startswith("buy") else "buy"
        register(
            intent.symbol, stop_price=intent.stop_loss, quantity=outcome.qty_filled, side=exit_side
        )

    def _refuse(self, intent: OrderIntent, status: OrderStatus, reason: str) -> OrderOutcome:
        outcome = OrderOutcome(
            status, intent.symbol, intent.side, float(intent.qty or 0), 0.0, "", None, reason
        )
        if self.audit_log is not None and status is OrderStatus.HALTED:
            try:
                from utils.audit_log import AuditEventType

                self.audit_log.log(
                    AuditEventType.ORDER_REJECTED,
                    {
                        "symbol": intent.symbol,
                        "side": intent.side,
                        "qty": intent.qty,
                        "reason": reason,
                        "strategy": intent.strategy_name,
                        "halted": True,
                    },
                )
            except Exception as e:  # audit must never block trading decisions
                logger.error(f"Audit log write failed: {e}")
        return outcome


def _is_crypto(builder: OrderBuilder) -> bool:
    flag = getattr(builder, "is_crypto", False)
    return bool(flag() if callable(flag) else flag)


def _to_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
