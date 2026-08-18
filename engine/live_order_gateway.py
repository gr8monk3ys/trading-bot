"""Live-path OrderGateway: the single choke point for real broker orders.

The 2026-05 cleanup removed the previous production gateway but left both
halves of its safety contract in place: `BaseStrategy.submit_entry_order` /
`submit_exit_order` refuse to run without a gateway, and `AlpacaBroker`
raises `GatewayBypassError` from its public submit methods once enforcement
is enabled. The net effect was that the live path could not place any order
at all (every file in `logs/` from that era is empty).

`LiveOrderGateway` restores the path and, in the same stroke, wires in the
circuit breaker's per-order interlock (`enforce_before_order`), which had no
production caller before this. Entries are rejected while the breaker is
tripped; exits pass `is_exit_order=True` so closing risk stays possible when
event blocking is active.

`StrategyManager.start_strategy` attaches one shared instance to every
strategy it constructs; the backtest engine attaches its own
`BacktestOrderGateway` instead and is unaffected.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from engine.backtest_order_gateway import OrderResult
from utils.circuit_breaker import TradingHaltedException

logger = logging.getLogger(__name__)


class LiveOrderGateway:
    """Forward strategy orders to a live broker under circuit-breaker control.

    Implements the two coroutine entry points `BaseStrategy` calls
    (`submit_order` for entries, `submit_exit_order` for exits) with the same
    `OrderResult` surface as `BacktestOrderGateway`. On construction it claims
    the broker's gateway authorization token, which simultaneously locks the
    broker's public submit methods against direct (un-checked) use.
    """

    def __init__(self, broker: Any, circuit_breaker: Any = None) -> None:
        self.broker = broker
        self.circuit_breaker = circuit_breaker
        if hasattr(broker, "enable_gateway_requirement"):
            self._gateway_token: Optional[str] = broker.enable_gateway_requirement()
        else:
            self._gateway_token = None

    async def _halt_reason(self, *, is_exit_order: bool) -> Optional[str]:
        """Run the breaker interlock; return a rejection reason if halted."""
        if self.circuit_breaker is None:
            return None
        try:
            await self.circuit_breaker.enforce_before_order(is_exit_order=is_exit_order)
        except TradingHaltedException as e:
            return str(e)
        return None

    async def _submit(self, order_request: Any, *, side_hint: str) -> OrderResult:
        try:
            order = await self.broker._internal_submit_order(
                order_request, gateway_token=self._gateway_token
            )
        except Exception as e:
            logger.error(f"Live order submission failed: {e}")
            return OrderResult(success=False, rejection_reason=str(e))
        if order is None:
            return OrderResult(success=False, rejection_reason="broker_returned_none")
        return OrderResult(
            success=True,
            order_id=str(getattr(order, "id", "")),
            side=str(getattr(order, "side", side_hint)),
            quantity=float(getattr(order, "filled_qty", 0) or getattr(order, "qty", 0)),
        )

    async def submit_order(
        self,
        *,
        order_request: Any,
        strategy_name: str,
        max_positions: Optional[int] = None,
        price_history: Any = None,
        is_exit_order: bool = False,
    ) -> OrderResult:
        halt = await self._halt_reason(is_exit_order=is_exit_order)
        if halt is not None:
            logger.warning(f"Entry order for {strategy_name} rejected by circuit breaker: {halt}")
            return OrderResult(success=False, rejection_reason=halt)
        return await self._submit(order_request, side_hint=getattr(order_request, "side", ""))

    async def submit_exit_order(
        self,
        *,
        symbol: str,
        quantity: float,
        strategy_name: str,
        side: str = "sell",
        reason: str = "exit",
    ) -> OrderResult:
        halt = await self._halt_reason(is_exit_order=True)
        if halt is not None:
            logger.warning(f"Exit order for {symbol} rejected by circuit breaker: {halt}")
            return OrderResult(success=False, rejection_reason=halt)

        # Import inside the method to avoid circular imports (repo convention).
        from brokers.order_builder import OrderBuilder

        order_request = OrderBuilder(symbol, side, quantity).market().day().build()
        return await self._submit(order_request, side_hint=side)
