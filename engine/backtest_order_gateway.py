"""Canonical OrderGateway implementation for the backtest engine.

PR #22 made gateway routing mandatory for ALL entry/exit orders
(`BaseStrategy.submit_entry_order` and `submit_exit_order` log
"No OrderGateway configured" and return None when `self.order_gateway`
is missing). That guard exists for live trading safety, but the backtest
engine doesn't have a real live-path `OrderGateway` (the production gateway
module was removed in the 2026-05 honest cleanup). Without this shim, every
backtest order is blocked and the backtest produces 0 trades that look
identical to a data-fetch failure.

`BacktestOrderGateway` restores the backtest path by forwarding orders to
the broker's `submit_order_advanced` (for entries) / `place_order` (for
exits), exposing the success/side/quantity/order_id surface that
`BaseStrategy` expects. It enforces no extra safety because the backtest
broker is already a sandboxed in-memory ledger; risk checks and position
limits in this code path would be redundant and the wrong place for them.

`BacktestEngine` attaches an instance of this gateway to every strategy it
constructs, so backtest authors do not need to wire this up themselves.
"""

from __future__ import annotations

from typing import Any, Optional


class OrderResult:
    """Duck-type stand-in for the OrderGateway result expected by `BaseStrategy`.

    `BaseStrategy.submit_entry_order` / `submit_exit_order` consume the
    `.success`, `.order_id`, `.side`, `.quantity`, and `.rejection_reason`
    attributes. Mirror that surface exactly.
    """

    def __init__(
        self,
        success: bool,
        order_id: Optional[str] = None,
        side: Optional[str] = None,
        quantity: Optional[float] = None,
        rejection_reason: Optional[str] = None,
    ) -> None:
        self.success = success
        self.order_id = order_id or ""
        self.side = side or ""
        self.quantity = quantity or 0
        self.rejection_reason = rejection_reason or ""


class BacktestOrderGateway:
    """Minimal OrderGateway implementation that forwards orders to a backtest broker.

    The class only implements the two coroutine entry points that
    `BaseStrategy` calls (`submit_order` for entries, `submit_exit_order`
    for exits) and forwards each to the underlying `BacktestBroker`. It
    deliberately adds no risk checks, position limits, or kill-switch
    behavior: the backtest broker is a sandboxed in-memory ledger and any
    extra safety in this code path would be redundant.
    """

    def __init__(self, broker: Any) -> None:
        self.broker = broker

    async def submit_order(
        self,
        *,
        order_request: Any,
        strategy_name: str,
        max_positions: Optional[int] = None,
        price_history: Any = None,
        is_exit_order: bool = False,
    ) -> OrderResult:
        mock_order = await self.broker.submit_order_advanced(order_request)
        if mock_order is None:
            return OrderResult(success=False, rejection_reason="broker_returned_none")
        side = getattr(mock_order, "side", "")
        return OrderResult(
            success=True,
            order_id=str(getattr(mock_order, "id", "")),
            side=str(side),
            quantity=float(
                getattr(mock_order, "filled_qty", 0) or getattr(mock_order, "qty", 0)
            ),
        )

    async def submit_exit_order(
        self,
        *,
        symbol: str,
        quantity: float,
        strategy_name: str,
        side: str = "sell",
        reason: str = "exit",
    ) -> OrderResult:
        # BacktestBroker.place_order is synchronous on this codepath.
        result = self.broker.place_order(symbol, int(quantity), side, order_type="market")
        if not result:
            return OrderResult(
                success=False, rejection_reason="broker_place_order_failed"
            )
        return OrderResult(
            success=True,
            order_id=str(result.get("id", "")),
            side=str(result.get("side", side)),
            quantity=float(
                result.get("filled_qty", 0) or result.get("quantity", quantity)
            ),
        )
