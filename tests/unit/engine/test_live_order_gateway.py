"""Tests for LiveOrderGateway — the live-path order choke point.

The 2026-05 cleanup deleted the production OrderGateway but kept the guard
in BaseStrategy (entries/exits blocked without a gateway) and the token lock
in AlpacaBroker (public submit methods raise GatewayBypassError once
enforcement is on). The live path therefore could not place a single order.

LiveOrderGateway restores it: it claims the broker's gateway token, runs the
circuit breaker's per-order interlock (enforce_before_order — dead code until
now), and forwards to broker._internal_submit_order.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from utils.circuit_breaker import TradingHaltedException


def _live_broker():
    broker = MagicMock()
    broker.enable_gateway_requirement = MagicMock(return_value="tok-123")
    broker._internal_submit_order = AsyncMock(
        return_value=SimpleNamespace(id="order-1", side="buy", filled_qty=10, qty=10)
    )
    return broker


def _order_request(symbol="SPY", side="buy", qty=10):
    return SimpleNamespace(symbol=symbol, side=side, qty=qty, type="market")


async def test_entry_order_routes_through_internal_submit_with_claimed_token():
    from engine.live_order_gateway import LiveOrderGateway

    broker = _live_broker()
    gateway = LiveOrderGateway(broker=broker)

    result = await gateway.submit_order(order_request=_order_request(), strategy_name="momentum")

    broker.enable_gateway_requirement.assert_called_once()
    broker._internal_submit_order.assert_awaited_once()
    assert broker._internal_submit_order.await_args.kwargs["gateway_token"] == "tok-123"
    assert result.success is True
    assert result.order_id == "order-1"
    assert result.side == "buy"
    assert result.quantity == 10


async def test_entry_order_rejected_when_circuit_breaker_halts():
    from engine.live_order_gateway import LiveOrderGateway

    broker = _live_broker()
    breaker = MagicMock()
    breaker.enforce_before_order = AsyncMock(side_effect=TradingHaltedException("daily loss limit"))
    gateway = LiveOrderGateway(broker=broker, circuit_breaker=breaker)

    result = await gateway.submit_order(order_request=_order_request(), strategy_name="momentum")

    breaker.enforce_before_order.assert_awaited_once_with(is_exit_order=False)
    broker._internal_submit_order.assert_not_awaited()
    assert result.success is False
    assert "daily loss limit" in result.rejection_reason


async def test_exit_order_passes_is_exit_flag_to_breaker_and_submits_market_sell():
    from engine.live_order_gateway import LiveOrderGateway

    broker = _live_broker()
    broker._internal_submit_order = AsyncMock(
        return_value=SimpleNamespace(id="order-2", side="sell", filled_qty=25, qty=25)
    )
    breaker = MagicMock()
    breaker.enforce_before_order = AsyncMock(return_value=None)
    gateway = LiveOrderGateway(broker=broker, circuit_breaker=breaker)

    result = await gateway.submit_exit_order(
        symbol="SPY", quantity=25, strategy_name="momentum", side="sell", reason="stop"
    )

    breaker.enforce_before_order.assert_awaited_once_with(is_exit_order=True)
    submitted = broker._internal_submit_order.await_args.args[0]
    assert getattr(submitted, "symbol", None) == "SPY"
    assert result.success is True
    assert result.quantity == 25


async def test_broker_without_token_protocol_still_works():
    from engine.live_order_gateway import LiveOrderGateway

    broker = MagicMock(spec=[])
    broker._internal_submit_order = AsyncMock(
        return_value=SimpleNamespace(id="order-3", side="buy", filled_qty=5, qty=5)
    )
    gateway = LiveOrderGateway(broker=broker)

    result = await gateway.submit_order(
        order_request=_order_request(qty=5), strategy_name="momentum"
    )

    assert result.success is True
    assert broker._internal_submit_order.await_args.kwargs["gateway_token"] is None


async def test_strategy_manager_attaches_live_gateway_to_new_strategies():
    from engine.live_order_gateway import LiveOrderGateway
    from engine.strategy_manager import StrategyManager

    broker = _live_broker()
    manager = StrategyManager(broker=broker, circuit_breaker=MagicMock())

    class FakeStrategy:
        NAME = "fake"

        def __init__(self, broker=None, parameters=None):
            self.broker = broker
            self.parameters = parameters or {}
            self.order_gateway = None

        async def initialize(self, **kwargs):
            return True

        async def start(self):
            return True

    manager.available_strategies = {"fake": FakeStrategy}
    started = await manager.start_strategy("fake", symbols=["SPY"], allocation=0.5)

    assert started is True
    strategy = manager.active_strategies["fake"]
    assert isinstance(strategy.order_gateway, LiveOrderGateway)
    assert strategy.order_gateway.circuit_breaker is manager.circuit_breaker


async def test_broker_error_returns_failed_result_instead_of_raising():
    from engine.live_order_gateway import LiveOrderGateway

    broker = _live_broker()
    broker._internal_submit_order = AsyncMock(side_effect=RuntimeError("api down"))
    gateway = LiveOrderGateway(broker=broker)

    result = await gateway.submit_order(order_request=_order_request(), strategy_name="momentum")

    assert result.success is False
    assert "api down" in result.rejection_reason
