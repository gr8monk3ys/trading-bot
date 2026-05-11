"""Tests for the canonical backtest OrderGateway and its wiring into BacktestEngine.

Background: PR #22 made `BaseStrategy.submit_entry_order` /
`submit_exit_order` require an `order_gateway` attribute. Before the
fix this file accompanies, `BacktestEngine` did not construct one, so
every backtest order was being silently rejected with "No OrderGateway
configured" — producing 0 trades indistinguishable from a data-fetch
failure. These tests pin that the engine attaches a `BacktestOrderGateway`
to strategies it instantiates and that orders flow end-to-end without
the historical shim.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from brokers.order_builder import OrderBuilder
from engine.backtest_engine import BacktestEngine
from engine.backtest_order_gateway import BacktestOrderGateway, OrderResult
from strategies.base_strategy import BaseStrategy


# =============================================================================
# UNIT TESTS: BacktestOrderGateway directly
# =============================================================================


class _StubBacktestBroker:
    """Minimal broker stub exposing only the surface BacktestOrderGateway uses."""

    def __init__(self) -> None:
        self.submit_calls = []
        self.place_calls = []

    async def submit_order_advanced(self, order_request):
        self.submit_calls.append(order_request)
        order = MagicMock()
        order.id = "stub-order-1"
        order.side = "buy"
        order.qty = 5
        order.filled_qty = 5
        return order

    def place_order(self, symbol, quantity, side, order_type="market"):
        self.place_calls.append((symbol, quantity, side, order_type))
        return {
            "id": "stub-exit-1",
            "side": side,
            "quantity": quantity,
            "filled_qty": quantity,
        }


@pytest.mark.asyncio
async def test_gateway_submit_order_forwards_to_broker():
    broker = _StubBacktestBroker()
    gateway = BacktestOrderGateway(broker=broker)

    order_request = OrderBuilder("AAPL", "buy", 5).market().day().build()
    result = await gateway.submit_order(
        order_request=order_request,
        strategy_name="UnitTest",
    )

    assert isinstance(result, OrderResult)
    assert result.success is True
    assert result.order_id == "stub-order-1"
    assert result.quantity == 5
    assert len(broker.submit_calls) == 1


@pytest.mark.asyncio
async def test_gateway_submit_order_handles_broker_returning_none():
    broker = _StubBacktestBroker()

    async def returns_none(_request):
        return None

    broker.submit_order_advanced = returns_none  # type: ignore[assignment]
    gateway = BacktestOrderGateway(broker=broker)

    order_request = OrderBuilder("AAPL", "buy", 5).market().day().build()
    result = await gateway.submit_order(
        order_request=order_request,
        strategy_name="UnitTest",
    )

    assert result.success is False
    assert result.rejection_reason == "broker_returned_none"


@pytest.mark.asyncio
async def test_gateway_submit_exit_order_forwards_to_place_order():
    broker = _StubBacktestBroker()
    gateway = BacktestOrderGateway(broker=broker)

    result = await gateway.submit_exit_order(
        symbol="AAPL",
        quantity=3,
        strategy_name="UnitTest",
        side="sell",
        reason="exit",
    )

    assert result.success is True
    assert result.order_id == "stub-exit-1"
    assert result.quantity == 3
    assert broker.place_calls == [("AAPL", 3, "sell", "market")]


# =============================================================================
# INTEGRATION TEST: BacktestEngine attaches a gateway to strategies
# =============================================================================


class _GatewayProbeStrategy(BaseStrategy):
    """Minimal strategy used to assert the engine wires a gateway.

    Submits a buy order via `submit_entry_order` on the first iteration.
    Without an attached gateway, `BaseStrategy.submit_entry_order` logs
    "No OrderGateway configured" and returns None.
    """

    NAME = "GatewayProbeStrategy"
    captured_gateway = None
    submit_attempts = 0
    submit_successes = 0

    async def initialize(self, **kwargs):
        # Don't call super().initialize() — base initialize spins up tasks.
        type(self).captured_gateway = self.order_gateway

    async def analyze_symbol(self, symbol):
        return {"action": "buy", "symbol": symbol}

    async def execute_trade(self, symbol, signal):
        type(self).submit_attempts += 1
        order_request = OrderBuilder(symbol, "buy", 1).market().day().build()
        result = await self.submit_entry_order(
            order_request=order_request,
            reason="probe",
        )
        if result is not None and getattr(result, "success", False):
            type(self).submit_successes += 1


def _make_bars(symbol: str, start: datetime, n: int = 10) -> list:
    """Build a list of bar-like objects the engine can consume."""

    class Bar:
        def __init__(self, ts, open_, high, low, close, volume):
            self.timestamp = ts
            self.open = open_
            self.high = high
            self.low = low
            self.close = close
            self.volume = volume

    bars = []
    base_price = 100.0
    for i in range(n):
        ts = pd.Timestamp(start) + pd.Timedelta(days=i)
        # Trend up so a momentum-ish signal could fire; for this probe
        # strategy any data is fine — we just need the engine to iterate.
        close = base_price + i
        bars.append(Bar(ts, close - 0.5, close + 0.5, close - 1.0, close, 1_000_000))
    return bars


@pytest.mark.asyncio
async def test_engine_attaches_backtest_order_gateway_to_strategy():
    """The engine must attach a BacktestOrderGateway when constructing strategies."""
    # Reset class-level state.
    _GatewayProbeStrategy.captured_gateway = None
    _GatewayProbeStrategy.submit_attempts = 0
    _GatewayProbeStrategy.submit_successes = 0

    data_broker = MagicMock()
    data_broker.get_bars = AsyncMock(
        return_value=_make_bars("AAPL", datetime(2024, 1, 2), n=10)
    )

    engine = BacktestEngine(broker=data_broker)
    result = await engine.run_backtest(
        strategy_class=_GatewayProbeStrategy,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 1, 12),
        initial_capital=100_000,
    )

    # The gateway must have been attached *before* initialize ran.
    assert _GatewayProbeStrategy.captured_gateway is not None, (
        "BacktestEngine did not attach an order_gateway before strategy.initialize()"
    )
    assert isinstance(_GatewayProbeStrategy.captured_gateway, BacktestOrderGateway)

    # The engine must produce a result dict, and the strategy must have at
    # least attempted to submit orders (the engine iterates over loaded
    # symbols/days). Without a gateway, every attempt would have been
    # rejected with the "No OrderGateway configured" log and returned None.
    assert isinstance(result, dict)
    assert _GatewayProbeStrategy.submit_attempts > 0, (
        "Engine never called execute_trade — test setup is wrong"
    )
    assert _GatewayProbeStrategy.submit_successes > 0, (
        "submit_entry_order returned None for every attempt — the gateway "
        "is not wired correctly"
    )


@pytest.mark.asyncio
async def test_engine_backtest_records_trades_through_gateway():
    """End-to-end: orders submitted through the gateway must reach the broker.

    Uses the simple `_GatewayProbeStrategy` (which always emits a buy
    signal) to prove that orders submitted via `BaseStrategy.submit_entry_order`
    are forwarded all the way to the backtest broker — i.e. the gateway
    chain works end to end, not just at the attachment site.

    Pre-fix this would have produced 0 trades because the gateway gate
    in BaseStrategy blocked every submission with "No OrderGateway
    configured".
    """
    _GatewayProbeStrategy.captured_gateway = None
    _GatewayProbeStrategy.submit_attempts = 0
    _GatewayProbeStrategy.submit_successes = 0

    data_broker = MagicMock()
    data_broker.get_bars = AsyncMock(
        return_value=_make_bars("AAPL", datetime(2024, 1, 2), n=10)
    )

    engine = BacktestEngine(broker=data_broker)
    result = await engine.run_backtest(
        strategy_class=_GatewayProbeStrategy,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 1, 12),
        initial_capital=100_000,
    )

    # The backtest must produce at least one recorded trade. Any trade
    # count > 0 demonstrates the order flowed from the strategy through
    # BaseStrategy.submit_entry_order through BacktestOrderGateway and into
    # the backtest broker's ledger, without being blocked by the
    # "No OrderGateway configured" gate.
    assert isinstance(result, dict)
    assert "trades" in result
    assert len(result["trades"]) > 0, (
        "Engine ran without recording any trades — orders are not flowing "
        "through the gateway. submit_attempts="
        f"{_GatewayProbeStrategy.submit_attempts}, "
        f"submit_successes={_GatewayProbeStrategy.submit_successes}"
    )
