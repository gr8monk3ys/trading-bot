#!/usr/bin/env python3
"""
Unit tests for OrderGateway crash-recovery idempotency protections.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils.order_gateway import OrderGateway


class _OrderRequest:
    def __init__(
        self,
        symbol: str = "AAPL",
        qty: float = 10,
        side: str = "buy",
        client_order_id: str = "cid-1",
        order_type: str = "market",
    ):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.client_order_id = client_order_id
        self.type = order_type


class _BrokerWithoutLookup:
    def __init__(self):
        self.submit_calls = 0

    async def submit_order_advanced(self, order_request):
        self.submit_calls += 1
        return SimpleNamespace(
            id=f"ord-{self.submit_calls}",
            filled_avg_price="100.0",
        )


class _BrokerWithLookup(_BrokerWithoutLookup):
    def __init__(self):
        super().__init__()
        self.orders_by_client_id: dict[str, str] = {}

    async def submit_order_advanced(self, order_request):
        result = await super().submit_order_advanced(order_request)
        client_order_id = str(getattr(order_request, "client_order_id", "") or "").strip()
        if client_order_id:
            self.orders_by_client_id[client_order_id] = str(result.id)
        return result

    async def get_order_by_client_id(self, client_order_id: str):
        order_id = self.orders_by_client_id.get(client_order_id)
        if not order_id:
            return None
        return SimpleNamespace(id=order_id)


@pytest.mark.asyncio
async def test_submit_order_suppresses_duplicate_client_order_id_in_same_process():
    broker = _BrokerWithoutLookup()
    gateway = OrderGateway(broker=broker, enforce_gateway=False)
    request = _OrderRequest(client_order_id="restart-001")

    first = await gateway.submit_order(request, strategy_name="StrategyA")
    second = await gateway.submit_order(request, strategy_name="StrategyA")

    assert first.success is True
    assert second.success is True
    assert first.order_id == second.order_id
    assert broker.submit_calls == 1
    assert gateway.get_statistics()["duplicate_orders_suppressed"] == 1
    lifecycle = gateway.lifecycle_tracker.export_state()
    assert lifecycle[first.order_id]["client_order_id"] == "restart-001"


@pytest.mark.asyncio
async def test_submit_order_suppresses_duplicate_after_crash_restart_with_lifecycle_restore():
    broker = _BrokerWithoutLookup()
    request = _OrderRequest(client_order_id="restart-002")

    first_gateway = OrderGateway(broker=broker, enforce_gateway=False)
    first = await first_gateway.submit_order(request, strategy_name="StrategyA")
    saved_lifecycle = first_gateway.lifecycle_tracker.export_state()

    restarted_gateway = OrderGateway(broker=broker, enforce_gateway=False)
    restarted_gateway.lifecycle_tracker.import_state(saved_lifecycle)
    second = await restarted_gateway.submit_order(request, strategy_name="StrategyA")

    assert first.success is True
    assert second.success is True
    assert first.order_id == second.order_id
    assert broker.submit_calls == 1
    assert restarted_gateway.get_statistics()["duplicate_orders_suppressed"] == 1


@pytest.mark.asyncio
async def test_submit_order_uses_broker_lookup_when_lifecycle_state_missing_on_restart():
    broker = _BrokerWithLookup()
    request = _OrderRequest(client_order_id="restart-003")

    first_gateway = OrderGateway(broker=broker, enforce_gateway=False)
    first = await first_gateway.submit_order(request, strategy_name="StrategyA")

    restarted_gateway = OrderGateway(broker=broker, enforce_gateway=False)
    second = await restarted_gateway.submit_order(request, strategy_name="StrategyA")

    assert first.success is True
    assert second.success is True
    assert first.order_id == second.order_id
    assert broker.submit_calls == 1
    assert restarted_gateway.get_statistics()["duplicate_orders_suppressed"] == 1
