#!/usr/bin/env python3
"""
Tests for restart-safe OrderGateway runtime state recovery.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils.order_gateway import OrderGateway


class _OrderRequest:
    def __init__(self, symbol: str = "AAPL", qty: float = 10, side: str = "buy"):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.type = "market"


class _Broker:
    def __init__(self):
        self.submit_calls = 0

    async def submit_order_advanced(self, order_request):
        self.submit_calls += 1
        return SimpleNamespace(id=f"ord-{self.submit_calls}", filled_avg_price="100")

    async def get_positions(self):
        return []


@pytest.mark.asyncio
async def test_kill_switch_persists_across_gateway_restart():
    broker = _Broker()
    source = OrderGateway(
        broker=broker,
        enforce_gateway=False,
        kill_switch_cooldown_minutes=30,
    )
    source.activate_kill_switch(
        reason="Critical reconciliation drift",
        source="reconciliation",
    )
    runtime_state = source.export_runtime_state()

    restored = OrderGateway(
        broker=broker,
        enforce_gateway=False,
        kill_switch_cooldown_minutes=30,
    )
    restored.import_runtime_state(runtime_state)
    result = await restored.submit_order(_OrderRequest(), strategy_name="StrategyA")

    assert result.success is False
    assert "kill switch" in (result.rejection_reason or "").lower()
    assert broker.submit_calls == 0
    stats = restored.get_statistics()
    assert stats["guardrails"]["halt_reason"] == "Critical reconciliation drift"
    assert stats["guardrails"]["trading_halted_until"] is not None
