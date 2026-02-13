#!/usr/bin/env python3
"""
Unit tests for OrderGateway portfolio guardrails.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from utils.order_gateway import OrderGateway


class _OrderRequest:
    def __init__(self, symbol: str = "AAPL", qty: float = 10, side: str = "buy"):
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.type = "market"


@pytest.mark.asyncio
async def test_guardrail_rejects_new_entries_after_drawdown_breach():
    broker = AsyncMock()
    broker.get_positions.return_value = []
    broker.get_account.side_effect = [
        MagicMock(equity="100000"),
        MagicMock(equity="90000"),
    ]
    broker.submit_order_advanced.return_value = MagicMock(id="ord-1", filled_avg_price="100")

    gateway = OrderGateway(
        broker=broker,
        enforce_gateway=False,
        max_intraday_drawdown_pct=0.05,
        kill_switch_cooldown_minutes=30,
    )

    first = await gateway.submit_order(_OrderRequest(), strategy_name="TestStrategy")
    second = await gateway.submit_order(_OrderRequest(), strategy_name="TestStrategy")

    assert first.success is True
    assert second.success is False
    assert "kill switch" in (second.rejection_reason or "").lower()


@pytest.mark.asyncio
async def test_guardrail_does_not_block_exit_orders():
    broker = AsyncMock()
    broker.get_positions.return_value = [MagicMock(symbol="AAPL", qty="10")]
    broker.get_account.return_value = MagicMock(equity="90000")
    broker.submit_order_advanced.return_value = MagicMock(id="ord-exit", filled_avg_price="99.5")

    gateway = OrderGateway(
        broker=broker,
        enforce_gateway=False,
        max_intraday_drawdown_pct=0.01,
        kill_switch_cooldown_minutes=30,
    )
    gateway._trading_halted_until = datetime.now() + timedelta(minutes=30)
    gateway._halt_reason = "test halt"

    result = await gateway.submit_exit_order(
        symbol="AAPL",
        quantity=10,
        strategy_name="TestStrategy",
        side="sell",
        reason="risk_exit",
    )

    assert result.success is True


@pytest.mark.asyncio
async def test_external_kill_switch_blocks_entries_even_without_drawdown_threshold():
    broker = AsyncMock()
    broker.get_positions.return_value = []
    broker.submit_order_advanced.return_value = MagicMock(id="ord-1", filled_avg_price="100")
    audit_log = MagicMock()

    gateway = OrderGateway(
        broker=broker,
        enforce_gateway=False,
        audit_log=audit_log,
        max_intraday_drawdown_pct=None,
    )
    gateway.activate_kill_switch(
        reason="Order reconciliation mismatch threshold breached",
        source="order_reconciliation",
    )

    result = await gateway.submit_order(_OrderRequest(), strategy_name="TestStrategy")

    assert result.success is False
    assert "kill switch" in (result.rejection_reason or "").lower()
    audit_log.log.assert_called()


def test_guardrail_state_exposed_in_statistics():
    broker = AsyncMock()
    gateway = OrderGateway(
        broker=broker,
        enforce_gateway=False,
        max_intraday_drawdown_pct=0.03,
        kill_switch_cooldown_minutes=45,
    )

    stats = gateway.get_statistics()
    assert "guardrails" in stats
    assert stats["guardrails"]["max_intraday_drawdown_pct"] == 0.03
    assert stats["guardrails"]["kill_switch_cooldown_minutes"] == 45
