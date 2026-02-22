from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from utils.position_manager import PositionManager


@pytest.mark.asyncio
async def test_sync_with_broker_respects_symbol_scope() -> None:
    broker = MagicMock()
    broker.get_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="AAPL", qty="10", avg_entry_price="150.0"),
            SimpleNamespace(symbol="BTCUSD", qty="0.5", avg_entry_price="68000.0"),
        ]
    )

    manager = PositionManager()
    await manager.sync_with_broker(
        broker,
        default_strategy="recovered",
        symbol_scope={"AAPL"},
    )

    positions = await manager.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"


@pytest.mark.asyncio
async def test_sync_with_broker_matches_crypto_scope_formats() -> None:
    broker = MagicMock()
    broker.get_positions = AsyncMock(
        return_value=[
            SimpleNamespace(symbol="BTCUSD", qty="0.25", avg_entry_price="70000.0"),
            SimpleNamespace(symbol="ETHUSD", qty="1.0", avg_entry_price="2000.0"),
        ]
    )

    manager = PositionManager()
    await manager.sync_with_broker(
        broker,
        default_strategy="recovered",
        symbol_scope={"BTC/USD"},
    )

    positions = await manager.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "BTCUSD"
