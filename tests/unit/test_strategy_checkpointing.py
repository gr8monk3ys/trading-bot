#!/usr/bin/env python3
"""
Unit tests for enhanced strategy runtime checkpointing.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


class _DummyStrategyForCheckpoint:
    def __init__(self):
        self.price_history = {"AAPL": deque([{"close": 100.0}, {"close": 101.0}], maxlen=10)}
        self.current_prices = {"AAPL": 101.5}
        self.signals = {"AAPL": "buy"}
        self.indicators = {"AAPL": {"rsi": 58.2}}
        self.circuit_breaker = SimpleNamespace(
            trading_halted=True,
            halt_triggered_at=datetime(2026, 1, 10, 14, 30, 0),
            last_reset_date=date(2026, 1, 10),
            peak_equity_today=102500.0,
            _halt_reason="test_halt",
            _halt_loss_pct=0.031,
        )

    async def export_state(self):
        return {"entry_prices": {"AAPL": 99.0}}


class _DummyStrategyForRestore:
    def __init__(self):
        self.imported_state = None
        self.price_history = {"AAPL": deque(maxlen=5)}
        self.current_prices = {}
        self.signals = {}
        self.indicators = {}
        self.circuit_breaker = SimpleNamespace(
            trading_halted=False,
            halt_triggered_at=None,
            last_reset_date=None,
            peak_equity_today=None,
            _halt_reason=None,
            _halt_loss_pct=None,
        )

    async def import_state(self, state):
        self.imported_state = state


@pytest.fixture
def manager():
    with patch("engine.strategy_manager.StrategyManager._load_available_strategies"):
        from engine.strategy_manager import StrategyManager

        broker = Mock()
        return StrategyManager(broker=broker)


@pytest.mark.asyncio
async def test_build_strategy_checkpoint_includes_internal_state(manager):
    strategy = _DummyStrategyForCheckpoint()

    checkpoint = await manager._build_strategy_checkpoint("DummyStrategy", strategy)

    assert checkpoint["version"] == 2
    assert checkpoint["exported_state"]["entry_prices"]["AAPL"] == 99.0
    assert checkpoint["internal_state"]["current_prices"]["AAPL"] == 101.5
    assert checkpoint["internal_state"]["signals"]["AAPL"] == "buy"
    assert checkpoint["internal_state"]["indicators"]["AAPL"]["rsi"] == 58.2
    assert checkpoint["internal_state"]["price_history"]["AAPL"][-1]["close"] == 101.0
    assert checkpoint["internal_state"]["circuit_breaker_state"]["trading_halted"] is True


@pytest.mark.asyncio
async def test_restore_strategy_checkpoint_applies_internal_state(manager):
    strategy = _DummyStrategyForRestore()
    saved = {
        "version": 2,
        "exported_state": {"entry_prices": {"AAPL": 98.0}},
        "internal_state": {
            "price_history": {"AAPL": [{"close": 97.0}, {"close": 98.5}]},
            "current_prices": {"AAPL": 98.5},
            "signals": {"AAPL": "sell"},
            "indicators": {"AAPL": {"rsi": 45.0}},
            "circuit_breaker_state": {
                "trading_halted": True,
                "halt_triggered_at": "2026-01-12T10:00:00",
                "last_reset_date": "2026-01-12",
                "peak_equity_today": 101000.0,
                "halt_reason": "data_quality",
                "halt_loss_pct": 0.02,
            },
        },
    }

    await manager._restore_strategy_checkpoint(strategy, saved)

    assert strategy.imported_state == {"entry_prices": {"AAPL": 98.0}}
    assert strategy.current_prices["AAPL"] == 98.5
    assert strategy.signals["AAPL"] == "sell"
    assert strategy.indicators["AAPL"]["rsi"] == 45.0
    assert isinstance(strategy.price_history["AAPL"], deque)
    assert strategy.price_history["AAPL"].maxlen == 5
    assert strategy.price_history["AAPL"][-1]["close"] == 98.5
    assert strategy.circuit_breaker.trading_halted is True
    assert strategy.circuit_breaker._halt_reason == "data_quality"
