from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import live_trader
from live_trader import LiveTrader


@pytest.mark.asyncio
async def test_initialize_logs_paper_trading_banner_on_failure(monkeypatch) -> None:
    trader = LiveTrader(strategy_name="momentum", symbols=["BTC/USD"], parameters={})
    info = MagicMock()

    async def _boom(*args, **kwargs):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(live_trader.logger, "info", info)
    monkeypatch.setattr(live_trader, "create_live_broker", _boom)

    result = await trader.initialize()

    assert result is False
    info.assert_any_call("🚀 PAPER TRADING INITIALIZATION")


@pytest.mark.asyncio
async def test_start_trading_logs_paper_banner_and_calls_shutdown(monkeypatch) -> None:
    trader = LiveTrader(strategy_name="momentum", symbols=["BTC/USD"], parameters={})
    info = MagicMock()
    shutdown = AsyncMock()
    broker = SimpleNamespace(start_websocket=AsyncMock())

    trader.broker = broker
    trader.shutdown_event.set()
    trader.monitor_performance = AsyncMock()
    trader._housekeeping_loop = AsyncMock()
    trader.shutdown = shutdown

    monkeypatch.setattr(live_trader.logger, "info", info)

    await trader.start_trading()

    broker.start_websocket.assert_awaited_once_with(["BTC/USD"])
    shutdown.assert_awaited_once()
    info.assert_any_call("📈 STARTING PAPER TRADING")


@pytest.mark.asyncio
async def test_shutdown_logs_paper_banner(monkeypatch) -> None:
    trader = LiveTrader(strategy_name="momentum", symbols=["BTC/USD"], parameters={})
    info = MagicMock()

    trader.broker = SimpleNamespace(
        get_account=AsyncMock(
            return_value=SimpleNamespace(
                equity="101000",
                id="paper-account",
                buying_power="101000",
            )
        ),
        get_positions=AsyncMock(return_value=[]),
    )
    trader.start_equity = 100000.0
    trader.start_time = datetime.now() - timedelta(minutes=5)

    monkeypatch.setattr(live_trader.logger, "info", info)
    monkeypatch.setattr(live_trader, "shutdown_live_broker_failover", AsyncMock())

    await trader.shutdown()

    info.assert_any_call("🛑 SHUTTING DOWN PAPER TRADING")


@pytest.mark.asyncio
async def test_main_uses_paper_trading_cli_entrypoint(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakeTrader:
        def __init__(self, strategy_name, symbols, parameters):
            calls["strategy_name"] = strategy_name
            calls["symbols"] = symbols
            calls["parameters"] = parameters

        async def initialize(self):
            calls["initialized"] = True
            return True

        async def start_trading(self):
            calls["started"] = True

        def handle_shutdown_signal(self, *_args):
            return None

    monkeypatch.setattr(
        live_trader.sys,
        "argv",
        ["live_trader.py", "--strategy", "momentum", "--symbols", "AAPL", "MSFT"],
    )
    monkeypatch.setattr(live_trader, "LiveTrader", _FakeTrader)
    monkeypatch.setattr(live_trader.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(live_trader, "SYMBOLS", ["TSLA", "NVDA", "META"])

    exit_code = await live_trader.main()

    assert exit_code == 0
    assert calls["strategy_name"] == "momentum"
    assert calls["symbols"] == ["AAPL", "MSFT"]
    assert calls["initialized"] is True
    assert calls["started"] is True
