from __future__ import annotations

import pytest

from brokers.alpaca_broker import AlpacaBroker
from utils import live_broker_factory


@pytest.mark.asyncio
async def test_create_live_broker_returns_primary_when_failover_disabled(monkeypatch):
    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_ENABLED", False)

    broker, manager = await live_broker_factory.create_live_broker(
        paper=True,
        source="unit-test",
    )

    assert isinstance(broker, AlpacaBroker)
    assert manager is None


@pytest.mark.asyncio
async def test_create_live_broker_returns_primary_when_backup_connect_fails(monkeypatch):
    class _FailingBackup:
        name = "ib-backup"

        def __init__(self, host, port, client_id):
            self.host = host
            self.port = port
            self.client_id = client_id

        async def connect(self):
            raise RuntimeError("connect failed")

    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_ENABLED", True)
    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_BACKUP_BROKER", "ib")
    monkeypatch.setattr(live_broker_factory, "InteractiveBrokersBroker", _FailingBackup)

    broker, manager = await live_broker_factory.create_live_broker(
        paper=True,
        source="unit-test",
    )

    assert isinstance(broker, AlpacaBroker)
    assert manager is None


@pytest.mark.asyncio
async def test_create_live_broker_returns_manager_when_backup_connects(monkeypatch):
    class _Backup:
        name = "ib-backup"

        def __init__(self, host, port, client_id):
            self.host = host
            self.port = port
            self.client_id = client_id
            self.connected = False

        async def connect(self):
            self.connected = True
            return True

    class _Manager:
        def __init__(
            self,
            primary,
            backups,
            health_check_interval,
            failure_threshold,
            recovery_threshold,
            operation_timeout,
            on_failover,
            auto_start_monitoring,
        ):
            self.primary = primary
            self.backups = backups
            self.started = False
            self.disconnected = False

        async def start_monitoring(self):
            self.started = True

        async def disconnect(self):
            self.disconnected = True

    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_ENABLED", True)
    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_BACKUP_BROKER", "ib")
    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_HEALTH_CHECK_INTERVAL", 30)
    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_FAILURE_THRESHOLD", 3)
    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_RECOVERY_THRESHOLD", 5)
    monkeypatch.setitem(live_broker_factory.RISK_PARAMS, "MULTI_BROKER_OPERATION_TIMEOUT_SECONDS", 15.0)
    monkeypatch.setattr(live_broker_factory, "InteractiveBrokersBroker", _Backup)
    monkeypatch.setattr(live_broker_factory, "MultiBrokerManager", _Manager)

    broker, manager = await live_broker_factory.create_live_broker(
        paper=True,
        source="unit-test",
    )

    assert isinstance(manager, _Manager)
    assert broker is manager
    assert manager.started is True

    await live_broker_factory.shutdown_live_broker_failover(manager)
    assert manager.disconnected is True
