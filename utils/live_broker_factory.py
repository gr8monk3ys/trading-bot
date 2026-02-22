"""
Live broker factory for primary/backup failover wiring.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

from brokers.alpaca_broker import AlpacaBroker
from brokers.ib_broker import InteractiveBrokersBroker
from brokers.multi_broker import FailoverLog, MultiBrokerManager
from config import RISK_PARAMS

logger = logging.getLogger(__name__)


def _parse_int_env(name: str, default: int) -> int:
    raw = str(os.environ.get(name, default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _build_failover_logger(source: str):
    def _on_failover(event: FailoverLog) -> None:
        logger.warning(
            "[%s] broker_failover event=%s from=%s to=%s reason=%s",
            source,
            event.event.value,
            event.from_broker,
            event.to_broker,
            event.reason,
        )

    return _on_failover


async def create_live_broker(
    *,
    paper: bool,
    source: str,
) -> Tuple[Any, Optional[MultiBrokerManager]]:
    """
    Build runtime broker for live/paper mode.

    Returns:
        (broker, failover_manager)
        - broker: object used by strategy runtime
        - failover_manager: manager instance when enabled, else None
    """
    primary = AlpacaBroker(paper=paper)

    if not bool(RISK_PARAMS.get("MULTI_BROKER_ENABLED", False)):
        return primary, None

    backup_kind = str(RISK_PARAMS.get("MULTI_BROKER_BACKUP_BROKER", "ib") or "ib").strip().lower()
    if backup_kind != "ib":
        logger.warning(
            "[%s] multi-broker enabled but unsupported backup kind=%s; using Alpaca only",
            source,
            backup_kind,
        )
        return primary, None

    ib_host = str(os.environ.get("IB_HOST", "127.0.0.1") or "127.0.0.1").strip()
    default_port = 7497 if paper else 7496
    ib_port = _parse_int_env("IB_PAPER_PORT" if paper else "IB_LIVE_PORT", default_port)
    ib_client_id = _parse_int_env("IB_CLIENT_ID", 1)

    backup = InteractiveBrokersBroker(
        host=ib_host,
        port=ib_port,
        client_id=ib_client_id,
    )
    manager = MultiBrokerManager(
        primary=primary,
        backups=[backup],
        health_check_interval=int(RISK_PARAMS.get("MULTI_BROKER_HEALTH_CHECK_INTERVAL", 30)),
        failure_threshold=int(RISK_PARAMS.get("MULTI_BROKER_FAILURE_THRESHOLD", 3)),
        recovery_threshold=int(RISK_PARAMS.get("MULTI_BROKER_RECOVERY_THRESHOLD", 5)),
        operation_timeout=float(RISK_PARAMS.get("MULTI_BROKER_OPERATION_TIMEOUT_SECONDS", 15.0)),
        on_failover=_build_failover_logger(source),
        auto_start_monitoring=False,
    )

    try:
        await backup.connect()
    except Exception as exc:
        logger.warning(
            "[%s] multi-broker backup connect failed (%s); continuing with Alpaca only",
            source,
            exc,
        )
        return primary, None

    await manager.start_monitoring()
    logger.info(
        "[%s] multi-broker failover enabled: primary=alpaca backup=ib host=%s port=%s",
        source,
        ib_host,
        ib_port,
    )
    return manager, manager


async def shutdown_live_broker_failover(manager: Optional[MultiBrokerManager]) -> None:
    """Gracefully tear down failover manager and backup connections."""
    if manager is None:
        return
    try:
        await manager.disconnect()
    except Exception as exc:
        logger.warning("Error stopping multi-broker failover manager: %s", exc)
