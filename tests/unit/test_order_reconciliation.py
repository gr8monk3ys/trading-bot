#!/usr/bin/env python3
"""
Unit tests for order lifecycle reconciliation against broker state.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from utils.audit_log import AuditEventType
from utils.order_lifecycle import OrderLifecycleTracker, OrderState
from utils.order_reconciliation import OrderReconciler


def _make_order(order_id: str, status=None, filled_qty=None):
    """Create a minimal broker order stub for reconciliation tests."""
    order = SimpleNamespace(id=order_id)
    if status is not None:
        order.status = status
    if filled_qty is not None:
        order.filled_qty = filled_qty
    return order


def _last_mismatch_type(audit_log: MagicMock) -> str:
    args, _ = audit_log.log.call_args
    assert args[0] == AuditEventType.RISK_WARNING
    return args[1]["mismatch_type"]


@pytest.mark.asyncio
async def test_missing_order_marked_canceled_when_lookup_fails():
    broker = MagicMock()
    broker.get_orders = AsyncMock(return_value=[])
    broker.get_order_by_id = AsyncMock(return_value=None)
    audit_log = MagicMock()

    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-1", "AAPL", "buy", 10, "TestStrategy")

    reconciler = OrderReconciler(broker=broker, lifecycle_tracker=tracker, audit_log=audit_log)
    await reconciler.reconcile()

    assert tracker.get_state("ord-1") == OrderState.CANCELED
    assert _last_mismatch_type(audit_log) == "missing_from_broker"


@pytest.mark.asyncio
async def test_lookup_filled_order_updates_state_to_filled():
    broker = MagicMock()
    broker.get_orders = AsyncMock(return_value=[])
    broker.get_order_by_id = AsyncMock(
        return_value=_make_order("ord-2", status="filled", filled_qty="10")
    )
    audit_log = MagicMock()

    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-2", "MSFT", "buy", 10, "TestStrategy")

    reconciler = OrderReconciler(broker=broker, lifecycle_tracker=tracker, audit_log=audit_log)
    await reconciler.reconcile()

    assert tracker.get_state("ord-2") == OrderState.FILLED
    assert _last_mismatch_type(audit_log) == "filled_qty_complete"


@pytest.mark.asyncio
async def test_open_order_partial_fill_updates_state_to_partial():
    broker = MagicMock()
    broker.get_orders = AsyncMock(
        return_value=[_make_order("ord-3", status="new", filled_qty="3.5")]
    )
    broker.get_order_by_id = AsyncMock()
    audit_log = MagicMock()

    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-3", "NVDA", "buy", 10, "TestStrategy")

    reconciler = OrderReconciler(broker=broker, lifecycle_tracker=tracker, audit_log=audit_log)
    await reconciler.reconcile()

    assert tracker.get_state("ord-3") == OrderState.PARTIAL
    assert _last_mismatch_type(audit_log) == "filled_qty_partial"
    broker.get_order_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_internal_state_with_open_broker_order_emits_mismatch():
    broker = MagicMock()
    broker.get_orders = AsyncMock(
        return_value=[_make_order("ord-4", status="new", filled_qty="0")]
    )
    audit_log = MagicMock()

    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-4", "TSLA", "buy", 10, "TestStrategy")
    tracker.update_state("ord-4", OrderState.FILLED)

    reconciler = OrderReconciler(broker=broker, lifecycle_tracker=tracker, audit_log=audit_log)
    await reconciler.reconcile()

    assert tracker.get_state("ord-4") == OrderState.FILLED
    assert _last_mismatch_type(audit_log) == "terminal_but_open"


@pytest.mark.asyncio
async def test_status_regression_is_reported_as_transition_blocked():
    broker = MagicMock()
    broker.get_orders = AsyncMock(
        return_value=[_make_order("ord-5", status="new", filled_qty="0")]
    )
    audit_log = MagicMock()

    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-5", "AMD", "buy", 10, "TestStrategy")
    tracker.update_state("ord-5", OrderState.PARTIAL)

    reconciler = OrderReconciler(broker=broker, lifecycle_tracker=tracker, audit_log=audit_log)
    await reconciler.reconcile()

    assert tracker.get_state("ord-5") == OrderState.PARTIAL
    assert _last_mismatch_type(audit_log) == "transition_blocked"


@pytest.mark.asyncio
async def test_recommends_halt_after_consecutive_mismatch_runs():
    broker = MagicMock()
    broker.get_orders = AsyncMock(
        return_value=[_make_order("ord-6", status="new", filled_qty="0")]
    )
    audit_log = MagicMock()

    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-6", "AAPL", "buy", 10, "TestStrategy")
    tracker.update_state("ord-6", OrderState.FILLED)

    reconciler = OrderReconciler(
        broker=broker,
        lifecycle_tracker=tracker,
        audit_log=audit_log,
        mismatch_halt_threshold=2,
    )

    await reconciler.reconcile()
    assert reconciler.should_halt_trading() is False

    await reconciler.reconcile()
    health = reconciler.get_health_snapshot()
    assert reconciler.should_halt_trading() is True
    assert health["consecutive_mismatch_runs"] == 2
    assert "threshold" in (health["halt_reason"] or "")


@pytest.mark.asyncio
async def test_clean_run_clears_halt_recommendation():
    broker = MagicMock()
    broker.get_orders = AsyncMock(
        side_effect=[
            [_make_order("ord-7", status="new", filled_qty="0")],
            [],
        ]
    )
    audit_log = MagicMock()

    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-7", "MSFT", "buy", 10, "TestStrategy")
    tracker.update_state("ord-7", OrderState.FILLED)

    reconciler = OrderReconciler(
        broker=broker,
        lifecycle_tracker=tracker,
        audit_log=audit_log,
        mismatch_halt_threshold=1,
    )

    await reconciler.reconcile()
    assert reconciler.should_halt_trading() is True

    await reconciler.reconcile()
    health = reconciler.get_health_snapshot()
    assert reconciler.should_halt_trading() is False
    assert health["consecutive_mismatch_runs"] == 0


@pytest.mark.asyncio
async def test_persists_reconciliation_health_snapshot(tmp_path):
    broker = MagicMock()
    broker.get_orders = AsyncMock(return_value=[])
    broker.get_order_by_id = AsyncMock(return_value=None)
    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-8", "AAPL", "buy", 5, "TestStrategy")

    events_path = tmp_path / "order_reconciliation_events.jsonl"
    reconciler = OrderReconciler(
        broker=broker,
        lifecycle_tracker=tracker,
        events_path=events_path,
        run_id="test_run",
    )
    await reconciler.reconcile()
    reconciler.close()

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"event_type":"order_reconciliation_snapshot"' in lines[0]
    assert '"run_id":"test_run"' in lines[0]
