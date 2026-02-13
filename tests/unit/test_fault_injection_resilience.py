#!/usr/bin/env python3
"""
Fault-injection resilience tests for reconciliation safety paths.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from utils.order_lifecycle import OrderLifecycleTracker
from utils.order_reconciliation import OrderReconciler


@pytest.mark.asyncio
async def test_order_reconciliation_fetch_failure_does_not_raise():
    broker = MagicMock()
    broker.get_orders = AsyncMock(side_effect=RuntimeError("injected broker failure"))

    tracker = OrderLifecycleTracker()
    tracker.register_order("ord-1", "AAPL", "buy", 10, "ChaosStrategy")

    reconciler = OrderReconciler(
        broker=broker,
        lifecycle_tracker=tracker,
        mismatch_halt_threshold=2,
    )

    # Should fail closed (no exception), preserving runtime loop stability.
    await reconciler.reconcile()
    health = reconciler.get_health_snapshot()
    assert health["runs_total"] == 1
    assert "injected broker failure" in (health.get("last_error") or "")
