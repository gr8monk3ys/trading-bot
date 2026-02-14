"""
Tests for Phase 1.3: Daily Position Reconciliation.

These tests verify that:
1. Reconciler correctly detects position mismatches
2. Tolerance thresholds work correctly
3. Mismatch types are correctly identified
4. History tracking works
5. ReconciliationError is raised when configured
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestPositionMismatch:
    """Tests for PositionMismatch dataclass."""

    def test_mismatch_calculates_discrepancy(self):
        """PositionMismatch should calculate discrepancy qty and value."""
        from utils.reconciliation import PositionMismatch

        mismatch = PositionMismatch(
            symbol="AAPL",
            broker_qty=100,
            internal_qty=90,
            broker_avg_price=150.0,
        )

        assert mismatch.discrepancy_qty == 10
        assert mismatch.discrepancy_value == 1500.0  # 10 shares * $150


class TestReconciliationResult:
    """Tests for ReconciliationResult dataclass."""

    def test_result_calculates_total_discrepancy(self):
        """ReconciliationResult should sum discrepancy values."""
        from utils.reconciliation import PositionMismatch, ReconciliationResult

        mismatches = [
            PositionMismatch("AAPL", 100, 90, broker_avg_price=150.0),
            PositionMismatch("MSFT", 50, 40, broker_avg_price=300.0),
        ]

        result = ReconciliationResult(
            positions_match=False,
            broker_positions={"AAPL": 100, "MSFT": 50},
            internal_positions={"AAPL": 90, "MSFT": 40},
            mismatches=mismatches,
        )

        # AAPL: 10 * 150 = 1500, MSFT: 10 * 300 = 3000
        assert result.total_discrepancy_value == 4500.0

    def test_result_generates_reconciliation_id(self):
        """ReconciliationResult should generate unique ID."""
        from utils.reconciliation import ReconciliationResult

        result = ReconciliationResult(
            positions_match=True,
            broker_positions={},
            internal_positions={},
            mismatches=[],
        )

        assert result.reconciliation_id.startswith("recon_")


class TestPositionReconciler:
    """Tests for PositionReconciler."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = MagicMock()
        return broker

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock internal tracker."""
        tracker = MagicMock()
        return tracker

    async def test_reconcile_passes_when_positions_match(self, mock_broker, mock_tracker):
        """Reconciliation should pass when positions match."""
        from utils.reconciliation import PositionReconciler

        # Set up broker positions
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        # Set up internal positions (matching)
        internal_pos = MagicMock()
        internal_pos.symbol = "AAPL"
        internal_pos.qty = "100"
        mock_tracker.get_positions = MagicMock(return_value=[internal_pos])

        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=mock_tracker,
            halt_on_mismatch=False,
        )

        result = await reconciler.reconcile()

        assert result.positions_match is True
        assert len(result.mismatches) == 0

    async def test_reconcile_detects_quantity_mismatch(self, mock_broker, mock_tracker):
        """Reconciliation should detect quantity mismatches."""
        from utils.reconciliation import PositionReconciler

        # Set up broker positions
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        # Set up internal positions (mismatch - 90 instead of 100)
        internal_pos = MagicMock()
        internal_pos.symbol = "AAPL"
        internal_pos.qty = "90"
        mock_tracker.get_positions = MagicMock(return_value=[internal_pos])

        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=mock_tracker,
            halt_on_mismatch=False,
        )

        result = await reconciler.reconcile()

        assert result.positions_match is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0].symbol == "AAPL"
        assert result.mismatches[0].mismatch_type == "quantity"

    async def test_reconcile_detects_missing_internal(self, mock_broker, mock_tracker):
        """Reconciliation should detect positions missing from internal tracker."""
        from utils.reconciliation import PositionReconciler

        # Set up broker positions
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        # Set up internal positions (empty - missing AAPL)
        mock_tracker.get_positions = MagicMock(return_value=[])

        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=mock_tracker,
            halt_on_mismatch=False,
        )

        result = await reconciler.reconcile()

        assert result.positions_match is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0].mismatch_type == "missing_internal"

    async def test_reconcile_detects_missing_broker(self, mock_broker, mock_tracker):
        """Reconciliation should detect positions missing from broker."""
        from utils.reconciliation import PositionReconciler

        # Set up broker positions (empty)
        mock_broker.get_positions = AsyncMock(return_value=[])

        # Set up internal positions (has AAPL)
        internal_pos = MagicMock()
        internal_pos.symbol = "AAPL"
        internal_pos.qty = "100"
        mock_tracker.get_positions = MagicMock(return_value=[internal_pos])

        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=mock_tracker,
            halt_on_mismatch=False,
        )

        result = await reconciler.reconcile()

        assert result.positions_match is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0].mismatch_type == "missing_broker"

    async def test_tolerance_allows_small_differences(self, mock_broker, mock_tracker):
        """Reconciliation should allow differences within tolerance."""
        from utils.reconciliation import PositionReconciler

        # Set up broker positions
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        # Set up internal positions (0.5% difference - within 1% tolerance)
        internal_pos = MagicMock()
        internal_pos.symbol = "AAPL"
        internal_pos.qty = "99.5"  # 0.5% less
        mock_tracker.get_positions = MagicMock(return_value=[internal_pos])

        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=mock_tracker,
            halt_on_mismatch=False,
            tolerance_pct=0.01,  # 1%
            tolerance_abs=1.0,  # 1 share
        )

        result = await reconciler.reconcile()

        # Should pass because 0.5 shares < 1 share tolerance
        assert result.positions_match is True

    async def test_raises_error_when_halt_on_mismatch(self, mock_broker, mock_tracker):
        """Reconciliation should raise ReconciliationError when halt_on_mismatch=True."""
        from utils.reconciliation import PositionReconciler, ReconciliationError

        # Set up broker positions
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        # Set up internal positions (mismatch)
        internal_pos = MagicMock()
        internal_pos.symbol = "AAPL"
        internal_pos.qty = "50"  # Large mismatch
        mock_tracker.get_positions = MagicMock(return_value=[internal_pos])

        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=mock_tracker,
            halt_on_mismatch=True,
        )

        with pytest.raises(ReconciliationError) as exc_info:
            await reconciler.reconcile()

        assert "Position mismatch" in str(exc_info.value)
        assert exc_info.value.result is not None

    async def test_tracks_reconciliation_history(self, mock_broker, mock_tracker):
        """Reconciler should maintain history of reconciliations."""
        from utils.reconciliation import PositionReconciler

        # Set up matching positions
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        internal_pos = MagicMock()
        internal_pos.symbol = "AAPL"
        internal_pos.qty = "100"
        mock_tracker.get_positions = MagicMock(return_value=[internal_pos])

        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=mock_tracker,
            halt_on_mismatch=False,
        )

        # Run multiple reconciliations
        await reconciler.reconcile()
        await reconciler.reconcile()
        await reconciler.reconcile()

        history = reconciler.get_reconciliation_history()
        assert len(history) == 3

        stats = reconciler.get_mismatch_statistics()
        assert stats["total_reconciliations"] == 3
        assert stats["success_rate"] == 1.0

    async def test_no_internal_tracker_compares_to_empty(self, mock_broker):
        """Reconciler without internal tracker should compare to empty state."""
        from utils.reconciliation import PositionReconciler

        # Set up broker positions
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        # No internal tracker
        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=None,
            halt_on_mismatch=False,
        )

        result = await reconciler.reconcile()

        # Should detect AAPL as "missing_internal" since internal state is empty
        assert result.positions_match is False
        assert result.mismatches[0].mismatch_type == "missing_internal"

    async def test_writes_reconciliation_snapshots(self, mock_broker, mock_tracker, tmp_path):
        """PositionReconciler should persist snapshot events when configured."""
        from utils.reconciliation import PositionReconciler

        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        internal_pos = MagicMock()
        internal_pos.symbol = "AAPL"
        internal_pos.qty = "100"
        mock_tracker.get_positions = MagicMock(return_value=[internal_pos])

        events_path = tmp_path / "position_reconciliation_events.jsonl"
        reconciler = PositionReconciler(
            broker=mock_broker,
            internal_tracker=mock_tracker,
            halt_on_mismatch=False,
            events_path=events_path,
            run_id="test_run",
        )
        await reconciler.reconcile()
        reconciler.close()

        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert '"event_type":"position_reconciliation_snapshot"' in lines[0]
        assert '"run_id":"test_run"' in lines[0]


class TestRunNightlyReconciliation:
    """Tests for the convenience function."""

    async def test_calls_notify_on_mismatch(self):
        """run_nightly_reconciliation should call notify_func on mismatch."""
        from utils.reconciliation import run_nightly_reconciliation

        mock_broker = MagicMock()
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        # Mock notify function
        notify_func = AsyncMock()

        result = await run_nightly_reconciliation(
            broker=mock_broker,
            internal_tracker=None,  # Will cause mismatch
            halt_on_mismatch=False,
            notify_func=notify_func,
        )

        assert result.positions_match is False
        notify_func.assert_called_once()
        assert "MISMATCH" in notify_func.call_args[0][0]
