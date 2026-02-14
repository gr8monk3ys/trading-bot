"""
Position Reconciliation - Daily verification that internal state matches broker.

INSTITUTIONAL SAFETY: Detects position mismatches that could cause:
- Incorrect risk calculations (VaR, correlation, exposure)
- Wrong P&L tracking
- Unintended position sizes

Usage:
    reconciler = PositionReconciler(broker, internal_tracker)

    # Run nightly reconciliation
    result = await reconciler.reconcile()

    if not result.positions_match:
        logger.critical(f"POSITION MISMATCH: {result.mismatches}")
        # Optionally halt trading until resolved
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.audit_log import AuditEventType, AuditLog
from utils.run_artifacts import JsonlWriter

logger = logging.getLogger(__name__)


@dataclass
class PositionMismatch:
    """Details of a position mismatch between broker and internal tracking."""

    symbol: str
    broker_qty: float
    internal_qty: float
    broker_avg_price: Optional[float] = None
    internal_avg_price: Optional[float] = None
    discrepancy_qty: float = 0
    discrepancy_value: float = 0
    mismatch_type: str = "quantity"  # 'quantity', 'missing_broker', 'missing_internal'
    detected_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Calculate discrepancy details."""
        self.discrepancy_qty = abs(self.broker_qty - self.internal_qty)
        if self.broker_avg_price:
            self.discrepancy_value = self.discrepancy_qty * self.broker_avg_price


@dataclass
class ReconciliationResult:
    """Result of a position reconciliation check."""

    positions_match: bool
    broker_positions: Dict[str, float]  # {symbol: qty}
    internal_positions: Dict[str, float]  # {symbol: qty}
    mismatches: List[PositionMismatch]
    timestamp: datetime = field(default_factory=datetime.now)
    total_discrepancy_value: float = 0
    reconciliation_id: str = ""

    def __post_init__(self):
        """Calculate totals."""
        self.total_discrepancy_value = sum(m.discrepancy_value for m in self.mismatches)
        self.reconciliation_id = f"recon_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"


class ReconciliationError(Exception):
    """Raised when reconciliation detects critical mismatch."""

    def __init__(self, message: str, result: ReconciliationResult):
        super().__init__(message)
        self.result = result


class PositionReconciler:
    """
    Reconciles internal position tracking with broker positions.

    INSTITUTIONAL SAFETY: Runs daily to detect:
    - Partial fills that weren't properly tracked
    - API failures that caused position updates to be missed
    - Manual trades made outside the system
    - System bugs in position tracking

    Can be configured to:
    - Log warnings (default)
    - Halt trading on mismatch
    - Automatically sync to broker state
    """

    # Tolerance for floating-point comparison (1% or 1 share, whichever is smaller)
    QUANTITY_TOLERANCE_PCT = 0.01  # 1%
    QUANTITY_TOLERANCE_ABS = 1.0  # 1 share

    def __init__(
        self,
        broker,
        internal_tracker=None,
        halt_on_mismatch: bool = True,
        sync_to_broker: bool = False,
        tolerance_pct: float = None,
        tolerance_abs: float = None,
        audit_log: Optional[AuditLog] = None,
        events_path: str | Path | None = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize the position reconciler.

        Args:
            broker: Broker instance with get_positions() method
            internal_tracker: Object with get_positions() that returns internal position state.
                            If None, reconciliation compares to empty state (broker-only audit)
            halt_on_mismatch: If True, raise ReconciliationError on mismatch
            sync_to_broker: If True, update internal tracker to match broker on mismatch
            tolerance_pct: Quantity tolerance as percentage (default 1%)
            tolerance_abs: Absolute quantity tolerance (default 1 share)
        """
        self.broker = broker
        self.internal_tracker = internal_tracker
        self.halt_on_mismatch = halt_on_mismatch
        self.sync_to_broker = sync_to_broker
        self.audit_log = audit_log
        self.run_id = run_id
        self._events_writer = JsonlWriter(events_path) if events_path else None

        self.tolerance_pct = tolerance_pct or self.QUANTITY_TOLERANCE_PCT
        self.tolerance_abs = tolerance_abs or self.QUANTITY_TOLERANCE_ABS

        # History for audit
        self._reconciliation_history: List[ReconciliationResult] = []
        self._max_history = 100  # Keep last 100 reconciliations

        logger.info(
            f"PositionReconciler initialized: halt_on_mismatch={halt_on_mismatch}, "
            f"sync_to_broker={sync_to_broker}, tolerance={self.tolerance_pct:.1%} or {self.tolerance_abs} shares"
        )

    async def reconcile(self) -> ReconciliationResult:
        """
        Perform position reconciliation.

        Compares broker positions with internal tracking and reports any mismatches.

        Returns:
            ReconciliationResult with match status and details

        Raises:
            ReconciliationError: If halt_on_mismatch is True and positions don't match
        """
        logger.info("=" * 80)
        logger.info("🔍 POSITION RECONCILIATION STARTING")
        logger.info("=" * 80)

        try:
            # Fetch broker positions
            broker_positions = await self._get_broker_positions()

            # Fetch internal positions
            internal_positions = await self._get_internal_positions()

            # Compare positions
            mismatches = self._compare_positions(broker_positions, internal_positions)

            # Build result
            positions_match = len(mismatches) == 0
            result = ReconciliationResult(
                positions_match=positions_match,
                broker_positions=broker_positions,
                internal_positions=internal_positions,
                mismatches=mismatches,
            )

            # Log result
            self._log_result(result)

            # Store in history
            self._reconciliation_history.append(result)
            if len(self._reconciliation_history) > self._max_history:
                self._reconciliation_history.pop(0)

            # Handle mismatch
            if not positions_match:
                if self.sync_to_broker:
                    await self._sync_to_broker_state(broker_positions)

                if self.halt_on_mismatch:
                    raise ReconciliationError(
                        f"Position mismatch detected: {len(mismatches)} discrepancies "
                        f"totaling ${result.total_discrepancy_value:,.2f}",
                        result=result,
                    )

            self._write_snapshot(result)
            return result

        except ReconciliationError as e:
            self._write_snapshot(result=e.result)
            raise
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}", exc_info=True)
            raise

    async def _get_broker_positions(self) -> Dict[str, float]:
        """
        Fetch positions from broker.

        Returns:
            Dict mapping symbol to quantity
        """
        positions = {}
        try:
            broker_positions = await self.broker.get_positions()

            for pos in broker_positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                positions[symbol] = qty

                logger.debug(
                    f"Broker position: {symbol} = {qty} shares @ ${float(pos.avg_entry_price):,.2f}"
                )

        except Exception as e:
            logger.error(f"Failed to fetch broker positions: {e}")
            raise

        logger.info(f"Fetched {len(positions)} positions from broker")
        return positions

    async def _get_internal_positions(self) -> Dict[str, float]:
        """
        Fetch positions from internal tracker.

        Returns:
            Dict mapping symbol to quantity
        """
        positions = {}

        if self.internal_tracker is None:
            logger.debug("No internal tracker configured - comparing to empty state")
            return positions

        try:
            # Support both sync and async interfaces
            if asyncio.iscoroutinefunction(self.internal_tracker.get_positions):
                internal_positions = await self.internal_tracker.get_positions()
            else:
                internal_positions = self.internal_tracker.get_positions()

            for pos in internal_positions:
                # Handle different position object formats
                if hasattr(pos, "symbol"):
                    symbol = pos.symbol
                    qty = float(pos.qty) if hasattr(pos, "qty") else float(pos.quantity)
                elif isinstance(pos, dict):
                    symbol = pos.get("symbol")
                    qty = float(pos.get("qty", pos.get("quantity", 0)))
                else:
                    continue

                positions[symbol] = qty
                logger.debug(f"Internal position: {symbol} = {qty} shares")

        except Exception as e:
            logger.error(f"Failed to fetch internal positions: {e}")
            raise

        logger.info(f"Fetched {len(positions)} positions from internal tracker")
        return positions

    def _compare_positions(
        self,
        broker: Dict[str, float],
        internal: Dict[str, float],
    ) -> List[PositionMismatch]:
        """
        Compare broker and internal positions.

        Args:
            broker: Broker positions {symbol: qty}
            internal: Internal positions {symbol: qty}

        Returns:
            List of PositionMismatch objects for any discrepancies
        """
        mismatches = []
        all_symbols = set(broker.keys()) | set(internal.keys())

        for symbol in all_symbols:
            broker_qty = broker.get(symbol, 0)
            internal_qty = internal.get(symbol, 0)

            # Check if quantities match within tolerance
            if not self._quantities_match(broker_qty, internal_qty):
                # Determine mismatch type
                if broker_qty != 0 and internal_qty == 0:
                    mismatch_type = "missing_internal"
                elif broker_qty == 0 and internal_qty != 0:
                    mismatch_type = "missing_broker"
                else:
                    mismatch_type = "quantity"

                mismatch = PositionMismatch(
                    symbol=symbol,
                    broker_qty=broker_qty,
                    internal_qty=internal_qty,
                    mismatch_type=mismatch_type,
                )
                mismatches.append(mismatch)

        return mismatches

    def _quantities_match(self, qty1: float, qty2: float) -> bool:
        """
        Check if two quantities match within tolerance.

        Uses the smaller of:
        - Percentage tolerance (default 1%)
        - Absolute tolerance (default 1 share)

        Args:
            qty1: First quantity
            qty2: Second quantity

        Returns:
            True if quantities match within tolerance
        """
        diff = abs(qty1 - qty2)

        # Calculate percentage-based tolerance
        max_qty = max(abs(qty1), abs(qty2))
        pct_tolerance = max_qty * self.tolerance_pct if max_qty > 0 else 0

        # Use smaller of percentage or absolute tolerance
        tolerance = min(pct_tolerance, self.tolerance_abs)

        return diff <= tolerance

    def _log_result(self, result: ReconciliationResult):
        """Log reconciliation result."""
        if result.positions_match:
            logger.info("✅ RECONCILIATION PASSED: All positions match")
            logger.info(f"   Broker positions: {len(result.broker_positions)}")
            logger.info(f"   Internal positions: {len(result.internal_positions)}")
            if self.audit_log:
                self.audit_log.log(
                    AuditEventType.POSITION_RECONCILIATION,
                    {
                        "positions_match": True,
                        "broker_positions": len(result.broker_positions),
                        "internal_positions": len(result.internal_positions),
                        "reconciliation_id": result.reconciliation_id,
                    },
                )
        else:
            logger.critical("=" * 80)
            logger.critical("❌ RECONCILIATION FAILED: Position mismatch detected!")
            logger.critical(f"   Mismatches: {len(result.mismatches)}")
            logger.critical(f"   Total discrepancy value: ${result.total_discrepancy_value:,.2f}")
            logger.critical("=" * 80)
            if self.audit_log:
                self.audit_log.log(
                    AuditEventType.POSITION_RECONCILIATION,
                    {
                        "positions_match": False,
                        "mismatch_count": len(result.mismatches),
                        "total_discrepancy_value": result.total_discrepancy_value,
                        "reconciliation_id": result.reconciliation_id,
                    },
                )
                self.audit_log.log(
                    AuditEventType.POSITION_MISMATCH,
                    {
                        "mismatches": [
                            {
                                "symbol": m.symbol,
                                "broker_qty": m.broker_qty,
                                "internal_qty": m.internal_qty,
                                "mismatch_type": m.mismatch_type,
                            }
                            for m in result.mismatches
                        ],
                        "reconciliation_id": result.reconciliation_id,
                    },
                )

            for mismatch in result.mismatches:
                logger.critical(
                    f"   {mismatch.symbol}: "
                    f"Broker={mismatch.broker_qty:.4f}, "
                    f"Internal={mismatch.internal_qty:.4f}, "
                    f"Diff={mismatch.discrepancy_qty:.4f} "
                    f"({mismatch.mismatch_type})"
                )

    async def _sync_to_broker_state(self, broker_positions: Dict[str, float]):
        """
        Update internal tracker to match broker state.

        WARNING: This overwrites internal state. Only use when broker is source of truth.
        """
        if self.internal_tracker is None:
            logger.warning("Cannot sync - no internal tracker configured")
            return

        if not hasattr(self.internal_tracker, "sync_positions"):
            logger.warning("Internal tracker does not support sync_positions()")
            return

        try:
            logger.info("Syncing internal tracker to broker state...")

            if asyncio.iscoroutinefunction(self.internal_tracker.sync_positions):
                await self.internal_tracker.sync_positions(broker_positions)
            else:
                self.internal_tracker.sync_positions(broker_positions)

            logger.info("✅ Internal tracker synced to broker state")

        except Exception as e:
            logger.error(f"Failed to sync internal tracker: {e}")

    def get_reconciliation_history(self) -> List[ReconciliationResult]:
        """Get reconciliation history for audit."""
        return self._reconciliation_history.copy()

    def get_last_result(self) -> Optional[ReconciliationResult]:
        """Get the most recent reconciliation result."""
        if self._reconciliation_history:
            return self._reconciliation_history[-1]
        return None

    def get_mismatch_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on reconciliation mismatches.

        Returns:
            Dict with mismatch statistics
        """
        if not self._reconciliation_history:
            return {
                "total_reconciliations": 0,
                "successful_reconciliations": 0,
                "failed_reconciliations": 0,
                "success_rate": 1.0,
            }

        total = len(self._reconciliation_history)
        successful = sum(1 for r in self._reconciliation_history if r.positions_match)
        failed = total - successful

        return {
            "total_reconciliations": total,
            "successful_reconciliations": successful,
            "failed_reconciliations": failed,
            "success_rate": successful / total if total > 0 else 1.0,
            "last_reconciliation": self._reconciliation_history[-1].timestamp.isoformat(),
            "last_result": "PASS" if self._reconciliation_history[-1].positions_match else "FAIL",
        }

    def close(self) -> None:
        if self._events_writer:
            self._events_writer.close()

    def _write_snapshot(self, result: Optional[ReconciliationResult]) -> None:
        """Persist reconciliation snapshot for replay/incident timelines."""
        if not self._events_writer or result is None:
            return

        self._events_writer.write(
            {
                "event_type": "position_reconciliation_snapshot",
                "timestamp": datetime.utcnow().isoformat(),
                "run_id": self.run_id,
                "reconciliation_id": result.reconciliation_id,
                "positions_match": result.positions_match,
                "mismatch_count": len(result.mismatches),
                "total_discrepancy_value": result.total_discrepancy_value,
                "mismatches": [
                    {
                        "symbol": m.symbol,
                        "broker_qty": m.broker_qty,
                        "internal_qty": m.internal_qty,
                        "mismatch_type": m.mismatch_type,
                    }
                    for m in result.mismatches
                ],
            }
        )


async def run_nightly_reconciliation(
    broker,
    internal_tracker=None,
    halt_on_mismatch: bool = True,
    notify_func=None,
) -> ReconciliationResult:
    """
    Convenience function to run nightly reconciliation.

    Args:
        broker: Broker instance
        internal_tracker: Optional internal position tracker
        halt_on_mismatch: If True, raise exception on mismatch
        notify_func: Optional async function to call on mismatch (e.g., send alert)

    Returns:
        ReconciliationResult

    Usage:
        # In nightly job:
        result = await run_nightly_reconciliation(
            broker=alpaca_broker,
            internal_tracker=position_manager,
            notify_func=send_discord_alert,
        )
    """
    reconciler = PositionReconciler(
        broker=broker,
        internal_tracker=internal_tracker,
        halt_on_mismatch=halt_on_mismatch,
    )

    try:
        result = await reconciler.reconcile()

        if not result.positions_match and notify_func:
            await notify_func(
                f"⚠️ POSITION MISMATCH DETECTED\n"
                f"Mismatches: {len(result.mismatches)}\n"
                f"Total discrepancy: ${result.total_discrepancy_value:,.2f}"
            )

        return result

    except ReconciliationError as e:
        if notify_func:
            await notify_func(
                f"🚨 CRITICAL: Position reconciliation failed!\n"
                f"{str(e)}\n"
                f"Trading halted until resolved."
            )
        raise
