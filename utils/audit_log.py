"""
Immutable Audit Trail for Institutional-Grade Trading Systems.

This module provides append-only, cryptographically verified logging
for all trading operations. Required for regulatory compliance and
forensic analysis.

Key Features:
- Append-only log entries (no modification or deletion)
- Cryptographic hash chaining (tamper detection)
- JSON-formatted structured logging
- Automatic rotation with archive integrity
- Query interface for compliance reporting
"""

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    # Order events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    ORDER_CANCELED = "order_canceled"
    ORDER_REJECTED = "order_rejected"
    ORDER_MODIFIED = "order_modified"

    # Risk events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    RISK_WARNING = "risk_warning"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    CORRELATION_LIMIT_EXCEEDED = "correlation_limit_exceeded"
    VAR_LIMIT_EXCEEDED = "var_limit_exceeded"

    # Circuit breaker events
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    CIRCUIT_BREAKER_RESET = "circuit_breaker_reset"
    TRADING_HALTED = "trading_halted"
    TRADING_RESUMED = "trading_resumed"

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_RECONCILIATION = "position_reconciliation"
    POSITION_MISMATCH = "position_mismatch"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    STRATEGY_CHANGE = "strategy_change"

    # Compliance events
    MARGIN_WARNING = "margin_warning"
    MARGIN_CALL = "margin_call"
    WASH_SALE_WARNING = "wash_sale_warning"
    GAP_RISK_EVENT = "gap_risk_event"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: datetime
    event_type: AuditEventType
    data: Dict[str, Any]
    sequence_number: int
    previous_hash: str
    entry_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "data": self.data,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditEntry":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            event_type=AuditEventType(d["event_type"]),
            data=d["data"],
            sequence_number=d["sequence_number"],
            previous_hash=d["previous_hash"],
            entry_hash=d["entry_hash"],
        )


class AuditLog:
    """
    Immutable, cryptographically verified audit log.

    Features:
    - Append-only: Entries cannot be modified or deleted
    - Hash chaining: Each entry includes hash of previous entry
    - Tamper detection: Chain verification detects modifications
    - Thread-safe: Multiple writers supported
    - File-based: Persistent storage with JSON format

    Usage:
        audit = AuditLog(log_dir="./audit_logs")

        # Log an order
        audit.log(AuditEventType.ORDER_SUBMITTED, {
            "order_id": "123",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
        })

        # Verify integrity
        is_valid = audit.verify_chain()
    """

    # Genesis hash for first entry
    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        log_dir: str = "./audit_logs",
        max_entries_per_file: int = 10000,
        auto_verify: bool = True,
    ):
        """
        Initialize the audit log.

        Args:
            log_dir: Directory for audit log files
            max_entries_per_file: Rotate file after this many entries
            auto_verify: Verify chain integrity on startup
        """
        self.log_dir = Path(log_dir)
        self.max_entries_per_file = max_entries_per_file
        self.auto_verify = auto_verify

        # Thread safety
        self._lock = threading.Lock()

        # State
        self._entries: List[AuditEntry] = []
        self._sequence_number = 0
        self._last_hash = self.GENESIS_HASH
        self._current_file: Optional[Path] = None
        self._file_handle = None

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Initialize log directory and load existing entries."""
        # Create directory if needed
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Find existing log files
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))

        if log_files:
            # Load existing entries to get last hash and sequence
            self._load_existing_entries(log_files)

            if self.auto_verify:
                if not self.verify_chain():
                    logger.error("AUDIT LOG INTEGRITY CHECK FAILED - Chain verification error")
                    raise RuntimeError("Audit log chain verification failed")
                logger.info("Audit log chain verified successfully")

        # Open current file for writing
        self._open_current_file()

        logger.info(
            f"Audit log initialized: {len(self._entries)} entries, "
            f"sequence {self._sequence_number}, log_dir={self.log_dir}"
        )

    def _load_existing_entries(self, log_files: List[Path]) -> None:
        """Load existing entries from log files."""
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry_dict = json.loads(line)
                            entry = AuditEntry.from_dict(entry_dict)
                            self._entries.append(entry)

                            # Update state
                            self._sequence_number = entry.sequence_number
                            self._last_hash = entry.entry_hash
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning(f"Invalid entry in {log_file}: {e}")
            except Exception as e:
                logger.error(f"Error loading {log_file}: {e}")

    def _open_current_file(self) -> None:
        """Open current log file for appending."""
        # Close existing handle
        if self._file_handle:
            self._file_handle.close()

        # Generate filename based on current date
        today = datetime.now().strftime("%Y%m%d")
        file_index = 0

        while True:
            filename = f"audit_{today}_{file_index:04d}.jsonl"
            filepath = self.log_dir / filename

            # Check if file exists and has room
            if filepath.exists():
                # Count lines
                with open(filepath, "r") as f:
                    line_count = sum(1 for _ in f)
                if line_count >= self.max_entries_per_file:
                    file_index += 1
                    continue

            self._current_file = filepath
            break

        # Open for appending
        self._file_handle = open(self._current_file, "a", encoding="utf-8")

    def _calculate_hash(self, entry: AuditEntry) -> str:
        """Calculate SHA-256 hash for an entry."""
        # Create canonical string representation
        canonical = json.dumps(
            {
                "timestamp": entry.timestamp.isoformat(),
                "event_type": entry.event_type.value,
                "data": entry.data,
                "sequence_number": entry.sequence_number,
                "previous_hash": entry.previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def log(
        self,
        event_type: AuditEventType,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> AuditEntry:
        """
        Log an audit event.

        This is the primary method for recording audit events.

        Args:
            event_type: Type of event
            data: Event data dictionary
            timestamp: Optional timestamp (defaults to now)

        Returns:
            The created AuditEntry
        """
        with self._lock:
            # Create entry
            self._sequence_number += 1
            entry = AuditEntry(
                timestamp=timestamp or datetime.now(),
                event_type=event_type,
                data=data,
                sequence_number=self._sequence_number,
                previous_hash=self._last_hash,
            )

            # Calculate hash
            entry.entry_hash = self._calculate_hash(entry)
            self._last_hash = entry.entry_hash

            # Store in memory
            self._entries.append(entry)

            # Write to file
            self._write_entry(entry)

            # Check for rotation
            if len(self._entries) % self.max_entries_per_file == 0:
                self._open_current_file()

            return entry

    def _write_entry(self, entry: AuditEntry) -> None:
        """Write entry to file."""
        try:
            line = json.dumps(entry.to_dict(), separators=(",", ":"))
            self._file_handle.write(line + "\n")
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())  # Force write to disk
        except Exception as e:
            logger.error(f"Error writing audit entry: {e}")
            raise

    def verify_chain(self) -> bool:
        """
        Verify the integrity of the entire audit chain.

        Returns:
            True if chain is valid, False if tampering detected
        """
        if not self._entries:
            return True

        expected_hash = self.GENESIS_HASH

        for entry in self._entries:
            # Check previous hash matches
            if entry.previous_hash != expected_hash:
                logger.error(
                    f"Chain broken at sequence {entry.sequence_number}: "
                    f"expected previous_hash={expected_hash}, got {entry.previous_hash}"
                )
                return False

            # Recalculate hash
            calculated_hash = self._calculate_hash(entry)
            if calculated_hash != entry.entry_hash:
                logger.error(
                    f"Hash mismatch at sequence {entry.sequence_number}: "
                    f"stored={entry.entry_hash}, calculated={calculated_hash}"
                )
                return False

            expected_hash = entry.entry_hash

        return True

    def verify_entry(self, sequence_number: int) -> bool:
        """
        Verify a specific entry's integrity.

        Args:
            sequence_number: Entry to verify

        Returns:
            True if entry is valid
        """
        for entry in self._entries:
            if entry.sequence_number == sequence_number:
                calculated_hash = self._calculate_hash(entry)
                return calculated_hash == entry.entry_hash
        return False

    def get_entries(
        self,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEntry]:
        """
        Query audit entries with filters.

        Args:
            event_type: Filter by event type
            start_time: Start of time range
            end_time: End of time range
            symbol: Filter by symbol (if in data)
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        results = []

        for entry in self._entries:
            # Apply filters
            if event_type and entry.event_type != event_type:
                continue

            if start_time and entry.timestamp < start_time:
                continue

            if end_time and entry.timestamp > end_time:
                continue

            if symbol and entry.data.get("symbol") != symbol:
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        return results

    def get_order_history(self, order_id: str) -> List[AuditEntry]:
        """
        Get all audit entries for an order.

        Args:
            order_id: Order ID to look up

        Returns:
            List of entries related to this order
        """
        return [
            entry
            for entry in self._entries
            if entry.data.get("order_id") == order_id
        ]

    def get_risk_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditEntry]:
        """Get all risk-related events."""
        risk_types = {
            AuditEventType.RISK_LIMIT_BREACH,
            AuditEventType.RISK_WARNING,
            AuditEventType.POSITION_LIMIT_EXCEEDED,
            AuditEventType.CORRELATION_LIMIT_EXCEEDED,
            AuditEventType.VAR_LIMIT_EXCEEDED,
            AuditEventType.CIRCUIT_BREAKER_TRIGGERED,
            AuditEventType.MARGIN_WARNING,
            AuditEventType.MARGIN_CALL,
        }

        return [
            entry
            for entry in self.get_entries(start_time=start_time, end_time=end_time)
            if entry.event_type in risk_types
        ]

    def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """
        Generate a compliance report for a time period.

        Returns:
            Dictionary with compliance metrics and event summaries
        """
        entries = self.get_entries(start_time=start_time, end_time=end_time, limit=100000)

        # Count by event type
        event_counts = {}
        for entry in entries:
            event_type = entry.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Identify risk events
        risk_events = [e for e in entries if e.event_type.value.startswith("risk_")]
        circuit_breaker_events = [
            e
            for e in entries
            if e.event_type in (AuditEventType.CIRCUIT_BREAKER_TRIGGERED, AuditEventType.TRADING_HALTED)
        ]

        # Order statistics
        order_events = [e for e in entries if e.event_type.value.startswith("order_")]
        orders_submitted = sum(1 for e in order_events if e.event_type == AuditEventType.ORDER_SUBMITTED)
        orders_rejected = sum(1 for e in order_events if e.event_type == AuditEventType.ORDER_REJECTED)

        return {
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "total_events": len(entries),
            "event_counts": event_counts,
            "risk_events_count": len(risk_events),
            "circuit_breaker_events_count": len(circuit_breaker_events),
            "orders_submitted": orders_submitted,
            "orders_rejected": orders_rejected,
            "rejection_rate": orders_rejected / orders_submitted if orders_submitted > 0 else 0,
            "chain_verified": self.verify_chain(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        return {
            "total_entries": len(self._entries),
            "current_sequence": self._sequence_number,
            "last_hash": self._last_hash[:16] + "...",
            "log_dir": str(self.log_dir),
            "current_file": str(self._current_file) if self._current_file else None,
            "chain_valid": self.verify_chain(),
        }

    def close(self) -> None:
        """Close the audit log (flush and close file handle)."""
        with self._lock:
            if self._file_handle:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
                self._file_handle.close()
                self._file_handle = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def log_order_event(
    audit_log: AuditLog,
    event_type: AuditEventType,
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: Optional[float] = None,
    **extra_data,
) -> AuditEntry:
    """Log an order-related event."""
    data = {
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        **extra_data,
    }
    if price is not None:
        data["price"] = price

    return audit_log.log(event_type, data)


def log_risk_event(
    audit_log: AuditLog,
    event_type: AuditEventType,
    risk_type: str,
    current_value: float,
    limit_value: float,
    **extra_data,
) -> AuditEntry:
    """Log a risk-related event."""
    return audit_log.log(
        event_type,
        {
            "risk_type": risk_type,
            "current_value": current_value,
            "limit_value": limit_value,
            "breach_pct": (current_value / limit_value - 1) * 100 if limit_value > 0 else 0,
            **extra_data,
        },
    )


def log_circuit_breaker_event(
    audit_log: AuditLog,
    triggered: bool,
    reason: str,
    equity: float,
    drawdown_pct: float,
    **extra_data,
) -> AuditEntry:
    """Log a circuit breaker event."""
    return audit_log.log(
        AuditEventType.CIRCUIT_BREAKER_TRIGGERED if triggered else AuditEventType.CIRCUIT_BREAKER_RESET,
        {
            "reason": reason,
            "equity": equity,
            "drawdown_pct": drawdown_pct,
            **extra_data,
        },
    )
