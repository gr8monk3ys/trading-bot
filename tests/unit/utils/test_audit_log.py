"""
Tests for Immutable Audit Trail.

Tests verify:
- Append-only behavior
- Cryptographic hash chaining
- Tamper detection
- Query interface
- Chain verification
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from utils.audit_log import (
    AuditEntry,
    AuditEventType,
    AuditLog,
    log_circuit_breaker_event,
    log_order_event,
    log_risk_event,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for audit logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def audit_log(temp_log_dir):
    """Create an AuditLog instance."""
    log = AuditLog(log_dir=temp_log_dir, auto_verify=False)
    yield log
    log.close()


# ============================================================================
# AUDIT ENTRY TESTS
# ============================================================================


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_entry_creation(self):
        """Test creating an AuditEntry."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            event_type=AuditEventType.ORDER_SUBMITTED,
            data={"order_id": "123", "symbol": "AAPL"},
            sequence_number=1,
            previous_hash="abc123",
            entry_hash="def456",
        )

        assert entry.event_type == AuditEventType.ORDER_SUBMITTED
        assert entry.sequence_number == 1
        assert entry.data["symbol"] == "AAPL"

    def test_entry_to_dict(self):
        """Test converting entry to dict."""
        now = datetime.now()
        entry = AuditEntry(
            timestamp=now,
            event_type=AuditEventType.ORDER_FILLED,
            data={"price": 150.0},
            sequence_number=5,
            previous_hash="prev",
            entry_hash="current",
        )

        d = entry.to_dict()

        assert d["timestamp"] == now.isoformat()
        assert d["event_type"] == "order_filled"
        assert d["sequence_number"] == 5
        assert d["data"]["price"] == 150.0

    def test_entry_from_dict(self):
        """Test creating entry from dict."""
        d = {
            "timestamp": "2024-01-15T10:30:00",
            "event_type": "order_canceled",
            "data": {"order_id": "456"},
            "sequence_number": 10,
            "previous_hash": "aaa",
            "entry_hash": "bbb",
        }

        entry = AuditEntry.from_dict(d)

        assert entry.event_type == AuditEventType.ORDER_CANCELED
        assert entry.sequence_number == 10


# ============================================================================
# AUDIT EVENT TYPE TESTS
# ============================================================================


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_order_events(self):
        """Test order event types exist."""
        assert AuditEventType.ORDER_SUBMITTED.value == "order_submitted"
        assert AuditEventType.ORDER_FILLED.value == "order_filled"
        assert AuditEventType.ORDER_REJECTED.value == "order_rejected"

    def test_risk_events(self):
        """Test risk event types exist."""
        assert AuditEventType.RISK_LIMIT_BREACH.value == "risk_limit_breach"
        assert AuditEventType.VAR_LIMIT_EXCEEDED.value == "var_limit_exceeded"

    def test_circuit_breaker_events(self):
        """Test circuit breaker event types exist."""
        assert AuditEventType.CIRCUIT_BREAKER_TRIGGERED.value == "circuit_breaker_triggered"
        assert AuditEventType.TRADING_HALTED.value == "trading_halted"


# ============================================================================
# BASIC LOGGING TESTS
# ============================================================================


class TestBasicLogging:
    """Tests for basic logging functionality."""

    def test_log_single_event(self, audit_log):
        """Test logging a single event."""
        entry = audit_log.log(
            AuditEventType.ORDER_SUBMITTED,
            {"order_id": "123", "symbol": "AAPL", "quantity": 100},
        )

        assert entry.event_type == AuditEventType.ORDER_SUBMITTED
        assert entry.sequence_number == 1
        assert entry.entry_hash != ""

    def test_log_multiple_events(self, audit_log):
        """Test logging multiple events."""
        entries = []
        for i in range(5):
            entry = audit_log.log(
                AuditEventType.ORDER_SUBMITTED,
                {"order_id": str(i), "symbol": "AAPL"},
            )
            entries.append(entry)

        assert entries[-1].sequence_number == 5
        # Each entry should have different hash
        hashes = [e.entry_hash for e in entries]
        assert len(set(hashes)) == 5

    def test_log_with_custom_timestamp(self, audit_log):
        """Test logging with custom timestamp."""
        custom_time = datetime(2024, 1, 15, 10, 30, 0)
        entry = audit_log.log(
            AuditEventType.ORDER_FILLED,
            {"order_id": "123"},
            timestamp=custom_time,
        )

        assert entry.timestamp == custom_time

    def test_sequence_numbers_increment(self, audit_log):
        """Test that sequence numbers always increment."""
        seq_nums = []
        for _ in range(10):
            entry = audit_log.log(AuditEventType.RISK_WARNING, {})
            seq_nums.append(entry.sequence_number)

        assert seq_nums == list(range(1, 11))


# ============================================================================
# HASH CHAIN TESTS
# ============================================================================


class TestHashChain:
    """Tests for cryptographic hash chaining."""

    def test_first_entry_uses_genesis_hash(self, audit_log):
        """Test first entry has genesis hash as previous."""
        entry = audit_log.log(AuditEventType.SYSTEM_START, {})

        assert entry.previous_hash == AuditLog.GENESIS_HASH

    def test_chain_links_entries(self, audit_log):
        """Test each entry links to previous via hash."""
        entry1 = audit_log.log(AuditEventType.ORDER_SUBMITTED, {"id": "1"})
        entry2 = audit_log.log(AuditEventType.ORDER_FILLED, {"id": "1"})
        entry3 = audit_log.log(AuditEventType.POSITION_CLOSED, {"id": "1"})

        assert entry2.previous_hash == entry1.entry_hash
        assert entry3.previous_hash == entry2.entry_hash

    def test_verify_chain_empty(self, audit_log):
        """Test chain verification on empty log."""
        assert audit_log.verify_chain() is True

    def test_verify_chain_valid(self, audit_log):
        """Test chain verification on valid chain."""
        for i in range(10):
            audit_log.log(AuditEventType.ORDER_SUBMITTED, {"i": i})

        assert audit_log.verify_chain() is True

    def test_verify_entry(self, audit_log):
        """Test verifying a single entry."""
        entry = audit_log.log(AuditEventType.ORDER_SUBMITTED, {"test": True})

        assert audit_log.verify_entry(entry.sequence_number) is True

    def test_verify_nonexistent_entry(self, audit_log):
        """Test verifying nonexistent entry."""
        assert audit_log.verify_entry(999) is False


# ============================================================================
# TAMPER DETECTION TESTS
# ============================================================================


class TestTamperDetection:
    """Tests for tamper detection."""

    def test_detect_modified_data(self, audit_log):
        """Test that modifying data is detected."""
        entry = audit_log.log(AuditEventType.ORDER_SUBMITTED, {"amount": 100})

        # Tamper with the data
        entry.data["amount"] = 999

        # Recalculate hash should differ
        calculated = audit_log._calculate_hash(entry)
        assert calculated != entry.entry_hash

    def test_detect_modified_timestamp(self, audit_log):
        """Test that modifying timestamp is detected."""
        entry = audit_log.log(AuditEventType.ORDER_SUBMITTED, {})
        original_hash = entry.entry_hash

        # Tamper with timestamp
        entry.timestamp = datetime(2020, 1, 1)

        calculated = audit_log._calculate_hash(entry)
        assert calculated != original_hash

    def test_detect_broken_chain(self, audit_log):
        """Test that breaking chain is detected."""
        for i in range(5):
            audit_log.log(AuditEventType.ORDER_SUBMITTED, {"i": i})

        # Tamper with middle entry's previous_hash
        audit_log._entries[2].previous_hash = "tampered_hash"

        assert audit_log.verify_chain() is False

    def test_detect_modified_hash(self, audit_log):
        """Test that modifying stored hash is detected."""
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {})

        # Tamper with the stored hash
        audit_log._entries[0].entry_hash = "fake_hash"

        assert audit_log.verify_chain() is False


# ============================================================================
# PERSISTENCE TESTS
# ============================================================================


class TestPersistence:
    """Tests for file persistence."""

    def test_entries_written_to_file(self, audit_log, temp_log_dir):
        """Test entries are written to file."""
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {"test": True})

        # Check file exists
        log_files = list(Path(temp_log_dir).glob("audit_*.jsonl"))
        assert len(log_files) > 0

        # Check file has content
        with open(log_files[0], "r") as f:
            content = f.read()
        assert "order_submitted" in content

    def test_reload_from_file(self, temp_log_dir):
        """Test entries can be reloaded from file."""
        # Create log and add entries
        log1 = AuditLog(log_dir=temp_log_dir, auto_verify=False)
        log1.log(AuditEventType.ORDER_SUBMITTED, {"order_id": "123"})
        log1.log(AuditEventType.ORDER_FILLED, {"order_id": "123"})
        log1.close()

        # Create new log instance - should load existing
        log2 = AuditLog(log_dir=temp_log_dir, auto_verify=False)

        assert len(log2._entries) == 2
        assert log2._sequence_number == 2

        log2.close()

    def test_chain_valid_after_reload(self, temp_log_dir):
        """Test chain is valid after reload."""
        # Create and populate log
        log1 = AuditLog(log_dir=temp_log_dir, auto_verify=False)
        for i in range(10):
            log1.log(AuditEventType.ORDER_SUBMITTED, {"i": i})
        log1.close()

        # Reload and verify
        log2 = AuditLog(log_dir=temp_log_dir, auto_verify=True)  # Auto verify
        assert log2.verify_chain() is True
        log2.close()


# ============================================================================
# QUERY TESTS
# ============================================================================


class TestQueries:
    """Tests for query interface."""

    def test_get_entries_all(self, audit_log):
        """Test getting all entries."""
        for i in range(5):
            audit_log.log(AuditEventType.ORDER_SUBMITTED, {"i": i})

        entries = audit_log.get_entries()
        assert len(entries) == 5

    def test_get_entries_by_type(self, audit_log):
        """Test filtering by event type."""
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {})
        audit_log.log(AuditEventType.ORDER_FILLED, {})
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {})

        entries = audit_log.get_entries(event_type=AuditEventType.ORDER_SUBMITTED)
        assert len(entries) == 2

    def test_get_entries_by_time_range(self, audit_log):
        """Test filtering by time range."""
        now = datetime.now()

        audit_log.log(AuditEventType.ORDER_SUBMITTED, {}, timestamp=now - timedelta(hours=2))
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {}, timestamp=now - timedelta(hours=1))
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {}, timestamp=now)

        entries = audit_log.get_entries(
            start_time=now - timedelta(hours=1, minutes=30),
            end_time=now + timedelta(minutes=1),
        )
        assert len(entries) == 2

    def test_get_entries_by_symbol(self, audit_log):
        """Test filtering by symbol."""
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {"symbol": "AAPL"})
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {"symbol": "MSFT"})
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {"symbol": "AAPL"})

        entries = audit_log.get_entries(symbol="AAPL")
        assert len(entries) == 2

    def test_get_entries_limit(self, audit_log):
        """Test limiting results."""
        for i in range(100):
            audit_log.log(AuditEventType.ORDER_SUBMITTED, {"i": i})

        entries = audit_log.get_entries(limit=10)
        assert len(entries) == 10

    def test_get_order_history(self, audit_log):
        """Test getting order history."""
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {"order_id": "123"})
        audit_log.log(AuditEventType.ORDER_FILLED, {"order_id": "123"})
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {"order_id": "456"})

        history = audit_log.get_order_history("123")
        assert len(history) == 2

    def test_get_risk_events(self, audit_log):
        """Test getting risk events."""
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {})
        audit_log.log(AuditEventType.RISK_WARNING, {"level": "high"})
        audit_log.log(AuditEventType.CIRCUIT_BREAKER_TRIGGERED, {})
        audit_log.log(AuditEventType.ORDER_FILLED, {})

        risk_events = audit_log.get_risk_events()
        assert len(risk_events) == 2


# ============================================================================
# COMPLIANCE REPORT TESTS
# ============================================================================


class TestComplianceReport:
    """Tests for compliance report generation."""

    def test_generate_report(self, audit_log):
        """Test generating a compliance report."""
        now = datetime.now()

        # Add various events
        audit_log.log(AuditEventType.ORDER_SUBMITTED, {}, timestamp=now)
        audit_log.log(AuditEventType.ORDER_FILLED, {}, timestamp=now)
        audit_log.log(AuditEventType.ORDER_REJECTED, {}, timestamp=now)
        audit_log.log(AuditEventType.RISK_WARNING, {}, timestamp=now)

        report = audit_log.generate_compliance_report(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        assert report["total_events"] == 4
        assert report["orders_submitted"] == 1
        assert report["orders_rejected"] == 1
        assert report["rejection_rate"] == 1.0  # 1 rejected / 1 submitted

    def test_report_chain_verified(self, audit_log):
        """Test that report includes chain verification."""
        now = datetime.now()
        audit_log.log(AuditEventType.SYSTEM_START, {}, timestamp=now)

        report = audit_log.generate_compliance_report(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        assert "chain_verified" in report
        assert report["chain_verified"] is True


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience logging functions."""

    def test_log_order_event(self, audit_log):
        """Test log_order_event helper."""
        entry = log_order_event(
            audit_log,
            AuditEventType.ORDER_SUBMITTED,
            order_id="123",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
        )

        assert entry.data["order_id"] == "123"
        assert entry.data["symbol"] == "AAPL"
        assert entry.data["price"] == 150.0

    def test_log_risk_event(self, audit_log):
        """Test log_risk_event helper."""
        entry = log_risk_event(
            audit_log,
            AuditEventType.VAR_LIMIT_EXCEEDED,
            risk_type="daily_var",
            current_value=50000,
            limit_value=40000,
        )

        assert entry.data["risk_type"] == "daily_var"
        assert entry.data["current_value"] == 50000
        assert entry.data["breach_pct"] == 25.0  # 50k/40k - 1 = 25%

    def test_log_circuit_breaker_event(self, audit_log):
        """Test log_circuit_breaker_event helper."""
        entry = log_circuit_breaker_event(
            audit_log,
            triggered=True,
            reason="Daily loss limit exceeded",
            equity=95000,
            drawdown_pct=5.0,
        )

        assert entry.event_type == AuditEventType.CIRCUIT_BREAKER_TRIGGERED
        assert entry.data["reason"] == "Daily loss limit exceeded"


# ============================================================================
# STATISTICS TESTS
# ============================================================================


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_statistics(self, audit_log):
        """Test getting statistics."""
        for i in range(5):
            audit_log.log(AuditEventType.ORDER_SUBMITTED, {"i": i})

        stats = audit_log.get_statistics()

        assert stats["total_entries"] == 5
        assert stats["current_sequence"] == 5
        assert stats["chain_valid"] is True
        assert "last_hash" in stats


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================


class TestContextManager:
    """Tests for context manager usage."""

    def test_context_manager(self, temp_log_dir):
        """Test using audit log as context manager."""
        with AuditLog(log_dir=temp_log_dir, auto_verify=False) as audit:
            audit.log(AuditEventType.SYSTEM_START, {})

        # File should be closed
        assert audit._file_handle is None
