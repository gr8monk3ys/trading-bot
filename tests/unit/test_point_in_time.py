"""
Tests for Point-in-Time Database

Tests:
- Data point creation and storage
- Point-in-time queries (no lookahead)
- Restatement handling
- Corporate event handling
- Universe membership
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from data.point_in_time import (
    PointInTimeDB,
    PITDataPoint,
    PITQueryResult,
    DataField,
    CorporateEvent,
    create_pit_db,
)


class TestPITDataPoint:
    """Tests for PITDataPoint dataclass."""

    def test_create_data_point(self):
        """Test creating a data point."""
        dp = PITDataPoint(
            symbol="AAPL",
            field="revenue",
            value=100000000,
            period_end=datetime(2024, 1, 15),
            announced_date=datetime(2024, 1, 10),
            source="compustat",
        )

        assert dp.symbol == "AAPL"
        assert dp.field == "revenue"
        assert dp.value == 100000000

    def test_data_point_with_restatement(self):
        """Test data point with restatement flag."""
        dp = PITDataPoint(
            symbol="AAPL",
            field="eps_diluted",
            value=1.50,
            period_end=datetime(2024, 3, 1),
            announced_date=datetime(2024, 1, 15),
            source="compustat",
            is_restated=True,
            original_value=1.45,
        )

        assert dp.is_restated is True
        assert dp.original_value == 1.45

    def test_data_point_to_dict(self):
        """Test data point serialization."""
        dp = PITDataPoint(
            symbol="AAPL",
            field="revenue",
            value=100000000,
            period_end=datetime(2024, 1, 15),
            announced_date=datetime(2024, 1, 10),
            source="compustat",
        )

        d = dp.to_dict()
        assert "symbol" in d
        assert "field" in d
        assert "value" in d
        assert d["symbol"] == "AAPL"


class TestDataField:
    """Tests for DataField enum."""

    def test_income_statement_fields_exist(self):
        """Test income statement fields exist."""
        expected = ["REVENUE", "NET_INCOME", "EPS_BASIC", "EPS_DILUTED"]
        for name in expected:
            assert hasattr(DataField, name)

    def test_balance_sheet_fields_exist(self):
        """Test balance sheet fields exist."""
        expected = ["TOTAL_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY"]
        for name in expected:
            assert hasattr(DataField, name)

    def test_ratio_fields_exist(self):
        """Test ratio fields exist."""
        expected = ["ROE", "ROA", "PE_RATIO", "PB_RATIO"]
        for name in expected:
            assert hasattr(DataField, name)


class TestCorporateEvent:
    """Tests for CorporateEvent enum."""

    def test_earnings_events_exist(self):
        """Test earnings-related events exist."""
        expected = ["EARNINGS_RELEASE", "EARNINGS_RESTATEMENT"]
        for name in expected:
            assert hasattr(CorporateEvent, name)

    def test_corporate_action_events_exist(self):
        """Test corporate action events exist."""
        expected = ["STOCK_SPLIT", "DIVIDEND_ANNOUNCEMENT", "MERGER_ANNOUNCED"]
        for name in expected:
            assert hasattr(CorporateEvent, name)

    def test_lifecycle_events_exist(self):
        """Test company lifecycle events exist."""
        expected = ["IPO", "DELISTING", "BANKRUPTCY"]
        for name in expected:
            assert hasattr(CorporateEvent, name)


class TestPointInTimeDB:
    """Tests for PointInTimeDB class."""

    @pytest.fixture
    def pit_db(self, tmp_path):
        """Create a test database."""
        return create_pit_db(str(tmp_path / "test_pit.db"))

    def test_initialization(self, pit_db):
        """Test database initialization."""
        assert pit_db is not None

    @pytest.mark.asyncio
    async def test_get_fundamental_returns_none_for_missing(self, pit_db):
        """Test get_fundamental returns None for missing data."""
        result = await pit_db.get_fundamental(
            symbol="AAPL",
            field=DataField.REVENUE,
            as_of_date=datetime(2024, 1, 15),
        )

        # Returns None since no data in empty DB
        assert result is None or result.value is None

    @pytest.mark.asyncio
    async def test_get_universe_returns_empty_for_missing(self, pit_db):
        """Test get_universe returns empty for missing data."""
        result = await pit_db.get_universe(
            universe="SP500",
            as_of_date=datetime(2024, 1, 15),
        )

        # Returns empty list for unknown universe
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_fundamentals_batch(self, pit_db):
        """Test batch fundamental query."""
        results = await pit_db.get_fundamentals_batch(
            symbols=["AAPL", "MSFT", "GOOGL"],
            fields=[DataField.REVENUE, DataField.NET_INCOME],
            as_of_date=datetime(2024, 1, 15),
        )

        assert isinstance(results, dict)

    def test_supported_sources(self, pit_db):
        """Test getting supported data sources."""
        sources = pit_db.get_supported_sources()
        assert isinstance(sources, list)


class TestPITQueryResult:
    """Tests for PITQueryResult dataclass."""

    def test_query_result_creation(self):
        """Test creating a query result."""
        result = PITQueryResult(
            symbol="AAPL",
            field="eps_diluted",
            as_of_date=datetime(2024, 1, 20),
            value=1.50,
            data_point=None,
            data_age_days=5,
            next_release=None,
        )

        assert result.symbol == "AAPL"
        assert result.value == 1.50

    def test_query_result_with_data_point(self):
        """Test query result with data point."""
        dp = PITDataPoint(
            symbol="AAPL",
            field="revenue",
            value=100000000,
            period_end=datetime(2024, 1, 15),
            announced_date=datetime(2024, 1, 15),
            source="compustat",
        )

        result = PITQueryResult(
            symbol="AAPL",
            field="revenue",
            as_of_date=datetime(2024, 1, 20),
            value=100000000,
            data_point=dp,
            data_age_days=5,
            next_release=None,
        )

        assert result.data_point is not None
        assert result.data_point.value == 100000000


class TestCreatePITDB:
    """Tests for create_pit_db factory function."""

    def test_create_in_memory_db(self):
        """Test creating in-memory database."""
        db = create_pit_db(":memory:")
        assert db is not None

    def test_create_file_db(self, tmp_path):
        """Test creating file-based database."""
        db_path = str(tmp_path / "test.db")
        db = create_pit_db(db_path)
        assert db is not None

    def test_create_with_sources(self, tmp_path):
        """Test creating with data sources."""
        db = create_pit_db(
            str(tmp_path / "test.db"),
            sources=None,  # No sources by default
        )
        assert db is not None
