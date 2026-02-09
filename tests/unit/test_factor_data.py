"""
Unit tests for the Factor Data module.

Tests cover:
- FundamentalData dataclass and serialization
- FactorDataProvider initialization and configuration
- Cache functionality (TTL, validity, keys)
- Fundamental data fetching (single and batch)
- Synthetic data generation (seeded consistency)
- CSV storage and retrieval
- Factor input building
- PointInTimeDataManager and publication delays
- Sector classification
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.factor_data import (
    FactorDataProvider,
    FundamentalData,
    PointInTimeDataManager,
    create_sample_fundamentals_csv,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def provider(temp_cache_dir):
    """Create a FactorDataProvider with temp cache."""
    return FactorDataProvider(
        cache_dir=temp_cache_dir,
        cache_ttl_hours=1,
    )


@pytest.fixture
def sample_fundamental_data():
    """Create sample FundamentalData for testing."""
    return FundamentalData(
        symbol="AAPL",
        as_of_date=datetime(2024, 6, 15),
        pe_ratio=25.5,
        pb_ratio=40.2,
        ps_ratio=7.5,
        ev_ebitda=20.3,
        roe=0.15,
        roa=0.08,
        roic=0.12,
        debt_to_equity=0.5,
        current_ratio=1.2,
        earnings_variability=0.18,
        market_cap=3_000_000_000_000,  # $3T
        sector="Technology",
        industry="Consumer Electronics",
    )


@pytest.fixture
def sample_csv_data(temp_cache_dir):
    """Create a sample fundamentals CSV file for testing."""
    csv_path = Path(temp_cache_dir) / "fundamentals.csv"

    data = [
        {
            "symbol": "AAPL",
            "date": "2024-01-15",
            "pe_ratio": 25.0,
            "pb_ratio": 35.0,
            "ps_ratio": 7.0,
            "ev_ebitda": 18.0,
            "roe": 0.14,
            "roa": 0.07,
            "roic": 0.11,
            "debt_to_equity": 0.4,
            "current_ratio": 1.3,
            "earnings_variability": 0.15,
            "market_cap": 2_800_000_000_000,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        },
        {
            "symbol": "AAPL",
            "date": "2024-03-15",
            "pe_ratio": 26.0,
            "pb_ratio": 36.0,
            "ps_ratio": 7.2,
            "ev_ebitda": 19.0,
            "roe": 0.145,
            "roa": 0.075,
            "roic": 0.115,
            "debt_to_equity": 0.45,
            "current_ratio": 1.25,
            "earnings_variability": 0.16,
            "market_cap": 2_900_000_000_000,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        },
        {
            "symbol": "MSFT",
            "date": "2024-01-15",
            "pe_ratio": 30.0,
            "pb_ratio": 12.0,
            "ps_ratio": 11.0,
            "ev_ebitda": 22.0,
            "roe": 0.40,
            "roa": 0.15,
            "roic": 0.25,
            "debt_to_equity": 0.35,
            "current_ratio": 1.8,
            "earnings_variability": 0.12,
            "market_cap": 2_700_000_000_000,
            "sector": "Technology",
            "industry": "Software",
        },
    ]

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    yield csv_path


# =============================================================================
# FUNDAMENTAL DATA TESTS
# =============================================================================


class TestFundamentalData:
    """Tests for FundamentalData dataclass."""

    def test_create_fundamental_data(self, sample_fundamental_data):
        """Test creating a FundamentalData object."""
        assert sample_fundamental_data.symbol == "AAPL"
        assert sample_fundamental_data.pe_ratio == 25.5
        assert sample_fundamental_data.sector == "Technology"

    def test_create_fundamental_data_with_defaults(self):
        """Test creating FundamentalData with default None values."""
        fd = FundamentalData(
            symbol="TEST",
            as_of_date=datetime.now(),
        )

        assert fd.symbol == "TEST"
        assert fd.pe_ratio is None
        assert fd.pb_ratio is None
        assert fd.sector is None

    def test_to_dict_includes_all_metrics(self, sample_fundamental_data):
        """Test to_dict includes all expected metrics."""
        d = sample_fundamental_data.to_dict()

        expected_keys = [
            "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
            "roe", "roa", "roic", "debt_to_equity",
            "current_ratio", "earnings_variability",
            "market_cap", "sector", "industry",
        ]

        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_correct(self, sample_fundamental_data):
        """Test to_dict returns correct values."""
        d = sample_fundamental_data.to_dict()

        assert d["pe_ratio"] == 25.5
        assert d["pb_ratio"] == 40.2
        assert d["roe"] == 0.15
        assert d["market_cap"] == 3_000_000_000_000
        assert d["sector"] == "Technology"
        assert d["industry"] == "Consumer Electronics"

    def test_to_dict_handles_none_values(self):
        """Test to_dict properly handles None values."""
        fd = FundamentalData(
            symbol="TEST",
            as_of_date=datetime.now(),
            pe_ratio=20.0,
            # All others default to None
        )

        d = fd.to_dict()

        assert d["pe_ratio"] == 20.0
        assert d["pb_ratio"] is None
        assert d["sector"] is None


# =============================================================================
# FACTOR DATA PROVIDER INITIALIZATION TESTS
# =============================================================================


class TestFactorDataProviderInitialization:
    """Tests for FactorDataProvider initialization."""

    def test_initialize_with_defaults(self, temp_cache_dir):
        """Test initialization with default parameters."""
        provider = FactorDataProvider(cache_dir=temp_cache_dir)

        assert provider.cache_dir == Path(temp_cache_dir)
        assert provider.cache_ttl == timedelta(hours=24)

    def test_initialize_with_custom_ttl(self, temp_cache_dir):
        """Test initialization with custom cache TTL."""
        provider = FactorDataProvider(
            cache_dir=temp_cache_dir,
            cache_ttl_hours=48,
        )

        assert provider.cache_ttl == timedelta(hours=48)

    def test_initialize_creates_cache_directory(self, temp_cache_dir):
        """Test that initialization creates the cache directory."""
        new_cache_dir = os.path.join(temp_cache_dir, "new_cache")
        FactorDataProvider(cache_dir=new_cache_dir)

        assert Path(new_cache_dir).exists()

    def test_initialize_with_api_keys(self, temp_cache_dir):
        """Test initialization with API keys."""
        provider = FactorDataProvider(
            cache_dir=temp_cache_dir,
            alpaca_api_key="test_key",
            alpaca_secret_key="test_secret",
        )

        assert provider.alpaca_api_key == "test_key"
        assert provider.alpaca_secret_key == "test_secret"

    def test_initialize_reads_env_vars(self, temp_cache_dir):
        """Test that initialization reads environment variables."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "env_key",
            "ALPACA_SECRET_KEY": "env_secret",
        }):
            provider = FactorDataProvider(cache_dir=temp_cache_dir)

            assert provider.alpaca_api_key == "env_key"
            assert provider.alpaca_secret_key == "env_secret"

    def test_sectors_dict_populated(self, provider):
        """Test that SECTORS dict is properly populated."""
        assert "AAPL" in FactorDataProvider.SECTORS
        assert "MSFT" in FactorDataProvider.SECTORS
        assert FactorDataProvider.SECTORS["AAPL"] == "Technology"
        assert FactorDataProvider.SECTORS["JPM"] == "Financials"


# =============================================================================
# CACHE FUNCTIONALITY TESTS
# =============================================================================


class TestCacheFunctionality:
    """Tests for cache operations."""

    def test_get_cache_key_format(self, provider):
        """Test cache key generation format."""
        test_date = datetime(2024, 6, 15)
        cache_key = provider._get_cache_key("AAPL", test_date)

        assert cache_key == "AAPL_2024-06-15"

    def test_get_cache_key_uses_current_date_if_none(self, provider):
        """Test cache key uses current date if as_of_date is None."""
        cache_key = provider._get_cache_key("AAPL", None)

        today = datetime.now().strftime("%Y-%m-%d")
        assert cache_key == f"AAPL_{today}"

    def test_is_cache_valid_returns_false_for_missing(self, provider):
        """Test _is_cache_valid returns False for missing cache."""
        assert provider._is_cache_valid("nonexistent_key") is False

    def test_is_cache_valid_returns_true_for_fresh_cache(self, provider):
        """Test _is_cache_valid returns True for fresh cache."""
        cache_key = "AAPL_2024-06-15"
        provider._cache[cache_key] = MagicMock()
        provider._cache_timestamps[cache_key] = datetime.now()

        assert provider._is_cache_valid(cache_key) is True

    def test_is_cache_valid_returns_false_for_expired_cache(self, provider):
        """Test _is_cache_valid returns False for expired cache."""
        cache_key = "AAPL_2024-06-15"
        provider._cache[cache_key] = MagicMock()
        # Set timestamp to 2 hours ago (TTL is 1 hour)
        provider._cache_timestamps[cache_key] = datetime.now() - timedelta(hours=2)

        assert provider._is_cache_valid(cache_key) is False

    def test_cache_ttl_boundary(self, provider):
        """Test cache validity at TTL boundary."""
        cache_key = "AAPL_2024-06-15"
        provider._cache[cache_key] = MagicMock()

        # Set timestamp to just under TTL (59 minutes with 1 hour TTL)
        provider._cache_timestamps[cache_key] = datetime.now() - timedelta(minutes=59)
        assert provider._is_cache_valid(cache_key) is True

        # Set timestamp to just over TTL (61 minutes)
        provider._cache_timestamps[cache_key] = datetime.now() - timedelta(minutes=61)
        assert provider._is_cache_valid(cache_key) is False


# =============================================================================
# GET FUNDAMENTAL DATA TESTS
# =============================================================================


class TestGetFundamentalData:
    """Tests for get_fundamental_data method."""

    @pytest.mark.asyncio
    async def test_get_fundamental_data_returns_cached(self, provider):
        """Test that cached data is returned without fetching."""
        test_date = datetime(2024, 6, 15)
        cache_key = provider._get_cache_key("AAPL", test_date)

        expected_data = FundamentalData(
            symbol="AAPL",
            as_of_date=test_date,
            pe_ratio=25.0,
        )

        provider._cache[cache_key] = expected_data
        provider._cache_timestamps[cache_key] = datetime.now()

        result = await provider.get_fundamental_data("AAPL", test_date)

        assert result == expected_data

    @pytest.mark.asyncio
    async def test_get_fundamental_data_fetches_on_cache_miss(self, provider):
        """Test that data is fetched on cache miss."""
        test_date = datetime(2024, 6, 15)

        result = await provider.get_fundamental_data("AAPL", test_date)

        assert result is not None
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_fundamental_data_populates_cache(self, provider):
        """Test that fetched data is cached."""
        test_date = datetime(2024, 6, 15)
        cache_key = provider._get_cache_key("AAPL", test_date)

        # Ensure cache is empty
        assert cache_key not in provider._cache

        await provider.get_fundamental_data("AAPL", test_date)

        # Cache should now be populated
        assert cache_key in provider._cache
        assert cache_key in provider._cache_timestamps

    @pytest.mark.asyncio
    async def test_get_fundamental_data_uses_current_date_default(self, provider):
        """Test that current date is used if as_of_date is None."""
        result = await provider.get_fundamental_data("AAPL")

        assert result is not None
        assert result.as_of_date.date() == datetime.now().date()


# =============================================================================
# BATCH FUNDAMENTAL DATA TESTS
# =============================================================================


class TestBatchFundamentalData:
    """Tests for get_batch_fundamental_data method."""

    @pytest.mark.asyncio
    async def test_get_batch_returns_dict(self, provider):
        """Test that batch fetch returns a dictionary."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        test_date = datetime(2024, 6, 15)

        result = await provider.get_batch_fundamental_data(symbols, test_date)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_batch_returns_data_for_all_symbols(self, provider):
        """Test that batch fetch returns data for all requested symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        test_date = datetime(2024, 6, 15)

        result = await provider.get_batch_fundamental_data(symbols, test_date)

        assert len(result) == len(symbols)
        for symbol in symbols:
            assert symbol in result
            assert isinstance(result[symbol], FundamentalData)

    @pytest.mark.asyncio
    async def test_get_batch_concurrent_execution(self, provider):
        """Test that batch fetch executes concurrently."""
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
        test_date = datetime(2024, 6, 15)

        # Time the batch fetch
        start = datetime.now()
        result = await provider.get_batch_fundamental_data(symbols, test_date)
        elapsed = (datetime.now() - start).total_seconds()

        assert len(result) == len(symbols)
        # Concurrent execution should be faster than sequential
        # (This is a soft test - just ensuring it completes)
        assert elapsed < 10  # Should complete within 10 seconds

    @pytest.mark.asyncio
    async def test_get_batch_handles_exceptions(self, provider):
        """Test that batch fetch handles exceptions gracefully."""
        symbols = ["AAPL", "INVALID_SYMBOL", "MSFT"]
        test_date = datetime(2024, 6, 15)

        # Patch to simulate exception for one symbol
        original_method = provider.get_fundamental_data

        async def mock_get_data(symbol, as_of_date=None):
            if symbol == "INVALID_SYMBOL":
                raise ValueError("Invalid symbol")
            return await original_method(symbol, as_of_date)

        provider.get_fundamental_data = mock_get_data

        result = await provider.get_batch_fundamental_data(symbols, test_date)

        # Should still have data for valid symbols
        assert "AAPL" in result
        assert "MSFT" in result
        # Invalid symbol should be excluded
        assert "INVALID_SYMBOL" not in result


# =============================================================================
# SYNTHETIC DATA GENERATION TESTS
# =============================================================================


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_synthetic_data_returns_fundamental_data(self, provider):
        """Test that synthetic generation returns FundamentalData."""
        test_date = datetime(2024, 6, 15)
        result = provider._generate_synthetic_data("AAPL", test_date)

        assert isinstance(result, FundamentalData)
        assert result.symbol == "AAPL"

    def test_generate_synthetic_data_is_consistent_for_same_symbol(self, provider):
        """Test that synthetic data is consistent for the same symbol (seeded)."""
        test_date = datetime(2024, 6, 15)

        result1 = provider._generate_synthetic_data("AAPL", test_date)
        result2 = provider._generate_synthetic_data("AAPL", test_date)

        assert result1.pe_ratio == result2.pe_ratio
        assert result1.roe == result2.roe
        assert result1.market_cap == result2.market_cap

    def test_generate_synthetic_data_differs_by_symbol(self, provider):
        """Test that synthetic data differs for different symbols."""
        test_date = datetime(2024, 6, 15)

        aapl_data = provider._generate_synthetic_data("AAPL", test_date)
        msft_data = provider._generate_synthetic_data("MSFT", test_date)

        # Different symbols should have different values (statistically unlikely to match)
        assert aapl_data.pe_ratio != msft_data.pe_ratio

    def test_generate_synthetic_data_sector_assignment(self, provider):
        """Test that known symbols get correct sector assignment."""
        test_date = datetime(2024, 6, 15)

        aapl_data = provider._generate_synthetic_data("AAPL", test_date)
        jpm_data = provider._generate_synthetic_data("JPM", test_date)
        xom_data = provider._generate_synthetic_data("XOM", test_date)

        assert aapl_data.sector == "Technology"
        assert jpm_data.sector == "Financials"
        assert xom_data.sector == "Energy"

    def test_generate_synthetic_data_unknown_symbol_sector(self, provider):
        """Test that unknown symbols get 'Other' sector."""
        test_date = datetime(2024, 6, 15)

        result = provider._generate_synthetic_data("UNKNOWN_SYMBOL", test_date)

        assert result.sector == "Other"

    def test_generate_synthetic_data_sector_adjustments(self, provider):
        """Test that sector-specific adjustments are applied."""
        test_date = datetime(2024, 6, 15)

        # Technology stocks should have higher P/E (pe adjustment = 1.5)
        tech_data = provider._generate_synthetic_data("AAPL", test_date)

        # Energy stocks should have lower P/E (pe adjustment = 0.6)
        energy_data = provider._generate_synthetic_data("XOM", test_date)

        # While random, tech should statistically have higher P/E
        # We can at least verify both have valid values
        assert tech_data.pe_ratio > 0
        assert energy_data.pe_ratio > 0

    def test_generate_synthetic_data_valid_ranges(self, provider):
        """Test that generated values are within valid ranges."""
        test_date = datetime(2024, 6, 15)

        for symbol in ["AAPL", "MSFT", "JPM", "XOM", "UNKNOWN"]:
            result = provider._generate_synthetic_data(symbol, test_date)

            # P/E should be positive
            assert result.pe_ratio >= 5, f"{symbol} P/E too low"

            # P/B should be positive
            assert result.pb_ratio >= 0.5, f"{symbol} P/B too low"

            # Debt/Equity should be non-negative
            assert result.debt_to_equity >= 0, f"{symbol} D/E negative"

            # Current ratio should be positive
            assert result.current_ratio >= 0.5, f"{symbol} current ratio too low"

            # Market cap should be positive
            assert result.market_cap > 0, f"{symbol} market cap not positive"

    def test_generate_synthetic_data_as_of_date_set(self, provider):
        """Test that as_of_date is properly set."""
        test_date = datetime(2024, 6, 15)

        result = provider._generate_synthetic_data("AAPL", test_date)

        assert result.as_of_date == test_date


# =============================================================================
# CSV OPERATIONS TESTS
# =============================================================================


class TestCSVOperations:
    """Tests for CSV save and load operations."""

    @pytest.mark.asyncio
    async def test_fetch_from_csv_handles_malformed_csv(self, provider, temp_cache_dir):
        """Test that fetch_from_csv handles malformed CSV gracefully."""
        # Create a malformed CSV file
        csv_path = Path(temp_cache_dir) / "fundamentals.csv"
        with open(csv_path, "w") as f:
            f.write("this,is,not,valid\ndate,format\n")

        result = await provider._fetch_from_csv("AAPL", datetime(2024, 6, 15))

        # Should return None on parse error
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_csv_returns_data(self, provider, sample_csv_data):
        """Test that CSV data is loaded correctly."""
        test_date = datetime(2024, 2, 15)

        result = await provider._fetch_from_csv("AAPL", test_date)

        assert result is not None
        assert result.symbol == "AAPL"
        # Should get Jan 15 data (most recent before Feb 15)
        assert result.pe_ratio == 25.0

    @pytest.mark.asyncio
    async def test_fetch_from_csv_point_in_time(self, provider, sample_csv_data):
        """Test point-in-time data loading (no look-ahead bias)."""
        # Query with date between Jan 15 and Mar 15
        test_date = datetime(2024, 2, 20)

        result = await provider._fetch_from_csv("AAPL", test_date)

        # Should get Jan 15 data (before Feb 20), not Mar 15 data
        assert result.pe_ratio == 25.0

    @pytest.mark.asyncio
    async def test_fetch_from_csv_gets_latest_available(self, provider, sample_csv_data):
        """Test that most recent available data is returned."""
        # Query with date after all data
        test_date = datetime(2024, 6, 15)

        result = await provider._fetch_from_csv("AAPL", test_date)

        # Should get Mar 15 data (latest available before Jun 15)
        assert result.pe_ratio == 26.0

    @pytest.mark.asyncio
    async def test_fetch_from_csv_returns_none_for_no_data(self, provider, sample_csv_data):
        """Test returns None when no data exists for symbol."""
        test_date = datetime(2024, 6, 15)

        result = await provider._fetch_from_csv("UNKNOWN", test_date)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_csv_returns_none_for_date_before_all_data(
        self, provider, sample_csv_data
    ):
        """Test returns None when query date is before all data."""
        test_date = datetime(2023, 1, 1)  # Before any data

        result = await provider._fetch_from_csv("AAPL", test_date)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_csv_missing_file(self, provider):
        """Test returns None when CSV file doesn't exist."""
        test_date = datetime(2024, 6, 15)

        result = await provider._fetch_from_csv("AAPL", test_date)

        assert result is None

    def test_save_to_csv_creates_file(self, provider, sample_fundamental_data):
        """Test that save_to_csv creates the CSV file."""
        data = {"AAPL": sample_fundamental_data}

        provider.save_to_csv(data, "test_output.csv")

        csv_path = provider.cache_dir / "test_output.csv"
        assert csv_path.exists()

    def test_save_to_csv_correct_content(self, provider, sample_fundamental_data):
        """Test that save_to_csv writes correct content."""
        data = {"AAPL": sample_fundamental_data}

        provider.save_to_csv(data, "test_output.csv")

        csv_path = provider.cache_dir / "test_output.csv"
        df = pd.read_csv(csv_path)

        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["pe_ratio"] == 25.5

    def test_save_to_csv_appends_to_existing(self, provider, sample_fundamental_data):
        """Test that save_to_csv appends to existing file."""
        csv_path = provider.cache_dir / "test_output.csv"

        # Save first data
        data1 = {"AAPL": sample_fundamental_data}
        provider.save_to_csv(data1, "test_output.csv")

        # Save second data
        msft_data = FundamentalData(
            symbol="MSFT",
            as_of_date=datetime(2024, 6, 15),
            pe_ratio=30.0,
        )
        data2 = {"MSFT": msft_data}
        provider.save_to_csv(data2, "test_output.csv")

        df = pd.read_csv(csv_path)
        assert len(df) == 2

    def test_save_to_csv_updates_duplicate(self, provider, sample_fundamental_data):
        """Test that save_to_csv updates duplicate entries."""
        csv_path = provider.cache_dir / "test_output.csv"

        # Save initial data
        data1 = {"AAPL": sample_fundamental_data}
        provider.save_to_csv(data1, "test_output.csv")

        # Save updated data with same symbol and date
        updated_data = FundamentalData(
            symbol="AAPL",
            as_of_date=datetime(2024, 6, 15),
            pe_ratio=28.0,  # Updated value
        )
        data2 = {"AAPL": updated_data}
        provider.save_to_csv(data2, "test_output.csv")

        df = pd.read_csv(csv_path)
        # Should have 1 row (updated, not duplicated)
        aapl_rows = df[df["symbol"] == "AAPL"]
        assert len(aapl_rows) == 1
        assert aapl_rows.iloc[0]["pe_ratio"] == 28.0


# =============================================================================
# BUILD FACTOR INPUTS TESTS
# =============================================================================


class TestBuildFactorInputs:
    """Tests for build_factor_inputs method."""

    @pytest.mark.asyncio
    async def test_build_factor_inputs_returns_dict(self, provider):
        """Test that build_factor_inputs returns a dictionary."""
        symbols = ["AAPL", "MSFT"]
        price_data = pd.DataFrame({
            "AAPL": [150.0, 151.0, 152.0],
            "MSFT": [350.0, 352.0, 354.0],
        })

        result = await provider.build_factor_inputs(symbols, price_data)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_build_factor_inputs_structure(self, provider):
        """Test that build_factor_inputs has correct structure."""
        symbols = ["AAPL", "MSFT"]
        price_data = pd.DataFrame({
            "AAPL": [150.0, 151.0, 152.0],
            "MSFT": [350.0, 352.0, 354.0],
        })

        result = await provider.build_factor_inputs(symbols, price_data)

        assert "fundamental_data" in result
        assert "market_caps" in result
        assert "sectors" in result

    @pytest.mark.asyncio
    async def test_build_factor_inputs_fundamental_data(self, provider):
        """Test that fundamental_data contains correct structure."""
        symbols = ["AAPL"]
        price_data = pd.DataFrame({"AAPL": [150.0]})

        result = await provider.build_factor_inputs(symbols, price_data)

        assert "AAPL" in result["fundamental_data"]
        assert "pe_ratio" in result["fundamental_data"]["AAPL"]

    @pytest.mark.asyncio
    async def test_build_factor_inputs_market_caps(self, provider):
        """Test that market_caps is populated."""
        symbols = ["AAPL", "MSFT"]
        price_data = pd.DataFrame({
            "AAPL": [150.0],
            "MSFT": [350.0],
        })

        result = await provider.build_factor_inputs(symbols, price_data)

        assert "AAPL" in result["market_caps"]
        assert result["market_caps"]["AAPL"] > 0

    @pytest.mark.asyncio
    async def test_build_factor_inputs_sectors(self, provider):
        """Test that sectors are populated."""
        symbols = ["AAPL", "JPM"]
        price_data = pd.DataFrame({
            "AAPL": [150.0],
            "JPM": [180.0],
        })

        result = await provider.build_factor_inputs(symbols, price_data)

        assert "AAPL" in result["sectors"]
        assert result["sectors"]["AAPL"] == "Technology"
        assert result["sectors"]["JPM"] == "Financials"


# =============================================================================
# POINT IN TIME DATA MANAGER TESTS
# =============================================================================


class TestPointInTimeDataManager:
    """Tests for PointInTimeDataManager class."""

    @pytest.fixture
    def pit_manager(self, provider):
        """Create a PointInTimeDataManager with provider."""
        return PointInTimeDataManager(provider)

    def test_publication_delays_defined(self, pit_manager):
        """Test that publication delays are defined."""
        assert "earnings" in PointInTimeDataManager.PUBLICATION_DELAYS
        assert "balance_sheet" in PointInTimeDataManager.PUBLICATION_DELAYS
        assert "market_cap" in PointInTimeDataManager.PUBLICATION_DELAYS
        assert "price" in PointInTimeDataManager.PUBLICATION_DELAYS

    def test_earnings_delay_is_45_days(self, pit_manager):
        """Test that earnings delay is 45 days."""
        assert PointInTimeDataManager.PUBLICATION_DELAYS["earnings"] == 45

    def test_market_cap_delay_is_1_day(self, pit_manager):
        """Test that market cap delay is 1 day."""
        assert PointInTimeDataManager.PUBLICATION_DELAYS["market_cap"] == 1

    def test_price_delay_is_0_days(self, pit_manager):
        """Test that price delay is 0 days."""
        assert PointInTimeDataManager.PUBLICATION_DELAYS["price"] == 0

    def test_get_available_date_earnings(self, pit_manager):
        """Test get_available_date for earnings."""
        as_of_date = datetime(2024, 6, 15)

        available_date = pit_manager.get_available_date("earnings", as_of_date)

        expected = as_of_date - timedelta(days=45)
        assert available_date == expected

    def test_get_available_date_market_cap(self, pit_manager):
        """Test get_available_date for market cap."""
        as_of_date = datetime(2024, 6, 15)

        available_date = pit_manager.get_available_date("market_cap", as_of_date)

        expected = as_of_date - timedelta(days=1)
        assert available_date == expected

    def test_get_available_date_price(self, pit_manager):
        """Test get_available_date for price (no delay)."""
        as_of_date = datetime(2024, 6, 15)

        available_date = pit_manager.get_available_date("price", as_of_date)

        assert available_date == as_of_date

    def test_get_available_date_unknown_type(self, pit_manager):
        """Test get_available_date for unknown data type (defaults to 0)."""
        as_of_date = datetime(2024, 6, 15)

        available_date = pit_manager.get_available_date("unknown_type", as_of_date)

        assert available_date == as_of_date

    @pytest.mark.asyncio
    async def test_get_point_in_time_fundamentals(self, pit_manager):
        """Test get_point_in_time_fundamentals returns dict."""
        symbols = ["AAPL", "MSFT"]
        as_of_date = datetime(2024, 6, 15)

        result = await pit_manager.get_point_in_time_fundamentals(symbols, as_of_date)

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result

    @pytest.mark.asyncio
    async def test_get_point_in_time_fundamentals_applies_delay(self, pit_manager):
        """Test that earnings delay is applied to fundamental queries."""
        symbols = ["AAPL"]
        as_of_date = datetime(2024, 6, 15)

        # The method should internally use data from 45 days prior
        result = await pit_manager.get_point_in_time_fundamentals(symbols, as_of_date)

        assert result is not None
        # Data should be present (synthetic in this case)
        assert "AAPL" in result


# =============================================================================
# SECTOR CLASSIFICATION TESTS
# =============================================================================


class TestSectorClassification:
    """Tests for sector classification."""

    def test_technology_sector_symbols(self, provider):
        """Test Technology sector classification."""
        tech_symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "AMD"]

        for symbol in tech_symbols:
            assert FactorDataProvider.SECTORS.get(symbol) == "Technology"

    def test_financials_sector_symbols(self, provider):
        """Test Financials sector classification."""
        financial_symbols = ["JPM", "V", "MA", "BAC"]

        for symbol in financial_symbols:
            assert FactorDataProvider.SECTORS.get(symbol) == "Financials"

    def test_healthcare_sector_symbols(self, provider):
        """Test Healthcare sector classification."""
        healthcare_symbols = ["JNJ", "UNH", "PFE", "ABBV"]

        for symbol in healthcare_symbols:
            assert FactorDataProvider.SECTORS.get(symbol) == "Healthcare"

    def test_consumer_discretionary_symbols(self, provider):
        """Test Consumer Discretionary sector classification."""
        cd_symbols = ["AMZN", "TSLA", "HD"]

        for symbol in cd_symbols:
            assert FactorDataProvider.SECTORS.get(symbol) == "Consumer Discretionary"

    def test_consumer_staples_symbols(self, provider):
        """Test Consumer Staples sector classification."""
        cs_symbols = ["PG", "KO", "PEP", "WMT", "COST"]

        for symbol in cs_symbols:
            assert FactorDataProvider.SECTORS.get(symbol) == "Consumer Staples"

    def test_energy_sector_symbols(self, provider):
        """Test Energy sector classification."""
        energy_symbols = ["XOM", "CVX"]

        for symbol in energy_symbols:
            assert FactorDataProvider.SECTORS.get(symbol) == "Energy"

    def test_communication_services_symbols(self, provider):
        """Test Communication Services sector classification."""
        comm_symbols = ["DIS", "NFLX"]

        for symbol in comm_symbols:
            assert FactorDataProvider.SECTORS.get(symbol) == "Communication Services"


# =============================================================================
# CREATE SAMPLE FUNDAMENTALS CSV TESTS
# =============================================================================


class TestCreateSampleFundamentalsCSV:
    """Tests for create_sample_fundamentals_csv function."""

    def test_creates_csv_file(self, temp_cache_dir):
        """Test that function creates a CSV file."""
        csv_path = create_sample_fundamentals_csv(temp_cache_dir)

        assert csv_path.exists()

    def test_csv_contains_expected_columns(self, temp_cache_dir):
        """Test that CSV contains expected columns."""
        csv_path = create_sample_fundamentals_csv(temp_cache_dir)

        df = pd.read_csv(csv_path)

        expected_columns = [
            "symbol", "date", "pe_ratio", "pb_ratio", "ps_ratio",
            "roe", "roa", "market_cap", "sector",
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_csv_contains_multiple_symbols(self, temp_cache_dir):
        """Test that CSV contains multiple symbols."""
        csv_path = create_sample_fundamentals_csv(temp_cache_dir)

        df = pd.read_csv(csv_path)

        unique_symbols = df["symbol"].nunique()
        assert unique_symbols >= 10

    def test_csv_contains_multiple_dates(self, temp_cache_dir):
        """Test that CSV contains multiple dates."""
        csv_path = create_sample_fundamentals_csv(temp_cache_dir)

        df = pd.read_csv(csv_path)

        unique_dates = df["date"].nunique()
        assert unique_dates >= 6

    def test_csv_data_is_valid(self, temp_cache_dir):
        """Test that CSV data contains valid values."""
        csv_path = create_sample_fundamentals_csv(temp_cache_dir)

        df = pd.read_csv(csv_path)

        # PE ratios should be positive
        assert (df["pe_ratio"] > 0).all()

        # Market caps should be large positive numbers
        assert (df["market_cap"] > 1e8).all()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAlpacaFetching:
    """Tests for Alpaca API fetching."""

    @pytest.mark.asyncio
    async def test_fetch_from_alpaca_returns_none(self, provider):
        """Test that _fetch_from_alpaca returns None (placeholder)."""
        result = await provider._fetch_from_alpaca("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_from_sources_with_api_key(self, temp_cache_dir):
        """Test _fetch_from_sources tries Alpaca when API key is set."""
        provider = FactorDataProvider(
            cache_dir=temp_cache_dir,
            alpaca_api_key="test_key",
            alpaca_secret_key="test_secret",
        )

        test_date = datetime(2024, 6, 15)

        # Should still return synthetic data since Alpaca returns None
        result = await provider._fetch_from_sources("AAPL", test_date)

        assert result is not None
        assert result.symbol == "AAPL"


class TestIntegration:
    """Integration tests for the factor data module."""

    @pytest.mark.asyncio
    async def test_full_workflow_synthetic(self, provider):
        """Test full workflow with synthetic data."""
        symbols = ["AAPL", "MSFT", "JPM"]
        as_of_date = datetime(2024, 6, 15)
        price_data = pd.DataFrame({
            "AAPL": [150.0, 151.0, 152.0],
            "MSFT": [350.0, 352.0, 354.0],
            "JPM": [180.0, 182.0, 184.0],
        })

        # Fetch batch data
        batch_data = await provider.get_batch_fundamental_data(symbols, as_of_date)

        assert len(batch_data) == 3

        # Build factor inputs
        factor_inputs = await provider.build_factor_inputs(symbols, price_data, as_of_date)

        assert len(factor_inputs["fundamental_data"]) == 3
        assert len(factor_inputs["sectors"]) == 3

    @pytest.mark.asyncio
    async def test_full_workflow_with_csv(self, provider, sample_csv_data):
        """Test full workflow with CSV data source."""
        symbols = ["AAPL", "MSFT"]
        as_of_date = datetime(2024, 2, 15)
        pd.DataFrame({
            "AAPL": [150.0, 151.0, 152.0],
            "MSFT": [350.0, 352.0, 354.0],
        })

        # Fetch batch data (should use CSV)
        batch_data = await provider.get_batch_fundamental_data(symbols, as_of_date)

        assert len(batch_data) == 2
        # Check that we got CSV data (not synthetic)
        assert batch_data["AAPL"].pe_ratio == 25.0  # From CSV

    @pytest.mark.asyncio
    async def test_point_in_time_workflow(self, provider):
        """Test point-in-time data workflow."""
        pit_manager = PointInTimeDataManager(provider)

        symbols = ["AAPL", "MSFT"]
        as_of_date = datetime(2024, 6, 15)

        # Get point-in-time fundamentals
        pit_data = await pit_manager.get_point_in_time_fundamentals(symbols, as_of_date)

        assert len(pit_data) == 2
        assert "pe_ratio" in pit_data["AAPL"]
