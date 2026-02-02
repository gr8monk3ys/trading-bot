"""
Tests for Feature Store

Tests:
- Feature definition and registration
- Feature storage and retrieval
- Point-in-time feature access
- Feature versioning
- Feature matrix operations
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from data.feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureValue,
    FeatureSet,
    FeatureMatrix,
    FeatureRegistry,
    FeatureType,
    ComputeFrequency,
    SQLiteFeatureBackend,
    create_feature_store,
    STANDARD_FEATURES,
)


class TestFeatureDefinition:
    """Tests for FeatureDefinition dataclass."""

    def test_create_feature_definition(self):
        """Test creating a feature definition."""
        fd = FeatureDefinition(
            name="rsi_14",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="14-day RSI",
        )

        assert fd.name == "rsi_14"
        assert fd.feature_type == FeatureType.TECHNICAL

    def test_full_name_with_version(self):
        """Test versioned full name."""
        fd = FeatureDefinition(
            name="momentum",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="Momentum factor",
            version="2.0.0",
        )

        assert fd.full_name == "momentum:v2.0.0"

    def test_compute_hash(self):
        """Test definition hash computation."""
        fd = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.DERIVED,
            compute_frequency=ComputeFrequency.DAILY,
            description="Test",
        )

        hash1 = fd.compute_hash()
        assert len(hash1) == 16

        # Same definition should have same hash
        fd2 = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.DERIVED,
            compute_frequency=ComputeFrequency.DAILY,
            description="Test",
        )
        assert fd.compute_hash() == fd2.compute_hash()

    def test_validate_value_within_range(self):
        """Test value validation within range."""
        fd = FeatureDefinition(
            name="rsi",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="RSI",
            min_value=0,
            max_value=100,
        )

        assert fd.validate_value(50) is True
        assert fd.validate_value(0) is True
        assert fd.validate_value(100) is True

    def test_validate_value_out_of_range(self):
        """Test value validation out of range."""
        fd = FeatureDefinition(
            name="rsi",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="RSI",
            min_value=0,
            max_value=100,
        )

        assert fd.validate_value(-1) is False
        assert fd.validate_value(101) is False

    def test_validate_null_value(self):
        """Test null value validation."""
        fd = FeatureDefinition(
            name="test",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="Test",
            allow_null=True,
        )
        assert fd.validate_value(None) is True

        fd2 = FeatureDefinition(
            name="test2",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="Test",
            allow_null=False,
        )
        assert fd2.validate_value(None) is False

    def test_to_dict(self):
        """Test serialization."""
        fd = FeatureDefinition(
            name="test",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="Test feature",
            tags=["momentum", "technical"],
        )

        d = fd.to_dict()
        assert d["name"] == "test"
        assert d["feature_type"] == "technical"
        assert "momentum" in d["tags"]


class TestFeatureValue:
    """Tests for FeatureValue dataclass."""

    def test_create_feature_value(self):
        """Test creating a feature value."""
        fv = FeatureValue(
            feature_name="rsi_14",
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            value=65.5,
        )

        assert fv.feature_name == "rsi_14"
        assert fv.value == 65.5

    def test_full_name(self):
        """Test versioned full name."""
        fv = FeatureValue(
            feature_name="momentum",
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            value=0.05,
            version="2.0.0",
        )

        assert fv.full_name == "momentum:v2.0.0"


class TestFeatureSet:
    """Tests for FeatureSet dataclass."""

    def test_create_feature_set(self):
        """Test creating a feature set."""
        fs = FeatureSet(
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            features={
                "rsi_14": 65.5,
                "momentum_20": 0.05,
                "volatility": 0.02,
            },
        )

        assert fs["rsi_14"] == 65.5
        assert fs.get("momentum_20") == 0.05
        assert fs.get("missing", 0) == 0

    def test_to_series(self):
        """Test conversion to pandas Series."""
        fs = FeatureSet(
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            features={
                "rsi_14": 65.5,
                "momentum_20": 0.05,
            },
        )

        series = fs.to_series()
        assert isinstance(series, pd.Series)
        assert series["rsi_14"] == 65.5


class TestFeatureMatrix:
    """Tests for FeatureMatrix dataclass."""

    @pytest.fixture
    def sample_matrix(self):
        """Create sample feature matrix."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        dates = [date(2024, 1, i) for i in range(1, 6)]
        feature_names = ["rsi", "momentum", "volatility"]

        data = np.random.randn(len(dates), len(symbols), len(feature_names))

        return FeatureMatrix(
            symbols=symbols,
            dates=dates,
            feature_names=feature_names,
            data=data,
        )

    def test_to_dataframe(self, sample_matrix):
        """Test conversion to DataFrame."""
        df = sample_matrix.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 15  # 5 dates * 3 symbols
        assert df.shape[1] == 3  # 3 features

    def test_get_symbol_features(self, sample_matrix):
        """Test getting features for a single symbol."""
        df = sample_matrix.get_symbol_features("AAPL")
        assert df.shape[0] == 5  # 5 dates
        assert df.shape[1] == 3  # 3 features

    def test_get_cross_section(self, sample_matrix):
        """Test getting cross-section at a point in time."""
        df = sample_matrix.get_cross_section(date(2024, 1, 3))
        assert len(df) == 3  # 3 symbols


class TestFeatureRegistry:
    """Tests for FeatureRegistry class."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create a registry."""
        backend = SQLiteFeatureBackend(str(tmp_path / "features.db"))
        return FeatureRegistry(backend)

    def test_register_feature(self, registry):
        """Test registering a feature."""
        fd = FeatureDefinition(
            name="test_feature",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="Test",
        )

        registry.register(fd)
        assert registry.get("test_feature") is not None

    def test_get_by_tag(self, registry):
        """Test getting features by tag."""
        fd1 = FeatureDefinition(
            name="rsi",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="RSI",
            tags=["momentum"],
        )
        fd2 = FeatureDefinition(
            name="macd",
            feature_type=FeatureType.TECHNICAL,
            compute_frequency=ComputeFrequency.DAILY,
            description="MACD",
            tags=["momentum", "trend"],
        )

        registry.register(fd1)
        registry.register(fd2)

        momentum_features = registry.get_by_tag("momentum")
        assert len(momentum_features) == 2

    def test_get_dependencies(self, registry):
        """Test getting feature dependencies."""
        fd = FeatureDefinition(
            name="combined",
            feature_type=FeatureType.DERIVED,
            compute_frequency=ComputeFrequency.DAILY,
            description="Combined feature",
            dependencies=["rsi", "macd"],
        )

        registry.register(fd)
        deps = registry.get_dependencies("combined")
        assert "rsi" in deps
        assert "macd" in deps


class TestSQLiteFeatureBackend:
    """Tests for SQLiteFeatureBackend class."""

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a backend."""
        return SQLiteFeatureBackend(str(tmp_path / "features.db"))

    @pytest.mark.asyncio
    async def test_store_and_retrieve_value(self, backend):
        """Test storing and retrieving a value."""
        fv = FeatureValue(
            feature_name="rsi_14",
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            value=65.5,
        )

        await backend.store_value(fv)

        result = await backend.get_value(
            feature_name="rsi_14",
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
        )

        assert result is not None
        assert result.value == 65.5

    @pytest.mark.asyncio
    async def test_store_multiple_values(self, backend):
        """Test storing multiple values."""
        values = [
            FeatureValue(
                feature_name="rsi_14",
                symbol=symbol,
                as_of_date=date(2024, 1, 15),
                value=50 + i * 5,
            )
            for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL"])
        ]

        await backend.store_values(values)

        for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL"]):
            result = await backend.get_value(
                feature_name="rsi_14",
                symbol=symbol,
                as_of_date=date(2024, 1, 15),
            )
            assert result is not None
            assert result.value == 50 + i * 5

    @pytest.mark.asyncio
    async def test_get_feature_history(self, backend):
        """Test getting feature history."""
        for i in range(5):
            fv = FeatureValue(
                feature_name="rsi_14",
                symbol="AAPL",
                as_of_date=date(2024, 1, 1) + timedelta(days=i),
                value=50 + i,
            )
            await backend.store_value(fv)

        history = await backend.get_feature_history(
            feature_name="rsi_14",
            symbol="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
        )

        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_get_cross_section(self, backend):
        """Test getting cross-section."""
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            for feature in ["rsi", "momentum"]:
                fv = FeatureValue(
                    feature_name=feature,
                    symbol=symbol,
                    as_of_date=date(2024, 1, 15),
                    value=50.0,
                )
                await backend.store_value(fv)

        result = await backend.get_cross_section(
            feature_names=["rsi", "momentum"],
            symbols=["AAPL", "MSFT", "GOOGL"],
            as_of_date=date(2024, 1, 15),
        )

        assert len(result) == 3
        assert "rsi" in result["AAPL"]
        assert "momentum" in result["AAPL"]


class TestFeatureStore:
    """Tests for FeatureStore class."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a feature store."""
        return create_feature_store(str(tmp_path / "features.db"))

    @pytest.mark.asyncio
    async def test_store_and_get_feature(self, store):
        """Test storing and getting a feature."""
        await store.store_feature(
            feature_name="rsi_14",
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
            value=65.5,
        )

        result = await store.get_feature(
            feature_name="rsi_14",
            symbol="AAPL",
            as_of_date=date(2024, 1, 15),
        )

        assert result == 65.5

    @pytest.mark.asyncio
    async def test_get_feature_history(self, store):
        """Test getting feature history."""
        for i in range(5):
            await store.store_feature(
                feature_name="rsi_14",
                symbol="AAPL",
                as_of_date=date(2024, 1, 1) + timedelta(days=i),
                value=50 + i,
            )

        series = await store.get_feature_history(
            feature_name="rsi_14",
            symbol="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
        )

        assert len(series) == 5

    def test_register_feature(self, store):
        """Test registering a feature."""
        fd = FeatureDefinition(
            name="custom_feature",
            feature_type=FeatureType.DERIVED,
            compute_frequency=ComputeFrequency.DAILY,
            description="Custom feature",
        )

        store.register_feature(fd)
        assert store.registry.get("custom_feature") is not None

    def test_get_lineage(self, store):
        """Test getting feature lineage."""
        fd = FeatureDefinition(
            name="combined",
            feature_type=FeatureType.DERIVED,
            compute_frequency=ComputeFrequency.DAILY,
            description="Combined",
            dependencies=["rsi_14", "momentum_20"],
            data_sources=["price_data"],
        )
        store.register_feature(fd)

        lineage = store.get_lineage("combined")
        assert "dependencies" in lineage
        assert "data_sources" in lineage


class TestStandardFeatures:
    """Tests for standard feature definitions."""

    def test_standard_features_defined(self):
        """Test that standard features are defined."""
        assert len(STANDARD_FEATURES) >= 3

    def test_rsi_feature_exists(self):
        """Test RSI feature exists."""
        rsi = next((f for f in STANDARD_FEATURES if f.name == "rsi_14"), None)
        assert rsi is not None
        assert rsi.min_value == 0
        assert rsi.max_value == 100


class TestFeatureType:
    """Tests for FeatureType enum."""

    def test_all_types_exist(self):
        """Test all feature types exist."""
        expected = ["TECHNICAL", "FUNDAMENTAL", "ALTERNATIVE", "DERIVED", "TARGET"]
        for name in expected:
            assert hasattr(FeatureType, name)


class TestComputeFrequency:
    """Tests for ComputeFrequency enum."""

    def test_all_frequencies_exist(self):
        """Test all frequencies exist."""
        expected = ["TICK", "MINUTE", "HOURLY", "DAILY", "WEEKLY", "MONTHLY"]
        for name in expected:
            assert hasattr(ComputeFrequency, name)
