"""
Versioned Feature Store

Provides institutional-grade feature management:
1. Point-in-Time Features: No lookahead bias in backtests
2. Versioning: Track feature definitions and computation
3. Lineage: Understand feature dependencies
4. Caching: Efficient storage and retrieval

Why feature stores matter:
- Computing features on-the-fly at 100+ features × 1000+ assets = bottleneck
- Feature versioning enables reproducibility
- Lineage tracking helps debug model degradation
"""

import asyncio
import hashlib
import json
import logging
import pickle
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FeatureType(Enum):
    """Type of feature."""
    TECHNICAL = "technical"      # RSI, MACD, etc.
    FUNDAMENTAL = "fundamental"  # P/E, ROE, etc.
    ALTERNATIVE = "alternative"  # Sentiment, satellite, etc.
    DERIVED = "derived"          # Combinations of above
    TARGET = "target"            # Labels for ML


class ComputeFrequency(Enum):
    """How often feature is recomputed."""
    TICK = "tick"           # Every tick
    MINUTE = "minute"       # Every minute
    HOURLY = "hourly"       # Every hour
    DAILY = "daily"         # End of day
    WEEKLY = "weekly"       # End of week
    MONTHLY = "monthly"     # End of month
    QUARTERLY = "quarterly" # End of quarter


@dataclass
class FeatureDefinition:
    """Defines a feature and how to compute it."""
    name: str
    feature_type: FeatureType
    compute_frequency: ComputeFrequency
    description: str
    version: str = "1.0.0"

    # Computation
    compute_fn: Optional[Callable] = None
    compute_sql: Optional[str] = None

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)

    # Validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_null: bool = True

    def __hash__(self):
        return hash((self.name, self.version))

    @property
    def full_name(self) -> str:
        """Get versioned name."""
        return f"{self.name}:v{self.version}"

    def compute_hash(self) -> str:
        """Compute hash of feature definition for change detection."""
        content = f"{self.name}|{self.version}|{self.compute_sql or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def validate_value(self, value: Any) -> bool:
        """Validate a feature value."""
        if value is None:
            return self.allow_null

        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "compute_frequency": self.compute_frequency.value,
            "description": self.description,
            "version": self.version,
            "dependencies": self.dependencies,
            "data_sources": self.data_sources,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }


@dataclass
class FeatureValue:
    """A single feature value at a point in time."""
    feature_name: str
    symbol: str
    as_of_date: date
    value: Any
    computed_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.feature_name}:v{self.version}"


@dataclass
class FeatureSet:
    """A collection of features for a symbol at a point in time."""
    symbol: str
    as_of_date: date
    features: Dict[str, Any]
    computed_at: datetime = field(default_factory=datetime.now)

    def __getitem__(self, key: str) -> Any:
        return self.features[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.features.get(key, default)

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.features, name=f"{self.symbol}_{self.as_of_date}")


@dataclass
class FeatureMatrix:
    """Features for multiple symbols across time."""
    symbols: List[str]
    dates: List[date]
    feature_names: List[str]
    data: np.ndarray  # Shape: (dates, symbols, features)
    computed_at: datetime = field(default_factory=datetime.now)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to multi-index DataFrame."""
        # Create multi-index
        index = pd.MultiIndex.from_product(
            [self.dates, self.symbols],
            names=["date", "symbol"]
        )

        # Reshape data
        reshaped = self.data.reshape(-1, len(self.feature_names))

        return pd.DataFrame(
            reshaped,
            index=index,
            columns=self.feature_names
        )

    def get_symbol_features(
        self,
        symbol: str,
        feature_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get features for a single symbol over time."""
        symbol_idx = self.symbols.index(symbol)
        data = self.data[:, symbol_idx, :]

        df = pd.DataFrame(
            data,
            index=pd.DatetimeIndex(self.dates),
            columns=self.feature_names
        )

        if feature_name:
            return df[feature_name]
        return df

    def get_cross_section(
        self,
        as_of_date: date,
        feature_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get features for all symbols at a point in time."""
        date_idx = self.dates.index(as_of_date)
        data = self.data[date_idx, :, :]

        df = pd.DataFrame(
            data,
            index=self.symbols,
            columns=self.feature_names
        )

        if feature_name:
            return df[feature_name]
        return df


class FeatureStorageBackend(ABC):
    """Abstract backend for feature storage."""

    @abstractmethod
    async def store_value(self, value: FeatureValue) -> None:
        """Store a single feature value."""
        pass

    @abstractmethod
    async def store_values(self, values: List[FeatureValue]) -> None:
        """Store multiple feature values."""
        pass

    @abstractmethod
    async def get_value(
        self,
        feature_name: str,
        symbol: str,
        as_of_date: date,
        version: Optional[str] = None,
    ) -> Optional[FeatureValue]:
        """Get a feature value."""
        pass

    @abstractmethod
    async def get_feature_history(
        self,
        feature_name: str,
        symbol: str,
        start_date: date,
        end_date: date,
        version: Optional[str] = None,
    ) -> List[FeatureValue]:
        """Get historical values for a feature."""
        pass

    @abstractmethod
    async def get_cross_section(
        self,
        feature_names: List[str],
        symbols: List[str],
        as_of_date: date,
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for multiple symbols at a point in time."""
        pass


class SQLiteFeatureBackend(FeatureStorageBackend):
    """SQLite-based feature storage backend."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Feature definitions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_definitions (
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                feature_type TEXT NOT NULL,
                compute_frequency TEXT NOT NULL,
                description TEXT,
                dependencies TEXT,
                data_sources TEXT,
                created_at TEXT NOT NULL,
                definition_hash TEXT NOT NULL,
                PRIMARY KEY (name, version)
            )
        """)

        # Feature values table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                feature_name TEXT NOT NULL,
                version TEXT NOT NULL,
                symbol TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                value BLOB NOT NULL,
                computed_at TEXT NOT NULL,
                metadata TEXT,
                PRIMARY KEY (feature_name, version, symbol, as_of_date)
            )
        """)

        # Indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_symbol_date
            ON feature_values (symbol, as_of_date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_date
            ON feature_values (as_of_date)
        """)

        # Lineage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_lineage (
                feature_name TEXT NOT NULL,
                version TEXT NOT NULL,
                parent_feature TEXT NOT NULL,
                parent_version TEXT NOT NULL,
                relationship TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (feature_name, version, parent_feature, parent_version)
            )
        """)

        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    async def store_value(self, value: FeatureValue) -> None:
        """Store a single feature value."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO feature_values
            (feature_name, version, symbol, as_of_date, value, computed_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            value.feature_name,
            value.version,
            value.symbol,
            value.as_of_date.isoformat(),
            pickle.dumps(value.value),
            value.computed_at.isoformat(),
            json.dumps(value.metadata),
        ))

        conn.commit()
        conn.close()

    async def store_values(self, values: List[FeatureValue]) -> None:
        """Store multiple feature values."""
        if not values:
            return

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.executemany("""
            INSERT OR REPLACE INTO feature_values
            (feature_name, version, symbol, as_of_date, value, computed_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                v.feature_name,
                v.version,
                v.symbol,
                v.as_of_date.isoformat(),
                pickle.dumps(v.value),
                v.computed_at.isoformat(),
                json.dumps(v.metadata),
            )
            for v in values
        ])

        conn.commit()
        conn.close()

    async def get_value(
        self,
        feature_name: str,
        symbol: str,
        as_of_date: date,
        version: Optional[str] = None,
    ) -> Optional[FeatureValue]:
        """Get a feature value."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if version:
            cursor.execute("""
                SELECT feature_name, version, symbol, as_of_date, value, computed_at, metadata
                FROM feature_values
                WHERE feature_name = ? AND symbol = ? AND as_of_date = ? AND version = ?
            """, (feature_name, symbol, as_of_date.isoformat(), version))
        else:
            # Get latest version
            cursor.execute("""
                SELECT feature_name, version, symbol, as_of_date, value, computed_at, metadata
                FROM feature_values
                WHERE feature_name = ? AND symbol = ? AND as_of_date = ?
                ORDER BY version DESC
                LIMIT 1
            """, (feature_name, symbol, as_of_date.isoformat()))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return FeatureValue(
            feature_name=row[0],
            version=row[1],
            symbol=row[2],
            as_of_date=date.fromisoformat(row[3]),
            value=pickle.loads(row[4]),
            computed_at=datetime.fromisoformat(row[5]),
            metadata=json.loads(row[6]) if row[6] else {},
        )

    async def get_feature_history(
        self,
        feature_name: str,
        symbol: str,
        start_date: date,
        end_date: date,
        version: Optional[str] = None,
    ) -> List[FeatureValue]:
        """Get historical values for a feature."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if version:
            cursor.execute("""
                SELECT feature_name, version, symbol, as_of_date, value, computed_at, metadata
                FROM feature_values
                WHERE feature_name = ? AND symbol = ?
                  AND as_of_date >= ? AND as_of_date <= ?
                  AND version = ?
                ORDER BY as_of_date
            """, (feature_name, symbol, start_date.isoformat(), end_date.isoformat(), version))
        else:
            cursor.execute("""
                SELECT feature_name, version, symbol, as_of_date, value, computed_at, metadata
                FROM feature_values
                WHERE feature_name = ? AND symbol = ?
                  AND as_of_date >= ? AND as_of_date <= ?
                ORDER BY as_of_date, version DESC
            """, (feature_name, symbol, start_date.isoformat(), end_date.isoformat()))

        rows = cursor.fetchall()
        conn.close()

        return [
            FeatureValue(
                feature_name=row[0],
                version=row[1],
                symbol=row[2],
                as_of_date=date.fromisoformat(row[3]),
                value=pickle.loads(row[4]),
                computed_at=datetime.fromisoformat(row[5]),
                metadata=json.loads(row[6]) if row[6] else {},
            )
            for row in rows
        ]

    async def get_cross_section(
        self,
        feature_names: List[str],
        symbols: List[str],
        as_of_date: date,
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for multiple symbols at a point in time."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Build query
        placeholders_f = ",".join("?" * len(feature_names))
        placeholders_s = ",".join("?" * len(symbols))

        cursor.execute(f"""
            SELECT feature_name, symbol, value
            FROM feature_values
            WHERE feature_name IN ({placeholders_f})
              AND symbol IN ({placeholders_s})
              AND as_of_date = ?
        """, (*feature_names, *symbols, as_of_date.isoformat()))

        rows = cursor.fetchall()
        conn.close()

        result = {s: {} for s in symbols}
        for feature_name, symbol, value_blob in rows:
            result[symbol][feature_name] = pickle.loads(value_blob)

        return result


class FeatureRegistry:
    """Registry for feature definitions with lineage tracking."""

    def __init__(self, backend: FeatureStorageBackend):
        self.backend = backend
        self._definitions: Dict[str, FeatureDefinition] = {}
        self._by_tag: Dict[str, Set[str]] = {}

    def register(self, definition: FeatureDefinition) -> None:
        """Register a feature definition."""
        key = definition.full_name
        self._definitions[key] = definition

        # Index by tags
        for tag in definition.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = set()
            self._by_tag[tag].add(key)

        logger.info(f"Registered feature: {key}")

    def get(self, name: str, version: Optional[str] = None) -> Optional[FeatureDefinition]:
        """Get a feature definition."""
        if version:
            key = f"{name}:v{version}"
            return self._definitions.get(key)

        # Return latest version
        matching = [k for k in self._definitions if k.startswith(f"{name}:v")]
        if not matching:
            return None
        return self._definitions[sorted(matching)[-1]]

    def get_by_tag(self, tag: str) -> List[FeatureDefinition]:
        """Get features by tag."""
        keys = self._by_tag.get(tag, set())
        return [self._definitions[k] for k in keys]

    def get_dependencies(
        self,
        name: str,
        recursive: bool = True,
    ) -> List[str]:
        """Get feature dependencies."""
        definition = self.get(name)
        if not definition:
            return []

        deps = list(definition.dependencies)

        if recursive:
            for dep in definition.dependencies:
                deps.extend(self.get_dependencies(dep, recursive=True))

        return list(set(deps))

    def validate_dependencies(self, name: str) -> List[str]:
        """Check if all dependencies are registered."""
        missing = []
        for dep in self.get_dependencies(name, recursive=True):
            if not self.get(dep):
                missing.append(dep)
        return missing

    def list_features(
        self,
        feature_type: Optional[FeatureType] = None,
        tag: Optional[str] = None,
    ) -> List[FeatureDefinition]:
        """List registered features."""
        features = list(self._definitions.values())

        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]

        if tag:
            tag_keys = self._by_tag.get(tag, set())
            features = [f for f in features if f.full_name in tag_keys]

        return features


class FeatureComputer:
    """Computes features from raw data."""

    def __init__(
        self,
        registry: FeatureRegistry,
        backend: FeatureStorageBackend,
    ):
        self.registry = registry
        self.backend = backend
        self._compute_cache: Dict[str, Any] = {}

    async def compute_feature(
        self,
        feature_name: str,
        symbol: str,
        as_of_date: date,
        raw_data: Dict[str, Any],
        force_recompute: bool = False,
    ) -> FeatureValue:
        """Compute a single feature value."""
        definition = self.registry.get(feature_name)
        if not definition:
            raise ValueError(f"Feature not registered: {feature_name}")

        # Check cache unless force recompute
        if not force_recompute:
            cached = await self.backend.get_value(
                feature_name, symbol, as_of_date, definition.version
            )
            if cached:
                return cached

        # Compute dependencies first
        dep_values = {}
        for dep in definition.dependencies:
            dep_value = await self.compute_feature(
                dep, symbol, as_of_date, raw_data, force_recompute
            )
            dep_values[dep] = dep_value.value

        # Compute feature
        if definition.compute_fn:
            value = definition.compute_fn(
                symbol=symbol,
                as_of_date=as_of_date,
                raw_data=raw_data,
                dependencies=dep_values,
            )
        else:
            raise ValueError(
                f"No compute function for feature: {feature_name}"
            )

        # Validate
        if not definition.validate_value(value):
            logger.warning(
                f"Feature value failed validation: {feature_name}={value}"
            )

        # Create and store
        feature_value = FeatureValue(
            feature_name=feature_name,
            symbol=symbol,
            as_of_date=as_of_date,
            value=value,
            version=definition.version,
        )

        await self.backend.store_value(feature_value)

        return feature_value

    async def compute_feature_set(
        self,
        feature_names: List[str],
        symbol: str,
        as_of_date: date,
        raw_data: Dict[str, Any],
    ) -> FeatureSet:
        """Compute multiple features for a symbol."""
        features = {}

        for name in feature_names:
            try:
                value = await self.compute_feature(
                    name, symbol, as_of_date, raw_data
                )
                features[name] = value.value
            except Exception as e:
                logger.error(f"Failed to compute {name}: {e}")
                features[name] = None

        return FeatureSet(
            symbol=symbol,
            as_of_date=as_of_date,
            features=features,
        )

    async def compute_cross_section(
        self,
        feature_names: List[str],
        symbols: List[str],
        as_of_date: date,
        raw_data_fn: Callable[[str, date], Dict[str, Any]],
    ) -> Dict[str, FeatureSet]:
        """Compute features for multiple symbols."""
        results = {}

        for symbol in symbols:
            raw_data = raw_data_fn(symbol, as_of_date)
            feature_set = await self.compute_feature_set(
                feature_names, symbol, as_of_date, raw_data
            )
            results[symbol] = feature_set

        return results


class FeatureStore:
    """
    Main interface for feature storage and retrieval.

    Provides:
    - Point-in-time feature access
    - Feature versioning
    - Bulk retrieval for backtesting
    - Lineage tracking
    """

    def __init__(
        self,
        backend: FeatureStorageBackend,
        registry: Optional[FeatureRegistry] = None,
    ):
        self.backend = backend
        self.registry = registry or FeatureRegistry(backend)
        self.computer = FeatureComputer(self.registry, backend)

    def register_feature(self, definition: FeatureDefinition) -> None:
        """Register a feature definition."""
        self.registry.register(definition)

    async def get_feature(
        self,
        feature_name: str,
        symbol: str,
        as_of_date: date,
        version: Optional[str] = None,
    ) -> Optional[Any]:
        """Get a feature value."""
        value = await self.backend.get_value(
            feature_name, symbol, as_of_date, version
        )
        return value.value if value else None

    async def get_features(
        self,
        feature_names: List[str],
        symbol: str,
        as_of_date: date,
    ) -> Dict[str, Any]:
        """Get multiple features for a symbol."""
        result = {}
        for name in feature_names:
            result[name] = await self.get_feature(name, symbol, as_of_date)
        return result

    async def get_feature_history(
        self,
        feature_name: str,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """Get historical feature values as a series."""
        values = await self.backend.get_feature_history(
            feature_name, symbol, start_date, end_date
        )

        if not values:
            return pd.Series(dtype=float)

        return pd.Series(
            {v.as_of_date: v.value for v in values},
            name=feature_name
        )

    async def get_feature_matrix(
        self,
        feature_names: List[str],
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> FeatureMatrix:
        """Get features for multiple symbols across time."""
        # Generate date range
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        # Initialize matrix
        data = np.full(
            (len(dates), len(symbols), len(feature_names)),
            np.nan
        )

        # Fill matrix
        for i, d in enumerate(dates):
            cross_section = await self.backend.get_cross_section(
                feature_names, symbols, d
            )
            for j, symbol in enumerate(symbols):
                symbol_features = cross_section.get(symbol, {})
                for k, feature_name in enumerate(feature_names):
                    value = symbol_features.get(feature_name)
                    if value is not None:
                        data[i, j, k] = value

        return FeatureMatrix(
            symbols=symbols,
            dates=dates,
            feature_names=feature_names,
            data=data,
        )

    async def store_feature(
        self,
        feature_name: str,
        symbol: str,
        as_of_date: date,
        value: Any,
        version: str = "1.0.0",
    ) -> None:
        """Store a feature value."""
        fv = FeatureValue(
            feature_name=feature_name,
            symbol=symbol,
            as_of_date=as_of_date,
            value=value,
            version=version,
        )
        await self.backend.store_value(fv)

    async def store_features_bulk(
        self,
        feature_name: str,
        df: pd.DataFrame,
        version: str = "1.0.0",
    ) -> int:
        """
        Store features from a DataFrame in bulk.

        DataFrame should have:
        - Index: dates
        - Columns: symbols
        - Values: feature values
        """
        values = []

        for d in df.index:
            as_of = d.date() if hasattr(d, 'date') else d
            for symbol in df.columns:
                v = df.loc[d, symbol]
                if pd.notna(v):
                    values.append(FeatureValue(
                        feature_name=feature_name,
                        symbol=symbol,
                        as_of_date=as_of,
                        value=float(v),
                        version=version,
                    ))

        await self.backend.store_values(values)
        return len(values)

    async def compute_and_store(
        self,
        feature_name: str,
        symbol: str,
        as_of_date: date,
        raw_data: Dict[str, Any],
    ) -> Any:
        """Compute a feature and store it."""
        value = await self.computer.compute_feature(
            feature_name, symbol, as_of_date, raw_data
        )
        return value.value

    def get_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get feature lineage (dependencies and sources)."""
        definition = self.registry.get(feature_name)
        if not definition:
            return {}

        deps = self.registry.get_dependencies(feature_name)

        return {
            "feature": feature_name,
            "version": definition.version,
            "dependencies": deps,
            "data_sources": definition.data_sources,
            "compute_frequency": definition.compute_frequency.value,
            "created_at": definition.created_at.isoformat(),
        }


# Pre-built feature definitions for common technical features

def _compute_rsi(
    symbol: str,
    as_of_date: date,
    raw_data: Dict[str, Any],
    dependencies: Dict[str, Any],
    period: int = 14,
) -> float:
    """Compute RSI."""
    prices = raw_data.get("close_prices", [])
    if len(prices) < period + 1:
        return np.nan

    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_momentum(
    symbol: str,
    as_of_date: date,
    raw_data: Dict[str, Any],
    dependencies: Dict[str, Any],
    period: int = 20,
) -> float:
    """Compute price momentum."""
    prices = raw_data.get("close_prices", [])
    if len(prices) < period + 1:
        return np.nan

    return (prices[-1] / prices[-period - 1]) - 1


def _compute_volatility(
    symbol: str,
    as_of_date: date,
    raw_data: Dict[str, Any],
    dependencies: Dict[str, Any],
    period: int = 20,
) -> float:
    """Compute realized volatility."""
    prices = raw_data.get("close_prices", [])
    if len(prices) < period + 1:
        return np.nan

    returns = np.diff(np.log(prices[-period-1:]))
    return np.std(returns) * np.sqrt(252)


# Standard feature definitions
STANDARD_FEATURES = [
    FeatureDefinition(
        name="rsi_14",
        feature_type=FeatureType.TECHNICAL,
        compute_frequency=ComputeFrequency.DAILY,
        description="14-day Relative Strength Index",
        compute_fn=lambda **kw: _compute_rsi(**kw, period=14),
        min_value=0,
        max_value=100,
        tags=["momentum", "oscillator"],
    ),
    FeatureDefinition(
        name="momentum_20",
        feature_type=FeatureType.TECHNICAL,
        compute_frequency=ComputeFrequency.DAILY,
        description="20-day price momentum",
        compute_fn=lambda **kw: _compute_momentum(**kw, period=20),
        tags=["momentum"],
    ),
    FeatureDefinition(
        name="volatility_20",
        feature_type=FeatureType.TECHNICAL,
        compute_frequency=ComputeFrequency.DAILY,
        description="20-day realized volatility (annualized)",
        compute_fn=lambda **kw: _compute_volatility(**kw, period=20),
        min_value=0,
        tags=["risk", "volatility"],
    ),
]


def create_feature_store(
    db_path: str = "features.db",
    register_standard: bool = True,
) -> FeatureStore:
    """
    Factory function to create a FeatureStore.

    Args:
        db_path: Path to SQLite database
        register_standard: Whether to register standard features
    """
    backend = SQLiteFeatureBackend(db_path)
    store = FeatureStore(backend)

    if register_standard:
        for feature in STANDARD_FEATURES:
            store.register_feature(feature)

    return store


def print_feature_report(
    store: FeatureStore,
    feature_name: str,
    symbol: str,
    start_date: date,
    end_date: date,
) -> None:
    """Print feature analysis report."""

    async def _get_data():
        return await store.get_feature_history(
            feature_name, symbol, start_date, end_date
        )

    series = asyncio.run(_get_data())

    print("\n" + "=" * 60)
    print(f"FEATURE REPORT: {feature_name}")
    print("=" * 60)

    print(f"\nSymbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Data Points: {len(series)}")

    if len(series) > 0:
        print(f"\n{'Statistics':-^40}")
        print(f"  Mean: {series.mean():.4f}")
        print(f"  Std: {series.std():.4f}")
        print(f"  Min: {series.min():.4f}")
        print(f"  Max: {series.max():.4f}")
        print(f"  Latest: {series.iloc[-1]:.4f}")

    lineage = store.get_lineage(feature_name)
    if lineage:
        print(f"\n{'Lineage':-^40}")
        print(f"  Version: {lineage.get('version', 'N/A')}")
        print(f"  Dependencies: {lineage.get('dependencies', [])}")
        print(f"  Sources: {lineage.get('data_sources', [])}")

    print("=" * 60 + "\n")
