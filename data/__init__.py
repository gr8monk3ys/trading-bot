"""
Data Package - Data Infrastructure

Provides:
- Point-in-Time Database: Survivorship-bias-free historical data
- Tick Data Integration: Microstructure data from Polygon, TAQ
- Feature Store: Versioned pre-computed features
- Cross-Asset Signals: VIX term structure, yield curve, FX (research-only)

The alternative-data framework (Reddit/order-flow/web-scraping) and the
LLM analysis pipeline (earnings/Fed/SEC/news) were removed in the 2026-05
cleanup; they had no validated edge.
"""

from data.cross_asset_provider import (
    CrossAssetAggregator,
    CrossAssetProvider,
    FxCorrelationProvider,
    VixTermStructureProvider,
    YieldCurveProvider,
)
from data.cross_asset_types import (
    CrossAssetAggregatedSignal,
    CrossAssetSource,
    FxCorrelationSignal,
    RiskAppetiteRegime,
    VixTermStructureSignal,
    VolatilityRegime,
    YieldCurveRegime,
    YieldCurveSignal,
)
from data.feature_store import (
    STANDARD_FEATURES,
    ComputeFrequency,
    FeatureComputer,
    FeatureDefinition,
    FeatureMatrix,
    FeatureRegistry,
    FeatureSet,
    FeatureStore,
    FeatureType,
    FeatureValue,
    SQLiteFeatureBackend,
    create_feature_store,
)
from data.point_in_time import (
    CorporateEvent,
    DataField,
    PITDataPoint,
    PITQueryResult,
    PointInTimeDB,
    create_pit_db,
)
from data.tick_data import (
    AggregatedBar,
    Exchange,
    MicrostructureSnapshot,
    PolygonTickProvider,
    Quote,
    TAQDataParser,
    TickAggregator,
    TickDataManager,
    TickDataProvider,
    TickType,
    Trade,
    create_tick_manager,
)

__all__ = [
    # Point-in-Time Database
    "PointInTimeDB",
    "PITDataPoint",
    "PITQueryResult",
    "DataField",
    "CorporateEvent",
    "create_pit_db",
    # Tick Data
    "Trade",
    "Quote",
    "AggregatedBar",
    "MicrostructureSnapshot",
    "TickDataProvider",
    "PolygonTickProvider",
    "TAQDataParser",
    "TickAggregator",
    "TickDataManager",
    "create_tick_manager",
    "Exchange",
    "TickType",
    # Feature Store
    "FeatureStore",
    "FeatureDefinition",
    "FeatureValue",
    "FeatureSet",
    "FeatureMatrix",
    "FeatureRegistry",
    "FeatureComputer",
    "FeatureType",
    "ComputeFrequency",
    "SQLiteFeatureBackend",
    "create_feature_store",
    # Cross-Asset Signals
    "CrossAssetSource",
    "VolatilityRegime",
    "YieldCurveRegime",
    "RiskAppetiteRegime",
    "VixTermStructureSignal",
    "YieldCurveSignal",
    "FxCorrelationSignal",
    "CrossAssetAggregatedSignal",
    "CrossAssetProvider",
    "VixTermStructureProvider",
    "YieldCurveProvider",
    "FxCorrelationProvider",
    "CrossAssetAggregator",
]
