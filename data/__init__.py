"""
Data Package - Institutional-Grade Data Infrastructure

Provides:
1. Point-in-Time Database: Survivorship-bias-free historical data
2. Tick Data Integration: Microstructure data from Polygon, TAQ
3. Feature Store: Versioned pre-computed features
"""

from data.alt_data_types import (
    AggregatedSignal,
    AltDataProviderStatus,
    AltDataSource,
    AlternativeSignal,
    OrderFlowSignal,
    SignalDirection,
    SignalStrength,
    SocialSentimentSignal,
    WebScrapingSignal,
)
from data.alternative_data_provider import (
    AltDataAggregator,
    AltDataCache,
    AlternativeDataProvider,
)
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
from data.web_scraper import (
    AppRankingsProvider,
    GlassdoorSentimentProvider,
    JobPostingsProvider,
    WebScraperAggregator,
    create_web_scraper_provider,
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
    "STANDARD_FEATURES",
    # Alternative Data
    "AltDataSource",
    "SignalDirection",
    "SignalStrength",
    "AlternativeSignal",
    "SocialSentimentSignal",
    "OrderFlowSignal",
    "WebScrapingSignal",
    "AggregatedSignal",
    "AltDataProviderStatus",
    "AlternativeDataProvider",
    "AltDataAggregator",
    "AltDataCache",
    # Web Scraper Providers
    "JobPostingsProvider",
    "GlassdoorSentimentProvider",
    "AppRankingsProvider",
    "WebScraperAggregator",
    "create_web_scraper_provider",
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
