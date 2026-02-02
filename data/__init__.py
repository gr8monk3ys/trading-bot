"""
Data Package - Institutional-Grade Data Infrastructure

Provides:
1. Point-in-Time Database: Survivorship-bias-free historical data
2. Tick Data Integration: Microstructure data from Polygon, TAQ
3. Feature Store: Versioned pre-computed features
"""

from data.point_in_time import (
    PointInTimeDB,
    PITDataPoint,
    PITQueryResult,
    DataField,
    CorporateEvent,
    create_pit_db,
)

from data.tick_data import (
    Trade,
    Quote,
    AggregatedBar,
    MicrostructureSnapshot,
    TickDataProvider,
    PolygonTickProvider,
    TAQDataParser,
    TickAggregator,
    TickDataManager,
    create_tick_manager,
    Exchange,
    TickType,
)

from data.feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureValue,
    FeatureSet,
    FeatureMatrix,
    FeatureRegistry,
    FeatureComputer,
    FeatureType,
    ComputeFrequency,
    SQLiteFeatureBackend,
    create_feature_store,
    STANDARD_FEATURES,
)

from data.alt_data_types import (
    AltDataSource,
    SignalDirection,
    SignalStrength,
    AlternativeSignal,
    SocialSentimentSignal,
    OrderFlowSignal,
    WebScrapingSignal,
    AggregatedSignal,
    AltDataProviderStatus,
)

from data.alternative_data_provider import (
    AlternativeDataProvider,
    AltDataAggregator,
    AltDataCache,
)

from data.web_scraper import (
    JobPostingsProvider,
    GlassdoorSentimentProvider,
    AppRankingsProvider,
    WebScraperAggregator,
    create_web_scraper_provider,
)

from data.cross_asset_types import (
    CrossAssetSource,
    VolatilityRegime,
    YieldCurveRegime,
    RiskAppetiteRegime,
    VixTermStructureSignal,
    YieldCurveSignal,
    FxCorrelationSignal,
    CrossAssetAggregatedSignal,
)

from data.cross_asset_provider import (
    CrossAssetProvider,
    VixTermStructureProvider,
    YieldCurveProvider,
    FxCorrelationProvider,
    CrossAssetAggregator,
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
