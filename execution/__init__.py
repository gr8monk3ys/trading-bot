"""
Execution Package - Institutional-Grade Order Execution

Provides:
1. Smart Order Router: Multi-venue order routing
2. Advanced Algorithms: Implementation Shortfall, POV, TWAP, VWAP
3. Low-Latency Framework: Optimized execution infrastructure
"""

from execution.advanced_algos import (
    AdaptiveTWAP,
    AdaptiveVWAP,
    AlgoMetrics,
    AlgoOrder,
    AlgorithmicExecutor,
    AlgoState,
    ExecutionAlgorithm,
    ExecutionSlice,
    ImplementationShortfall,
    POVAlgorithm,
    SweepAlgorithm,
    Urgency,
    create_algo_executor,
)
from execution.low_latency import (
    ConnectionPool,
    ExecutionConfig,
    LatencyBucket,
    LatencyMonitor,
    LatencyOptimizer,
    LatencyStats,
    LowLatencyExecutor,
    OrderMessage,
    OrderQueue,
    create_low_latency_executor,
)
from execution.smart_order_router import (
    RouteResult,
    RoutingDecision,
    RoutingStrategy,
    SmartOrderRouter,
    Venue,
    VenueQuote,
    create_smart_router,
)

__all__ = [
    # Smart Order Router
    "SmartOrderRouter",
    "Venue",
    "VenueQuote",
    "RoutingDecision",
    "RouteResult",
    "RoutingStrategy",
    "create_smart_router",
    # Advanced Algorithms
    "AlgorithmicExecutor",
    "ExecutionAlgorithm",
    "ImplementationShortfall",
    "POVAlgorithm",
    "AdaptiveTWAP",
    "AdaptiveVWAP",
    "SweepAlgorithm",
    "AlgoOrder",
    "AlgoState",
    "AlgoMetrics",
    "ExecutionSlice",
    "Urgency",
    "create_algo_executor",
    # Low-Latency Framework
    "LowLatencyExecutor",
    "LatencyMonitor",
    "LatencyStats",
    "LatencyBucket",
    "ConnectionPool",
    "OrderQueue",
    "OrderMessage",
    "ExecutionConfig",
    "LatencyOptimizer",
    "create_low_latency_executor",
]
