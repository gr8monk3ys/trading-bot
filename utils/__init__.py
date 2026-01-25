"""
Utilities package for the trading bot

This package contains utility functions and classes used throughout the trading bot.
"""

from utils.database import (
    TradingDatabase,
    Trade,
    DailyMetrics,
    Position,
    DatabaseError,
    create_database,
)
from utils.visualization import (
    create_performance_report,
    plot_drawdown_periods,
    plot_equity_curve,
    plot_monthly_returns,
    plot_returns_distribution,
    plot_rolling_performance,
    plot_trade_analysis,
)

__all__ = [
    # Database
    "TradingDatabase",
    "Trade",
    "DailyMetrics",
    "Position",
    "DatabaseError",
    "create_database",
    # Visualization
    "plot_equity_curve",
    "plot_returns_distribution",
    "plot_drawdown_periods",
    "plot_monthly_returns",
    "plot_rolling_performance",
    "plot_trade_analysis",
    "create_performance_report",
]
