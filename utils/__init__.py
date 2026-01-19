"""
Utilities package for the trading bot

This package contains utility functions and classes used throughout the trading bot.
"""

from utils.sentiment_analysis import analyze_sentiment
from utils.stock_scanner import StockScanner
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
    "analyze_sentiment",
    "StockScanner",
    "plot_equity_curve",
    "plot_returns_distribution",
    "plot_drawdown_periods",
    "plot_monthly_returns",
    "plot_rolling_performance",
    "plot_trade_analysis",
    "create_performance_report",
]
