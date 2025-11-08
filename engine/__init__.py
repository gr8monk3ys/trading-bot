"""
Trading Engine Package

Core components for the trading bot engine including:
- Strategy Manager: Manages multiple trading strategies
- Performance Metrics: Calculates trading performance metrics
- Backtest Engine: Provides backtesting functionality
- Strategy Evaluator: Evaluates and ranks strategy performance
"""

from engine.strategy_manager import StrategyManager
from engine.performance_metrics import PerformanceMetrics
from engine.backtest_engine import BacktestEngine
from engine.strategy_evaluator import StrategyEvaluator

__all__ = [
    'StrategyManager',
    'PerformanceMetrics',
    'BacktestEngine',
    'StrategyEvaluator'
]
