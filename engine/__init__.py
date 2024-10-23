"""
Trading Engine package.

Exports are loaded lazily to avoid import-time side effects when tools
(like coverage collection) import submodules by dotted path.
"""

from importlib import import_module

__all__ = ["StrategyManager", "PerformanceMetrics", "BacktestEngine", "StrategyEvaluator"]

_EXPORTS = {
    "StrategyManager": ("engine.strategy_manager", "StrategyManager"),
    "PerformanceMetrics": ("engine.performance_metrics", "PerformanceMetrics"),
    "BacktestEngine": ("engine.backtest_engine", "BacktestEngine"),
    "StrategyEvaluator": ("engine.strategy_evaluator", "StrategyEvaluator"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
