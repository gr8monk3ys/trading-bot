"""
BaseStrategy — thin facade re-exporting the abstract class from the
``strategies/base/`` sub-package.

The implementation lives in:

    - strategies/base/strategy.py        — abstract class, init, lifecycle,
                                           state/order plumbing
    - strategies/base/position_sizing.py — Kelly Criterion, position-size
                                           limits, volatility/streak
                                           adjustments, position queries

External callers continue to do ``from strategies.base_strategy import
BaseStrategy`` exactly as before.
"""

from strategies.base import BasePositionSizingMixin, BaseStrategy

__all__ = [
    "BaseStrategy",
    "BasePositionSizingMixin",
]
