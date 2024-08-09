"""
BaseStrategy sub-package.

The original ``strategies/base_strategy.py`` (~875 LOC) was split into focused
modules so future maintenance touches only one concern at a time:

    - strategy.py        — abstract class, init, lifecycle, state/order plumbing
    - position_sizing.py — Kelly Criterion, position-size limits, vol/streak
                           adjustments, position queries (mixin)

The concrete ``BaseStrategy`` class lives in ``strategy.py`` and mixes in
``BasePositionSizingMixin`` from ``position_sizing.py``.

External callers continue to import ``BaseStrategy`` from
``strategies.base_strategy`` — this sub-package is an internal
implementation detail.
"""

from strategies.base.position_sizing import BasePositionSizingMixin
from strategies.base.strategy import BaseStrategy

__all__ = [
    "BaseStrategy",
    "BasePositionSizingMixin",
]
