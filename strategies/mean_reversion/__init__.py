"""
MeanReversionStrategy sub-package.

The original ``strategies/mean_reversion_strategy.py`` (~885 LOC) was split
into focused modules:

    - strategy.py — concrete class, init, on_bar, signal-execution dispatch,
                    state/backtest helpers
    - signals.py  — indicator updates, signal generation, smart-exit
                    condition checks (mixin)

External callers continue to import ``MeanReversionStrategy`` from
``strategies.mean_reversion_strategy`` — this sub-package is an internal
implementation detail.
"""

from strategies.mean_reversion.signals import MeanReversionSignalsMixin
from strategies.mean_reversion.strategy import MeanReversionStrategy

__all__ = [
    "MeanReversionStrategy",
    "MeanReversionSignalsMixin",
]
