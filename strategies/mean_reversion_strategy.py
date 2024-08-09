"""
MeanReversionStrategy — thin facade re-exporting the concrete class from the
``strategies/mean_reversion/`` sub-package.

The implementation lives in:

    - strategies/mean_reversion/strategy.py — concrete class, init, on_bar,
                                              signal-execution dispatch,
                                              state/backtest helpers
    - strategies/mean_reversion/signals.py  — indicator updates, signal
                                              generation, smart-exit checks

External callers continue to do ``from strategies.mean_reversion_strategy
import MeanReversionStrategy`` exactly as before.
"""

from strategies.mean_reversion import MeanReversionSignalsMixin, MeanReversionStrategy

__all__ = [
    "MeanReversionStrategy",
    "MeanReversionSignalsMixin",
]
