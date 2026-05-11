"""
MomentumStrategy — thin facade re-exporting the concrete class from the
``strategies/momentum/`` sub-package.

The implementation lives in:

    - strategies/momentum/strategy.py   — concrete class, init, on_bar,
                                          signal-execution dispatch,
                                          position-value math,
                                          state/backtest helpers
    - strategies/momentum/indicators.py — TA-Lib indicator pipeline
                                          (RSI / MACD / ADX / SMAs / ATR /
                                          Bollinger Bands)
    - strategies/momentum/signals.py    — signal-generation heuristic and
                                          trailing-stop / exit-condition
                                          checks

External callers continue to do ``from strategies.momentum_strategy import
MomentumStrategy`` exactly as before.
"""

from strategies.momentum import (
    MomentumIndicatorsMixin,
    MomentumSignalsMixin,
    MomentumStrategy,
)

__all__ = [
    "MomentumStrategy",
    "MomentumIndicatorsMixin",
    "MomentumSignalsMixin",
]
