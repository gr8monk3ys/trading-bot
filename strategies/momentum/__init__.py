"""
MomentumStrategy sub-package.

The original ``strategies/momentum_strategy.py`` (~1229 LOC) was split into
focused modules:

    - strategy.py    — concrete class, init, on_bar, signal-execution
                       dispatch, position-value math, state/backtest helpers
    - indicators.py  — TA-Lib indicator pipeline (RSI / MACD / ADX / SMAs /
                       ATR / Bollinger Bands) — mixin
    - signals.py     — signal-generation heuristic and trailing-stop /
                       exit-condition checks — mixin

External callers continue to import ``MomentumStrategy`` from
``strategies.momentum_strategy`` — this sub-package is an internal
implementation detail.
"""

from strategies.momentum.indicators import MomentumIndicatorsMixin
from strategies.momentum.signals import MomentumSignalsMixin
from strategies.momentum.strategy import MomentumStrategy

__all__ = [
    "MomentumStrategy",
    "MomentumIndicatorsMixin",
    "MomentumSignalsMixin",
]
