"""Trade recorder: turns order outcomes into completed trades.

Called by OrderSubmission on every fill. Entries open a lot; exits close it
and produce a Trade with realised P&L, which every listener (the strategy's
Kelly estimator, trade history) receives. One writer, many readers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from utils.kelly_criterion import Trade

logger = logging.getLogger(__name__)


@dataclass
class _Lot:
    qty: float  # signed: positive long, negative short
    entry_price: float
    entry_time: datetime


@dataclass
class TradeRecorder:
    trades: List[Trade] = field(default_factory=list)
    _lots: Dict[Tuple[str, str], _Lot] = field(default_factory=dict)
    _listeners: Dict[str, List[Callable[[Trade], None]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def subscribe(self, strategy_name: str, listener: Callable[[Trade], None]) -> None:
        self._listeners[strategy_name].append(listener)

    def subscribe_all(self, listener: Callable[[Trade, str], None]) -> None:
        """Hear every completed trade with its strategy name (trade history uses this)."""
        self._listeners["*"].append(listener)

    def on_outcome(self, intent, outcome, when: Optional[datetime] = None) -> Optional[Trade]:
        if not outcome.ok or outcome.qty_filled <= 0 or outcome.fill_price is None:
            return None
        when = when or datetime.now()
        key = (intent.strategy_name, intent.symbol)
        signed = (
            outcome.qty_filled if intent.side.lower().startswith("buy") else -outcome.qty_filled
        )
        lot = self._lots.get(key)
        if lot is None or (lot.qty > 0) == (signed > 0):
            # Opening or adding: average the entry.
            if lot is None:
                self._lots[key] = _Lot(signed, outcome.fill_price, when)
            else:
                total = lot.qty + signed
                lot.entry_price = (lot.entry_price * lot.qty + outcome.fill_price * signed) / total
                lot.qty = total
            return None
        # Closing (part of) the lot.
        closed = min(abs(signed), abs(lot.qty))
        direction = 1.0 if lot.qty > 0 else -1.0
        pnl = (outcome.fill_price - lot.entry_price) * closed * direction
        cost = lot.entry_price * closed
        trade = Trade(
            symbol=intent.symbol,
            entry_time=lot.entry_time,
            exit_time=when,
            entry_price=lot.entry_price,
            exit_price=outcome.fill_price,
            quantity=closed,
            pnl=pnl,
            pnl_pct=(pnl / cost) if cost else 0.0,
            is_winner=pnl > 0,
        )
        lot.qty -= closed * direction
        if abs(lot.qty) < 1e-9:
            del self._lots[key]
        self.trades.append(trade)
        for listener in self._listeners.get(intent.strategy_name, []):
            try:
                listener(trade)
            except Exception as e:  # a learner must never block the order path
                logger.error(f"trade listener failed: {e}")
        for listener in self._listeners.get("*", []):
            try:
                listener(trade, intent.strategy_name)
            except Exception as e:
                logger.error(f"trade history listener failed: {e}")
        return trade
