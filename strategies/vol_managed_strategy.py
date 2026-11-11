"""Volatility-managed equity: hold less of an index when it has just been volatile.

Why an edge might exist (written down before any backtest, ADR 0012):
volatility clusters, so last month's volatility predicts next month's well,
but expected returns do not rise to match. Cutting exposure after volatile
months therefore cuts risk more than return, which raises return per unit of
exposure (Moreira and Muir 2017, "Volatility-Managed Portfolios"). Known
weakness: later work (Cederburg et al. 2020) finds the gain is unreliable in
real time for many factors, and the market factor is one of the stronger
cases. The pass bar compares against SPY at the same average exposure, which
is exactly the claim.

Rules (all fixed in ``VolManagedParams``):
    * On the first session of each month, for each ETF, weight =
      min(1, target_vol / realised volatility over the last 21 sessions).
    * Each slot is ``target_gross / len(symbols)`` of equity times its weight.
      Long only, no margin. A position is resized only when its target moves
      by more than ``rebalance_band`` of a full slot, to keep costs down.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from engine.order_submission import OrderIntent
from strategies.base_strategy import BaseStrategy
from strategies.params import VolManagedParams

logger = logging.getLogger(__name__)


class VolManagedStrategy(BaseStrategy):
    Params = VolManagedParams
    NAME = "VolManaged"

    def __init__(
        self, broker=None, parameters: Optional[Dict[str, Any]] = None, order_submission=None
    ):
        super().__init__(broker=broker, parameters=parameters, order_submission=order_submission)
        self.lookback = int(self.parameters["vol_lookback_days"])
        self.target_vol = float(self.parameters["target_vol"])
        self.band = float(self.parameters["rebalance_band"])
        self.target_gross = float(self.parameters["target_gross"])
        # Fraction of a full slot to hold; None = not enough history yet.
        self.weights: Dict[str, Optional[float]] = {}
        self._last_rebalance: Optional[tuple] = None
        self._rebalance_today = False
        self._reductions_sent = False

    async def initialize(self, **kwargs):
        return True

    async def prepare(self, when, histories) -> None:
        month = (when.year, when.month)
        self._rebalance_today = month != self._last_rebalance
        self._reductions_sent = False
        if not self._rebalance_today:
            return
        ready = 0
        for symbol in self.symbols:
            df = histories.get(symbol)
            if df is None or len(df) <= self.lookback:
                self.weights[symbol] = None
                continue
            r = df["close"].pct_change().iloc[-self.lookback :]
            vol = float(r.std()) * math.sqrt(252)
            self.weights[symbol] = 1.0 if vol <= 0 else min(1.0, self.target_vol / vol)
            ready += 1
        if ready:
            self._last_rebalance = month
        else:
            self._rebalance_today = False

    def _target_qty(self, symbol: str, equity: float, price: float) -> Optional[int]:
        w = self.weights.get(symbol)
        if w is None or price <= 0:
            return None
        return int(equity * self.target_gross / len(self.symbols) * w / price)

    def _band_qty(self, equity: float, price: float) -> float:
        return equity * self.target_gross / len(self.symbols) / price * self.band

    async def decide(self, symbol, when, portfolio) -> List[OrderIntent]:
        if not self._rebalance_today:
            return []
        intents: List[OrderIntent] = []
        # Reductions for every symbol go out first so their cash funds the increases.
        if not self._reductions_sent:
            self._reductions_sent = True
            for s in self.symbols:
                held = portfolio.position(s)
                qty = int(float(held.qty)) if held is not None else 0
                if qty <= 0:
                    continue
                price = await portfolio.ask_price(s)
                target = self._target_qty(s, portfolio.equity, price)
                if target is None:
                    continue
                if qty - target > self._band_qty(portfolio.equity, price):
                    intents.append(
                        OrderIntent(
                            symbol=s,
                            side="sell",
                            qty=qty - target,
                            is_exit=True,
                            strategy_name=self.name,
                            reason="vol_reduce",
                        )
                    )
            if intents:
                return intents

        held = portfolio.position(symbol)
        qty = int(float(held.qty)) if held is not None else 0
        price = await portfolio.ask_price(symbol)
        target = self._target_qty(symbol, portfolio.equity, price)
        if target is None:
            return intents
        add = target - qty
        if add > self._band_qty(portfolio.equity, price) or (qty == 0 and add > 0):
            add = min(add, int(portfolio.cash * 0.995 / price))
            if add > 0:
                intents.append(
                    OrderIntent(
                        symbol=symbol,
                        side="buy",
                        qty=add,
                        strategy_name=self.name,
                        reason="vol_increase",
                    )
                )
        return intents

    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        w = self.weights.get(symbol)
        return {"action": "neutral", "symbol": symbol, "strategy": self.NAME, "weight": w}

    async def execute_trade(self, symbol, signal):
        """Orders go through decide() and the session (ADR 0001)."""
        return None
