"""Time-series trend following across a diversified ETF universe.

Why an edge might exist (written down before any backtest, ADR 0011):
assets that rose over the past 12 months tend to keep rising over the next
month, across stocks, bonds, commodities and currencies. The effect is
documented out of sample over a century of data (Moskowitz, Ooi and Pedersen
2012; Hurst, Ooi and Pedersen 2017). The usual explanations are slow
reaction to news and hedging demand. Its known shape: it lags in steady bull
markets and earns most of its keep in long sell-offs. So the realistic hope
is "SPY-like return per unit of exposure with smaller drawdowns", not
"beats SPY outright".

Rules (all fixed in ``TrendParams``):
    * On the first session of each month, for each ETF, hold it if its
      12-month price return is above zero, otherwise hold nothing in its slot.
    * Each slot is ``target_gross / len(symbols)`` of equity, long only, no
      margin, no shorts. Open positions are not resized month to month.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from engine.order_submission import OrderIntent
from strategies.base_strategy import BaseStrategy
from strategies.params import TrendParams

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    Params = TrendParams
    NAME = "TrendFollowing"

    def __init__(
        self, broker=None, parameters: Optional[Dict[str, Any]] = None, order_submission=None
    ):
        super().__init__(broker=broker, parameters=parameters, order_submission=order_submission)
        self.lookback = int(self.parameters["lookback_days"])
        self.target_gross = float(self.parameters["target_gross"])
        # True = in trend, False = out, None = not enough history yet.
        self.signals: Dict[str, Optional[bool]] = {}
        self._last_rebalance: Optional[tuple] = None
        self._rebalance_today = False
        self._exits_sent = False

    async def initialize(self, **kwargs):
        return True

    async def prepare(self, when, histories) -> None:
        """On the first session of a month, compute each symbol's 12-month trend."""
        month = (when.year, when.month)
        self._rebalance_today = month != self._last_rebalance
        self._exits_sent = False
        if not self._rebalance_today:
            return
        ready = 0
        for symbol in self.symbols:
            df = histories.get(symbol)
            if df is None or len(df) <= self.lookback:
                self.signals[symbol] = None
                continue
            closes = df["close"]
            past = float(closes.iloc[-1 - self.lookback])
            self.signals[symbol] = past > 0 and float(closes.iloc[-1]) / past - 1 > 0
            ready += 1
        if ready:
            self._last_rebalance = month
        else:
            # Still warming up: try again next session.
            self._rebalance_today = False

    def _exit(self, symbol: str, qty: int) -> OrderIntent:
        return OrderIntent(
            symbol=symbol,
            side="sell",
            qty=qty,
            is_exit=True,
            strategy_name=self.name,
            reason="trend_off",
        )

    async def decide(self, symbol, when, portfolio) -> List[OrderIntent]:
        if not self._rebalance_today:
            return []
        intents: List[OrderIntent] = []
        # The session asks symbol by symbol; sells for every symbol go out on
        # the first call so the cash they free is there for the buys.
        if not self._exits_sent:
            self._exits_sent = True
            for s in self.symbols:
                held = portfolio.position(s)
                qty = int(float(held.qty)) if held is not None else 0
                if qty > 0 and self.signals.get(s) is False:
                    intents.append(self._exit(s, qty))
            if intents:
                return intents  # the session refreshes the portfolio before the next symbol

        if self.signals.get(symbol) is not True:
            return intents
        held = portfolio.position(symbol)
        if held is not None and int(float(held.qty)) != 0:
            return intents
        price = await portfolio.ask_price(symbol)
        if price <= 0:
            return intents
        slot = portfolio.equity * self.target_gross / len(self.symbols)
        qty = int(min(slot, portfolio.cash * 0.995) / price)
        if qty > 0:
            intents.append(
                OrderIntent(
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    strategy_name=self.name,
                    reason="trend_on",
                )
            )
        return intents

    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        sig = self.signals.get(symbol)
        action = "buy" if sig is True else "sell" if sig is False else "neutral"
        return {"action": action, "symbol": symbol, "strategy": self.NAME}

    async def execute_trade(self, symbol, signal):
        """Orders go through decide() and the session (ADR 0001)."""
        return None
