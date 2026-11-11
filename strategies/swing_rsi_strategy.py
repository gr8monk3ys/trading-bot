"""Swing mean reversion: buy short, sharp dips in index ETFs that are in an uptrend.

Why an edge might exist (written down before any backtest, ADR 0013):
forced selling (margin calls, fund outflows, volatility targeting) pushes
index prices below fair value for a few days, and buyers who supply that
liquidity are paid by the bounce. Rules from Connors and Alvarez (2008),
never tuned. Known weakness: the effect is reported to have shrunk after
about 2010, and each trade earns well under 1%, so costs decide it.

Rules (all fixed in ``SwingRSIParams``), evaluated on the last completed
close; the engine fills at the next session's close (one day later than the
book, see ADR 0013):
    * Buy when close > SMA(trend_sma_days) and RSI(rsi_period) < entry_rsi.
    * Sell the whole position when close > SMA(exit_sma_days).
    * Each position is ``target_gross / len(symbols)`` of equity. Long only,
      no stop, no margin.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from engine.order_submission import OrderIntent
from strategies.base_strategy import BaseStrategy
from strategies.params import SwingRSIParams

logger = logging.getLogger(__name__)


def wilder_rsi(close: pd.Series, period: int) -> float:
    """Wilder's RSI of the last value. 100 when there were no down moves."""
    delta = close.diff().dropna()
    if delta.empty:
        return 50.0
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if down <= 0:
        return 100.0 if up > 0 else 50.0
    return float(100 - 100 / (1 + up / down))


class SwingRSIStrategy(BaseStrategy):
    Params = SwingRSIParams
    NAME = "SwingRSI"

    def __init__(
        self, broker=None, parameters: Optional[Dict[str, Any]] = None, order_submission=None
    ):
        super().__init__(broker=broker, parameters=parameters, order_submission=order_submission)
        self.rsi_period = int(self.parameters["rsi_period"])
        self.entry_rsi = float(self.parameters["entry_rsi"])
        self.trend_days = int(self.parameters["trend_sma_days"])
        self.exit_days = int(self.parameters["exit_sma_days"])
        self.target_gross = float(self.parameters["target_gross"])
        # Per symbol: "enter", "exit" or None (no signal / not enough history).
        self.signals: Dict[str, Optional[str]] = {}

    async def initialize(self, **kwargs):
        return True

    async def prepare(self, when, histories) -> None:
        for symbol in self.symbols:
            self.signals[symbol] = None
            df = histories.get(symbol)
            if df is None or len(df) < self.trend_days:
                continue
            close = df["close"].iloc[-(self.trend_days + 60) :].astype(float)
            last = float(close.iloc[-1])
            if last > float(close.iloc[-self.exit_days :].mean()):
                self.signals[symbol] = "exit"
            elif (
                last > float(close.iloc[-self.trend_days :].mean())
                and wilder_rsi(close, self.rsi_period) < self.entry_rsi
            ):
                self.signals[symbol] = "enter"

    async def decide(self, symbol, when, portfolio) -> List[OrderIntent]:
        signal = self.signals.get(symbol)
        held = portfolio.position(symbol)
        qty = int(float(held.qty)) if held is not None else 0
        if signal == "exit" and qty > 0:
            return [
                OrderIntent(
                    symbol=symbol,
                    side="sell",
                    qty=qty,
                    is_exit=True,
                    strategy_name=self.name,
                    reason="close_above_exit_sma",
                )
            ]
        if signal == "enter" and qty == 0:
            price = await portfolio.ask_price(symbol)
            if price <= 0:
                return []
            slot = portfolio.equity * self.target_gross / len(self.symbols)
            buy = min(int(slot / price), int(portfolio.cash * 0.995 / price))
            if buy > 0:
                return [
                    OrderIntent(
                        symbol=symbol,
                        side="buy",
                        qty=buy,
                        strategy_name=self.name,
                        reason="rsi_dip_in_uptrend",
                    )
                ]
        return []

    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        return {
            "action": "neutral",
            "symbol": symbol,
            "strategy": self.NAME,
            "signal": self.signals.get(symbol),
        }

    async def execute_trade(self, symbol, signal):
        """Orders go through decide() and the session (ADR 0001)."""
        return None
