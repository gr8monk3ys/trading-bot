"""
Simple Moving Average Crossover Strategy for backtest validation.

This is a basic strategy used primarily to validate the backtesting engine works.
Uses a simple dual moving average crossover with minimal filters.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from engine.order_submission import OrderIntent
from strategies.base_strategy import BaseStrategy
from strategies.params import SimpleMAParams

logger = logging.getLogger(__name__)


class SimpleMACrossoverStrategy(BaseStrategy):
    Params = SimpleMAParams
    """
    Simple dual moving average crossover strategy.

    Buy when fast MA crosses above slow MA.
    Sell when fast MA crosses below slow MA.

    This is intentionally simple to validate the backtest engine.
    """

    NAME = "SimpleMACrossover"

    def __init__(self, broker=None, parameters: Dict[str, Any] = None, order_submission=None):
        """Initialize the strategy."""
        super().__init__(broker=broker, parameters=parameters, order_submission=order_submission)

        # Simple parameters
        self.fast_period = self.parameters["fast_period"]
        self.slow_period = self.parameters["slow_period"]
        self.min_history = self.slow_period + 5

        # State tracking
        self.signals: Dict[str, str] = {}
        self.previous_crossover: Dict[str, Optional[str]] = {}

        logger.info(
            f"SimpleMACrossover initialized: fast={self.fast_period}, slow={self.slow_period}"
        )

    async def initialize(self):
        """Initialize strategy state."""
        symbols = self.parameters["symbols"]
        for symbol in symbols:
            self.signals[symbol] = "neutral"
            self.previous_crossover[symbol] = None
        logger.info(f"SimpleMACrossover: Tracking {len(symbols)} symbols")

    async def generate_signals(self):
        """Generate signals for all symbols."""
        symbols = self.parameters["symbols"]
        for symbol in symbols:
            await self._update_signal(symbol)

    async def _update_signal(self, symbol: str):
        """Update signal for a single symbol."""
        try:
            # Get price data from current_data (set by backtest engine)
            if not hasattr(self, "current_data") or symbol not in self.current_data:
                self.signals[symbol] = "neutral"
                return

            df = self.current_data[symbol]
            if len(df) < self.min_history:
                self.signals[symbol] = "neutral"
                return

            # Calculate moving averages
            closes = df["close"].values
            fast_ma = np.mean(closes[-self.fast_period :])
            slow_ma = np.mean(closes[-self.slow_period :])

            # Determine crossover state
            if fast_ma > slow_ma:
                current_state = "bullish"
            else:
                current_state = "bearish"

            # Generate signal on state change
            prev_state = self.previous_crossover.get(symbol)

            if prev_state == "bearish" and current_state == "bullish":
                self.signals[symbol] = "buy"
                logger.debug(
                    f"{symbol}: MA crossover BUY signal (fast={fast_ma:.2f} > slow={slow_ma:.2f})"
                )
            elif prev_state == "bullish" and current_state == "bearish":
                self.signals[symbol] = "sell"
                logger.debug(
                    f"{symbol}: MA crossover SELL signal (fast={fast_ma:.2f} < slow={slow_ma:.2f})"
                )
            else:
                self.signals[symbol] = "neutral"

            self.previous_crossover[symbol] = current_state

        except Exception as e:
            logger.error(f"Error updating signal for {symbol}: {e}")
            self.signals[symbol] = "neutral"

    async def decide(self, symbol, when, portfolio):
        """Buy 20% of cash on a bullish crossover when flat; sell the position on a bearish one."""
        action = self._signal_action(symbol)
        if action == "neutral":
            return []
        held = portfolio.position(symbol)
        price = await portfolio.ask_price(symbol)
        name = getattr(self, "name", self.__class__.__name__)
        if action == "buy" and held is None:
            qty = int(portfolio.cash * 0.20 / price)
            if qty > 0:
                return [
                    OrderIntent(
                        symbol=symbol,
                        side="buy",
                        qty=qty,
                        strategy_name=name,
                        reason="simple_ma_entry",
                    )
                ]
        elif action == "sell" and held is not None and int(held.qty) > 0:
            return [
                OrderIntent(
                    symbol=symbol,
                    side="sell",
                    qty=int(held.qty),
                    is_exit=True,
                    strategy_name=name,
                    reason="simple_ma_exit",
                )
            ]
        return []

    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze a symbol and return trading signal.

        Returns:
            Signal dict with 'action' key
        """
        signal = self.signals.get(symbol, "neutral")

        return {"action": signal, "symbol": symbol, "strategy": self.NAME}
