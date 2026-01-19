"""
Simple Moving Average Crossover Strategy for backtest validation.

This is a basic strategy used primarily to validate the backtesting engine works.
Uses a simple dual moving average crossover with minimal filters.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class SimpleMACrossoverStrategy(BaseStrategy):
    """
    Simple dual moving average crossover strategy.

    Buy when fast MA crosses above slow MA.
    Sell when fast MA crosses below slow MA.

    This is intentionally simple to validate the backtest engine.
    """

    NAME = "SimpleMACrossover"

    def __init__(self, broker=None, parameters: Dict[str, Any] = None):
        """Initialize the strategy."""
        super().__init__(broker=broker, parameters=parameters)

        # Simple parameters
        self.fast_period = self.parameters.get('fast_period', 10)
        self.slow_period = self.parameters.get('slow_period', 30)
        self.min_history = self.slow_period + 5

        # State tracking
        self.signals: Dict[str, str] = {}
        self.previous_crossover: Dict[str, Optional[str]] = {}

        logger.info(f"SimpleMACrossover initialized: fast={self.fast_period}, slow={self.slow_period}")

    async def initialize(self):
        """Initialize strategy state."""
        symbols = self.parameters.get('symbols', [])
        for symbol in symbols:
            self.signals[symbol] = 'neutral'
            self.previous_crossover[symbol] = None
        logger.info(f"SimpleMACrossover: Tracking {len(symbols)} symbols")

    async def generate_signals(self):
        """Generate signals for all symbols."""
        symbols = self.parameters.get('symbols', [])
        for symbol in symbols:
            await self._update_signal(symbol)

    async def _update_signal(self, symbol: str):
        """Update signal for a single symbol."""
        try:
            # Get price data from current_data (set by backtest engine)
            if not hasattr(self, 'current_data') or symbol not in self.current_data:
                self.signals[symbol] = 'neutral'
                return

            df = self.current_data[symbol]
            if len(df) < self.min_history:
                self.signals[symbol] = 'neutral'
                return

            # Calculate moving averages
            closes = df['close'].values
            fast_ma = np.mean(closes[-self.fast_period:])
            slow_ma = np.mean(closes[-self.slow_period:])

            # Determine crossover state
            if fast_ma > slow_ma:
                current_state = 'bullish'
            else:
                current_state = 'bearish'

            # Generate signal on state change
            prev_state = self.previous_crossover.get(symbol)

            if prev_state == 'bearish' and current_state == 'bullish':
                self.signals[symbol] = 'buy'
                logger.debug(f"{symbol}: MA crossover BUY signal (fast={fast_ma:.2f} > slow={slow_ma:.2f})")
            elif prev_state == 'bullish' and current_state == 'bearish':
                self.signals[symbol] = 'sell'
                logger.debug(f"{symbol}: MA crossover SELL signal (fast={fast_ma:.2f} < slow={slow_ma:.2f})")
            else:
                self.signals[symbol] = 'neutral'

            self.previous_crossover[symbol] = current_state

        except Exception as e:
            logger.error(f"Error updating signal for {symbol}: {e}")
            self.signals[symbol] = 'neutral'

    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze a symbol and return trading signal.

        Returns:
            Signal dict with 'action' key
        """
        signal = self.signals.get(symbol, 'neutral')

        return {
            'action': signal,
            'symbol': symbol,
            'strategy': self.NAME
        }

    async def execute_trade(self, symbol: str, signal: Dict[str, Any]):
        """Execute a trade based on signal."""
        action = signal.get('action', 'neutral')

        if action == 'neutral':
            return

        try:
            # Get current position
            position = None
            # Use async method if available
            if hasattr(self.broker, 'get_all_positions'):
                positions = await self.broker.get_all_positions()
            else:
                positions = self.broker.get_positions() if hasattr(self.broker, 'get_positions') else []
            for pos in positions:
                pos_symbol = pos.get('symbol') if isinstance(pos, dict) else getattr(pos, 'symbol', None)
                if pos_symbol == symbol:
                    position = pos
                    break

            # Get account info
            account = await self.broker.get_account()
            cash = float(account.cash)

            # Simple position sizing: 20% of cash per position
            position_size = cash * 0.20

            # Get current price
            quote = await self.broker.get_latest_quote(symbol)
            price = float(quote.ask_price)

            if action == 'buy' and position is None:
                # Open new position
                qty = int(position_size / price)
                if qty > 0:
                    await self._place_order(symbol, qty, 'buy')
                    logger.info(f"BUY {qty} shares of {symbol} @ ${price:.2f}")

            elif action == 'sell' and position is not None:
                # Close position
                if isinstance(position, dict):
                    pos_qty = int(position.get('quantity', 0))
                else:
                    pos_qty = int(getattr(position, 'quantity', 0) or float(getattr(position, 'qty', 0)))
                if pos_qty > 0:
                    await self._place_order(symbol, pos_qty, 'sell')
                    logger.info(f"SELL {pos_qty} shares of {symbol} @ ${price:.2f}")

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")

    async def _place_order(self, symbol: str, qty: int, side: str):
        """Place an order through the broker."""
        try:
            # Create simple order request object
            class SimpleOrder:
                def __init__(self, sym, q, s):
                    self.symbol = sym
                    self.qty = q
                    self.side = s
                    self.type = 'market'

            order = SimpleOrder(symbol, qty, side)
            await self.broker.submit_order_advanced(order)

        except Exception as e:
            logger.error(f"Error placing order: {e}")
