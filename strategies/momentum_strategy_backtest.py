"""
MomentumStrategy configured for backtesting with daily data.

This version disables features that require intraday data and uses
less strict parameters to generate realistic trade signals.
"""

import logging
from datetime import datetime
from strategies.momentum_strategy import MomentumStrategy
from brokers.order_builder import OrderBuilder

logger = logging.getLogger(__name__)


class MomentumStrategyBacktest(MomentumStrategy):
    """
    MomentumStrategy variant optimized for daily-data backtesting.

    Key differences from production MomentumStrategy:
    - Uses standard RSI-14 with 30/70 thresholds (not RSI-2 aggressive)
    - Disables multi-timeframe filtering (requires intraday data)
    - Lower volume threshold for confirmation
    - Disabled strict timeframe alignment

    This allows realistic backtesting while maintaining the core momentum logic.
    """

    NAME = "MomentumStrategyBacktest"

    def default_parameters(self):
        """Override parameters for backtesting with daily data."""
        # Get base parameters from parent class
        params = MomentumStrategy.default_parameters(self)

        # === BACKTEST-FRIENDLY OVERRIDES ===

        # Use standard RSI-14 with 30/70 thresholds
        # RSI-2 with 10/90 is too aggressive for daily data
        params['rsi_mode'] = 'standard'
        params['rsi_period'] = 14
        params['rsi_overbought'] = 70
        params['rsi_oversold'] = 30

        # Disable multi-timeframe filtering (requires intraday data)
        params['use_multi_timeframe'] = False

        # Lower volume threshold (1.5x is too strict for daily data)
        params['volume_factor'] = 1.2

        # Lower ADX threshold for more signals
        params['adx_threshold'] = 20

        # Disable strict alignment (not available in daily backtest)
        params['mtf_require_alignment'] = False

        # Keep other features enabled
        params['use_bollinger_filter'] = True
        params['enable_short_selling'] = True
        params['use_kelly_criterion'] = False  # Disable for clean backtest
        params['use_volatility_regime'] = False  # Simpler for backtesting
        params['use_streak_sizing'] = False  # Use fixed sizing

        logger.info("MomentumStrategyBacktest: Using daily-data optimized parameters")
        logger.info(f"  RSI mode: standard (14-period, 30/70 thresholds)")
        logger.info(f"  Multi-timeframe: disabled")
        logger.info(f"  Volume factor: 1.2x")
        logger.info(f"  ADX threshold: 20")

        return params

    async def execute_trade(self, symbol, signal):
        """
        Execute a trade based on the signal.

        Override of base MomentumStrategy which has empty execute_trade.
        This version implements actual trade execution for backtesting.
        """
        try:
            # Handle both string and dict signal formats
            if isinstance(signal, str):
                action = signal
            else:
                action = signal.get('action', 'neutral') if signal else 'neutral'

            if action in ['neutral', 'hold', None]:
                return

            # Get current position
            if hasattr(self.broker, 'get_all_positions'):
                positions = await self.broker.get_all_positions()
            else:
                positions = self.broker.get_positions() if hasattr(self.broker, 'get_positions') else []

            current_position = None
            for pos in positions:
                pos_symbol = getattr(pos, 'symbol', None) or pos.get('symbol') if isinstance(pos, dict) else None
                if pos_symbol == symbol:
                    current_position = pos
                    break

            # Get account info
            account = await self.broker.get_account()
            cash = float(account.cash)

            # Get current price
            quote = await self.broker.get_latest_quote(symbol)
            price = float(quote.ask_price)

            # Calculate position size (10% of cash)
            position_value = cash * 0.10
            qty = int(position_value / price)

            if qty <= 0:
                return

            # Execute based on action
            if action == 'buy' and current_position is None:
                # Open long position
                await self._place_backtest_order(symbol, qty, 'buy')
                logger.info(f"BUY {qty} shares of {symbol} @ ${price:.2f}")

            elif action == 'short' and current_position is None:
                # Open short position (simulate with sell)
                await self._place_backtest_order(symbol, qty, 'sell')
                logger.info(f"SHORT {qty} shares of {symbol} @ ${price:.2f}")

            elif action == 'sell' and current_position is not None:
                # Close position
                pos_qty = int(getattr(current_position, 'quantity', 0) or float(getattr(current_position, 'qty', 0)))
                if pos_qty > 0:
                    await self._place_backtest_order(symbol, pos_qty, 'sell')
                    logger.info(f"SELL {pos_qty} shares of {symbol} @ ${price:.2f}")

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")

    async def _place_backtest_order(self, symbol, qty, side):
        """Place an order through the backtest broker."""
        try:
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
