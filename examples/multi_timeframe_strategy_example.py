#!/usr/bin/env python3
"""
Multi-Timeframe Strategy Example

Demonstrates how to use multi-timeframe analysis for better trading signals.

Strategy Logic:
1. Long-term timeframe (1Hour) determines overall trend direction
2. Medium-term timeframe (5Min) confirms trend
3. Short-term timeframe (1Min) times entries

Only enters trades when all timeframes are aligned (multi-timeframe confirmation).
"""

import logging
import asyncio
from strategies.base_strategy import BaseStrategy
from brokers.order_builder import OrderBuilder
from utils.multi_timeframe import MultiTimeframeAnalyzer

logger = logging.getLogger(__name__)


class MultiTimeframeStrategy(BaseStrategy):
    """
    Example strategy using multi-timeframe analysis.

    Buys when:
    - 1Hour trend is bullish (long-term uptrend)
    - 5Min trend is bullish (medium-term confirmation)
    - 1Min shows bullish momentum (short-term entry timing)

    Sells when any timeframe turns bearish.
    """

    def default_parameters(self):
        """Default strategy parameters."""
        return {
            'position_size': 0.10,  # 10% of capital per position
            'stop_loss': 0.02,      # 2% stop loss
            'take_profit': 0.05,    # 5% take profit

            # Multi-timeframe settings
            'timeframes': ['1Min', '5Min', '1Hour'],
            'require_full_alignment': True,  # All timeframes must agree
        }

    async def initialize(self, **kwargs):
        """Initialize the multi-timeframe strategy."""
        try:
            # Initialize base strategy
            await super().initialize(**kwargs)

            # Set parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            self.position_size = self.parameters['position_size']
            self.stop_loss = self.parameters['stop_loss']
            self.take_profit = self.parameters['take_profit']
            self.require_full_alignment = self.parameters['require_full_alignment']

            # Initialize multi-timeframe analyzer
            self.mtf = MultiTimeframeAnalyzer(
                timeframes=self.parameters['timeframes'],
                history_length=200
            )

            # Track current positions
            self.current_positions = set()
            self.current_prices = {}

            # Add strategy as subscriber to broker
            if hasattr(self.broker, '_add_subscriber'):
                self.broker._add_subscriber(self)

            logger.info("Multi-timeframe strategy initialized")
            logger.info(f"Timeframes: {', '.join(self.parameters['timeframes'])}")
            logger.info(f"Require full alignment: {self.require_full_alignment}")

            return True

        except Exception as e:
            logger.error(f"Error initializing multi-timeframe strategy: {e}", exc_info=True)
            return False

    async def on_bar(self, symbol, open_price, high_price, low_price, close_price, volume, timestamp):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return

            # Update current price
            self.current_prices[symbol] = close_price

            # Update multi-timeframe analyzer
            await self.mtf.update(symbol, timestamp, close_price, volume)

            # Check if we have enough data
            status = self.mtf.get_status(symbol)
            if 'error' in status:
                return

            # Get aligned signal
            aligned_signal = status['aligned_signal']

            # Log multi-timeframe status periodically
            if timestamp.second == 0:  # Log every minute
                logger.info(f"\n{symbol} Multi-Timeframe Status:")
                for tf, data in status['timeframes'].items():
                    logger.info(f"  {tf}: {data['trend']} (momentum: {data['momentum']:.2f}%)")
                logger.info(f"  Aligned Signal: {aligned_signal}")
                if status['divergence']:
                    logger.info(f"  ⚠️  Divergence: {status['divergence']}")

            # Get current position
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            # Trading logic
            if aligned_signal == 'bullish' and not current_position:
                await self._execute_buy(symbol, close_price, status)
            elif (aligned_signal == 'bearish' or aligned_signal == 'neutral') and current_position:
                await self._execute_sell(symbol, current_position, status)

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    async def _execute_buy(self, symbol, price, mtf_status):
        """Execute buy order with multi-timeframe confirmation."""
        try:
            # CRITICAL SAFETY: Check if trading is allowed
            if not await self.check_trading_allowed():
                logger.warning("Trading halted by circuit breaker")
                return

            # Get account info
            account = await self.broker.get_account()
            buying_power = float(account.buying_power)

            # Calculate position size
            position_value = buying_power * self.position_size

            # CRITICAL SAFETY: Enforce maximum position size limit (5% of portfolio)
            position_value, quantity = await self.enforce_position_size_limit(symbol, position_value, price)

            # Allow fractional shares (Alpaca minimum is typically 0.01)
            if quantity < 0.01:
                logger.info(f"Position size too small for {symbol}, need at least 0.01 shares")
                return

            # Calculate bracket order levels
            take_profit_price = price * (1 + self.take_profit)
            stop_loss_price = price * (1 - self.stop_loss)

            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 MULTI-TIMEFRAME BUY SIGNAL - {symbol}")
            logger.info(f"{'='*60}")
            logger.info(f"All timeframes aligned: {mtf_status['aligned_signal']}")
            for tf, data in mtf_status['timeframes'].items():
                logger.info(f"  {tf}: {data['trend']} ({data['momentum']:+.2f}%)")
            logger.info("\nOrder Details:")
            logger.info(f"  Quantity: {quantity:.4f} shares")
            logger.info(f"  Entry: ${price:.2f}")
            logger.info(f"  Take Profit: ${take_profit_price:.2f} (+{self.take_profit*100:.1f}%)")
            logger.info(f"  Stop Loss: ${stop_loss_price:.2f} (-{self.stop_loss*100:.1f}%)")
            logger.info(f"{'='*60}\n")

            # Create and submit bracket order
            order = (OrderBuilder(symbol, 'buy', quantity)
                     .market()
                     .bracket(take_profit=take_profit_price, stop_loss=stop_loss_price)
                     .gtc()
                     .build())

            result = await self.broker.submit_order_advanced(order)

            if result:
                logger.info(f"✅ Multi-timeframe BUY order submitted: {result.id}")
                self.current_positions.add(symbol)

        except Exception as e:
            logger.error(f"Error executing multi-timeframe buy for {symbol}: {e}", exc_info=True)

    async def _execute_sell(self, symbol, position, mtf_status):
        """Execute sell order (exit on timeframe divergence)."""
        try:
            quantity = float(position.qty)

            logger.info(f"\n{'='*60}")
            logger.info(f"🔴 MULTI-TIMEFRAME SELL SIGNAL - {symbol}")
            logger.info(f"{'='*60}")
            logger.info(f"Signal: {mtf_status['aligned_signal']}")
            if mtf_status['divergence']:
                logger.info(f"Divergence detected: {mtf_status['divergence']}")
            for tf, data in mtf_status['timeframes'].items():
                logger.info(f"  {tf}: {data['trend']} ({data['momentum']:+.2f}%)")
            logger.info(f"\nClosing {quantity:.4f} shares")
            logger.info(f"{'='*60}\n")

            # Create and submit sell order
            order = (OrderBuilder(symbol, 'sell', quantity)
                     .market()
                     .day()
                     .build())

            result = await self.broker.submit_order_advanced(order)

            if result:
                logger.info(f"✅ Multi-timeframe SELL order submitted: {result.id}")
                self.current_positions.discard(symbol)

        except Exception as e:
            logger.error(f"Error executing multi-timeframe sell for {symbol}: {e}", exc_info=True)

    async def analyze_symbol(self, symbol):
        """Required by base strategy - returns multi-timeframe signal."""
        status = self.mtf.get_status(symbol)
        return status.get('aligned_signal', 'neutral')

    async def execute_trade(self, symbol, signal):
        """Required by base strategy - handled in on_bar."""
        pass


# Test function
async def test_multi_timeframe_strategy():
    """Test the multi-timeframe strategy with paper trading."""
    from brokers.alpaca_broker import AlpacaBroker
    from config import SYMBOLS

    # Initialize broker
    broker = AlpacaBroker(paper=True)

    # Initialize strategy
    strategy = MultiTimeframeStrategy(
        broker=broker,
        parameters={
            'symbols': SYMBOLS[:2],  # Test with 2 symbols
            'timeframes': ['1Min', '5Min', '15Min'],  # 3 timeframes
            'position_size': 0.05,  # 5% per position
            'stop_loss': 0.02,
            'take_profit': 0.04,
        }
    )

    # Initialize
    await strategy.initialize()

    logger.info("Multi-timeframe strategy initialized for paper trading")
    logger.info(f"Trading symbols: {strategy.symbols}")
    logger.info(f"Timeframes: {strategy.mtf.timeframes}")

    # Start WebSocket
    await broker.start_websocket()

    logger.info("Strategy running. Press Ctrl+C to stop.")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run test
    asyncio.run(test_multi_timeframe_strategy())
