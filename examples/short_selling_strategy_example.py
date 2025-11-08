#!/usr/bin/env python3
"""
Short Selling Strategy Example

Demonstrates how to implement short selling (profit from declining stocks).

Short Selling Basics:
1. SELL shares you don't own (borrow from broker)
2. Price drops
3. BUY back shares at lower price (return to broker)
4. Keep the difference as profit

Example:
- Short AAPL at $200 (sell 10 shares = $2000 credit)
- AAPL drops to $190
- Cover short at $190 (buy 10 shares = $1900 debit)
- Profit: $100 ($2000 - $1900)

Risk: If price goes UP instead, you lose money:
- Short at $200
- Price rises to $210
- Cover at $210 (forced to buy back at higher price)
- Loss: $100

This strategy shorts stocks showing bearish signals and covers when bearish momentum fades.
"""

import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy
from brokers.order_builder import OrderBuilder

logger = logging.getLogger(__name__)


class ShortSellingStrategy(BaseStrategy):
    """
    Example strategy that implements short selling.

    Shorts when:
    - Strong bearish momentum (RSI > 70 and falling)
    - Price breaking below support
    - Negative sentiment (if available)

    Covers (buys back) when:
    - Bearish momentum weakens (RSI < 50)
    - Price recovers above resistance
    - Stop loss hit (price rises too much)
    """

    def default_parameters(self):
        """Default strategy parameters."""
        return {
            'position_size': 0.08,  # 8% of capital per short position
            'max_positions': 3,     # Maximum concurrent short positions

            # Short-specific risk management (more conservative)
            'stop_loss': 0.03,      # 3% stop loss (price rises 3%)
            'take_profit': 0.06,    # 6% take profit (price drops 6%)

            # Technical indicators for short signals
            'rsi_period': 14,
            'rsi_short_threshold': 65,  # Short when RSI > 65 (overbought)
            'rsi_cover_threshold': 45,  # Cover when RSI < 45 (momentum fading)

            'sma_period': 20,           # Simple moving average
            'price_below_sma': 0.98,    # Short when price falls below 98% of SMA
        }

    async def initialize(self, **kwargs):
        """Initialize the short selling strategy."""
        try:
            # Initialize base strategy
            await super().initialize(**kwargs)

            # Set parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            self.position_size = self.parameters['position_size']
            self.max_positions = self.parameters['max_positions']
            self.stop_loss = self.parameters['stop_loss']
            self.take_profit = self.parameters['take_profit']
            self.rsi_period = self.parameters['rsi_period']
            self.rsi_short_threshold = self.parameters['rsi_short_threshold']
            self.rsi_cover_threshold = self.parameters['rsi_cover_threshold']
            self.sma_period = self.parameters['sma_period']
            self.price_below_sma = self.parameters['price_below_sma']

            # Track price history and indicators
            self.price_history = {symbol: [] for symbol in self.symbols}
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.current_prices = {}

            # Track short positions separately
            self.short_positions = {}  # symbol -> entry_price

            # Add strategy as subscriber to broker
            if hasattr(self.broker, '_add_subscriber'):
                self.broker._add_subscriber(self)

            logger.info(f"Short selling strategy initialized")
            logger.info(f"Max short positions: {self.max_positions}")
            logger.info(f"Stop loss: {self.stop_loss:.1%}, Take profit: {self.take_profit:.1%}")
            logger.info("⚠️  WARNING: Short selling has UNLIMITED risk. Use stop losses!")

            return True

        except Exception as e:
            logger.error(f"Error initializing short strategy: {e}", exc_info=True)
            return False

    async def on_bar(self, symbol, open_price, high_price, low_price, close_price, volume, timestamp):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return

            # Update price history
            self.current_prices[symbol] = close_price
            self.price_history[symbol].append(close_price)

            # Keep only necessary history
            max_history = max(self.sma_period, self.rsi_period) + 20
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]

            # Update indicators
            await self._update_indicators(symbol)

            # Check if we have enough data
            if len(self.price_history[symbol]) < self.sma_period:
                return

            # Get current positions
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            # Determine if we have a short position
            is_short = current_position and float(current_position.qty) < 0

            # Generate signal
            signal = await self._generate_signal(symbol, is_short)

            # Execute trades
            if signal == 'short' and not is_short:
                await self._execute_short(symbol, close_price)
            elif signal == 'cover' and is_short:
                await self._execute_cover(symbol, current_position)

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    async def _update_indicators(self, symbol):
        """Update technical indicators for short signal generation."""
        try:
            if len(self.price_history[symbol]) < self.rsi_period + 1:
                return

            prices = np.array(self.price_history[symbol])

            # Calculate RSI
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = np.mean(gain[-self.rsi_period:])
            avg_loss = np.mean(loss[-self.rsi_period:])

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # Calculate SMA
            sma = np.mean(prices[-self.sma_period:])

            # Store indicators
            self.indicators[symbol] = {
                'rsi': rsi,
                'sma': sma,
                'current_price': prices[-1]
            }

        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}", exc_info=True)

    async def _generate_signal(self, symbol, is_short):
        """
        Generate short/cover signals based on indicators.

        Args:
            symbol: Stock symbol
            is_short: True if we currently have a short position

        Returns:
            'short', 'cover', or 'neutral'
        """
        try:
            if not self.indicators.get(symbol):
                return 'neutral'

            ind = self.indicators[symbol]
            rsi = ind['rsi']
            sma = ind['sma']
            price = ind['current_price']

            # SHORT SIGNAL: Overbought + price breaking below SMA
            if not is_short:
                # Check for overbought conditions
                rsi_bearish = rsi > self.rsi_short_threshold

                # Check if price is falling below SMA
                price_weakness = price < (sma * self.price_below_sma)

                if rsi_bearish and price_weakness:
                    logger.info(f"📉 Short signal for {symbol}:")
                    logger.info(f"  RSI: {rsi:.1f} (overbought)")
                    logger.info(f"  Price: ${price:.2f}, SMA: ${sma:.2f}")
                    logger.info(f"  Price is {((price/sma - 1) * 100):.1f}% vs SMA")
                    return 'short'

            # COVER SIGNAL: Bearish momentum fading
            else:
                # Check if RSI shows momentum fading
                momentum_fading = rsi < self.rsi_cover_threshold

                # Check if price is recovering above SMA
                price_recovery = price > sma

                if momentum_fading or price_recovery:
                    logger.info(f"📈 Cover signal for {symbol}:")
                    logger.info(f"  RSI: {rsi:.1f} (momentum fading)")
                    logger.info(f"  Price: ${price:.2f}, SMA: ${sma:.2f}")
                    return 'cover'

            return 'neutral'

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return 'neutral'

    async def _execute_short(self, symbol, price):
        """
        Execute a short sale with bracket orders.

        Note: We SELL first (open short position), then BUY to cover later.
        """
        try:
            # CRITICAL SAFETY: Check if trading is allowed
            if not await self.check_trading_allowed():
                logger.warning("Trading halted by circuit breaker")
                return

            # Check max positions limit
            positions = await self.broker.get_positions()
            short_count = sum(1 for p in positions if float(p.qty) < 0)

            if short_count >= self.max_positions:
                logger.info(f"Max short positions reached ({self.max_positions})")
                return

            # Get account info
            account = await self.broker.get_account()
            buying_power = float(account.buying_power)

            # Calculate position size
            position_value = buying_power * self.position_size

            # CRITICAL SAFETY: Enforce maximum position size limit (5% of portfolio)
            position_value, quantity = await self.enforce_position_size_limit(symbol, position_value, price)

            # Allow fractional shares
            if quantity < 0.01:
                logger.info(f"Position size too small for {symbol}")
                return

            # Calculate bracket order levels for SHORT position
            # Stop loss: price RISES above entry (lose money)
            # Take profit: price FALLS below entry (make money)
            stop_loss_price = price * (1 + self.stop_loss)      # Price rises 3% -> stop loss
            take_profit_price = price * (1 - self.take_profit)  # Price falls 6% -> take profit

            logger.info(f"\n{'='*60}")
            logger.info(f"📉 SHORT SELL SIGNAL - {symbol}")
            logger.info(f"{'='*60}")
            logger.info(f"Indicators:")
            logger.info(f"  RSI: {self.indicators[symbol]['rsi']:.1f}")
            logger.info(f"  Price: ${price:.2f}")
            logger.info(f"  SMA: ${self.indicators[symbol]['sma']:.2f}")
            logger.info(f"\nShort Position:")
            logger.info(f"  Quantity: {quantity:.4f} shares (SHORT)")
            logger.info(f"  Entry: ${price:.2f}")
            logger.info(f"  Take Profit: ${take_profit_price:.2f} (-{self.take_profit*100:.1f}%)")
            logger.info(f"  Stop Loss: ${stop_loss_price:.2f} (+{self.stop_loss*100:.1f}%)")
            logger.info(f"\n⚠️  WARNING: SHORT position has unlimited upside risk!")
            logger.info(f"{'='*60}\n")

            # Create SHORT order (SELL to open)
            # Note: For short positions, the bracket orders are inverted:
            # - "take_profit" becomes a BUY at lower price
            # - "stop_loss" becomes a BUY at higher price
            order = (OrderBuilder(symbol, 'sell', quantity)  # SELL to open short
                     .market()
                     .bracket(
                         take_profit=take_profit_price,    # Buy at lower price (profit)
                         stop_loss=stop_loss_price          # Buy at higher price (loss)
                     )
                     .gtc()
                     .build())

            result = await self.broker.submit_order_advanced(order)

            if result:
                logger.info(f"✅ SHORT order submitted: {result.id}")
                self.short_positions[symbol] = price

        except Exception as e:
            logger.error(f"Error executing short for {symbol}: {e}", exc_info=True)

    async def _execute_cover(self, symbol, position):
        """
        Cover (close) a short position by buying back the shares.

        Args:
            position: Current short position to close
        """
        try:
            # For short positions, quantity is negative
            quantity = abs(float(position.qty))

            logger.info(f"\n{'='*60}")
            logger.info(f"📈 COVER SHORT POSITION - {symbol}")
            logger.info(f"{'='*60}")
            logger.info(f"Covering {quantity:.4f} shares")
            logger.info(f"Entry price: ${self.short_positions.get(symbol, 0):.2f}")
            logger.info(f"Current price: ${self.current_prices[symbol]:.2f}")

            entry_price = self.short_positions.get(symbol, self.current_prices[symbol])
            current_price = self.current_prices[symbol]
            pnl_pct = ((entry_price - current_price) / entry_price) * 100

            logger.info(f"P/L: {pnl_pct:+.2f}%")
            logger.info(f"{'='*60}\n")

            # Create BUY order to cover short (close position)
            order = (OrderBuilder(symbol, 'buy', quantity)  # BUY to cover short
                     .market()
                     .day()
                     .build())

            result = await self.broker.submit_order_advanced(order)

            if result:
                logger.info(f"✅ COVER order submitted: {result.id}")
                if symbol in self.short_positions:
                    del self.short_positions[symbol]

        except Exception as e:
            logger.error(f"Error covering short for {symbol}: {e}", exc_info=True)

    async def analyze_symbol(self, symbol):
        """Required by base strategy."""
        return 'neutral'

    async def execute_trade(self, symbol, signal):
        """Required by base strategy - handled in on_bar."""
        pass


# Test function
async def test_short_selling_strategy():
    """Test the short selling strategy with paper trading."""
    from brokers.alpaca_broker import AlpacaBroker
    from config import SYMBOLS

    # Initialize broker
    broker = AlpacaBroker(paper=True)

    # Initialize strategy
    strategy = ShortSellingStrategy(
        broker=broker,
        parameters={
            'symbols': SYMBOLS[:3],  # Test with 3 symbols
            'position_size': 0.05,   # 5% per short position
            'max_positions': 2,
            'stop_loss': 0.02,       # Tight stop loss for shorts
            'take_profit': 0.04,
        }
    )

    # Initialize
    await strategy.initialize()

    logger.info("Short selling strategy initialized for paper trading")
    logger.info(f"Trading symbols: {strategy.symbols}")
    logger.info("⚠️  WARNING: Short selling has unlimited risk. Paper trade first!")

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
    asyncio.run(test_short_selling_strategy())
