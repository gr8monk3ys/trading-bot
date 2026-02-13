#!/usr/bin/env python3
"""
Bracket Momentum Strategy

This strategy demonstrates the use of advanced order types including:
- Bracket orders (entry + take-profit + stop-loss)
- Trailing stop orders
- Extended time-in-force options (GTC)

The strategy identifies momentum using technical indicators and automatically
sets profit targets and stop losses using bracket orders.
"""

import asyncio
import logging
import warnings
from datetime import datetime

import numpy as np
import talib

from brokers.order_builder import OrderBuilder
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BracketMomentumStrategy(BaseStrategy):
    """
    Advanced momentum strategy using bracket orders for risk management.

    Each entry automatically includes:
    - Take-profit order at calculated resistance level
    - Stop-loss order at support level
    - All orders use GTC (Good-Till-Canceled)
    """

    NAME = "BracketMomentumStrategy"

    def default_parameters(self):
        """Return default parameters for the strategy."""
        return {
            # Basic parameters
            "position_size": 0.1,  # 10% of available capital per position
            "max_positions": 3,  # Maximum number of concurrent positions
            # Momentum parameters
            "rsi_period": 14,
            "rsi_buy_threshold": 35,  # Buy when RSI is low but not oversold
            "rsi_sell_threshold": 65,  # Sell when RSI is high but not overbought
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            # Bracket order parameters
            "profit_target_pct": 0.08,  # 8% profit target
            "stop_loss_pct": 0.03,  # 3% stop loss
            "use_stop_limit": True,  # Use stop-limit instead of stop
            "stop_limit_offset": 0.005,  # 0.5% offset for stop-limit
            # ATR-based dynamic stops
            "use_atr_stops": True,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            # Price MA parameters
            "fast_ma_period": 20,
            "slow_ma_period": 50,
        }

    async def initialize(self, **kwargs):
        """Initialize the bracket momentum strategy."""
        warnings.warn(
            f"{self.__class__.__name__} is experimental and has not been fully validated. "
            "Use in production at your own risk.",
            category=UserWarning,
            stacklevel=2,
        )
        try:
            # Initialize the base strategy
            await super().initialize(**kwargs)

            # Set strategy-specific parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            # Extract parameters
            self.position_size = self.parameters["position_size"]
            self.max_positions = self.parameters["max_positions"]

            # Technical indicator parameters
            self.rsi_period = self.parameters["rsi_period"]
            self.rsi_buy_threshold = self.parameters["rsi_buy_threshold"]
            self.rsi_sell_threshold = self.parameters["rsi_sell_threshold"]
            self.macd_fast = self.parameters["macd_fast_period"]
            self.macd_slow = self.parameters["macd_slow_period"]
            self.macd_signal = self.parameters["macd_signal_period"]
            self.fast_ma = self.parameters["fast_ma_period"]
            self.slow_ma = self.parameters["slow_ma_period"]

            # Bracket parameters
            self.profit_target_pct = self.parameters["profit_target_pct"]
            self.stop_loss_pct = self.parameters["stop_loss_pct"]
            self.use_stop_limit = self.parameters["use_stop_limit"]
            self.stop_limit_offset = self.parameters["stop_limit_offset"]

            # ATR parameters
            self.use_atr_stops = self.parameters["use_atr_stops"]
            self.atr_period = self.parameters["atr_period"]
            self.atr_multiplier = self.parameters["atr_multiplier"]

            # Initialize tracking dictionaries
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.signals = dict.fromkeys(self.symbols, "neutral")
            self.current_prices = {}
            self.price_history = {symbol: [] for symbol in self.symbols}
            self.active_bracket_orders = {}  # Track bracket orders by symbol

            # Add strategy as subscriber to broker
            if hasattr(self.broker, "_add_subscriber"):
                subscribe_result = self.broker._add_subscriber(self)
                if asyncio.iscoroutine(subscribe_result):
                    await subscribe_result

            logger.info(f"Initialized {self.NAME} with {len(self.symbols)} symbols")
            logger.info(
                f"Profit target: {self.profit_target_pct:.1%}, Stop loss: {self.stop_loss_pct:.1%}"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing {self.NAME}: {e}", exc_info=True)
            return False

    async def on_bar(
        self, symbol, open_price, high_price, low_price, close_price, volume, timestamp
    ):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return

            # Store latest price
            self.current_prices[symbol] = close_price

            # Update price history
            self.price_history[symbol].append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

            # Keep only necessary history
            max_history = (
                max(
                    self.slow_ma,
                    self.rsi_period,
                    self.macd_slow + self.macd_signal,
                    self.atr_period,
                )
                + 10
            )
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]

            # Update technical indicators
            await self._update_indicators(symbol)

            # Check for signals
            signal = await self._generate_signal(symbol)
            self.signals[symbol] = signal

            # Execute trades if needed
            if signal == "buy":
                await self._execute_bracket_buy(symbol)

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    async def _update_indicators(self, symbol):
        """Update technical indicators for a symbol."""
        try:
            # Ensure we have enough price history
            if len(self.price_history[symbol]) < self.slow_ma:
                return

            # Extract price data into arrays
            closes = np.array([bar["close"] for bar in self.price_history[symbol]])
            highs = np.array([bar["high"] for bar in self.price_history[symbol]])
            lows = np.array([bar["low"] for bar in self.price_history[symbol]])

            # Calculate RSI
            rsi = talib.RSI(closes, timeperiod=self.rsi_period)

            # Calculate MACD
            macd, signal, hist = talib.MACD(
                closes,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal,
            )

            # Calculate moving averages
            fast_ma = talib.SMA(closes, timeperiod=self.fast_ma)
            slow_ma = talib.SMA(closes, timeperiod=self.slow_ma)

            # Calculate ATR for dynamic stops
            atr = talib.ATR(highs, lows, closes, timeperiod=self.atr_period)

            # Store indicators
            self.indicators[symbol] = {
                "rsi": rsi[-1] if len(rsi) > 0 else None,
                "macd": macd[-1] if len(macd) > 0 else None,
                "macd_signal": signal[-1] if len(signal) > 0 else None,
                "macd_hist": hist[-1] if len(hist) > 0 else None,
                "fast_ma": fast_ma[-1] if len(fast_ma) > 0 else None,
                "slow_ma": slow_ma[-1] if len(slow_ma) > 0 else None,
                "atr": atr[-1] if len(atr) > 0 else None,
                "close": closes[-1] if len(closes) > 0 else None,
            }

        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}", exc_info=True)

    async def _generate_signal(self, symbol):
        """Generate trading signal based on indicators."""
        try:
            # Check if indicators are available
            if not self.indicators.get(symbol) or self.indicators[symbol]["rsi"] is None:
                return "neutral"

            ind = self.indicators[symbol]

            # Get current indicator values
            rsi = ind["rsi"]
            macd = ind["macd"]
            macd_signal = ind["macd_signal"]
            macd_hist = ind["macd_hist"]
            fast_ma = ind["fast_ma"]
            slow_ma = ind["slow_ma"]

            # Buy signal: RSI recovering from low + MACD bullish + MA uptrend
            buy_signal = (
                rsi < self.rsi_buy_threshold  # RSI low but not extreme
                and macd > macd_signal  # MACD bullish crossover
                and macd_hist > 0  # MACD histogram positive
                and fast_ma > slow_ma  # Fast MA above slow MA (uptrend)
            )

            if buy_signal:
                return "buy"

            return "neutral"

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return "neutral"

    async def _execute_bracket_buy(self, symbol):
        """Execute a bracket buy order with automatic take-profit and stop-loss."""
        try:
            # Get current positions
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            # Don't buy if we already have a position
            if current_position:
                logger.debug(f"Already have position in {symbol}, skipping buy")
                return

            # Check if we're already at max positions
            if len(positions) >= self.max_positions:
                logger.info(
                    f"Max positions reached ({self.max_positions}), skipping buy for {symbol}"
                )
                return

            # Get account info
            account = await self.broker.get_account()
            buying_power = float(account.buying_power)

            # Calculate position size
            current_price = self.current_prices[symbol]
            position_value = buying_power * self.position_size

            # CRITICAL SAFETY: Enforce maximum position size limit (5% of portfolio)
            position_value, quantity = await self.enforce_position_size_limit(
                symbol, position_value, current_price
            )

            # Allow fractional shares (Alpaca minimum is typically 0.01)
            if quantity < 0.01:
                logger.info(f"Position size too small for {symbol}, need at least 0.01 shares")
                return

            # Calculate take-profit and stop-loss levels
            if self.use_atr_stops and self.indicators[symbol].get("atr"):
                # Use ATR for dynamic stops
                atr = self.indicators[symbol]["atr"]
                take_profit_price = current_price + (atr * self.atr_multiplier)
                stop_loss_price = current_price - (atr * self.atr_multiplier)
            else:
                # Use percentage-based stops
                take_profit_price = current_price * (1 + self.profit_target_pct)
                stop_loss_price = current_price * (1 - self.stop_loss_pct)

            # Calculate stop-limit price if using stop-limit
            stop_limit_price = None
            if self.use_stop_limit:
                stop_limit_price = stop_loss_price * (1 - self.stop_limit_offset)

            # Build bracket order using OrderBuilder
            logger.info(f"Creating bracket order for {symbol}:")
            logger.info(f"  Entry: ${current_price:.2f} x {quantity} shares")
            logger.info(f"  Take-profit: ${take_profit_price:.2f}")
            logger.info(
                f"  Stop-loss: ${stop_loss_price:.2f}"
                + (f" (limit: ${stop_limit_price:.2f})" if stop_limit_price else "")
            )

            order = (
                OrderBuilder(symbol, "buy", quantity)
                .market()
                .bracket(
                    take_profit=take_profit_price,
                    stop_loss=stop_loss_price,
                    stop_limit=stop_limit_price,
                )
                .gtc()  # Good-Till-Canceled
                .build()
            )

            # Submit the bracket order via gateway
            result = await self.submit_entry_order(
                order,
                reason="bracket_momentum_entry",
                max_positions=self.max_positions,
            )

            if result and (not hasattr(result, "success") or result.success):
                order_id = None
                if isinstance(result, dict):
                    for key in ("order_id", "id"):
                        value = result.get(key)
                        if isinstance(value, (str, int, float)):
                            order_id = str(value)
                            break
                else:
                    for attr in ("order_id", "id"):
                        value = getattr(result, attr, None)
                        if isinstance(value, (str, int, float)):
                            order_id = str(value)
                            break

                if order_id is None:
                    order_id = "unknown"
                logger.info(f"✅ Bracket order submitted for {symbol}: {order_id}")
                logger.info(
                    f"   Risk/Reward: {self.stop_loss_pct:.1%} / {self.profit_target_pct:.1%}"
                )

                # Track the bracket order
                self.active_bracket_orders[symbol] = {
                    "order_id": order_id,
                    "entry_price": current_price,
                    "take_profit": take_profit_price,
                    "stop_loss": stop_loss_price,
                    "quantity": quantity,
                    "timestamp": datetime.now(),
                }

        except Exception as e:
            logger.error(f"Error executing bracket buy for {symbol}: {e}", exc_info=True)

    async def analyze_symbol(self, symbol):
        """Analyze a symbol and determine if we should trade it."""
        return self.signals.get(symbol, "neutral")

    async def execute_trade(self, symbol, signal):
        """Execute a trade based on the signal."""
        # Bracket orders are handled in _execute_bracket_buy
        pass

    async def export_state(self) -> dict:
        """Export state for restart recovery."""
        def _dt(v):
            return v.isoformat() if hasattr(v, "isoformat") else v

        orders = {}
        for sym, data in self.active_bracket_orders.items():
            item = data.copy()
            if "timestamp" in item:
                item["timestamp"] = _dt(item["timestamp"])
            orders[sym] = item

        return {"active_bracket_orders": orders}

    async def import_state(self, state: dict) -> None:
        """Restore state after restart."""
        from datetime import datetime

        def _parse_dt(v):
            return datetime.fromisoformat(v) if isinstance(v, str) else v

        orders = {}
        for sym, data in state.get("active_bracket_orders", {}).items():
            item = data.copy()
            if "timestamp" in item:
                item["timestamp"] = _parse_dt(item["timestamp"])
            orders[sym] = item
        self.active_bracket_orders = orders


# Convenience function for testing
async def test_bracket_strategy():
    """Test the bracket momentum strategy with paper trading."""
    from brokers.alpaca_broker import AlpacaBroker
    from config import SYMBOLS

    # Initialize broker
    broker = AlpacaBroker(paper=True)

    # Initialize strategy
    strategy = BracketMomentumStrategy(
        broker=broker, symbols=SYMBOLS[:3]  # Test with first 3 symbols
    )

    # Initialize strategy parameters
    await strategy.initialize(
        position_size=0.05,  # 5% per position for testing
        max_positions=2,
        profit_target_pct=0.05,  # 5% profit target
        stop_loss_pct=0.02,  # 2% stop loss
    )

    logger.info("Bracket momentum strategy initialized for paper trading")
    logger.info(f"Trading symbols: {strategy.symbols}")

    # Start WebSocket connection
    await broker.start_websocket()

    logger.info("Strategy running. Press Ctrl+C to stop.")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run test
    asyncio.run(test_bracket_strategy())
