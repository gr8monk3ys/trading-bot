import logging
import asyncio
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager
from brokers.order_builder import OrderBuilder

logger = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy that identifies overbought/oversold conditions and
    trades on the expectation that prices will revert to the mean. Uses Bollinger
    Bands, RSI, and standard deviation to identify entry/exit points.
    """
    
    NAME = "MeanReversionStrategy"
    
    def default_parameters(self):
        """Return default parameters for the strategy."""
        return {
            # Basic parameters
            'position_size': 0.1,  # 10% of available capital per position
            'max_positions': 5,    # Maximum number of concurrent positions
            'max_portfolio_risk': 0.02,  # Maximum portfolio risk (2%)
            'stop_loss': 0.02,     # 2% stop loss
            'take_profit': 0.04,   # 4% take profit
            
            # Mean reversion parameters
            'bb_period': 20,       # Bollinger Bands period
            'bb_std': 2.0,         # Bollinger Bands standard deviation
            'rsi_period': 14,      # RSI period
            'rsi_overbought': 70,  # RSI overbought threshold
            'rsi_oversold': 30,    # RSI oversold threshold
            'sma_period': 50,      # SMA period for mean
            'mean_lookback': 20,   # Lookback period for mean reversion
            'std_threshold': 1.5,  # Standard deviation threshold
            
            # Exit parameters
            'profit_target_std': 0.5,  # Exit when price reverts to this many stdevs from mean
            'max_hold_days': 5,    # Maximum holding period in days
            'trailing_stop': 0.015, # 1.5% trailing stop once in profit
            
            # Risk management
            'max_correlation': 0.7,  # Maximum correlation between positions
            'max_sector_exposure': 0.3,  # Maximum exposure to any one sector
        }
    
    async def initialize(self, **kwargs):
        """Initialize the mean reversion strategy."""
        try:
            # Initialize the base strategy
            await super().initialize(**kwargs)
            
            # Set strategy-specific parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params
            
            # Extract parameters
            self.position_size = self.parameters['position_size']
            self.max_positions = self.parameters['max_positions']
            self.stop_loss = self.parameters['stop_loss']
            self.take_profit = self.parameters['take_profit']
            
            # Mean reversion parameters
            self.bb_period = self.parameters['bb_period']
            self.bb_std = self.parameters['bb_std']
            self.rsi_period = self.parameters['rsi_period']
            self.rsi_overbought = self.parameters['rsi_overbought']
            self.rsi_oversold = self.parameters['rsi_oversold']
            self.sma_period = self.parameters['sma_period']
            self.mean_lookback = self.parameters['mean_lookback']
            self.std_threshold = self.parameters['std_threshold']
            
            # Exit parameters
            self.profit_target_std = self.parameters['profit_target_std']
            self.max_hold_days = self.parameters['max_hold_days']
            self.trailing_stop = self.parameters['trailing_stop']
            
            # Initialize tracking dictionaries
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.signals = {symbol: 'neutral' for symbol in self.symbols}
            self.last_signal_time = {symbol: None for symbol in self.symbols}
            self.position_entries = {}  # Track entry times and prices
            self.highest_prices = {}    # For trailing stops
            self.lowest_prices = {}     # For trailing stops
            self.current_prices = {}
            self.price_history = {symbol: [] for symbol in self.symbols}
            
            # Risk manager initialization
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.parameters['max_portfolio_risk'],
                max_position_risk=self.parameters.get('max_position_risk', 0.01),
                max_correlation=self.parameters['max_correlation']
            )
            
            # Add strategy as subscriber to broker
            if hasattr(self.broker, '_add_subscriber'):
                self.broker._add_subscriber(self)
            
            logger.info(f"Initialized {self.NAME} with {len(self.symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {self.NAME}: {e}", exc_info=True)
            return False
    
    async def on_bar(self, symbol, open_price, high_price, low_price, close_price, volume, timestamp):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return
                
            # Store latest price
            self.current_prices[symbol] = close_price
            
            # Update price history
            self.price_history[symbol].append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            # Keep only necessary history
            max_history = max(self.sma_period, self.bb_period, self.rsi_period) + self.mean_lookback + 10
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                
            # Update technical indicators
            await self._update_indicators(symbol)
            
            # Check for signals
            signal = await self._generate_signal(symbol)
            self.signals[symbol] = signal
            
            # Execute trades if needed
            if signal != 'neutral':
                await self._execute_signal(symbol, signal)
                
            # Check exit conditions for existing positions
            await self._check_exit_conditions(symbol)
            
        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)
    
    async def _update_indicators(self, symbol):
        """Update technical indicators for a symbol."""
        try:
            # Ensure we have enough price history
            if len(self.price_history[symbol]) < self.sma_period:
                return
                
            # Extract price data into arrays
            closes = np.array([bar['close'] for bar in self.price_history[symbol]])
            highs = np.array([bar['high'] for bar in self.price_history[symbol]])
            lows = np.array([bar['low'] for bar in self.price_history[symbol]])
            volumes = np.array([bar['volume'] for bar in self.price_history[symbol]])
            
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                closes, 
                timeperiod=self.bb_period, 
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0
            )
            
            # Calculate RSI
            rsi = talib.RSI(closes, timeperiod=self.rsi_period)
            
            # Calculate SMA (mean)
            sma = talib.SMA(closes, timeperiod=self.sma_period)
            
            # Calculate standard deviation
            std = talib.STDDEV(closes, timeperiod=self.mean_lookback)
            
            # Calculate Stochastic
            slowk, slowd = talib.STOCH(
                highs, 
                lows, 
                closes, 
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            
            # Calculate ATR for stop loss
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            
            # Calculate distance from mean (z-score)
            z_score = (closes[-1] - sma[-1]) / std[-1] if std[-1] > 0 else 0
            
            # Calculate percent of BB range
            bb_range = upper[-1] - lower[-1]
            bb_position = (closes[-1] - lower[-1]) / bb_range if bb_range > 0 else 0.5
            
            # Store indicators
            self.indicators[symbol] = {
                'upper_band': upper[-1] if len(upper) > 0 else None,
                'middle_band': middle[-1] if len(middle) > 0 else None,
                'lower_band': lower[-1] if len(lower) > 0 else None,
                'rsi': rsi[-1] if len(rsi) > 0 else None,
                'sma': sma[-1] if len(sma) > 0 else None,
                'std': std[-1] if len(std) > 0 else None,
                'z_score': z_score,
                'bb_position': bb_position,
                'slowk': slowk[-1] if len(slowk) > 0 else None,
                'slowd': slowd[-1] if len(slowd) > 0 else None,
                'atr': atr[-1] if len(atr) > 0 else None,
                'close': closes[-1] if len(closes) > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}: {e}", exc_info=True)
    
    async def _generate_signal(self, symbol):
        """Generate trading signal based on indicators."""
        try:
            # Check if indicators are available
            if not self.indicators.get(symbol) or self.indicators[symbol]['rsi'] is None:
                return 'neutral'
                
            ind = self.indicators[symbol]
            
            # Get current indicator values
            close = ind['close']
            upper_band = ind['upper_band']
            lower_band = ind['lower_band']
            middle_band = ind['middle_band']
            rsi = ind['rsi']
            z_score = ind['z_score']
            bb_position = ind['bb_position']
            stoch_k = ind['slowk']
            stoch_d = ind['slowd']
            
            # Buy signal: Price is below lower Bollinger Band + RSI is oversold + far from mean
            buy_signal = (
                close < lower_band and 
                rsi < self.rsi_oversold and 
                z_score < -self.std_threshold and
                bb_position < 0.05 and  # Near bottom of BB
                stoch_k < 20 and stoch_k > stoch_d  # Stoch turning up
            )
            
            # Sell signal: Price is above upper Bollinger Band + RSI is overbought + far from mean
            sell_signal = (
                close > upper_band and 
                rsi > self.rsi_overbought and 
                z_score > self.std_threshold and
                bb_position > 0.95 and  # Near top of BB
                stoch_k > 80 and stoch_k < stoch_d  # Stoch turning down
            )
            
            # Determine final signal
            if buy_signal:
                return 'buy'
            elif sell_signal:
                return 'sell'
                
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return 'neutral'
    
    async def _execute_signal(self, symbol, signal):
        """Execute a trading signal."""
        try:
            # Check if we have enough time since last signal to avoid overtrading
            current_time = datetime.now()
            if (self.last_signal_time.get(symbol) and 
                (current_time - self.last_signal_time[symbol]).total_seconds() < 3600):  # 1 hour cooldown
                return
                
            # Get current positions
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)
            
            # Get account info
            account = await self.broker.get_account()
            buying_power = float(account.buying_power)
            
            # Execute buy signal
            if signal == 'buy' and not current_position:
                # Check if we're already at max positions
                if len(positions) >= self.max_positions:
                    logger.info(f"Max positions reached ({self.max_positions}), skipping buy for {symbol}")
                    return
                
                # Calculate position size
                price = self.current_prices[symbol]
                position_value = buying_power * self.position_size
                
                # Risk-adjust position size
                current_positions = {}
                for pos in positions:
                    pos_symbol = pos.symbol
                    if pos_symbol in self.price_history:
                        price_history = self.price_history[pos_symbol]
                        close_prices = [bar['close'] for bar in price_history]
                        current_positions[pos_symbol] = {
                            'value': float(pos.market_value),
                            'price_history': close_prices,
                            'risk': None
                        }
                
                # Use risk manager to adjust position size if we have price history
                if len(self.price_history[symbol]) > 20:
                    close_prices = [bar['close'] for bar in self.price_history[symbol]]
                    adjusted_value = self.risk_manager.adjust_position_size(
                        symbol, 
                        position_value,
                        close_prices,
                        current_positions
                    )
                    position_value = adjusted_value
                
                if position_value <= 0:
                    logger.info(f"Risk manager rejected position for {symbol}")
                    return

                # CRITICAL SAFETY: Enforce maximum position size limit (5% of portfolio)
                position_value, quantity = await self.enforce_position_size_limit(symbol, position_value, price)

                # Allow fractional shares (Alpaca minimum is typically 0.01)
                if quantity < 0.01:
                    logger.info(f"Position size too small for {symbol}, need at least 0.01 shares")
                    return

                # Calculate bracket order levels
                take_profit_price = price * (1 + self.take_profit)  # 4% profit target
                stop_loss_price = price * (1 - self.stop_loss)      # 2% stop loss

                # Create and submit bracket order for automatic risk management
                # Use fractional shares for precise position sizing
                order = (OrderBuilder(symbol, 'buy', quantity)
                         .market()
                         .bracket(take_profit=take_profit_price, stop_loss=stop_loss_price)
                         .gtc()
                         .build())
                result = await self.broker.submit_order_advanced(order)

                if result:
                    logger.info(f"BUY bracket order submitted for {symbol}: {quantity:.4f} shares at ~${price:.2f}")
                    logger.info(f"  Take Profit: ${take_profit_price:.2f} (+{self.take_profit*100:.1f}%)")
                    logger.info(f"  Stop Loss: ${stop_loss_price:.2f} (-{self.stop_loss*100:.1f}%)")

                    # Store entry details
                    self.position_entries[symbol] = {
                        'time': current_time,
                        'price': price,
                        'quantity': quantity  # Keep fractional quantity
                    }

                    # Initialize trailing stop tracking (track highest price for long positions)
                    self.highest_prices[symbol] = price

                    # Update last signal time
                    self.last_signal_time[symbol] = current_time
                    
            # Execute sell signal
            elif signal == 'sell' and current_position:
                # We have a position and should sell it
                quantity = float(current_position.qty)

                # Create and submit market sell order
                order = (OrderBuilder(symbol, 'sell', quantity)
                         .market()
                         .day()
                         .build())
                result = await self.broker.submit_order_advanced(order)

                if result:
                    price = self.current_prices[symbol]
                    logger.info(f"SELL order submitted for {symbol}: {quantity} shares at ~${price:.2f}")

                    # Clear position tracking
                    if symbol in self.position_entries:
                        del self.position_entries[symbol]
                    if symbol in self.highest_prices:
                        del self.highest_prices[symbol]
                    if symbol in self.lowest_prices:
                        del self.lowest_prices[symbol]

                    # Update last signal time
                    self.last_signal_time[symbol] = current_time
                
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}", exc_info=True)
    
    async def _check_exit_conditions(self, symbol):
        """
        Check advanced exit conditions for the given symbol.

        Note: Bracket orders automatically handle basic stop-loss (-2%) and take-profit (+4%)
        at the broker level. This method implements ADDITIONAL "smart exits" that can optimize
        returns beyond the bracket limits:

        1. Max holding period (5 days) - Frees capital from stale positions
        2. Mean reversion target - Exits when price returns to mean (strategy's core edge)
        3. Trailing stop (1.5%) - Locks in profits beyond 4% if stock keeps running

        These exits work IN ADDITION TO the bracket orders for maximum profit potential.
        """
        try:
            # Get current position
            positions = await self.broker.get_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)

            if not current_position:
                # Position was closed (likely by bracket order), clean up tracking
                if symbol in self.position_entries:
                    logger.debug(f"Position {symbol} closed (likely by bracket order), cleaning up tracking")
                    del self.position_entries[symbol]
                if symbol in self.lowest_prices:
                    del self.lowest_prices[symbol]
                if symbol in self.highest_prices:
                    del self.highest_prices[symbol]
                return

            current_price = self.current_prices.get(symbol)
            if not current_price:
                return
                
            # Get entry details
            entry = self.position_entries.get(symbol)
            if not entry:
                return
                
            entry_price = entry['price']
            entry_time = entry['time']
            current_time = datetime.now()

            # Calculate unrealized profit/loss
            unrealized_pnl = (current_price - entry_price) / entry_price

            # SMART EXIT 1: Max holding period - Free up capital from stale positions
            holding_days = (current_time - entry_time).days
            if holding_days >= self.max_hold_days:
                logger.info(f"SMART EXIT: Max holding period ({self.max_hold_days} days) reached for {symbol}, "
                           f"exiting position to free capital (P/L: {unrealized_pnl*100:.1f}%)")
                quantity = float(current_position.qty)
                order = (OrderBuilder(symbol, 'sell', quantity)
                         .market()
                         .day()
                         .build())
                await self.broker.submit_order_advanced(order)
                return

            # SMART EXIT 2: Mean reversion target - Exit when price returns to mean (strategy's core edge)
            ind = self.indicators[symbol]
            sma = ind.get('sma')
            std = ind.get('std')

            if sma and std and std > 0:
                z_score = (current_price - sma) / std

                # Check if price has reverted to mean (or beyond)
                if (entry_price < sma and current_price >= sma - self.profit_target_std * std) or \
                   (entry_price > sma and current_price <= sma + self.profit_target_std * std):
                    logger.info(f"SMART EXIT: Mean reversion target reached for {symbol} "
                               f"(z-score: {z_score:.2f}, P/L: {unrealized_pnl*100:.1f}%), exiting position")
                    quantity = float(current_position.qty)
                    order = (OrderBuilder(symbol, 'sell', quantity)
                             .market()
                             .day()
                             .build())
                    await self.broker.submit_order_advanced(order)
                    return

            # SMART EXIT 3: Trailing stop - Lock in profits beyond 4% take-profit
            # For LONG positions: track HIGHEST price (peak) and trail DOWN from it
            # Update peak price tracking
            if symbol in self.highest_prices:
                self.highest_prices[symbol] = max(self.highest_prices[symbol], current_price)
            else:
                self.highest_prices[symbol] = current_price

            # Trailing stop triggers only if in profit (can capture more than bracket's fixed 4%)
            if unrealized_pnl > 0:
                # Calculate trailing stop price: peak price minus trailing percentage
                peak_price = self.highest_prices[symbol]
                trailing_stop_price = peak_price * (1 - self.trailing_stop)  # Trail DOWN by 1.5%

                if current_price <= trailing_stop_price:
                    logger.info(f"SMART EXIT: Trailing stop triggered for {symbol} at ${current_price:.2f} "
                               f"(peak: ${peak_price:.2f}, trailing stop: ${trailing_stop_price:.2f}, "
                               f"profit locked: {unrealized_pnl*100:.1f}%)")
                    quantity = float(current_position.qty)
                    order = (OrderBuilder(symbol, 'sell', quantity)
                             .market()
                             .day()
                             .build())
                    await self.broker.submit_order_advanced(order)
                    return

            # Monitor bracket order levels for logging/debugging
            # Bracket order stop-loss at -2%, take-profit at +4%
            bracket_stop_loss = entry_price * (1 - self.stop_loss)  # -2%
            bracket_take_profit = entry_price * (1 + self.take_profit)  # +4%

            # Log when approaching bracket levels (for monitoring)
            if current_price <= bracket_stop_loss * 1.005:  # Within 0.5% of stop
                logger.debug(f"{symbol} approaching bracket stop-loss: ${current_price:.2f} near ${bracket_stop_loss:.2f}")

            if current_price >= bracket_take_profit * 0.995:  # Within 0.5% of take-profit
                logger.debug(f"{symbol} approaching bracket take-profit: ${current_price:.2f} near ${bracket_take_profit:.2f}")

            # Note: Basic stop-loss and take-profit are handled by bracket orders at broker level
            # No manual sell orders needed - bracket will execute automatically
                
        except Exception as e:
            logger.error(f"Error checking exit conditions for {symbol}: {e}", exc_info=True)
    
    async def analyze_symbol(self, symbol):
        """Analyze a symbol and return trading signal."""
        return self.signals.get(symbol, 'neutral')
    
    async def execute_trade(self, symbol, signal):
        """Execute a trade based on the signal."""
        # This is already handled in _execute_signal
        pass
    
    async def generate_signals(self):
        """Generate signals for all symbols (used in backtest mode)."""
        for symbol in self.symbols:
            if symbol in self.current_data:
                df = self.current_data[symbol]
                if len(df) < self.sma_period:
                    continue
                    
                # Extract price data
                closes = df['close'].values
                highs = df['high'].values
                lows = df['low'].values
                
                # Calculate indicators
                upper, middle, lower = talib.BBANDS(
                    closes, 
                    timeperiod=self.bb_period,
                    nbdevup=self.bb_std,
                    nbdevdn=self.bb_std,
                    matype=0
                )
                
                rsi = talib.RSI(closes, timeperiod=self.rsi_period)
                sma = talib.SMA(closes, timeperiod=self.sma_period)
                std = talib.STDDEV(closes, timeperiod=self.mean_lookback)
                
                slowk, slowd = talib.STOCH(
                    highs, 
                    lows, 
                    closes, 
                    fastk_period=14,
                    slowk_period=3,
                    slowk_matype=0,
                    slowd_period=3,
                    slowd_matype=0
                )
                
                # Calculate z-score
                z_score = (closes[-1] - sma[-1]) / std[-1] if len(std) > 0 and std[-1] > 0 else 0
                
                # Calculate BB position
                bb_range = upper[-1] - lower[-1] if len(upper) > 0 else 0
                bb_position = (closes[-1] - lower[-1]) / bb_range if bb_range > 0 else 0.5
                
                # Store the indicators
                self.indicators[symbol] = {
                    'upper_band': upper[-1] if len(upper) > 0 else None,
                    'middle_band': middle[-1] if len(middle) > 0 else None,
                    'lower_band': lower[-1] if len(lower) > 0 else None,
                    'rsi': rsi[-1] if len(rsi) > 0 else None,
                    'sma': sma[-1] if len(sma) > 0 else None,
                    'std': std[-1] if len(std) > 0 else None,
                    'z_score': z_score,
                    'bb_position': bb_position,
                    'slowk': slowk[-1] if len(slowk) > 0 else None,
                    'slowd': slowd[-1] if len(slowd) > 0 else None,
                    'close': closes[-1] if len(closes) > 0 else None
                }
                
                # Generate signal
                signal = await self._generate_signal(symbol)
                self.signals[symbol] = signal
    
    def get_orders(self):
        """Get orders for backtest mode."""
        orders = []
        
        for symbol, signal in self.signals.items():
            if signal == 'neutral':
                continue
                
            # Get current positions (for backtest)
            current_positions = getattr(self, 'positions', {})
            has_position = symbol in current_positions
            
            # Current price
            price = self.indicators[symbol]['close']
            if not price:
                continue
                
            # Buy signal
            if signal == 'buy' and not has_position:
                # Calculate position size (simplified for backtest)
                capital = getattr(self, 'capital', 100000)
                position_size = capital * self.position_size
                quantity = position_size / price

                # Allow fractional shares (minimum 0.01 shares)
                if quantity >= 0.01:
                    orders.append({
                        'symbol': symbol,
                        'quantity': quantity,  # Keep fractional quantity
                        'side': 'buy',
                        'type': 'market'
                    })
                    
            # Sell signal
            elif signal == 'sell' and has_position:
                position = current_positions[symbol]
                quantity = position.get('quantity', 0)
                
                if quantity > 0:
                    orders.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'side': 'sell',
                        'type': 'market'
                    })
        
        return orders
