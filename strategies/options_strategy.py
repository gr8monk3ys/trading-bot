"""
Options Trading Strategy

Advanced options strategies with defined risk and income generation.

Strategies Implemented:
1. Covered Calls - Income generation on long stock positions
2. Cash-Secured Puts - Get paid to wait for entry price
3. Call Debit Spreads - Bullish with defined risk
4. Put Debit Spreads - Bearish with defined risk
5. Iron Condor - High volatility, range-bound strategy
6. Protective Puts - Portfolio insurance

Expected Use Cases:
- Income generation (selling premium)
- Defined-risk directional trades
- Hedging existing positions
- Leverage with controlled risk

Risk Profile: Medium to High (requires options knowledge)
Best For: Experienced traders familiar with options mechanics
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import ContractType, OptionSide

from strategies.base_strategy import BaseStrategy
from strategies.risk_manager import RiskManager
from brokers.order_builder import OrderBuilder
from utils.indicators import TechnicalIndicators, analyze_trend, analyze_momentum, analyze_volatility

logger = logging.getLogger(__name__)


class OptionsStrategy(BaseStrategy):
    """
    Advanced options trading strategy.

    Features:
    - Multiple option strategies (covered calls, spreads, condors)
    - Implied volatility analysis
    - Greeks monitoring (delta, theta, vega)
    - Risk-defined trades
    - Income generation

    WARNING: Options trading is complex and risky. Only use with paper trading first.
    """

    NAME = "OptionsStrategy"

    def default_parameters(self):
        """Return default parameters."""
        return {
            # Basic parameters
            'position_size': 0.10,  # 10% of portfolio per trade
            'max_positions': 3,  # Limit options exposure
            'max_portfolio_risk': 0.02,

            # Options-specific parameters
            'option_allocation': 0.20,  # Only use 20% of portfolio for options
            'min_days_to_expiry': 20,  # Minimum days before expiration
            'max_days_to_expiry': 45,  # Maximum days before expiration
            'preferred_expiry_days': 30,  # Target 30-day options

            # Strategy selection
            'enable_covered_calls': True,
            'enable_cash_secured_puts': True,
            'enable_call_spreads': True,
            'enable_put_spreads': True,
            'enable_iron_condor': False,  # Advanced - disabled by default
            'enable_protective_puts': True,

            # Entry criteria
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std': 2.0,
            'high_iv_threshold': 0.30,  # 30% IV = high
            'low_iv_threshold': 0.15,   # 15% IV = low

            # Option strike selection
            'call_strike_otm_pct': 0.05,  # 5% out of the money
            'put_strike_otm_pct': 0.05,
            'spread_width': 5,  # $5 spread width
            'iron_condor_otm_pct': 0.10,  # 10% OTM for iron condor

            # Exit criteria
            'profit_target_pct': 0.50,  # Close at 50% of max profit
            'stop_loss_pct': 0.30,  # Stop at 30% loss
            'close_days_before_expiry': 7,  # Close 1 week before expiration

            # Risk management
            'max_option_allocation': 0.20,  # Max 20% in options
            'max_correlation': 0.7,
        }

    async def initialize(self, **kwargs):
        """Initialize options strategy."""
        try:
            await super().initialize(**kwargs)

            # Set parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            # Extract key parameters
            self.position_size = self.parameters['position_size']
            self.max_positions = self.parameters['max_positions']
            self.option_allocation = self.parameters['option_allocation']

            # Initialize tracking
            self.current_prices = {}
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.option_positions = {}  # Track option positions separately
            self.underlying_positions = {}  # Track underlying stock positions
            self.signals = {symbol: 'neutral' for symbol in self.symbols}

            # Risk manager
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.parameters['max_portfolio_risk'],
                max_position_risk=self.parameters.get('max_position_risk', 0.01),
                max_correlation=self.parameters['max_correlation']
            )

            logger.info(f"Initialized {self.NAME} with {len(self.symbols)} symbols")
            logger.info(f"  Enabled strategies: {self._get_enabled_strategies()}")
            logger.info(f"  Option allocation: {self.option_allocation:.0%}")
            logger.info(f"  Preferred expiry: {self.parameters['preferred_expiry_days']} days")
            logger.warning(f"  ⚠️  OPTIONS TRADING: Paper trade first! Complex and risky.")

            return True

        except Exception as e:
            logger.error(f"Error initializing {self.NAME}: {e}", exc_info=True)
            return False

    def _get_enabled_strategies(self) -> str:
        """Get list of enabled option strategies."""
        strategies = []
        if self.parameters['enable_covered_calls']:
            strategies.append('Covered Calls')
        if self.parameters['enable_cash_secured_puts']:
            strategies.append('Cash-Secured Puts')
        if self.parameters['enable_call_spreads']:
            strategies.append('Call Spreads')
        if self.parameters['enable_put_spreads']:
            strategies.append('Put Spreads')
        if self.parameters['enable_iron_condor']:
            strategies.append('Iron Condor')
        if self.parameters['enable_protective_puts']:
            strategies.append('Protective Puts')
        return ', '.join(strategies)

    async def on_bar(self, symbol, open_price, high_price, low_price, close_price, volume, timestamp):
        """Handle incoming bar data."""
        try:
            if symbol not in self.symbols:
                return

            # Store current price
            self.current_prices[symbol] = close_price

            # Update price history for indicator calculations
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            self.price_history[symbol].append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

            # Keep only recent history (200 bars for indicators)
            if len(self.price_history[symbol]) > 200:
                self.price_history[symbol] = self.price_history[symbol][-200:]

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    async def on_trading_iteration(self):
        """Main trading logic - called periodically."""
        try:
            # Check circuit breaker
            if not await self.circuit_breaker.check_trading_allowed():
                logger.warning("Circuit breaker triggered - trading halted")
                return

            # Get account info
            account = await self.broker.get_account()
            portfolio_value = float(account.equity)

            # Check option allocation limit
            total_option_value = await self._get_total_option_value()
            if total_option_value > portfolio_value * self.max_option_allocation:
                logger.warning(f"Max option allocation reached: {total_option_value/portfolio_value:.1%}")
                return

            # Analyze each symbol
            for symbol in self.symbols:
                try:
                    # Need sufficient price history
                    if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
                        continue

                    # Analyze market conditions
                    await self._analyze_symbol(symbol)

                    # Generate option strategy signal
                    strategy_type = await self._select_strategy(symbol)

                    if strategy_type:
                        # Execute the selected strategy
                        await self._execute_strategy(symbol, strategy_type)

                    # Manage existing positions
                    await self._manage_positions(symbol)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}", exc_info=True)

    async def _analyze_symbol(self, symbol: str):
        """Analyze symbol with technical indicators."""
        try:
            # Extract price data
            df = pd.DataFrame(self.price_history[symbol])
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values

            # Calculate indicators using our library
            ind = TechnicalIndicators(high=highs, low=lows, close=closes, volume=volumes)

            # Get analysis
            trend = analyze_trend(closes, highs, lows)
            momentum = analyze_momentum(closes, highs, lows)
            volatility = analyze_volatility(closes, highs, lows)

            # Calculate specific indicators
            rsi = ind.rsi(period=self.parameters['rsi_period'])
            bb_upper, bb_middle, bb_lower = ind.bollinger_bands(
                period=self.parameters['bb_period'],
                std=self.parameters['bb_std']
            )

            # Estimate implied volatility (using historical volatility as proxy)
            # In production, would use actual IV from options chain
            historical_vol = volatility.get('atr_pct', 0.02)

            # Store indicators
            self.indicators[symbol] = {
                'rsi': rsi[-1] if len(rsi) > 0 else 50,
                'bb_upper': bb_upper[-1] if len(bb_upper) > 0 else closes[-1],
                'bb_middle': bb_middle[-1] if len(bb_middle) > 0 else closes[-1],
                'bb_lower': bb_lower[-1] if len(bb_lower) > 0 else closes[-1],
                'trend': trend.get('direction', 'neutral'),
                'trend_strength': trend.get('strength', 'weak'),
                'momentum_condition': momentum.get('condition', 'neutral'),
                'volatility_state': volatility.get('state', 'normal'),
                'historical_vol': historical_vol,
                'current_price': closes[-1]
            }

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)

    async def _select_strategy(self, symbol: str) -> Optional[str]:
        """Select appropriate option strategy based on market conditions."""
        try:
            if symbol not in self.indicators:
                return None

            ind = self.indicators[symbol]
            rsi = ind['rsi']
            trend = ind['trend']
            volatility = ind['historical_vol']
            price = ind['current_price']

            # Check if we have existing stock position
            has_stock = await self._has_stock_position(symbol)

            # Strategy selection logic

            # 1. Covered Calls (need to own stock)
            if (self.parameters['enable_covered_calls'] and
                has_stock and
                rsi > self.parameters['rsi_overbought']):
                return 'covered_call'

            # 2. Protective Puts (hedging existing stock)
            if (self.parameters['enable_protective_puts'] and
                has_stock and
                trend == 'bearish' and
                volatility < self.parameters['low_iv_threshold']):
                return 'protective_put'

            # 3. Cash-Secured Puts (want to own stock at lower price)
            if (self.parameters['enable_cash_secured_puts'] and
                not has_stock and
                rsi < self.parameters['rsi_oversold'] and
                trend != 'strongly_bearish'):
                return 'cash_secured_put'

            # 4. Call Debit Spread (bullish, high IV)
            if (self.parameters['enable_call_spreads'] and
                trend == 'bullish' and
                rsi < 60 and
                volatility > self.parameters['high_iv_threshold']):
                return 'call_debit_spread'

            # 5. Put Debit Spread (bearish, high IV)
            if (self.parameters['enable_put_spreads'] and
                trend == 'bearish' and
                rsi > 40 and
                volatility > self.parameters['high_iv_threshold']):
                return 'put_debit_spread'

            # 6. Iron Condor (range-bound, high IV)
            if (self.parameters['enable_iron_condor'] and
                trend == 'neutral' and
                volatility > self.parameters['high_iv_threshold'] and
                40 < rsi < 60):
                return 'iron_condor'

            return None

        except Exception as e:
            logger.error(f"Error selecting strategy for {symbol}: {e}", exc_info=True)
            return None

    async def _execute_strategy(self, symbol: str, strategy_type: str):
        """Execute the selected option strategy."""
        try:
            logger.info(f"Executing {strategy_type} for {symbol}")

            # Route to specific strategy implementation
            if strategy_type == 'covered_call':
                await self._execute_covered_call(symbol)
            elif strategy_type == 'protective_put':
                await self._execute_protective_put(symbol)
            elif strategy_type == 'cash_secured_put':
                await self._execute_cash_secured_put(symbol)
            elif strategy_type == 'call_debit_spread':
                await self._execute_call_spread(symbol)
            elif strategy_type == 'put_debit_spread':
                await self._execute_put_spread(symbol)
            elif strategy_type == 'iron_condor':
                await self._execute_iron_condor(symbol)

        except Exception as e:
            logger.error(f"Error executing {strategy_type} for {symbol}: {e}", exc_info=True)

    async def _execute_covered_call(self, symbol: str):
        """
        Covered Call: Sell call against existing stock position.
        Income strategy - collect premium while owning stock.
        """
        try:
            # Note: This is a skeleton implementation
            # Alpaca options API integration would go here

            logger.info(f"Covered call strategy for {symbol}")
            logger.warning(f"Options API integration required - not yet implemented")

            # TODO: Implement when options API is fully available
            # 1. Get stock position size
            # 2. Find call option at strike price OTM
            # 3. Sell call options (quantity = stock_shares / 100)
            # 4. Track position

        except Exception as e:
            logger.error(f"Error in covered call for {symbol}: {e}", exc_info=True)

    async def _execute_protective_put(self, symbol: str):
        """
        Protective Put: Buy put to hedge stock position.
        Insurance strategy - protect against downside.
        """
        try:
            logger.info(f"Protective put strategy for {symbol}")
            logger.warning(f"Options API integration required - not yet implemented")

            # TODO: Implement
            # 1. Get stock position size
            # 2. Buy put option at appropriate strike
            # 3. Track as hedge position

        except Exception as e:
            logger.error(f"Error in protective put for {symbol}: {e}", exc_info=True)

    async def _execute_cash_secured_put(self, symbol: str):
        """
        Cash-Secured Put: Sell put, reserve cash to buy stock if assigned.
        Entry strategy - get paid to wait for lower entry price.
        """
        try:
            logger.info(f"Cash-secured put strategy for {symbol}")
            logger.warning(f"Options API integration required - not yet implemented")

            # TODO: Implement
            # 1. Calculate strike price (current_price * 0.95)
            # 2. Find put option at strike
            # 3. Sell put option
            # 4. Reserve cash (strike * 100 * quantity)

        except Exception as e:
            logger.error(f"Error in cash-secured put for {symbol}: {e}", exc_info=True)

    async def _execute_call_spread(self, symbol: str):
        """
        Call Debit Spread: Buy call, sell higher call.
        Defined-risk bullish strategy.
        """
        try:
            logger.info(f"Call debit spread strategy for {symbol}")
            logger.warning(f"Options API integration required - not yet implemented")

            # TODO: Implement
            # 1. Buy call at lower strike (ATM or slightly ITM)
            # 2. Sell call at higher strike (OTM)
            # 3. Max profit = spread width - net debit
            # 4. Max loss = net debit paid

        except Exception as e:
            logger.error(f"Error in call spread for {symbol}: {e}", exc_info=True)

    async def _execute_put_spread(self, symbol: str):
        """
        Put Debit Spread: Buy put, sell lower put.
        Defined-risk bearish strategy.
        """
        try:
            logger.info(f"Put debit spread strategy for {symbol}")
            logger.warning(f"Options API integration required - not yet implemented")

            # TODO: Implement
            # Similar to call spread but with puts

        except Exception as e:
            logger.error(f"Error in put spread for {symbol}: {e}", exc_info=True)

    async def _execute_iron_condor(self, symbol: str):
        """
        Iron Condor: Sell OTM call spread + sell OTM put spread.
        Neutral strategy - profit from low volatility and range-bound movement.
        """
        try:
            logger.info(f"Iron condor strategy for {symbol}")
            logger.warning(f"Options API integration required - not yet implemented")

            # TODO: Implement
            # 1. Sell OTM put spread (lower side)
            # 2. Sell OTM call spread (upper side)
            # 3. Max profit = net credit received
            # 4. Max loss = spread width - net credit

        except Exception as e:
            logger.error(f"Error in iron condor for {symbol}: {e}", exc_info=True)

    async def _manage_positions(self, symbol: str):
        """Manage existing option positions - take profits, stop losses, close before expiry."""
        try:
            # TODO: Implement position management
            # 1. Check days to expiration
            # 2. Check P/L vs profit target / stop loss
            # 3. Close positions meeting exit criteria
            pass

        except Exception as e:
            logger.error(f"Error managing positions for {symbol}: {e}", exc_info=True)

    async def _has_stock_position(self, symbol: str) -> bool:
        """Check if we have an existing stock position."""
        try:
            positions = await self.broker.get_positions()
            for pos in positions:
                if pos.symbol == symbol and float(pos.qty) > 0:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking stock position for {symbol}: {e}")
            return False

    async def _get_total_option_value(self) -> float:
        """Calculate total value of option positions."""
        try:
            # TODO: Calculate total option position value
            # Would need to track option positions separately
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating option value: {e}")
            return 0.0
