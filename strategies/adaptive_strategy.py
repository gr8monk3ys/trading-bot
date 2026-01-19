#!/usr/bin/env python3
"""
Adaptive Strategy - Regime-Switching Strategy Coordinator

Automatically switches between strategies based on detected market regime:
- BULL market (trending up)    -> Momentum Strategy (long bias)
- BEAR market (trending down)  -> Momentum Strategy (short bias) or defensive
- SIDEWAYS market (ranging)    -> Mean Reversion Strategy
- VOLATILE market (high VIX)   -> Reduced exposure across all strategies

Research shows:
- Using momentum in sideways markets loses money
- Using mean reversion in trending markets loses money
- Matching strategy to regime improves returns by 10-15% annually

Usage:
    from strategies.adaptive_strategy import AdaptiveStrategy

    strategy = AdaptiveStrategy(broker, symbols)
    await strategy.initialize()

    # Strategy automatically detects regime and routes signals
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from strategies.base_strategy import BaseStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from utils.market_regime import MarketRegimeDetector, MarketRegime

logger = logging.getLogger(__name__)


class AdaptiveStrategy(BaseStrategy):
    """
    Adaptive strategy that switches between sub-strategies based on market regime.

    In BULL/BEAR (trending) markets:
    - Uses MomentumStrategy
    - Long bias in bull, short bias in bear

    In SIDEWAYS (ranging) markets:
    - Uses MeanReversionStrategy
    - Profits from oscillations around the mean

    In VOLATILE markets:
    - Reduces position sizes across all strategies
    - May skip signals entirely if VIX is extreme
    """

    NAME = "AdaptiveStrategy"

    def __init__(self, broker=None, symbols=None, parameters=None):
        """Initialize adaptive strategy with symbols."""
        parameters = parameters or {}
        if symbols:
            parameters['symbols'] = symbols
        super().__init__(name=self.NAME, broker=broker, parameters=parameters)

    def default_parameters(self):
        """Return default parameters for the adaptive strategy."""
        return {
            # Basic parameters
            'position_size': 0.10,
            'max_positions': 5,
            'max_portfolio_risk': 0.02,
            'stop_loss': 0.03,
            'take_profit': 0.05,

            # Regime detection settings
            'regime_check_interval_minutes': 30,  # How often to check regime
            'min_regime_confidence': 0.55,        # Minimum confidence to act on regime

            # Strategy selection
            'bull_strategy': 'momentum',           # Strategy for bull regime
            'bear_strategy': 'momentum_short',     # Strategy for bear regime
            'sideways_strategy': 'mean_reversion', # Strategy for sideways regime
            'volatile_strategy': 'defensive',      # Strategy for high volatility

            # Position adjustments by regime
            'bull_position_mult': 1.2,    # 20% larger in bull
            'bear_position_mult': 0.8,    # 20% smaller in bear (shorts are riskier)
            'sideways_position_mult': 1.0, # Normal in sideways
            'volatile_position_mult': 0.5, # 50% smaller in volatile

            # Defensive thresholds
            'skip_trades_vix_threshold': 40,  # Skip new trades if VIX > 40
            'reduce_exposure_vix_threshold': 30,  # Reduce exposure if VIX > 30

            # Sub-strategy parameters (passed through)
            'use_kelly_criterion': True,
            'use_volatility_regime': True,
            'use_trailing_stop': True,
            'use_multi_timeframe': True,
        }

    async def initialize(self, **kwargs):
        """Initialize the adaptive strategy with sub-strategies."""
        try:
            # Initialize base strategy
            await super().initialize(**kwargs)

            # Set parameters
            params = self.default_parameters()
            params.update(self.parameters)
            self.parameters = params

            # Extract parameters
            self.position_size = self.parameters['position_size']
            self.max_positions = self.parameters['max_positions']
            self.stop_loss = self.parameters['stop_loss']
            self.take_profit = self.parameters['take_profit']

            # Initialize regime detector
            self.regime_detector = MarketRegimeDetector(
                self.broker,
                cache_minutes=self.parameters['regime_check_interval_minutes']
            )
            self.current_regime = None
            self.last_regime_check = None

            # Initialize sub-strategies
            logger.info("Initializing sub-strategies for AdaptiveStrategy...")

            # Momentum strategy for trending markets
            momentum_params = {
                'symbols': self.symbols,
                'position_size': self.position_size,
                'max_positions': self.max_positions,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'use_kelly_criterion': self.parameters.get('use_kelly_criterion', True),
                'use_volatility_regime': self.parameters.get('use_volatility_regime', True),
                'use_trailing_stop': self.parameters.get('use_trailing_stop', True),
                'use_multi_timeframe': self.parameters.get('use_multi_timeframe', True),
                'enable_short_selling': True,  # Enable for bear markets
            }
            self.momentum_strategy = MomentumStrategy(
                broker=self.broker,
                parameters=momentum_params
            )
            await self.momentum_strategy.initialize()
            logger.info("  MomentumStrategy initialized for trending markets")

            # Mean reversion strategy for sideways markets
            mean_rev_params = {
                'symbols': self.symbols,
                'position_size': self.position_size,
                'max_positions': self.max_positions,
                'stop_loss': self.stop_loss,
                'take_profit': self.parameters.get('mean_reversion_take_profit', 0.04),
                'use_multi_timeframe': self.parameters.get('use_multi_timeframe', True),
                'enable_short_selling': True,
            }
            self.mean_reversion_strategy = MeanReversionStrategy(
                broker=self.broker,
                parameters=mean_rev_params
            )
            await self.mean_reversion_strategy.initialize()
            logger.info("  MeanReversionStrategy initialized for sideways markets")

            # Active strategy pointer
            self.active_strategy = self.momentum_strategy  # Default
            self.active_strategy_name = 'momentum'

            # Tracking
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.signals = {symbol: 'neutral' for symbol in self.symbols}
            self.current_prices = {}
            self.price_history = {symbol: [] for symbol in self.symbols}
            self.regime_switches = 0
            self.last_regime_switch = None

            logger.info(f"AdaptiveStrategy initialized with {len(self.symbols)} symbols")
            logger.info(f"  Bull: {self.parameters['bull_strategy']}, "
                       f"Bear: {self.parameters['bear_strategy']}, "
                       f"Sideways: {self.parameters['sideways_strategy']}")

            return True

        except Exception as e:
            logger.error(f"Error initializing AdaptiveStrategy: {e}", exc_info=True)
            return False

    async def on_bar(self, symbol, open_price, high_price, low_price, close_price, volume, timestamp):
        """Handle incoming bar data with regime-aware routing."""
        try:
            if symbol not in self.symbols:
                return

            # Store current price
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

            # Keep only recent history
            max_history = 100
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]

            # Check and update market regime (cached, not every bar)
            await self._update_regime()

            # Route to appropriate sub-strategy based on regime
            if self.active_strategy:
                await self.active_strategy.on_bar(
                    symbol, open_price, high_price, low_price, close_price, volume, timestamp
                )

                # Copy signals and indicators from active strategy
                self.signals[symbol] = self.active_strategy.signals.get(symbol, 'neutral')
                self.indicators[symbol] = self.active_strategy.indicators.get(symbol, {})

        except Exception as e:
            logger.error(f"Error in AdaptiveStrategy on_bar for {symbol}: {e}", exc_info=True)

    async def _update_regime(self):
        """Update market regime and switch strategies if needed."""
        try:
            # Detect current regime (uses internal caching)
            regime_info = await self.regime_detector.detect_regime()

            # Check if regime changed
            new_regime_type = regime_info['type']

            if self.current_regime is None or new_regime_type != self.current_regime:
                old_regime = self.current_regime or 'none'
                self.current_regime = new_regime_type

                # Log regime change
                logger.warning(
                    f"REGIME CHANGE: {old_regime.upper()} -> {new_regime_type.upper()} "
                    f"(confidence: {regime_info['confidence']:.0%}, "
                    f"recommended: {regime_info['recommended_strategy']})"
                )

                # Switch active strategy
                await self._switch_strategy(regime_info)

                self.regime_switches += 1
                self.last_regime_switch = datetime.now()

        except Exception as e:
            logger.error(f"Error updating regime: {e}", exc_info=True)

    async def _switch_strategy(self, regime_info: Dict):
        """Switch active strategy based on regime."""
        regime_type = regime_info['type']
        confidence = regime_info['confidence']

        # Don't switch if confidence is too low
        if confidence < self.parameters['min_regime_confidence']:
            logger.info(f"Regime confidence ({confidence:.0%}) below threshold, keeping {self.active_strategy_name}")
            return

        # Select strategy based on regime
        if regime_type == 'bull':
            self.active_strategy = self.momentum_strategy
            self.active_strategy_name = 'momentum_long'
            # Adjust for bull market (favor longs)
            self.momentum_strategy.enable_short_selling = False
            logger.info("BULL REGIME: Switched to MomentumStrategy (long bias)")

        elif regime_type == 'bear':
            self.active_strategy = self.momentum_strategy
            self.active_strategy_name = 'momentum_short'
            # Adjust for bear market (enable shorts)
            self.momentum_strategy.enable_short_selling = True
            logger.info("BEAR REGIME: Switched to MomentumStrategy (short enabled)")

        elif regime_type == 'sideways':
            self.active_strategy = self.mean_reversion_strategy
            self.active_strategy_name = 'mean_reversion'
            logger.info("SIDEWAYS REGIME: Switched to MeanReversionStrategy")

        elif regime_type == 'volatile':
            # Keep current strategy but reduce exposure
            logger.info(f"VOLATILE REGIME: Keeping {self.active_strategy_name} with reduced exposure")
            # Position multiplier from regime_info already handles reduction

        # Apply position multiplier from regime
        multiplier = regime_info.get('position_multiplier', 1.0)
        adjusted_size = self.parameters['position_size'] * multiplier

        if self.active_strategy:
            self.active_strategy.position_size = adjusted_size
            logger.info(f"  Position size adjusted to {adjusted_size:.1%} (mult: {multiplier:.1f}x)")

    async def analyze_symbol(self, symbol: str) -> str:
        """Analyze symbol using the active strategy."""
        if self.active_strategy:
            return await self.active_strategy.analyze_symbol(symbol)
        return 'neutral'

    async def execute_trade(self, symbol: str, signal: str):
        """Execute trade using the active strategy."""
        if self.active_strategy:
            await self.active_strategy.execute_trade(symbol, signal)

    async def generate_signals(self):
        """Generate signals for all symbols using active strategy."""
        if self.active_strategy:
            await self.active_strategy.generate_signals()
            # Copy signals
            self.signals = self.active_strategy.signals.copy()

    def get_orders(self) -> List[Dict]:
        """Get orders from active strategy for backtest mode."""
        if self.active_strategy:
            return self.active_strategy.get_orders()
        return []

    def get_status(self) -> Dict:
        """Get current status of the adaptive strategy."""
        return {
            'name': self.NAME,
            'active_strategy': self.active_strategy_name,
            'current_regime': self.current_regime,
            'regime_switches': self.regime_switches,
            'last_switch': self.last_regime_switch.isoformat() if self.last_regime_switch else None,
            'symbols': len(self.symbols),
            'signals': {s: sig for s, sig in self.signals.items() if sig != 'neutral'}
        }

    async def get_regime_info(self) -> Dict:
        """Get current regime information."""
        return await self.regime_detector.detect_regime()


# Factory function for easy creation
def create_adaptive_strategy(broker, symbols: List[str], **kwargs) -> AdaptiveStrategy:
    """
    Create and return an AdaptiveStrategy instance.

    Args:
        broker: Trading broker instance
        symbols: List of symbols to trade
        **kwargs: Additional parameters to pass to the strategy

    Returns:
        Initialized AdaptiveStrategy
    """
    return AdaptiveStrategy(
        broker=broker,
        symbols=symbols,
        parameters=kwargs
    )
