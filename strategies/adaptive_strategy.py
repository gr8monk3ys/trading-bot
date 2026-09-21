#!/usr/bin/env python3
"""
Adaptive Strategy - Regime-Switching Strategy Coordinator

Automatically switches between strategies based on detected market regime:
- BULL market (trending up)    -> Momentum Strategy (long bias)
- BEAR market (trending down)  -> Momentum Strategy (short bias)
- SIDEWAYS market (ranging)    -> Mean Reversion Strategy
- VOLATILE market (high VIX)   -> Reduced exposure across all strategies

Status: UNVALIDATED. This coordinator has never been backtested, and neither
has its sideways arm (MeanReversionStrategy). It cannot be backtested by the
current engine either, because all of its routing lives in `on_bar` and the
engine never calls it. A previous version of this docstring claimed regime
matching "improves returns by 10-15% annually"; nothing in this repository
supports that number, so it has been removed rather than cited.

Arms never subscribe to bars; the LiveSession that drives this coordinator does.

Usage:
    from strategies.adaptive_strategy import AdaptiveStrategy

    strategy = AdaptiveStrategy(broker, symbols)
    await strategy.initialize()

    # Strategy automatically detects regime and routes signals
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from strategies.params import AdaptiveParams
from utils.market_regime import MarketRegimeDetector

logger = logging.getLogger(__name__)


class AdaptiveStrategy(BaseStrategy):
    Params = AdaptiveParams
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
    """

    NAME = "AdaptiveStrategy"

    def __init__(
        self,
        broker=None,
        symbols=None,
        parameters=None,
        order_submission=None,
    ):
        """
        Initialize adaptive strategy with symbols.

        Args:
            broker: Trading broker instance
            symbols: List of symbols to trade
            parameters: Strategy parameters
            order_submission: Optional OrderSubmission shared with both arms
        """
        parameters = parameters or {}
        if symbols:
            parameters["symbols"] = symbols
        super().__init__(
            name=self.NAME,
            broker=broker,
            parameters=parameters,
            order_submission=order_submission,
        )

    def default_parameters(self):
        """The defaults live once, on ``Params`` (strategies/params.py)."""
        return dict(self.Params.defaults())

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
            self.position_size = self.parameters["position_size"]
            self.max_positions = self.parameters["max_positions"]
            self.stop_loss = self.parameters["stop_loss"]
            self.take_profit = self.parameters["take_profit"]

            # Initialize regime detector
            self.regime_detector = MarketRegimeDetector(
                self.broker, cache_minutes=self.parameters["regime_check_interval_minutes"]
            )
            self.current_regime = None
            self.last_regime_check = None

            # Initialize sub-strategies
            logger.info("Initializing sub-strategies for AdaptiveStrategy...")

            # Momentum strategy for trending markets
            momentum_params = {
                "symbols": self.symbols,
                "position_size": self.position_size,
                "max_positions": self.max_positions,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "use_kelly_criterion": self.parameters["use_kelly_criterion"],
                "use_volatility_regime": self.parameters["use_volatility_regime"],
                "use_trailing_stop": self.parameters["use_trailing_stop"],
                "use_multi_timeframe": self.parameters["use_multi_timeframe"],
                "enable_short_selling": True,  # Enable for bear markets
            }
            self.momentum_strategy = MomentumStrategy(
                broker=self.broker,
                parameters=momentum_params,
                order_submission=self.order_submission,
            )

            # Mean reversion strategy for sideways markets
            mean_rev_params = {
                "symbols": self.symbols,
                "position_size": self.position_size,
                "max_positions": self.max_positions,
                "stop_loss": self.stop_loss,
                "take_profit": self.parameters["mean_reversion_take_profit"],
                "use_multi_timeframe": self.parameters["use_multi_timeframe"],
                "enable_short_selling": True,
            }
            self.mean_reversion_strategy = MeanReversionStrategy(
                broker=self.broker,
                parameters=mean_rev_params,
                order_submission=self.order_submission,
            )

            # Performance optimization: Initialize sub-strategies in parallel
            await asyncio.gather(
                self.momentum_strategy.initialize(), self.mean_reversion_strategy.initialize()
            )
            logger.info("  MomentumStrategy initialized for trending markets")
            logger.info("  MeanReversionStrategy initialized for sideways markets")

            # Take ownership of the bar feed. Each sub-strategy subscribes
            # itself during initialize(); left alone, both arms receive every
            # bar and trade independently through their own gateways, while
            # this coordinator never runs on_bar and so never detects a regime
            # at all. The arms must only ever be driven via on_bar routing.

            # Active strategy pointer
            self.active_strategy = self.momentum_strategy  # Default
            self.active_strategy_name = "momentum"

            # Tracking
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.signals = dict.fromkeys(self.symbols, "neutral")
            self.current_prices = {}
            # Performance optimization: Use deque with maxlen for O(1) append and auto-trimming
            self.price_history = {symbol: deque(maxlen=100) for symbol in self.symbols}
            self.regime_switches = 0
            self.last_regime_switch = None

            logger.info(f"AdaptiveStrategy initialized with {len(self.symbols)} symbols")
            logger.info(
                f"  Bull: {self.parameters['bull_strategy']}, "
                f"Bear: {self.parameters['bear_strategy']}, "
                f"Sideways: {self.parameters['sideways_strategy']}"
            )

            return True

        except Exception as e:
            logger.error(f"Error initializing AdaptiveStrategy: {e}", exc_info=True)
            return False

    async def _update_regime(self):
        """Update market regime and switch strategies if needed."""
        try:
            # Detect current regime (uses internal caching)
            regime_info = await self.regime_detector.detect_regime()

            # Check if regime changed
            new_regime_type = regime_info["type"]

            if self.current_regime is None or new_regime_type != self.current_regime:
                old_regime = self.current_regime or "none"
                self.current_regime = new_regime_type

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
        regime_type = regime_info["type"]
        confidence = regime_info["confidence"]

        # Don't switch if confidence is too low
        if confidence < self.parameters["min_regime_confidence"]:
            logger.info(
                f"Regime confidence ({confidence:.0%}) below threshold, keeping {self.active_strategy_name}"
            )
            return

        # Select strategy based on regime
        if regime_type == "bull":
            self.active_strategy = self.momentum_strategy
            self.active_strategy_name = "momentum_long"
            # Adjust for bull market (favor longs)
            self.momentum_strategy.enable_short_selling = False
            logger.info("BULL REGIME: Switched to MomentumStrategy (long bias)")

        elif regime_type == "bear":
            self.active_strategy = self.momentum_strategy
            self.active_strategy_name = "momentum_short"
            # Adjust for bear market (enable shorts)
            self.momentum_strategy.enable_short_selling = True
            logger.info("BEAR REGIME: Switched to MomentumStrategy (short enabled)")

        elif regime_type == "sideways":
            self.active_strategy = self.mean_reversion_strategy
            self.active_strategy_name = "mean_reversion"
            logger.info("SIDEWAYS REGIME: Switched to MeanReversionStrategy")

        elif regime_type == "volatile":
            # Keep current strategy but reduce exposure
            logger.info(
                f"VOLATILE REGIME: Keeping {self.active_strategy_name} with reduced exposure"
            )
            # Position multiplier from regime_info already handles reduction

        # Apply position multiplier from regime
        multiplier = regime_info.get("position_multiplier", 1.0)
        adjusted_size = self.parameters["position_size"] * multiplier

        if self.active_strategy:
            self.active_strategy.position_size = adjusted_size
            logger.info(
                f"  Position size adjusted to {adjusted_size:.1%} (mult: {multiplier:.1f}x)"
            )

    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze symbol using the active strategy.

        Returns:
            Dict with action and confidence from the active sub-strategy.
        """
        if not self.active_strategy:
            return {"action": "neutral", "confidence": 0.0}

        # Get technical signal from active strategy
        technical_result = await self.active_strategy.analyze_symbol(symbol)

        # Convert to dict if it's a string
        if isinstance(technical_result, str):
            return {
                "action": technical_result,
                "confidence": 0.5 if technical_result != "neutral" else 0.0,
                "reason": f"Technical analysis ({self.active_strategy_name})",
            }
        if isinstance(technical_result, dict):
            return technical_result
        return {"action": "neutral", "confidence": 0.0}

    async def prepare(self, when, histories) -> None:
        mode = getattr(self, "execution_mode", "daily")
        for arm in (self.momentum_strategy, self.mean_reversion_strategy):
            if arm is not None:
                arm.execution_mode = mode
        if mode == "live":
            for symbol, df in histories.items():
                if len(df):
                    self.current_prices[symbol] = float(df["close"].iloc[-1])
            await self._update_regime()
        if self.active_strategy:
            await self.active_strategy.prepare(when, histories)
            self.signals = self.active_strategy.signals.copy()

    async def decide(self, symbol, when, portfolio):
        if not self.active_strategy:
            return []
        return await self.active_strategy.decide(symbol, when, portfolio)

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
            "name": self.NAME,
            "active_strategy": self.active_strategy_name,
            "current_regime": self.current_regime,
            "regime_switches": self.regime_switches,
            "last_switch": self.last_regime_switch.isoformat() if self.last_regime_switch else None,
            "symbols": len(self.symbols),
            "signals": {s: sig for s, sig in self.signals.items() if sig != "neutral"},
        }

    async def get_regime_info(self) -> Dict:
        """Get current regime information."""
        return await self.regime_detector.detect_regime()


# Factory function for easy creation
def create_adaptive_strategy(
    broker,
    symbols: List[str],
    **kwargs,
) -> AdaptiveStrategy:
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
        parameters=kwargs,
    )
