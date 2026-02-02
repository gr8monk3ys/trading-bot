"""
Ensemble Voting Strategy

Combines multiple sub-strategies with calibrated voting:
- Weighted voting based on recent performance
- Regime-adjusted weights for different market conditions
- Only trades when agreement exceeds threshold (60%+ default)
- Automatic weight rebalancing based on rolling performance

This replaces the experimental ensemble with production-ready implementation.

Usage:
    from strategies.ensemble_voting_strategy import EnsembleVotingStrategy
    from strategies.momentum_strategy import MomentumStrategy
    from strategies.mean_reversion_strategy import MeanReversionStrategy

    ensemble = EnsembleVotingStrategy(
        broker=broker,
        sub_strategies=[MomentumStrategy, MeanReversionStrategy],
        min_agreement=0.6,
    )
    await ensemble.initialize()
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from strategies.base_strategy import BaseStrategy
from utils.market_regime import MarketRegimeDetector
from utils.strategy_performance_tracker import StrategyPerformanceTracker

logger = logging.getLogger(__name__)


class EnsembleVotingStrategy(BaseStrategy):
    """
    Ensemble strategy using weighted voting from multiple sub-strategies.

    Key features:
    - Adaptive weights based on rolling performance
    - Regime-aware weight adjustment
    - Minimum agreement threshold before trading
    - Tracks and records outcomes for continuous improvement
    """

    NAME = "EnsembleVotingStrategy"

    # Default weights if no performance history
    DEFAULT_WEIGHTS = {
        "MomentumStrategy": 0.35,
        "MeanReversionStrategy": 0.25,
        "AdaptiveStrategy": 0.20,
        "SimpleMACrossover": 0.10,
        "RSIDivergenceStrategy": 0.10,
    }

    # Regime-specific weight adjustments
    REGIME_ADJUSTMENTS = {
        "bull": {
            "MomentumStrategy": 1.3,
            "MeanReversionStrategy": 0.7,
            "AdaptiveStrategy": 1.0,
        },
        "bear": {
            "MomentumStrategy": 0.8,
            "MeanReversionStrategy": 0.9,
            "AdaptiveStrategy": 1.2,
        },
        "sideways": {
            "MomentumStrategy": 0.6,
            "MeanReversionStrategy": 1.4,
            "AdaptiveStrategy": 1.0,
        },
        "volatile": {
            "MomentumStrategy": 0.5,
            "MeanReversionStrategy": 0.7,
            "AdaptiveStrategy": 1.3,
        },
    }

    def __init__(
        self,
        name=None,
        broker=None,
        parameters=None,
        sub_strategies: Optional[List[Type[BaseStrategy]]] = None,
        min_agreement: float = 0.6,
        use_adaptive_weights: bool = True,
    ):
        """
        Initialize ensemble voting strategy.

        Args:
            name: Strategy name
            broker: Trading broker instance
            parameters: Strategy parameters
            sub_strategies: List of strategy classes to use
            min_agreement: Minimum agreement ratio to trade (0-1)
            use_adaptive_weights: Enable adaptive weight adjustment
        """
        super().__init__(name=name or self.NAME, broker=broker, parameters=parameters or {})

        self.sub_strategy_classes = sub_strategies or []
        self.sub_strategies: List[BaseStrategy] = []
        self.min_agreement = min_agreement
        self.use_adaptive_weights = use_adaptive_weights

        # Initialize performance tracker
        self.performance_tracker = StrategyPerformanceTracker(
            lookback_trades=100,
            min_trades_for_adjustment=20,
            decay_half_life_days=30,
        )

        # Regime detector
        self.regime_detector = None

        # Current weights
        self.current_weights: Dict[str, float] = {}

        # Track pending signals for outcome recording
        self._pending_signals: Dict[str, Dict[str, Any]] = {}

    async def initialize(self, **kwargs):
        """Initialize ensemble and all sub-strategies."""
        await super().initialize(**kwargs)

        # Initialize sub-strategies
        for strategy_class in self.sub_strategy_classes:
            try:
                strategy = strategy_class(
                    broker=self.broker,
                    parameters=self.parameters,
                )
                await strategy.initialize(**kwargs)
                self.sub_strategies.append(strategy)
                logger.info(f"Initialized sub-strategy: {strategy.name}")
            except Exception as e:
                logger.error(f"Failed to initialize {strategy_class.__name__}: {e}")

        if not self.sub_strategies:
            logger.warning("No sub-strategies initialized - ensemble will not function")
            return False

        # Initialize regime detector
        if self.broker:
            self.regime_detector = MarketRegimeDetector(self.broker)

        # Set initial weights
        self._update_weights()

        logger.info(
            f"EnsembleVotingStrategy initialized with {len(self.sub_strategies)} strategies: "
            f"{', '.join(s.name for s in self.sub_strategies)}"
        )

        return True

    def _update_weights(self, regime: Optional[str] = None):
        """Update strategy weights based on performance and regime."""
        strategy_names = [s.name for s in self.sub_strategies]

        if self.use_adaptive_weights:
            # Get performance-based weights
            if regime:
                weights = self.performance_tracker.get_regime_adjusted_weights(
                    strategy_names, regime
                )
            else:
                weights = self.performance_tracker.get_adaptive_weights(strategy_names)
        else:
            # Use default weights
            weights = {}
            total = 0
            for name in strategy_names:
                w = self.DEFAULT_WEIGHTS.get(name, 1.0)
                weights[name] = w
                total += w
            weights = {k: v / total for k, v in weights.items()}

        # Apply regime adjustments if available
        if regime and regime in self.REGIME_ADJUSTMENTS:
            adjustments = self.REGIME_ADJUSTMENTS[regime]
            for name in weights:
                if name in adjustments:
                    weights[name] *= adjustments[name]

            # Renormalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        self.current_weights = weights

        logger.debug(
            f"Updated weights ({regime or 'default'}): "
            + ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
        )

    async def analyze_symbol(self, symbol: str) -> dict:
        """
        Analyze symbol using ensemble voting.

        Collects signals from all sub-strategies, applies weighted voting,
        and returns consensus signal if agreement threshold is met.
        """
        if not self.sub_strategies:
            return {"action": "neutral", "reason": "no_sub_strategies"}

        # Detect current regime
        regime = None
        if self.regime_detector:
            try:
                regime_result = await self.regime_detector.detect_regime()
                if isinstance(regime_result, tuple):
                    regime = regime_result[0].value if hasattr(regime_result[0], 'value') else str(regime_result[0])
                else:
                    regime = str(regime_result)
            except Exception as e:
                logger.debug(f"Regime detection failed: {e}")

        # Update weights for current regime
        self._update_weights(regime)

        # Collect signals from all sub-strategies
        signals = await self._collect_signals(symbol)

        if not signals:
            return {"action": "neutral", "reason": "no_signals_received"}

        # Calculate weighted vote
        vote_result = self._calculate_weighted_vote(signals)

        # Check if agreement meets threshold
        if vote_result["agreement"] < self.min_agreement:
            logger.debug(
                f"{symbol}: Agreement {vote_result['agreement']:.0%} < "
                f"threshold {self.min_agreement:.0%}, skipping"
            )
            return {
                "action": "neutral",
                "reason": "insufficient_agreement",
                "agreement": vote_result["agreement"],
                "votes": vote_result["votes"],
            }

        # Store pending signal for outcome tracking
        self._pending_signals[symbol] = {
            "timestamp": datetime.now(),
            "signals": signals,
            "vote_result": vote_result,
            "regime": regime,
        }

        # Build result
        action = vote_result["consensus"]
        confidence = vote_result["agreement"] * vote_result["avg_confidence"]

        result = {
            "action": action,
            "confidence": confidence,
            "agreement": vote_result["agreement"],
            "regime": regime,
            "votes": vote_result["votes"],
            "contributing_strategies": vote_result["contributing"],
        }

        logger.info(
            f"{symbol}: Ensemble signal = {action.upper()} "
            f"(agreement: {vote_result['agreement']:.0%}, "
            f"confidence: {confidence:.0%}, regime: {regime})"
        )

        return result

    async def _collect_signals(
        self, symbol: str
    ) -> Dict[str, Dict[str, Any]]:
        """Collect signals from all sub-strategies concurrently."""
        signals = {}

        async def get_signal(strategy):
            try:
                signal = await asyncio.wait_for(
                    strategy.analyze_symbol(symbol),
                    timeout=10.0,
                )
                return strategy.name, signal
            except asyncio.TimeoutError:
                logger.warning(f"{strategy.name} timed out analyzing {symbol}")
                return strategy.name, None
            except Exception as e:
                logger.error(f"{strategy.name} error for {symbol}: {e}")
                return strategy.name, None

        # Collect signals concurrently
        tasks = [get_signal(s) for s in self.sub_strategies]
        results = await asyncio.gather(*tasks)

        for name, signal in results:
            if signal and signal.get("action"):
                signals[name] = signal

        return signals

    def _calculate_weighted_vote(
        self, signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate weighted vote from collected signals.

        Returns:
            Dict with consensus, agreement, votes breakdown
        """
        votes = {"buy": 0.0, "sell": 0.0, "neutral": 0.0}
        confidences = []
        contributing = []

        for strategy_name, signal in signals.items():
            action = signal.get("action", "neutral")
            confidence = signal.get("confidence", 0.5)
            weight = self.current_weights.get(strategy_name, 0.1)

            if action in votes:
                votes[action] += weight
                confidences.append(confidence)
                contributing.append(strategy_name)

        # Normalize votes
        total = sum(votes.values())
        if total > 0:
            votes = {k: v / total for k, v in votes.items()}

        # Determine consensus
        consensus = max(votes, key=votes.get)
        agreement = votes[consensus]

        return {
            "consensus": consensus,
            "agreement": agreement,
            "votes": votes,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.5,
            "contributing": contributing,
        }

    async def execute_trade(self, symbol: str, signal: dict):
        """Execute trade based on ensemble signal."""
        if signal.get("action") == "neutral":
            return None

        # Check trading allowed
        if not await self.check_trading_allowed():
            logger.warning(f"Trading halted by circuit breaker for {symbol}")
            return None

        # Check sentiment filter if enabled
        if hasattr(self, 'sentiment_analyzer') and self.sentiment_analyzer:
            direction = "long" if signal["action"] == "buy" else "short"
            if not await self.check_sentiment_filter(symbol, direction):
                return None

        try:
            from brokers.order_builder import OrderBuilder

            # Get current price
            bars = await self.broker.get_bars(symbol, timeframe="1Min", limit=1)
            if not bars:
                return None
            current_price = float(bars[-1].close)

            # Calculate position size
            position_value, position_fraction, base_qty = await self.calculate_kelly_position_size(
                symbol, current_price
            )

            if base_qty <= 0:
                return None

            # Apply volatility adjustments
            adj_size, adj_stop, regime = await self.apply_volatility_adjustments(
                position_fraction, self.stop_loss_pct
            )

            # Apply sentiment adjustment if available
            if hasattr(self, 'get_sentiment_adjusted_size'):
                adj_size = await self.get_sentiment_adjusted_size(symbol, adj_size)

            # Recalculate quantity with adjusted size
            account = await self.broker.get_account()
            account_value = float(account.equity)
            final_qty = int((account_value * adj_size) / current_price)

            if final_qty <= 0:
                return None

            # Build order
            side = "buy" if signal["action"] == "buy" else "sell"

            # Calculate stop and target
            if side == "buy":
                stop_price = current_price * (1 - adj_stop)
                target_price = current_price * (1 + self.take_profit_pct)
            else:
                stop_price = current_price * (1 + adj_stop)
                target_price = current_price * (1 - self.take_profit_pct)

            # Create bracket order for safety
            order = (
                OrderBuilder(symbol, side, final_qty)
                .market()
                .bracket(take_profit=target_price, stop_loss=stop_price)
                .day()
                .build()
            )

            result = await self.broker.submit_order_advanced(order)

            if result:
                self.track_position_entry(symbol, current_price)
                logger.info(
                    f"ENSEMBLE TRADE: {side.upper()} {final_qty} {symbol} @ ${current_price:.2f} "
                    f"(agreement: {signal['agreement']:.0%})"
                )

            return result

        except Exception as e:
            logger.error(f"Error executing ensemble trade for {symbol}: {e}")
            return None

    def record_trade_outcome(
        self,
        symbol: str,
        actual_direction: str,
        pnl: float,
    ):
        """
        Record trade outcome for performance tracking.

        Call this when a trade is closed to update strategy performance metrics.
        """
        if symbol not in self._pending_signals:
            return

        pending = self._pending_signals.pop(symbol)
        signals = pending["signals"]
        regime = pending["regime"]

        for strategy_name, signal in signals.items():
            predicted = signal.get("action", "neutral")
            confidence = signal.get("confidence", 0.5)

            self.performance_tracker.record_signal_outcome(
                strategy_name=strategy_name,
                symbol=symbol,
                predicted=predicted,
                actual=actual_direction,
                pnl=pnl,
                confidence=confidence,
                regime=regime,
            )

    async def on_trading_iteration(self):
        """Main trading loop iteration."""
        if not await self.check_trading_allowed():
            logger.debug("Trading halted, skipping iteration")
            return

        for symbol in self.symbols:
            try:
                signal = await self.analyze_symbol(symbol)

                if signal.get("action") in ["buy", "sell"]:
                    await self.execute_trade(symbol, signal)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current ensemble status and performance metrics."""
        performance_report = self.performance_tracker.get_performance_report()

        return {
            "name": self.NAME,
            "num_sub_strategies": len(self.sub_strategies),
            "sub_strategies": [s.name for s in self.sub_strategies],
            "current_weights": self.current_weights,
            "min_agreement": self.min_agreement,
            "use_adaptive_weights": self.use_adaptive_weights,
            "performance": performance_report,
        }

    async def cleanup(self):
        """Cleanup ensemble and all sub-strategies."""
        for strategy in self.sub_strategies:
            try:
                await strategy.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {strategy.name}: {e}")

        await super().cleanup()
