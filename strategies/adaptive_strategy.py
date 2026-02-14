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

import asyncio
import logging
import os
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.momentum_strategy import MomentumStrategy
from utils.market_regime import MarketRegimeDetector

logger = logging.getLogger(__name__)


# Lazy imports for optional quant features
SignalAggregator = None
PortfolioOptimizer = None
TailHedgeManager = None
EnsemblePredictor = None
CrossAssetAggregator = None


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

    def __init__(
        self,
        broker=None,
        symbols=None,
        parameters=None,
        enable_signal_aggregator: bool = True,
        enable_portfolio_optimizer: bool = True,
        enable_ml_signals: bool = True,
        enable_ensemble: bool = True,
        enable_cross_asset: bool = True,
        order_gateway=None,
    ):
        """
        Initialize adaptive strategy with symbols.

        Args:
            broker: Trading broker instance
            symbols: List of symbols to trade
            parameters: Strategy parameters
            enable_signal_aggregator: Use multi-source signal enrichment
            enable_portfolio_optimizer: Use portfolio optimization for sizing
            enable_ml_signals: Use LSTM predictions as supplementary signals
            enable_ensemble: Use ML ensemble for combined signals
            enable_cross_asset: Use cross-asset signals for regime detection
        """
        parameters = parameters or {}
        if symbols:
            parameters["symbols"] = symbols
        super().__init__(
            name=self.NAME,
            broker=broker,
            parameters=parameters,
            order_gateway=order_gateway,
        )

        # Quant feature flags
        self.enable_signal_aggregator = enable_signal_aggregator
        self.enable_portfolio_optimizer = enable_portfolio_optimizer
        self.enable_ml_signals = enable_ml_signals
        self.enable_ensemble = enable_ensemble
        self.enable_cross_asset = enable_cross_asset

        # Quant components (initialized in initialize())
        self.signal_aggregator = None
        self.portfolio_optimizer = None
        self.tail_hedge_manager = None
        self.lstm_predictor = None
        self.ensemble_predictor = None
        self.cross_asset_aggregator = None
        self.cached_portfolio_weights: Dict[str, float] = {}
        self._cross_asset_signal = None
        self._cross_asset_timestamp: Optional[datetime] = None

    def default_parameters(self):
        """Return default parameters for the adaptive strategy."""
        return {
            # Basic parameters
            "position_size": 0.10,
            "max_positions": 5,
            "max_portfolio_risk": 0.02,
            "stop_loss": 0.03,
            "take_profit": 0.05,
            # Regime detection settings
            "regime_check_interval_minutes": 30,  # How often to check regime
            "min_regime_confidence": 0.55,  # Minimum confidence to act on regime
            # Strategy selection
            "bull_strategy": "momentum",  # Strategy for bull regime
            "bear_strategy": "momentum_short",  # Strategy for bear regime
            "sideways_strategy": "mean_reversion",  # Strategy for sideways regime
            "volatile_strategy": "defensive",  # Strategy for high volatility
            # Position adjustments by regime
            "bull_position_mult": 1.2,  # 20% larger in bull
            "bear_position_mult": 0.8,  # 20% smaller in bear (shorts are riskier)
            "sideways_position_mult": 1.0,  # Normal in sideways
            "volatile_position_mult": 0.5,  # 50% smaller in volatile
            # Defensive thresholds
            "skip_trades_vix_threshold": 40,  # Skip new trades if VIX > 40
            "reduce_exposure_vix_threshold": 30,  # Reduce exposure if VIX > 30
            # Sub-strategy parameters (passed through)
            "use_kelly_criterion": True,
            "use_volatility_regime": True,
            "use_trailing_stop": True,
            "use_multi_timeframe": True,
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
                "use_kelly_criterion": self.parameters.get("use_kelly_criterion", True),
                "use_volatility_regime": self.parameters.get("use_volatility_regime", True),
                "use_trailing_stop": self.parameters.get("use_trailing_stop", True),
                "use_multi_timeframe": self.parameters.get("use_multi_timeframe", True),
                "enable_short_selling": True,  # Enable for bear markets
            }
            self.momentum_strategy = MomentumStrategy(
                broker=self.broker, parameters=momentum_params
            )

            # Mean reversion strategy for sideways markets
            mean_rev_params = {
                "symbols": self.symbols,
                "position_size": self.position_size,
                "max_positions": self.max_positions,
                "stop_loss": self.stop_loss,
                "take_profit": self.parameters.get("mean_reversion_take_profit", 0.04),
                "use_multi_timeframe": self.parameters.get("use_multi_timeframe", True),
                "enable_short_selling": True,
            }
            self.mean_reversion_strategy = MeanReversionStrategy(
                broker=self.broker, parameters=mean_rev_params
            )

            # Performance optimization: Initialize sub-strategies in parallel
            await asyncio.gather(
                self.momentum_strategy.initialize(),
                self.mean_reversion_strategy.initialize()
            )
            logger.info("  MomentumStrategy initialized for trending markets")
            logger.info("  MeanReversionStrategy initialized for sideways markets")

            # Active strategy pointer
            self.active_strategy = self.momentum_strategy  # Default
            self.active_strategy_name = "momentum"

            # Tracking
            self.indicators = {symbol: {} for symbol in self.symbols}
            self.signals = dict.fromkeys(self.symbols, "neutral")
            self.current_prices = {}
            # Performance optimization: Use deque with maxlen for O(1) append and auto-trimming
            # This avoids memory churn from list slicing
            self.price_history = {symbol: deque(maxlen=100) for symbol in self.symbols}
            self.regime_switches = 0
            self.last_regime_switch = None

            logger.info(f"AdaptiveStrategy initialized with {len(self.symbols)} symbols")
            logger.info(
                f"  Bull: {self.parameters['bull_strategy']}, "
                f"Bear: {self.parameters['bear_strategy']}, "
                f"Sideways: {self.parameters['sideways_strategy']}"
            )

            # Initialize quant features
            await self._initialize_quant_features()

            return True

        except Exception as e:
            logger.error(f"Error initializing AdaptiveStrategy: {e}", exc_info=True)
            return False

    async def _initialize_quant_features(self):
        """Initialize optional quant enhancement features."""
        global SignalAggregator, PortfolioOptimizer, CrossAssetAggregator

        # Initialize Cross-Asset Aggregator for VIX, yield curve, FX signals
        if self.enable_cross_asset:
            try:
                from data.cross_asset_provider import (
                    CrossAssetAggregator as CAA,
                )
                from data.cross_asset_provider import (
                    FxCorrelationProvider,
                    VixTermStructureProvider,
                    YieldCurveProvider,
                )

                CrossAssetAggregator = CAA

                self.cross_asset_aggregator = CrossAssetAggregator()
                self.cross_asset_aggregator.register_provider(VixTermStructureProvider())
                self.cross_asset_aggregator.register_provider(YieldCurveProvider())
                self.cross_asset_aggregator.register_provider(FxCorrelationProvider())
                await self.cross_asset_aggregator.initialize_all()
                logger.info("  CrossAssetAggregator initialized for VIX/yield curve/FX signals")
            except Exception as e:
                logger.warning(f"Could not initialize CrossAssetAggregator: {e}")
                self.cross_asset_aggregator = None

        # Initialize Signal Aggregator for multi-source signal enrichment
        if self.enable_signal_aggregator:
            try:
                from utils.signal_aggregator import SignalAggregator as SA

                SignalAggregator = SA

                api_key = os.getenv("ALPACA_API_KEY")
                secret_key = os.getenv("ALPACA_SECRET_KEY")

                self.signal_aggregator = SignalAggregator(
                    broker=self.broker,
                    api_key=api_key,
                    secret_key=secret_key,
                    enable_sentiment=True,
                    enable_ml=True,
                    min_agreement=0.5,
                )
                await self.signal_aggregator.initialize()
                logger.info("  SignalAggregator initialized for multi-source signal enrichment")
            except Exception as e:
                logger.warning(f"Could not initialize SignalAggregator: {e}")
                self.signal_aggregator = None

        # Initialize Portfolio Optimizer for position sizing
        if self.enable_portfolio_optimizer:
            try:
                from utils.portfolio_optimizer import PortfolioOptimizer as PO

                PortfolioOptimizer = PO

                self.portfolio_optimizer = PortfolioOptimizer(
                    broker=self.broker,
                    lookback_days=60,
                    risk_free_rate=0.05,
                )
                logger.info("  PortfolioOptimizer initialized for position sizing")
            except Exception as e:
                logger.warning(f"Could not initialize PortfolioOptimizer: {e}")
                self.portfolio_optimizer = None

        # Initialize Tail Hedge Manager for crash protection
        try:
            from utils.tail_hedge_manager import TailHedgeManager as THM
            from utils.volatility_regime import VolatilityRegimeDetector

            global TailHedgeManager
            TailHedgeManager = THM

            volatility_detector = VolatilityRegimeDetector(
                broker=self.broker,
                cache_minutes=5,
            )

            self.tail_hedge_manager = TailHedgeManager(
                broker=self.broker,
                volatility_detector=volatility_detector,
            )
            await self.tail_hedge_manager.initialize()
            logger.info("  TailHedgeManager initialized for crash protection")
        except Exception as e:
            logger.warning(f"Could not initialize TailHedgeManager: {e}")
            self.tail_hedge_manager = None

        # Initialize LSTM Predictor for ML-based signals
        if self.enable_ml_signals:
            try:
                from ml.lstm_predictor import LSTMPredictor

                self.lstm_predictor = LSTMPredictor(
                    sequence_length=60,
                    prediction_horizon=5,
                    hidden_size=64,
                    num_layers=2,
                )
                # Load pre-trained models if available
                models_loaded = 0
                for symbol in self.symbols:
                    if self.lstm_predictor.load_model(symbol):
                        models_loaded += 1
                if models_loaded > 0:
                    logger.info(f"  LSTMPredictor initialized with {models_loaded} pre-trained models")
                else:
                    logger.info("  LSTMPredictor initialized (no pre-trained models found)")
            except Exception as e:
                logger.warning(f"Could not initialize LSTMPredictor: {e}")
                self.lstm_predictor = None

        # Initialize Ensemble Predictor for combined ML signals
        if self.enable_ensemble:
            try:
                from ml.ensemble_predictor import (
                    EnsemblePredictor as EP,
                )
                from ml.ensemble_predictor import (
                    SignalComponent,
                    SignalSource,
                )

                global EnsemblePredictor
                EnsemblePredictor = EP

                self.ensemble_predictor = EnsemblePredictor(
                    min_sources_required=1,  # Allow single-source predictions
                    confidence_threshold=0.5,
                    use_performance_weighting=True,
                )

                # Register LSTM as signal source if available
                if self.lstm_predictor:
                    def lstm_signal_fn(symbol: str, data: Any) -> Optional[SignalComponent]:
                        prices = data.get("prices", [])
                        if len(prices) < 60:
                            return None

                        result = self.lstm_predictor.predict_with_uncertainty(symbol, prices)
                        if result is None:
                            return None

                        direction_map = {"up": "long", "down": "short", "neutral": "neutral"}
                        signal_value = 1.0 if result.predicted_direction == "up" else (
                            -1.0 if result.predicted_direction == "down" else 0.0
                        )

                        return SignalComponent(
                            source=SignalSource.LSTM,
                            signal_value=signal_value,
                            confidence=result.confidence,
                            direction=direction_map.get(result.predicted_direction, "neutral"),
                            metadata={"std": result.std_prediction},
                        )

                    self.ensemble_predictor.register_source(SignalSource.LSTM, lstm_signal_fn)
                    logger.info("  Registered LSTM as ensemble signal source")

                # Register momentum strategy signal
                def momentum_signal_fn(symbol: str, data: Any) -> Optional[SignalComponent]:
                    try:
                        # Use momentum strategy's analysis
                        signal = self.momentum_strategy.signals.get(symbol, "neutral")
                        confidence = 0.5

                        if signal == "neutral":
                            return SignalComponent(
                                source=SignalSource.MOMENTUM,
                                signal_value=0.0,
                                confidence=0.3,
                                direction="neutral",
                            )

                        signal_value = 1.0 if signal in ("buy", "strong_buy") else -1.0
                        direction = "long" if signal_value > 0 else "short"

                        return SignalComponent(
                            source=SignalSource.MOMENTUM,
                            signal_value=signal_value,
                            confidence=confidence,
                            direction=direction,
                        )
                    except Exception:
                        return None

                self.ensemble_predictor.register_source(SignalSource.MOMENTUM, momentum_signal_fn)
                logger.info("  Registered Momentum as ensemble signal source")

                # Register DQN agent as signal source
                try:
                    from ml.rl_agent import DQNAgent

                    self.dqn_agent = DQNAgent(
                        state_size=20,
                        action_size=3,
                        hidden_sizes=[128, 64],
                        epsilon_end=0.01,  # Low exploration for inference
                    )
                    # Try to load pre-trained model
                    if self.dqn_agent.load("dqn_trading_agent.pt"):
                        logger.info("  Loaded pre-trained DQN agent")
                    else:
                        logger.info("  DQN agent initialized (no pre-trained model)")
                    self.dqn_agent.set_inference_mode()

                    def dqn_signal_fn(symbol: str, data: Any) -> Optional[SignalComponent]:
                        try:
                            prices = data.get("prices", [])
                            if len(prices) < 10:
                                return None

                            # Build state from price data
                            state = self.dqn_agent.build_state(
                                prices=prices[-10:],
                                position=0.0,  # Neutral position assumption
                                account_equity=100000.0,  # Normalized
                            )

                            action, confidence = self.dqn_agent.act(state, return_confidence=True)

                            # Map DQN action to signal
                            if action == 0:  # HOLD
                                return SignalComponent(
                                    source=SignalSource.DQN,
                                    signal_value=0.0,
                                    confidence=confidence * 0.5,  # Lower confidence for hold
                                    direction="neutral",
                                )
                            elif action == 1:  # BUY
                                return SignalComponent(
                                    source=SignalSource.DQN,
                                    signal_value=1.0,
                                    confidence=confidence,
                                    direction="long",
                                )
                            else:  # SELL
                                return SignalComponent(
                                    source=SignalSource.DQN,
                                    signal_value=-1.0,
                                    confidence=confidence,
                                    direction="short",
                                )
                        except Exception as e:
                            logger.debug(f"DQN signal error for {symbol}: {e}")
                            return None

                    self.ensemble_predictor.register_source(SignalSource.DQN, dqn_signal_fn)
                    logger.info("  Registered DQN as ensemble signal source")

                except Exception as e:
                    logger.warning(f"Could not initialize DQN for ensemble: {e}")
                    self.dqn_agent = None

                # Register Factor Model as signal source
                try:
                    from strategies.factor_models import FactorModel, FactorType
                    from utils.factor_data import FactorDataProvider

                    self.factor_model = FactorModel(
                        factor_weights={
                            # Slightly overweight momentum for trading
                            FactorType.VALUE: 0.15,
                            FactorType.QUALITY: 0.20,
                            FactorType.MOMENTUM: 0.35,
                            FactorType.LOW_VOLATILITY: 0.20,
                            FactorType.SIZE: 0.10,
                        }
                    )
                    self.factor_data_provider = FactorDataProvider()
                    self._factor_scores_cache: Dict[str, Any] = {}
                    self._factor_scores_timestamp: Optional[datetime] = None

                    def factor_signal_fn(symbol: str, data: Any) -> Optional[SignalComponent]:
                        try:
                            # Use cached factor scores if available and recent
                            if (
                                self._factor_scores_cache
                                and self._factor_scores_timestamp
                                and (datetime.now() - self._factor_scores_timestamp).total_seconds() < 3600
                            ):
                                signal_dict = self.factor_model.get_signal(
                                    symbol, self._factor_scores_cache
                                )
                            else:
                                # Factor scores need price data for momentum calculation
                                # This would typically be pre-calculated during initialization
                                return None

                            if signal_dict["action"] == "hold":
                                return None

                            composite_z = signal_dict.get("composite_z", 0)
                            confidence = signal_dict.get("confidence", 0.5)

                            return SignalComponent(
                                source=SignalSource.FACTOR,
                                signal_value=min(1.0, max(-1.0, composite_z / 2)),
                                confidence=confidence,
                                direction=signal_dict["action"],
                                metadata=signal_dict.get("factor_breakdown"),
                            )
                        except Exception as e:
                            logger.debug(f"Factor signal error for {symbol}: {e}")
                            return None

                    self.ensemble_predictor.register_source(SignalSource.FACTOR, factor_signal_fn)
                    logger.info("  Registered Factor Model as ensemble signal source")

                except Exception as e:
                    logger.warning(f"Could not initialize Factor Model for ensemble: {e}")
                    self.factor_model = None
                    self.factor_data_provider = None

                # Register Cross-Asset signals if aggregator is available
                if self.cross_asset_aggregator:
                    try:
                        async def cross_asset_signal_fn_async(symbol: str, data: Any) -> Optional[SignalComponent]:
                            try:
                                signal = await self.cross_asset_aggregator.get_aggregated_signal()
                                if signal is None:
                                    return None

                                return SignalComponent(
                                    source=SignalSource.CROSS_ASSET,
                                    signal_value=signal.composite_signal,
                                    confidence=signal.composite_confidence,
                                    direction="long" if signal.composite_signal > 0 else (
                                        "short" if signal.composite_signal < 0 else "neutral"
                                    ),
                                    metadata={
                                        "sources": [s.value for s in signal.sources],
                                        "should_reduce_exposure": signal.should_reduce_exposure,
                                    },
                                )
                            except Exception as e:
                                logger.debug(f"Cross-asset signal error: {e}")
                                return None

                        # Wrap async function for synchronous ensemble interface
                        def cross_asset_signal_fn(symbol: str, data: Any) -> Optional[SignalComponent]:
                            try:
                                # Use cached signal if available and fresh
                                if (
                                    self._cross_asset_signal is not None
                                    and self._cross_asset_timestamp is not None
                                    and (datetime.now() - self._cross_asset_timestamp).total_seconds() < 900
                                ):
                                    signal = self._cross_asset_signal
                                    return SignalComponent(
                                        source=SignalSource.CROSS_ASSET,
                                        signal_value=signal.composite_signal,
                                        confidence=signal.composite_confidence,
                                        direction="long" if signal.composite_signal > 0 else (
                                            "short" if signal.composite_signal < 0 else "neutral"
                                        ),
                                        metadata={
                                            "sources": [s.value for s in signal.sources],
                                            "should_reduce_exposure": signal.should_reduce_exposure,
                                        },
                                    )
                                return None
                            except Exception as e:
                                logger.debug(f"Cross-asset signal error: {e}")
                                return None

                        self.ensemble_predictor.register_source(SignalSource.CROSS_ASSET, cross_asset_signal_fn)
                        logger.info("  Registered Cross-Asset as ensemble signal source")

                    except Exception as e:
                        logger.warning(f"Could not register Cross-Asset for ensemble: {e}")

                logger.info("  EnsemblePredictor initialized for combined ML signals")

            except Exception as e:
                logger.warning(f"Could not initialize EnsemblePredictor: {e}")
                self.ensemble_predictor = None

    async def update_portfolio_weights(self):
        """
        Update cached portfolio weights using optimization.

        Called daily (or on demand) to recalculate optimal weights.
        """
        if not self.portfolio_optimizer:
            return

        try:
            # Use risk parity for equal risk contribution
            result = await self.portfolio_optimizer.optimize_risk_parity(
                symbols=self.symbols,
                max_weight=0.25,  # No more than 25% in any position
            )

            if result and result.weights:
                self.cached_portfolio_weights = result.weights
                logger.info(
                    f"Portfolio weights updated: {', '.join(f'{k}={v:.1%}' for k, v in result.weights.items())}"
                )
                logger.info(f"  Expected return: {result.expected_return:.2%}, Risk: {result.expected_volatility:.2%}")
            else:
                logger.warning("Portfolio optimization returned no weights")

        except Exception as e:
            logger.error(f"Error updating portfolio weights: {e}")

    async def update_factor_scores(self):
        """
        Update cached factor scores for all symbols.

        This should be called daily (or on demand) to refresh factor data.
        Factor scores are used by the ensemble predictor for signal generation.
        """
        if not hasattr(self, "factor_model") or self.factor_model is None:
            return

        if not hasattr(self, "factor_data_provider") or self.factor_data_provider is None:
            return

        try:
            import pandas as pd

            # Build price DataFrame from cached history
            price_dict = {}
            for symbol in self.symbols:
                if symbol in self.price_history and len(self.price_history[symbol]) >= 252:
                    prices = [bar.get("close", bar.get("c", 0)) for bar in self.price_history[symbol]]
                    price_dict[symbol] = prices

            if not price_dict:
                logger.warning("Insufficient price history for factor calculation")
                return

            # Create price DataFrame
            max_len = max(len(p) for p in price_dict.values())
            for symbol in price_dict:
                # Pad shorter series
                if len(price_dict[symbol]) < max_len:
                    padding = [price_dict[symbol][0]] * (max_len - len(price_dict[symbol]))
                    price_dict[symbol] = padding + price_dict[symbol]

            price_df = pd.DataFrame(price_dict)

            # Get fundamental data
            factor_inputs = await self.factor_data_provider.build_factor_inputs(
                symbols=self.symbols,
                price_data=price_df,
            )

            # Provenance gate: never let synthetic fundamentals silently drive paper/live decisions.
            # We still compute price-only factors (momentum/low_vol) when fundamentals are gated.
            fundamental_data_for_scoring = factor_inputs.get("fundamental_data")
            market_caps_for_scoring = factor_inputs.get("market_caps")

            provenance = factor_inputs.get("data_provenance") or {}
            ratios = (provenance.get("ratios") or {}) if isinstance(provenance, dict) else {}

            synthetic_ratio = ratios.get("synthetic_ratio")
            real_ratio = ratios.get("real_ratio")
            coverage_ratio = ratios.get("coverage_ratio")

            try:
                from config import RISK_PARAMS

                max_synth = float(RISK_PARAMS.get("FACTOR_DATA_MAX_SYNTHETIC_RATIO", 0.0))
                min_real = float(RISK_PARAMS.get("FACTOR_DATA_MIN_REAL_COVERAGE_RATIO", 0.0))
                min_cov = float(RISK_PARAMS.get("FACTOR_DATA_MIN_COVERAGE_RATIO", 0.0))

                gate_fundamentals = False
                if synthetic_ratio is not None and synthetic_ratio > max_synth:
                    gate_fundamentals = True
                if real_ratio is not None and real_ratio < min_real:
                    gate_fundamentals = True
                if coverage_ratio is not None and coverage_ratio < min_cov:
                    gate_fundamentals = True

                if gate_fundamentals:
                    if ratios:
                        logger.warning(
                            "Gating fundamental/size factors due to data provenance "
                            f"(coverage={coverage_ratio:.0%}, real={real_ratio:.0%}, synthetic={synthetic_ratio:.0%}; "
                            f"thresholds: min_real={min_real:.0%}, max_synth={max_synth:.0%}, min_cov={min_cov:.0%}). "
                            "Proceeding with price-only factors."
                        )
                    fundamental_data_for_scoring = None
                    market_caps_for_scoring = None
                else:
                    # Prefer non-synthetic subsets when available.
                    fundamental_data_for_scoring = (
                        factor_inputs.get("fundamental_data_real")
                        or factor_inputs.get("fundamental_data")
                    )
                    market_caps_for_scoring = (
                        factor_inputs.get("market_caps_real")
                        or factor_inputs.get("market_caps")
                    )
            except Exception as e:
                logger.debug(f"Factor provenance gate unavailable: {e}")

            # Score the universe
            self._factor_scores_cache = self.factor_model.score_universe(
                symbols=self.symbols,
                price_data=price_df,
                fundamental_data=fundamental_data_for_scoring,
                market_caps=market_caps_for_scoring,
            )
            self._factor_scores_timestamp = datetime.now()

            used_factors = set()
            for score in self._factor_scores_cache.values():
                try:
                    used_factors.update(ft.value for ft in score.factor_scores.keys())
                except Exception:
                    continue

            logger.info(
                f"Factor scores updated for {len(self._factor_scores_cache)} symbols "
                f"(factors used: {', '.join(sorted(used_factors)) if used_factors else 'none'})"
            )

        except Exception as e:
            logger.error(f"Error updating factor scores: {e}")

    async def enrich_signal(
        self,
        symbol: str,
        technical_signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enrich a technical signal with multi-source confirmation.

        Takes the raw technical signal from the strategy and:
        1. Adds sentiment analysis confirmation
        2. Checks economic calendar
        3. Incorporates ML predictions
        4. Adjusts confidence based on agreement

        Args:
            symbol: Stock symbol
            technical_signal: Raw signal from strategy (action, confidence, reason)

        Returns:
            Enriched signal with adjusted confidence and metadata
        """
        if not self.signal_aggregator:
            return technical_signal

        try:
            # Get price history for ML
            price_history = None
            if symbol in self.price_history and len(self.price_history[symbol]) >= 60:
                price_history = [bar["close"] for bar in self.price_history[symbol]]

            # Get composite signal from aggregator
            composite = await self.signal_aggregator.get_composite_signal(
                symbol=symbol,
                technical_signal=technical_signal,
                price_history=price_history,
            )

            # Check if blocked
            if composite.blocked_by:
                logger.info(f"{symbol}: Signal BLOCKED by {', '.join(composite.blocked_by)}")
                return {
                    "action": "neutral",
                    "confidence": 0.0,
                    "reason": f"Blocked by: {', '.join(composite.blocked_by)}",
                    "blocked": True,
                }

            # Adjust confidence based on multi-source agreement
            original_confidence = technical_signal.get("confidence", 0.5)
            enriched_confidence = original_confidence

            # Boost if sources agree
            if composite.agreement_pct >= 0.6:
                enriched_confidence = min(1.0, original_confidence * 1.2)
                logger.debug(f"{symbol}: Confidence boosted (agreement: {composite.agreement_pct:.0%})")
            # Reduce if sources disagree
            elif composite.agreement_pct < 0.4:
                enriched_confidence = original_confidence * 0.8
                logger.debug(f"{symbol}: Confidence reduced (low agreement: {composite.agreement_pct:.0%})")

            # Apply position size multiplier from economic/regime
            position_multiplier = composite.position_size_multiplier

            # Get portfolio weight if available
            portfolio_weight = self.cached_portfolio_weights.get(symbol, 1.0)

            # Build enriched signal
            enriched_signal = {
                "action": technical_signal.get("action", "neutral"),
                "confidence": enriched_confidence,
                "original_confidence": original_confidence,
                "reason": technical_signal.get("reason", ""),
                "agreement_pct": composite.agreement_pct,
                "position_multiplier": position_multiplier * portfolio_weight,
                "contributing_sources": [s.source.value for s in composite.contributing_signals],
                "enriched": True,
            }

            # Log enrichment summary
            if enriched_signal["action"] != "neutral":
                logger.info(
                    f"{symbol}: Signal enriched - {enriched_signal['action'].upper()} "
                    f"(conf: {original_confidence:.0%} -> {enriched_confidence:.0%}, "
                    f"agreement: {composite.agreement_pct:.0%}, "
                    f"sources: {len(composite.contributing_signals)})"
                )

            return enriched_signal

        except Exception as e:
            logger.warning(f"Error enriching signal for {symbol}: {e}")
            return technical_signal

    async def on_bar(
        self, symbol, open_price, high_price, low_price, close_price, volume, timestamp
    ):
        """Handle incoming bar data with regime-aware routing."""
        try:
            if symbol not in self.symbols:
                return

            # Store current price
            self.current_prices[symbol] = close_price

            # Update price history (deque auto-trims to maxlen=100 via maxlen)
            # Performance optimization: O(1) append, no list slicing needed
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

            # Check and update market regime (cached, not every bar)
            await self._update_regime()

            # Route to appropriate sub-strategy based on regime
            if self.active_strategy:
                await self.active_strategy.on_bar(
                    symbol, open_price, high_price, low_price, close_price, volume, timestamp
                )

                # Copy signals and indicators from active strategy
                self.signals[symbol] = self.active_strategy.signals.get(symbol, "neutral")
                self.indicators[symbol] = self.active_strategy.indicators.get(symbol, {})

        except Exception as e:
            logger.error(f"Error in AdaptiveStrategy on_bar for {symbol}: {e}", exc_info=True)

    async def _update_regime(self):
        """Update market regime and switch strategies if needed."""
        try:
            # Detect current regime (uses internal caching)
            regime_info = await self.regime_detector.detect_regime()

            # Enhance regime detection with cross-asset signals
            if self.cross_asset_aggregator:
                regime_info = await self._enhance_regime_with_cross_asset(regime_info)

            # Check if regime changed
            new_regime_type = regime_info["type"]

            if self.current_regime is None or new_regime_type != self.current_regime:
                old_regime = self.current_regime or "none"
                self.current_regime = new_regime_type

                # Log regime change
                cross_asset_note = ""
                if regime_info.get("cross_asset_override"):
                    cross_asset_note = f" [cross-asset override: {regime_info.get('cross_asset_reason', '')}]"

                logger.warning(
                    f"REGIME CHANGE: {old_regime.upper()} -> {new_regime_type.upper()} "
                    f"(confidence: {regime_info['confidence']:.0%}, "
                    f"recommended: {regime_info['recommended_strategy']}){cross_asset_note}"
                )

                # Switch active strategy
                await self._switch_strategy(regime_info)

                self.regime_switches += 1
                self.last_regime_switch = datetime.now()

        except Exception as e:
            logger.error(f"Error updating regime: {e}", exc_info=True)

    async def _enhance_regime_with_cross_asset(self, regime_info: Dict) -> Dict:
        """
        Enhance regime detection using cross-asset signals.

        Uses VIX term structure, yield curve, and FX correlations to refine
        or override the technical-based regime detection.

        Args:
            regime_info: Base regime info from MarketRegimeDetector

        Returns:
            Enhanced regime info with cross-asset adjustments
        """
        try:
            # Get cross-asset signal (cache for 15 minutes)
            now = datetime.now()
            if (
                self._cross_asset_signal is None
                or self._cross_asset_timestamp is None
                or (now - self._cross_asset_timestamp).total_seconds() > 900
            ):
                self._cross_asset_signal = await self.cross_asset_aggregator.get_aggregated_signal()
                self._cross_asset_timestamp = now

            if self._cross_asset_signal is None:
                return regime_info

            signal = self._cross_asset_signal
            enhanced_info = regime_info.copy()
            original_type = regime_info["type"]

            # --- VIX Term Structure Override ---
            # Steep backwardation (fear) → Override to VOLATILE
            # Steep contango (complacency) → Boost BULL confidence
            vix_signal = signal.vix_signal
            if vix_signal:
                from data.cross_asset_types import VolatilityRegime

                # Crisis volatility or backwardation = force VOLATILE regime
                if vix_signal.volatility_regime == VolatilityRegime.CRISIS:
                    enhanced_info["type"] = "volatile"
                    enhanced_info["position_multiplier"] = 0.3  # 70% reduction
                    enhanced_info["cross_asset_override"] = True
                    enhanced_info["cross_asset_reason"] = f"VIX crisis ({vix_signal.vix_spot:.1f})"
                    logger.info(
                        f"Cross-asset: VIX CRISIS regime detected (VIX={vix_signal.vix_spot:.1f}), "
                        f"overriding to VOLATILE with 0.3x position multiplier"
                    )

                elif vix_signal.volatility_regime == VolatilityRegime.HIGH_VOL:
                    if original_type != "volatile":
                        enhanced_info["type"] = "volatile"
                        enhanced_info["position_multiplier"] = min(
                            regime_info.get("position_multiplier", 1.0), 0.5
                        )
                        enhanced_info["cross_asset_override"] = True
                        enhanced_info["cross_asset_reason"] = f"VIX high ({vix_signal.vix_spot:.1f})"
                        logger.info(
                            f"Cross-asset: High VIX detected ({vix_signal.vix_spot:.1f}), "
                            f"overriding to VOLATILE"
                        )

                elif vix_signal.is_backwardation and vix_signal.vix_spot > 20:
                    # Backwardation with elevated VIX = fear, be defensive
                    if original_type in ("bull", "sideways"):
                        enhanced_info["type"] = "volatile"
                        enhanced_info["position_multiplier"] = min(
                            regime_info.get("position_multiplier", 1.0), 0.6
                        )
                        enhanced_info["cross_asset_override"] = True
                        enhanced_info["cross_asset_reason"] = "VIX backwardation (fear)"
                        logger.info(
                            f"Cross-asset: VIX backwardation detected (slope={vix_signal.term_slope:.2%}), "
                            f"overriding {original_type} to VOLATILE"
                        )

                elif vix_signal.is_complacent and original_type == "bull":
                    # Very low VIX with steep contango = complacency, boost confidence
                    enhanced_info["confidence"] = min(0.95, regime_info["confidence"] * 1.15)
                    enhanced_info["position_multiplier"] = min(
                        1.3, regime_info.get("position_multiplier", 1.0) * 1.1
                    )
                    logger.debug(
                        "Cross-asset: VIX complacency detected, boosting BULL confidence"
                    )

            # --- Yield Curve Override ---
            # Inverted curve = reduce equity exposure (recession risk)
            yield_signal = signal.yield_curve_signal
            if yield_signal and not enhanced_info.get("cross_asset_override"):
                if yield_signal.is_recessionary:
                    # Strongly inverted curve = reduce exposure regardless of regime
                    current_mult = enhanced_info.get("position_multiplier", 1.0)
                    enhanced_info["position_multiplier"] = current_mult * 0.8  # 20% reduction
                    enhanced_info["yield_curve_warning"] = True
                    logger.info(
                        f"Cross-asset: Yield curve inverted (slope={yield_signal.curve_slope:.2%}), "
                        f"reducing exposure by 20% (recession prob: {yield_signal.recession_probability:.0%})"
                    )

                elif yield_signal.is_steepening and original_type == "sideways":
                    # Steepening curve can signal recovery - slightly more bullish
                    enhanced_info["confidence"] = min(0.9, regime_info["confidence"] * 1.05)
                    logger.debug("Cross-asset: Yield curve steepening, slight bullish tilt")

            # --- FX Correlation Override ---
            # Strong risk-off in FX = reduce exposure
            fx_signal = signal.fx_signal
            if fx_signal and not enhanced_info.get("cross_asset_override"):
                if fx_signal.is_risk_off and fx_signal.confidence > 0.7:
                    # Strong risk-off signal from FX
                    if original_type == "bull":
                        # Don't override bull, but reduce position size
                        current_mult = enhanced_info.get("position_multiplier", 1.0)
                        enhanced_info["position_multiplier"] = current_mult * 0.85
                        enhanced_info["fx_risk_off_warning"] = True
                        logger.info(
                            f"Cross-asset: FX risk-off signal detected "
                            f"(appetite={fx_signal.risk_appetite_score:.2f}), reducing exposure 15%"
                        )

                elif fx_signal.is_risk_on and fx_signal.confidence > 0.7 and original_type == "bull":
                    # Strong risk-on confirmation
                    enhanced_info["confidence"] = min(0.95, regime_info["confidence"] * 1.1)
                    logger.debug("Cross-asset: FX risk-on confirmed, boosting BULL confidence")

            # --- Overall Cross-Asset Agreement ---
            # If all cross-asset signals agree on direction, boost confidence
            if signal.agreement_ratio >= 0.9 and not enhanced_info.get("cross_asset_override"):
                direction = signal.direction
                if direction.value == "bullish" and original_type == "bull":
                    enhanced_info["confidence"] = min(0.95, regime_info["confidence"] * 1.15)
                    logger.info("Cross-asset: All signals bullish, high conviction BULL regime")
                elif direction.value == "bearish" and original_type == "bear":
                    enhanced_info["confidence"] = min(0.95, regime_info["confidence"] * 1.15)
                    logger.info("Cross-asset: All signals bearish, high conviction BEAR regime")

            # If cross-asset says reduce exposure strongly, honor it
            if signal.should_reduce_exposure and not enhanced_info.get("cross_asset_override"):
                current_mult = enhanced_info.get("position_multiplier", 1.0)
                enhanced_info["position_multiplier"] = min(current_mult, 0.6)
                enhanced_info["cross_asset_reduce_exposure"] = True
                logger.warning(
                    "Cross-asset: Multiple signals recommend reducing exposure, "
                    "capping position multiplier at 0.6x"
                )

            return enhanced_info

        except Exception as e:
            logger.warning(f"Error enhancing regime with cross-asset: {e}")
            return regime_info

    async def get_cross_asset_signal(self) -> Optional[Dict]:
        """
        Get current cross-asset signal for external monitoring.

        Returns:
            Dict with VIX, yield curve, and FX signal details
        """
        if not self.cross_asset_aggregator:
            return None

        try:
            signal = await self.cross_asset_aggregator.get_aggregated_signal()
            if signal is None:
                return None

            result = {
                "composite_signal": signal.composite_signal,
                "composite_confidence": signal.composite_confidence,
                "direction": signal.direction.value if signal.direction else "neutral",
                "agreement_ratio": signal.agreement_ratio,
                "should_reduce_exposure": signal.should_reduce_exposure,
                "sources": [s.value for s in signal.sources],
            }

            # Add VIX details
            if signal.vix_signal:
                result["vix"] = {
                    "spot": signal.vix_signal.vix_spot,
                    "term_slope": signal.vix_signal.term_slope,
                    "regime": signal.vix_signal.volatility_regime.value,
                    "is_contango": signal.vix_signal.is_contango,
                    "is_fear_elevated": signal.vix_signal.is_fear_elevated,
                }

            # Add yield curve details
            if signal.yield_curve_signal:
                result["yield_curve"] = {
                    "slope": signal.yield_curve_signal.curve_slope,
                    "regime": signal.yield_curve_signal.yield_curve_regime.value,
                    "is_inverted": signal.yield_curve_signal.is_inverted,
                    "recession_probability": signal.yield_curve_signal.recession_probability,
                }

            # Add FX details
            if signal.fx_signal:
                result["fx"] = {
                    "risk_appetite": signal.fx_signal.risk_appetite_score,
                    "regime": signal.fx_signal.risk_appetite_regime.value,
                    "is_risk_on": signal.fx_signal.is_risk_on,
                    "is_risk_off": signal.fx_signal.is_risk_off,
                }

            return result

        except Exception as e:
            logger.warning(f"Error getting cross-asset signal: {e}")
            return None

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

    async def _get_ml_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get LSTM prediction as a supplementary ML signal.

        The ML signal provides an overlay on technical analysis when
        the model has high confidence (>60%).

        Args:
            symbol: Stock symbol to predict

        Returns:
            Dict with direction and confidence, or None if unavailable
        """
        if self.lstm_predictor is None:
            return None

        # Check if we have a model for this symbol
        if not self.lstm_predictor.has_model(symbol):
            return None

        # Check if we have enough price history
        if symbol not in self.price_history or len(self.price_history[symbol]) < 60:
            return None

        try:
            # Prepare price data for LSTM
            price_data = list(self.price_history[symbol])

            # Get prediction
            prediction = self.lstm_predictor.predict(symbol, price_data)

            if prediction is None:
                return None

            # Only use if confidence > 60%
            if prediction.confidence < 0.6:
                logger.debug(
                    f"{symbol}: ML signal confidence too low ({prediction.confidence:.0%}), skipping"
                )
                return None

            # Convert prediction to signal
            direction_map = {"up": 1.0, "down": -1.0, "neutral": 0.0}
            direction = direction_map.get(prediction.predicted_direction, 0.0)

            logger.info(
                f"{symbol}: ML signal - {prediction.predicted_direction.upper()} "
                f"(conf: {prediction.confidence:.0%}, predicted: ${prediction.predicted_price:.2f}, "
                f"current: ${prediction.current_price:.2f}, change: {prediction.price_change_pct:.1f}%)"
            )

            return {
                "direction": direction,
                "confidence": prediction.confidence,
                "predicted_price": prediction.predicted_price,
                "price_change_pct": prediction.price_change_pct,
            }

        except Exception as e:
            logger.warning(f"Error getting ML signal for {symbol}: {e}")
            return None

    async def _get_ensemble_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get ensemble prediction combining multiple ML signal sources.

        The ensemble combines:
        - LSTM predictions with MC Dropout uncertainty
        - Momentum strategy signals
        - Factor model signals (if configured)

        This is the institutional-grade approach to signal generation.

        Args:
            symbol: Stock symbol to predict

        Returns:
            Dict with ensemble signal details or None if unavailable
        """
        if self.ensemble_predictor is None:
            return None

        # Check if we have enough price history
        if symbol not in self.price_history or len(self.price_history[symbol]) < 60:
            return None

        try:
            # Prepare data for ensemble
            price_data = list(self.price_history[symbol])

            # Map current regime to ensemble regime
            regime = None
            if self.current_regime:
                from ml.ensemble_predictor import MarketRegime
                regime_map = {
                    "bull": MarketRegime.BULL,
                    "bear": MarketRegime.BEAR,
                    "sideways": MarketRegime.SIDEWAYS,
                    "volatile": MarketRegime.VOLATILE,
                }
                regime = regime_map.get(self.current_regime)

            # Get ensemble prediction
            prediction = self.ensemble_predictor.predict(
                symbol=symbol,
                data={"prices": price_data},
                regime=regime,
            )

            if prediction is None:
                return None

            # Check if signal is strong enough to trade
            if not prediction.should_trade(min_confidence=0.5):
                logger.debug(
                    f"{symbol}: Ensemble signal not strong enough "
                    f"(signal={prediction.ensemble_signal:.2f}, conf={prediction.ensemble_confidence:.0%})"
                )
                return None

            logger.info(
                f"{symbol}: ENSEMBLE signal - {prediction.direction.upper()} "
                f"(signal: {prediction.ensemble_signal:.2f}, conf: {prediction.ensemble_confidence:.0%}, "
                f"sources: {len(prediction.components)}, regime: {prediction.regime})"
            )

            return {
                "direction": prediction.ensemble_signal,
                "confidence": prediction.ensemble_confidence,
                "action": prediction.direction,
                "source_count": len(prediction.components),
                "weights_used": {k.value: v for k, v in prediction.weights_used.items()},
                "position_multiplier": prediction.get_position_size_multiplier(),
            }

        except Exception as e:
            logger.warning(f"Error getting ensemble signal for {symbol}: {e}")
            return None

    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze symbol using the active strategy with ML overlay and signal enrichment.

        The signal is generated in layers:
        1. Technical analysis from active strategy (base signal)
        2. ML overlay from LSTM predictions (20% weight when confident)
        3. Multi-source enrichment from signal aggregator

        Returns:
            Dict with action, confidence, and enrichment metadata
        """
        if not self.active_strategy:
            return {"action": "neutral", "confidence": 0.0}

        # Get technical signal from active strategy
        technical_result = await self.active_strategy.analyze_symbol(symbol)

        # Convert to dict if it's a string
        if isinstance(technical_result, str):
            technical_signal = {
                "action": technical_result,
                "confidence": 0.5 if technical_result != "neutral" else 0.0,
                "reason": f"Technical analysis ({self.active_strategy_name})",
            }
        elif isinstance(technical_result, dict):
            technical_signal = technical_result
        else:
            technical_signal = {"action": "neutral", "confidence": 0.0}

        # Apply ML overlay if enabled and available
        # Try ensemble first (institutional-grade), fall back to single LSTM
        ml_signal = None
        if self.enable_ml_signals:
            if self.enable_ensemble and self.ensemble_predictor:
                ml_signal = await self._get_ensemble_signal(symbol)
            if ml_signal is None:
                ml_signal = await self._get_ml_signal(symbol)

        if ml_signal is not None:
            # Blend technical and ML signals (80% technical + 20% ML)
            technical_direction = self._action_to_direction(technical_signal.get("action", "neutral"))
            ml_direction = ml_signal["direction"]

            # Calculate blended direction
            blended_direction = 0.8 * technical_direction + 0.2 * ml_direction

            # Convert back to action
            if blended_direction > 0.3:
                blended_action = "buy"
            elif blended_direction < -0.3:
                blended_action = "sell"
            else:
                blended_action = "neutral"

            # Adjust confidence based on agreement
            original_confidence = technical_signal.get("confidence", 0.5)
            if (technical_direction > 0 and ml_direction > 0) or (technical_direction < 0 and ml_direction < 0):
                # Signals agree - boost confidence
                blended_confidence = min(1.0, original_confidence * 1.15)
                agreement = "confirming"
            elif technical_direction != 0 and ml_direction != 0 and technical_direction * ml_direction < 0:
                # Signals disagree - reduce confidence
                blended_confidence = original_confidence * 0.85
                agreement = "conflicting"
            else:
                # One is neutral - keep original
                blended_confidence = original_confidence
                agreement = "partial"

            technical_signal = {
                **technical_signal,
                "action": blended_action,
                "confidence": blended_confidence,
                "original_action": technical_signal.get("action", "neutral"),
                "original_confidence": original_confidence,
                "ml_overlay": True,
                "ml_direction": ml_signal["direction"],
                "ml_confidence": ml_signal["confidence"],
                "ml_agreement": agreement,
                "reason": f"{technical_signal.get('reason', 'Technical')} + ML overlay ({agreement})",
            }

            logger.debug(
                f"{symbol}: Blended signal - {blended_action} "
                f"(tech: {technical_direction:.2f}, ml: {ml_direction:.2f}, "
                f"blended: {blended_direction:.2f}, {agreement})"
            )

        # Enrich with multi-source signals if aggregator is available
        if self.signal_aggregator:
            enriched_signal = await self.enrich_signal(symbol, technical_signal)
            return enriched_signal

        return technical_signal

    def _action_to_direction(self, action: str) -> float:
        """Convert action string to directional value."""
        direction_map = {
            "buy": 1.0,
            "strong_buy": 1.0,
            "long": 1.0,
            "sell": -1.0,
            "strong_sell": -1.0,
            "short": -1.0,
            "neutral": 0.0,
            "hold": 0.0,
        }
        return direction_map.get(action.lower(), 0.0)

    async def execute_trade(self, symbol: str, signal):
        """
        Execute trade using the active strategy with enriched sizing.

        Args:
            symbol: Stock symbol
            signal: Signal dict with action, confidence, position_multiplier
        """
        if not self.active_strategy:
            return

        # Extract action from signal
        if isinstance(signal, dict):
            action = signal.get("action", "neutral")
            position_multiplier = signal.get("position_multiplier", 1.0)

            # Check if blocked
            if signal.get("blocked"):
                logger.info(f"{symbol}: Trade blocked by signal enrichment")
                return

            # Apply position multiplier to strategy
            original_size = self.active_strategy.position_size
            self.active_strategy.position_size *= position_multiplier

            try:
                await self.active_strategy.execute_trade(symbol, action)
            finally:
                # Restore original position size
                self.active_strategy.position_size = original_size
        else:
            # Legacy string signal
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

    async def manage_hedges(self) -> Dict:
        """
        Manage tail hedge positions based on volatility regime.

        Should be called periodically (e.g., at start of each trading cycle)
        to establish or close protective put positions.

        Returns:
            Dict with hedge management status
        """
        if self.tail_hedge_manager is None:
            return {"status": "disabled", "action": None}

        try:
            result = await self.tail_hedge_manager.manage_hedge()
            return result
        except Exception as e:
            logger.warning(f"Error managing hedges: {e}")
            return {"status": "error", "error": str(e)}

    def get_hedge_status(self) -> Dict:
        """Get current tail hedge status."""
        if self.tail_hedge_manager is None:
            return {"enabled": False}
        return self.tail_hedge_manager.get_hedge_status()

    def get_status(self) -> Dict:
        """Get current status of the adaptive strategy."""
        status = {
            "name": self.NAME,
            "active_strategy": self.active_strategy_name,
            "current_regime": self.current_regime,
            "regime_switches": self.regime_switches,
            "last_switch": self.last_regime_switch.isoformat() if self.last_regime_switch else None,
            "symbols": len(self.symbols),
            "signals": {s: sig for s, sig in self.signals.items() if sig != "neutral"},
            "cross_asset_enabled": self.cross_asset_aggregator is not None,
        }

        # Add cached cross-asset signal summary if available
        if self._cross_asset_signal:
            status["cross_asset_summary"] = {
                "direction": self._cross_asset_signal.direction.value if self._cross_asset_signal.direction else "neutral",
                "composite_signal": round(self._cross_asset_signal.composite_signal, 2),
                "should_reduce_exposure": self._cross_asset_signal.should_reduce_exposure,
            }

        return status

    async def get_regime_info(self) -> Dict:
        """Get current regime information."""
        return await self.regime_detector.detect_regime()


# Factory function for easy creation
def create_adaptive_strategy(
    broker,
    symbols: List[str],
    enable_cross_asset: bool = True,
    **kwargs,
) -> AdaptiveStrategy:
    """
    Create and return an AdaptiveStrategy instance.

    Args:
        broker: Trading broker instance
        symbols: List of symbols to trade
        enable_cross_asset: Enable cross-asset signals for regime enhancement
        **kwargs: Additional parameters to pass to the strategy

    Returns:
        Initialized AdaptiveStrategy
    """
    return AdaptiveStrategy(
        broker=broker,
        symbols=symbols,
        parameters=kwargs,
        enable_cross_asset=enable_cross_asset,
    )
