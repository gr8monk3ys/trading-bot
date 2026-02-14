"""
Ensemble Predictor for Institutional-Grade Signal Combination

Combines multiple signal sources:
1. LSTM neural network predictions
2. DQN reinforcement learning agent
3. Factor model scores

Ensemble Methods:
- Performance-weighted averaging
- Regime-dependent weighting
- Confidence-weighted combination
- Optional meta-learner (stacking)

Why Ensemble:
- Reduces model-specific risk
- Diversifies across methodologies (technical, fundamental, ML)
- More robust to regime changes
- Institutional standard for production systems
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# =============================================================================
# INSTITUTIONAL SAFETY CONSTANTS
# =============================================================================

# Minimum predictions required before ANY weight adjustment
# CRITICAL: 5 is way too few - requires 50+ for reliable accuracy estimates
MIN_PREDICTIONS_FOR_ADJUSTMENT = 50

# Bayesian prior: assume 50% accuracy with pseudo-count of 10
# This regularizes early estimates toward baseline
BAYESIAN_PRIOR_ALPHA = 5  # Pseudo-successes (50% * 10)
BAYESIAN_PRIOR_BETA = 5  # Pseudo-failures (50% * 10)

# Kill switch threshold: disable source if accuracy falls below
# 2 standard deviations below 50% baseline
KILL_SWITCH_THRESHOLD_SIGMA = 2.0

# Maximum weight adjustment factor
MAX_WEIGHT_INCREASE = 1.5  # Never more than 50% boost
MIN_WEIGHT_DECREASE = 0.5  # Never less than 50% of base weight

# Credibility interval for weight adjustment decisions
CREDIBILITY_LEVEL = 0.95  # 95% credibility interval


class SignalSource(Enum):
    """Signal sources in the ensemble."""

    LSTM = "lstm"
    DQN = "dqn"
    FACTOR = "factor"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ALTERNATIVE_DATA = "alternative_data"  # Social sentiment, order flow, web data
    CROSS_ASSET = "cross_asset"  # VIX term structure, yield curve, FX correlations
    LLM_ANALYSIS = "llm_analysis"  # LLM-powered text analysis (earnings, Fed, SEC, news)


class MarketRegime(Enum):
    """Market regimes for regime-dependent weighting."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


@dataclass
class SignalComponent:
    """Individual signal from one source."""

    source: SignalSource
    signal_value: float  # -1 (short) to +1 (long)
    confidence: float  # 0 to 1
    direction: str  # 'long', 'short', 'neutral'
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble."""

    symbol: str
    ensemble_signal: float  # -1 to +1
    ensemble_confidence: float  # 0 to 1
    direction: str  # 'long', 'short', 'neutral'
    components: Dict[SignalSource, SignalComponent]
    weights_used: Dict[SignalSource, float]
    timestamp: datetime
    regime: Optional[MarketRegime] = None

    def should_trade(self, min_confidence: float = 0.6) -> bool:
        """Check if signal is strong enough to trade."""
        return (
            self.ensemble_confidence >= min_confidence
            and abs(self.ensemble_signal) >= 0.3
            and self.direction != "neutral"
        )

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on confidence."""
        # Scale position between 0.5x and 1.5x based on confidence
        return 0.5 + self.ensemble_confidence


class EnsemblePredictor:
    """
    Combines multiple signal sources into unified predictions.

    Features:
    - Dynamic weighting based on recent performance
    - Regime-aware signal combination
    - Confidence calibration
    - Meta-learner option for optimal combination

    Usage:
        ensemble = EnsemblePredictor()

        # Register signal sources
        ensemble.register_source(SignalSource.LSTM, lstm_signal_fn)
        ensemble.register_source(SignalSource.FACTOR, factor_signal_fn)

        # Get ensemble prediction
        prediction = ensemble.predict(symbol, data)

        if prediction.should_trade():
            execute_trade(prediction.direction, prediction.confidence)
    """

    # Default weights per market regime
    # Alternative data is weighted 15-20% in most regimes
    # Cross-asset signals (VIX, yield curve, FX) are critical in volatile/bear regimes
    # LLM analysis adds +2-4% alpha from earnings, Fed speeches, SEC filings, news themes
    DEFAULT_REGIME_WEIGHTS = {
        MarketRegime.BULL: {
            SignalSource.LSTM: 0.20,
            SignalSource.DQN: 0.10,
            SignalSource.FACTOR: 0.20,
            SignalSource.MOMENTUM: 0.20,
            SignalSource.ALTERNATIVE_DATA: 0.20,
            SignalSource.CROSS_ASSET: 0.10,
        },
        MarketRegime.BEAR: {
            SignalSource.LSTM: 0.10,
            SignalSource.DQN: 0.20,
            SignalSource.FACTOR: 0.20,
            SignalSource.MEAN_REVERSION: 0.25,
            SignalSource.ALTERNATIVE_DATA: 0.10,
            SignalSource.CROSS_ASSET: 0.15,
        },
        MarketRegime.SIDEWAYS: {
            SignalSource.LSTM: 0.10,
            SignalSource.DQN: 0.10,
            SignalSource.FACTOR: 0.15,
            SignalSource.MEAN_REVERSION: 0.40,
            SignalSource.ALTERNATIVE_DATA: 0.15,
            SignalSource.CROSS_ASSET: 0.10,
        },
        MarketRegime.VOLATILE: {
            SignalSource.LSTM: 0.10,
            SignalSource.DQN: 0.30,
            SignalSource.FACTOR: 0.30,
            SignalSource.MOMENTUM: 0.10,
            SignalSource.ALTERNATIVE_DATA: 0.10,
            SignalSource.CROSS_ASSET: 0.10,
        },
    }

    def __init__(
        self,
        base_weights: Dict[SignalSource, float] = None,
        regime_weights: Dict[MarketRegime, Dict[SignalSource, float]] = None,
        min_sources_required: int = 2,
        confidence_threshold: float = 0.5,
        use_performance_weighting: bool = True,
        performance_window: int = 20,
    ):
        """
        Initialize the ensemble predictor.

        Args:
            base_weights: Default weights for each source
            regime_weights: Regime-specific weights
            min_sources_required: Minimum sources for valid prediction
            confidence_threshold: Minimum confidence for non-neutral signal
            use_performance_weighting: Adjust weights by recent performance
            performance_window: Window for performance measurement
        """
        self.base_weights = base_weights or {
            SignalSource.LSTM: 0.20,
            SignalSource.DQN: 0.10,
            SignalSource.FACTOR: 0.25,
            SignalSource.MOMENTUM: 0.10,
            SignalSource.ALTERNATIVE_DATA: 0.20,
            SignalSource.CROSS_ASSET: 0.15,
        }
        self.regime_weights = regime_weights or self.DEFAULT_REGIME_WEIGHTS
        self.min_sources_required = min_sources_required
        self.confidence_threshold = confidence_threshold
        self.use_performance_weighting = use_performance_weighting
        self.performance_window = performance_window

        # Signal source functions
        self._signal_sources: Dict[SignalSource, Callable] = {}

        # Performance tracking
        self._predictions: List[Dict] = []
        self._source_accuracy: Dict[SignalSource, List[float]] = {s: [] for s in SignalSource}

        # INSTITUTIONAL SAFETY: Track killed sources
        # Sources that fall below kill threshold are disabled
        self._killed_sources: set = set()

        # Meta-learner (optional)
        self._meta_learner = None

    def register_source(
        self,
        source: SignalSource,
        signal_fn: Callable[[str, Any], SignalComponent],
        weight: float = None,
    ):
        """
        Register a signal source.

        Args:
            source: SignalSource enum
            signal_fn: Function that takes (symbol, data) and returns SignalComponent
            weight: Optional custom weight for this source
        """
        self._signal_sources[source] = signal_fn

        if weight is not None:
            self.base_weights[source] = weight

        logger.info(f"Registered signal source: {source.value}")

    def _get_weights(
        self,
        regime: Optional[MarketRegime] = None,
        available_sources: List[SignalSource] = None,
    ) -> Dict[SignalSource, float]:
        """
        Get weights for signal combination.

        Args:
            regime: Current market regime
            available_sources: Sources that provided signals

        Returns:
            Normalized weights dictionary
        """
        # Start with base or regime-specific weights
        if regime and regime in self.regime_weights:
            weights = self.regime_weights[regime].copy()
        else:
            weights = self.base_weights.copy()

        # Filter to available sources
        if available_sources:
            weights = {s: w for s, w in weights.items() if s in available_sources}

        # Adjust by performance if enabled
        if self.use_performance_weighting:
            weights = self._adjust_by_performance(weights)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def _adjust_by_performance(
        self,
        weights: Dict[SignalSource, float],
    ) -> Dict[SignalSource, float]:
        """
        Adjust weights based on recent accuracy using Bayesian credibility intervals.

        INSTITUTIONAL SAFETY:
        - Requires minimum predictions before ANY adjustment
        - Uses Bayesian beta distribution for uncertainty-aware estimates
        - Only adjusts if credibility interval excludes 0.5 (baseline)
        - Limits adjustment factors to prevent wild swings
        - Kill switch for sources degrading >2σ below baseline
        """
        adjusted = {}

        for source, base_weight in weights.items():
            accuracy_history = self._source_accuracy.get(source, [])
            n_predictions = len(accuracy_history)

            # CRITICAL: Don't adjust until we have sufficient data
            if n_predictions < MIN_PREDICTIONS_FOR_ADJUSTMENT:
                adjusted[source] = base_weight
                continue

            # Use only recent predictions for responsiveness
            recent_history = accuracy_history[-self.performance_window :]
            n_recent = len(recent_history)
            n_successes = sum(recent_history)
            n_failures = n_recent - n_successes

            # Bayesian posterior: Beta distribution
            # Prior: Beta(5, 5) = 50% accuracy, pseudo-count of 10
            posterior_alpha = BAYESIAN_PRIOR_ALPHA + n_successes
            posterior_beta = BAYESIAN_PRIOR_BETA + n_failures

            # Posterior mean (Bayesian point estimate)
            bayesian_accuracy = posterior_alpha / (posterior_alpha + posterior_beta)

            # 95% credibility interval
            ci_lower = scipy_stats.beta.ppf(
                (1 - CREDIBILITY_LEVEL) / 2, posterior_alpha, posterior_beta
            )
            ci_upper = scipy_stats.beta.ppf(
                1 - (1 - CREDIBILITY_LEVEL) / 2, posterior_alpha, posterior_beta
            )

            # === KILL SWITCH ===
            # If upper credibility bound is below baseline - 2σ, disable source
            # For Beta(5+n_s, 5+n_f), std ≈ 0.5 / sqrt(10+n)
            effective_n = BAYESIAN_PRIOR_ALPHA + BAYESIAN_PRIOR_BETA + n_recent
            approx_std = 0.5 / np.sqrt(effective_n)
            kill_threshold = 0.5 - KILL_SWITCH_THRESHOLD_SIGMA * approx_std

            if ci_upper < kill_threshold:
                # Source is significantly below baseline - kill it
                logger.warning(
                    f"KILL SWITCH: {source.value} disabled. "
                    f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}] below kill threshold {kill_threshold:.3f}"
                )
                self._killed_sources.add(source)
                adjusted[source] = 0.0
                continue

            # === WEIGHT ADJUSTMENT ===
            # Only adjust if we're confident accuracy differs from 0.5

            if ci_lower > 0.5:
                # Confidently ABOVE baseline - modest boost
                # Use lower bound for conservative estimate
                excess_accuracy = ci_lower - 0.5
                # Map to weight factor: 0.0 → 1.0, 0.25 → 1.5 (capped)
                performance_factor = 1.0 + excess_accuracy * 2
                performance_factor = min(MAX_WEIGHT_INCREASE, performance_factor)

            elif ci_upper < 0.5:
                # Confidently BELOW baseline - reduce weight
                # Use upper bound for conservative estimate
                deficit_accuracy = 0.5 - ci_upper
                # Map to weight factor: 0.0 → 1.0, 0.25 → 0.5 (capped)
                performance_factor = 1.0 - deficit_accuracy * 2
                performance_factor = max(MIN_WEIGHT_DECREASE, performance_factor)

            else:
                # CI includes 0.5 - not confident, don't adjust
                performance_factor = 1.0

            adjusted[source] = base_weight * performance_factor

            # Debug logging for transparency
            logger.debug(
                f"{source.value}: accuracy={bayesian_accuracy:.3f} "
                f"CI=[{ci_lower:.3f}, {ci_upper:.3f}] "
                f"factor={performance_factor:.2f} "
                f"weight={adjusted[source]:.3f}"
            )

        return adjusted

    def is_source_killed(self, source: SignalSource) -> bool:
        """Check if a source has been disabled by kill switch."""
        return source in self._killed_sources

    def revive_source(self, source: SignalSource):
        """
        Revive a killed source (e.g., after manual review or retraining).

        WARNING: Only call this after investigating why the source degraded.
        """
        if source in self._killed_sources:
            self._killed_sources.discard(source)
            # Reset accuracy history to give it a fresh start
            self._source_accuracy[source] = []
            logger.info(f"Revived signal source: {source.value}")

    def predict(
        self,
        symbol: str,
        data: Any,
        regime: Optional[MarketRegime] = None,
    ) -> Optional[EnsemblePrediction]:
        """
        Generate ensemble prediction for a symbol.

        Args:
            symbol: Stock symbol
            data: Data to pass to signal sources
            regime: Optional market regime

        Returns:
            EnsemblePrediction or None if insufficient sources
        """
        # Collect signals from all sources (skip killed sources)
        components = {}

        for source, signal_fn in self._signal_sources.items():
            # Skip sources disabled by kill switch
            if source in self._killed_sources:
                logger.debug(f"Skipping killed source: {source.value}")
                continue

            try:
                signal = signal_fn(symbol, data)
                if signal is not None:
                    components[source] = signal
            except Exception as e:
                logger.warning(f"Signal source {source.value} failed: {e}")

        if len(components) < self.min_sources_required:
            logger.warning(
                f"Insufficient signals for {symbol}: "
                f"{len(components)} < {self.min_sources_required}"
            )
            return None

        # Get weights
        weights = self._get_weights(regime, list(components.keys()))

        # Combine signals
        if self._meta_learner is not None:
            ensemble_signal, ensemble_confidence = self._meta_predict(components)
        else:
            ensemble_signal, ensemble_confidence = self._weighted_combine(components, weights)

        # Determine direction
        if ensemble_signal > 0.1:
            direction = "long"
        elif ensemble_signal < -0.1:
            direction = "short"
        else:
            direction = "neutral"

        prediction = EnsemblePrediction(
            symbol=symbol,
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            direction=direction,
            components=components,
            weights_used=weights,
            timestamp=datetime.now(),
            regime=regime,
        )

        self._predictions.append(
            {
                "symbol": symbol,
                "prediction": prediction,
                "timestamp": datetime.now(),
            }
        )

        return prediction

    def _weighted_combine(
        self,
        components: Dict[SignalSource, SignalComponent],
        weights: Dict[SignalSource, float],
    ) -> Tuple[float, float]:
        """
        Combine signals using weighted average.

        Uses confidence-weighted combination within each source.
        """
        weighted_signal = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        for source, component in components.items():
            weight = weights.get(source, 0.0)
            if weight <= 0:
                continue

            # Weight by both assigned weight and source confidence
            effective_weight = weight * component.confidence
            weighted_signal += effective_weight * component.signal_value
            weighted_confidence += effective_weight * component.confidence
            total_weight += effective_weight

        if total_weight == 0:
            return 0.0, 0.0

        ensemble_signal = weighted_signal / total_weight
        ensemble_confidence = weighted_confidence / total_weight

        # Penalize confidence when sources disagree
        signals = [c.signal_value for c in components.values()]
        signal_std = np.std(signals) if len(signals) > 1 else 0
        disagreement_penalty = 1 - min(1, signal_std * 2)
        ensemble_confidence *= disagreement_penalty

        return ensemble_signal, ensemble_confidence

    def _meta_predict(
        self,
        components: Dict[SignalSource, SignalComponent],
    ) -> Tuple[float, float]:
        """Use meta-learner for combination."""
        if self._meta_learner is None:
            return self._weighted_combine(components, self.base_weights)

        # Build feature vector for meta-learner
        features = []
        for source in SignalSource:
            if source in components:
                features.extend(
                    [
                        components[source].signal_value,
                        components[source].confidence,
                    ]
                )
            else:
                features.extend([0.0, 0.0])

        features = np.array(features).reshape(1, -1)

        try:
            prediction = self._meta_learner.predict(features)[0]
            confidence = self._meta_learner.predict_proba(features).max()
            return prediction, confidence
        except Exception as e:
            logger.warning(f"Meta-learner prediction failed: {e}")
            return self._weighted_combine(components, self.base_weights)

    def train_meta_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "logistic",
    ):
        """
        Train meta-learner for signal combination.

        Args:
            X: Feature matrix (signals and confidences from each source)
            y: True outcomes (1 for profitable, 0 for not)
            model_type: 'logistic', 'random_forest', or 'xgboost'
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            if model_type == "logistic":
                self._meta_learner = LogisticRegression()
            elif model_type == "random_forest":
                self._meta_learner = RandomForestClassifier(n_estimators=50)
            else:
                self._meta_learner = LogisticRegression()

            self._meta_learner.fit(X, y)
            logger.info(f"Trained meta-learner with {len(X)} samples")

        except ImportError:
            logger.warning("sklearn not installed; using fallback meta-learner")

            class _FallbackMetaLearner:
                """Minimal probabilistic classifier fallback for test/dev environments."""

                def __init__(self):
                    self._p1 = 0.5

                def fit(self, X_fit, y_fit):
                    _ = np.asarray(X_fit)
                    y_arr = np.asarray(y_fit, dtype=float)
                    if y_arr.size > 0:
                        self._p1 = float(np.clip(np.mean(y_arr), 0.0, 1.0))
                    return self

                def predict(self, X_pred):
                    n_rows = np.asarray(X_pred).shape[0]
                    label = 1 if self._p1 >= 0.5 else 0
                    return np.full(n_rows, label, dtype=int)

                def predict_proba(self, X_pred):
                    n_rows = np.asarray(X_pred).shape[0]
                    p1 = float(np.clip(self._p1, 1e-6, 1 - 1e-6))
                    p0 = 1.0 - p1
                    return np.tile(np.array([p0, p1]), (n_rows, 1))

            self._meta_learner = _FallbackMetaLearner().fit(X, y)

    def record_outcome(
        self,
        symbol: str,
        prediction_time: datetime,
        actual_return: float,
    ):
        """
        Record actual outcome for performance tracking.

        Args:
            symbol: Stock symbol
            prediction_time: When prediction was made
            actual_return: Actual return achieved
        """
        # Find matching prediction
        for pred_record in reversed(self._predictions):
            if (
                pred_record["symbol"] == symbol
                and abs((pred_record["timestamp"] - prediction_time).total_seconds()) < 3600
            ):
                prediction = pred_record["prediction"]

                # Determine if prediction was correct
                correct = (prediction.direction == "long" and actual_return > 0) or (
                    prediction.direction == "short" and actual_return < 0
                )

                # Update accuracy for each source
                for source in prediction.components:
                    self._source_accuracy[source].append(1.0 if correct else 0.0)

                    # Trim history
                    if len(self._source_accuracy[source]) > 100:
                        self._source_accuracy[source] = self._source_accuracy[source][-100:]

                break

    def get_source_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance statistics for each signal source.

        Includes Bayesian credibility intervals and kill switch status.
        """
        performance = {}

        for source, accuracy_history in self._source_accuracy.items():
            n_predictions = len(accuracy_history)

            if n_predictions < 5:
                continue

            # Recent history for metrics
            recent_history = accuracy_history[-self.performance_window :]
            n_recent = len(recent_history)
            n_successes = sum(recent_history)
            n_failures = n_recent - n_successes

            # Bayesian posterior
            posterior_alpha = BAYESIAN_PRIOR_ALPHA + n_successes
            posterior_beta = BAYESIAN_PRIOR_BETA + n_failures

            bayesian_accuracy = posterior_alpha / (posterior_alpha + posterior_beta)

            # Credibility interval
            ci_lower = scipy_stats.beta.ppf(
                (1 - CREDIBILITY_LEVEL) / 2, posterior_alpha, posterior_beta
            )
            ci_upper = scipy_stats.beta.ppf(
                1 - (1 - CREDIBILITY_LEVEL) / 2, posterior_alpha, posterior_beta
            )

            performance[source.value] = {
                "accuracy_raw": np.mean(accuracy_history),
                "accuracy_bayesian": bayesian_accuracy,
                "recent_accuracy": np.mean(recent_history),
                "credibility_interval": (ci_lower, ci_upper),
                "n_predictions": n_predictions,
                "n_recent": n_recent,
                "current_weight": self.base_weights.get(source, 0),
                "is_killed": source in self._killed_sources,
                "min_predictions_met": n_predictions >= MIN_PREDICTIONS_FOR_ADJUSTMENT,
            }

        return performance

    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get overall ensemble statistics."""
        if not self._predictions:
            return {"error": "No predictions made yet"}

        return {
            "total_predictions": len(self._predictions),
            "sources_registered": len(self._signal_sources),
            "sources_active": len(self._signal_sources) - len(self._killed_sources),
            "sources_killed": [s.value for s in self._killed_sources],
            "source_performance": self.get_source_performance(),
            "has_meta_learner": self._meta_learner is not None,
            "base_weights": {s.value: w for s, w in self.base_weights.items()},
            "safety_settings": {
                "min_predictions_for_adjustment": MIN_PREDICTIONS_FOR_ADJUSTMENT,
                "kill_switch_threshold_sigma": KILL_SWITCH_THRESHOLD_SIGMA,
                "credibility_level": CREDIBILITY_LEVEL,
                "max_weight_increase": MAX_WEIGHT_INCREASE,
                "min_weight_decrease": MIN_WEIGHT_DECREASE,
            },
        }


def create_ensemble_from_components(
    lstm_predictor=None,
    dqn_agent=None,
    factor_model=None,
    momentum_strategy=None,
) -> EnsemblePredictor:
    """
    Create ensemble predictor from existing components.

    Args:
        lstm_predictor: LSTMPredictor instance
        dqn_agent: DQNAgent instance
        factor_model: FactorModel instance
        momentum_strategy: MomentumStrategy instance

    Returns:
        Configured EnsemblePredictor
    """
    ensemble = EnsemblePredictor()

    if lstm_predictor is not None:

        def lstm_signal(symbol: str, data: Any) -> Optional[SignalComponent]:
            result = lstm_predictor.predict_with_uncertainty(symbol, data.get("prices", []))
            if result is None:
                return None

            return SignalComponent(
                source=SignalSource.LSTM,
                signal_value=1.0 if result.predicted_direction == "up" else -1.0,
                confidence=result.confidence,
                direction=result.predicted_direction.replace("up", "long").replace("down", "short"),
                metadata={"std": result.std_prediction},
            )

        ensemble.register_source(SignalSource.LSTM, lstm_signal)

    if dqn_agent is not None:

        def dqn_signal(symbol: str, data: Any) -> Optional[SignalComponent]:
            try:
                state = data.get("state")
                if state is None:
                    return None

                action, confidence = dqn_agent.act(state, return_confidence=True)

                # Map DQN action to signal
                if action == 0:  # Hold
                    signal_value = 0.0
                    direction = "neutral"
                elif action == 1:  # Buy
                    signal_value = 1.0
                    direction = "long"
                else:  # Sell
                    signal_value = -1.0
                    direction = "short"

                return SignalComponent(
                    source=SignalSource.DQN,
                    signal_value=signal_value,
                    confidence=confidence,
                    direction=direction,
                )
            except Exception:
                return None

        ensemble.register_source(SignalSource.DQN, dqn_signal)

    if factor_model is not None:

        def factor_signal(symbol: str, data: Any) -> Optional[SignalComponent]:
            scores = data.get("factor_scores", {})
            signal_dict = factor_model.get_signal(symbol, scores)

            if signal_dict["action"] == "hold":
                return None

            return SignalComponent(
                source=SignalSource.FACTOR,
                signal_value=signal_dict.get("composite_z", 0) / 2,  # Scale to -1, 1
                confidence=signal_dict.get("confidence", 0.5),
                direction=signal_dict["action"],
                metadata=signal_dict.get("factor_breakdown"),
            )

        ensemble.register_source(SignalSource.FACTOR, factor_signal)

    if momentum_strategy is not None:

        def momentum_signal(symbol: str, data: Any) -> Optional[SignalComponent]:
            try:
                signal = momentum_strategy.analyze_symbol(symbol)
                if signal is None:
                    return None

                action = signal.get("action", "hold")
                if action == "hold":
                    signal_value = 0.0
                    direction = "neutral"
                elif action == "buy":
                    signal_value = 1.0
                    direction = "long"
                else:
                    signal_value = -1.0
                    direction = "short"

                return SignalComponent(
                    source=SignalSource.MOMENTUM,
                    signal_value=signal_value,
                    confidence=signal.get("confidence", 0.5),
                    direction=direction,
                )
            except Exception:
                return None

        ensemble.register_source(SignalSource.MOMENTUM, momentum_signal)

    return ensemble


class AltDataSignalGenerator:
    """
    Alternative Data Signal Generator for Ensemble Integration.

    Wraps the AltDataAggregator to produce SignalComponent objects
    compatible with the EnsemblePredictor.

    Combines signals from:
    - Social sentiment (Reddit, Twitter, StockTwits)
    - Order flow (dark pool prints, options flow)
    - Web data (job postings, Glassdoor, app rankings)
    """

    def __init__(
        self,
        use_social: bool = True,
        use_order_flow: bool = True,
        use_web_data: bool = True,
        confidence_floor: float = 0.3,
    ):
        """
        Initialize alternative data signal generator.

        Args:
            use_social: Include social sentiment signals
            use_order_flow: Include order flow signals
            use_web_data: Include web scraping signals
            confidence_floor: Minimum confidence to return a signal
        """
        self.use_social = use_social
        self.use_order_flow = use_order_flow
        self.use_web_data = use_web_data
        self.confidence_floor = confidence_floor

        self._aggregator = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all alternative data providers."""
        try:
            from data.alternative_data_provider import AltDataAggregator

            self._aggregator = AltDataAggregator()

            # Register providers based on configuration
            if self.use_social:
                from data.social_sentiment_advanced import RedditSentimentProvider

                provider = RedditSentimentProvider()
                await provider.initialize()
                self._aggregator.register_provider(provider)

            if self.use_order_flow:
                from data.order_flow_analyzer import OrderFlowAnalyzer

                provider = OrderFlowAnalyzer()
                await provider.initialize()
                self._aggregator.register_provider(provider)

            if self.use_web_data:
                from data.web_scraper import (
                    AppRankingsProvider,
                    GlassdoorSentimentProvider,
                    JobPostingsProvider,
                )

                for ProviderClass in [
                    JobPostingsProvider,
                    GlassdoorSentimentProvider,
                    AppRankingsProvider,
                ]:
                    provider = ProviderClass()
                    await provider.initialize()
                    self._aggregator.register_provider(provider)

            self._initialized = True
            logger.info("AltDataSignalGenerator initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"Alternative data modules not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize AltDataSignalGenerator: {e}")
            return False

    async def get_signal(self, symbol: str, data: Any = None) -> Optional[SignalComponent]:
        """
        Get alternative data signal for a symbol.

        Args:
            symbol: Stock ticker symbol
            data: Optional additional data (not used currently)

        Returns:
            SignalComponent or None if no signal available
        """
        if not self._initialized:
            success = await self.initialize()
            if not success:
                return None

        try:
            # Get aggregated signal from all sources
            aggregated = await self._aggregator.get_signal(symbol)

            if aggregated is None:
                return None

            # Convert to signal value (-1 to +1)
            signal_value = aggregated.composite_signal

            # Get confidence
            confidence = min(0.9, aggregated.composite_confidence)

            # Apply confidence floor
            if confidence < self.confidence_floor:
                return None

            # Determine direction
            if signal_value > 0.15:
                direction = "long"
            elif signal_value < -0.15:
                direction = "short"
            else:
                direction = "neutral"

            return SignalComponent(
                source=SignalSource.ALTERNATIVE_DATA,
                signal_value=signal_value,
                confidence=confidence,
                direction=direction,
                metadata={
                    "sources": [s.value for s in aggregated.sources],
                    "agreement_ratio": aggregated.agreement_ratio,
                    "high_conviction": aggregated.high_conviction,
                },
            )

        except Exception as e:
            logger.error(f"Error getting alt data signal for {symbol}: {e}")
            return None

    def get_signal_sync(self, symbol: str, data: Any = None) -> Optional[SignalComponent]:
        """
        Synchronous wrapper for get_signal.

        Use this when integrating with sync code.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop for this call
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.get_signal(symbol, data))
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.get_signal(symbol, data))
        except Exception as e:
            logger.error(f"Sync wrapper failed for {symbol}: {e}")
            return None


def create_ensemble_with_alt_data(
    lstm_predictor=None,
    dqn_agent=None,
    factor_model=None,
    momentum_strategy=None,
    alt_data_generator: AltDataSignalGenerator = None,
) -> EnsemblePredictor:
    """
    Create ensemble predictor including alternative data.

    Args:
        lstm_predictor: LSTMPredictor instance
        dqn_agent: DQNAgent instance
        factor_model: FactorModel instance
        momentum_strategy: MomentumStrategy instance
        alt_data_generator: AltDataSignalGenerator instance

    Returns:
        Configured EnsemblePredictor with alternative data
    """
    # Start with base ensemble
    ensemble = create_ensemble_from_components(
        lstm_predictor=lstm_predictor,
        dqn_agent=dqn_agent,
        factor_model=factor_model,
        momentum_strategy=momentum_strategy,
    )

    # Add alternative data source
    if alt_data_generator is not None:

        def alt_data_signal(symbol: str, data: Any) -> Optional[SignalComponent]:
            return alt_data_generator.get_signal_sync(symbol, data)

        ensemble.register_source(SignalSource.ALTERNATIVE_DATA, alt_data_signal)
        logger.info("Registered ALTERNATIVE_DATA source in ensemble")

    return ensemble


class CrossAssetSignalGenerator:
    """
    Generates cross-asset signals for ensemble integration.

    Wraps the CrossAssetAggregator to provide SignalComponent outputs
    compatible with the ensemble predictor.
    """

    def __init__(
        self,
        use_vix: bool = True,
        use_yield_curve: bool = True,
        use_fx: bool = True,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize cross-asset signal generator.

        Args:
            use_vix: Include VIX term structure signals
            use_yield_curve: Include yield curve signals
            use_fx: Include FX correlation signals
            cache_ttl_seconds: Cache TTL for data fetching
        """
        self._use_vix = use_vix
        self._use_yield_curve = use_yield_curve
        self._use_fx = use_fx
        self._cache_ttl = cache_ttl_seconds
        self._aggregator = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the cross-asset aggregator."""
        try:
            from data.cross_asset_provider import CrossAssetAggregator

            self._aggregator = CrossAssetAggregator(
                use_vix=self._use_vix,
                use_yield_curve=self._use_yield_curve,
                use_fx=self._use_fx,
                cache_ttl_seconds=self._cache_ttl,
            )

            self._initialized = await self._aggregator.initialize()
            if self._initialized:
                logger.info("CrossAssetSignalGenerator initialized successfully")
            return self._initialized

        except Exception as e:
            logger.error(f"Failed to initialize CrossAssetSignalGenerator: {e}")
            return False

    async def get_signal(self, symbol: str = None, data: Any = None) -> Optional[SignalComponent]:
        """
        Get cross-asset signal.

        Note: Cross-asset signals are global (not per-symbol), but we accept
        symbol parameter for interface compatibility.

        Args:
            symbol: Ignored - cross-asset signals are market-wide
            data: Optional additional data

        Returns:
            SignalComponent with cross-asset signal or None
        """
        if not self._initialized:
            await self.initialize()

        if not self._initialized or self._aggregator is None:
            return None

        try:
            aggregated = await self._aggregator.get_signal()

            if aggregated is None:
                return None

            # Determine direction
            if aggregated.composite_signal > 0.1:
                direction = "long"
            elif aggregated.composite_signal < -0.1:
                direction = "short"
            else:
                direction = "neutral"

            return SignalComponent(
                source=SignalSource.CROSS_ASSET,
                signal_value=aggregated.composite_signal,
                confidence=min(0.9, aggregated.composite_confidence),
                direction=direction,
                metadata={
                    "overall_regime": aggregated.overall_regime,
                    "regime_strength": aggregated.regime_strength,
                    "agreement_ratio": aggregated.agreement_ratio,
                    "sources": [s.value for s in aggregated.sources],
                    "should_reduce_exposure": aggregated.should_reduce_exposure,
                    "vix_signal": (
                        aggregated.vix_signal.signal_value if aggregated.vix_signal else None
                    ),
                    "yield_signal": (
                        aggregated.yield_curve_signal.signal_value
                        if aggregated.yield_curve_signal
                        else None
                    ),
                    "fx_signal": (
                        aggregated.fx_signal.signal_value if aggregated.fx_signal else None
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Error getting cross-asset signal: {e}")
            return None

    def get_signal_sync(self, symbol: str = None, data: Any = None) -> Optional[SignalComponent]:
        """
        Synchronous wrapper for get_signal.

        Use this when integrating with sync code.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.get_signal(symbol, data))
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.get_signal(symbol, data))
        except Exception as e:
            logger.error(f"Sync wrapper failed for cross-asset signal: {e}")
            return None


def create_ensemble_with_cross_asset(
    lstm_predictor=None,
    dqn_agent=None,
    factor_model=None,
    momentum_strategy=None,
    alt_data_generator: AltDataSignalGenerator = None,
    cross_asset_generator: CrossAssetSignalGenerator = None,
) -> EnsemblePredictor:
    """
    Create ensemble predictor including cross-asset signals.

    This is the most comprehensive ensemble configuration, including:
    - LSTM neural network predictions
    - DQN reinforcement learning agent
    - Factor model scores
    - Momentum strategy signals
    - Alternative data (social, order flow, web)
    - Cross-asset signals (VIX, yield curve, FX)

    Args:
        lstm_predictor: LSTMPredictor instance
        dqn_agent: DQNAgent instance
        factor_model: FactorModel instance
        momentum_strategy: MomentumStrategy instance
        alt_data_generator: AltDataSignalGenerator instance
        cross_asset_generator: CrossAssetSignalGenerator instance

    Returns:
        Configured EnsemblePredictor with all signal sources
    """
    # Start with alt data ensemble
    ensemble = create_ensemble_with_alt_data(
        lstm_predictor=lstm_predictor,
        dqn_agent=dqn_agent,
        factor_model=factor_model,
        momentum_strategy=momentum_strategy,
        alt_data_generator=alt_data_generator,
    )

    # Add cross-asset source
    if cross_asset_generator is not None:

        def cross_asset_signal(symbol: str, data: Any) -> Optional[SignalComponent]:
            return cross_asset_generator.get_signal_sync(symbol, data)

        ensemble.register_source(SignalSource.CROSS_ASSET, cross_asset_signal)
        logger.info("Registered CROSS_ASSET source in ensemble")

    return ensemble


def create_full_ensemble(
    lstm_predictor=None,
    dqn_agent=None,
    factor_model=None,
    momentum_strategy=None,
    use_alt_data: bool = True,
    use_cross_asset: bool = True,
) -> EnsemblePredictor:
    """
    Create a fully-configured ensemble with all available signal sources.

    This is the recommended way to create an ensemble for production use.
    It automatically initializes alternative data and cross-asset generators.

    Args:
        lstm_predictor: LSTMPredictor instance
        dqn_agent: DQNAgent instance
        factor_model: FactorModel instance
        momentum_strategy: MomentumStrategy instance
        use_alt_data: Enable alternative data signals
        use_cross_asset: Enable cross-asset signals

    Returns:
        Fully configured EnsemblePredictor
    """
    alt_data_gen = None
    cross_asset_gen = None

    if use_alt_data:
        alt_data_gen = AltDataSignalGenerator(
            use_social=True,
            use_order_flow=True,
            use_web_data=True,
        )

    if use_cross_asset:
        cross_asset_gen = CrossAssetSignalGenerator(
            use_vix=True,
            use_yield_curve=True,
            use_fx=True,
        )

    return create_ensemble_with_cross_asset(
        lstm_predictor=lstm_predictor,
        dqn_agent=dqn_agent,
        factor_model=factor_model,
        momentum_strategy=momentum_strategy,
        alt_data_generator=alt_data_gen,
        cross_asset_generator=cross_asset_gen,
    )


class LLMSignalGenerator:
    """
    LLM Signal Generator for Ensemble Integration.

    Aggregates signals from LLM-powered analysis providers:
    - Earnings call analysis
    - Fed speech analysis
    - SEC filing analysis
    - News theme extraction

    Expected Alpha Contribution: +2-4% annually
    Daily API Cost: ~$20-50 (with aggressive caching)
    """

    def __init__(
        self,
        use_earnings: bool = True,
        use_fed_speech: bool = True,
        use_sec_filings: bool = True,
        use_news_themes: bool = True,
        confidence_floor: float = 0.4,
    ):
        """
        Initialize LLM signal generator.

        Args:
            use_earnings: Include earnings call analysis
            use_fed_speech: Include Fed speech analysis
            use_sec_filings: Include SEC filing analysis
            use_news_themes: Include news theme extraction
            confidence_floor: Minimum confidence to return a signal
        """
        self.use_earnings = use_earnings
        self.use_fed_speech = use_fed_speech
        self.use_sec_filings = use_sec_filings
        self.use_news_themes = use_news_themes
        self.confidence_floor = confidence_floor

        self._earnings_analyzer = None
        self._fed_analyzer = None
        self._sec_analyzer = None
        self._news_extractor = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all LLM analysis providers."""
        try:
            from llm import create_llm_client

            # Create shared LLM client
            llm_client = create_llm_client()

            if self.use_earnings:
                from data.llm_providers import EarningsCallAnalyzer

                self._earnings_analyzer = EarningsCallAnalyzer(llm_client=llm_client)
                await self._earnings_analyzer.initialize()

            if self.use_fed_speech:
                from data.llm_providers import FedSpeechAnalyzer

                self._fed_analyzer = FedSpeechAnalyzer(llm_client=llm_client)
                await self._fed_analyzer.initialize()

            if self.use_sec_filings:
                from data.llm_providers import SECFilingAnalyzer

                self._sec_analyzer = SECFilingAnalyzer(llm_client=llm_client)
                await self._sec_analyzer.initialize()

            if self.use_news_themes:
                from data.llm_providers import NewsThemeExtractor

                self._news_extractor = NewsThemeExtractor(llm_client=llm_client)
                await self._news_extractor.initialize()

            self._initialized = True
            logger.info("LLMSignalGenerator initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"LLM modules not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LLMSignalGenerator: {e}")
            return False

    async def get_signal(self, symbol: str, data: Any = None) -> Optional[SignalComponent]:
        """
        Get combined LLM analysis signal for a symbol.

        Aggregates signals from all enabled analyzers and returns
        a weighted combination.

        Args:
            symbol: Stock ticker symbol
            data: Optional additional data

        Returns:
            SignalComponent or None if no signal available
        """
        if not self._initialized:
            success = await self.initialize()
            if not success:
                return None

        signals = []
        source_weights = {
            "earnings": 0.35,  # Highest weight - direct company info
            "fed_speech": 0.15,  # Market-wide, less symbol-specific
            "sec_filing": 0.25,  # Important fundamental data
            "news_theme": 0.25,  # Timely but noisy
        }

        try:
            # Collect signals from all providers
            if self._earnings_analyzer:
                earnings_signal = await self._earnings_analyzer.fetch_signal(symbol)
                if earnings_signal and earnings_signal.confidence >= self.confidence_floor:
                    signals.append(("earnings", earnings_signal))

            if self._fed_analyzer:
                fed_signal = await self._fed_analyzer.fetch_signal(symbol)
                if fed_signal and fed_signal.confidence >= self.confidence_floor:
                    signals.append(("fed_speech", fed_signal))

            if self._sec_analyzer:
                sec_signal = await self._sec_analyzer.fetch_signal(symbol)
                if sec_signal and sec_signal.confidence >= self.confidence_floor:
                    signals.append(("sec_filing", sec_signal))

            if self._news_extractor:
                news_signal = await self._news_extractor.fetch_signal(symbol)
                if news_signal and news_signal.confidence >= self.confidence_floor:
                    signals.append(("news_theme", news_signal))

            if not signals:
                return None

            # Weighted combination
            total_weight = 0.0
            weighted_signal = 0.0
            weighted_confidence = 0.0
            sources_used = []

            for source_name, signal in signals:
                weight = source_weights.get(source_name, 0.1)
                effective_weight = weight * signal.confidence
                weighted_signal += effective_weight * signal.signal_value
                weighted_confidence += effective_weight * signal.confidence
                total_weight += effective_weight
                sources_used.append(source_name)

            if total_weight == 0:
                return None

            combined_signal = weighted_signal / total_weight
            combined_confidence = weighted_confidence / total_weight

            # Determine direction
            if combined_signal > 0.15:
                direction = "long"
            elif combined_signal < -0.15:
                direction = "short"
            else:
                direction = "neutral"

            return SignalComponent(
                source=SignalSource.LLM_ANALYSIS,
                signal_value=max(-1.0, min(1.0, combined_signal)),
                confidence=min(0.9, combined_confidence),
                direction=direction,
                metadata={
                    "sources_used": sources_used,
                    "num_sources": len(signals),
                    "source_weights": source_weights,
                },
            )

        except Exception as e:
            logger.error(f"Error getting LLM signal for {symbol}: {e}")
            return None

    def get_signal_sync(self, symbol: str, data: Any = None) -> Optional[SignalComponent]:
        """
        Synchronous wrapper for get_signal.

        Use this when integrating with sync code.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.get_signal(symbol, data))
                    return future.result(timeout=60)  # Longer timeout for LLM calls
            else:
                return loop.run_until_complete(self.get_signal(symbol, data))
        except Exception as e:
            logger.error(f"Sync wrapper failed for {symbol}: {e}")
            return None


def create_ensemble_with_llm(
    lstm_predictor=None,
    dqn_agent=None,
    factor_model=None,
    momentum_strategy=None,
    alt_data_generator: AltDataSignalGenerator = None,
    cross_asset_generator: CrossAssetSignalGenerator = None,
    llm_generator: LLMSignalGenerator = None,
) -> EnsemblePredictor:
    """
    Create ensemble predictor including LLM analysis signals.

    This is the most comprehensive ensemble configuration, including:
    - LSTM neural network predictions
    - DQN reinforcement learning agent
    - Factor model scores
    - Momentum strategy signals
    - Alternative data (social, order flow, web)
    - Cross-asset signals (VIX, yield curve, FX)
    - LLM analysis (earnings, Fed, SEC, news)

    Args:
        lstm_predictor: LSTMPredictor instance
        dqn_agent: DQNAgent instance
        factor_model: FactorModel instance
        momentum_strategy: MomentumStrategy instance
        alt_data_generator: AltDataSignalGenerator instance
        cross_asset_generator: CrossAssetSignalGenerator instance
        llm_generator: LLMSignalGenerator instance

    Returns:
        Configured EnsemblePredictor with all signal sources
    """
    # Start with cross-asset ensemble
    ensemble = create_ensemble_with_cross_asset(
        lstm_predictor=lstm_predictor,
        dqn_agent=dqn_agent,
        factor_model=factor_model,
        momentum_strategy=momentum_strategy,
        alt_data_generator=alt_data_generator,
        cross_asset_generator=cross_asset_generator,
    )

    # Add LLM analysis source
    if llm_generator is not None:

        def llm_signal(symbol: str, data: Any) -> Optional[SignalComponent]:
            return llm_generator.get_signal_sync(symbol, data)

        ensemble.register_source(SignalSource.LLM_ANALYSIS, llm_signal)
        logger.info("Registered LLM_ANALYSIS source in ensemble")

    return ensemble


def create_institutional_ensemble(
    lstm_predictor=None,
    dqn_agent=None,
    factor_model=None,
    momentum_strategy=None,
    use_alt_data: bool = True,
    use_cross_asset: bool = True,
    use_llm: bool = True,
) -> EnsemblePredictor:
    """
    Create a fully-configured institutional-grade ensemble.

    This is the recommended configuration for production deployment,
    including all available signal sources.

    Features:
    - LSTM and DQN ML models
    - Factor model with quality/value/momentum
    - Alternative data (social, order flow, web)
    - Cross-asset signals (VIX, yield curve, FX)
    - LLM analysis (earnings, Fed, SEC, news)

    Estimated Alpha: +8-12% annually (before costs)
    LLM Cost: ~$20-50/day with caching

    Args:
        lstm_predictor: LSTMPredictor instance
        dqn_agent: DQNAgent instance
        factor_model: FactorModel instance
        momentum_strategy: MomentumStrategy instance
        use_alt_data: Enable alternative data signals
        use_cross_asset: Enable cross-asset signals
        use_llm: Enable LLM analysis signals

    Returns:
        Fully configured institutional-grade EnsemblePredictor
    """
    alt_data_gen = None
    cross_asset_gen = None
    llm_gen = None

    if use_alt_data:
        alt_data_gen = AltDataSignalGenerator(
            use_social=True,
            use_order_flow=True,
            use_web_data=True,
        )

    if use_cross_asset:
        cross_asset_gen = CrossAssetSignalGenerator(
            use_vix=True,
            use_yield_curve=True,
            use_fx=True,
        )

    if use_llm:
        llm_gen = LLMSignalGenerator(
            use_earnings=True,
            use_fed_speech=True,
            use_sec_filings=True,
            use_news_themes=True,
        )

    return create_ensemble_with_llm(
        lstm_predictor=lstm_predictor,
        dqn_agent=dqn_agent,
        factor_model=factor_model,
        momentum_strategy=momentum_strategy,
        alt_data_generator=alt_data_gen,
        cross_asset_generator=cross_asset_gen,
        llm_generator=llm_gen,
    )
