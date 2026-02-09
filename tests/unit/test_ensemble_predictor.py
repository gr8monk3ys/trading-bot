"""
Unit tests for EnsemblePredictor.

Tests signal source registration, regime-dependent weighting,
weighted signal combination, confidence calculations, and
performance tracking.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml.ensemble_predictor import (
    EnsemblePrediction,
    EnsemblePredictor,
    MarketRegime,
    SignalComponent,
    SignalSource,
    create_ensemble_from_components,
)

# =============================================================================
# TEST FIXTURES AND HELPERS
# =============================================================================


def create_signal_component(
    source: SignalSource,
    signal_value: float = 0.5,
    confidence: float = 0.7,
    direction: str = "long",
    metadata: Dict[str, Any] = None,
) -> SignalComponent:
    """Helper to create SignalComponent with defaults."""
    return SignalComponent(
        source=source,
        signal_value=signal_value,
        confidence=confidence,
        direction=direction,
        metadata=metadata,
    )


def create_mock_signal_fn(
    source: SignalSource,
    signal_value: float = 0.5,
    confidence: float = 0.7,
    direction: str = "long",
    metadata: Dict[str, Any] = None,
    should_return_none: bool = False,
    should_raise: bool = False,
):
    """Create a mock signal function that returns a SignalComponent."""
    def signal_fn(symbol: str, data: Any) -> Optional[SignalComponent]:
        if should_raise:
            raise ValueError("Simulated signal failure")
        if should_return_none:
            return None
        return create_signal_component(
            source=source,
            signal_value=signal_value,
            confidence=confidence,
            direction=direction,
            metadata=metadata,
        )
    return signal_fn


@pytest.fixture
def ensemble_predictor() -> EnsemblePredictor:
    """Create a default EnsemblePredictor instance."""
    return EnsemblePredictor()


@pytest.fixture
def ensemble_with_sources() -> EnsemblePredictor:
    """Create an EnsemblePredictor with multiple signal sources registered."""
    ensemble = EnsemblePredictor(min_sources_required=2)

    # Register LSTM source - bullish signal
    ensemble.register_source(
        SignalSource.LSTM,
        create_mock_signal_fn(SignalSource.LSTM, signal_value=0.8, confidence=0.9, direction="long"),
    )

    # Register DQN source - bullish signal
    ensemble.register_source(
        SignalSource.DQN,
        create_mock_signal_fn(SignalSource.DQN, signal_value=0.6, confidence=0.7, direction="long"),
    )

    # Register FACTOR source - slightly bullish
    ensemble.register_source(
        SignalSource.FACTOR,
        create_mock_signal_fn(SignalSource.FACTOR, signal_value=0.4, confidence=0.8, direction="long"),
    )

    return ensemble


@pytest.fixture
def ensemble_with_conflicting_signals() -> EnsemblePredictor:
    """Create an EnsemblePredictor with conflicting signal sources."""
    ensemble = EnsemblePredictor(min_sources_required=2)

    # Register LSTM source - bullish
    ensemble.register_source(
        SignalSource.LSTM,
        create_mock_signal_fn(SignalSource.LSTM, signal_value=0.9, confidence=0.8, direction="long"),
    )

    # Register DQN source - bearish
    ensemble.register_source(
        SignalSource.DQN,
        create_mock_signal_fn(SignalSource.DQN, signal_value=-0.9, confidence=0.8, direction="short"),
    )

    return ensemble


# =============================================================================
# SIGNAL COMPONENT TESTS
# =============================================================================


class TestSignalComponent:
    """Tests for SignalComponent dataclass."""

    def test_create_signal_component_with_all_fields(self):
        """Test creating SignalComponent with all fields specified."""
        component = SignalComponent(
            source=SignalSource.LSTM,
            signal_value=0.75,
            confidence=0.85,
            direction="long",
            metadata={"feature": "value"},
        )

        assert component.source == SignalSource.LSTM
        assert component.signal_value == 0.75
        assert component.confidence == 0.85
        assert component.direction == "long"
        assert component.metadata == {"feature": "value"}

    def test_create_signal_component_metadata_defaults_to_empty_dict(self):
        """Test that metadata defaults to empty dict if not provided."""
        component = SignalComponent(
            source=SignalSource.DQN,
            signal_value=0.5,
            confidence=0.6,
            direction="neutral",
        )

        assert component.metadata == {}

    def test_signal_component_accepts_negative_signal_value(self):
        """Test that signal_value can be negative (short signal)."""
        component = SignalComponent(
            source=SignalSource.FACTOR,
            signal_value=-0.8,
            confidence=0.7,
            direction="short",
        )

        assert component.signal_value == -0.8
        assert component.direction == "short"

    def test_signal_component_accepts_zero_signal_value(self):
        """Test that signal_value can be zero (neutral)."""
        component = SignalComponent(
            source=SignalSource.MOMENTUM,
            signal_value=0.0,
            confidence=0.5,
            direction="neutral",
        )

        assert component.signal_value == 0.0
        assert component.direction == "neutral"


# =============================================================================
# ENSEMBLE PREDICTION TESTS
# =============================================================================


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction dataclass."""

    def create_prediction(
        self,
        ensemble_signal: float = 0.5,
        ensemble_confidence: float = 0.7,
        direction: str = "long",
    ) -> EnsemblePrediction:
        """Helper to create EnsemblePrediction."""
        return EnsemblePrediction(
            symbol="AAPL",
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            direction=direction,
            components={},
            weights_used={},
            timestamp=datetime.now(),
            regime=None,
        )

    def test_should_trade_returns_true_when_all_conditions_met(self):
        """Test should_trade returns True with high confidence and strong signal."""
        prediction = self.create_prediction(
            ensemble_signal=0.5,
            ensemble_confidence=0.7,
            direction="long",
        )

        assert prediction.should_trade(min_confidence=0.6) is True

    def test_should_trade_returns_false_when_confidence_too_low(self):
        """Test should_trade returns False when confidence below threshold."""
        prediction = self.create_prediction(
            ensemble_signal=0.5,
            ensemble_confidence=0.5,  # Below default 0.6 threshold
            direction="long",
        )

        assert prediction.should_trade(min_confidence=0.6) is False

    def test_should_trade_returns_false_when_signal_too_weak(self):
        """Test should_trade returns False when signal magnitude below 0.3."""
        prediction = self.create_prediction(
            ensemble_signal=0.2,  # Below 0.3 threshold
            ensemble_confidence=0.8,
            direction="long",
        )

        assert prediction.should_trade() is False

    def test_should_trade_returns_false_when_direction_neutral(self):
        """Test should_trade returns False when direction is neutral."""
        prediction = self.create_prediction(
            ensemble_signal=0.5,
            ensemble_confidence=0.8,
            direction="neutral",
        )

        assert prediction.should_trade() is False

    def test_should_trade_with_short_direction(self):
        """Test should_trade works correctly for short signals."""
        prediction = self.create_prediction(
            ensemble_signal=-0.5,
            ensemble_confidence=0.7,
            direction="short",
        )

        assert prediction.should_trade(min_confidence=0.6) is True

    def test_should_trade_with_custom_min_confidence(self):
        """Test should_trade with custom minimum confidence threshold."""
        prediction = self.create_prediction(
            ensemble_signal=0.5,
            ensemble_confidence=0.4,
            direction="long",
        )

        assert prediction.should_trade(min_confidence=0.3) is True
        assert prediction.should_trade(min_confidence=0.5) is False

    def test_get_position_size_multiplier_min_confidence(self):
        """Test position size multiplier at minimum confidence (0)."""
        prediction = self.create_prediction(ensemble_confidence=0.0)

        # 0.5 + 0.0 = 0.5
        assert prediction.get_position_size_multiplier() == 0.5

    def test_get_position_size_multiplier_max_confidence(self):
        """Test position size multiplier at maximum confidence (1)."""
        prediction = self.create_prediction(ensemble_confidence=1.0)

        # 0.5 + 1.0 = 1.5
        assert prediction.get_position_size_multiplier() == 1.5

    def test_get_position_size_multiplier_mid_confidence(self):
        """Test position size multiplier at mid confidence (0.5)."""
        prediction = self.create_prediction(ensemble_confidence=0.5)

        # 0.5 + 0.5 = 1.0
        assert prediction.get_position_size_multiplier() == 1.0

    def test_get_position_size_multiplier_scales_linearly(self):
        """Test position size multiplier scales linearly with confidence."""
        confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = [0.5, 0.75, 1.0, 1.25, 1.5]

        for conf, exp in zip(confidences, expected, strict=False):
            prediction = self.create_prediction(ensemble_confidence=conf)
            assert prediction.get_position_size_multiplier() == exp


# =============================================================================
# ENSEMBLE PREDICTOR INITIALIZATION TESTS
# =============================================================================


class TestEnsemblePredictorInit:
    """Tests for EnsemblePredictor initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        ensemble = EnsemblePredictor()

        assert ensemble.min_sources_required == 2
        assert ensemble.confidence_threshold == 0.5
        assert ensemble.use_performance_weighting is True
        assert ensemble.performance_window == 20
        assert len(ensemble._signal_sources) == 0
        assert ensemble._meta_learner is None

    def test_custom_initialization(self):
        """Test custom initialization values."""
        custom_weights = {SignalSource.LSTM: 0.5, SignalSource.DQN: 0.5}

        ensemble = EnsemblePredictor(
            base_weights=custom_weights,
            min_sources_required=3,
            confidence_threshold=0.7,
            use_performance_weighting=False,
            performance_window=30,
        )

        assert ensemble.base_weights == custom_weights
        assert ensemble.min_sources_required == 3
        assert ensemble.confidence_threshold == 0.7
        assert ensemble.use_performance_weighting is False
        assert ensemble.performance_window == 30

    def test_default_base_weights(self):
        """Test default base weights are set correctly."""
        ensemble = EnsemblePredictor()

        assert SignalSource.LSTM in ensemble.base_weights
        assert SignalSource.DQN in ensemble.base_weights
        assert SignalSource.FACTOR in ensemble.base_weights
        assert SignalSource.MOMENTUM in ensemble.base_weights
        assert SignalSource.ALTERNATIVE_DATA in ensemble.base_weights
        assert SignalSource.CROSS_ASSET in ensemble.base_weights

        # Default weights including alternative data and cross-asset (sum to 1.0)
        assert ensemble.base_weights[SignalSource.LSTM] == 0.20
        assert ensemble.base_weights[SignalSource.DQN] == 0.10
        assert ensemble.base_weights[SignalSource.FACTOR] == 0.25
        assert ensemble.base_weights[SignalSource.MOMENTUM] == 0.10
        assert ensemble.base_weights[SignalSource.ALTERNATIVE_DATA] == 0.20
        assert ensemble.base_weights[SignalSource.CROSS_ASSET] == 0.15

    def test_default_regime_weights_exist(self):
        """Test that regime weights are initialized for all regimes."""
        ensemble = EnsemblePredictor()

        assert MarketRegime.BULL in ensemble.regime_weights
        assert MarketRegime.BEAR in ensemble.regime_weights
        assert MarketRegime.SIDEWAYS in ensemble.regime_weights
        assert MarketRegime.VOLATILE in ensemble.regime_weights

    def test_source_accuracy_initialized_for_all_sources(self):
        """Test that source accuracy tracking is initialized for all sources."""
        ensemble = EnsemblePredictor()

        for source in SignalSource:
            assert source in ensemble._source_accuracy
            assert ensemble._source_accuracy[source] == []


# =============================================================================
# SIGNAL SOURCE REGISTRATION TESTS
# =============================================================================


class TestRegisterSource:
    """Tests for register_source method."""

    def test_register_single_source(self, ensemble_predictor):
        """Test registering a single signal source."""
        signal_fn = create_mock_signal_fn(SignalSource.LSTM)

        ensemble_predictor.register_source(SignalSource.LSTM, signal_fn)

        assert SignalSource.LSTM in ensemble_predictor._signal_sources
        assert ensemble_predictor._signal_sources[SignalSource.LSTM] == signal_fn

    def test_register_multiple_sources(self, ensemble_predictor):
        """Test registering multiple signal sources."""
        lstm_fn = create_mock_signal_fn(SignalSource.LSTM)
        dqn_fn = create_mock_signal_fn(SignalSource.DQN)
        factor_fn = create_mock_signal_fn(SignalSource.FACTOR)

        ensemble_predictor.register_source(SignalSource.LSTM, lstm_fn)
        ensemble_predictor.register_source(SignalSource.DQN, dqn_fn)
        ensemble_predictor.register_source(SignalSource.FACTOR, factor_fn)

        assert len(ensemble_predictor._signal_sources) == 3

    def test_register_source_with_custom_weight(self, ensemble_predictor):
        """Test registering source with custom weight."""
        signal_fn = create_mock_signal_fn(SignalSource.LSTM)

        ensemble_predictor.register_source(SignalSource.LSTM, signal_fn, weight=0.5)

        assert ensemble_predictor.base_weights[SignalSource.LSTM] == 0.5

    def test_register_source_overrides_existing(self, ensemble_predictor):
        """Test that registering same source twice replaces the function."""
        signal_fn1 = create_mock_signal_fn(SignalSource.LSTM, signal_value=0.5)
        signal_fn2 = create_mock_signal_fn(SignalSource.LSTM, signal_value=0.9)

        ensemble_predictor.register_source(SignalSource.LSTM, signal_fn1)
        ensemble_predictor.register_source(SignalSource.LSTM, signal_fn2)

        assert ensemble_predictor._signal_sources[SignalSource.LSTM] == signal_fn2


# =============================================================================
# WEIGHT CALCULATION TESTS
# =============================================================================


class TestGetWeights:
    """Tests for _get_weights method."""

    def test_get_weights_without_regime_uses_base_weights(self, ensemble_predictor):
        """Test that base weights are used when no regime specified."""
        available = [SignalSource.LSTM, SignalSource.DQN]

        weights = ensemble_predictor._get_weights(regime=None, available_sources=available)

        # Weights should be normalized for available sources
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001  # Normalized to 1

    def test_get_weights_with_bull_regime(self, ensemble_predictor):
        """Test weights in BULL regime."""
        available = [SignalSource.LSTM, SignalSource.DQN, SignalSource.FACTOR, SignalSource.MOMENTUM]

        weights = ensemble_predictor._get_weights(
            regime=MarketRegime.BULL,
            available_sources=available
        )

        # In BULL regime, momentum should get 0.20 (from DEFAULT_REGIME_WEIGHTS)
        assert SignalSource.MOMENTUM in weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_get_weights_with_bear_regime(self, ensemble_predictor):
        """Test weights in BEAR regime."""
        available = [SignalSource.LSTM, SignalSource.DQN, SignalSource.FACTOR, SignalSource.MEAN_REVERSION]

        weights = ensemble_predictor._get_weights(
            regime=MarketRegime.BEAR,
            available_sources=available
        )

        # In BEAR regime, mean reversion should get 0.25
        assert SignalSource.MEAN_REVERSION in weights

    def test_get_weights_with_sideways_regime(self, ensemble_predictor):
        """Test weights in SIDEWAYS regime favor mean reversion."""
        available = [SignalSource.LSTM, SignalSource.DQN, SignalSource.FACTOR, SignalSource.MEAN_REVERSION]

        weights = ensemble_predictor._get_weights(
            regime=MarketRegime.SIDEWAYS,
            available_sources=available
        )

        # In SIDEWAYS, mean reversion has highest weight (0.40)
        # After normalization, should still be highest
        assert weights[SignalSource.MEAN_REVERSION] > weights[SignalSource.LSTM]

    def test_get_weights_with_volatile_regime(self, ensemble_predictor):
        """Test weights in VOLATILE regime."""
        available = [SignalSource.LSTM, SignalSource.DQN, SignalSource.FACTOR, SignalSource.MOMENTUM]

        weights = ensemble_predictor._get_weights(
            regime=MarketRegime.VOLATILE,
            available_sources=available
        )

        # In VOLATILE, DQN and FACTOR have highest weights (0.30 each)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_get_weights_filters_unavailable_sources(self, ensemble_predictor):
        """Test that only available sources get weights."""
        available = [SignalSource.LSTM, SignalSource.DQN]  # Only 2 sources

        weights = ensemble_predictor._get_weights(regime=None, available_sources=available)

        assert SignalSource.LSTM in weights
        assert SignalSource.DQN in weights
        assert SignalSource.FACTOR not in weights
        assert SignalSource.MOMENTUM not in weights

    def test_get_weights_normalizes_to_one(self, ensemble_predictor):
        """Test that weights are normalized to sum to 1."""
        for regime in [None, MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]:
            available = [SignalSource.LSTM, SignalSource.DQN]
            weights = ensemble_predictor._get_weights(regime=regime, available_sources=available)

            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"Failed for regime {regime}"


# =============================================================================
# PERFORMANCE-BASED WEIGHT ADJUSTMENT TESTS
# =============================================================================


class TestAdjustByPerformance:
    """Tests for _adjust_by_performance method."""

    def test_no_adjustment_with_insufficient_history(self, ensemble_predictor):
        """Test no adjustment when accuracy history is too short."""
        weights = {SignalSource.LSTM: 0.5, SignalSource.DQN: 0.5}

        # No accuracy history
        adjusted = ensemble_predictor._adjust_by_performance(weights)

        assert adjusted[SignalSource.LSTM] == 0.5
        assert adjusted[SignalSource.DQN] == 0.5

    def test_adjustment_requires_minimum_predictions(self, ensemble_predictor):
        """Test that adjustment only happens with sufficient predictions."""
        weights = {SignalSource.LSTM: 0.5, SignalSource.DQN: 0.5}

        # Add high accuracy but insufficient history (< 50 predictions)
        ensemble_predictor._source_accuracy[SignalSource.LSTM] = [1.0] * 10
        ensemble_predictor._source_accuracy[SignalSource.DQN] = [0.0] * 10

        adjusted = ensemble_predictor._adjust_by_performance(weights)

        # INSTITUTIONAL SAFETY: No adjustment should happen with < 50 predictions
        assert adjusted[SignalSource.LSTM] == weights[SignalSource.LSTM]
        assert adjusted[SignalSource.DQN] == weights[SignalSource.DQN]

    def test_adjustment_boosts_accurate_source_with_sufficient_data(self, ensemble_predictor):
        """Test that high accuracy boosts weight when sufficient data exists."""
        weights = {SignalSource.LSTM: 0.5, SignalSource.DQN: 0.5}

        # Add sufficient history (50+ predictions) with high accuracy for LSTM
        ensemble_predictor._source_accuracy[SignalSource.LSTM] = [1.0] * 60
        # Add sufficient history with low accuracy for DQN
        ensemble_predictor._source_accuracy[SignalSource.DQN] = [0.0] * 60

        adjusted = ensemble_predictor._adjust_by_performance(weights)

        # LSTM should have higher weight after adjustment
        assert adjusted[SignalSource.LSTM] > weights[SignalSource.LSTM]
        # DQN may be killed (all zeros is extremely bad)
        # If not killed, should have lower weight
        if adjusted[SignalSource.DQN] > 0:
            assert adjusted[SignalSource.DQN] < weights[SignalSource.DQN]

    def test_adjustment_with_neutral_accuracy(self, ensemble_predictor):
        """Test that 50% accuracy results in no adjustment (CI includes 0.5)."""
        weights = {SignalSource.LSTM: 0.5}

        # 50% accuracy with sufficient data
        # Bayesian approach: CI will include 0.5, so no adjustment
        ensemble_predictor._source_accuracy[SignalSource.LSTM] = [1.0, 0.0] * 30  # 60 total

        adjusted = ensemble_predictor._adjust_by_performance(weights)

        # With CI-based adjustment, neutral accuracy means no change
        # because the 95% CI for 50% accuracy includes 0.5
        assert abs(adjusted[SignalSource.LSTM] - weights[SignalSource.LSTM]) < 0.1

    def test_adjustment_factor_clamped_to_bounds(self, ensemble_predictor):
        """Test that performance factor is clamped between 0.2 and 2.0."""
        weights = {SignalSource.LSTM: 0.5}

        # Perfect accuracy
        ensemble_predictor._source_accuracy[SignalSource.LSTM] = [1.0] * 20

        adjusted = ensemble_predictor._adjust_by_performance(weights)

        # Should not exceed 2.0x original weight
        assert adjusted[SignalSource.LSTM] <= 1.0  # 0.5 * 2.0 = 1.0


# =============================================================================
# PREDICTION TESTS
# =============================================================================


class TestPredict:
    """Tests for predict method."""

    def test_predict_with_sufficient_sources(self, ensemble_with_sources):
        """Test prediction with sufficient signal sources."""
        prediction = ensemble_with_sources.predict("AAPL", {})

        assert prediction is not None
        assert prediction.symbol == "AAPL"
        assert isinstance(prediction.ensemble_signal, float)
        assert isinstance(prediction.ensemble_confidence, float)
        assert prediction.direction in ["long", "short", "neutral"]

    def test_predict_returns_none_with_insufficient_sources(self, ensemble_predictor):
        """Test that predict returns None when insufficient sources."""
        ensemble_predictor.min_sources_required = 2

        # Register only 1 source
        ensemble_predictor.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM),
        )

        prediction = ensemble_predictor.predict("AAPL", {})

        assert prediction is None

    def test_predict_handles_failed_source(self, ensemble_predictor):
        """Test prediction continues when one source fails."""
        ensemble_predictor.min_sources_required = 2

        # Register working sources
        ensemble_predictor.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM),
        )
        ensemble_predictor.register_source(
            SignalSource.DQN,
            create_mock_signal_fn(SignalSource.DQN),
        )
        # Register failing source
        ensemble_predictor.register_source(
            SignalSource.FACTOR,
            create_mock_signal_fn(SignalSource.FACTOR, should_raise=True),
        )

        prediction = ensemble_predictor.predict("AAPL", {})

        # Should still work with 2 sources
        assert prediction is not None
        assert SignalSource.FACTOR not in prediction.components

    def test_predict_handles_source_returning_none(self, ensemble_predictor):
        """Test prediction continues when source returns None."""
        ensemble_predictor.min_sources_required = 2

        # Register working sources
        ensemble_predictor.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM),
        )
        ensemble_predictor.register_source(
            SignalSource.DQN,
            create_mock_signal_fn(SignalSource.DQN),
        )
        # Register source that returns None
        ensemble_predictor.register_source(
            SignalSource.FACTOR,
            create_mock_signal_fn(SignalSource.FACTOR, should_return_none=True),
        )

        prediction = ensemble_predictor.predict("AAPL", {})

        assert prediction is not None
        assert SignalSource.FACTOR not in prediction.components

    def test_predict_direction_long_when_signal_positive(self, ensemble_predictor):
        """Test direction is 'long' when ensemble signal > 0.1."""
        ensemble_predictor.min_sources_required = 1

        ensemble_predictor.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=0.5, direction="long"),
        )

        prediction = ensemble_predictor.predict("AAPL", {})

        assert prediction.direction == "long"

    def test_predict_direction_short_when_signal_negative(self, ensemble_predictor):
        """Test direction is 'short' when ensemble signal < -0.1."""
        ensemble_predictor.min_sources_required = 1

        ensemble_predictor.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=-0.5, direction="short"),
        )

        prediction = ensemble_predictor.predict("AAPL", {})

        assert prediction.direction == "short"

    def test_predict_direction_neutral_when_signal_near_zero(self, ensemble_predictor):
        """Test direction is 'neutral' when ensemble signal between -0.1 and 0.1."""
        ensemble_predictor.min_sources_required = 1

        ensemble_predictor.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=0.05, direction="neutral"),
        )

        prediction = ensemble_predictor.predict("AAPL", {})

        assert prediction.direction == "neutral"

    def test_predict_stores_prediction_in_history(self, ensemble_with_sources):
        """Test that prediction is stored in history."""
        initial_count = len(ensemble_with_sources._predictions)

        ensemble_with_sources.predict("AAPL", {})

        assert len(ensemble_with_sources._predictions) == initial_count + 1

    def test_predict_with_regime(self, ensemble_with_sources):
        """Test prediction with market regime specified."""
        prediction = ensemble_with_sources.predict("AAPL", {}, regime=MarketRegime.BULL)

        assert prediction is not None
        assert prediction.regime == MarketRegime.BULL

    def test_predict_components_contain_source_signals(self, ensemble_with_sources):
        """Test that prediction components contain signals from each source."""
        prediction = ensemble_with_sources.predict("AAPL", {})

        assert SignalSource.LSTM in prediction.components
        assert SignalSource.DQN in prediction.components
        assert SignalSource.FACTOR in prediction.components

    def test_predict_weights_used_are_normalized(self, ensemble_with_sources):
        """Test that weights_used sum to 1."""
        prediction = ensemble_with_sources.predict("AAPL", {})

        total_weight = sum(prediction.weights_used.values())
        assert abs(total_weight - 1.0) < 0.001


# =============================================================================
# WEIGHTED COMBINE TESTS
# =============================================================================


class TestWeightedCombine:
    """Tests for _weighted_combine method."""

    def test_weighted_combine_single_source(self, ensemble_predictor):
        """Test weighted combine with single source."""
        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.8, confidence=0.9
            ),
        }
        weights = {SignalSource.LSTM: 1.0}

        signal, confidence = ensemble_predictor._weighted_combine(components, weights)

        # With single source and weight 1.0, signal should be signal * confidence / (weight * confidence)
        # = 0.8 * 0.9 * 0.9 / (1.0 * 0.9 * 0.9) = 0.8
        assert abs(signal - 0.8) < 0.01

    def test_weighted_combine_multiple_sources_agreeing(self, ensemble_predictor):
        """Test weighted combine when sources agree."""
        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.8, confidence=0.8
            ),
            SignalSource.DQN: create_signal_component(
                SignalSource.DQN, signal_value=0.7, confidence=0.8
            ),
        }
        weights = {SignalSource.LSTM: 0.5, SignalSource.DQN: 0.5}

        signal, confidence = ensemble_predictor._weighted_combine(components, weights)

        # Signal should be between 0.7 and 0.8
        assert 0.7 <= signal <= 0.8
        # Confidence should be high since sources agree
        assert confidence > 0.5

    def test_weighted_combine_sources_disagreeing_penalizes_confidence(self, ensemble_predictor):
        """Test that disagreeing sources reduce confidence."""
        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.9, confidence=0.8
            ),
            SignalSource.DQN: create_signal_component(
                SignalSource.DQN, signal_value=-0.9, confidence=0.8
            ),
        }
        weights = {SignalSource.LSTM: 0.5, SignalSource.DQN: 0.5}

        signal, confidence = ensemble_predictor._weighted_combine(components, weights)

        # Signal should be near 0 (canceling out)
        assert abs(signal) < 0.1
        # Confidence should be penalized due to high disagreement
        assert confidence < 0.7

    def test_weighted_combine_respects_weights(self, ensemble_predictor):
        """Test that heavier weights have more influence."""
        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=1.0, confidence=0.8
            ),
            SignalSource.DQN: create_signal_component(
                SignalSource.DQN, signal_value=0.0, confidence=0.8
            ),
        }

        # LSTM gets 90% weight
        weights = {SignalSource.LSTM: 0.9, SignalSource.DQN: 0.1}
        signal, _ = ensemble_predictor._weighted_combine(components, weights)

        # Signal should be closer to 1.0 (LSTM value)
        assert signal > 0.8

    def test_weighted_combine_with_zero_total_weight(self, ensemble_predictor):
        """Test weighted combine with zero total weight returns zeros."""
        components = {
            SignalSource.LSTM: create_signal_component(SignalSource.LSTM),
        }
        weights = {SignalSource.LSTM: 0.0}  # Zero weight

        signal, confidence = ensemble_predictor._weighted_combine(components, weights)

        assert signal == 0.0
        assert confidence == 0.0

    def test_weighted_combine_confidence_weighted_by_source_confidence(self, ensemble_predictor):
        """Test that source confidence affects effective weight."""
        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.8, confidence=0.9  # High confidence
            ),
            SignalSource.DQN: create_signal_component(
                SignalSource.DQN, signal_value=0.2, confidence=0.1  # Low confidence
            ),
        }
        weights = {SignalSource.LSTM: 0.5, SignalSource.DQN: 0.5}

        signal, _ = ensemble_predictor._weighted_combine(components, weights)

        # Signal should be closer to LSTM (0.8) due to higher confidence
        assert signal > 0.5

    def test_weighted_combine_disagreement_penalty_calculation(self, ensemble_predictor):
        """Test the disagreement penalty calculation."""
        # No disagreement
        components_agree = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.5, confidence=0.8
            ),
            SignalSource.DQN: create_signal_component(
                SignalSource.DQN, signal_value=0.5, confidence=0.8
            ),
        }
        weights = {SignalSource.LSTM: 0.5, SignalSource.DQN: 0.5}

        _, conf_agree = ensemble_predictor._weighted_combine(components_agree, weights)

        # High disagreement
        components_disagree = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=1.0, confidence=0.8
            ),
            SignalSource.DQN: create_signal_component(
                SignalSource.DQN, signal_value=-1.0, confidence=0.8
            ),
        }

        _, conf_disagree = ensemble_predictor._weighted_combine(components_disagree, weights)

        # Agreement should yield higher confidence
        assert conf_agree > conf_disagree


# =============================================================================
# META LEARNER TESTS
# =============================================================================


class TestMetaPredict:
    """Tests for _meta_predict method."""

    def test_meta_predict_falls_back_without_learner(self, ensemble_predictor):
        """Test meta_predict falls back to weighted combine without learner."""
        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.5, confidence=0.8
            ),
        }

        signal, confidence = ensemble_predictor._meta_predict(components)

        # Should return valid values from fallback
        assert isinstance(signal, float)
        assert isinstance(confidence, float)

    def test_meta_predict_with_learner(self, ensemble_predictor):
        """Test meta_predict with trained meta-learner."""
        # Create mock meta-learner
        mock_learner = MagicMock()
        mock_learner.predict.return_value = np.array([0.7])
        mock_learner.predict_proba.return_value = np.array([[0.2, 0.8]])
        ensemble_predictor._meta_learner = mock_learner

        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.5, confidence=0.8
            ),
        }

        signal, confidence = ensemble_predictor._meta_predict(components)

        assert signal == 0.7
        assert confidence == 0.8

    def test_meta_predict_handles_learner_error(self, ensemble_predictor):
        """Test meta_predict handles learner errors gracefully."""
        # Create failing meta-learner
        mock_learner = MagicMock()
        mock_learner.predict.side_effect = ValueError("Prediction failed")
        ensemble_predictor._meta_learner = mock_learner

        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.5, confidence=0.8
            ),
        }

        # Should fall back to weighted combine
        signal, confidence = ensemble_predictor._meta_predict(components)

        assert isinstance(signal, float)
        assert isinstance(confidence, float)

    def test_meta_predict_builds_correct_feature_vector(self, ensemble_predictor):
        """Test that meta_predict builds correct feature vector."""
        mock_learner = MagicMock()
        mock_learner.predict.return_value = np.array([0.5])
        mock_learner.predict_proba.return_value = np.array([[0.5, 0.5]])
        ensemble_predictor._meta_learner = mock_learner

        components = {
            SignalSource.LSTM: create_signal_component(
                SignalSource.LSTM, signal_value=0.7, confidence=0.9
            ),
        }

        ensemble_predictor._meta_predict(components)

        # Verify predict was called
        mock_learner.predict.assert_called_once()

        # Check feature vector shape (2 features per SignalSource)
        features = mock_learner.predict.call_args[0][0]
        assert features.shape[1] == len(SignalSource) * 2


# =============================================================================
# TRAIN META LEARNER TESTS
# =============================================================================


class TestTrainMetaLearner:
    """Tests for train_meta_learner method."""

    def test_train_meta_learner_logistic(self, ensemble_predictor):
        """Test training logistic regression meta-learner."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        ensemble_predictor.train_meta_learner(X, y, model_type="logistic")

        assert ensemble_predictor._meta_learner is not None

    def test_train_meta_learner_random_forest(self, ensemble_predictor):
        """Test training random forest meta-learner."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        ensemble_predictor.train_meta_learner(X, y, model_type="random_forest")

        assert ensemble_predictor._meta_learner is not None

    def test_train_meta_learner_default_is_logistic(self, ensemble_predictor):
        """Test that unknown model type defaults to logistic regression."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        ensemble_predictor.train_meta_learner(X, y, model_type="unknown_type")

        assert ensemble_predictor._meta_learner is not None


# =============================================================================
# RECORD OUTCOME TESTS
# =============================================================================


class TestRecordOutcome:
    """Tests for record_outcome method."""

    def test_record_outcome_updates_accuracy_for_correct_long(self, ensemble_with_sources):
        """Test accuracy updates when long prediction is correct."""
        # Make a prediction
        prediction = ensemble_with_sources.predict("AAPL", {})
        prediction_time = datetime.now()

        # Record positive outcome for long prediction
        ensemble_with_sources.record_outcome("AAPL", prediction_time, actual_return=0.05)

        # Check accuracy was updated
        for source in prediction.components:
            assert len(ensemble_with_sources._source_accuracy[source]) > 0

    def test_record_outcome_updates_accuracy_for_incorrect_long(self, ensemble_with_sources):
        """Test accuracy updates when long prediction is wrong."""
        prediction = ensemble_with_sources.predict("AAPL", {})
        prediction_time = datetime.now()

        # Record negative outcome for long prediction (wrong)
        ensemble_with_sources.record_outcome("AAPL", prediction_time, actual_return=-0.05)

        # Should record 0.0 (incorrect)
        for source in prediction.components:
            if ensemble_with_sources._source_accuracy[source]:
                assert ensemble_with_sources._source_accuracy[source][-1] == 0.0

    def test_record_outcome_updates_accuracy_for_correct_short(self, ensemble_predictor):
        """Test accuracy updates when short prediction is correct."""
        ensemble_predictor.min_sources_required = 1
        ensemble_predictor.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=-0.8, direction="short"),
        )

        ensemble_predictor.predict("AAPL", {})
        prediction_time = datetime.now()

        # Record negative return (correct for short)
        ensemble_predictor.record_outcome("AAPL", prediction_time, actual_return=-0.05)

        # Should record 1.0 (correct)
        assert ensemble_predictor._source_accuracy[SignalSource.LSTM][-1] == 1.0

    def test_record_outcome_no_match_does_nothing(self, ensemble_with_sources):
        """Test that unmatched outcome doesn't update accuracy."""
        initial_lengths = {
            source: len(acc) for source, acc in ensemble_with_sources._source_accuracy.items()
        }

        # Record outcome without matching prediction (wrong time)
        ensemble_with_sources.record_outcome(
            "AAPL",
            datetime.now() - timedelta(hours=5),  # Too old
            actual_return=0.05,
        )

        # Accuracy should not have changed
        for source, length in initial_lengths.items():
            assert len(ensemble_with_sources._source_accuracy[source]) == length

    def test_record_outcome_trims_history_to_100(self, ensemble_with_sources):
        """Test that accuracy history is trimmed to 100 entries."""
        # Make multiple predictions and record outcomes
        for _i in range(110):
            ensemble_with_sources.predict("AAPL", {})
            prediction_time = datetime.now()
            ensemble_with_sources.record_outcome("AAPL", prediction_time, actual_return=0.01)

        # Check that history doesn't exceed 100
        for source in SignalSource:
            assert len(ensemble_with_sources._source_accuracy[source]) <= 100


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestGetSourcePerformance:
    """Tests for get_source_performance method."""

    def test_get_source_performance_empty_when_insufficient_history(self, ensemble_predictor):
        """Test returns empty when insufficient accuracy history."""
        performance = ensemble_predictor.get_source_performance()

        assert performance == {}

    def test_get_source_performance_with_history(self, ensemble_predictor):
        """Test returns performance when sufficient history."""
        # Add accuracy history
        ensemble_predictor._source_accuracy[SignalSource.LSTM] = [1.0, 0.0, 1.0, 1.0, 0.0]
        ensemble_predictor.base_weights[SignalSource.LSTM] = 0.3

        performance = ensemble_predictor.get_source_performance()

        assert "lstm" in performance
        # New API uses 'accuracy_raw' and 'accuracy_bayesian'
        assert "accuracy_raw" in performance["lstm"]
        assert "accuracy_bayesian" in performance["lstm"]
        assert "credibility_interval" in performance["lstm"]
        assert "recent_accuracy" in performance["lstm"]
        assert "n_predictions" in performance["lstm"]
        assert "current_weight" in performance["lstm"]
        assert "is_killed" in performance["lstm"]
        assert "min_predictions_met" in performance["lstm"]

    def test_get_source_performance_calculates_accuracy(self, ensemble_predictor):
        """Test accuracy calculation is correct."""
        # 60% accuracy
        ensemble_predictor._source_accuracy[SignalSource.LSTM] = [1.0, 1.0, 1.0, 0.0, 0.0]
        ensemble_predictor.base_weights[SignalSource.LSTM] = 0.3

        performance = ensemble_predictor.get_source_performance()

        # Now uses 'accuracy_raw' for the raw mean
        assert performance["lstm"]["accuracy_raw"] == 0.6
        assert performance["lstm"]["n_predictions"] == 5
        # Bayesian accuracy is regularized toward 0.5
        # With prior Beta(5,5) and data (3 successes, 2 failures), posterior is Beta(8, 7)
        # Mean = 8/15 ≈ 0.533
        assert 0.5 < performance["lstm"]["accuracy_bayesian"] < 0.6


class TestGetEnsembleStatistics:
    """Tests for get_ensemble_statistics method."""

    def test_get_ensemble_statistics_empty_predictions(self, ensemble_predictor):
        """Test returns error when no predictions made."""
        stats = ensemble_predictor.get_ensemble_statistics()

        assert "error" in stats
        assert stats["error"] == "No predictions made yet"

    def test_get_ensemble_statistics_with_predictions(self, ensemble_with_sources):
        """Test returns full statistics with predictions."""
        # Make a prediction
        ensemble_with_sources.predict("AAPL", {})

        stats = ensemble_with_sources.get_ensemble_statistics()

        assert "total_predictions" in stats
        assert "sources_registered" in stats
        assert "source_performance" in stats
        assert "has_meta_learner" in stats
        assert "base_weights" in stats

        assert stats["total_predictions"] == 1
        assert stats["sources_registered"] == 3  # LSTM, DQN, FACTOR
        assert stats["has_meta_learner"] is False


# =============================================================================
# CREATE ENSEMBLE FROM COMPONENTS TESTS
# =============================================================================


class TestCreateEnsembleFromComponents:
    """Tests for create_ensemble_from_components helper function."""

    def test_create_ensemble_no_components(self):
        """Test creating ensemble with no components."""
        ensemble = create_ensemble_from_components()

        assert isinstance(ensemble, EnsemblePredictor)
        assert len(ensemble._signal_sources) == 0

    def test_create_ensemble_with_lstm_predictor(self):
        """Test creating ensemble with LSTM predictor."""
        mock_lstm = MagicMock()
        mock_result = MagicMock()
        mock_result.predicted_direction = "up"
        mock_result.confidence = 0.8
        mock_result.std_prediction = 0.1
        mock_lstm.predict_with_uncertainty.return_value = mock_result

        ensemble = create_ensemble_from_components(lstm_predictor=mock_lstm)

        assert SignalSource.LSTM in ensemble._signal_sources

    def test_create_ensemble_with_dqn_agent(self):
        """Test creating ensemble with DQN agent."""
        mock_dqn = MagicMock()
        mock_dqn.act.return_value = (1, 0.9)  # Buy action with confidence

        ensemble = create_ensemble_from_components(dqn_agent=mock_dqn)

        assert SignalSource.DQN in ensemble._signal_sources

    def test_create_ensemble_with_factor_model(self):
        """Test creating ensemble with factor model."""
        mock_factor = MagicMock()
        mock_factor.get_signal.return_value = {
            "action": "long",
            "composite_z": 1.5,
            "confidence": 0.7,
            "factor_breakdown": {},
        }

        ensemble = create_ensemble_from_components(factor_model=mock_factor)

        assert SignalSource.FACTOR in ensemble._signal_sources

    def test_create_ensemble_with_momentum_strategy(self):
        """Test creating ensemble with momentum strategy."""
        mock_momentum = MagicMock()
        mock_momentum.analyze_symbol.return_value = {
            "action": "buy",
            "confidence": 0.8,
        }

        ensemble = create_ensemble_from_components(momentum_strategy=mock_momentum)

        assert SignalSource.MOMENTUM in ensemble._signal_sources

    def test_create_ensemble_with_all_components(self):
        """Test creating ensemble with all components."""
        mock_lstm = MagicMock()
        mock_dqn = MagicMock()
        mock_factor = MagicMock()
        mock_momentum = MagicMock()

        mock_lstm.predict_with_uncertainty.return_value = MagicMock(
            predicted_direction="up", confidence=0.8, std_prediction=0.1
        )
        mock_dqn.act.return_value = (1, 0.9)
        mock_factor.get_signal.return_value = {"action": "long", "composite_z": 1.5, "confidence": 0.7}
        mock_momentum.analyze_symbol.return_value = {"action": "buy", "confidence": 0.8}

        ensemble = create_ensemble_from_components(
            lstm_predictor=mock_lstm,
            dqn_agent=mock_dqn,
            factor_model=mock_factor,
            momentum_strategy=mock_momentum,
        )

        assert len(ensemble._signal_sources) == 4

    def test_lstm_signal_function_handles_none_result(self):
        """Test LSTM signal function handles None result."""
        mock_lstm = MagicMock()
        mock_lstm.predict_with_uncertainty.return_value = None

        ensemble = create_ensemble_from_components(lstm_predictor=mock_lstm)

        # Call the registered function
        signal = ensemble._signal_sources[SignalSource.LSTM]("AAPL", {"prices": []})

        assert signal is None

    def test_dqn_signal_function_handles_no_state(self):
        """Test DQN signal function handles missing state."""
        mock_dqn = MagicMock()

        ensemble = create_ensemble_from_components(dqn_agent=mock_dqn)

        # Call with no state
        signal = ensemble._signal_sources[SignalSource.DQN]("AAPL", {})

        assert signal is None

    def test_dqn_signal_function_maps_actions_correctly(self):
        """Test DQN signal function maps actions to correct directions."""
        mock_dqn = MagicMock()

        # Test hold action (0)
        mock_dqn.act.return_value = (0, 0.5)
        ensemble = create_ensemble_from_components(dqn_agent=mock_dqn)
        signal = ensemble._signal_sources[SignalSource.DQN]("AAPL", {"state": [1, 2, 3]})
        assert signal.direction == "neutral"
        assert signal.signal_value == 0.0

        # Test buy action (1)
        mock_dqn.act.return_value = (1, 0.8)
        signal = ensemble._signal_sources[SignalSource.DQN]("AAPL", {"state": [1, 2, 3]})
        assert signal.direction == "long"
        assert signal.signal_value == 1.0

        # Test sell action (2)
        mock_dqn.act.return_value = (2, 0.9)
        signal = ensemble._signal_sources[SignalSource.DQN]("AAPL", {"state": [1, 2, 3]})
        assert signal.direction == "short"
        assert signal.signal_value == -1.0

    def test_factor_signal_function_handles_hold_action(self):
        """Test factor signal function returns None for hold."""
        mock_factor = MagicMock()
        mock_factor.get_signal.return_value = {"action": "hold"}

        ensemble = create_ensemble_from_components(factor_model=mock_factor)
        signal = ensemble._signal_sources[SignalSource.FACTOR]("AAPL", {"factor_scores": {}})

        assert signal is None

    def test_momentum_signal_function_handles_hold_action(self):
        """Test momentum signal function handles hold action."""
        mock_momentum = MagicMock()
        mock_momentum.analyze_symbol.return_value = {"action": "hold"}

        ensemble = create_ensemble_from_components(momentum_strategy=mock_momentum)
        signal = ensemble._signal_sources[SignalSource.MOMENTUM]("AAPL", {})

        assert signal.direction == "neutral"
        assert signal.signal_value == 0.0

    def test_momentum_signal_function_handles_exception(self):
        """Test momentum signal function handles exceptions."""
        mock_momentum = MagicMock()
        mock_momentum.analyze_symbol.side_effect = ValueError("Error")

        ensemble = create_ensemble_from_components(momentum_strategy=mock_momentum)
        signal = ensemble._signal_sources[SignalSource.MOMENTUM]("AAPL", {})

        assert signal is None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEnsembleIntegration:
    """Integration tests for full ensemble workflow."""

    def test_full_prediction_workflow(self):
        """Test complete workflow: register, predict, record, check stats."""
        ensemble = EnsemblePredictor(min_sources_required=2)

        # Register sources
        ensemble.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=0.7, confidence=0.8),
        )
        ensemble.register_source(
            SignalSource.DQN,
            create_mock_signal_fn(SignalSource.DQN, signal_value=0.6, confidence=0.7),
        )

        # Make prediction
        prediction = ensemble.predict("AAPL", {}, regime=MarketRegime.BULL)

        assert prediction is not None
        assert prediction.should_trade(min_confidence=0.5)

        # Record outcome
        ensemble.record_outcome("AAPL", datetime.now(), actual_return=0.05)

        # Check statistics
        stats = ensemble.get_ensemble_statistics()

        assert stats["total_predictions"] == 1
        assert stats["sources_registered"] == 2

    def test_multi_symbol_predictions(self):
        """Test predictions across multiple symbols."""
        ensemble = EnsemblePredictor(min_sources_required=2)

        ensemble.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=0.5, confidence=0.7),
        )
        ensemble.register_source(
            SignalSource.DQN,
            create_mock_signal_fn(SignalSource.DQN, signal_value=0.5, confidence=0.7),
        )

        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        predictions = {}

        for symbol in symbols:
            predictions[symbol] = ensemble.predict(symbol, {})

        assert len(predictions) == 4
        assert all(p is not None for p in predictions.values())

    def test_regime_switching_affects_weights(self):
        """Test that regime changes affect weight distribution."""
        ensemble = EnsemblePredictor(min_sources_required=1)

        ensemble.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=0.5, confidence=0.7),
        )
        ensemble.register_source(
            SignalSource.MEAN_REVERSION,
            create_mock_signal_fn(SignalSource.MEAN_REVERSION, signal_value=0.5, confidence=0.7),
        )

        # Get weights in different regimes
        bull_weights = ensemble._get_weights(
            MarketRegime.BULL,
            [SignalSource.LSTM, SignalSource.MEAN_REVERSION]
        )
        sideways_weights = ensemble._get_weights(
            MarketRegime.SIDEWAYS,
            [SignalSource.LSTM, SignalSource.MEAN_REVERSION]
        )

        # In sideways, mean reversion should have higher weight
        assert sideways_weights.get(SignalSource.MEAN_REVERSION, 0) > bull_weights.get(SignalSource.MEAN_REVERSION, 0)

    def test_performance_tracking_updates_weights_over_time(self):
        """Test that performance tracking influences weights."""
        ensemble = EnsemblePredictor(
            min_sources_required=2,
            use_performance_weighting=True,
        )

        ensemble.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=0.5, confidence=0.7),
        )
        ensemble.register_source(
            SignalSource.DQN,
            create_mock_signal_fn(SignalSource.DQN, signal_value=0.5, confidence=0.7),
        )

        # Simulate LSTM being more accurate
        ensemble._source_accuracy[SignalSource.LSTM] = [1.0] * 10  # 100% accurate
        ensemble._source_accuracy[SignalSource.DQN] = [0.0] * 10   # 0% accurate

        weights = ensemble._get_weights(
            regime=None,
            available_sources=[SignalSource.LSTM, SignalSource.DQN]
        )

        # LSTM should have higher weight due to performance
        assert weights[SignalSource.LSTM] > weights[SignalSource.DQN]


# =============================================================================
# ALTERNATIVE DATA INTEGRATION TESTS
# =============================================================================


class TestAlternativeDataSignalSource:
    """Tests for ALTERNATIVE_DATA signal source in ensemble."""

    def test_alternative_data_in_signal_source_enum(self):
        """Test that ALTERNATIVE_DATA is a valid SignalSource."""
        assert hasattr(SignalSource, "ALTERNATIVE_DATA")
        assert SignalSource.ALTERNATIVE_DATA.value == "alternative_data"

    def test_alternative_data_in_default_weights(self):
        """Test that ALTERNATIVE_DATA has default weight."""
        ensemble = EnsemblePredictor()
        assert SignalSource.ALTERNATIVE_DATA in ensemble.base_weights
        assert ensemble.base_weights[SignalSource.ALTERNATIVE_DATA] > 0

    def test_alternative_data_in_regime_weights(self):
        """Test that ALTERNATIVE_DATA has weights in all regimes."""
        ensemble = EnsemblePredictor()

        for regime in MarketRegime:
            assert SignalSource.ALTERNATIVE_DATA in ensemble.regime_weights[regime]
            assert ensemble.regime_weights[regime][SignalSource.ALTERNATIVE_DATA] > 0

    def test_register_alternative_data_source(self):
        """Test registering an alternative data signal source."""
        ensemble = EnsemblePredictor()

        alt_data_fn = create_mock_signal_fn(
            SignalSource.ALTERNATIVE_DATA,
            signal_value=0.6,
            confidence=0.75,
            direction="long",
        )

        ensemble.register_source(SignalSource.ALTERNATIVE_DATA, alt_data_fn)

        assert SignalSource.ALTERNATIVE_DATA in ensemble._signal_sources

    def test_predict_with_alternative_data(self):
        """Test ensemble prediction including alternative data."""
        ensemble = EnsemblePredictor(min_sources_required=2)

        # Register traditional sources
        ensemble.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=0.5, confidence=0.7),
        )
        ensemble.register_source(
            SignalSource.FACTOR,
            create_mock_signal_fn(SignalSource.FACTOR, signal_value=0.4, confidence=0.6),
        )

        # Register alternative data
        ensemble.register_source(
            SignalSource.ALTERNATIVE_DATA,
            create_mock_signal_fn(
                SignalSource.ALTERNATIVE_DATA,
                signal_value=0.7,
                confidence=0.8,
                direction="long",
            ),
        )

        prediction = ensemble.predict("AAPL", {})

        assert prediction is not None
        assert SignalSource.ALTERNATIVE_DATA in prediction.components
        assert SignalSource.ALTERNATIVE_DATA in prediction.weights_used

    def test_alternative_data_weight_varies_by_regime(self):
        """Test that alternative data weight varies across regimes."""
        ensemble = EnsemblePredictor()

        bull_weight = ensemble.regime_weights[MarketRegime.BULL][SignalSource.ALTERNATIVE_DATA]
        bear_weight = ensemble.regime_weights[MarketRegime.BEAR][SignalSource.ALTERNATIVE_DATA]
        ensemble.regime_weights[MarketRegime.VOLATILE][SignalSource.ALTERNATIVE_DATA]

        # Alt data should have different weights in different regimes
        # In bear markets, alt data is weighted less (sentiment can mislead)
        assert bear_weight < bull_weight

    def test_alternative_data_contributes_to_ensemble_signal(self):
        """Test that alternative data signal contributes to final signal."""
        ensemble = EnsemblePredictor(min_sources_required=2)

        # Traditional source with neutral signal
        ensemble.register_source(
            SignalSource.LSTM,
            create_mock_signal_fn(SignalSource.LSTM, signal_value=0.0, confidence=0.8),
        )

        # Alternative data with strong bullish signal
        ensemble.register_source(
            SignalSource.ALTERNATIVE_DATA,
            create_mock_signal_fn(
                SignalSource.ALTERNATIVE_DATA,
                signal_value=0.9,
                confidence=0.85,
            ),
        )

        prediction = ensemble.predict("AAPL", {})

        # Final signal should be positive due to alt data influence
        assert prediction.ensemble_signal > 0

    def test_alternative_data_accuracy_tracked(self):
        """Test that alternative data accuracy is tracked."""
        ensemble = EnsemblePredictor()

        assert SignalSource.ALTERNATIVE_DATA in ensemble._source_accuracy
        assert ensemble._source_accuracy[SignalSource.ALTERNATIVE_DATA] == []

        # Record some accuracy
        ensemble._source_accuracy[SignalSource.ALTERNATIVE_DATA].append(1.0)
        ensemble._source_accuracy[SignalSource.ALTERNATIVE_DATA].append(1.0)
        ensemble._source_accuracy[SignalSource.ALTERNATIVE_DATA].append(0.0)

        assert len(ensemble._source_accuracy[SignalSource.ALTERNATIVE_DATA]) == 3

    def test_regime_weights_sum_to_one(self):
        """Test that regime weights still sum to approximately 1.0."""
        ensemble = EnsemblePredictor()

        for regime, weights in ensemble.regime_weights.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"Regime {regime} weights sum to {total}"
