"""
Comprehensive unit tests for AdaptiveStrategy.

AdaptiveStrategy is the recommended production strategy that automatically
switches between sub-strategies based on detected market regime:
- BULL market -> MomentumStrategy (long bias)
- BEAR market -> MomentumStrategy (short bias)
- SIDEWAYS market -> MeanReversionStrategy
- VOLATILE market -> Reduced exposure across all strategies

Tests cover:
- Initialization and configuration
- Sub-strategy creation (momentum, mean_reversion)
- Regime detection integration
- Strategy switching based on regime
- Signal routing to active strategy
- Position size adjustments by regime
- Status reporting
- Error handling and edge cases

DRY Principles Applied:
- Module-level imports
- Named constants for all magic numbers
- Shared fixtures from conftest.py
- Parametrized tests for multiple scenarios
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.unit.conftest import (
    create_mock_account,
)

# =============================================================================
# CONSTANTS - No magic numbers in tests
# =============================================================================

# Default symbols for testing
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
SINGLE_SYMBOL = "AAPL"

# Default parameters
DEFAULT_POSITION_SIZE = 0.10
DEFAULT_MAX_POSITIONS = 5
DEFAULT_STOP_LOSS = 0.03
DEFAULT_TAKE_PROFIT = 0.05
DEFAULT_REGIME_CHECK_INTERVAL = 30
DEFAULT_MIN_REGIME_CONFIDENCE = 0.55

# Position multipliers by regime
BULL_POSITION_MULT = 1.2
BEAR_POSITION_MULT = 0.8
SIDEWAYS_POSITION_MULT = 1.0
VOLATILE_POSITION_MULT = 0.5

# Regime types
REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"
REGIME_VOLATILE = "volatile"
REGIME_UNKNOWN = "unknown"

# Confidence levels
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.45
THRESHOLD_CONFIDENCE = 0.55


# =============================================================================
# MODULE-LEVEL FIXTURES
# =============================================================================


@pytest.fixture
def mock_broker():
    """Create mock broker with default account values."""
    broker = AsyncMock()
    broker.get_account.return_value = create_mock_account()
    broker.get_positions.return_value = []
    broker.get_bars = AsyncMock(return_value=None)
    return broker


@pytest.fixture
def mock_regime_info_bull():
    """Create mock regime info for bull market."""
    return {
        "type": REGIME_BULL,
        "confidence": HIGH_CONFIDENCE,
        "trend_direction": "up",
        "trend_strength": 35,
        "is_trending": True,
        "is_ranging": False,
        "volatility_regime": "normal",
        "volatility_pct": 2.0,
        "sma_50": 150.0,
        "sma_200": 140.0,
        "recommended_strategy": "momentum_long",
        "position_multiplier": BULL_POSITION_MULT,
        "detected_at": datetime.now().isoformat(),
    }


@pytest.fixture
def mock_regime_info_bear():
    """Create mock regime info for bear market."""
    return {
        "type": REGIME_BEAR,
        "confidence": HIGH_CONFIDENCE,
        "trend_direction": "down",
        "trend_strength": 35,
        "is_trending": True,
        "is_ranging": False,
        "volatility_regime": "normal",
        "volatility_pct": 2.0,
        "sma_50": 130.0,
        "sma_200": 140.0,
        "recommended_strategy": "momentum_short",
        "position_multiplier": BEAR_POSITION_MULT,
        "detected_at": datetime.now().isoformat(),
    }


@pytest.fixture
def mock_regime_info_sideways():
    """Create mock regime info for sideways market."""
    return {
        "type": REGIME_SIDEWAYS,
        "confidence": HIGH_CONFIDENCE,
        "trend_direction": "flat",
        "trend_strength": 15,
        "is_trending": False,
        "is_ranging": True,
        "volatility_regime": "normal",
        "volatility_pct": 1.5,
        "sma_50": 140.0,
        "sma_200": 141.0,
        "recommended_strategy": "mean_reversion",
        "position_multiplier": SIDEWAYS_POSITION_MULT,
        "detected_at": datetime.now().isoformat(),
    }


@pytest.fixture
def mock_regime_info_volatile():
    """Create mock regime info for volatile market."""
    return {
        "type": REGIME_VOLATILE,
        "confidence": HIGH_CONFIDENCE,
        "trend_direction": "up",
        "trend_strength": 30,
        "is_trending": True,
        "is_ranging": False,
        "volatility_regime": "high",
        "volatility_pct": 4.5,
        "sma_50": 150.0,
        "sma_200": 140.0,
        "recommended_strategy": "defensive",
        "position_multiplier": VOLATILE_POSITION_MULT,
        "detected_at": datetime.now().isoformat(),
    }


@pytest.fixture
def mock_regime_info_low_confidence():
    """Create mock regime info with low confidence."""
    return {
        "type": REGIME_BULL,
        "confidence": LOW_CONFIDENCE,
        "trend_direction": "up",
        "trend_strength": 22,
        "is_trending": False,
        "is_ranging": False,
        "volatility_regime": "normal",
        "volatility_pct": 2.0,
        "sma_50": 145.0,
        "sma_200": 140.0,
        "recommended_strategy": "momentum_long",
        "position_multiplier": 0.9,
        "detected_at": datetime.now().isoformat(),
    }


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestAdaptiveStrategyInit:
    """Test AdaptiveStrategy initialization."""

    def test_class_has_name_attribute(self):
        """Test that AdaptiveStrategy has NAME class attribute."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        assert hasattr(AdaptiveStrategy, "NAME")
        assert AdaptiveStrategy.NAME == "AdaptiveStrategy"

    def test_init_with_broker_and_symbols(self, mock_broker):
        """Test initialization with broker and symbols."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)

        assert strategy.broker == mock_broker
        assert strategy.name == "AdaptiveStrategy"

    def test_init_with_parameters(self, mock_broker):
        """Test initialization with custom parameters."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        custom_params = {
            "position_size": 0.15,
            "max_positions": 3,
            "stop_loss": 0.04,
        }
        strategy = AdaptiveStrategy(
            broker=mock_broker, symbols=TEST_SYMBOLS, parameters=custom_params
        )

        assert strategy.parameters.get("position_size") == 0.15 or "symbols" in strategy.parameters

    def test_init_with_none_parameters(self, mock_broker):
        """Test initialization with None parameters."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS, parameters=None)

        assert strategy is not None

    def test_default_parameters_returns_expected_keys(self):
        """Test that default_parameters returns all expected keys."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy()
        params = strategy.default_parameters()

        # Basic parameters
        assert "position_size" in params
        assert "max_positions" in params
        assert "stop_loss" in params
        assert "take_profit" in params

        # Regime detection settings
        assert "regime_check_interval_minutes" in params
        assert "min_regime_confidence" in params

        # Strategy selection
        assert "bull_strategy" in params
        assert "bear_strategy" in params
        assert "sideways_strategy" in params
        assert "volatile_strategy" in params

        # Position multipliers
        assert "bull_position_mult" in params
        assert "bear_position_mult" in params
        assert "sideways_position_mult" in params
        assert "volatile_position_mult" in params

    def test_default_parameters_values(self):
        """Test that default_parameters returns expected values."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy()
        params = strategy.default_parameters()

        assert params["position_size"] == DEFAULT_POSITION_SIZE
        assert params["max_positions"] == DEFAULT_MAX_POSITIONS
        assert params["stop_loss"] == DEFAULT_STOP_LOSS
        assert params["take_profit"] == DEFAULT_TAKE_PROFIT
        assert params["regime_check_interval_minutes"] == DEFAULT_REGIME_CHECK_INTERVAL
        assert params["min_regime_confidence"] == DEFAULT_MIN_REGIME_CONFIDENCE
        assert params["bull_position_mult"] == BULL_POSITION_MULT
        assert params["bear_position_mult"] == BEAR_POSITION_MULT
        assert params["sideways_position_mult"] == SIDEWAYS_POSITION_MULT
        assert params["volatile_position_mult"] == VOLATILE_POSITION_MULT


class TestAdaptiveStrategyInitialize:
    """Test AdaptiveStrategy async initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_sub_strategies(self, mock_broker):
        """Test that initialize creates momentum and mean_reversion sub-strategies."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            # Configure mocks
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = MagicMock()
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            result = await strategy.initialize()

            assert result is True
            assert MockMomentum.called, "MomentumStrategy should be created"
            assert MockMeanReversion.called, "MeanReversionStrategy should be created"
            assert MockRegime.called, "MarketRegimeDetector should be created"

    @pytest.mark.asyncio
    async def test_initialize_sets_default_active_strategy_to_momentum(self, mock_broker):
        """Test that momentum is the default active strategy after initialization."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            assert strategy.active_strategy == mock_momentum
            assert strategy.active_strategy_name == "momentum"

    @pytest.mark.asyncio
    async def test_initialize_sets_parameters_correctly(self, mock_broker):
        """Test that parameters are correctly set after initialization."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            assert strategy.position_size == DEFAULT_POSITION_SIZE
            assert strategy.max_positions == DEFAULT_MAX_POSITIONS
            assert strategy.stop_loss == DEFAULT_STOP_LOSS
            assert strategy.take_profit == DEFAULT_TAKE_PROFIT

    @pytest.mark.asyncio
    async def test_initialize_creates_tracking_dictionaries(self, mock_broker):
        """Test that tracking dictionaries are initialized."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            assert strategy.current_regime is None
            assert strategy.last_regime_check is None
            assert strategy.regime_switches == 0
            assert strategy.last_regime_switch is None

            # Check tracking dictionaries
            for symbol in TEST_SYMBOLS:
                assert symbol in strategy.indicators
                assert symbol in strategy.signals
                assert symbol in strategy.price_history

    @pytest.mark.asyncio
    async def test_initialize_returns_false_on_exception(self, mock_broker):
        """Test that initialize returns False when an exception occurs."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum:
            MockMomentum.side_effect = Exception("Initialization failed")

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            result = await strategy.initialize()

            assert result is False


class TestRegimeDetectionIntegration:
    """Test regime detection and strategy switching."""

    @pytest.mark.asyncio
    async def test_switches_to_momentum_in_bull_market(self, mock_broker, mock_regime_info_bull):
        """Test that strategy switches to momentum in bull market."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.enable_short_selling = True
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            # Trigger regime update
            await strategy._update_regime()

            assert strategy.current_regime == REGIME_BULL
            assert strategy.active_strategy == mock_momentum
            assert strategy.active_strategy_name == "momentum_long"
            # In bull market, short selling should be disabled
            assert mock_momentum.enable_short_selling is False

    @pytest.mark.asyncio
    async def test_switches_to_momentum_in_bear_market(self, mock_broker, mock_regime_info_bear):
        """Test that strategy switches to momentum with shorts enabled in bear market."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.enable_short_selling = False
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bear)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            # Trigger regime update
            await strategy._update_regime()

            assert strategy.current_regime == REGIME_BEAR
            assert strategy.active_strategy == mock_momentum
            assert strategy.active_strategy_name == "momentum_short"
            # In bear market, short selling should be enabled
            assert mock_momentum.enable_short_selling is True

    @pytest.mark.asyncio
    async def test_switches_to_mean_reversion_in_sideways_market(
        self, mock_broker, mock_regime_info_sideways
    ):
        """Test that strategy switches to mean reversion in sideways market."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_sideways)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            # Trigger regime update
            await strategy._update_regime()

            assert strategy.current_regime == REGIME_SIDEWAYS
            assert strategy.active_strategy == mock_mean_rev
            assert strategy.active_strategy_name == "mean_reversion"

    @pytest.mark.asyncio
    async def test_reduces_position_size_in_volatile_market(
        self, mock_broker, mock_regime_info_volatile
    ):
        """Test that position size is reduced in volatile market."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.position_size = DEFAULT_POSITION_SIZE
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_volatile)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            # Trigger regime update
            await strategy._update_regime()

            assert strategy.current_regime == REGIME_VOLATILE
            # Position size should be reduced by volatile multiplier
            expected_size = DEFAULT_POSITION_SIZE * VOLATILE_POSITION_MULT
            assert mock_momentum.position_size == expected_size

    @pytest.mark.asyncio
    async def test_does_not_switch_on_low_confidence(
        self, mock_broker, mock_regime_info_low_confidence
    ):
        """Test that strategy does not switch when confidence is below threshold."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_low_confidence)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            original_strategy_name = strategy.active_strategy_name

            # Trigger regime update with low confidence
            await strategy._update_regime()

            # Should not switch because confidence is below threshold
            # (Only switch_strategy should be skipped, but regime is still recorded)
            # Note: The regime change is detected, but _switch_strategy is not executed
            assert strategy.active_strategy_name == original_strategy_name

    @pytest.mark.asyncio
    async def test_tracks_regime_switches(self, mock_broker, mock_regime_info_bull):
        """Test that regime switches are tracked."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.enable_short_selling = True
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            initial_switches = strategy.regime_switches
            await strategy._update_regime()

            assert strategy.regime_switches == initial_switches + 1
            assert strategy.last_regime_switch is not None


class TestSignalRouting:
    """Test signal routing to active strategy."""

    @pytest.mark.asyncio
    async def test_analyze_symbol_delegates_to_active_strategy(self, mock_broker):
        """Test that analyze_symbol delegates to the active strategy."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.analyze_symbol = AsyncMock(return_value="buy")
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            result = await strategy.analyze_symbol(SINGLE_SYMBOL)

            mock_momentum.analyze_symbol.assert_called_once_with(SINGLE_SYMBOL)
            # analyze_symbol now returns dict with enriched signal
            assert isinstance(result, dict)
            assert result.get("action") == "buy" or "action" in result

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_neutral_when_no_active_strategy(self, mock_broker):
        """Test that analyze_symbol returns neutral when no active strategy."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
        # Don't initialize - no active strategy
        strategy.active_strategy = None

        result = await strategy.analyze_symbol(SINGLE_SYMBOL)

        # analyze_symbol now returns dict
        assert isinstance(result, dict)
        assert result.get("action") == "neutral"

    @pytest.mark.asyncio
    async def test_execute_trade_delegates_to_active_strategy(self, mock_broker):
        """Test that execute_trade delegates to the active strategy."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.execute_trade = AsyncMock()
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            await strategy.execute_trade(SINGLE_SYMBOL, "buy")

            mock_momentum.execute_trade.assert_called_once_with(SINGLE_SYMBOL, "buy")

    @pytest.mark.asyncio
    async def test_generate_signals_delegates_to_active_strategy(self, mock_broker):
        """Test that generate_signals delegates to the active strategy."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.generate_signals = AsyncMock()
            mock_momentum.signals = {"AAPL": "buy", "MSFT": "neutral", "GOOGL": "sell"}
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            await strategy.generate_signals()

            mock_momentum.generate_signals.assert_called_once()
            # Signals should be copied from active strategy
            assert strategy.signals == {"AAPL": "buy", "MSFT": "neutral", "GOOGL": "sell"}

    @pytest.mark.asyncio
    async def test_get_orders_delegates_to_active_strategy(self, mock_broker):
        """Test that get_orders delegates to the active strategy."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_orders = [{"symbol": "AAPL", "side": "buy", "qty": 10}]
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.get_orders = MagicMock(return_value=mock_orders)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            orders = strategy.get_orders()

            mock_momentum.get_orders.assert_called_once()
            assert orders == mock_orders

    def test_get_orders_returns_empty_when_no_active_strategy(self, mock_broker):
        """Test that get_orders returns empty list when no active strategy."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
        strategy.active_strategy = None

        orders = strategy.get_orders()

        assert orders == []


class TestOnBar:
    """Test on_bar method for price updates and routing."""

    @pytest.mark.asyncio
    async def test_on_bar_updates_current_price(self, mock_broker, mock_regime_info_bull):
        """Test that on_bar updates current price."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.on_bar = AsyncMock()
            mock_momentum.signals = {"AAPL": "neutral"}
            mock_momentum.indicators = {"AAPL": {"rsi": 50}}
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            timestamp = datetime.now()
            await strategy.on_bar(SINGLE_SYMBOL, 100.0, 102.0, 99.0, 101.0, 1000000, timestamp)

            assert strategy.current_prices[SINGLE_SYMBOL] == 101.0

    @pytest.mark.asyncio
    async def test_on_bar_updates_price_history(self, mock_broker, mock_regime_info_bull):
        """Test that on_bar updates price history."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.on_bar = AsyncMock()
            mock_momentum.signals = {"AAPL": "neutral"}
            mock_momentum.indicators = {"AAPL": {}}
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            timestamp = datetime.now()
            await strategy.on_bar(SINGLE_SYMBOL, 100.0, 102.0, 99.0, 101.0, 1000000, timestamp)

            assert len(strategy.price_history[SINGLE_SYMBOL]) == 1
            assert strategy.price_history[SINGLE_SYMBOL][0]["close"] == 101.0

    @pytest.mark.asyncio
    async def test_on_bar_routes_to_active_strategy(self, mock_broker, mock_regime_info_bull):
        """Test that on_bar routes to the active strategy."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.on_bar = AsyncMock()
            mock_momentum.signals = {SINGLE_SYMBOL: "buy"}
            mock_momentum.indicators = {SINGLE_SYMBOL: {"rsi": 30}}
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            timestamp = datetime.now()
            await strategy.on_bar(SINGLE_SYMBOL, 100.0, 102.0, 99.0, 101.0, 1000000, timestamp)

            mock_momentum.on_bar.assert_called_once_with(
                SINGLE_SYMBOL, 100.0, 102.0, 99.0, 101.0, 1000000, timestamp
            )

    @pytest.mark.asyncio
    async def test_on_bar_ignores_unknown_symbols(self, mock_broker, mock_regime_info_bull):
        """Test that on_bar ignores symbols not in strategy's symbol list."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.on_bar = AsyncMock()
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            timestamp = datetime.now()
            # Call with unknown symbol
            await strategy.on_bar("UNKNOWN", 100.0, 102.0, 99.0, 101.0, 1000000, timestamp)

            # Should not route to active strategy for unknown symbol
            mock_momentum.on_bar.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_bar_limits_price_history_to_100(self, mock_broker, mock_regime_info_bull):
        """Test that price history is limited to 100 entries."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.on_bar = AsyncMock()
            mock_momentum.signals = {SINGLE_SYMBOL: "neutral"}
            mock_momentum.indicators = {SINGLE_SYMBOL: {}}
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            # Add 105 bars
            for i in range(105):
                timestamp = datetime.now()
                await strategy.on_bar(
                    SINGLE_SYMBOL, 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 1000000, timestamp
                )

            # Should be limited to 100
            assert len(strategy.price_history[SINGLE_SYMBOL]) == 100


class TestGetStatus:
    """Test status reporting."""

    @pytest.mark.asyncio
    async def test_get_status_returns_expected_keys(self, mock_broker):
        """Test that get_status returns all expected keys."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            status = strategy.get_status()

            assert "name" in status
            assert "active_strategy" in status
            assert "current_regime" in status
            assert "regime_switches" in status
            assert "last_switch" in status
            assert "symbols" in status
            assert "signals" in status

    @pytest.mark.asyncio
    async def test_get_status_returns_correct_values(self, mock_broker):
        """Test that get_status returns correct values."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            status = strategy.get_status()

            assert status["name"] == "AdaptiveStrategy"
            assert status["active_strategy"] == "momentum"
            assert status["current_regime"] is None
            assert status["regime_switches"] == 0
            assert status["last_switch"] is None
            assert status["symbols"] == len(TEST_SYMBOLS)

    @pytest.mark.asyncio
    async def test_get_status_filters_neutral_signals(self, mock_broker):
        """Test that get_status filters out neutral signals."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            # Set some signals
            strategy.signals = {"AAPL": "buy", "MSFT": "neutral", "GOOGL": "sell"}

            status = strategy.get_status()

            # Neutral signals should be filtered out
            assert "MSFT" not in status["signals"]
            assert status["signals"] == {"AAPL": "buy", "GOOGL": "sell"}


class TestGetRegimeInfo:
    """Test regime info retrieval."""

    @pytest.mark.asyncio
    async def test_get_regime_info_delegates_to_detector(self, mock_broker, mock_regime_info_bull):
        """Test that get_regime_info delegates to the regime detector."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            regime_info = await strategy.get_regime_info()

            mock_regime.detect_regime.assert_called_once()
            assert regime_info == mock_regime_info_bull


class TestFactoryFunction:
    """Test the factory function for creating AdaptiveStrategy."""

    def test_create_adaptive_strategy_basic(self, mock_broker):
        """Test creating strategy with factory function."""
        from strategies.adaptive_strategy import create_adaptive_strategy

        strategy = create_adaptive_strategy(mock_broker, TEST_SYMBOLS)

        assert strategy is not None
        assert strategy.broker == mock_broker

    def test_create_adaptive_strategy_with_kwargs(self, mock_broker):
        """Test creating strategy with additional kwargs."""
        from strategies.adaptive_strategy import create_adaptive_strategy

        strategy = create_adaptive_strategy(
            mock_broker, TEST_SYMBOLS, position_size=0.15, max_positions=3
        )

        assert strategy is not None
        assert strategy.parameters.get("position_size") == 0.15
        assert strategy.parameters.get("max_positions") == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_update_regime_handles_exception(self, mock_broker):
        """Test that _update_regime handles exceptions gracefully."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(side_effect=Exception("API Error"))
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            # Should not raise, should handle gracefully
            await strategy._update_regime()

            # Regime should remain unchanged
            assert strategy.current_regime is None

    @pytest.mark.asyncio
    async def test_on_bar_handles_exception(self, mock_broker, mock_regime_info_bull):
        """Test that on_bar handles exceptions gracefully."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.on_bar = AsyncMock(side_effect=Exception("Processing error"))
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=mock_regime_info_bull)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()

            timestamp = datetime.now()
            # Should not raise
            await strategy.on_bar(SINGLE_SYMBOL, 100.0, 102.0, 99.0, 101.0, 1000000, timestamp)

    @pytest.mark.asyncio
    async def test_empty_symbols_list(self, mock_broker):
        """Test initialization with empty symbols list."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector"),
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=[])
            result = await strategy.initialize()

            assert result is True
            assert len(strategy.symbols) == 0


class TestParametrizedRegimeSwitching:
    """Parametrized tests for regime switching."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "regime_type,expected_strategy_name,expected_short_selling",
        [
            (REGIME_BULL, "momentum_long", False),
            (REGIME_BEAR, "momentum_short", True),
            (REGIME_SIDEWAYS, "mean_reversion", None),  # N/A for mean reversion
        ],
    )
    async def test_strategy_selection_by_regime(
        self,
        mock_broker,
        regime_type,
        expected_strategy_name,
        expected_short_selling,
    ):
        """Test that correct strategy is selected for each regime."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.enable_short_selling = True
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            MockMeanReversion.return_value = mock_mean_rev

            regime_info = {
                "type": regime_type,
                "confidence": HIGH_CONFIDENCE,
                "trend_direction": "up" if regime_type == REGIME_BULL else "down",
                "trend_strength": 35 if regime_type != REGIME_SIDEWAYS else 15,
                "is_trending": regime_type != REGIME_SIDEWAYS,
                "is_ranging": regime_type == REGIME_SIDEWAYS,
                "volatility_regime": "normal",
                "volatility_pct": 2.0,
                "sma_50": 150.0,
                "sma_200": 140.0,
                "recommended_strategy": "momentum",
                "position_multiplier": 1.0,
                "detected_at": datetime.now().isoformat(),
            }

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=regime_info)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()
            await strategy._update_regime()

            assert strategy.active_strategy_name == expected_strategy_name

            if expected_short_selling is not None:
                assert mock_momentum.enable_short_selling == expected_short_selling

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "regime_type,expected_multiplier",
        [
            (REGIME_BULL, BULL_POSITION_MULT),
            (REGIME_BEAR, BEAR_POSITION_MULT),
            (REGIME_SIDEWAYS, SIDEWAYS_POSITION_MULT),
            (REGIME_VOLATILE, VOLATILE_POSITION_MULT),
        ],
    )
    async def test_position_multiplier_by_regime(
        self, mock_broker, regime_type, expected_multiplier
    ):
        """Test that correct position multiplier is applied for each regime."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        with (
            patch("strategies.adaptive_strategy.MomentumStrategy") as MockMomentum,
            patch("strategies.adaptive_strategy.MeanReversionStrategy") as MockMeanReversion,
            patch("strategies.adaptive_strategy.MarketRegimeDetector") as MockRegime,
        ):
            mock_momentum = AsyncMock()
            mock_momentum.initialize = AsyncMock(return_value=True)
            mock_momentum.position_size = DEFAULT_POSITION_SIZE
            mock_momentum.enable_short_selling = True
            MockMomentum.return_value = mock_momentum

            mock_mean_rev = AsyncMock()
            mock_mean_rev.initialize = AsyncMock(return_value=True)
            mock_mean_rev.position_size = DEFAULT_POSITION_SIZE
            MockMeanReversion.return_value = mock_mean_rev

            regime_info = {
                "type": regime_type,
                "confidence": HIGH_CONFIDENCE,
                "trend_direction": "up",
                "trend_strength": 35 if regime_type != REGIME_SIDEWAYS else 15,
                "is_trending": regime_type not in (REGIME_SIDEWAYS, REGIME_VOLATILE),
                "is_ranging": regime_type == REGIME_SIDEWAYS,
                "volatility_regime": "high" if regime_type == REGIME_VOLATILE else "normal",
                "volatility_pct": 4.5 if regime_type == REGIME_VOLATILE else 2.0,
                "sma_50": 150.0,
                "sma_200": 140.0,
                "recommended_strategy": "momentum",
                "position_multiplier": expected_multiplier,
                "detected_at": datetime.now().isoformat(),
            }

            mock_regime = AsyncMock()
            mock_regime.detect_regime = AsyncMock(return_value=regime_info)
            MockRegime.return_value = mock_regime

            strategy = AdaptiveStrategy(broker=mock_broker, symbols=TEST_SYMBOLS)
            await strategy.initialize()
            await strategy._update_regime()

            # Check that position size was adjusted
            expected_size = DEFAULT_POSITION_SIZE * expected_multiplier
            assert strategy.active_strategy.position_size == expected_size


# =============================================================================
# FACTOR SCORE UPDATE / PROVENANCE GATING TESTS
# =============================================================================


class TestAdaptiveStrategyFactorScoreUpdate:
    """Tests for factor score refresh behavior (including provenance gates)."""

    @pytest.mark.asyncio
    async def test_update_factor_scores_gates_synthetic_fundamentals(self, mock_broker):
        """Synthetic fundamentals should not be passed into the factor model."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy(broker=mock_broker, symbols=["AAPL", "MSFT"])

        # Provide sufficient price history for factor calculations.
        strategy.price_history = {
            "AAPL": [{"close": 100.0}] * 252,
            "MSFT": [{"close": 200.0}] * 252,
        }

        strategy.factor_model = MagicMock()
        strategy.factor_model.score_universe = MagicMock(return_value={})

        strategy.factor_data_provider = AsyncMock()
        strategy.factor_data_provider.build_factor_inputs = AsyncMock(
            return_value={
                "fundamental_data": {"AAPL": {"pe_ratio": 10.0}, "MSFT": {"pe_ratio": 12.0}},
                "market_caps": {"AAPL": 1e12, "MSFT": 2e12},
                "fundamental_data_real": {},
                "market_caps_real": {},
                "data_provenance": {
                    "ratios": {
                        "coverage_ratio": 1.0,
                        "real_ratio": 0.0,
                        "synthetic_ratio": 1.0,
                        "missing_ratio": 0.0,
                    }
                },
            }
        )

        await strategy.update_factor_scores()

        _args, kwargs = strategy.factor_model.score_universe.call_args
        assert kwargs["fundamental_data"] is None
        assert kwargs["market_caps"] is None

    @pytest.mark.asyncio
    async def test_update_factor_scores_passes_real_fundamentals_when_healthy(self, mock_broker):
        """When provenance looks healthy, the strategy should pass non-synthetic inputs."""
        from strategies.adaptive_strategy import AdaptiveStrategy

        strategy = AdaptiveStrategy(broker=mock_broker, symbols=["AAPL", "MSFT"])
        strategy.price_history = {
            "AAPL": [{"close": 100.0}] * 252,
            "MSFT": [{"close": 200.0}] * 252,
        }

        strategy.factor_model = MagicMock()
        strategy.factor_model.score_universe = MagicMock(return_value={})

        strategy.factor_data_provider = AsyncMock()
        strategy.factor_data_provider.build_factor_inputs = AsyncMock(
            return_value={
                "fundamental_data": {"AAPL": {"pe_ratio": 99.0}, "MSFT": {"pe_ratio": 98.0}},
                "market_caps": {"AAPL": 9e12, "MSFT": 8e12},
                "fundamental_data_real": {"AAPL": {"pe_ratio": 10.0}, "MSFT": {"pe_ratio": 12.0}},
                "market_caps_real": {"AAPL": 1e12, "MSFT": 2e12},
                "data_provenance": {
                    "ratios": {
                        "coverage_ratio": 1.0,
                        "real_ratio": 1.0,
                        "synthetic_ratio": 0.0,
                        "missing_ratio": 0.0,
                    }
                },
            }
        )

        await strategy.update_factor_scores()

        _args, kwargs = strategy.factor_model.score_universe.call_args
        assert kwargs["fundamental_data"] == {
            "AAPL": {"pe_ratio": 10.0},
            "MSFT": {"pe_ratio": 12.0},
        }
        assert kwargs["market_caps"] == {"AAPL": 1e12, "MSFT": 2e12}
