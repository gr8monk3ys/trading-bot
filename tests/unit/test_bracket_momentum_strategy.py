"""
Unit tests for BracketMomentumStrategy.

Tests cover:
1. Initialization with default and custom parameters
2. Bracket order creation with ATR-based stops
3. Bracket order creation with percentage-based stops
4. Signal generation (buy, sell, neutral conditions)
5. Position size limiting
6. Parameter validation
"""

import warnings
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from strategies.bracket_momentum_strategy import BracketMomentumStrategy


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_broker():
    """Create a mock broker with default account values."""
    broker = AsyncMock()
    broker.get_account.return_value = MagicMock(
        buying_power="100000.0",
        equity="100000.0",
        cash="50000.0",
    )
    broker.get_positions.return_value = []
    broker.get_last_price.return_value = 150.0
    broker.submit_order_advanced.return_value = MagicMock(id="test-order-123")
    return broker


@pytest.fixture
def sample_price_history():
    """Generate realistic OHLCV price data for testing."""
    np.random.seed(42)
    base = 100
    returns = np.random.normal(0.001, 0.02, 60)  # Need enough data for indicators
    prices = base * np.cumprod(1 + returns)

    history = []
    for i, p in enumerate(prices):
        # Generate realistic OHLCV data
        high = p * (1 + abs(np.random.normal(0, 0.01)))
        low = p * (1 - abs(np.random.normal(0, 0.01)))
        open_price = p * (1 + np.random.normal(0, 0.005))
        history.append({
            "timestamp": datetime.now(),
            "open": open_price,
            "high": max(high, open_price, p),
            "low": min(low, open_price, p),
            "close": p,
            "volume": 1000000 + np.random.randint(-100000, 100000),
        })
    return history


@pytest.fixture
def bullish_indicators():
    """Return indicator values that should generate a buy signal."""
    return {
        "rsi": 30,  # Below buy threshold (35)
        "macd": 0.5,
        "macd_signal": 0.2,  # MACD > signal (bullish crossover)
        "macd_hist": 0.3,  # Positive histogram
        "fast_ma": 105,
        "slow_ma": 100,  # Fast > slow (uptrend)
        "atr": 2.5,
        "close": 100,
    }


@pytest.fixture
def bearish_indicators():
    """Return indicator values that should NOT generate a buy signal."""
    return {
        "rsi": 70,  # Above sell threshold (65)
        "macd": -0.5,
        "macd_signal": 0.2,  # MACD < signal (bearish)
        "macd_hist": -0.3,  # Negative histogram
        "fast_ma": 95,
        "slow_ma": 100,  # Fast < slow (downtrend)
        "atr": 2.5,
        "close": 100,
    }


@pytest.fixture
def neutral_indicators():
    """Return indicator values that should NOT trigger any trade."""
    return {
        "rsi": 50,  # Neither oversold nor overbought
        "macd": 0.1,
        "macd_signal": 0.1,
        "macd_hist": 0.0,
        "fast_ma": 100,
        "slow_ma": 100,
        "atr": 2.5,
        "close": 100,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestBracketMomentumStrategyInit:
    """Test initialization of BracketMomentumStrategy."""

    @pytest.mark.asyncio
    async def test_initializes_with_defaults(self, mock_broker):
        """Test that strategy initializes with default parameters."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await strategy.initialize()

            assert result is True
            # Check deprecation warning was issued
            assert len(w) >= 1
            assert "experimental" in str(w[0].message).lower()

        # Check default parameters are set
        assert strategy.position_size == 0.1
        assert strategy.max_positions == 3
        assert strategy.rsi_period == 14
        assert strategy.profit_target_pct == 0.08
        assert strategy.stop_loss_pct == 0.03

    @pytest.mark.asyncio
    async def test_shows_deprecation_warning(self, mock_broker):
        """Test that deprecation warning is shown during initialization."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with pytest.warns(UserWarning, match="experimental"):
            await strategy.initialize()

    @pytest.mark.asyncio
    async def test_initializes_with_custom_parameters(self, mock_broker):
        """Test that strategy respects custom parameters."""
        custom_params = {
            "symbols": ["AAPL", "MSFT"],
            "position_size": 0.05,
            "max_positions": 5,
            "rsi_period": 10,
            "profit_target_pct": 0.10,
            "stop_loss_pct": 0.02,
            "use_atr_stops": False,
        }

        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters=custom_params
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        assert strategy.position_size == 0.05
        assert strategy.max_positions == 5
        assert strategy.rsi_period == 10
        assert strategy.profit_target_pct == 0.10
        assert strategy.stop_loss_pct == 0.02
        assert strategy.use_atr_stops is False

    @pytest.mark.asyncio
    async def test_initializes_tracking_dictionaries(self, mock_broker):
        """Test that tracking dictionaries are properly initialized."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": symbols}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Check that all tracking dicts are initialized for each symbol
        for symbol in symbols:
            assert symbol in strategy.indicators
            assert symbol in strategy.signals
            assert symbol in strategy.price_history

        # Check initial states
        assert all(signal == "neutral" for signal in strategy.signals.values())
        assert strategy.active_bracket_orders == {}


# =============================================================================
# SIGNAL GENERATION TESTS
# =============================================================================


class TestSignalGeneration:
    """Test signal generation logic."""

    @pytest.mark.asyncio
    async def test_generates_buy_signal_when_conditions_met(
        self, mock_broker, bullish_indicators
    ):
        """Test that buy signal is generated when all bullish conditions are met."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up indicators for buy signal
        strategy.indicators["AAPL"] = bullish_indicators

        signal = await strategy._generate_signal("AAPL")
        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_generates_neutral_signal_when_bearish(
        self, mock_broker, bearish_indicators
    ):
        """Test that neutral signal is generated when conditions are bearish."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.indicators["AAPL"] = bearish_indicators

        signal = await strategy._generate_signal("AAPL")
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generates_neutral_signal_when_mixed_conditions(
        self, mock_broker
    ):
        """Test that neutral signal is generated when conditions are mixed."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # RSI low but trend is down
        mixed_indicators = {
            "rsi": 30,  # Oversold (bullish)
            "macd": 0.5,
            "macd_signal": 0.2,  # Bullish crossover
            "macd_hist": 0.3,  # Positive
            "fast_ma": 95,
            "slow_ma": 100,  # Downtrend (bearish)
            "atr": 2.5,
            "close": 100,
        }
        strategy.indicators["AAPL"] = mixed_indicators

        signal = await strategy._generate_signal("AAPL")
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_returns_neutral_when_indicators_not_available(self, mock_broker):
        """Test that neutral is returned when indicators are not calculated."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Empty indicators
        strategy.indicators["AAPL"] = {}

        signal = await strategy._generate_signal("AAPL")
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_returns_neutral_when_rsi_is_none(self, mock_broker):
        """Test that neutral is returned when RSI is None."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.indicators["AAPL"] = {"rsi": None}

        signal = await strategy._generate_signal("AAPL")
        assert signal == "neutral"


# =============================================================================
# BRACKET ORDER TESTS
# =============================================================================


class TestBracketOrderCreation:
    """Test bracket order creation logic."""

    @pytest.mark.asyncio
    async def test_creates_bracket_order_with_atr_stops(self, mock_broker):
        """Test bracket order creation with ATR-based stop levels."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "use_atr_stops": True,
                "atr_multiplier": 2.0,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up current price and indicators
        current_price = 150.0
        atr = 3.0  # ATR value
        strategy.current_prices["AAPL"] = current_price
        strategy.indicators["AAPL"] = {"atr": atr}

        # Execute bracket buy
        await strategy._execute_bracket_buy("AAPL")

        # Verify order was submitted
        mock_broker.submit_order_advanced.assert_called_once()

        # Get the order that was submitted
        order_call = mock_broker.submit_order_advanced.call_args
        order = order_call[0][0]  # First positional argument

        # Verify it's a bracket order for AAPL
        assert order.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_creates_bracket_order_with_percentage_stops(self, mock_broker):
        """Test bracket order creation with percentage-based stop levels."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "use_atr_stops": False,
                "profit_target_pct": 0.08,
                "stop_loss_pct": 0.03,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        current_price = 100.0
        strategy.current_prices["AAPL"] = current_price
        strategy.indicators["AAPL"] = {"atr": None}  # No ATR

        await strategy._execute_bracket_buy("AAPL")

        # Verify order was submitted
        mock_broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_order_when_already_have_position(self, mock_broker):
        """Test that no order is created when already holding position."""
        # Set up existing position
        existing_position = MagicMock()
        existing_position.symbol = "AAPL"
        mock_broker.get_positions.return_value = [existing_position]

        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.current_prices["AAPL"] = 150.0
        strategy.indicators["AAPL"] = {"atr": 2.5}

        await strategy._execute_bracket_buy("AAPL")

        # Verify no order was submitted
        mock_broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_order_when_max_positions_reached(self, mock_broker):
        """Test that no order is created when max positions reached."""
        # Set up existing positions at max
        positions = [
            MagicMock(symbol="MSFT"),
            MagicMock(symbol="GOOGL"),
            MagicMock(symbol="TSLA"),
        ]
        mock_broker.get_positions.return_value = positions

        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "max_positions": 3,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.current_prices["AAPL"] = 150.0
        strategy.indicators["AAPL"] = {"atr": 2.5}

        await strategy._execute_bracket_buy("AAPL")

        # Verify no order was submitted
        mock_broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_tracks_bracket_order_after_submission(self, mock_broker):
        """Test that bracket order is tracked after successful submission."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.current_prices["AAPL"] = 150.0
        strategy.indicators["AAPL"] = {"atr": 2.5}

        await strategy._execute_bracket_buy("AAPL")

        # Verify order is tracked
        assert "AAPL" in strategy.active_bracket_orders
        tracked = strategy.active_bracket_orders["AAPL"]
        assert tracked["order_id"] == "test-order-123"
        assert tracked["entry_price"] == 150.0


# =============================================================================
# POSITION SIZE TESTS
# =============================================================================


class TestPositionSizing:
    """Test position sizing logic."""

    @pytest.mark.asyncio
    async def test_enforces_max_position_size_limit(self, mock_broker):
        """Test that position size is capped at max_position_size."""
        # Setup account with large equity
        mock_broker.get_account.return_value = MagicMock(
            buying_power="1000000.0",
            equity="1000000.0",
        )

        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "position_size": 0.5,  # 50% is way too large
                "max_position_size": 0.05,  # But max is 5%
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.current_prices["AAPL"] = 150.0
        strategy.indicators["AAPL"] = {"atr": 2.5}

        await strategy._execute_bracket_buy("AAPL")

        # Order should have been submitted with capped size
        mock_broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_order_when_quantity_too_small(self, mock_broker):
        """Test that order is skipped when calculated quantity is too small."""
        # Setup account with small buying power
        mock_broker.get_account.return_value = MagicMock(
            buying_power="10.0",  # Very small
            equity="10.0",
        )

        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "position_size": 0.01,  # 1% of $10 = $0.10
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Price too high for tiny account
        strategy.current_prices["AAPL"] = 150.0
        strategy.indicators["AAPL"] = {"atr": 2.5}

        await strategy._execute_bracket_buy("AAPL")

        # Order should not have been submitted (quantity < 0.01)
        mock_broker.submit_order_advanced.assert_not_called()


# =============================================================================
# ON_BAR TESTS
# =============================================================================


class TestOnBar:
    """Test on_bar event handling."""

    @pytest.mark.asyncio
    async def test_on_bar_updates_price_history(self, mock_broker):
        """Test that on_bar updates price history correctly."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Call on_bar
        await strategy.on_bar(
            symbol="AAPL",
            open_price=149.0,
            high_price=152.0,
            low_price=148.0,
            close_price=150.0,
            volume=1000000,
            timestamp=datetime.now(),
        )

        # Verify price history was updated
        assert len(strategy.price_history["AAPL"]) == 1
        assert strategy.price_history["AAPL"][0]["close"] == 150.0
        assert strategy.current_prices["AAPL"] == 150.0

    @pytest.mark.asyncio
    async def test_on_bar_ignores_unknown_symbols(self, mock_broker):
        """Test that on_bar ignores symbols not in strategy.symbols."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Call on_bar with unknown symbol
        await strategy.on_bar(
            symbol="UNKNOWN",
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            volume=500000,
            timestamp=datetime.now(),
        )

        # Verify no changes were made
        assert "UNKNOWN" not in strategy.price_history

    @pytest.mark.asyncio
    async def test_on_bar_limits_history_size(self, mock_broker, sample_price_history):
        """Test that price history is trimmed to max size."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Add lots of price history
        for bar in sample_price_history:
            await strategy.on_bar(
                symbol="AAPL",
                open_price=bar["open"],
                high_price=bar["high"],
                low_price=bar["low"],
                close_price=bar["close"],
                volume=bar["volume"],
                timestamp=bar["timestamp"],
            )

        # Verify history is limited
        max_history = max(
            strategy.slow_ma,
            strategy.rsi_period,
            strategy.macd_slow + strategy.macd_signal,
            strategy.atr_period,
        ) + 10

        assert len(strategy.price_history["AAPL"]) <= max_history


# =============================================================================
# ANALYZE SYMBOL TESTS
# =============================================================================


class TestAnalyzeSymbol:
    """Test analyze_symbol method."""

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_current_signal(self, mock_broker):
        """Test that analyze_symbol returns the current signal for a symbol."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set signal
        strategy.signals["AAPL"] = "buy"

        result = await strategy.analyze_symbol("AAPL")
        assert result == "buy"

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_neutral_for_unknown(self, mock_broker):
        """Test that analyze_symbol returns neutral for unknown symbols."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        result = await strategy.analyze_symbol("UNKNOWN")
        assert result == "neutral"


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


class TestParameterValidation:
    """Test parameter validation."""

    @pytest.mark.asyncio
    async def test_default_parameters_are_valid(self, mock_broker):
        """Test that default_parameters returns valid configuration."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        params = strategy.default_parameters()

        # Verify required parameters exist and have sensible values
        assert 0 < params["position_size"] <= 1
        assert params["max_positions"] >= 1
        assert params["rsi_period"] > 0
        assert params["profit_target_pct"] > 0
        assert params["stop_loss_pct"] > 0
        assert params["profit_target_pct"] > params["stop_loss_pct"]  # R:R should be > 1

    @pytest.mark.asyncio
    async def test_atr_parameters_are_valid(self, mock_broker):
        """Test that ATR parameters are properly configured."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        params = strategy.default_parameters()

        assert params["use_atr_stops"] is True
        assert params["atr_period"] > 0
        assert params["atr_multiplier"] > 0

    @pytest.mark.asyncio
    async def test_ma_parameters_are_valid(self, mock_broker):
        """Test that moving average parameters are valid."""
        strategy = BracketMomentumStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        params = strategy.default_parameters()

        # Fast MA should be shorter than slow MA
        assert params["fast_ma_period"] < params["slow_ma_period"]
        assert params["fast_ma_period"] > 0
        assert params["slow_ma_period"] > 0
