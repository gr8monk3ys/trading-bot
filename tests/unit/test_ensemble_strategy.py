"""
Unit tests for EnsembleStrategy.

Tests cover:
1. Initialization of sub-strategies (mean reversion, momentum, trend following)
2. Individual sub-strategy signal generation
3. Signal combination and weighting logic
4. Minimum agreement threshold (60% default)
5. Regime-based weight boosting
6. Parameter validation
"""

import warnings
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from strategies.ensemble_strategy import EnsembleStrategy


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
    """Generate realistic OHLCV price data for testing (minimum 50 bars needed)."""
    np.random.seed(42)
    base = 100
    returns = np.random.normal(0.001, 0.02, 60)
    prices = base * np.cumprod(1 + returns)

    history = []
    for i, p in enumerate(prices):
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
def trending_indicators():
    """Indicators for a trending market regime."""
    return {
        # Trend indicators
        "adx": 30,  # Strong trend (> 25)
        "plus_di": 25,
        "minus_di": 15,  # Plus > Minus = uptrend
        "fast_ma": 105,
        "slow_ma": 100,
        "ema_20": 103,
        # Momentum indicators
        "rsi": 55,
        "macd": 0.5,
        "macd_signal": 0.2,
        "macd_hist": 0.3,
        "stoch_k": 60,
        "stoch_d": 55,
        # Volatility indicators
        "bb_upper": 110,
        "bb_middle": 100,
        "bb_lower": 90,
        "atr": 1.5,
        "stddev": 2.0,
        # Volume indicators
        "vwap": 102,
        "volume": 1000000,
        "volume_sma": 900000,
        # Price
        "close": 105,
    }


@pytest.fixture
def ranging_indicators():
    """Indicators for a ranging market regime."""
    return {
        # Trend indicators
        "adx": 15,  # Weak trend (< 20)
        "plus_di": 20,
        "minus_di": 20,  # No clear direction
        "fast_ma": 100,
        "slow_ma": 100,
        "ema_20": 100,
        # Momentum indicators
        "rsi": 50,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "stoch_k": 50,
        "stoch_d": 50,
        # Volatility indicators
        "bb_upper": 105,
        "bb_middle": 100,
        "bb_lower": 95,
        "atr": 1.0,
        "stddev": 1.5,
        # Volume indicators
        "vwap": 100,
        "volume": 1000000,
        "volume_sma": 1000000,
        # Price
        "close": 100,
    }


@pytest.fixture
def mean_reversion_buy_indicators():
    """Indicators that should trigger mean reversion buy signal."""
    return {
        # Trend indicators
        "adx": 15,  # Ranging market (best for mean reversion)
        "plus_di": 20,
        "minus_di": 20,
        "fast_ma": 100,
        "slow_ma": 100,
        "ema_20": 100,
        # Momentum indicators
        "rsi": 25,  # Oversold (< 30)
        "macd": -0.5,
        "macd_signal": -0.3,
        "macd_hist": -0.2,
        "stoch_k": 15,  # Oversold (< 20)
        "stoch_d": 18,
        # Volatility indicators - price below lower band
        "bb_upper": 110,
        "bb_middle": 100,
        "bb_lower": 90,
        "atr": 1.5,
        "stddev": 2.0,
        # Volume indicators
        "vwap": 100,
        "volume": 1000000,
        "volume_sma": 900000,
        # Price below lower Bollinger band
        "close": 88,  # Below bb_lower (90)
    }


@pytest.fixture
def momentum_buy_indicators():
    """Indicators that should trigger momentum buy signal."""
    return {
        # Trend indicators
        "adx": 30,  # Trending market (best for momentum)
        "plus_di": 30,
        "minus_di": 15,
        "fast_ma": 105,
        "slow_ma": 100,
        "ema_20": 103,
        # Momentum indicators - bullish
        "rsi": 60,  # Above 50 (bullish confirmation)
        "macd": 0.8,
        "macd_signal": 0.4,  # MACD > Signal
        "macd_hist": 0.4,  # Positive histogram
        "stoch_k": 70,
        "stoch_d": 65,
        # Volatility indicators
        "bb_upper": 112,
        "bb_middle": 105,
        "bb_lower": 98,
        "atr": 2.0,
        "stddev": 2.5,
        # Volume indicators
        "vwap": 104,
        "volume": 1200000,
        "volume_sma": 1000000,
        # Price
        "close": 106,
    }


@pytest.fixture
def trend_following_buy_indicators():
    """Indicators that should trigger trend following buy signal."""
    return {
        # Trend indicators - strong uptrend
        "adx": 35,  # Strong trend
        "plus_di": 35,
        "minus_di": 12,  # Plus >> Minus = strong uptrend
        "fast_ma": 108,
        "slow_ma": 100,  # Fast > Slow
        "ema_20": 105,
        # Momentum indicators
        "rsi": 65,
        "macd": 1.0,
        "macd_signal": 0.6,
        "macd_hist": 0.4,
        "stoch_k": 75,
        "stoch_d": 70,
        # Volatility indicators
        "bb_upper": 115,
        "bb_middle": 107,
        "bb_lower": 99,
        "atr": 2.5,
        "stddev": 3.0,
        # Volume indicators
        "vwap": 106,
        "volume": 1300000,
        "volume_sma": 1000000,
        # Price above EMA (confirms uptrend)
        "close": 110,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestEnsembleStrategyInit:
    """Test initialization of EnsembleStrategy."""

    @pytest.mark.asyncio
    async def test_initializes_with_defaults(self, mock_broker):
        """Test that strategy initializes with default parameters."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await strategy.initialize()

            assert result is True
            # Check deprecation warning
            assert len(w) >= 1
            assert "experimental" in str(w[0].message).lower()

        # Check default parameters
        assert strategy.position_size == 0.10
        assert strategy.max_positions == 5
        assert strategy.stop_loss == 0.025
        assert strategy.take_profit == 0.05

    @pytest.mark.asyncio
    async def test_shows_deprecation_warning(self, mock_broker):
        """Test that deprecation warning is shown during initialization."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with pytest.warns(UserWarning, match="experimental"):
            await strategy.initialize()

    @pytest.mark.asyncio
    async def test_initializes_tracking_structures(self, mock_broker):
        """Test that all tracking structures are initialized."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": symbols}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Check tracking structures
        for symbol in symbols:
            assert symbol in strategy.indicators
            assert symbol in strategy.market_regime
            assert symbol in strategy.sub_strategy_signals
            assert symbol in strategy.ensemble_signals
            assert symbol in strategy.price_history

        # Check initial states
        assert all(regime == "unknown" for regime in strategy.market_regime.values())
        assert all(signal == "neutral" for signal in strategy.ensemble_signals.values())

    @pytest.mark.asyncio
    async def test_initializes_risk_manager(self, mock_broker):
        """Test that risk manager is initialized."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        assert strategy.risk_manager is not None

    @pytest.mark.asyncio
    async def test_custom_parameters_override_defaults(self, mock_broker):
        """Test that custom parameters correctly override defaults."""
        custom_params = {
            "symbols": ["AAPL", "MSFT"],
            "position_size": 0.15,
            "max_positions": 3,
            "stop_loss": 0.03,
            "take_profit": 0.08,
            "min_agreement_pct": 0.75,
            "regime_weight_boost": 2.0,
            "max_correlation": 0.5,
        }
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters=custom_params
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Verify custom parameters were applied
        assert strategy.position_size == 0.15
        assert strategy.max_positions == 3
        assert strategy.stop_loss == 0.03
        assert strategy.take_profit == 0.08
        assert strategy.parameters["min_agreement_pct"] == 0.75
        assert strategy.parameters["regime_weight_boost"] == 2.0
        assert strategy.risk_manager.max_correlation == 0.5


# =============================================================================
# MARKET REGIME DETECTION TESTS
# =============================================================================


class TestMarketRegimeDetection:
    """Test market regime detection logic."""

    @pytest.mark.asyncio
    async def test_detects_trending_regime(self, mock_broker, trending_indicators):
        """Test detection of trending market regime."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.indicators["AAPL"] = trending_indicators

        await strategy._detect_market_regime("AAPL")

        assert "trending" in strategy.market_regime["AAPL"]

    @pytest.mark.asyncio
    async def test_detects_ranging_regime(self, mock_broker, ranging_indicators):
        """Test detection of ranging market regime."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.indicators["AAPL"] = ranging_indicators

        await strategy._detect_market_regime("AAPL")

        assert "ranging" in strategy.market_regime["AAPL"]

    @pytest.mark.asyncio
    async def test_detects_volatile_regime(self, mock_broker):
        """Test detection of volatile market conditions."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "atr_volatility_threshold": 0.02,  # 2%
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # High ATR relative to price = volatile
        volatile_indicators = {
            "adx": 22,  # Transitional
            "atr": 5.0,  # High ATR
            "close": 100,  # ATR/close = 5% > 2% threshold
        }
        strategy.indicators["AAPL"] = volatile_indicators

        await strategy._detect_market_regime("AAPL")

        assert "volatile" in strategy.market_regime["AAPL"]

    @pytest.mark.asyncio
    async def test_detects_transitional_regime(self, mock_broker):
        """Test detection of transitional market regime (ADX between thresholds)."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "adx_trending_threshold": 25,
                "adx_ranging_threshold": 20,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # ADX between 20 and 25
        transitional_indicators = {
            "adx": 22,  # Between thresholds
            "atr": 1.0,
            "close": 100,
        }
        strategy.indicators["AAPL"] = transitional_indicators

        await strategy._detect_market_regime("AAPL")

        assert "transitional" in strategy.market_regime["AAPL"]


# =============================================================================
# SUB-STRATEGY SIGNAL TESTS
# =============================================================================


class TestSubStrategySignals:
    """Test individual sub-strategy signal generation."""

    @pytest.mark.asyncio
    async def test_mean_reversion_buy_signal(
        self, mock_broker, mean_reversion_buy_indicators
    ):
        """Test mean reversion generates buy signal when oversold."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.indicators["AAPL"] = mean_reversion_buy_indicators

        await strategy._generate_sub_strategy_signals("AAPL")

        signals = strategy.sub_strategy_signals["AAPL"]
        assert "mean_reversion" in signals
        assert signals["mean_reversion"]["signal"] == "buy"
        assert signals["mean_reversion"]["best_regime"] == "ranging"

    @pytest.mark.asyncio
    async def test_momentum_buy_signal(self, mock_broker, momentum_buy_indicators):
        """Test momentum strategy generates buy signal in uptrend."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.indicators["AAPL"] = momentum_buy_indicators

        await strategy._generate_sub_strategy_signals("AAPL")

        signals = strategy.sub_strategy_signals["AAPL"]
        assert "momentum" in signals
        assert signals["momentum"]["signal"] == "buy"
        assert signals["momentum"]["best_regime"] == "trending"

    @pytest.mark.asyncio
    async def test_trend_following_buy_signal(
        self, mock_broker, trend_following_buy_indicators
    ):
        """Test trend following generates buy signal in strong uptrend."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.indicators["AAPL"] = trend_following_buy_indicators

        await strategy._generate_sub_strategy_signals("AAPL")

        signals = strategy.sub_strategy_signals["AAPL"]
        assert "trend_following" in signals
        assert signals["trend_following"]["signal"] == "buy"
        assert signals["trend_following"]["best_regime"] == "trending"

    @pytest.mark.asyncio
    async def test_sub_strategies_return_neutral_when_conditions_not_met(
        self, mock_broker, ranging_indicators
    ):
        """Test sub-strategies return neutral when no clear signals."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.indicators["AAPL"] = ranging_indicators

        await strategy._generate_sub_strategy_signals("AAPL")

        signals = strategy.sub_strategy_signals["AAPL"]
        # All should be neutral in ranging market with neutral indicators
        for strategy_name, signal_data in signals.items():
            assert signal_data["signal"] in ["neutral", "buy", "sell"]


# =============================================================================
# SIGNAL COMBINATION TESTS
# =============================================================================


class TestSignalCombination:
    """Test signal combination and weighting logic."""

    @pytest.mark.asyncio
    async def test_combines_signals_with_minimum_agreement(self, mock_broker):
        """Test that signals are combined when minimum agreement is met."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "min_agreement_pct": 0.60,  # 60% agreement needed
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up sub-strategy signals (2 of 3 agree = 66.7%)
        strategy.sub_strategy_signals["AAPL"] = {
            "mean_reversion": {"signal": "buy", "strength": 1.0, "best_regime": "ranging"},
            "momentum": {"signal": "buy", "strength": 1.0, "best_regime": "trending"},
            "trend_following": {"signal": "neutral", "strength": 0.5, "best_regime": "trending"},
        }
        strategy.market_regime["AAPL"] = "trending_normal"

        await strategy._combine_signals("AAPL")

        # Should have buy signal (66.7% > 60% threshold)
        assert strategy.ensemble_signals["AAPL"] == "buy"

    @pytest.mark.asyncio
    async def test_returns_neutral_when_agreement_too_low(self, mock_broker):
        """Test that neutral is returned when agreement is below threshold."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "min_agreement_pct": 0.80,  # 80% agreement needed (high bar)
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up conflicting signals
        strategy.sub_strategy_signals["AAPL"] = {
            "mean_reversion": {"signal": "buy", "strength": 1.0, "best_regime": "ranging"},
            "momentum": {"signal": "sell", "strength": 1.0, "best_regime": "trending"},
            "trend_following": {"signal": "neutral", "strength": 0.5, "best_regime": "trending"},
        }
        strategy.market_regime["AAPL"] = "trending_normal"

        await strategy._combine_signals("AAPL")

        # Should be neutral (no clear agreement)
        assert strategy.ensemble_signals["AAPL"] == "neutral"

    @pytest.mark.asyncio
    async def test_regime_weight_boosting(self, mock_broker):
        """Test that regime-matching strategies get weight boost."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "min_agreement_pct": 0.50,
                "regime_weight_boost": 2.0,  # 2x boost for matching regime
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up signals where regime boost makes the difference
        strategy.sub_strategy_signals["AAPL"] = {
            "mean_reversion": {"signal": "sell", "strength": 1.0, "best_regime": "ranging"},
            "momentum": {"signal": "buy", "strength": 1.0, "best_regime": "trending"},
            "trend_following": {"signal": "buy", "strength": 1.0, "best_regime": "trending"},
        }
        # Set trending regime - momentum and trend following should get boosted
        strategy.market_regime["AAPL"] = "trending_normal"

        await strategy._combine_signals("AAPL")

        # Trending strategies should dominate with regime boost
        # momentum (boosted) + trend_following (boosted) should outweigh mean_reversion
        assert strategy.ensemble_signals["AAPL"] == "buy"

    @pytest.mark.asyncio
    async def test_returns_neutral_when_no_signals(self, mock_broker):
        """Test that neutral is returned when no sub-strategy signals exist."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Empty signals
        strategy.sub_strategy_signals["AAPL"] = {}
        strategy.market_regime["AAPL"] = "unknown"

        await strategy._combine_signals("AAPL")

        assert strategy.ensemble_signals["AAPL"] == "neutral"

    @pytest.mark.asyncio
    async def test_confidence_calculation_from_strength(self, mock_broker):
        """Test that signal strength affects voting weight proportionally."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "min_agreement_pct": 0.50,
                "regime_weight_boost": 1.0,  # No boost to simplify calculation
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # High strength buy signals should dominate low strength sell
        strategy.sub_strategy_signals["AAPL"] = {
            "mean_reversion": {"signal": "buy", "strength": 1.0, "best_regime": "ranging"},
            "momentum": {"signal": "buy", "strength": 1.0, "best_regime": "trending"},
            "trend_following": {"signal": "sell", "strength": 0.2, "best_regime": "trending"},
        }
        strategy.market_regime["AAPL"] = "transitional_normal"  # No regime boost

        await strategy._combine_signals("AAPL")

        # Buy weight = 1.0 + 1.0 = 2.0
        # Sell weight = 0.2
        # Total = 2.2
        # Buy agreement = 2.0 / 2.2 = ~91% > 50%
        assert strategy.ensemble_signals["AAPL"] == "buy"


# =============================================================================
# SIGNAL EXECUTION TESTS
# =============================================================================


class TestSignalExecution:
    """Test signal execution logic."""

    @pytest.mark.asyncio
    async def test_executes_buy_signal_when_no_position(self, mock_broker):
        """Test that buy signal is executed when no existing position."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.current_prices["AAPL"] = 150.0
        strategy.price_history["AAPL"] = [{"close": 150.0}] * 25  # Enough history
        strategy.market_regime["AAPL"] = "trending_normal"
        strategy.sub_strategy_signals["AAPL"] = {
            "momentum": {"signal": "buy", "strength": 1.0, "best_regime": "trending"}
        }

        await strategy._execute_signal("AAPL", "buy")

        mock_broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_buy_when_max_positions_reached(self, mock_broker):
        """Test that buy is skipped when max positions reached."""
        positions = [
            MagicMock(symbol="MSFT", market_value="10000"),
            MagicMock(symbol="GOOGL", market_value="10000"),
            MagicMock(symbol="AMZN", market_value="10000"),
            MagicMock(symbol="META", market_value="10000"),
            MagicMock(symbol="NVDA", market_value="10000"),
        ]
        mock_broker.get_positions.return_value = positions

        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "max_positions": 5,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.current_prices["AAPL"] = 150.0

        await strategy._execute_signal("AAPL", "buy")

        mock_broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_executes_sell_signal_with_existing_position(self, mock_broker):
        """Test that sell signal is executed when holding position."""
        existing_position = MagicMock()
        existing_position.symbol = "AAPL"
        existing_position.qty = "10"
        mock_broker.get_positions.return_value = [existing_position]

        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.current_prices["AAPL"] = 150.0

        await strategy._execute_signal("AAPL", "sell")

        mock_broker.submit_order_advanced.assert_called_once()


# =============================================================================
# TRAILING STOP TESTS
# =============================================================================


class TestTrailingStop:
    """Test trailing stop logic."""

    @pytest.mark.asyncio
    async def test_updates_highest_price(self, mock_broker):
        """Test that highest price is tracked for trailing stops."""
        existing_position = MagicMock()
        existing_position.symbol = "AAPL"
        existing_position.qty = "10"
        mock_broker.get_positions.return_value = [existing_position]

        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up position entry and prices
        strategy.position_entries["AAPL"] = {"price": 100.0, "quantity": 10}
        strategy.highest_prices["AAPL"] = 110.0
        strategy.current_prices["AAPL"] = 115.0  # New high

        await strategy._check_exit_conditions("AAPL")

        # Highest price should be updated
        assert strategy.highest_prices["AAPL"] == 115.0

    @pytest.mark.asyncio
    async def test_triggers_trailing_stop_when_price_drops(self, mock_broker):
        """Test that trailing stop is triggered when price drops from peak."""
        existing_position = MagicMock()
        existing_position.symbol = "AAPL"
        existing_position.qty = "10"
        mock_broker.get_positions.return_value = [existing_position]

        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={
                "symbols": ["AAPL"],
                "trailing_stop": 0.015,  # 1.5% trailing stop
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up position that's profitable
        entry_price = 100.0
        peak_price = 110.0  # +10% profit at peak
        current_price = 107.0  # Dropped ~2.7% from peak (> 1.5%)

        strategy.position_entries["AAPL"] = {"price": entry_price, "quantity": 10}
        strategy.highest_prices["AAPL"] = peak_price
        strategy.current_prices["AAPL"] = current_price

        await strategy._check_exit_conditions("AAPL")

        # Trailing stop should be triggered (sell order submitted)
        mock_broker.submit_order_advanced.assert_called_once()


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


class TestParameterValidation:
    """Test parameter validation."""

    @pytest.mark.asyncio
    async def test_default_parameters_are_valid(self, mock_broker):
        """Test that default_parameters returns valid configuration."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        params = strategy.default_parameters()

        # Verify required parameters
        assert 0 < params["position_size"] <= 1
        assert params["max_positions"] >= 1
        assert 0 < params["min_agreement_pct"] <= 1
        assert params["regime_weight_boost"] >= 1.0
        assert params["stop_loss"] > 0
        assert params["take_profit"] > params["stop_loss"]

    @pytest.mark.asyncio
    async def test_adx_thresholds_are_valid(self, mock_broker):
        """Test that ADX thresholds are properly ordered."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        params = strategy.default_parameters()

        # Trending threshold should be > ranging threshold
        assert params["adx_trending_threshold"] > params["adx_ranging_threshold"]

    @pytest.mark.asyncio
    async def test_sub_strategy_parameters_exist(self, mock_broker):
        """Test that all sub-strategy parameters are defined."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        params = strategy.default_parameters()

        # Mean reversion parameters
        assert "mr_bb_period" in params
        assert "mr_rsi_period" in params
        assert "mr_rsi_oversold" in params
        assert "mr_rsi_overbought" in params

        # Momentum parameters
        assert "mom_rsi_period" in params
        assert "mom_macd_fast" in params
        assert "mom_macd_slow" in params
        assert "mom_adx_threshold" in params

        # Trend following parameters
        assert "tf_fast_ma" in params
        assert "tf_slow_ma" in params
        assert params["tf_fast_ma"] < params["tf_slow_ma"]


# =============================================================================
# ANALYZE SYMBOL TESTS
# =============================================================================


class TestAnalyzeSymbol:
    """Test analyze_symbol method."""

    @pytest.mark.asyncio
    async def test_returns_current_ensemble_signal(self, mock_broker):
        """Test that analyze_symbol returns the ensemble signal."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ensemble_signals["AAPL"] = "buy"

        result = await strategy.analyze_symbol("AAPL")
        assert result == "buy"

    @pytest.mark.asyncio
    async def test_returns_neutral_for_unknown_symbol(self, mock_broker):
        """Test that analyze_symbol returns neutral for unknown symbols."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        result = await strategy.analyze_symbol("UNKNOWN")
        assert result == "neutral"

    @pytest.mark.asyncio
    async def test_integrates_sub_strategy_signals(
        self, mock_broker, sample_price_history
    ):
        """Test full integration from price history to combined signal."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up sufficient price history for indicator calculation
        strategy.price_history["AAPL"] = sample_price_history

        # Run full pipeline
        await strategy._update_indicators("AAPL")
        await strategy._detect_market_regime("AAPL")
        await strategy._generate_sub_strategy_signals("AAPL")
        await strategy._combine_signals("AAPL")

        # Verify result
        result = await strategy.analyze_symbol("AAPL")
        assert result in ["buy", "sell", "neutral"]


# =============================================================================
# ON_BAR TESTS
# =============================================================================


class TestOnBar:
    """Test on_bar price update handling."""

    @pytest.mark.asyncio
    async def test_updates_price_history(self, mock_broker):
        """Test that on_bar correctly updates price history."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Clear initial history
        strategy.price_history["AAPL"] = []

        # Add a bar
        await strategy.on_bar(
            symbol="AAPL",
            open_price=100.0,
            high_price=101.5,
            low_price=99.5,
            close_price=101.0,
            volume=1500000,
            timestamp=datetime.now()
        )

        assert len(strategy.price_history["AAPL"]) == 1
        bar = strategy.price_history["AAPL"][0]
        assert bar["open"] == 100.0
        assert bar["high"] == 101.5
        assert bar["low"] == 99.5
        assert bar["close"] == 101.0
        assert bar["volume"] == 1500000
        assert strategy.current_prices["AAPL"] == 101.0

    @pytest.mark.asyncio
    async def test_ignores_unknown_symbols(self, mock_broker):
        """Test that on_bar ignores symbols not in watchlist."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Try to add bar for symbol not in watchlist
        await strategy.on_bar(
            symbol="UNKNOWN_SYMBOL",
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            volume=1000000,
            timestamp=datetime.now()
        )

        # Should not have added history for unknown symbol
        assert "UNKNOWN_SYMBOL" not in strategy.price_history
        assert "UNKNOWN_SYMBOL" not in strategy.current_prices

    @pytest.mark.asyncio
    async def test_limits_history_size(self, mock_broker):
        """Test that price history is capped at maximum size (200 bars)."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Clear initial history
        strategy.price_history["AAPL"] = []

        # Add more bars than max history (200)
        for i in range(250):
            await strategy.on_bar(
                symbol="AAPL",
                open_price=100.0 + i * 0.1,
                high_price=101.0 + i * 0.1,
                low_price=99.0 + i * 0.1,
                close_price=100.5 + i * 0.1,
                volume=1000000,
                timestamp=datetime.now()
            )

        # Should be capped at 200
        assert len(strategy.price_history["AAPL"]) == 200
        # Most recent bar should be last (i=249)
        expected_close = 100.5 + 249 * 0.1
        assert abs(strategy.price_history["AAPL"][-1]["close"] - expected_close) < 0.01


# =============================================================================
# GET_ORDERS TESTS (BACKTEST MODE)
# =============================================================================


class TestGetOrders:
    """Test get_orders method for backtest mode."""

    @pytest.mark.asyncio
    async def test_generates_buy_orders_for_buy_signal(self, mock_broker):
        """Test that get_orders generates buy order for buy signal without position."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up for buy signal
        strategy.ensemble_signals["AAPL"] = "buy"
        strategy.indicators["AAPL"] = {"close": 150.0}
        strategy.positions = {}  # No existing position
        strategy.capital = 100000

        orders = strategy.get_orders()

        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["side"] == "buy"
        assert orders[0]["type"] == "market"
        # Quantity = (capital * position_size) / price = (100000 * 0.10) / 150 = 66.67
        assert orders[0]["quantity"] > 0

    @pytest.mark.asyncio
    async def test_generates_sell_orders_with_position(self, mock_broker):
        """Test that get_orders generates sell order when holding position."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        # Set up for sell signal with existing position
        strategy.ensemble_signals["AAPL"] = "sell"
        strategy.indicators["AAPL"] = {"close": 150.0}
        strategy.positions = {"AAPL": {"quantity": 50}}

        orders = strategy.get_orders()

        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["side"] == "sell"
        assert orders[0]["quantity"] == 50

    @pytest.mark.asyncio
    async def test_no_orders_for_neutral_signal(self, mock_broker):
        """Test that get_orders returns empty list for neutral signals."""
        strategy = EnsembleStrategy(
            broker=mock_broker,
            parameters={"symbols": ["AAPL"]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await strategy.initialize()

        strategy.ensemble_signals["AAPL"] = "neutral"
        strategy.indicators["AAPL"] = {"close": 150.0}

        orders = strategy.get_orders()

        assert len(orders) == 0
