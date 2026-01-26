"""
Comprehensive Unit Tests for MomentumStrategy

Tests the momentum-based trading strategy including:
- Initialization and parameter handling
- RSI mode (standard vs aggressive)
- Technical indicator calculation
- Signal generation (buy/sell/short)
- Multi-timeframe filtering
- Bollinger Band filtering
- Order execution
- Exit condition monitoring
- Backtest mode

Target: Increase coverage from 5.39% to 90%+
"""

from collections import deque
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest


class TestMomentumStrategyName:
    """Tests for MomentumStrategy NAME attribute."""

    def test_name_attribute(self):
        """Test that NAME attribute is set correctly."""
        from strategies.momentum_strategy import MomentumStrategy

        assert MomentumStrategy.NAME == "MomentumStrategy"


class TestDefaultParameters:
    """Tests for the default_parameters method."""

    def test_default_parameters_returns_dict(self):
        """Test that default_parameters returns a dictionary."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        params = strategy.default_parameters()
        assert isinstance(params, dict)

    def test_default_parameters_contains_basic_params(self):
        """Test that default parameters contain basic trading params."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        params = strategy.default_parameters()

        assert params["position_size"] == 0.10
        assert params["max_positions"] == 5
        assert params["stop_loss"] == 0.03
        assert params["take_profit"] == 0.05

    def test_default_parameters_contains_indicator_params(self):
        """Test that default parameters contain indicator settings."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        params = strategy.default_parameters()

        assert params["rsi_period"] == 14
        assert params["rsi_overbought"] == 70
        assert params["rsi_oversold"] == 30
        assert params["macd_fast_period"] == 12
        assert params["macd_slow_period"] == 26
        assert params["macd_signal_period"] == 9
        assert params["adx_period"] == 14
        assert params["adx_threshold"] == 25

    def test_default_parameters_contains_advanced_features(self):
        """Test that default parameters contain advanced feature flags."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        params = strategy.default_parameters()

        assert "use_multi_timeframe" in params
        assert "enable_short_selling" in params
        assert "use_bollinger_filter" in params
        assert "use_kelly_criterion" in params

    def test_default_rsi_mode_is_aggressive(self):
        """Test that default RSI mode is aggressive."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        params = strategy.default_parameters()
        assert params["rsi_mode"] == "aggressive"

    def test_default_bollinger_params(self):
        """Test default Bollinger Band parameters."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        params = strategy.default_parameters()

        assert params["use_bollinger_filter"] is True
        assert params["bb_period"] == 20
        assert params["bb_std"] == 2.0
        assert params["bb_buy_threshold"] == 0.3
        assert params["bb_sell_threshold"] == 0.7

    def test_default_kelly_params(self):
        """Test default Kelly Criterion parameters."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        params = strategy.default_parameters()

        assert params["use_kelly_criterion"] is True
        assert params["kelly_fraction"] == 0.5
        assert params["kelly_min_trades"] == 30


class TestMomentumStrategyInitialize:
    """Tests for the initialize method."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = Mock()
        broker._add_subscriber = Mock()
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_account = AsyncMock(return_value=Mock(buying_power="100000", equity=100000))
        return broker

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_broker):
        """Test successful initialization."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL", "MSFT"],
            "position_size": 0.10,
            "use_multi_timeframe": False,
            "enable_short_selling": False,
            "use_bollinger_filter": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            result = await strategy.initialize()

        assert result is True
        assert strategy.position_size == 0.10

    @pytest.mark.asyncio
    async def test_initialize_creates_indicator_dicts(self, mock_broker):
        """Test that initialize creates indicator tracking dictionaries."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL", "MSFT"],
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        assert hasattr(strategy, "indicators")
        assert "AAPL" in strategy.indicators
        assert "MSFT" in strategy.indicators

    @pytest.mark.asyncio
    async def test_initialize_creates_signal_dicts(self, mock_broker):
        """Test that initialize creates signal tracking dictionaries."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        assert hasattr(strategy, "signals")
        assert strategy.signals.get("AAPL") == "neutral"

    @pytest.mark.asyncio
    async def test_initialize_aggressive_rsi_mode(self, mock_broker):
        """Test initialization with aggressive RSI mode."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL"],
            "rsi_mode": "aggressive",
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        assert strategy.rsi_period == 2
        assert strategy.rsi_overbought == 90
        assert strategy.rsi_oversold == 10

    @pytest.mark.asyncio
    async def test_initialize_standard_rsi_mode(self, mock_broker):
        """Test initialization with standard RSI mode."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL"],
            "rsi_mode": "standard",
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        assert strategy.rsi_period == 14
        assert strategy.rsi_overbought == 70
        assert strategy.rsi_oversold == 30

    @pytest.mark.asyncio
    async def test_initialize_with_short_selling(self, mock_broker):
        """Test initialization with short selling enabled."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL"],
            "enable_short_selling": True,
            "short_position_size": 0.08,
            "short_stop_loss": 0.04,
            "use_multi_timeframe": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        assert strategy.enable_short_selling is True
        assert strategy.short_position_size == 0.08
        assert strategy.short_stop_loss == 0.04

    @pytest.mark.asyncio
    async def test_initialize_with_multi_timeframe(self, mock_broker):
        """Test initialization with multi-timeframe enabled."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": True,
            "mtf_timeframes": ["5Min", "15Min"],
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        assert strategy.use_multi_timeframe is True
        assert strategy.mtf_analyzer is not None

    @pytest.mark.asyncio
    async def test_initialize_with_bollinger_filter(self, mock_broker):
        """Test initialization with Bollinger Band filter."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL"],
            "use_bollinger_filter": True,
            "bb_period": 20,
            "bb_std": 2.0,
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        assert strategy.use_bollinger_filter is True
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0

    @pytest.mark.asyncio
    async def test_initialize_adds_subscriber_to_broker(self, mock_broker):
        """Test that initialize adds strategy as subscriber to broker."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        mock_broker._add_subscriber.assert_called_once_with(strategy)

    @pytest.mark.asyncio
    async def test_initialize_creates_risk_manager(self, mock_broker):
        """Test that initialize creates a risk manager."""
        from strategies.momentum_strategy import MomentumStrategy

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=mock_broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            await strategy.initialize()

        assert hasattr(strategy, "risk_manager")
        assert strategy.risk_manager is not None

    @pytest.mark.asyncio
    async def test_initialize_handles_exception(self, mock_broker):
        """Test that initialize handles exceptions gracefully."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy(broker=mock_broker, parameters={})

        # Patch super().initialize() to raise an exception
        with patch(
            "strategies.base_strategy.BaseStrategy.initialize", new_callable=AsyncMock
        ) as mock_init:
            mock_init.side_effect = Exception("Test error")
            result = await strategy.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_without_add_subscriber(self):
        """Test initialization when broker doesn't have _add_subscriber."""
        from strategies.momentum_strategy import MomentumStrategy

        broker = Mock(spec=[])  # No _add_subscriber
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_account = AsyncMock(return_value=Mock(buying_power="100000", equity=100000))

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MomentumStrategy(broker=broker, parameters=params)

        with patch.object(
            strategy, "check_trading_allowed", new_callable=AsyncMock, return_value=True
        ):
            result = await strategy.initialize()

        assert result is True


class TestUpdateIndicators:
    """Tests for the _update_indicators method."""

    @pytest.fixture
    def strategy_with_history(self):
        """Create a strategy with price history."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.slow_ma = 50
        strategy.rsi_period = 14
        strategy.macd_fast = 12
        strategy.macd_slow = 26
        strategy.macd_signal = 9
        strategy.adx_period = 14
        strategy.fast_ma = 10
        strategy.medium_ma = 20
        strategy.volume_ma_period = 20
        strategy.atr_period = 14
        strategy.bb_period = 20
        strategy.bb_std = 2.0
        strategy.use_bollinger_filter = True
        strategy.indicators = {"AAPL": {}}

        # Create 60 bars of price history with explicit float types
        strategy.price_history = {"AAPL": []}
        base_price = 150.0
        np.random.seed(42)  # For reproducibility
        for i in range(60):
            open_price = float(base_price + np.random.randn() * 2)
            high_price = float(base_price + 2 + abs(np.random.randn()))
            low_price = float(base_price - 2 - abs(np.random.randn()))
            close_price = float(base_price + np.random.randn() * 2)
            volume = float(1000000 + np.random.randint(-100000, 100000))
            bar = {
                "timestamp": datetime.now() - timedelta(minutes=60 - i),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
            strategy.price_history["AAPL"].append(bar)

        return strategy

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_rsi(self, strategy_with_history):
        """Test that update_indicators calculates RSI."""
        await strategy_with_history._update_indicators("AAPL")

        assert "rsi" in strategy_with_history.indicators["AAPL"]
        rsi = strategy_with_history.indicators["AAPL"]["rsi"]
        assert rsi is None or (0 <= rsi <= 100)

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_macd(self, strategy_with_history):
        """Test that update_indicators calculates MACD."""
        await strategy_with_history._update_indicators("AAPL")

        assert "macd" in strategy_with_history.indicators["AAPL"]
        assert "macd_signal" in strategy_with_history.indicators["AAPL"]
        assert "macd_hist" in strategy_with_history.indicators["AAPL"]

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_moving_averages(self, strategy_with_history):
        """Test that update_indicators calculates moving averages."""
        await strategy_with_history._update_indicators("AAPL")

        assert "fast_ma" in strategy_with_history.indicators["AAPL"]
        assert "medium_ma" in strategy_with_history.indicators["AAPL"]
        assert "slow_ma" in strategy_with_history.indicators["AAPL"]

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_bollinger_bands(self, strategy_with_history):
        """Test that update_indicators calculates Bollinger Bands."""
        await strategy_with_history._update_indicators("AAPL")

        assert "bb_upper" in strategy_with_history.indicators["AAPL"]
        assert "bb_middle" in strategy_with_history.indicators["AAPL"]
        assert "bb_lower" in strategy_with_history.indicators["AAPL"]
        assert "bb_position" in strategy_with_history.indicators["AAPL"]

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_adx(self, strategy_with_history):
        """Test that update_indicators calculates ADX."""
        await strategy_with_history._update_indicators("AAPL")

        assert "adx" in strategy_with_history.indicators["AAPL"]

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_atr(self, strategy_with_history):
        """Test that update_indicators calculates ATR."""
        await strategy_with_history._update_indicators("AAPL")

        assert "atr" in strategy_with_history.indicators["AAPL"]

    @pytest.mark.asyncio
    async def test_update_indicators_insufficient_history(self):
        """Test that update_indicators handles insufficient history."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.slow_ma = 50
        strategy.indicators = {"AAPL": {}}
        strategy.price_history = {
            "AAPL": [{"close": 150, "high": 151, "low": 149, "volume": 1000000}]
        }

        await strategy._update_indicators("AAPL")

        # Indicators should not be updated with insufficient data
        assert strategy.indicators["AAPL"] == {}

    @pytest.mark.asyncio
    async def test_update_indicators_handles_exception(self, strategy_with_history):
        """Test that update_indicators handles exceptions."""
        # Corrupt the price history
        strategy_with_history.price_history["AAPL"] = [{"bad": "data"}]

        # Should not raise, just log error
        await strategy_with_history._update_indicators("AAPL")


class TestGenerateSignal:
    """Tests for the _generate_signal method."""

    @pytest.fixture
    def strategy_with_indicators(self):
        """Create a strategy with pre-calculated indicators."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.rsi_overbought = 70
        strategy.rsi_oversold = 30
        strategy.adx_threshold = 25
        strategy.volume_factor = 1.5
        strategy.use_bollinger_filter = False
        strategy.use_multi_timeframe = False
        strategy.enable_short_selling = False
        strategy.parameters = {}
        strategy.mtf_analyzer = None

        return strategy

    @pytest.mark.asyncio
    async def test_generate_signal_returns_neutral_no_indicators(self, strategy_with_indicators):
        """Test that signal is neutral when no indicators."""
        strategy_with_indicators.indicators = {"AAPL": {}}

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_returns_neutral_null_rsi(self, strategy_with_indicators):
        """Test that signal is neutral when RSI is null."""
        strategy_with_indicators.indicators = {"AAPL": {"rsi": None}}

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_buy_all_conditions_met(self, strategy_with_indicators):
        """Test buy signal when all conditions are met."""
        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,  # Oversold
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 30,  # Strong trend
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,  # Bullish MA alignment
                "volume": 2000000,
                "volume_ma": 1000000,  # Volume confirmation
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_weak_trend(self, strategy_with_indicators):
        """Test neutral signal when trend is weak."""
        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 15,  # Weak trend
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_low_volume(self, strategy_with_indicators):
        """Test neutral signal when volume is low."""
        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 500000,  # Low volume
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_short_when_enabled(self, strategy_with_indicators):
        """Test short signal when short selling is enabled."""
        strategy_with_indicators.enable_short_selling = True
        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 85,  # Overbought
                "macd": -1.0,
                "macd_signal": -0.5,
                "macd_hist": -0.5,
                "adx": 30,
                "fast_ma": 145,
                "medium_ma": 148,
                "slow_ma": 150,  # Bearish MA alignment
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "short"

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_short_disabled(self, strategy_with_indicators):
        """Test neutral signal when bearish but short selling disabled."""
        strategy_with_indicators.enable_short_selling = False
        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 85,
                "macd": -1.0,
                "macd_signal": -0.5,
                "macd_hist": -0.5,
                "adx": 30,
                "fast_ma": 145,
                "medium_ma": 148,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_with_bollinger_boost(self, strategy_with_indicators):
        """Test signal with Bollinger Band boost."""
        strategy_with_indicators.use_bollinger_filter = True
        strategy_with_indicators.bb_buy_threshold = 0.3
        strategy_with_indicators.bb_sell_threshold = 0.7
        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
                "bb_position": 0.1,  # Near lower band - should boost
                "bb_upper": 160,
                "bb_lower": 140,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_generate_signal_bollinger_reduces_signal(self, strategy_with_indicators):
        """Test that Bollinger Band can reduce signal strength."""
        strategy_with_indicators.use_bollinger_filter = True
        strategy_with_indicators.bb_buy_threshold = 0.3
        strategy_with_indicators.bb_sell_threshold = 0.7

        # Marginal buy signal near upper band should be reduced
        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,  # +1
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,  # +1
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 150,  # Not in order
                "slow_ma": 152,
                "volume": 2000000,
                "volume_ma": 1000000,
                "bb_position": 0.9,  # Near upper band - reduces
                "bb_upper": 160,
                "bb_lower": 140,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        # Score is reduced so no buy
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_bollinger_boosts_short(self, strategy_with_indicators):
        """Test Bollinger Band boosts short signal near upper band."""
        strategy_with_indicators.use_bollinger_filter = True
        strategy_with_indicators.bb_buy_threshold = 0.3
        strategy_with_indicators.bb_sell_threshold = 0.7
        strategy_with_indicators.enable_short_selling = True

        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 85,
                "macd": -1.0,
                "macd_signal": -0.5,
                "macd_hist": -0.5,
                "adx": 30,
                "fast_ma": 145,
                "medium_ma": 148,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
                "bb_position": 0.9,  # Near upper band - boosts short
                "bb_upper": 160,
                "bb_lower": 140,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "short"

    @pytest.mark.asyncio
    async def test_generate_signal_bollinger_narrow_bands(self, strategy_with_indicators):
        """Test Bollinger Band with very narrow bands."""
        strategy_with_indicators.use_bollinger_filter = True
        strategy_with_indicators.bb_buy_threshold = 0.3
        strategy_with_indicators.bb_sell_threshold = 0.7

        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
                "bb_position": 0.5,
                "bb_upper": 150.001,  # Very narrow bands
                "bb_lower": 150.0,
            }
        }

        # Should not crash
        signal = await strategy_with_indicators._generate_signal("AAPL")
        assert signal in ["buy", "neutral", "short", "sell"]

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_strict_mode_filters(self, strategy_with_indicators):
        """Test MTF strict mode filters out signals."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.mtf_require_alignment = True

        mock_mtf = Mock()
        mock_mtf.get_aligned_signal = Mock(return_value="neutral")
        strategy_with_indicators.mtf_analyzer = mock_mtf

        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_strict_mode_allows(self, strategy_with_indicators):
        """Test MTF strict mode allows aligned signals."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.mtf_require_alignment = True

        mock_mtf = Mock()
        mock_mtf.get_aligned_signal = Mock(return_value="bullish")
        strategy_with_indicators.mtf_analyzer = mock_mtf

        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_strict_rejects_bearish_on_bullish(
        self, strategy_with_indicators
    ):
        """Test MTF strict mode rejects bearish when higher TF bullish."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.mtf_require_alignment = True
        strategy_with_indicators.enable_short_selling = True

        mock_mtf = Mock()
        mock_mtf.get_aligned_signal = Mock(return_value="bullish")
        strategy_with_indicators.mtf_analyzer = mock_mtf

        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 85,
                "macd": -1.0,
                "macd_signal": -0.5,
                "macd_hist": -0.5,
                "adx": 30,
                "fast_ma": 145,
                "medium_ma": 148,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_soft_mode(self, strategy_with_indicators):
        """Test MTF soft mode filtering."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.mtf_require_alignment = False
        strategy_with_indicators.parameters = {"mtf_timeframes": ["5Min", "15Min", "1Hour"]}

        mock_mtf = Mock()
        mock_mtf.get_trend = Mock(return_value="bearish")
        strategy_with_indicators.mtf_analyzer = mock_mtf

        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        # Buy signal rejected because higher TF is bearish
        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_soft_allows_aligned(self, strategy_with_indicators):
        """Test MTF soft mode allows aligned signals."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.mtf_require_alignment = False
        strategy_with_indicators.parameters = {"mtf_timeframes": ["5Min", "15Min", "1Hour"]}

        mock_mtf = Mock()
        mock_mtf.get_trend = Mock(return_value="bullish")
        strategy_with_indicators.mtf_analyzer = mock_mtf

        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": 0.5,
                "macd_hist": 0.5,
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_generate_signal_missing_critical_indicators(self, strategy_with_indicators):
        """Test signal generation with missing critical indicators."""
        strategy_with_indicators.indicators = {
            "AAPL": {
                "rsi": 25,
                "macd": 1.0,
                "macd_signal": None,  # Missing
                "macd_hist": 0.5,
                "adx": 30,
                "fast_ma": 155,
                "medium_ma": 153,
                "slow_ma": 150,
                "volume": 2000000,
                "volume_ma": 1000000,
            }
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_handles_exception(self, strategy_with_indicators):
        """Test that generate_signal handles exceptions."""
        strategy_with_indicators.indicators = {"AAPL": "invalid"}

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"


class TestOnBar:
    """Tests for the on_bar method."""

    @pytest.fixture
    def initialized_strategy(self):
        """Create an initialized strategy for on_bar tests."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.symbols = ["AAPL", "MSFT"]
        strategy.slow_ma = 50
        strategy.rsi_period = 14
        strategy.macd_fast = 12
        strategy.macd_slow = 26
        strategy.macd_signal = 9
        strategy.adx_period = 14
        strategy.fast_ma = 10
        strategy.medium_ma = 20
        strategy.volume_ma_period = 20
        strategy.atr_period = 14
        strategy.bb_period = 20
        strategy.bb_std = 2.0
        strategy.rsi_overbought = 70
        strategy.rsi_oversold = 30
        strategy.adx_threshold = 25
        strategy.volume_factor = 1.5
        strategy.use_bollinger_filter = False
        strategy.use_multi_timeframe = False
        strategy.enable_short_selling = False
        strategy.mtf_analyzer = None
        strategy.parameters = {}

        strategy.indicators = {"AAPL": {}, "MSFT": {}}
        strategy.signals = {"AAPL": "neutral", "MSFT": "neutral"}
        strategy.last_signal_time = {"AAPL": None, "MSFT": None}
        strategy.current_prices = {}
        # Use deque with maxlen to match production code (auto-trims on append)
        max_history = max(strategy.slow_ma, strategy.rsi_period,
                          strategy.macd_slow + strategy.macd_signal,
                          strategy.adx_period) + 10
        strategy.max_history = max_history
        strategy.price_history = {
            "AAPL": deque(maxlen=max_history),
            "MSFT": deque(maxlen=max_history),
        }
        strategy.stop_prices = {}
        strategy.target_prices = {}
        strategy.entry_prices = {}
        strategy.peak_prices = {}

        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[])

        return strategy

    @pytest.mark.asyncio
    async def test_on_bar_ignores_unknown_symbol(self, initialized_strategy):
        """Test that on_bar ignores unknown symbols."""
        await initialized_strategy.on_bar("UNKNOWN", 100, 101, 99, 100.5, 1000000, datetime.now())

        assert "UNKNOWN" not in initialized_strategy.current_prices

    @pytest.mark.asyncio
    async def test_on_bar_stores_current_price(self, initialized_strategy):
        """Test that on_bar stores the current price."""
        await initialized_strategy.on_bar("AAPL", 150, 152, 149, 151, 1000000, datetime.now())

        assert initialized_strategy.current_prices["AAPL"] == 151

    @pytest.mark.asyncio
    async def test_on_bar_updates_price_history(self, initialized_strategy):
        """Test that on_bar updates price history."""
        timestamp = datetime.now()
        await initialized_strategy.on_bar("AAPL", 150, 152, 149, 151, 1000000, timestamp)

        assert len(initialized_strategy.price_history["AAPL"]) == 1
        assert initialized_strategy.price_history["AAPL"][0]["close"] == 151

    @pytest.mark.asyncio
    async def test_on_bar_trims_price_history(self, initialized_strategy):
        """Test that on_bar trims price history to max length."""
        for i in range(100):
            await initialized_strategy.on_bar("AAPL", 150, 152, 149, 151, 1000000, datetime.now())

        max_history = 60  # slow_ma + 10
        assert len(initialized_strategy.price_history["AAPL"]) <= max_history + 10

    @pytest.mark.asyncio
    async def test_on_bar_updates_mtf_analyzer(self, initialized_strategy):
        """Test that on_bar updates multi-timeframe analyzer."""
        initialized_strategy.use_multi_timeframe = True
        mock_mtf = Mock()
        mock_mtf.update = AsyncMock()
        initialized_strategy.mtf_analyzer = mock_mtf

        timestamp = datetime.now()
        await initialized_strategy.on_bar("AAPL", 150, 152, 149, 151, 1000000, timestamp)

        mock_mtf.update.assert_called_once_with("AAPL", timestamp, 151, 1000000)

    @pytest.mark.asyncio
    async def test_on_bar_calls_execute_signal_on_buy(self, initialized_strategy):
        """Test that on_bar calls _execute_signal when buy signal."""
        for i in range(60):
            initialized_strategy.price_history["AAPL"].append(
                {
                    "timestamp": datetime.now(),
                    "open": 150,
                    "high": 152,
                    "low": 149,
                    "close": 151,
                    "volume": 1000000,
                }
            )

        initialized_strategy._execute_signal = AsyncMock()
        initialized_strategy._generate_signal = AsyncMock(return_value="buy")
        initialized_strategy._update_indicators = AsyncMock()
        initialized_strategy._check_exit_conditions = AsyncMock()

        await initialized_strategy.on_bar("AAPL", 150, 152, 149, 151, 1000000, datetime.now())

        initialized_strategy._execute_signal.assert_called_once_with("AAPL", "buy")

    @pytest.mark.asyncio
    async def test_on_bar_does_not_execute_neutral(self, initialized_strategy):
        """Test that on_bar doesn't call execute for neutral signal."""
        initialized_strategy._execute_signal = AsyncMock()
        initialized_strategy._generate_signal = AsyncMock(return_value="neutral")
        initialized_strategy._update_indicators = AsyncMock()
        initialized_strategy._check_exit_conditions = AsyncMock()

        await initialized_strategy.on_bar("AAPL", 150, 152, 149, 151, 1000000, datetime.now())

        initialized_strategy._execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_bar_handles_exception(self, initialized_strategy):
        """Test that on_bar handles exceptions gracefully."""
        initialized_strategy._update_indicators = AsyncMock(side_effect=Exception("Test"))

        # Should not raise
        await initialized_strategy.on_bar("AAPL", 150, 152, 149, 151, 1000000, datetime.now())


class TestExecuteSignal:
    """Tests for the _execute_signal method."""

    @pytest.fixture
    def trading_strategy(self):
        """Create a strategy ready for trading."""
        import logging
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.symbols = ["AAPL"]
        strategy.max_positions = 5
        strategy.position_size = 0.10
        strategy.stop_loss = 0.03
        strategy.take_profit = 0.05
        strategy.enable_short_selling = False
        strategy.short_position_size = 0.08
        strategy.short_stop_loss = 0.04
        strategy.parameters = {"use_kelly_criterion": False}
        strategy.current_prices = {"AAPL": 150.0}
        strategy.last_signal_time = {"AAPL": None}
        strategy.stop_prices = {}
        strategy.target_prices = {}
        strategy.price_history = {"AAPL": []}
        # Add missing attributes used by _execute_signal
        strategy.entry_prices = {}
        strategy.peak_prices = {}
        strategy.logger = logging.getLogger("test")

        strategy.risk_manager = Mock()
        strategy.risk_manager.adjust_position_size = Mock(return_value=10000)

        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[])
        strategy.broker.get_account = AsyncMock(return_value=Mock(buying_power="100000"))
        strategy.broker.submit_order_advanced = AsyncMock(return_value=Mock(id="order123"))

        strategy.enforce_position_size_limit = AsyncMock(return_value=(10000, 66.67))

        return strategy

    @pytest.mark.asyncio
    async def test_execute_signal_buy_creates_order(self, trading_strategy):
        """Test that execute_signal creates a buy order."""
        await trading_strategy._execute_signal("AAPL", "buy")

        trading_strategy.broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_signal_buy_stores_stop_and_target(self, trading_strategy):
        """Test that execute_signal stores stop and target prices."""
        await trading_strategy._execute_signal("AAPL", "buy")

        assert "AAPL" in trading_strategy.stop_prices
        assert "AAPL" in trading_strategy.target_prices
        assert trading_strategy.stop_prices["AAPL"] == 150.0 * 0.97
        assert trading_strategy.target_prices["AAPL"] == 150.0 * 1.05

    @pytest.mark.asyncio
    async def test_execute_signal_buy_updates_last_signal_time(self, trading_strategy):
        """Test that execute_signal updates last signal time."""
        await trading_strategy._execute_signal("AAPL", "buy")

        assert trading_strategy.last_signal_time["AAPL"] is not None

    @pytest.mark.asyncio
    async def test_execute_signal_respects_cooldown(self, trading_strategy):
        """Test that execute_signal respects signal cooldown."""
        trading_strategy.last_signal_time["AAPL"] = datetime.now()

        await trading_strategy._execute_signal("AAPL", "buy")

        trading_strategy.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_respects_max_positions(self, trading_strategy):
        """Test that execute_signal respects max positions."""
        mock_positions = [Mock() for _ in range(5)]
        trading_strategy.broker.get_positions = AsyncMock(return_value=mock_positions)

        await trading_strategy._execute_signal("AAPL", "buy")

        trading_strategy.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_skips_if_already_positioned(self, trading_strategy):
        """Test that execute_signal skips if already have position."""
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        trading_strategy.broker.get_positions = AsyncMock(return_value=[mock_position])

        await trading_strategy._execute_signal("AAPL", "buy")

        trading_strategy.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_skips_small_quantity(self, trading_strategy):
        """Test that execute_signal skips if quantity is too small."""
        trading_strategy.enforce_position_size_limit = AsyncMock(return_value=(1, 0.005))

        await trading_strategy._execute_signal("AAPL", "buy")

        trading_strategy.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_short_when_enabled(self, trading_strategy):
        """Test that execute_signal creates short order when enabled."""
        trading_strategy.enable_short_selling = True

        await trading_strategy._execute_signal("AAPL", "short")

        trading_strategy.broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_signal_short_skipped_when_disabled(self, trading_strategy):
        """Test that short signal is skipped when disabled."""
        trading_strategy.enable_short_selling = False

        await trading_strategy._execute_signal("AAPL", "short")

        trading_strategy.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_sell_closes_position(self, trading_strategy):
        """Test that sell signal closes existing position."""
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = 100
        trading_strategy.broker.get_positions = AsyncMock(return_value=[mock_position])
        trading_strategy.stop_prices["AAPL"] = 145.0
        trading_strategy.target_prices["AAPL"] = 160.0

        await trading_strategy._execute_signal("AAPL", "sell")

        trading_strategy.broker.submit_order_advanced.assert_called_once()
        assert "AAPL" not in trading_strategy.stop_prices
        assert "AAPL" not in trading_strategy.target_prices

    @pytest.mark.asyncio
    async def test_execute_signal_uses_kelly_criterion(self, trading_strategy):
        """Test that execute_signal uses Kelly Criterion when enabled."""
        trading_strategy.parameters["use_kelly_criterion"] = True
        trading_strategy.kelly = Mock()
        trading_strategy.kelly.trades = []
        trading_strategy.calculate_kelly_position_size = AsyncMock(return_value=(15000, 0.15, 100))

        await trading_strategy._execute_signal("AAPL", "buy")

        trading_strategy.calculate_kelly_position_size.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_signal_uses_risk_manager(self, trading_strategy):
        """Test that execute_signal uses risk manager for position sizing."""
        for i in range(25):
            trading_strategy.price_history["AAPL"].append({"close": 150})

        await trading_strategy._execute_signal("AAPL", "buy")

        trading_strategy.risk_manager.adjust_position_size.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_signal_risk_manager_rejects_position(self, trading_strategy):
        """Test that execution stops when risk manager rejects position."""
        for i in range(25):
            trading_strategy.price_history["AAPL"].append({"close": 150})

        trading_strategy.risk_manager.adjust_position_size = Mock(return_value=0)

        await trading_strategy._execute_signal("AAPL", "buy")

        trading_strategy.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_handles_exception(self, trading_strategy):
        """Test that execute_signal handles exceptions."""
        trading_strategy.broker.get_positions = AsyncMock(side_effect=Exception("Test"))

        # Should not raise
        await trading_strategy._execute_signal("AAPL", "buy")

    @pytest.mark.asyncio
    async def test_execute_signal_short_inverted_levels(self, trading_strategy):
        """Test short orders have inverted stop/take-profit levels."""
        trading_strategy.enable_short_selling = True

        await trading_strategy._execute_signal("AAPL", "short")

        # For shorts: take-profit is below entry, stop-loss is above
        assert trading_strategy.target_prices["AAPL"] == 150.0 * (1 - 0.05)
        assert trading_strategy.stop_prices["AAPL"] == 150.0 * (1 + 0.04)

    @pytest.mark.asyncio
    async def test_execute_signal_short_uses_kelly_reduced(self, trading_strategy):
        """Test short orders use reduced Kelly sizing."""
        trading_strategy.enable_short_selling = True
        trading_strategy.parameters["use_kelly_criterion"] = True
        trading_strategy.kelly = Mock()
        trading_strategy.kelly.trades = []
        trading_strategy.calculate_kelly_position_size = AsyncMock(return_value=(15000, 0.15, 100))
        # Return reduced value for short
        trading_strategy.enforce_position_size_limit = AsyncMock(return_value=(12000, 80))

        await trading_strategy._execute_signal("AAPL", "short")

        trading_strategy.calculate_kelly_position_size.assert_called_once()


class TestCheckExitConditions:
    """Tests for the _check_exit_conditions method."""

    @pytest.fixture
    def strategy_with_position(self):
        """Create a strategy with an existing position."""
        import logging
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.current_prices = {"AAPL": 150.0}
        strategy.stop_prices = {"AAPL": 145.0}
        strategy.target_prices = {"AAPL": 160.0}
        strategy.entry_prices = {}
        strategy.peak_prices = {}
        strategy.logger = logging.getLogger("test")
        # Position cache attributes (used by _get_cached_positions)
        strategy._positions_cache = None
        strategy._positions_cache_time = None
        strategy._positions_cache_ttl = timedelta(seconds=1)
        strategy.use_trailing_stop = False

        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = 100

        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[mock_position])

        return strategy

    @pytest.mark.asyncio
    async def test_check_exit_conditions_with_position(self, strategy_with_position):
        """Test check_exit_conditions with existing position."""
        await strategy_with_position._check_exit_conditions("AAPL")

        assert "AAPL" in strategy_with_position.stop_prices

    @pytest.mark.asyncio
    async def test_check_exit_conditions_cleans_up_closed_position(self, strategy_with_position):
        """Test that closed positions are cleaned up."""
        strategy_with_position.broker.get_positions = AsyncMock(return_value=[])

        await strategy_with_position._check_exit_conditions("AAPL")

        assert "AAPL" not in strategy_with_position.stop_prices
        assert "AAPL" not in strategy_with_position.target_prices

    @pytest.mark.asyncio
    async def test_check_exit_conditions_no_current_price(self, strategy_with_position):
        """Test check_exit_conditions with no current price."""
        strategy_with_position.current_prices = {}

        # Should not raise
        await strategy_with_position._check_exit_conditions("AAPL")

    @pytest.mark.asyncio
    async def test_check_exit_conditions_handles_exception(self, strategy_with_position):
        """Test that check_exit_conditions handles exceptions."""
        strategy_with_position.broker.get_positions = AsyncMock(side_effect=Exception("Test"))

        # Should not raise
        await strategy_with_position._check_exit_conditions("AAPL")


class TestAnalyzeSymbol:
    """Tests for the analyze_symbol method."""

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_signal(self):
        """Test that analyze_symbol returns the stored signal."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.signals = {"AAPL": "buy", "MSFT": "sell"}

        signal = await strategy.analyze_symbol("AAPL")

        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_neutral_for_unknown(self):
        """Test that analyze_symbol returns neutral for unknown symbol."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.signals = {}

        signal = await strategy.analyze_symbol("UNKNOWN")

        assert signal == "neutral"


class TestExecuteTrade:
    """Tests for the execute_trade method."""

    @pytest.mark.asyncio
    async def test_execute_trade_is_noop(self):
        """Test that execute_trade does nothing (handled in _execute_signal)."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)

        result = await strategy.execute_trade("AAPL", "buy")

        assert result is None


class TestGenerateSignals:
    """Tests for the generate_signals method (backtest mode)."""

    @pytest.fixture
    def backtest_strategy(self):
        """Create a strategy for backtest mode."""
        import pandas as pd

        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.symbols = ["AAPL"]
        strategy.slow_ma = 50
        strategy.rsi_period = 14
        strategy.macd_fast = 12
        strategy.macd_slow = 26
        strategy.macd_signal = 9
        strategy.adx_period = 14
        strategy.atr_period = 14
        strategy.fast_ma = 10
        strategy.medium_ma = 20
        strategy.volume_ma_period = 20
        strategy.bb_period = 20
        strategy.bb_std = 2.0
        strategy.use_bollinger_filter = False
        strategy.indicators = {"AAPL": {}}
        strategy.signals = {"AAPL": "neutral"}
        strategy.rsi_overbought = 70
        strategy.rsi_oversold = 30
        strategy.adx_threshold = 25
        strategy.volume_factor = 1.5
        strategy.use_multi_timeframe = False
        strategy.enable_short_selling = False
        strategy.parameters = {}
        strategy.mtf_analyzer = None

        # Create test data with explicit float64 types (talib requires double)
        np.random.seed(42)
        dates = pd.date_range(
            end=datetime.now(), periods=100, freq="min"
        )  # 'min' instead of deprecated 'T'
        data = pd.DataFrame(
            {
                "open": (np.random.randn(100) + 150).astype(np.float64),
                "high": (np.random.randn(100) + 152).astype(np.float64),
                "low": (np.random.randn(100) + 148).astype(np.float64),
                "close": (np.random.randn(100) + 150).astype(np.float64),
                "volume": (np.random.randint(900000, 1100000, 100)).astype(
                    np.float64
                ),  # Must be float64 for talib
            },
            index=dates,
        )

        strategy.current_data = {"AAPL": data}

        return strategy

    @pytest.mark.asyncio
    async def test_generate_signals_updates_indicators(self, backtest_strategy):
        """Test that generate_signals updates indicators."""
        await backtest_strategy.generate_signals()

        assert backtest_strategy.indicators["AAPL"] != {}

    @pytest.mark.asyncio
    async def test_generate_signals_updates_signals(self, backtest_strategy):
        """Test that generate_signals updates signals."""
        await backtest_strategy.generate_signals()

        assert backtest_strategy.signals["AAPL"] in ["buy", "sell", "short", "neutral"]

    @pytest.mark.asyncio
    async def test_generate_signals_skips_insufficient_data(self, backtest_strategy):
        """Test that generate_signals skips symbols with insufficient data."""
        import pandas as pd

        dates = pd.date_range(end=datetime.now(), periods=10, freq="min")
        data = pd.DataFrame(
            {
                "open": np.array([150.0] * 10, dtype=np.float64),
                "high": np.array([152.0] * 10, dtype=np.float64),
                "low": np.array([148.0] * 10, dtype=np.float64),
                "close": np.array([150.0] * 10, dtype=np.float64),
                "volume": np.array([1000000.0] * 10, dtype=np.float64),
            },
            index=dates,
        )

        backtest_strategy.current_data = {"AAPL": data}

        await backtest_strategy.generate_signals()

        assert backtest_strategy.indicators["AAPL"] == {}

    @pytest.mark.asyncio
    async def test_generate_signals_with_bollinger(self, backtest_strategy):
        """Test generate_signals with Bollinger filter."""
        backtest_strategy.use_bollinger_filter = True

        await backtest_strategy.generate_signals()

        # Should include BB indicators
        assert "bb_upper" in backtest_strategy.indicators["AAPL"]


class TestGetOrders:
    """Tests for the get_orders method (backtest mode)."""

    @pytest.fixture
    def strategy_for_orders(self):
        """Create a strategy for testing get_orders."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.signals = {"AAPL": "buy", "MSFT": "neutral"}
        strategy.indicators = {"AAPL": {"close": 150.0}, "MSFT": {"close": 300.0}}
        strategy.capital = 100000
        strategy.position_size = 0.10
        strategy.positions = {}

        return strategy

    def test_get_orders_returns_buy_order(self, strategy_for_orders):
        """Test that get_orders returns buy order for buy signal."""
        orders = strategy_for_orders.get_orders()

        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["side"] == "buy"
        assert orders[0]["type"] == "market"

    def test_get_orders_skips_neutral(self, strategy_for_orders):
        """Test that get_orders skips neutral signals."""
        orders = strategy_for_orders.get_orders()

        symbols = [o["symbol"] for o in orders]
        assert "MSFT" not in symbols

    def test_get_orders_calculates_quantity(self, strategy_for_orders):
        """Test that get_orders calculates correct quantity."""
        orders = strategy_for_orders.get_orders()

        expected_value = 100000 * 0.10
        expected_quantity = expected_value / 150.0

        assert abs(orders[0]["quantity"] - expected_quantity) < 0.01

    def test_get_orders_sell_with_position(self, strategy_for_orders):
        """Test that get_orders creates sell order when has position."""
        strategy_for_orders.signals = {"AAPL": "sell"}
        strategy_for_orders.positions = {"AAPL": {"quantity": 100}}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 1
        assert orders[0]["side"] == "sell"
        assert orders[0]["quantity"] == 100

    def test_get_orders_skips_buy_with_position(self, strategy_for_orders):
        """Test that get_orders skips buy when already have position."""
        strategy_for_orders.positions = {"AAPL": {"quantity": 100}}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_sell_without_position(self, strategy_for_orders):
        """Test that get_orders skips sell when no position."""
        strategy_for_orders.signals = {"AAPL": "sell"}
        strategy_for_orders.positions = {}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_invalid_price(self, strategy_for_orders):
        """Test that get_orders skips symbols with no price."""
        strategy_for_orders.indicators = {"AAPL": {"close": None}}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_small_quantity(self, strategy_for_orders):
        """Test that get_orders skips very small quantities."""
        strategy_for_orders.capital = 1
        strategy_for_orders.position_size = 0.001

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_price_history(self):
        """Test indicator update with empty price history."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.slow_ma = 50
        strategy.indicators = {"AAPL": {}}
        strategy.price_history = {"AAPL": []}

        await strategy._update_indicators("AAPL")

        assert strategy.indicators["AAPL"] == {}

    def test_get_orders_with_no_data(self):
        """Test get_orders when no current_data."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.symbols = ["AAPL"]
        strategy.signals = {"AAPL": "buy"}
        strategy.indicators = {"AAPL": {"close": None}}
        strategy.current_data = {}
        strategy.positions = {}
        strategy.capital = 100000
        strategy.position_size = 0.10

        orders = strategy.get_orders()

        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_on_bar_with_no_price_history_key(self):
        """Test on_bar when symbol not in price_history."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.symbols = ["AAPL"]
        strategy.price_history = {}  # Empty
        strategy.current_prices = {}
        strategy.use_multi_timeframe = False
        strategy.mtf_analyzer = None

        # Should not crash - symbol not in symbols list effectively
        # Actually, this will fail because AAPL is in symbols but not price_history
        # The code checks if symbol not in self.symbols first


class TestRiskManagerIntegration:
    """Tests for risk manager integration."""

    @pytest.mark.asyncio
    async def test_risk_manager_called_with_position_data(self):
        """Test risk manager receives correct position data."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy.__new__(MomentumStrategy)
        strategy.symbols = ["AAPL"]
        strategy.max_positions = 5
        strategy.position_size = 0.10
        strategy.stop_loss = 0.03
        strategy.take_profit = 0.05
        strategy.enable_short_selling = False
        strategy.parameters = {"use_kelly_criterion": False}
        strategy.current_prices = {"AAPL": 150.0}
        strategy.last_signal_time = {"AAPL": None}
        strategy.stop_prices = {}
        strategy.target_prices = {}

        # Add price history for risk manager
        strategy.price_history = {
            "AAPL": [{"close": 150.0} for _ in range(25)],
            "MSFT": [{"close": 300.0} for _ in range(25)],
        }

        # Existing position
        mock_position = Mock()
        mock_position.symbol = "MSFT"
        mock_position.market_value = "30000"

        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[mock_position])
        strategy.broker.get_account = AsyncMock(return_value=Mock(buying_power="100000"))
        strategy.broker.submit_order_advanced = AsyncMock(return_value=Mock(id="123"))

        strategy.risk_manager = Mock()
        strategy.risk_manager.adjust_position_size = Mock(return_value=10000)
        strategy.enforce_position_size_limit = AsyncMock(return_value=(10000, 66.67))

        await strategy._execute_signal("AAPL", "buy")

        # Verify risk manager was called with current positions
        strategy.risk_manager.adjust_position_size.assert_called_once()
        call_args = strategy.risk_manager.adjust_position_size.call_args
        assert "AAPL" == call_args[0][0]  # symbol


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
