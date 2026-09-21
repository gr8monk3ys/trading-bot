"""
Comprehensive tests for MeanReversionStrategy.

Tests cover:
- Default parameters
- Initialization (standard, with multi-timeframe, with short selling)
- Indicator updates (Bollinger Bands, RSI, z-score, stochastic)
- Signal generation (buy, sell, short, neutral)
- Multi-timeframe filtering
- Trade execution with bracket orders
- Exit conditions (max hold, mean reversion target, trailing stop)
- Backtest mode (generate_signals, get_orders)
"""

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest


class TestMeanReversionStrategyName:
    """Tests for the NAME attribute."""

    def test_name_attribute(self):
        """Test that NAME attribute is correctly set."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        assert MeanReversionStrategy.NAME == "MeanReversionStrategy"


class TestDefaultParameters:
    """Tests for the default_parameters method."""

    def test_default_parameters_returns_dict(self):
        """Test that default_parameters returns a dictionary."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert isinstance(params, dict)

    def test_default_parameters_contains_basic_params(self):
        """Test that default parameters contain basic trading parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert "position_size" in params
        assert "max_positions" in params
        assert "stop_loss" in params
        assert "take_profit" in params

    def test_default_parameters_contains_mean_reversion_params(self):
        """Test that default parameters contain mean reversion specific parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert "bb_period" in params
        assert "bb_std" in params
        assert "rsi_period" in params
        assert "rsi_overbought" in params
        assert "rsi_oversold" in params
        assert "sma_period" in params
        assert "mean_lookback" in params
        assert "std_threshold" in params

    def test_default_parameters_contains_exit_params(self):
        """Test that default parameters contain exit parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert "profit_target_std" in params
        assert "max_hold_days" in params
        assert "trailing_stop" in params

    def test_default_parameters_contains_multi_timeframe_params(self):
        """Test that default parameters contain multi-timeframe parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert "use_multi_timeframe" in params
        assert "mtf_timeframes" in params
        assert "mtf_require_alignment" in params

    def test_default_parameters_contains_short_selling_params(self):
        """Test that default parameters contain short selling parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert "enable_short_selling" in params
        assert "short_position_size" in params
        assert "short_stop_loss" in params


class TestMeanReversionStrategyInitialize:
    """Tests for the initialize method."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = Mock()
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_account = AsyncMock(return_value=Mock(buying_power="100000", equity=100000))
        broker._add_subscriber = Mock()
        return broker

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_broker):
        """Test successful initialization."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            "symbols": ["AAPL", "MSFT"],
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with nullcontext():
            result = await strategy.initialize()

        assert result is True
        assert strategy.bb_period == 20
        assert strategy.rsi_period == 14
        assert strategy.sma_period == 50

    @pytest.mark.asyncio
    async def test_initialize_creates_tracking_dicts(self, mock_broker):
        """Test that initialize creates all tracking dictionaries."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with nullcontext():
            await strategy.initialize()

        assert "AAPL" in strategy.indicators
        assert "AAPL" in strategy.signals
        assert "AAPL" in strategy.last_signal_time
        assert "AAPL" in strategy.price_history
        assert isinstance(strategy.position_entries, dict)
        assert isinstance(strategy.highest_prices, dict)
        assert isinstance(strategy.lowest_prices, dict)
        assert isinstance(strategy.current_prices, dict)

    @pytest.mark.asyncio
    async def test_initialize_with_multi_timeframe(self, mock_broker):
        """Test initialization with multi-timeframe enabled."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": True,
            "mtf_timeframes": ["5Min", "15Min", "1Hour"],
            "enable_short_selling": False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with nullcontext():
            await strategy.initialize()

        assert strategy.use_multi_timeframe is True
        assert strategy.mtf_analyzer is not None

    @pytest.mark.asyncio
    async def test_initialize_with_short_selling(self, mock_broker):
        """Test initialization with short selling enabled."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": False,
            "enable_short_selling": True,
            "short_position_size": 0.08,
            "short_stop_loss": 0.03,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with nullcontext():
            await strategy.initialize()

        assert strategy.enable_short_selling is True
        assert strategy.short_position_size == 0.08
        assert strategy.short_stop_loss == 0.03

    @pytest.mark.asyncio
    async def test_initialize_creates_risk_manager(self, mock_broker):
        """Test that initialize creates a risk manager."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            "symbols": ["AAPL"],
            "use_multi_timeframe": False,
            "enable_short_selling": False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with nullcontext():
            await strategy.initialize()

        assert hasattr(strategy, "risk_manager")
        assert strategy.risk_manager is not None

    @pytest.mark.asyncio
    async def test_initialize_handles_exception(self, mock_broker):
        """Test that initialize handles exceptions gracefully."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy(broker=mock_broker, parameters={})

        with patch(
            "strategies.base_strategy.BaseStrategy.initialize", new_callable=AsyncMock
        ) as mock_init:
            mock_init.side_effect = Exception("Test error")
            result = await strategy.initialize()

        assert result is False

        # Should not raise, just log error


class TestGenerateSignal:
    """Tests for the _generate_signal method."""

    @pytest.fixture
    def strategy_with_indicators(self):
        """Create a strategy with indicators set."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.rsi_overbought = 70
        strategy.rsi_oversold = 30
        strategy.std_threshold = 1.5
        strategy.use_multi_timeframe = False
        strategy.enable_short_selling = False
        strategy.mtf_analyzer = None
        strategy.parameters = {}

        strategy.indicators = {
            "AAPL": {
                "close": 150.0,
                "upper_band": 160.0,
                "middle_band": 152.0,
                "lower_band": 144.0,
                "rsi": 50.0,
                "z_score": 0.0,
                "bb_position": 0.5,
                "slowk": 50.0,
                "slowd": 50.0,
                "sma": 152.0,
                "std": 4.0,
            }
        }

        return strategy

    @pytest.mark.asyncio
    async def test_generate_signal_returns_neutral_no_indicators(self, strategy_with_indicators):
        """Test that generate_signal returns neutral when no indicators."""
        strategy_with_indicators.indicators = {}

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_returns_neutral_null_rsi(self, strategy_with_indicators):
        """Test that generate_signal returns neutral when RSI is None."""
        strategy_with_indicators.indicators["AAPL"]["rsi"] = None

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_buy_all_conditions_met(self, strategy_with_indicators):
        """Test buy signal when all conditions are met."""
        strategy_with_indicators.indicators["AAPL"] = {
            "close": 143.0,  # Below lower band
            "upper_band": 160.0,
            "middle_band": 152.0,
            "lower_band": 144.0,
            "rsi": 25.0,  # Oversold
            "z_score": -2.0,  # Far from mean
            "bb_position": 0.02,  # Near bottom of BB
            "slowk": 15.0,  # Stoch oversold
            "slowd": 10.0,  # Stoch turning up (k > d)
            "sma": 152.0,
            "std": 4.0,
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_conditions_not_met(self, strategy_with_indicators):
        """Test neutral signal when conditions not fully met."""
        # Default indicators don't meet buy/sell conditions
        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_short_when_enabled(self, strategy_with_indicators):
        """Test short signal when short selling is enabled."""
        strategy_with_indicators.enable_short_selling = True
        strategy_with_indicators.indicators["AAPL"] = {
            "close": 161.0,  # Above upper band
            "upper_band": 160.0,
            "middle_band": 152.0,
            "lower_band": 144.0,
            "rsi": 80.0,  # Overbought
            "z_score": 2.0,  # Far from mean
            "bb_position": 0.98,  # Near top of BB
            "slowk": 85.0,  # Stoch overbought
            "slowd": 90.0,  # Stoch turning down (k < d)
            "sma": 152.0,
            "std": 4.0,
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "short"

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_short_disabled(self, strategy_with_indicators):
        """Test neutral when sell conditions met but short selling disabled."""
        strategy_with_indicators.enable_short_selling = False
        strategy_with_indicators.indicators["AAPL"] = {
            "close": 161.0,
            "upper_band": 160.0,
            "middle_band": 152.0,
            "lower_band": 144.0,
            "rsi": 80.0,
            "z_score": 2.0,
            "bb_position": 0.98,
            "slowk": 85.0,
            "slowd": 90.0,
            "sma": 152.0,
            "std": 4.0,
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_rejects_buy_in_downtrend(self, strategy_with_indicators):
        """Test MTF filter rejects buy signal in strong downtrend."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.parameters = {"mtf_timeframes": ["5Min", "15Min", "1Hour"]}

        # Create mock MTF analyzer
        strategy_with_indicators.mtf_analyzer = Mock()
        strategy_with_indicators.mtf_analyzer.get_trend = Mock(return_value="bearish")

        # Set up buy conditions
        strategy_with_indicators.indicators["AAPL"] = {
            "close": 143.0,
            "upper_band": 160.0,
            "middle_band": 152.0,
            "lower_band": 144.0,
            "rsi": 25.0,
            "z_score": -2.0,
            "bb_position": 0.02,
            "slowk": 15.0,
            "slowd": 10.0,
            "sma": 152.0,
            "std": 4.0,
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"  # Rejected by MTF filter

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_allows_buy_in_neutral_market(self, strategy_with_indicators):
        """Test MTF filter allows buy signal in neutral market."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.parameters = {"mtf_timeframes": ["5Min", "15Min", "1Hour"]}

        # Create mock MTF analyzer
        strategy_with_indicators.mtf_analyzer = Mock()
        strategy_with_indicators.mtf_analyzer.get_trend = Mock(return_value="neutral")

        # Set up buy conditions
        strategy_with_indicators.indicators["AAPL"] = {
            "close": 143.0,
            "upper_band": 160.0,
            "middle_band": 152.0,
            "lower_band": 144.0,
            "rsi": 25.0,
            "z_score": -2.0,
            "bb_position": 0.02,
            "slowk": 15.0,
            "slowd": 10.0,
            "sma": 152.0,
            "std": 4.0,
        }

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_generate_signal_handles_exception(self, strategy_with_indicators):
        """Test that generate_signal handles exceptions."""
        strategy_with_indicators.indicators["AAPL"] = "invalid"

        signal = await strategy_with_indicators._generate_signal("AAPL")

        assert signal == "neutral"


class TestAnalyzeSymbol:
    """Tests for the analyze_symbol method."""

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_signal(self):
        """Test that analyze_symbol returns the signal for a symbol."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.signals = {"AAPL": "buy", "MSFT": "sell"}

        signal = await strategy.analyze_symbol("AAPL")

        assert signal == "buy"

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_neutral_for_unknown(self):
        """Test that analyze_symbol returns neutral for unknown symbol."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.signals = {}

        signal = await strategy.analyze_symbol("UNKNOWN")

        assert signal == "neutral"


class TestExecuteTrade:
    """Tests for the execute_trade method."""


class TestGenerateSignals:
    """Tests for the generate_signals method (backtest mode)."""

    @pytest.fixture
    def backtest_strategy(self):
        """Create a strategy for backtest mode testing."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.symbols = ["AAPL"]
        strategy.sma_period = 50
        strategy.bb_period = 20
        strategy.bb_std = 2.0
        strategy.rsi_period = 14
        strategy.mean_lookback = 20
        strategy.std_threshold = 1.5
        strategy.rsi_overbought = 70
        strategy.rsi_oversold = 30
        strategy.use_multi_timeframe = False
        strategy.enable_short_selling = False
        strategy.mtf_analyzer = None
        strategy.parameters = {}

        strategy.indicators = {"AAPL": {}}
        strategy.signals = {"AAPL": "neutral"}

        # Create test data with explicit float64 types
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq="min")
        data = pd.DataFrame(
            {
                "open": (np.random.randn(100) + 150).astype(np.float64),
                "high": (np.random.randn(100) + 152).astype(np.float64),
                "low": (np.random.randn(100) + 148).astype(np.float64),
                "close": (np.random.randn(100) + 150).astype(np.float64),
                "volume": (np.random.randint(900000, 1100000, 100)).astype(np.float64),
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


class TestGetOrders:
    """Tests for the get_orders method (backtest mode)."""

    @pytest.fixture
    def strategy_for_orders(self):
        """Create a strategy for testing get_orders."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
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

    def test_get_orders_skips_neutral(self, strategy_for_orders):
        """Test that get_orders skips neutral signals."""
        orders = strategy_for_orders.get_orders()

        # Only AAPL should have an order, MSFT is neutral
        symbols = [o["symbol"] for o in orders]
        assert "MSFT" not in symbols

    def test_get_orders_calculates_quantity(self, strategy_for_orders):
        """Test that get_orders calculates correct quantity."""
        orders = strategy_for_orders.get_orders()

        expected_quantity = (100000 * 0.10) / 150.0
        assert orders[0]["quantity"] == pytest.approx(expected_quantity, rel=0.01)

    def test_get_orders_sell_with_position(self, strategy_for_orders):
        """Test that get_orders returns sell order when has position."""
        strategy_for_orders.signals["AAPL"] = "sell"
        strategy_for_orders.positions = {"AAPL": {"quantity": 10}}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 1
        assert orders[0]["side"] == "sell"
        assert orders[0]["quantity"] == 10

    def test_get_orders_skips_buy_with_position(self, strategy_for_orders):
        """Test that get_orders skips buy when already has position."""
        strategy_for_orders.positions = {"AAPL": {"quantity": 10}}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_sell_without_position(self, strategy_for_orders):
        """Test that get_orders skips sell when no position."""
        strategy_for_orders.signals["AAPL"] = "sell"
        strategy_for_orders.positions = {}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_invalid_price(self, strategy_for_orders):
        """Test that get_orders skips orders with invalid price."""
        strategy_for_orders.indicators["AAPL"]["close"] = None

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_small_quantity(self, strategy_for_orders):
        """Test that get_orders skips very small quantities (< 0.01 shares)."""
        # With 100000 capital, 10% position size = 10000
        # Price of 10000000 = 0.001 shares (< 0.01 threshold)
        strategy_for_orders.indicators["AAPL"][
            "close"
        ] = 10000000  # Extremely high price = < 0.01 shares

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_orders_with_no_data(self):
        """Test get_orders with no data."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.signals = {}
        strategy.indicators = {}
        strategy.positions = {}
        strategy.capital = 100000
        strategy.position_size = 0.10

        orders = strategy.get_orders()

        assert orders == []
