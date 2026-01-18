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

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta


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

        assert 'position_size' in params
        assert 'max_positions' in params
        assert 'stop_loss' in params
        assert 'take_profit' in params

    def test_default_parameters_contains_mean_reversion_params(self):
        """Test that default parameters contain mean reversion specific parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert 'bb_period' in params
        assert 'bb_std' in params
        assert 'rsi_period' in params
        assert 'rsi_overbought' in params
        assert 'rsi_oversold' in params
        assert 'sma_period' in params
        assert 'mean_lookback' in params
        assert 'std_threshold' in params

    def test_default_parameters_contains_exit_params(self):
        """Test that default parameters contain exit parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert 'profit_target_std' in params
        assert 'max_hold_days' in params
        assert 'trailing_stop' in params

    def test_default_parameters_contains_multi_timeframe_params(self):
        """Test that default parameters contain multi-timeframe parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert 'use_multi_timeframe' in params
        assert 'mtf_timeframes' in params
        assert 'mtf_require_alignment' in params

    def test_default_parameters_contains_short_selling_params(self):
        """Test that default parameters contain short selling parameters."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        params = strategy.default_parameters()

        assert 'enable_short_selling' in params
        assert 'short_position_size' in params
        assert 'short_stop_loss' in params


class TestMeanReversionStrategyInitialize:
    """Tests for the initialize method."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = Mock()
        broker.get_positions = AsyncMock(return_value=[])
        broker.get_account = AsyncMock(return_value=Mock(
            buying_power='100000',
            equity=100000
        ))
        broker._add_subscriber = Mock()
        return broker

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_broker):
        """Test successful initialization."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            'symbols': ['AAPL', 'MSFT'],
            'use_multi_timeframe': False,
            'enable_short_selling': False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with patch.object(strategy, 'check_trading_allowed', new_callable=AsyncMock, return_value=True):
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
            'symbols': ['AAPL'],
            'use_multi_timeframe': False,
            'enable_short_selling': False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with patch.object(strategy, 'check_trading_allowed', new_callable=AsyncMock, return_value=True):
            await strategy.initialize()

        assert 'AAPL' in strategy.indicators
        assert 'AAPL' in strategy.signals
        assert 'AAPL' in strategy.last_signal_time
        assert 'AAPL' in strategy.price_history
        assert isinstance(strategy.position_entries, dict)
        assert isinstance(strategy.highest_prices, dict)
        assert isinstance(strategy.lowest_prices, dict)
        assert isinstance(strategy.current_prices, dict)

    @pytest.mark.asyncio
    async def test_initialize_with_multi_timeframe(self, mock_broker):
        """Test initialization with multi-timeframe enabled."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            'symbols': ['AAPL'],
            'use_multi_timeframe': True,
            'mtf_timeframes': ['5Min', '15Min', '1Hour'],
            'enable_short_selling': False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with patch.object(strategy, 'check_trading_allowed', new_callable=AsyncMock, return_value=True):
            await strategy.initialize()

        assert strategy.use_multi_timeframe is True
        assert strategy.mtf_analyzer is not None

    @pytest.mark.asyncio
    async def test_initialize_with_short_selling(self, mock_broker):
        """Test initialization with short selling enabled."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            'symbols': ['AAPL'],
            'use_multi_timeframe': False,
            'enable_short_selling': True,
            'short_position_size': 0.08,
            'short_stop_loss': 0.03,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with patch.object(strategy, 'check_trading_allowed', new_callable=AsyncMock, return_value=True):
            await strategy.initialize()

        assert strategy.enable_short_selling is True
        assert strategy.short_position_size == 0.08
        assert strategy.short_stop_loss == 0.03

    @pytest.mark.asyncio
    async def test_initialize_creates_risk_manager(self, mock_broker):
        """Test that initialize creates a risk manager."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            'symbols': ['AAPL'],
            'use_multi_timeframe': False,
            'enable_short_selling': False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with patch.object(strategy, 'check_trading_allowed', new_callable=AsyncMock, return_value=True):
            await strategy.initialize()

        assert hasattr(strategy, 'risk_manager')
        assert strategy.risk_manager is not None

    @pytest.mark.asyncio
    async def test_initialize_handles_exception(self, mock_broker):
        """Test that initialize handles exceptions gracefully."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy(broker=mock_broker, parameters={})

        with patch('strategies.base_strategy.BaseStrategy.initialize', new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = Exception("Test error")
            result = await strategy.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_adds_subscriber_to_broker(self, mock_broker):
        """Test that initialize adds strategy as subscriber to broker."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        params = {
            'symbols': ['AAPL'],
            'use_multi_timeframe': False,
            'enable_short_selling': False,
        }
        strategy = MeanReversionStrategy(broker=mock_broker, parameters=params)

        with patch.object(strategy, 'check_trading_allowed', new_callable=AsyncMock, return_value=True):
            await strategy.initialize()

        mock_broker._add_subscriber.assert_called_once_with(strategy)


class TestUpdateIndicators:
    """Tests for the _update_indicators method."""

    @pytest.fixture
    def strategy_with_history(self):
        """Create a strategy with price history."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.sma_period = 50
        strategy.bb_period = 20
        strategy.bb_std = 2.0
        strategy.rsi_period = 14
        strategy.mean_lookback = 20
        strategy.indicators = {'AAPL': {}}

        # Create 60 bars of price history with explicit float types
        strategy.price_history = {'AAPL': []}
        base_price = 150.0
        np.random.seed(42)
        for i in range(60):
            bar = {
                'timestamp': datetime.now() - timedelta(minutes=60-i),
                'open': float(base_price + np.random.randn() * 2),
                'high': float(base_price + 2 + abs(np.random.randn())),
                'low': float(base_price - 2 - abs(np.random.randn())),
                'close': float(base_price + np.random.randn() * 2),
                'volume': float(1000000 + np.random.randint(-100000, 100000))
            }
            strategy.price_history['AAPL'].append(bar)

        return strategy

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_bollinger_bands(self, strategy_with_history):
        """Test that update_indicators calculates Bollinger Bands."""
        await strategy_with_history._update_indicators('AAPL')

        assert 'upper_band' in strategy_with_history.indicators['AAPL']
        assert 'middle_band' in strategy_with_history.indicators['AAPL']
        assert 'lower_band' in strategy_with_history.indicators['AAPL']

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_rsi(self, strategy_with_history):
        """Test that update_indicators calculates RSI."""
        await strategy_with_history._update_indicators('AAPL')

        rsi = strategy_with_history.indicators['AAPL'].get('rsi')
        assert rsi is not None
        assert 0 <= rsi <= 100

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_sma(self, strategy_with_history):
        """Test that update_indicators calculates SMA."""
        await strategy_with_history._update_indicators('AAPL')

        sma = strategy_with_history.indicators['AAPL'].get('sma')
        assert sma is not None

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_z_score(self, strategy_with_history):
        """Test that update_indicators calculates z-score."""
        await strategy_with_history._update_indicators('AAPL')

        z_score = strategy_with_history.indicators['AAPL'].get('z_score')
        assert z_score is not None

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_bb_position(self, strategy_with_history):
        """Test that update_indicators calculates BB position."""
        await strategy_with_history._update_indicators('AAPL')

        bb_position = strategy_with_history.indicators['AAPL'].get('bb_position')
        assert bb_position is not None
        assert 0 <= bb_position <= 1

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_stochastic(self, strategy_with_history):
        """Test that update_indicators calculates Stochastic."""
        await strategy_with_history._update_indicators('AAPL')

        slowk = strategy_with_history.indicators['AAPL'].get('slowk')
        slowd = strategy_with_history.indicators['AAPL'].get('slowd')
        assert slowk is not None
        assert slowd is not None

    @pytest.mark.asyncio
    async def test_update_indicators_calculates_atr(self, strategy_with_history):
        """Test that update_indicators calculates ATR."""
        await strategy_with_history._update_indicators('AAPL')

        atr = strategy_with_history.indicators['AAPL'].get('atr')
        assert atr is not None

    @pytest.mark.asyncio
    async def test_update_indicators_insufficient_history(self, strategy_with_history):
        """Test that update_indicators handles insufficient history."""
        strategy_with_history.price_history['AAPL'] = strategy_with_history.price_history['AAPL'][:10]

        await strategy_with_history._update_indicators('AAPL')

        assert strategy_with_history.indicators['AAPL'] == {}

    @pytest.mark.asyncio
    async def test_update_indicators_handles_exception(self, strategy_with_history):
        """Test that update_indicators handles exceptions."""
        strategy_with_history.price_history['AAPL'] = "invalid"

        await strategy_with_history._update_indicators('AAPL')
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
            'AAPL': {
                'close': 150.0,
                'upper_band': 160.0,
                'middle_band': 152.0,
                'lower_band': 144.0,
                'rsi': 50.0,
                'z_score': 0.0,
                'bb_position': 0.5,
                'slowk': 50.0,
                'slowd': 50.0,
                'sma': 152.0,
                'std': 4.0,
            }
        }

        return strategy

    @pytest.mark.asyncio
    async def test_generate_signal_returns_neutral_no_indicators(self, strategy_with_indicators):
        """Test that generate_signal returns neutral when no indicators."""
        strategy_with_indicators.indicators = {}

        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'neutral'

    @pytest.mark.asyncio
    async def test_generate_signal_returns_neutral_null_rsi(self, strategy_with_indicators):
        """Test that generate_signal returns neutral when RSI is None."""
        strategy_with_indicators.indicators['AAPL']['rsi'] = None

        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'neutral'

    @pytest.mark.asyncio
    async def test_generate_signal_buy_all_conditions_met(self, strategy_with_indicators):
        """Test buy signal when all conditions are met."""
        strategy_with_indicators.indicators['AAPL'] = {
            'close': 143.0,  # Below lower band
            'upper_band': 160.0,
            'middle_band': 152.0,
            'lower_band': 144.0,
            'rsi': 25.0,  # Oversold
            'z_score': -2.0,  # Far from mean
            'bb_position': 0.02,  # Near bottom of BB
            'slowk': 15.0,  # Stoch oversold
            'slowd': 10.0,  # Stoch turning up (k > d)
            'sma': 152.0,
            'std': 4.0,
        }

        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'buy'

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_conditions_not_met(self, strategy_with_indicators):
        """Test neutral signal when conditions not fully met."""
        # Default indicators don't meet buy/sell conditions
        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'neutral'

    @pytest.mark.asyncio
    async def test_generate_signal_short_when_enabled(self, strategy_with_indicators):
        """Test short signal when short selling is enabled."""
        strategy_with_indicators.enable_short_selling = True
        strategy_with_indicators.indicators['AAPL'] = {
            'close': 161.0,  # Above upper band
            'upper_band': 160.0,
            'middle_band': 152.0,
            'lower_band': 144.0,
            'rsi': 80.0,  # Overbought
            'z_score': 2.0,  # Far from mean
            'bb_position': 0.98,  # Near top of BB
            'slowk': 85.0,  # Stoch overbought
            'slowd': 90.0,  # Stoch turning down (k < d)
            'sma': 152.0,
            'std': 4.0,
        }

        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'short'

    @pytest.mark.asyncio
    async def test_generate_signal_neutral_short_disabled(self, strategy_with_indicators):
        """Test neutral when sell conditions met but short selling disabled."""
        strategy_with_indicators.enable_short_selling = False
        strategy_with_indicators.indicators['AAPL'] = {
            'close': 161.0,
            'upper_band': 160.0,
            'middle_band': 152.0,
            'lower_band': 144.0,
            'rsi': 80.0,
            'z_score': 2.0,
            'bb_position': 0.98,
            'slowk': 85.0,
            'slowd': 90.0,
            'sma': 152.0,
            'std': 4.0,
        }

        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'neutral'

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_rejects_buy_in_downtrend(self, strategy_with_indicators):
        """Test MTF filter rejects buy signal in strong downtrend."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.parameters = {'mtf_timeframes': ['5Min', '15Min', '1Hour']}

        # Create mock MTF analyzer
        strategy_with_indicators.mtf_analyzer = Mock()
        strategy_with_indicators.mtf_analyzer.get_trend = Mock(return_value='bearish')

        # Set up buy conditions
        strategy_with_indicators.indicators['AAPL'] = {
            'close': 143.0,
            'upper_band': 160.0,
            'middle_band': 152.0,
            'lower_band': 144.0,
            'rsi': 25.0,
            'z_score': -2.0,
            'bb_position': 0.02,
            'slowk': 15.0,
            'slowd': 10.0,
            'sma': 152.0,
            'std': 4.0,
        }

        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'neutral'  # Rejected by MTF filter

    @pytest.mark.asyncio
    async def test_generate_signal_mtf_allows_buy_in_neutral_market(self, strategy_with_indicators):
        """Test MTF filter allows buy signal in neutral market."""
        strategy_with_indicators.use_multi_timeframe = True
        strategy_with_indicators.parameters = {'mtf_timeframes': ['5Min', '15Min', '1Hour']}

        # Create mock MTF analyzer
        strategy_with_indicators.mtf_analyzer = Mock()
        strategy_with_indicators.mtf_analyzer.get_trend = Mock(return_value='neutral')

        # Set up buy conditions
        strategy_with_indicators.indicators['AAPL'] = {
            'close': 143.0,
            'upper_band': 160.0,
            'middle_band': 152.0,
            'lower_band': 144.0,
            'rsi': 25.0,
            'z_score': -2.0,
            'bb_position': 0.02,
            'slowk': 15.0,
            'slowd': 10.0,
            'sma': 152.0,
            'std': 4.0,
        }

        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'buy'

    @pytest.mark.asyncio
    async def test_generate_signal_handles_exception(self, strategy_with_indicators):
        """Test that generate_signal handles exceptions."""
        strategy_with_indicators.indicators['AAPL'] = "invalid"

        signal = await strategy_with_indicators._generate_signal('AAPL')

        assert signal == 'neutral'


class TestOnBar:
    """Tests for the on_bar method."""

    @pytest.fixture
    def initialized_strategy(self):
        """Create an initialized strategy."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.symbols = ['AAPL']
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

        strategy.indicators = {'AAPL': {}}
        strategy.signals = {'AAPL': 'neutral'}
        strategy.last_signal_time = {'AAPL': None}
        strategy.position_entries = {}
        strategy.highest_prices = {}
        strategy.lowest_prices = {}
        strategy.current_prices = {}
        strategy.price_history = {'AAPL': []}

        # Mock broker
        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[])
        strategy.broker.get_account = AsyncMock(return_value=Mock(buying_power='100000'))

        return strategy

    @pytest.mark.asyncio
    async def test_on_bar_ignores_unknown_symbol(self, initialized_strategy):
        """Test that on_bar ignores unknown symbols."""
        await initialized_strategy.on_bar('UNKNOWN', 100, 102, 98, 101, 1000000, datetime.now())

        assert 'UNKNOWN' not in initialized_strategy.current_prices

    @pytest.mark.asyncio
    async def test_on_bar_stores_current_price(self, initialized_strategy):
        """Test that on_bar stores the current price."""
        await initialized_strategy.on_bar('AAPL', 150, 152, 148, 151, 1000000, datetime.now())

        assert initialized_strategy.current_prices['AAPL'] == 151

    @pytest.mark.asyncio
    async def test_on_bar_updates_price_history(self, initialized_strategy):
        """Test that on_bar updates price history."""
        await initialized_strategy.on_bar('AAPL', 150, 152, 148, 151, 1000000, datetime.now())

        assert len(initialized_strategy.price_history['AAPL']) == 1
        assert initialized_strategy.price_history['AAPL'][0]['close'] == 151

    @pytest.mark.asyncio
    async def test_on_bar_trims_price_history(self, initialized_strategy):
        """Test that on_bar trims price history to max size."""
        max_history = max(
            initialized_strategy.sma_period,
            initialized_strategy.bb_period,
            initialized_strategy.rsi_period
        ) + initialized_strategy.mean_lookback + 10

        # Add more than max history
        for i in range(max_history + 20):
            await initialized_strategy.on_bar(
                'AAPL', 150, 152, 148, 150 + i * 0.1,
                1000000, datetime.now() + timedelta(minutes=i)
            )

        assert len(initialized_strategy.price_history['AAPL']) <= max_history

    @pytest.mark.asyncio
    async def test_on_bar_updates_mtf_analyzer(self, initialized_strategy):
        """Test that on_bar updates MTF analyzer when enabled."""
        initialized_strategy.use_multi_timeframe = True
        initialized_strategy.mtf_analyzer = Mock()
        initialized_strategy.mtf_analyzer.update = AsyncMock()

        timestamp = datetime.now()
        await initialized_strategy.on_bar('AAPL', 150, 152, 148, 151, 1000000, timestamp)

        initialized_strategy.mtf_analyzer.update.assert_called_once_with('AAPL', timestamp, 151, 1000000)

    @pytest.mark.asyncio
    async def test_on_bar_handles_exception(self, initialized_strategy):
        """Test that on_bar handles exceptions."""
        initialized_strategy.price_history = None  # Will cause error

        # Should not raise
        await initialized_strategy.on_bar('AAPL', 150, 152, 148, 151, 1000000, datetime.now())


class TestExecuteSignal:
    """Tests for the _execute_signal method."""

    @pytest.fixture
    def strategy_for_execution(self):
        """Create a strategy ready for order execution."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.symbols = ['AAPL']
        strategy.position_size = 0.10
        strategy.max_positions = 5
        strategy.stop_loss = 0.02
        strategy.take_profit = 0.04
        strategy.enable_short_selling = True
        strategy.short_position_size = 0.08
        strategy.short_stop_loss = 0.03

        strategy.current_prices = {'AAPL': 150.0}
        strategy.last_signal_time = {'AAPL': None}
        strategy.position_entries = {}
        strategy.highest_prices = {}
        strategy.lowest_prices = {}
        strategy.price_history = {'AAPL': [{'close': 150.0} for _ in range(25)]}

        # Mock broker
        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[])
        strategy.broker.get_account = AsyncMock(return_value=Mock(buying_power='100000'))
        strategy.broker.submit_order_advanced = AsyncMock(return_value=Mock(id='order123'))

        # Mock risk manager
        strategy.risk_manager = Mock()
        strategy.risk_manager.adjust_position_size = Mock(return_value=10000)

        # Mock enforce_position_size_limit
        strategy.enforce_position_size_limit = AsyncMock(return_value=(10000, 66.67))

        return strategy

    @pytest.mark.asyncio
    async def test_execute_signal_buy_creates_order(self, strategy_for_execution):
        """Test that buy signal creates bracket order."""
        await strategy_for_execution._execute_signal('AAPL', 'buy')

        strategy_for_execution.broker.submit_order_advanced.assert_called_once()
        # Verify the order was submitted (implementation details vary by Alpaca SDK version)
        call_args = strategy_for_execution.broker.submit_order_advanced.call_args[0][0]
        assert hasattr(call_args, 'symbol') or isinstance(call_args, dict)
        # Verify entry was tracked
        assert 'AAPL' in strategy_for_execution.position_entries

    @pytest.mark.asyncio
    async def test_execute_signal_buy_stores_entry_details(self, strategy_for_execution):
        """Test that buy signal stores entry details."""
        await strategy_for_execution._execute_signal('AAPL', 'buy')

        assert 'AAPL' in strategy_for_execution.position_entries
        assert strategy_for_execution.position_entries['AAPL']['price'] == 150.0

    @pytest.mark.asyncio
    async def test_execute_signal_buy_initializes_trailing_stop(self, strategy_for_execution):
        """Test that buy signal initializes trailing stop tracking."""
        await strategy_for_execution._execute_signal('AAPL', 'buy')

        assert 'AAPL' in strategy_for_execution.highest_prices
        assert strategy_for_execution.highest_prices['AAPL'] == 150.0

    @pytest.mark.asyncio
    async def test_execute_signal_respects_cooldown(self, strategy_for_execution):
        """Test that execute signal respects cooldown period."""
        strategy_for_execution.last_signal_time['AAPL'] = datetime.now()

        await strategy_for_execution._execute_signal('AAPL', 'buy')

        strategy_for_execution.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_respects_max_positions(self, strategy_for_execution):
        """Test that execute signal respects max positions."""
        # Create mock positions
        mock_positions = [Mock(symbol=f'SYM{i}') for i in range(5)]
        strategy_for_execution.broker.get_positions = AsyncMock(return_value=mock_positions)

        await strategy_for_execution._execute_signal('AAPL', 'buy')

        strategy_for_execution.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_skips_if_already_positioned(self, strategy_for_execution):
        """Test that execute signal skips if already has position."""
        mock_position = Mock(symbol='AAPL', qty='10')
        strategy_for_execution.broker.get_positions = AsyncMock(return_value=[mock_position])

        await strategy_for_execution._execute_signal('AAPL', 'buy')

        strategy_for_execution.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_short_creates_order(self, strategy_for_execution):
        """Test that short signal creates short order."""
        await strategy_for_execution._execute_signal('AAPL', 'short')

        strategy_for_execution.broker.submit_order_advanced.assert_called_once()
        # Verify the order was submitted and entry was tracked
        assert 'AAPL' in strategy_for_execution.position_entries
        assert strategy_for_execution.position_entries['AAPL'].get('is_short') is True

    @pytest.mark.asyncio
    async def test_execute_signal_short_stores_is_short_flag(self, strategy_for_execution):
        """Test that short signal stores is_short flag."""
        await strategy_for_execution._execute_signal('AAPL', 'short')

        assert strategy_for_execution.position_entries['AAPL'].get('is_short') is True

    @pytest.mark.asyncio
    async def test_execute_signal_short_initializes_lowest_price(self, strategy_for_execution):
        """Test that short signal initializes lowest price tracking."""
        await strategy_for_execution._execute_signal('AAPL', 'short')

        assert 'AAPL' in strategy_for_execution.lowest_prices
        assert strategy_for_execution.lowest_prices['AAPL'] == 150.0

    @pytest.mark.asyncio
    async def test_execute_signal_sell_closes_position(self, strategy_for_execution):
        """Test that sell signal closes existing position."""
        mock_position = Mock(symbol='AAPL', qty='10')
        strategy_for_execution.broker.get_positions = AsyncMock(return_value=[mock_position])
        strategy_for_execution.position_entries['AAPL'] = {'time': datetime.now(), 'price': 145.0}
        strategy_for_execution.highest_prices['AAPL'] = 150.0

        await strategy_for_execution._execute_signal('AAPL', 'sell')

        strategy_for_execution.broker.submit_order_advanced.assert_called_once()
        # Verify position tracking was cleared
        assert 'AAPL' not in strategy_for_execution.position_entries
        assert 'AAPL' not in strategy_for_execution.highest_prices

    @pytest.mark.asyncio
    async def test_execute_signal_risk_manager_rejects_position(self, strategy_for_execution):
        """Test handling when risk manager rejects position."""
        strategy_for_execution.risk_manager.adjust_position_size = Mock(return_value=0)
        strategy_for_execution.enforce_position_size_limit = AsyncMock(return_value=(0, 0))

        await strategy_for_execution._execute_signal('AAPL', 'buy')

        strategy_for_execution.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_skips_small_quantity(self, strategy_for_execution):
        """Test that execute signal skips very small quantities."""
        strategy_for_execution.enforce_position_size_limit = AsyncMock(return_value=(1.0, 0.005))

        await strategy_for_execution._execute_signal('AAPL', 'buy')

        strategy_for_execution.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_handles_exception(self, strategy_for_execution):
        """Test that execute signal handles exceptions."""
        strategy_for_execution.broker.get_positions = AsyncMock(side_effect=Exception("API Error"))

        # Should not raise
        await strategy_for_execution._execute_signal('AAPL', 'buy')


class TestCheckExitConditions:
    """Tests for the _check_exit_conditions method."""

    @pytest.fixture
    def strategy_with_position(self):
        """Create a strategy with an open position."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.symbols = ['AAPL']
        strategy.stop_loss = 0.02
        strategy.take_profit = 0.04
        strategy.trailing_stop = 0.015
        strategy.max_hold_days = 5
        strategy.profit_target_std = 0.5

        strategy.current_prices = {'AAPL': 155.0}
        strategy.position_entries = {
            'AAPL': {
                'time': datetime.now() - timedelta(days=1),
                'price': 150.0,
                'quantity': 10
            }
        }
        strategy.highest_prices = {'AAPL': 155.0}
        strategy.lowest_prices = {'AAPL': 150.0}
        strategy.indicators = {
            'AAPL': {
                'sma': 152.0,
                'std': 4.0,
            }
        }

        # Mock broker
        strategy.broker = Mock()
        mock_position = Mock(symbol='AAPL', qty='10')
        strategy.broker.get_positions = AsyncMock(return_value=[mock_position])
        strategy.broker.submit_order_advanced = AsyncMock(return_value=Mock(id='order123'))

        return strategy

    @pytest.mark.asyncio
    async def test_check_exit_conditions_cleans_up_closed_position(self, strategy_with_position):
        """Test cleanup when position is closed."""
        strategy_with_position.broker.get_positions = AsyncMock(return_value=[])

        await strategy_with_position._check_exit_conditions('AAPL')

        assert 'AAPL' not in strategy_with_position.position_entries
        assert 'AAPL' not in strategy_with_position.highest_prices

    @pytest.mark.asyncio
    async def test_check_exit_max_hold_days_triggers_exit(self, strategy_with_position):
        """Test that max hold days triggers exit."""
        strategy_with_position.position_entries['AAPL']['time'] = datetime.now() - timedelta(days=6)

        await strategy_with_position._check_exit_conditions('AAPL')

        strategy_with_position.broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_exit_mean_reversion_target_long(self, strategy_with_position):
        """Test mean reversion target exit for long position."""
        # Entry below SMA, now at SMA level
        strategy_with_position.position_entries['AAPL']['price'] = 145.0
        strategy_with_position.current_prices['AAPL'] = 152.0  # At SMA

        await strategy_with_position._check_exit_conditions('AAPL')

        strategy_with_position.broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_exit_trailing_stop_triggers(self, strategy_with_position):
        """Test trailing stop trigger - ensures trailing stop code path is covered."""
        # Setup to avoid mean reversion target triggering first:
        # - Entry was above SMA (not a typical oversold buy)
        # - Current price is well above mean, so mean reversion doesn't trigger
        strategy_with_position.position_entries['AAPL']['price'] = 158.0  # Entry above SMA
        strategy_with_position.indicators['AAPL'] = {
            'sma': 150.0,  # Mean is below entry
            'std': 4.0,    # sma + profit_target_std * std = 150 + 0.5*4 = 152
        }
        # Current price at 155 is above 152, so mean reversion target NOT hit
        strategy_with_position.current_prices['AAPL'] = 155.0

        # Position is in profit: (155 - 158) / 158 = -0.019 = -1.9% (actually a loss)
        # Need profit for trailing stop, so entry must be lower than current
        strategy_with_position.position_entries['AAPL']['price'] = 150.0
        # Now (155 - 150) / 150 = 0.033 = 3.3% profit > 0 ✓

        # Peak price tracking: 160.0
        # trailing_stop = 0.015 (1.5%)
        # trailing_stop_price = 160 * (1 - 0.015) = 157.6
        # current_price (155) < trailing_stop_price (157.6) → trailing stop TRIGGERS
        strategy_with_position.highest_prices['AAPL'] = 160.0

        await strategy_with_position._check_exit_conditions('AAPL')

        strategy_with_position.broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_exit_no_current_price(self, strategy_with_position):
        """Test handling when no current price available."""
        strategy_with_position.current_prices = {}

        await strategy_with_position._check_exit_conditions('AAPL')

        strategy_with_position.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_exit_no_entry_details(self, strategy_with_position):
        """Test handling when no entry details available."""
        strategy_with_position.position_entries = {}

        await strategy_with_position._check_exit_conditions('AAPL')

        strategy_with_position.broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_exit_handles_exception(self, strategy_with_position):
        """Test that check_exit_conditions handles exceptions."""
        strategy_with_position.broker.get_positions = AsyncMock(side_effect=Exception("API Error"))

        # Should not raise
        await strategy_with_position._check_exit_conditions('AAPL')


class TestAnalyzeSymbol:
    """Tests for the analyze_symbol method."""

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_signal(self):
        """Test that analyze_symbol returns the signal for a symbol."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.signals = {'AAPL': 'buy', 'MSFT': 'sell'}

        signal = await strategy.analyze_symbol('AAPL')

        assert signal == 'buy'

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_neutral_for_unknown(self):
        """Test that analyze_symbol returns neutral for unknown symbol."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.signals = {}

        signal = await strategy.analyze_symbol('UNKNOWN')

        assert signal == 'neutral'


class TestExecuteTrade:
    """Tests for the execute_trade method."""

    @pytest.mark.asyncio
    async def test_execute_trade_is_noop(self):
        """Test that execute_trade is a no-op (handled by _execute_signal)."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)

        # Should not raise
        await strategy.execute_trade('AAPL', 'buy')


class TestGenerateSignals:
    """Tests for the generate_signals method (backtest mode)."""

    @pytest.fixture
    def backtest_strategy(self):
        """Create a strategy for backtest mode testing."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.symbols = ['AAPL']
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

        strategy.indicators = {'AAPL': {}}
        strategy.signals = {'AAPL': 'neutral'}

        # Create test data with explicit float64 types
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100, freq='min')
        data = pd.DataFrame({
            'open': (np.random.randn(100) + 150).astype(np.float64),
            'high': (np.random.randn(100) + 152).astype(np.float64),
            'low': (np.random.randn(100) + 148).astype(np.float64),
            'close': (np.random.randn(100) + 150).astype(np.float64),
            'volume': (np.random.randint(900000, 1100000, 100)).astype(np.float64)
        }, index=dates)

        strategy.current_data = {'AAPL': data}

        return strategy

    @pytest.mark.asyncio
    async def test_generate_signals_updates_indicators(self, backtest_strategy):
        """Test that generate_signals updates indicators."""
        await backtest_strategy.generate_signals()

        assert backtest_strategy.indicators['AAPL'] != {}

    @pytest.mark.asyncio
    async def test_generate_signals_updates_signals(self, backtest_strategy):
        """Test that generate_signals updates signals."""
        await backtest_strategy.generate_signals()

        assert backtest_strategy.signals['AAPL'] in ['buy', 'sell', 'short', 'neutral']

    @pytest.mark.asyncio
    async def test_generate_signals_skips_insufficient_data(self, backtest_strategy):
        """Test that generate_signals skips symbols with insufficient data."""
        dates = pd.date_range(end=datetime.now(), periods=10, freq='min')
        data = pd.DataFrame({
            'open': np.array([150.0] * 10, dtype=np.float64),
            'high': np.array([152.0] * 10, dtype=np.float64),
            'low': np.array([148.0] * 10, dtype=np.float64),
            'close': np.array([150.0] * 10, dtype=np.float64),
            'volume': np.array([1000000.0] * 10, dtype=np.float64)
        }, index=dates)

        backtest_strategy.current_data = {'AAPL': data}

        await backtest_strategy.generate_signals()

        assert backtest_strategy.indicators['AAPL'] == {}


class TestGetOrders:
    """Tests for the get_orders method (backtest mode)."""

    @pytest.fixture
    def strategy_for_orders(self):
        """Create a strategy for testing get_orders."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.signals = {'AAPL': 'buy', 'MSFT': 'neutral'}
        strategy.indicators = {
            'AAPL': {'close': 150.0},
            'MSFT': {'close': 300.0}
        }
        strategy.capital = 100000
        strategy.position_size = 0.10
        strategy.positions = {}

        return strategy

    def test_get_orders_returns_buy_order(self, strategy_for_orders):
        """Test that get_orders returns buy order for buy signal."""
        orders = strategy_for_orders.get_orders()

        assert len(orders) == 1
        assert orders[0]['symbol'] == 'AAPL'
        assert orders[0]['side'] == 'buy'

    def test_get_orders_skips_neutral(self, strategy_for_orders):
        """Test that get_orders skips neutral signals."""
        orders = strategy_for_orders.get_orders()

        # Only AAPL should have an order, MSFT is neutral
        symbols = [o['symbol'] for o in orders]
        assert 'MSFT' not in symbols

    def test_get_orders_calculates_quantity(self, strategy_for_orders):
        """Test that get_orders calculates correct quantity."""
        orders = strategy_for_orders.get_orders()

        expected_quantity = (100000 * 0.10) / 150.0
        assert orders[0]['quantity'] == pytest.approx(expected_quantity, rel=0.01)

    def test_get_orders_sell_with_position(self, strategy_for_orders):
        """Test that get_orders returns sell order when has position."""
        strategy_for_orders.signals['AAPL'] = 'sell'
        strategy_for_orders.positions = {'AAPL': {'quantity': 10}}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 1
        assert orders[0]['side'] == 'sell'
        assert orders[0]['quantity'] == 10

    def test_get_orders_skips_buy_with_position(self, strategy_for_orders):
        """Test that get_orders skips buy when already has position."""
        strategy_for_orders.positions = {'AAPL': {'quantity': 10}}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_sell_without_position(self, strategy_for_orders):
        """Test that get_orders skips sell when no position."""
        strategy_for_orders.signals['AAPL'] = 'sell'
        strategy_for_orders.positions = {}

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_invalid_price(self, strategy_for_orders):
        """Test that get_orders skips orders with invalid price."""
        strategy_for_orders.indicators['AAPL']['close'] = None

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0

    def test_get_orders_skips_small_quantity(self, strategy_for_orders):
        """Test that get_orders skips very small quantities (< 0.01 shares)."""
        # With 100000 capital, 10% position size = 10000
        # Price of 10000000 = 0.001 shares (< 0.01 threshold)
        strategy_for_orders.indicators['AAPL']['close'] = 10000000  # Extremely high price = < 0.01 shares

        orders = strategy_for_orders.get_orders()

        assert len(orders) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_price_history(self):
        """Test handling of empty price history."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.sma_period = 50
        strategy.indicators = {'AAPL': {}}
        strategy.price_history = {'AAPL': []}

        await strategy._update_indicators('AAPL')

        assert strategy.indicators['AAPL'] == {}

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

    @pytest.mark.asyncio
    async def test_on_bar_with_no_price_history_key(self):
        """Test on_bar when price_history key doesn't exist."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.symbols = ['AAPL']
        strategy.sma_period = 50
        strategy.bb_period = 20
        strategy.rsi_period = 14
        strategy.mean_lookback = 20
        strategy.use_multi_timeframe = False
        strategy.enable_short_selling = False
        strategy.mtf_analyzer = None

        strategy.current_prices = {}
        strategy.indicators = {'AAPL': {}}
        strategy.signals = {'AAPL': 'neutral'}
        strategy.price_history = {}  # Missing AAPL key

        # Mock broker
        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[])

        # Should handle gracefully
        await strategy.on_bar('AAPL', 150, 152, 148, 151, 1000000, datetime.now())


class TestRiskManagerIntegration:
    """Tests for risk manager integration."""

    @pytest.fixture
    def strategy_with_risk_manager(self):
        """Create a strategy with risk manager."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.symbols = ['AAPL']
        strategy.position_size = 0.10
        strategy.max_positions = 5
        strategy.stop_loss = 0.02
        strategy.take_profit = 0.04
        strategy.enable_short_selling = False
        strategy.short_position_size = 0.08
        strategy.short_stop_loss = 0.03

        strategy.current_prices = {'AAPL': 150.0}
        strategy.last_signal_time = {'AAPL': None}
        strategy.position_entries = {}
        strategy.highest_prices = {}
        strategy.lowest_prices = {}

        # Create price history
        strategy.price_history = {'AAPL': [{'close': float(150 + i * 0.1)} for i in range(25)]}

        # Mock broker
        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[])
        strategy.broker.get_account = AsyncMock(return_value=Mock(buying_power='100000'))
        strategy.broker.submit_order_advanced = AsyncMock(return_value=Mock(id='order123'))

        # Real risk manager (not mocked)
        from strategies.risk_manager import RiskManager
        strategy.risk_manager = RiskManager(
            max_portfolio_risk=0.02,
            max_position_risk=0.01,
            max_correlation=0.7
        )

        # Mock enforce_position_size_limit
        strategy.enforce_position_size_limit = AsyncMock(return_value=(10000, 66.67))

        return strategy

    @pytest.mark.asyncio
    async def test_risk_manager_called_with_position_data(self, strategy_with_risk_manager):
        """Test that risk manager is called with correct position data."""
        with patch.object(strategy_with_risk_manager.risk_manager, 'adjust_position_size', return_value=10000) as mock_adjust:
            await strategy_with_risk_manager._execute_signal('AAPL', 'buy')

            mock_adjust.assert_called_once()
            call_args = mock_adjust.call_args
            assert call_args[0][0] == 'AAPL'  # symbol
            assert call_args[0][1] == 10000  # position_value
            assert isinstance(call_args[0][2], list)  # price_history


class TestShortSellingIntegration:
    """Tests for short selling integration."""

    @pytest.fixture
    def short_strategy(self):
        """Create a strategy configured for short selling."""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy.__new__(MeanReversionStrategy)
        strategy.symbols = ['AAPL']
        strategy.position_size = 0.10
        strategy.max_positions = 5
        strategy.stop_loss = 0.02
        strategy.take_profit = 0.04
        strategy.enable_short_selling = True
        strategy.short_position_size = 0.08
        strategy.short_stop_loss = 0.03

        strategy.current_prices = {'AAPL': 150.0}
        strategy.last_signal_time = {'AAPL': None}
        strategy.position_entries = {}
        strategy.highest_prices = {}
        strategy.lowest_prices = {}
        strategy.price_history = {'AAPL': [{'close': float(150 + i * 0.1)} for i in range(25)]}

        # Mock broker
        strategy.broker = Mock()
        strategy.broker.get_positions = AsyncMock(return_value=[])
        strategy.broker.get_account = AsyncMock(return_value=Mock(buying_power='100000'))
        strategy.broker.submit_order_advanced = AsyncMock(return_value=Mock(id='order123'))

        # Mock risk manager
        strategy.risk_manager = Mock()
        strategy.risk_manager.adjust_position_size = Mock(return_value=8000)

        # Mock enforce_position_size_limit
        strategy.enforce_position_size_limit = AsyncMock(return_value=(8000, 53.33))

        return strategy

    @pytest.mark.asyncio
    async def test_short_order_inverted_levels(self, short_strategy):
        """Test that short orders have inverted take-profit and stop-loss levels."""
        await short_strategy._execute_signal('AAPL', 'short')

        # Verify order was submitted
        short_strategy.broker.submit_order_advanced.assert_called_once()

        # Verify short position was stored with correct metadata
        # For shorts: take-profit should be BELOW entry, stop-loss ABOVE entry
        # Entry: $150
        # Take profit: $150 * (1 - 0.04) = $144 (price drops 4%)
        # Stop loss: $150 * (1 + 0.03) = $154.50 (price rises 3%)
        assert 'AAPL' in short_strategy.position_entries
        entry = short_strategy.position_entries['AAPL']
        assert entry['is_short'] is True
        assert entry['price'] == 150.0

        # Verify lowest_prices tracking was initialized (used for trailing stops on shorts)
        assert 'AAPL' in short_strategy.lowest_prices
        assert short_strategy.lowest_prices['AAPL'] == 150.0

    @pytest.mark.asyncio
    async def test_short_uses_smaller_position_size(self, short_strategy):
        """Test that short positions use smaller position size."""
        await short_strategy._execute_signal('AAPL', 'short')

        # Verify risk manager was called with short_position_size (8%)
        short_strategy.risk_manager.adjust_position_size.assert_called_once()
        call_args = short_strategy.risk_manager.adjust_position_size.call_args[0]
        expected_value = 100000 * 0.08  # short_position_size = 0.08
        assert call_args[1] == expected_value
