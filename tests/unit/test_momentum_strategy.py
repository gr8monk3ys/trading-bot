"""
Unit tests for MomentumStrategy
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from strategies.momentum_strategy import MomentumStrategy
from tests.fixtures.mock_broker import MockAlpacaBroker
from tests.fixtures.mock_data import (
    generate_momentum_scenario,
    generate_mean_reversion_scenario,
    generate_volatile_scenario
)
from tests.fixtures.test_helpers import (
    assert_approximately_equal,
    assert_in_range,
    assert_valid_signal,
    create_mock_strategy_params
)


@pytest.mark.unit
@pytest.mark.strategy
@pytest.mark.asyncio
class TestMomentumStrategy:
    """Tests for MomentumStrategy"""

    @pytest.fixture
    async def mock_broker(self):
        """Create a mock broker for testing"""
        broker = MockAlpacaBroker(paper=True, initial_capital=100000.0)
        return broker

    @pytest.fixture
    async def strategy(self, mock_broker):
        """Create a strategy instance for testing"""
        symbols = ['TEST', 'AAPL', 'TSLA']
        strategy = MomentumStrategy(broker=mock_broker)
        await strategy.initialize(symbols=symbols)
        return strategy

    async def test_initialization(self, strategy):
        """Test strategy initializes correctly"""
        assert strategy.NAME == "MomentumStrategy"
        assert len(strategy.symbols) == 3
        assert 'TEST' in strategy.symbols
        assert strategy.position_size == 0.10  # 10% default
        assert strategy.stop_loss == 0.03  # 3% default
        assert strategy.take_profit == 0.05  # 5% default

    async def test_default_parameters(self, strategy):
        """Test default parameters are set correctly"""
        params = strategy.default_parameters()

        # Basic parameters
        assert params['position_size'] == 0.10
        assert params['max_positions'] == 3
        assert params['stop_loss'] == 0.03
        assert params['take_profit'] == 0.05

        # Technical indicators
        assert params['rsi_period'] == 14
        assert params['rsi_overbought'] == 70
        assert params['rsi_oversold'] == 30
        assert params['macd_fast_period'] == 12
        assert params['macd_slow_period'] == 26
        assert params['adx_threshold'] == 25

        # Advanced features (disabled by default)
        assert params['use_multi_timeframe'] is False
        assert params['enable_short_selling'] is False
        assert params['use_kelly_criterion'] is False

    async def test_indicator_calculation(self, strategy):
        """Test that indicators are calculated correctly"""
        symbol = 'TEST'

        # Generate mock price data (100 days of momentum)
        df = generate_momentum_scenario(days=100)

        # Populate price history
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Update indicators
        await strategy._update_indicators(symbol)

        # Verify indicators were calculated
        assert symbol in strategy.indicators
        indicators = strategy.indicators[symbol]

        # Check RSI
        assert indicators['rsi'] is not None
        assert 0 <= indicators['rsi'] <= 100

        # Check MACD
        assert indicators['macd'] is not None
        assert indicators['macd_signal'] is not None
        assert indicators['macd_hist'] is not None

        # Check ADX
        assert indicators['adx'] is not None
        assert indicators['adx'] >= 0

        # Check moving averages
        assert indicators['fast_ma'] is not None
        assert indicators['medium_ma'] is not None
        assert indicators['slow_ma'] is not None

        # Check volume indicators
        assert indicators['volume'] is not None
        assert indicators['volume_ma'] is not None

    async def test_buy_signal_generation(self, strategy):
        """Test that buy signals are generated correctly"""
        symbol = 'TEST'

        # Generate strong momentum scenario
        df = generate_momentum_scenario(days=100)

        # Populate price history
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume'] * 2)  # High volume for signal
            })

        # Update indicators
        await strategy._update_indicators(symbol)

        # Manually set indicators to create buy signal
        strategy.indicators[symbol].update({
            'rsi': 25,  # Oversold
            'macd': 0.5,
            'macd_signal': 0.3,  # MACD > signal
            'macd_hist': 0.2,  # Positive histogram
            'adx': 30,  # Strong trend
            'fast_ma': 105,
            'medium_ma': 103,
            'slow_ma': 100,  # Bullish MA alignment
            'volume': 5_000_000,
            'volume_ma': 2_000_000  # High volume
        })

        # Generate signal
        signal = await strategy._generate_signal(symbol)

        # Should generate buy signal
        assert signal == 'buy'

    async def test_sell_signal_generation(self, strategy):
        """Test that sell signals are generated correctly (for shorts)"""
        symbol = 'TEST'

        # Enable short selling
        strategy.enable_short_selling = True

        # Generate mean reversion scenario
        df = generate_mean_reversion_scenario(days=100)

        # Populate price history
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume'] * 2)
            })

        # Update indicators
        await strategy._update_indicators(symbol)

        # Manually set indicators to create sell signal (short)
        strategy.indicators[symbol].update({
            'rsi': 75,  # Overbought
            'macd': -0.5,
            'macd_signal': -0.3,  # MACD < signal
            'macd_hist': -0.2,  # Negative histogram
            'adx': 30,  # Strong trend
            'fast_ma': 95,
            'medium_ma': 97,
            'slow_ma': 100,  # Bearish MA alignment
            'volume': 5_000_000,
            'volume_ma': 2_000_000  # High volume
        })

        # Generate signal
        signal = await strategy._generate_signal(symbol)

        # Should generate short signal
        assert signal == 'short'

    async def test_neutral_signal_on_weak_momentum(self, strategy):
        """Test that neutral signal is generated when momentum is weak"""
        symbol = 'TEST'

        # Generate sideways/ranging data
        df = generate_mean_reversion_scenario(days=100)

        # Populate price history
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Update indicators
        await strategy._update_indicators(symbol)

        # Set weak indicators
        strategy.indicators[symbol].update({
            'rsi': 50,  # Neutral
            'macd': 0.1,
            'macd_signal': 0.1,  # Weak MACD
            'macd_hist': 0.0,
            'adx': 15,  # Weak trend
            'fast_ma': 100,
            'medium_ma': 100,
            'slow_ma': 100,  # Flat MAs
            'volume': 2_000_000,
            'volume_ma': 2_000_000  # Normal volume
        })

        # Generate signal
        signal = await strategy._generate_signal(symbol)

        # Should generate neutral signal
        assert signal == 'neutral'

    async def test_on_bar_updates_price_history(self, strategy):
        """Test that on_bar correctly updates price history"""
        symbol = 'TEST'
        initial_count = len(strategy.price_history[symbol])

        # Simulate bar data
        await strategy.on_bar(
            symbol=symbol,
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            volume=1_000_000,
            timestamp=datetime.now()
        )

        # Verify price history was updated
        assert len(strategy.price_history[symbol]) == initial_count + 1

        # Verify latest bar
        latest_bar = strategy.price_history[symbol][-1]
        assert latest_bar['close'] == 101.0
        assert latest_bar['volume'] == 1_000_000

    async def test_price_history_limit(self, strategy):
        """Test that price history is limited to max size"""
        symbol = 'TEST'

        # Add more bars than the limit
        max_history = max(
            strategy.slow_ma,
            strategy.rsi_period,
            strategy.macd_slow + strategy.macd_signal,
            strategy.adx_period
        ) + 10

        for i in range(max_history + 50):
            await strategy.on_bar(
                symbol=symbol,
                open_price=100.0,
                high_price=102.0,
                low_price=99.0,
                close_price=101.0,
                volume=1_000_000,
                timestamp=datetime.now() - timedelta(days=max_history + 50 - i)
            )

        # Verify history is limited
        assert len(strategy.price_history[symbol]) == max_history

    async def test_execute_buy_signal(self, strategy, mock_broker):
        """Test that buy signals are executed correctly"""
        symbol = 'TEST'
        strategy.current_prices[symbol] = 100.0
        strategy.last_signal_time[symbol] = datetime.now() - timedelta(hours=2)

        # Set up price history
        df = generate_momentum_scenario(days=50)
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Execute buy signal
        await strategy._execute_signal(symbol, 'buy')

        # Verify order was submitted
        orders = await mock_broker.get_orders()
        assert len(orders) > 0

        buy_order = orders[-1]
        assert buy_order.symbol == symbol
        assert buy_order.side == 'buy'
        assert buy_order.status == 'filled'

    async def test_position_size_calculation(self, strategy, mock_broker):
        """Test that position sizes are calculated correctly"""
        symbol = 'TEST'
        strategy.current_prices[symbol] = 100.0
        strategy.last_signal_time[symbol] = datetime.now() - timedelta(hours=2)

        # Get account buying power
        account = await mock_broker.get_account()
        buying_power = float(account.buying_power)

        # Set up price history
        df = generate_momentum_scenario(days=50)
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Execute buy signal
        await strategy._execute_signal(symbol, 'buy')

        # Get the order
        orders = await mock_broker.get_orders()
        order = orders[-1]

        # Verify position size is calculated
        # Note: enforce_position_size_limit caps at 5% of portfolio equity
        actual_value = order.qty * order.filled_avg_price

        # Position size should be positive and reasonable
        assert actual_value > 0
        assert actual_value < buying_power  # Less than buying power

        # Position should be at most 5% of portfolio (due to enforce_position_size_limit)
        account = await mock_broker.get_account()
        max_position = account.equity * 0.05
        assert actual_value <= max_position * 1.02  # Allow 2% tolerance for slippage

    async def test_max_positions_limit(self, strategy, mock_broker):
        """Test that max positions limit is enforced"""
        # Set max positions to 2
        strategy.max_positions = 2

        # Use symbols that were initialized in the strategy fixture
        symbols = ['TEST', 'AAPL', 'TSLA']
        for symbol in symbols:
            strategy.current_prices[symbol] = 100.0
            strategy.last_signal_time[symbol] = datetime.now() - timedelta(hours=2)

            # Set up price history
            df = generate_momentum_scenario(days=50)
            for _, row in df.iterrows():
                strategy.price_history[symbol].append({
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })

        # Try to buy 3 symbols
        for symbol in symbols:
            await strategy._execute_signal(symbol, 'buy')

        # Verify only 2 positions were created
        positions = await mock_broker.get_positions()
        assert len(positions) <= strategy.max_positions

    async def test_cooldown_period(self, strategy):
        """Test that cooldown period prevents overtrading"""
        symbol = 'TEST'
        strategy.current_prices[symbol] = 100.0

        # Set up price history
        df = generate_momentum_scenario(days=50)
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Execute first signal
        await strategy._execute_signal(symbol, 'buy')
        first_signal_time = strategy.last_signal_time[symbol]

        # Try to execute another signal immediately (should be blocked)
        await strategy._execute_signal(symbol, 'buy')
        second_signal_time = strategy.last_signal_time[symbol]

        # Verify signal time didn't change (second signal was blocked)
        assert first_signal_time == second_signal_time

    async def test_stop_loss_and_take_profit_levels(self, strategy):
        """Test that stop-loss and take-profit levels are set correctly"""
        symbol = 'TEST'
        entry_price = 100.0
        strategy.current_prices[symbol] = entry_price
        strategy.last_signal_time[symbol] = datetime.now() - timedelta(hours=2)

        # Set up price history
        df = generate_momentum_scenario(days=50)
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Execute buy signal
        await strategy._execute_signal(symbol, 'buy')

        # Verify stop and target prices were set
        assert symbol in strategy.stop_prices
        assert symbol in strategy.target_prices

        # Verify levels are correct
        expected_stop = entry_price * (1 - strategy.stop_loss)
        expected_target = entry_price * (1 + strategy.take_profit)

        assert_approximately_equal(strategy.stop_prices[symbol], expected_stop, tolerance=0.01)
        assert_approximately_equal(strategy.target_prices[symbol], expected_target, tolerance=0.01)

    async def test_short_selling_disabled_by_default(self, strategy):
        """Test that short selling is disabled by default"""
        assert strategy.enable_short_selling is False

        symbol = 'TEST'
        strategy.current_prices[symbol] = 100.0

        # Try to execute short signal (should be ignored)
        await strategy._execute_signal(symbol, 'short')

        # Verify no position was created
        positions = await strategy.broker.get_positions()
        assert len(positions) == 0

    async def test_risk_manager_integration(self, strategy):
        """Test that risk manager is properly integrated"""
        # Verify risk manager exists
        assert strategy.risk_manager is not None

        # Verify risk parameters
        assert strategy.risk_manager.max_portfolio_risk == strategy.parameters['max_portfolio_risk']
        assert strategy.risk_manager.max_correlation == strategy.parameters['max_correlation']

    async def test_indicator_none_handling(self, strategy):
        """Test that strategy handles None indicators gracefully"""
        symbol = 'TEST'

        # Don't populate price history (indicators will be None)
        signal = await strategy._generate_signal(symbol)

        # Should return neutral when indicators are None
        assert signal == 'neutral'

    async def test_insufficient_price_history(self, strategy):
        """Test behavior with insufficient price history"""
        symbol = 'TEST'

        # Add only a few bars (less than required for indicators)
        for i in range(5):
            await strategy.on_bar(
                symbol=symbol,
                open_price=100.0,
                high_price=102.0,
                low_price=99.0,
                close_price=101.0,
                volume=1_000_000,
                timestamp=datetime.now() - timedelta(days=5 - i)
            )

        # Indicators should not be updated (not enough data)
        assert strategy.indicators[symbol].get('rsi') is None

    async def test_volume_confirmation_requirement(self, strategy):
        """Test that volume confirmation is required for signals"""
        symbol = 'TEST'

        df = generate_momentum_scenario(days=100)
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        await strategy._update_indicators(symbol)

        # Set strong indicators but LOW volume
        strategy.indicators[symbol].update({
            'rsi': 25,
            'macd': 0.5,
            'macd_signal': 0.3,
            'macd_hist': 0.2,
            'adx': 30,
            'fast_ma': 105,
            'medium_ma': 103,
            'slow_ma': 100,
            'volume': 1_000_000,
            'volume_ma': 5_000_000  # Volume is BELOW average
        })

        signal = await strategy._generate_signal(symbol)

        # Should NOT generate signal due to low volume
        assert signal == 'neutral'


@pytest.mark.unit
@pytest.mark.strategy
@pytest.mark.asyncio
class TestMomentumStrategyAdvanced:
    """Advanced tests for MomentumStrategy"""

    @pytest.fixture
    async def strategy_with_short_selling(self):
        """Create strategy with short selling enabled"""
        broker = MockAlpacaBroker(paper=True, initial_capital=100000.0)
        symbols = ['TEST']
        strategy = MomentumStrategy(broker=broker)

        # Initialize with short selling enabled
        await strategy.initialize(symbols=symbols, enable_short_selling=True)
        return strategy

    async def test_short_signal_execution(self, strategy_with_short_selling):
        """Test that short signals are executed when enabled"""
        strategy = strategy_with_short_selling
        symbol = 'TEST'
        strategy.current_prices[symbol] = 100.0
        strategy.last_signal_time[symbol] = datetime.now() - timedelta(hours=2)

        # Set up price history
        df = generate_momentum_scenario(days=50)
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Execute short signal
        await strategy._execute_signal(symbol, 'short')

        # Verify short order was submitted
        orders = await strategy.broker.get_orders()
        assert len(orders) > 0

        short_order = orders[-1]
        assert short_order.symbol == symbol
        assert short_order.side == 'sell'  # Short = sell
        assert short_order.status == 'filled'

    async def test_backtest_mode_signal_generation(self):
        """Test signal generation in backtest mode"""
        broker = MockAlpacaBroker(paper=True)
        symbols = ['TEST']
        strategy = MomentumStrategy(broker=broker)
        await strategy.initialize(symbols=symbols)

        # Create sample data
        df = generate_momentum_scenario(days=100)
        strategy.current_data = {'TEST': df}

        # Generate signals
        await strategy.generate_signals()

        # Verify signal was generated
        assert 'TEST' in strategy.signals
        assert strategy.signals['TEST'] in ['buy', 'sell', 'short', 'neutral']

    async def test_backtest_get_orders(self):
        """Test order generation in backtest mode"""
        broker = MockAlpacaBroker(paper=True)
        symbols = ['TEST']
        strategy = MomentumStrategy(broker=broker)
        await strategy.initialize(symbols=symbols)

        # Set up indicators for buy signal
        strategy.indicators['TEST'] = {
            'rsi': 25,
            'macd': 0.5,
            'macd_signal': 0.3,
            'macd_hist': 0.2,
            'adx': 30,
            'fast_ma': 105,
            'medium_ma': 103,
            'slow_ma': 100,
            'volume': 5_000_000,
            'volume_ma': 2_000_000,
            'close': 100.0
        }
        strategy.signals['TEST'] = 'buy'
        strategy.capital = 100000

        # Get orders
        orders = strategy.get_orders()

        # Verify order was created
        assert len(orders) > 0
        assert orders[0]['symbol'] == 'TEST'
        assert orders[0]['side'] == 'buy'
        assert orders[0]['quantity'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
