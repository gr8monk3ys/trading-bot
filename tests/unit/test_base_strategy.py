#!/usr/bin/env python3
"""
Comprehensive unit tests for strategies/base_strategy.py

Tests cover:
- BaseStrategy initialization
- Parameter initialization
- Circuit breaker integration
- Position size enforcement
- Kelly Criterion position sizing
- Trade tracking and recording
- Volatility regime adjustments
- Streak-based adjustments
- Multi-timeframe signal checking
- Position helpers (is_short, get_pnl)
- Create order
- Risk limit checking
- Volatility calculation
- Performance metrics
- Cleanup and shutdown
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# Mock the utility modules before importing BaseStrategy
import sys

# Create mock modules
mock_circuit_breaker = Mock()
mock_circuit_breaker.CircuitBreaker = Mock()

mock_kelly = Mock()
mock_kelly.KellyCriterion = Mock()
mock_kelly.Trade = Mock()

mock_volatility = Mock()
mock_volatility.VolatilityRegimeDetector = Mock()

mock_streak = Mock()
mock_streak.StreakSizer = Mock()

mock_mtf = Mock()
mock_mtf.MultiTimeframeAnalyzer = Mock()

sys.modules['utils.circuit_breaker'] = mock_circuit_breaker
sys.modules['utils.kelly_criterion'] = mock_kelly
sys.modules['utils.volatility_regime'] = mock_volatility
sys.modules['utils.streak_sizing'] = mock_streak
sys.modules['utils.multi_timeframe_analyzer'] = mock_mtf


# ============================================================================
# Concrete Strategy for Testing
# ============================================================================

from strategies.base_strategy import BaseStrategy


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""

    async def analyze_symbol(self, symbol):
        """Implement abstract method."""
        return 'buy'

    async def execute_trade(self, symbol, signal):
        """Implement abstract method."""
        pass


# ============================================================================
# Test Initialization
# ============================================================================

class TestBaseStrategyInit:
    """Test BaseStrategy initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        strategy = ConcreteStrategy()

        assert strategy.name == "ConcreteStrategy"
        assert strategy.broker is None
        assert strategy.parameters == {}
        assert strategy.interval == 60
        assert strategy.symbols == []
        assert strategy.running is False
        assert strategy.tasks == []
        assert strategy.price_history == {}

    def test_init_with_custom_name(self):
        """Should accept custom name."""
        strategy = ConcreteStrategy(name="MyStrategy")

        assert strategy.name == "MyStrategy"

    def test_init_with_broker(self):
        """Should accept broker."""
        mock_broker = Mock()
        strategy = ConcreteStrategy(broker=mock_broker)

        assert strategy.broker is mock_broker

    def test_init_with_parameters(self):
        """Should accept parameters."""
        params = {
            'interval': 30,
            'symbols': ['AAPL', 'MSFT'],
            'position_size': 0.15
        }
        strategy = ConcreteStrategy(parameters=params)

        assert strategy.interval == 30
        assert strategy.symbols == ['AAPL', 'MSFT']
        assert strategy.parameters['position_size'] == 0.15

    def test_init_circuit_breaker_default(self):
        """Should initialize circuit breaker with default max_daily_loss."""
        strategy = ConcreteStrategy()

        # Just verify it has a circuit_breaker attribute
        assert strategy.circuit_breaker is not None

    def test_init_circuit_breaker_custom(self):
        """Should initialize circuit breaker with custom max_daily_loss."""
        params = {'max_daily_loss': 0.05}
        strategy = ConcreteStrategy(parameters=params)

        # Just verify it has a circuit_breaker attribute
        assert strategy.circuit_breaker is not None

    def test_init_kelly_disabled_by_default(self):
        """Kelly Criterion should be disabled by default."""
        strategy = ConcreteStrategy()

        assert strategy.kelly is None

    def test_init_kelly_enabled(self):
        """Should initialize Kelly Criterion when enabled."""
        params = {'use_kelly_criterion': True, 'kelly_fraction': 0.25}

        strategy = ConcreteStrategy(parameters=params)

        # Kelly should be initialized (not None)
        assert strategy.kelly is not None

    def test_init_volatility_regime_disabled_by_default(self):
        """Volatility regime should be disabled by default."""
        strategy = ConcreteStrategy()

        assert strategy.volatility_regime is None

    def test_init_streak_sizing_disabled_by_default(self):
        """Streak sizing should be disabled by default."""
        strategy = ConcreteStrategy()

        assert strategy.streak_sizer is None

    def test_init_streak_sizing_enabled(self):
        """Should initialize streak sizer when enabled."""
        params = {'use_streak_sizing': True}

        strategy = ConcreteStrategy(parameters=params)

        # Streak sizer should be initialized (not None)
        assert strategy.streak_sizer is not None

    def test_init_multi_timeframe_disabled_by_default(self):
        """Multi-timeframe should be disabled by default."""
        strategy = ConcreteStrategy()

        assert strategy.multi_timeframe is None


# ============================================================================
# Test Initialize Method
# ============================================================================

class TestInitialize:
    """Test async initialize method."""

    @pytest.mark.asyncio
    async def test_initialize_updates_parameters(self):
        """Should update parameters from kwargs."""
        strategy = ConcreteStrategy()

        await strategy.initialize(symbols=['AAPL'], interval=120)

        assert strategy.symbols == ['AAPL']
        assert strategy.interval == 120

    @pytest.mark.asyncio
    async def test_initialize_calls_initialize_parameters(self):
        """Should call _initialize_parameters."""
        strategy = ConcreteStrategy(parameters={'position_size': 0.15})

        result = await strategy.initialize()

        assert result is True
        assert strategy.position_size == 0.15

    @pytest.mark.asyncio
    async def test_initialize_with_broker_initializes_circuit_breaker(self):
        """Should initialize circuit breaker with broker."""
        mock_broker = Mock()
        mock_cb = AsyncMock()
        mock_cb.initialize = AsyncMock()
        mock_cb.max_daily_loss = 0.03

        strategy = ConcreteStrategy(broker=mock_broker)
        strategy.circuit_breaker = mock_cb

        await strategy.initialize()

        mock_cb.initialize.assert_called_once_with(mock_broker)

    @pytest.mark.asyncio
    async def test_initialize_returns_false_on_error(self):
        """Should return False on initialization error."""
        strategy = ConcreteStrategy()

        # Force an error
        with patch.object(strategy, '_initialize_parameters', side_effect=Exception("Test error")):
            result = await strategy.initialize()

        assert result is False


# ============================================================================
# Test _initialize_parameters
# ============================================================================

class TestInitializeParameters:
    """Test _initialize_parameters method."""

    @pytest.mark.asyncio
    async def test_initialize_parameters_defaults(self):
        """Should set default parameter values."""
        strategy = ConcreteStrategy()
        await strategy._initialize_parameters()

        assert strategy.sentiment_threshold == 0.6
        assert strategy.position_size == 0.1
        assert strategy.max_position_size == 0.05
        assert strategy.stop_loss_pct == 0.02
        assert strategy.take_profit_pct == 0.05
        assert strategy.price_history_window == 30

    @pytest.mark.asyncio
    async def test_initialize_parameters_custom(self):
        """Should use custom parameter values."""
        params = {
            'sentiment_threshold': 0.8,
            'position_size': 0.2,
            'stop_loss_pct': 0.03
        }
        strategy = ConcreteStrategy(parameters=params)
        await strategy._initialize_parameters()

        assert strategy.sentiment_threshold == 0.8
        assert strategy.position_size == 0.2
        assert strategy.stop_loss_pct == 0.03


# ============================================================================
# Test Lifecycle Methods
# ============================================================================

class TestLifecycleMethods:
    """Test lifecycle methods."""

    def test_before_market_opens(self):
        """before_market_opens should be callable."""
        strategy = ConcreteStrategy()

        # Should not raise
        strategy.before_market_opens()

    def test_before_starting(self):
        """before_starting should be callable."""
        strategy = ConcreteStrategy()

        # Should not raise
        strategy.before_starting()

    def test_after_market_closes(self):
        """after_market_closes should be callable."""
        strategy = ConcreteStrategy()

        # Should not raise
        strategy.after_market_closes()

    def test_on_abrupt_closing(self):
        """on_abrupt_closing should be callable."""
        strategy = ConcreteStrategy()

        # Should not raise
        strategy.on_abrupt_closing()

    def test_on_bot_crash(self):
        """on_bot_crash should log error."""
        strategy = ConcreteStrategy()

        # Should not raise
        strategy.on_bot_crash(Exception("Test crash"))

    def test_get_parameters(self):
        """get_parameters should return parameters."""
        params = {'position_size': 0.1}
        strategy = ConcreteStrategy(parameters=params)

        result = strategy.get_parameters()

        assert result['position_size'] == 0.1

    def test_set_parameters(self):
        """set_parameters should update parameters."""
        strategy = ConcreteStrategy()

        strategy.set_parameters({'position_size': 0.2})

        assert strategy.parameters['position_size'] == 0.2


# ============================================================================
# Test Circuit Breaker Integration
# ============================================================================

class TestCircuitBreaker:
    """Test circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_check_trading_allowed_when_not_halted(self):
        """Should return True when not halted."""
        strategy = ConcreteStrategy()
        strategy.circuit_breaker = AsyncMock()
        strategy.circuit_breaker.check_and_halt = AsyncMock(return_value=False)

        result = await strategy.check_trading_allowed()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_trading_allowed_when_halted(self):
        """Should return False when halted."""
        strategy = ConcreteStrategy()
        strategy.circuit_breaker = AsyncMock()
        strategy.circuit_breaker.check_and_halt = AsyncMock(return_value=True)

        result = await strategy.check_trading_allowed()

        assert result is False


# ============================================================================
# Test Position Size Enforcement
# ============================================================================

class TestPositionSizeEnforcement:
    """Test enforce_position_size_limit method."""

    @pytest.mark.asyncio
    async def test_enforce_position_size_returns_zero_for_invalid_price(self):
        """Should return 0 for invalid price (zero or negative)."""
        strategy = ConcreteStrategy()

        value, qty = await strategy.enforce_position_size_limit("AAPL", 10000, 0)

        assert value == 0
        assert qty == 0

    @pytest.mark.asyncio
    async def test_enforce_position_size_returns_zero_for_negative_price(self):
        """Should return 0 for negative price."""
        strategy = ConcreteStrategy()

        value, qty = await strategy.enforce_position_size_limit("AAPL", 10000, -100)

        assert value == 0
        assert qty == 0

    @pytest.mark.asyncio
    async def test_enforce_position_size_allows_under_limit(self):
        """Should allow position under limit."""
        mock_broker = Mock()
        mock_account = Mock()
        mock_account.equity = "100000.00"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        strategy = ConcreteStrategy(broker=mock_broker)
        strategy.max_position_size = 0.10  # 10% max

        # Request $5000 (5% of $100k) - under limit
        value, qty = await strategy.enforce_position_size_limit("AAPL", 5000, 150)

        assert value == 5000
        assert qty == pytest.approx(5000 / 150, rel=0.01)

    @pytest.mark.asyncio
    async def test_enforce_position_size_caps_over_limit(self):
        """Should cap position over limit."""
        mock_broker = Mock()
        mock_account = Mock()
        mock_account.equity = "100000.00"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        strategy = ConcreteStrategy(broker=mock_broker)
        strategy.max_position_size = 0.05  # 5% max

        # Request $10000 (10% of $100k) - over limit
        value, qty = await strategy.enforce_position_size_limit("AAPL", 10000, 150)

        # Should cap at 5% = $5000
        assert value == 5000
        assert qty == pytest.approx(5000 / 150, rel=0.01)

    @pytest.mark.asyncio
    async def test_enforce_position_size_handles_error(self):
        """Should return 0 on error."""
        mock_broker = Mock()
        mock_broker.get_account = AsyncMock(side_effect=Exception("API error"))

        strategy = ConcreteStrategy(broker=mock_broker)

        value, qty = await strategy.enforce_position_size_limit("AAPL", 10000, 150)

        assert value == 0
        assert qty == 0


# ============================================================================
# Test Kelly Criterion Position Sizing
# ============================================================================

class TestKellyPositionSizing:
    """Test calculate_kelly_position_size method."""

    @pytest.mark.asyncio
    async def test_kelly_returns_zero_for_invalid_price(self):
        """Should return 0 for invalid price."""
        strategy = ConcreteStrategy()

        value, fraction, qty = await strategy.calculate_kelly_position_size("AAPL", 0)

        assert value == 0
        assert fraction == 0
        assert qty == 0

    @pytest.mark.asyncio
    async def test_kelly_uses_fixed_sizing_when_disabled(self):
        """Should use fixed sizing when Kelly disabled."""
        mock_broker = Mock()
        mock_account = Mock()
        mock_account.equity = "100000.00"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        strategy = ConcreteStrategy(broker=mock_broker)
        strategy.kelly = None
        strategy.position_size = 0.10  # 10% fixed

        value, fraction, qty = await strategy.calculate_kelly_position_size("AAPL", 150)

        assert fraction == 0.10
        assert value == pytest.approx(10000, rel=0.01)  # 10% of $100k
        assert qty == pytest.approx(10000 / 150, rel=0.01)

    @pytest.mark.asyncio
    async def test_kelly_uses_kelly_when_enabled(self):
        """Should use Kelly formula when enabled."""
        mock_broker = Mock()
        mock_account = Mock()
        mock_account.equity = "100000.00"
        mock_broker.get_account = AsyncMock(return_value=mock_account)

        mock_kelly_instance = Mock()
        mock_kelly_instance.calculate_position_size = Mock(return_value=(8000, 0.08))
        mock_kelly_instance.win_rate = 0.55
        mock_kelly_instance.profit_factor = 1.5

        strategy = ConcreteStrategy(broker=mock_broker)
        strategy.kelly = mock_kelly_instance

        value, fraction, qty = await strategy.calculate_kelly_position_size("AAPL", 150)

        assert value == 8000
        assert fraction == 0.08
        assert qty == pytest.approx(8000 / 150, rel=0.01)

    @pytest.mark.asyncio
    async def test_kelly_handles_error(self):
        """Should return 0 on error."""
        mock_broker = Mock()
        mock_broker.get_account = AsyncMock(side_effect=Exception("API error"))

        strategy = ConcreteStrategy(broker=mock_broker)

        value, fraction, qty = await strategy.calculate_kelly_position_size("AAPL", 150)

        assert value == 0
        assert fraction == 0
        assert qty == 0


# ============================================================================
# Test Trade Tracking
# ============================================================================

class TestTradeTracking:
    """Test track_position_entry and record_completed_trade."""

    def test_track_position_entry(self):
        """Should track position entry."""
        strategy = ConcreteStrategy()

        strategy.track_position_entry("AAPL", 150.00)

        assert "AAPL" in strategy.closed_positions
        assert strategy.closed_positions["AAPL"]["entry_price"] == 150.00
        assert "entry_time" in strategy.closed_positions["AAPL"]

    def test_track_position_entry_with_custom_time(self):
        """Should track entry with custom time."""
        strategy = ConcreteStrategy()
        custom_time = datetime(2024, 1, 1, 10, 0, 0)

        strategy.track_position_entry("AAPL", 150.00, entry_time=custom_time)

        assert strategy.closed_positions["AAPL"]["entry_time"] == custom_time

    def test_record_completed_trade_without_kelly(self):
        """Should do nothing when Kelly disabled."""
        strategy = ConcreteStrategy()
        strategy.kelly = None

        # Should not raise
        strategy.record_completed_trade("AAPL", 160.00, datetime.now(), 100)

    def test_record_completed_trade_without_entry_tracking(self):
        """Should warn when entry not tracked."""
        mock_kelly_instance = Mock()
        strategy = ConcreteStrategy()
        strategy.kelly = mock_kelly_instance

        # Record trade without tracking entry
        strategy.record_completed_trade("AAPL", 160.00, datetime.now(), 100)

        # add_trade should NOT be called
        mock_kelly_instance.add_trade.assert_not_called()

    def test_record_completed_trade_long_win(self):
        """Should record long winning trade."""
        mock_kelly_instance = Mock()
        mock_kelly_instance.trades = []  # Add trades attribute for len()
        strategy = ConcreteStrategy()
        strategy.kelly = mock_kelly_instance

        # Track entry
        entry_time = datetime(2024, 1, 1, 10, 0, 0)
        strategy.track_position_entry("AAPL", 150.00, entry_time=entry_time)

        # Record exit
        exit_time = datetime(2024, 1, 2, 10, 0, 0)
        strategy.record_completed_trade("AAPL", 160.00, exit_time, 100, side='long')

        mock_kelly_instance.add_trade.assert_called_once()

        # Symbol should be removed from tracking
        assert "AAPL" not in strategy.closed_positions

    def test_record_completed_trade_short_win(self):
        """Should record short winning trade."""
        mock_kelly_instance = Mock()
        mock_kelly_instance.trades = []  # Add trades attribute for len()
        strategy = ConcreteStrategy()
        strategy.kelly = mock_kelly_instance

        # Track entry
        strategy.track_position_entry("AAPL", 150.00)

        # Record exit (short = profit when price goes down)
        strategy.record_completed_trade("AAPL", 140.00, datetime.now(), 100, side='short')

        mock_kelly_instance.add_trade.assert_called_once()


# ============================================================================
# Test Volatility Adjustments
# ============================================================================

class TestVolatilityAdjustments:
    """Test apply_volatility_adjustments method."""

    @pytest.mark.asyncio
    async def test_returns_base_values_when_disabled(self):
        """Should return base values when volatility regime disabled."""
        strategy = ConcreteStrategy()
        strategy.volatility_regime = None

        pos_size, stop_loss, regime = await strategy.apply_volatility_adjustments(0.10, 0.03)

        assert pos_size == 0.10
        assert stop_loss == 0.03
        assert regime == 'normal'

    @pytest.mark.asyncio
    async def test_applies_adjustments_when_enabled(self):
        """Should apply adjustments when enabled."""
        mock_regime = AsyncMock()
        mock_regime.get_current_regime = AsyncMock(return_value=('high', {'pos_mult': 0.5, 'stop_mult': 1.5}))
        mock_regime.adjust_position_size = Mock(return_value=0.05)  # 10% * 0.5
        mock_regime.adjust_stop_loss = Mock(return_value=0.045)  # 3% * 1.5

        strategy = ConcreteStrategy()
        strategy.volatility_regime = mock_regime

        pos_size, stop_loss, regime = await strategy.apply_volatility_adjustments(0.10, 0.03)

        assert pos_size == 0.05
        assert stop_loss == 0.045
        assert regime == 'high'

    @pytest.mark.asyncio
    async def test_handles_error(self):
        """Should return base values on error."""
        mock_regime = AsyncMock()
        mock_regime.get_current_regime = AsyncMock(side_effect=Exception("API error"))

        strategy = ConcreteStrategy()
        strategy.volatility_regime = mock_regime

        pos_size, stop_loss, regime = await strategy.apply_volatility_adjustments(0.10, 0.03)

        assert pos_size == 0.10
        assert stop_loss == 0.03
        assert regime == 'normal'


# ============================================================================
# Test Streak Adjustments
# ============================================================================

class TestStreakAdjustments:
    """Test apply_streak_adjustments method."""

    def test_returns_base_value_when_disabled(self):
        """Should return base value when streak sizing disabled."""
        strategy = ConcreteStrategy()
        strategy.streak_sizer = None

        result = strategy.apply_streak_adjustments(0.10)

        assert result == 0.10

    def test_applies_adjustment_when_enabled(self):
        """Should apply adjustment when enabled."""
        mock_sizer = Mock()
        mock_sizer.adjust_for_streak = Mock(return_value=0.12)  # Hot streak

        strategy = ConcreteStrategy()
        strategy.streak_sizer = mock_sizer

        result = strategy.apply_streak_adjustments(0.10)

        assert result == 0.12
        mock_sizer.adjust_for_streak.assert_called_once_with(0.10)

    def test_handles_error(self):
        """Should return base value on error."""
        mock_sizer = Mock()
        mock_sizer.adjust_for_streak = Mock(side_effect=Exception("Error"))

        strategy = ConcreteStrategy()
        strategy.streak_sizer = mock_sizer

        result = strategy.apply_streak_adjustments(0.10)

        assert result == 0.10


# ============================================================================
# Test Multi-Timeframe Signal
# ============================================================================

class TestMultiTimeframeSignal:
    """Test check_multi_timeframe_signal method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """Should return None when multi-timeframe disabled."""
        strategy = ConcreteStrategy()
        strategy.multi_timeframe = None

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_signal_when_confirmed(self):
        """Should return signal when confirmed."""
        mock_mtf = AsyncMock()
        mock_mtf.analyze = AsyncMock(return_value={
            'should_enter': True,
            'signal': 'buy',
            'confidence': 0.85
        })

        strategy = ConcreteStrategy()
        strategy.multi_timeframe = mock_mtf
        strategy.mtf_min_confidence = 0.70
        strategy.mtf_require_daily = True

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result == 'buy'

    @pytest.mark.asyncio
    async def test_returns_none_when_rejected(self):
        """Should return None when signal rejected."""
        mock_mtf = AsyncMock()
        mock_mtf.analyze = AsyncMock(return_value={
            'should_enter': False,
            'signal': 'buy',
            'confidence': 0.50
        })

        strategy = ConcreteStrategy()
        strategy.multi_timeframe = mock_mtf
        strategy.mtf_min_confidence = 0.70
        strategy.mtf_require_daily = True

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_analysis_failure(self):
        """Should return None when analysis fails."""
        mock_mtf = AsyncMock()
        mock_mtf.analyze = AsyncMock(return_value=None)

        strategy = ConcreteStrategy()
        strategy.multi_timeframe = mock_mtf

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        """Should return None on error."""
        mock_mtf = AsyncMock()
        mock_mtf.analyze = AsyncMock(side_effect=Exception("Error"))

        strategy = ConcreteStrategy()
        strategy.multi_timeframe = mock_mtf

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result is None


# ============================================================================
# Test Position Helpers
# ============================================================================

class TestPositionHelpers:
    """Test is_short_position and get_position_pnl methods."""

    @pytest.mark.asyncio
    async def test_is_short_position_true(self):
        """Should return True for short position."""
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "-100"  # Negative = short

        mock_broker = Mock()
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.is_short_position("AAPL")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_short_position_false_for_long(self):
        """Should return False for long position."""
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"

        mock_broker = Mock()
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.is_short_position("AAPL")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_short_position_false_for_no_position(self):
        """Should return False when no position."""
        mock_broker = Mock()
        mock_broker.get_positions = AsyncMock(return_value=[])

        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.is_short_position("AAPL")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_short_position_handles_error(self):
        """Should return False on error."""
        mock_broker = Mock()
        mock_broker.get_positions = AsyncMock(side_effect=Exception("Error"))

        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.is_short_position("AAPL")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_position_pnl_returns_data(self):
        """Should return P/L data for position."""
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.unrealized_pl = "500.00"
        mock_position.unrealized_plpc = "0.05"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_position.current_price = "155.00"
        mock_position.market_value = "15500.00"

        mock_broker = Mock()
        mock_broker.get_positions = AsyncMock(return_value=[mock_position])

        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.get_position_pnl("AAPL")

        assert result['unrealized_pl'] == 500.00
        assert result['unrealized_plpc'] == 0.05
        assert result['qty'] == 100
        assert result['is_short'] is False

    @pytest.mark.asyncio
    async def test_get_position_pnl_returns_none_for_no_position(self):
        """Should return None when no position."""
        mock_broker = Mock()
        mock_broker.get_positions = AsyncMock(return_value=[])

        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.get_position_pnl("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_position_pnl_handles_error(self):
        """Should return None on error."""
        mock_broker = Mock()
        mock_broker.get_positions = AsyncMock(side_effect=Exception("Error"))

        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.get_position_pnl("AAPL")

        assert result is None


# ============================================================================
# Test Create Order
# ============================================================================

class TestCreateOrder:
    """Test create_order method."""

    def test_create_market_order(self):
        """Should create market order."""
        strategy = ConcreteStrategy()

        order = strategy.create_order("AAPL", 100, "buy")

        assert order["symbol"] == "AAPL"
        assert order["quantity"] == 100
        assert order["side"] == "buy"
        assert order["type"] == "market"

    def test_create_limit_order(self):
        """Should create limit order."""
        strategy = ConcreteStrategy()

        order = strategy.create_order("AAPL", 100, "sell", type="limit", limit_price=155.00)

        assert order["symbol"] == "AAPL"
        assert order["quantity"] == 100
        assert order["side"] == "sell"
        assert order["type"] == "limit"
        assert order["limit_price"] == 155.00

    def test_create_stop_order(self):
        """Should create stop order."""
        strategy = ConcreteStrategy()

        order = strategy.create_order("AAPL", 100, "sell", type="stop", stop_price=145.00)

        assert order["type"] == "stop"
        assert order["stop_price"] == 145.00


# ============================================================================
# Test Volatility Calculation
# ============================================================================

class TestVolatilityCalculation:
    """Test _calculate_volatility method."""

    def test_calculate_volatility_with_insufficient_data(self):
        """Should return 0 with insufficient data."""
        strategy = ConcreteStrategy()
        strategy.price_history = {"AAPL": [150.0] * 10}  # Only 10 prices
        strategy.price_history_window = 30

        result = strategy._calculate_volatility("AAPL")

        assert result == 0

    def test_calculate_volatility_with_no_data(self):
        """Should return 0 with no data."""
        strategy = ConcreteStrategy()
        strategy.price_history = {}
        strategy.price_history_window = 30

        result = strategy._calculate_volatility("AAPL")

        assert result == 0

    def test_calculate_volatility_with_sufficient_data(self):
        """Should calculate volatility with sufficient data."""
        strategy = ConcreteStrategy()

        # Create price data with some volatility
        prices = [100 + i * 0.5 + np.random.randn() for i in range(50)]
        strategy.price_history = {"AAPL": prices}
        strategy.price_history_window = 30

        result = strategy._calculate_volatility("AAPL")

        # Should be a positive number
        assert result > 0


# ============================================================================
# Test Performance Metrics
# ============================================================================

class TestPerformanceMetrics:
    """Test update_performance_metrics method."""

    def test_update_metrics_winning_trade(self):
        """Should update metrics for winning trade."""
        strategy = ConcreteStrategy()
        strategy.trades_made = 0
        strategy.successful_trades = 0
        strategy.total_profit_loss = 0

        strategy.update_performance_metrics(100, "AAPL")

        assert strategy.trades_made == 1
        assert strategy.successful_trades == 1
        assert strategy.total_profit_loss == 100

    def test_update_metrics_losing_trade(self):
        """Should update metrics for losing trade."""
        strategy = ConcreteStrategy()
        strategy.trades_made = 0
        strategy.successful_trades = 0
        strategy.total_profit_loss = 0

        strategy.update_performance_metrics(-50, "AAPL")

        assert strategy.trades_made == 1
        assert strategy.successful_trades == 0
        assert strategy.total_profit_loss == -50


# ============================================================================
# Test Cleanup and Shutdown
# ============================================================================

class TestCleanupAndShutdown:
    """Test cleanup and shutdown methods."""

    @pytest.mark.asyncio
    async def test_cleanup_cancels_tasks(self):
        """Should cancel running tasks."""
        strategy = ConcreteStrategy()
        strategy.running = True

        # Create a real async task that we can cancel
        async def dummy_coro():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy_coro())
        strategy.tasks = [task]

        await strategy.cleanup()

        assert strategy.running is False
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_cleanup_with_no_tasks(self):
        """Should handle cleanup with no tasks."""
        strategy = ConcreteStrategy()
        strategy.running = True
        strategy.tasks = []

        await strategy.cleanup()

        assert strategy.running is False

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self):
        """Should set shutdown event."""
        strategy = ConcreteStrategy()

        await strategy.shutdown()

        assert strategy._shutdown_event.is_set()


# ============================================================================
# Test Abstract Methods
# ============================================================================

class TestAbstractMethods:
    """Test that abstract methods raise NotImplementedError."""

    @pytest.mark.asyncio
    async def test_on_trading_iteration_raises(self):
        """on_trading_iteration should raise NotImplementedError."""
        strategy = ConcreteStrategy()

        with pytest.raises(NotImplementedError):
            await strategy.on_trading_iteration()

    @pytest.mark.asyncio
    async def test_backtest_raises(self):
        """backtest should raise NotImplementedError."""
        strategy = ConcreteStrategy()

        with pytest.raises(NotImplementedError):
            await strategy.backtest()


# ============================================================================
# Test Legacy Initialize
# ============================================================================

class TestLegacyInitialize:
    """Test _legacy_initialize method."""

    def test_legacy_initialize_sets_attributes(self):
        """Should set legacy attributes."""
        strategy = ConcreteStrategy()
        strategy.portfolio_value = 100000

        strategy._legacy_initialize(
            symbols=['AAPL', 'MSFT'],
            cash_at_risk=0.4,
            max_positions=5,
            stop_loss_pct=0.03,
            take_profit_pct=0.15,
            max_drawdown=0.10
        )

        assert strategy.symbols == ['AAPL', 'MSFT']
        assert strategy.cash_at_risk == 0.4
        assert strategy.max_positions == 5
        assert strategy.stop_loss_pct == 0.03
        assert strategy.take_profit_pct == 0.15
        assert strategy.max_drawdown == 0.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
