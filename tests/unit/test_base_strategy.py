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

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

# Import the actual modules - we'll use @patch decorators for isolation
from strategies.base_strategy import BaseStrategy


# ============================================================================
# Concrete Strategy for Testing
# ============================================================================


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""

    async def analyze_symbol(self, symbol):
        """Implement abstract method."""
        return "buy"

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
        params = {"interval": 30, "symbols": ["AAPL", "MSFT"], "position_size": 0.15}
        strategy = ConcreteStrategy(parameters=params)

        assert strategy.interval == 30
        assert strategy.symbols == ["AAPL", "MSFT"]
        assert strategy.parameters["position_size"] == 0.15

    def test_init_circuit_breaker_default(self):
        """Should initialize circuit breaker with default max_daily_loss."""
        strategy = ConcreteStrategy()

        assert strategy.circuit_breaker is not None

    def test_init_circuit_breaker_custom(self):
        """Should initialize circuit breaker with custom max_daily_loss."""
        strategy = ConcreteStrategy(parameters={"max_daily_loss": 0.05})

        assert strategy.circuit_breaker is not None

    def test_init_kelly_disabled_by_default(self):
        """Kelly should be None when not enabled."""
        strategy = ConcreteStrategy()

        assert strategy.kelly is None

    def test_init_kelly_enabled(self):
        """Kelly should be initialized when enabled."""
        strategy = ConcreteStrategy(parameters={"use_kelly_criterion": True})

        assert strategy.kelly is not None

    def test_init_volatility_regime_disabled_by_default(self):
        """Volatility regime should be None when not enabled."""
        strategy = ConcreteStrategy()

        assert strategy.volatility_regime is None

    def test_init_volatility_regime_enabled(self):
        """Volatility regime should be marked for init when enabled."""
        strategy = ConcreteStrategy(parameters={"use_volatility_regime": True})

        # Gets initialized in async initialize(), so it's None here
        assert strategy.volatility_regime is None

    def test_init_streak_sizer_disabled_by_default(self):
        """Streak sizer should be None when not enabled."""
        strategy = ConcreteStrategy()

        assert strategy.streak_sizer is None

    def test_init_streak_sizer_enabled(self):
        """Streak sizer should be initialized when enabled."""
        strategy = ConcreteStrategy(parameters={"use_streak_sizing": True})

        assert strategy.streak_sizer is not None

    def test_init_multi_timeframe_disabled_by_default(self):
        """Multi-timeframe should be None when not enabled."""
        strategy = ConcreteStrategy()

        assert strategy.multi_timeframe is None


# ============================================================================
# Test Initialize Method
# ============================================================================


class TestInitialize:
    """Test async initialize method."""

    @pytest.mark.asyncio
    async def test_initialize_returns_true(self):
        """Initialize should return True on success."""
        strategy = ConcreteStrategy()

        result = await strategy.initialize()

        assert result is True

    @pytest.mark.asyncio
    async def test_initialize_updates_parameters(self):
        """Initialize should update parameters with kwargs."""
        strategy = ConcreteStrategy()

        await strategy.initialize(symbols=["AAPL"], interval=120)

        assert strategy.symbols == ["AAPL"]
        assert strategy.interval == 120

    @pytest.mark.asyncio
    async def test_initialize_with_broker_initializes_circuit_breaker(self):
        """Initialize with broker should set up circuit breaker."""
        mock_broker = AsyncMock()
        mock_broker.get_account = AsyncMock(
            return_value=MagicMock(equity="100000", cash="50000")
        )
        strategy = ConcreteStrategy(broker=mock_broker)

        await strategy.initialize()

        # Circuit breaker should have been initialized with broker
        assert strategy.circuit_breaker is not None


# ============================================================================
# Test Lifecycle Methods
# ============================================================================


class TestLifecycleMethods:
    """Test lifecycle methods."""

    def test_set_parameters(self):
        """set_parameters should update strategy parameters."""
        strategy = ConcreteStrategy()

        strategy.set_parameters({"position_size": 0.15, "stop_loss": 0.03})

        assert strategy.parameters.get("position_size") == 0.15
        assert strategy.parameters.get("stop_loss") == 0.03


# ============================================================================
# Test Position Size Enforcement
# ============================================================================


class TestPositionSizeEnforcement:
    """Test position size limit enforcement."""

    @pytest.mark.asyncio
    async def test_enforce_position_size_limit_allows_under_limit(self):
        """Should allow position sizes under the limit."""
        mock_broker = AsyncMock()
        mock_broker.get_account.return_value = MagicMock(
            equity="100000", cash="50000"
        )
        mock_broker.get_positions.return_value = []

        strategy = ConcreteStrategy(broker=mock_broker)
        await strategy.initialize()

        # Request size under max_position_size (5% of 100k = 5000)
        capped_value, capped_qty = await strategy.enforce_position_size_limit(
            "AAPL", 3000, 150.0
        )

        assert capped_value == 3000
        assert capped_qty == 20.0  # 3000 / 150

    @pytest.mark.asyncio
    async def test_enforce_position_size_limit_caps_over_limit(self):
        """Should cap position sizes over the limit."""
        mock_broker = AsyncMock()
        mock_broker.get_account.return_value = MagicMock(
            equity="100000", cash="50000"
        )
        mock_broker.get_positions.return_value = []

        strategy = ConcreteStrategy(broker=mock_broker)
        await strategy.initialize()

        # Request size over max_position_size (5% of 100k = 5000)
        capped_value, capped_qty = await strategy.enforce_position_size_limit(
            "AAPL", 10000, 150.0
        )

        assert capped_value == 5000  # Capped at 5%
        assert abs(capped_qty - 33.33) < 0.1  # 5000 / 150

    @pytest.mark.asyncio
    async def test_enforce_position_size_limit_returns_zero_on_invalid_price(self):
        """Should return 0 for invalid price."""
        mock_broker = AsyncMock()
        strategy = ConcreteStrategy(broker=mock_broker)

        capped_value, capped_qty = await strategy.enforce_position_size_limit(
            "AAPL", 5000, 0
        )

        assert capped_value == 0
        assert capped_qty == 0


# ============================================================================
# Test Kelly Criterion Position Sizing
# ============================================================================


class TestKellyPositionSizing:
    """Test Kelly Criterion position sizing."""

    @pytest.mark.asyncio
    async def test_calculate_kelly_position_size_returns_values(self):
        """Should return position size values."""
        mock_broker = AsyncMock()
        mock_broker.get_account.return_value = MagicMock(equity="100000")

        strategy = ConcreteStrategy(
            broker=mock_broker,
            parameters={"use_kelly_criterion": True}
        )
        await strategy.initialize()

        position_value, position_fraction, quantity = (
            await strategy.calculate_kelly_position_size("AAPL", 150.0)
        )

        assert position_value >= 0
        assert 0 <= position_fraction <= 1
        assert quantity >= 0

    @pytest.mark.asyncio
    async def test_calculate_kelly_position_size_without_kelly(self):
        """Should use fixed sizing when Kelly is disabled."""
        mock_broker = AsyncMock()
        mock_broker.get_account.return_value = MagicMock(equity="100000")

        strategy = ConcreteStrategy(broker=mock_broker)
        await strategy.initialize()

        position_value, position_fraction, quantity = (
            await strategy.calculate_kelly_position_size("AAPL", 150.0)
        )

        # Should use position_size parameter (default 0.1)
        assert position_value == 10000  # 10% of 100000
        assert position_fraction == 0.1
        assert abs(quantity - 66.67) < 0.1  # 10000 / 150


# ============================================================================
# Test Trade Tracking
# ============================================================================


class TestTradeTracking:
    """Test trade tracking functionality."""

    @pytest.mark.asyncio
    async def test_track_position_entry(self):
        """Should track position entry."""
        strategy = ConcreteStrategy()
        await strategy.initialize()

        strategy.track_position_entry("AAPL", 150.0)

        assert "AAPL" in strategy.closed_positions
        assert strategy.closed_positions["AAPL"]["entry_price"] == 150.0

    @pytest.mark.asyncio
    async def test_track_position_entry_with_custom_time(self):
        """Should track position entry with custom time."""
        strategy = ConcreteStrategy()
        await strategy.initialize()

        entry_time = datetime(2024, 1, 15, 10, 30)
        strategy.track_position_entry("AAPL", 150.0, entry_time)

        assert strategy.closed_positions["AAPL"]["entry_time"] == entry_time

    @pytest.mark.asyncio
    async def test_record_completed_trade_without_kelly(self):
        """Should handle recording trade when Kelly is disabled."""
        strategy = ConcreteStrategy()
        await strategy.initialize()

        # Should not raise - returns early when Kelly is None
        strategy.record_completed_trade(
            "AAPL", 155.0, datetime.now(), 10, "long"
        )

    @pytest.mark.asyncio
    async def test_record_completed_trade_without_entry_tracking(self):
        """Should handle recording trade without prior entry tracking."""
        strategy = ConcreteStrategy(parameters={"use_kelly_criterion": True})
        await strategy.initialize()

        # Should not raise - logs warning and returns
        strategy.record_completed_trade(
            "AAPL", 155.0, datetime.now(), 10, "long"
        )

    @pytest.mark.asyncio
    async def test_record_completed_trade_long_win(self):
        """Should record winning long trade."""
        strategy = ConcreteStrategy(parameters={"use_kelly_criterion": True})
        await strategy.initialize()

        strategy.track_position_entry("AAPL", 150.0)
        strategy.record_completed_trade(
            "AAPL", 155.0, datetime.now(), 10, "long"
        )

        # Entry should be cleared
        assert "AAPL" not in strategy.closed_positions

    @pytest.mark.asyncio
    async def test_record_completed_trade_short_win(self):
        """Should record winning short trade."""
        strategy = ConcreteStrategy(parameters={"use_kelly_criterion": True})
        await strategy.initialize()

        strategy.track_position_entry("TSLA", 300.0)
        strategy.record_completed_trade(
            "TSLA", 280.0, datetime.now(), 10, "short"
        )

        assert "TSLA" not in strategy.closed_positions


# ============================================================================
# Test Volatility Adjustments
# ============================================================================


class TestVolatilityAdjustments:
    """Test volatility regime adjustments."""

    @pytest.mark.asyncio
    async def test_returns_base_values_when_disabled(self):
        """Should return base values when volatility regime is disabled."""
        strategy = ConcreteStrategy()
        await strategy.initialize()

        adj_pos, adj_stop, regime = await strategy.apply_volatility_adjustments(
            0.10, 0.03
        )

        assert adj_pos == 0.10
        assert adj_stop == 0.03
        assert regime == "normal"

    @pytest.mark.asyncio
    async def test_applies_adjustments_when_enabled(self):
        """Should apply adjustments when volatility regime is enabled."""
        mock_broker = AsyncMock()
        mock_broker.get_latest_quote.return_value = MagicMock(ask_price=15.0)

        strategy = ConcreteStrategy(
            broker=mock_broker,
            parameters={"use_volatility_regime": True}
        )
        await strategy.initialize()

        adj_pos, adj_stop, regime = await strategy.apply_volatility_adjustments(
            0.10, 0.03
        )

        assert isinstance(adj_pos, (int, float))
        assert isinstance(adj_stop, (int, float))
        assert isinstance(regime, str)

    @pytest.mark.asyncio
    async def test_handles_error(self):
        """Should handle errors gracefully."""
        strategy = ConcreteStrategy(parameters={"use_volatility_regime": True})
        await strategy.initialize()

        # Force error by setting volatility_regime to a mock that raises
        strategy.volatility_regime = MagicMock()
        strategy.volatility_regime.get_current_regime = AsyncMock(
            side_effect=Exception("VIX unavailable")
        )

        adj_pos, adj_stop, regime = await strategy.apply_volatility_adjustments(
            0.10, 0.03
        )

        # Should return base values on error
        assert adj_pos == 0.10
        assert adj_stop == 0.03
        assert regime == "normal"


# ============================================================================
# Test Streak Adjustments
# ============================================================================


class TestStreakAdjustments:
    """Test streak-based position adjustments."""

    def test_returns_base_value_when_disabled(self):
        """Should return base value when streak sizing is disabled."""
        strategy = ConcreteStrategy()

        adjusted = strategy.apply_streak_adjustments(0.10)

        assert adjusted == 0.10

    def test_applies_adjustment_when_enabled(self):
        """Should apply adjustment when streak sizing is enabled."""
        strategy = ConcreteStrategy(parameters={"use_streak_sizing": True})

        adjusted = strategy.apply_streak_adjustments(0.10)

        assert isinstance(adjusted, float)

    def test_handles_error(self):
        """Should handle errors gracefully."""
        strategy = ConcreteStrategy(parameters={"use_streak_sizing": True})

        # Force error by patching the streak_sizer method
        strategy.streak_sizer.adjust_for_streak = Mock(
            side_effect=Exception("Test error")
        )

        adjusted = strategy.apply_streak_adjustments(0.10)

        # Should return base value on error
        assert adjusted == 0.10


# ============================================================================
# Test Multi-Timeframe Signal
# ============================================================================


class TestMultiTimeframeSignal:
    """Test multi-timeframe signal checking."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """Should return None when MTF is disabled."""
        strategy = ConcreteStrategy()
        await strategy.initialize()

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_signal_when_confirmed(self):
        """Should return signal when MTF confirms."""
        mock_broker = AsyncMock()
        strategy = ConcreteStrategy(
            broker=mock_broker,
            parameters={"use_multi_timeframe": True}
        )
        await strategy.initialize()

        # Set up mock MTF analyzer
        strategy.multi_timeframe = MagicMock()
        strategy.multi_timeframe.analyze = AsyncMock(return_value={
            "should_enter": True,
            "signal": "buy",
            "confidence": 0.85
        })

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result == "buy"

    @pytest.mark.asyncio
    async def test_returns_none_when_rejected(self):
        """Should return None when MTF rejects signal."""
        mock_broker = AsyncMock()
        strategy = ConcreteStrategy(
            broker=mock_broker,
            parameters={"use_multi_timeframe": True}
        )
        await strategy.initialize()

        # Set up mock MTF analyzer
        strategy.multi_timeframe = MagicMock()
        strategy.multi_timeframe.analyze = AsyncMock(return_value={
            "should_enter": False,
            "signal": "neutral",
            "confidence": 0.45
        })

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_analysis_failure(self):
        """Should return None when MTF analysis fails."""
        mock_broker = AsyncMock()
        strategy = ConcreteStrategy(
            broker=mock_broker,
            parameters={"use_multi_timeframe": True}
        )
        await strategy.initialize()

        # Set up mock MTF analyzer that returns None
        strategy.multi_timeframe = MagicMock()
        strategy.multi_timeframe.analyze = AsyncMock(return_value=None)

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        """Should return None on error."""
        mock_broker = AsyncMock()
        strategy = ConcreteStrategy(
            broker=mock_broker,
            parameters={"use_multi_timeframe": True}
        )
        await strategy.initialize()

        # Set up mock MTF analyzer that raises
        strategy.multi_timeframe = MagicMock()
        strategy.multi_timeframe.analyze = AsyncMock(
            side_effect=Exception("Test error")
        )

        result = await strategy.check_multi_timeframe_signal("AAPL")

        assert result is None


# ============================================================================
# Test Position Helpers
# ============================================================================


class TestPositionHelpers:
    """Test position helper methods."""

    @pytest.mark.asyncio
    async def test_is_short_position_true(self):
        """Should return True for short position."""
        mock_broker = AsyncMock()
        mock_broker.get_positions.return_value = [
            MagicMock(symbol="AAPL", qty="-10")
        ]
        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.is_short_position("AAPL")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_short_position_false_for_long(self):
        """Should return False for long position."""
        mock_broker = AsyncMock()
        mock_broker.get_positions.return_value = [
            MagicMock(symbol="AAPL", qty="10")
        ]
        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.is_short_position("AAPL")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_short_position_false_for_no_position(self):
        """Should return False when no position exists."""
        mock_broker = AsyncMock()
        mock_broker.get_positions.return_value = []
        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.is_short_position("AAPL")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_short_position_handles_error(self):
        """Should handle errors gracefully."""
        mock_broker = AsyncMock()
        mock_broker.get_positions.side_effect = Exception("API Error")
        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.is_short_position("AAPL")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_position_pnl_returns_data(self):
        """Should return P&L data for position."""
        mock_broker = AsyncMock()
        mock_broker.get_positions.return_value = [
            MagicMock(
                symbol="AAPL",
                qty="10",
                unrealized_pl="100.00",
                unrealized_plpc="0.05",
                avg_entry_price="145.00",
                current_price="155.00",
                market_value="1550.00"
            )
        ]
        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.get_position_pnl("AAPL")

        assert result is not None
        assert result["unrealized_pl"] == 100.0
        assert result["unrealized_plpc"] == 0.05

    @pytest.mark.asyncio
    async def test_get_position_pnl_returns_none_for_no_position(self):
        """Should return None when no position exists."""
        mock_broker = AsyncMock()
        mock_broker.get_positions.return_value = []
        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.get_position_pnl("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_position_pnl_handles_error(self):
        """Should handle errors gracefully."""
        mock_broker = AsyncMock()
        mock_broker.get_positions.side_effect = Exception("API Error")
        strategy = ConcreteStrategy(broker=mock_broker)

        result = await strategy.get_position_pnl("AAPL")

        assert result is None


# ============================================================================
# Test Create Order
# ============================================================================


class TestCreateOrder:
    """Test order creation."""

    def test_create_market_order(self):
        """Should create market order request."""
        strategy = ConcreteStrategy()

        order = strategy.create_order(
            symbol="AAPL",
            quantity=10,
            side="buy",
            type="market"
        )

        assert order is not None
        assert order["symbol"] == "AAPL"
        assert order["quantity"] == 10
        assert order["side"] == "buy"
        assert order["type"] == "market"

    def test_create_limit_order(self):
        """Should create limit order request."""
        strategy = ConcreteStrategy()

        order = strategy.create_order(
            symbol="AAPL",
            quantity=10,
            side="buy",
            type="limit",
            limit_price=150.0
        )

        assert order is not None
        assert order["limit_price"] == 150.0

    def test_create_stop_order(self):
        """Should create stop order request."""
        strategy = ConcreteStrategy()

        order = strategy.create_order(
            symbol="AAPL",
            quantity=10,
            side="sell",
            type="stop",
            stop_price=145.0
        )

        assert order is not None
        assert order["stop_price"] == 145.0


# ============================================================================
# Test Volatility Calculation
# ============================================================================


class TestVolatilityCalculation:
    """Test volatility calculation methods."""

    def test_calculate_volatility_with_insufficient_data(self):
        """Should handle insufficient data."""
        strategy = ConcreteStrategy()
        strategy.price_history_window = 30
        strategy.price_history["AAPL"] = [100.0, 101.0]  # Not enough data

        vol = strategy._calculate_volatility("AAPL")

        assert vol == 0  # Returns 0 for insufficient data

    def test_calculate_volatility_with_no_data(self):
        """Should handle missing price history."""
        strategy = ConcreteStrategy()
        strategy.price_history_window = 30

        vol = strategy._calculate_volatility("AAPL")

        assert vol == 0  # Returns 0 for missing data

    def test_calculate_volatility_with_sufficient_data(self):
        """Should calculate volatility with sufficient data."""
        strategy = ConcreteStrategy()
        # Generate reasonable price data
        np.random.seed(42)
        base = 100
        prices = [base * (1 + np.random.normal(0, 0.02)) for _ in range(30)]
        strategy.price_history["AAPL"] = prices
        strategy.price_history_window = 20

        vol = strategy._calculate_volatility("AAPL")

        assert vol is not None
        assert vol >= 0


# ============================================================================
# Test Performance Metrics
# ============================================================================


# ============================================================================
# Test Cleanup and Shutdown
# ============================================================================


class TestCleanupAndShutdown:
    """Test cleanup and shutdown methods."""

    @pytest.mark.asyncio
    async def test_cleanup_cancels_tasks(self):
        """Should cancel running tasks."""
        strategy = ConcreteStrategy()

        # Add mock task
        async def dummy():
            await asyncio.sleep(10)

        task = asyncio.create_task(dummy())
        strategy.tasks = [task]

        await strategy.cleanup()

        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_cleanup_with_no_tasks(self):
        """Should handle cleanup with no tasks."""
        strategy = ConcreteStrategy()
        strategy.tasks = []

        # Should not raise
        await strategy.cleanup()

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
    """Test that abstract methods must be implemented."""

    @pytest.mark.asyncio
    async def test_analyze_symbol_works(self):
        """analyze_symbol should work in concrete class."""
        strategy = ConcreteStrategy()

        result = await strategy.analyze_symbol("AAPL")
        assert result == "buy"

    @pytest.mark.asyncio
    async def test_on_trading_iteration_raises(self):
        """on_trading_iteration should raise NotImplementedError in base."""
        strategy = ConcreteStrategy()

        with pytest.raises(NotImplementedError):
            await strategy.on_trading_iteration()


# ============================================================================
# Test Legacy Initialize
# ============================================================================


class TestInitAttributes:
    """Test that __init__ sets expected attributes."""

    def test_init_sets_expected_attributes(self):
        """__init__ should set expected attributes."""
        strategy = ConcreteStrategy()

        # Access attributes set in __init__
        assert hasattr(strategy, "circuit_breaker")
        assert hasattr(strategy, "kelly")
        assert hasattr(strategy, "volatility_regime")
        assert hasattr(strategy, "streak_sizer")
        assert hasattr(strategy, "multi_timeframe")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
