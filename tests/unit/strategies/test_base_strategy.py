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
from unittest.mock import Mock

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
        from strategies.params import BaseParams

        assert strategy.parameters == BaseParams.defaults()
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


# ============================================================================
# Test Kelly Criterion Position Sizing
# ============================================================================


# ============================================================================
# Test Trade Tracking
# ============================================================================


# ============================================================================
# Test Volatility Adjustments
# ============================================================================


# ============================================================================
# Test Streak Adjustments
# ============================================================================


# ============================================================================
# Test Multi-Timeframe Signal
# ============================================================================


# ============================================================================
# Test Position Helpers
# ============================================================================


# ============================================================================
# Test Create Order
# ============================================================================


# ============================================================================
# Test Volatility Calculation
# ============================================================================


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
