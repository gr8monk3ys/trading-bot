"""
Unit tests for SimpleMACrossoverStrategy.

Tests the simple moving average crossover strategy including:
- Initialization and configuration
- Signal generation on MA crossovers
- Trade execution logic
"""

from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest


class TestSimpleMACrossoverStrategyInit:
    """Test SimpleMACrossoverStrategy initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy()

        assert strategy.fast_period == 10
        assert strategy.slow_period == 30
        assert strategy.min_history == 35  # slow_period + 5
        assert strategy.NAME == "SimpleMACrossover"

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy(parameters={"fast_period": 5, "slow_period": 20})

        assert strategy.fast_period == 5
        assert strategy.slow_period == 20
        assert strategy.min_history == 25  # 20 + 5

    def test_init_with_broker(self):
        """Test initialization with broker."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        assert strategy.broker == mock_broker

    def test_init_signals_dict(self):
        """Test that signals dict is initialized."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy()

        assert strategy.signals == {}
        assert strategy.previous_crossover == {}


class TestInitialize:
    """Test strategy initialization."""

    @pytest.mark.asyncio
    async def test_initialize_with_symbols(self):
        """Test initialization sets up symbol tracking."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy(parameters={"symbols": ["AAPL", "MSFT", "GOOGL"]})
        await strategy.initialize()

        assert strategy.signals["AAPL"] == "neutral"
        assert strategy.signals["MSFT"] == "neutral"
        assert strategy.signals["GOOGL"] == "neutral"
        assert strategy.previous_crossover["AAPL"] is None
        assert strategy.previous_crossover["MSFT"] is None
        assert strategy.previous_crossover["GOOGL"] is None

    @pytest.mark.asyncio
    async def test_initialize_empty_symbols(self):
        """Test initialization with no symbols."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy(parameters={})
        await strategy.initialize()

        assert strategy.signals == {}


class TestUpdateSignal:
    """Test signal update logic."""

    @pytest.mark.asyncio
    async def test_update_signal_no_current_data(self):
        """Test signal is neutral when no current data."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy()
        strategy.signals = {"AAPL": "buy"}

        await strategy._update_signal("AAPL")

        assert strategy.signals["AAPL"] == "neutral"

    @pytest.mark.asyncio
    async def test_update_signal_symbol_not_in_data(self):
        """Test signal is neutral when symbol not in current data."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy()
        strategy.signals = {"AAPL": "buy"}
        strategy.current_data = {"MSFT": pd.DataFrame()}

        await strategy._update_signal("AAPL")

        assert strategy.signals["AAPL"] == "neutral"

    @pytest.mark.asyncio
    async def test_update_signal_insufficient_history(self):
        """Test signal is neutral with insufficient price history."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy()  # min_history = 35
        strategy.signals = {"AAPL": "buy"}

        # Only 20 data points
        df = pd.DataFrame({"close": np.linspace(100, 110, 20)})
        strategy.current_data = {"AAPL": df}

        await strategy._update_signal("AAPL")

        assert strategy.signals["AAPL"] == "neutral"

    @pytest.mark.asyncio
    async def test_update_signal_bullish_crossover(self):
        """Test buy signal on bullish crossover."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy(parameters={"fast_period": 5, "slow_period": 10})
        strategy.signals = {"AAPL": "neutral"}
        strategy.previous_crossover = {"AAPL": "bearish"}

        # Create data where fast MA > slow MA (bullish)
        # Fast MA (last 5): average of [115, 116, 117, 118, 119] = 117
        # Slow MA (last 10): average of [105-114 + 115-119] = ~112
        closes = list(range(100, 120))  # 20 points: 100, 101, ..., 119
        df = pd.DataFrame({"close": closes})
        strategy.current_data = {"AAPL": df}

        await strategy._update_signal("AAPL")

        assert strategy.signals["AAPL"] == "buy"
        assert strategy.previous_crossover["AAPL"] == "bullish"

    @pytest.mark.asyncio
    async def test_update_signal_bearish_crossover(self):
        """Test sell signal on bearish crossover."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy(parameters={"fast_period": 5, "slow_period": 10})
        strategy.signals = {"AAPL": "neutral"}
        strategy.previous_crossover = {"AAPL": "bullish"}

        # Create data where fast MA < slow MA (bearish)
        # Descending prices
        closes = list(range(120, 100, -1))  # 20 points: 120, 119, ..., 101
        df = pd.DataFrame({"close": closes})
        strategy.current_data = {"AAPL": df}

        await strategy._update_signal("AAPL")

        assert strategy.signals["AAPL"] == "sell"
        assert strategy.previous_crossover["AAPL"] == "bearish"

    @pytest.mark.asyncio
    async def test_update_signal_no_crossover(self):
        """Test neutral signal when no crossover occurs."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy(parameters={"fast_period": 5, "slow_period": 10})
        strategy.signals = {"AAPL": "neutral"}
        strategy.previous_crossover = {"AAPL": "bullish"}

        # Still bullish (uptrend continues)
        closes = list(range(100, 120))
        df = pd.DataFrame({"close": closes})
        strategy.current_data = {"AAPL": df}

        await strategy._update_signal("AAPL")

        assert strategy.signals["AAPL"] == "neutral"  # No crossover
        assert strategy.previous_crossover["AAPL"] == "bullish"

    @pytest.mark.asyncio
    async def test_update_signal_handles_exception(self):
        """Test error handling in signal update."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy()
        strategy.signals = {"AAPL": "buy"}

        # Invalid data that will cause an exception
        strategy.current_data = {"AAPL": "not a dataframe"}

        await strategy._update_signal("AAPL")

        assert strategy.signals["AAPL"] == "neutral"


class TestGenerateSignals:
    """Test signal generation for all symbols."""

    @pytest.mark.asyncio
    async def test_generate_signals_all_symbols(self):
        """Test signals are generated for all symbols."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy(
            parameters={"symbols": ["AAPL", "MSFT"], "fast_period": 5, "slow_period": 10}
        )
        await strategy.initialize()

        # Set up data for both symbols
        closes = list(range(100, 120))
        df = pd.DataFrame({"close": closes})
        strategy.current_data = {"AAPL": df, "MSFT": df}
        strategy.previous_crossover = {"AAPL": "bearish", "MSFT": "bearish"}

        await strategy.generate_signals()

        # Both should have buy signals (bullish crossover from bearish)
        assert strategy.signals["AAPL"] == "buy"
        assert strategy.signals["MSFT"] == "buy"


class TestAnalyzeSymbol:
    """Test symbol analysis."""

    @pytest.mark.asyncio
    async def test_analyze_symbol_returns_signal(self):
        """Test analyze_symbol returns signal dict."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy()
        strategy.signals = {"AAPL": "buy"}

        result = await strategy.analyze_symbol("AAPL")

        assert result["action"] == "buy"
        assert result["symbol"] == "AAPL"
        assert result["strategy"] == "SimpleMACrossover"

    @pytest.mark.asyncio
    async def test_analyze_symbol_unknown_returns_neutral(self):
        """Test unknown symbol returns neutral."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        strategy = SimpleMACrossoverStrategy()
        strategy.signals = {}

        result = await strategy.analyze_symbol("UNKNOWN")

        assert result["action"] == "neutral"
