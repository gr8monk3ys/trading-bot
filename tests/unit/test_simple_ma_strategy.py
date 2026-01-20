"""
Unit tests for SimpleMACrossoverStrategy.

Tests the simple moving average crossover strategy including:
- Initialization and configuration
- Signal generation on MA crossovers
- Trade execution logic
"""

from unittest.mock import AsyncMock, MagicMock, patch

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

        strategy = SimpleMACrossoverStrategy(
            parameters={"fast_period": 5, "slow_period": 20}
        )

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

        strategy = SimpleMACrossoverStrategy(
            parameters={"symbols": ["AAPL", "MSFT", "GOOGL"]}
        )
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

        strategy = SimpleMACrossoverStrategy(
            parameters={"fast_period": 5, "slow_period": 10}
        )
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

        strategy = SimpleMACrossoverStrategy(
            parameters={"fast_period": 5, "slow_period": 10}
        )
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

        strategy = SimpleMACrossoverStrategy(
            parameters={"fast_period": 5, "slow_period": 10}
        )
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


class TestExecuteTrade:
    """Test trade execution."""

    @pytest.mark.asyncio
    async def test_execute_trade_neutral_does_nothing(self):
        """Test neutral signal doesn't execute trade."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy.execute_trade("AAPL", {"action": "neutral"})

        mock_broker.get_account.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_trade_buy_opens_position(self):
        """Test buy signal opens new position."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        mock_broker.get_all_positions.return_value = []
        mock_account = MagicMock()
        mock_account.cash = "100000"
        mock_broker.get_account.return_value = mock_account
        mock_quote = MagicMock()
        mock_quote.ask_price = "150.00"
        mock_broker.get_latest_quote.return_value = mock_quote

        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy.execute_trade("AAPL", {"action": "buy"})

        mock_broker.submit_order_advanced.assert_called_once()
        order = mock_broker.submit_order_advanced.call_args[0][0]
        assert order.symbol == "AAPL"
        assert order.side == "buy"
        # 20% of 100000 / 150 = 133 shares
        assert order.qty == 133

    @pytest.mark.asyncio
    async def test_execute_trade_buy_no_duplicate_position(self):
        """Test buy signal doesn't open if position exists."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        mock_position.quantity = 0

        mock_broker = AsyncMock()
        mock_broker.get_all_positions.return_value = [mock_position]
        mock_account = MagicMock()
        mock_account.cash = "100000"
        mock_broker.get_account.return_value = mock_account
        mock_quote = MagicMock()
        mock_quote.ask_price = "150.00"
        mock_broker.get_latest_quote.return_value = mock_quote

        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy.execute_trade("AAPL", {"action": "buy"})

        mock_broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_trade_sell_closes_position(self):
        """Test sell signal closes existing position."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "100"
        # Set quantity to 0 so the code uses qty instead
        mock_position.quantity = 0

        mock_broker = AsyncMock()
        mock_broker.get_all_positions.return_value = [mock_position]
        mock_account = MagicMock()
        mock_account.cash = "100000"
        mock_broker.get_account.return_value = mock_account
        mock_quote = MagicMock()
        mock_quote.ask_price = "150.00"
        mock_broker.get_latest_quote.return_value = mock_quote

        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy.execute_trade("AAPL", {"action": "sell"})

        mock_broker.submit_order_advanced.assert_called_once()
        order = mock_broker.submit_order_advanced.call_args[0][0]
        assert order.symbol == "AAPL"
        assert order.side == "sell"
        assert order.qty == 100

    @pytest.mark.asyncio
    async def test_execute_trade_sell_no_position(self):
        """Test sell signal does nothing without position."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        mock_broker.get_all_positions.return_value = []
        mock_account = MagicMock()
        mock_account.cash = "100000"
        mock_broker.get_account.return_value = mock_account
        mock_quote = MagicMock()
        mock_quote.ask_price = "150.00"
        mock_broker.get_latest_quote.return_value = mock_quote

        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy.execute_trade("AAPL", {"action": "sell"})

        mock_broker.submit_order_advanced.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_trade_with_dict_position(self):
        """Test trade execution with dict-style position."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        mock_broker.get_all_positions.return_value = [
            {"symbol": "AAPL", "quantity": 50}
        ]
        mock_account = MagicMock()
        mock_account.cash = "100000"
        mock_broker.get_account.return_value = mock_account
        mock_quote = MagicMock()
        mock_quote.ask_price = "150.00"
        mock_broker.get_latest_quote.return_value = mock_quote

        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy.execute_trade("AAPL", {"action": "sell"})

        mock_broker.submit_order_advanced.assert_called_once()
        order = mock_broker.submit_order_advanced.call_args[0][0]
        assert order.qty == 50

    @pytest.mark.asyncio
    async def test_execute_trade_uses_get_positions_fallback(self):
        """Test fallback to get_positions if get_all_positions not available."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = MagicMock()
        # Remove get_all_positions
        del mock_broker.get_all_positions
        mock_broker.get_positions.return_value = []
        mock_account = MagicMock()
        mock_account.cash = "100000"
        mock_broker.get_account = AsyncMock(return_value=mock_account)
        mock_quote = MagicMock()
        mock_quote.ask_price = "150.00"
        mock_broker.get_latest_quote = AsyncMock(return_value=mock_quote)
        mock_broker.submit_order_advanced = AsyncMock()

        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy.execute_trade("AAPL", {"action": "buy"})

        mock_broker.get_positions.assert_called_once()
        mock_broker.submit_order_advanced.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_trade_handles_exception(self):
        """Test error handling in trade execution."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        mock_broker.get_all_positions.side_effect = Exception("API Error")

        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        # Should not raise
        await strategy.execute_trade("AAPL", {"action": "buy"})


class TestPlaceOrder:
    """Test order placement."""

    @pytest.mark.asyncio
    async def test_place_order_buy(self):
        """Test placing a buy order."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy._place_order("AAPL", 100, "buy")

        mock_broker.submit_order_advanced.assert_called_once()
        order = mock_broker.submit_order_advanced.call_args[0][0]
        assert order.symbol == "AAPL"
        assert order.qty == 100
        assert order.side == "buy"
        assert order.type == "market"

    @pytest.mark.asyncio
    async def test_place_order_sell(self):
        """Test placing a sell order."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        await strategy._place_order("MSFT", 50, "sell")

        mock_broker.submit_order_advanced.assert_called_once()
        order = mock_broker.submit_order_advanced.call_args[0][0]
        assert order.symbol == "MSFT"
        assert order.qty == 50
        assert order.side == "sell"

    @pytest.mark.asyncio
    async def test_place_order_handles_exception(self):
        """Test error handling in order placement."""
        from strategies.simple_ma_strategy import SimpleMACrossoverStrategy

        mock_broker = AsyncMock()
        mock_broker.submit_order_advanced.side_effect = Exception("Order Error")

        strategy = SimpleMACrossoverStrategy(broker=mock_broker)

        # Should not raise
        await strategy._place_order("AAPL", 100, "buy")
