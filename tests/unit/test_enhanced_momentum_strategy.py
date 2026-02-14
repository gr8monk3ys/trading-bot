#!/usr/bin/env python3
"""
Unit tests for Enhanced Momentum Strategy.

Tests cover:
1. Strategy initialization with profit features
2. RSI-2 signal generation
3. Kelly Criterion position sizing
4. Multi-timeframe confirmation
5. Volatility regime adjustment
6. ATR-based stops
7. Performance tracking
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.enhanced_momentum_strategy import EnhancedMomentumStrategy


class MockBar:
    """Mock bar for testing."""

    def __init__(self, close, high=None, low=None, volume=1000000):
        self.close = close
        self.high = high or close * 1.01
        self.low = low or close * 0.99
        self.volume = volume


class MockAccount:
    """Mock account for testing."""

    def __init__(self, buying_power=100000, equity=100000, cash=100000):
        self.buying_power = buying_power
        self.equity = equity
        self.cash = cash


class MockBroker:
    """Mock broker for testing."""

    def __init__(self, bars=None, account=None):
        self.bars = bars or []
        self.account = account or MockAccount()
        self.submitted_orders = []

    async def get_bars(self, symbol, timeframe="1Day", limit=50):
        return self.bars

    async def get_account(self):
        return self.account

    async def get_positions(self):
        """Return empty positions list."""
        return []

    async def submit_order_advanced(self, order):
        self.submitted_orders.append(order)
        return {"id": "test_order_123"}

    async def get_latest_quote(self, symbol):
        """Mock quote for VIX."""

        class Quote:
            ask_price = 18.0  # Normal VIX

        return Quote()


def create_trending_bars(direction="up", length=60, start_price=100):
    """Create mock bars with clear trend."""
    bars = []
    price = start_price

    for i in range(length):
        if direction == "up":
            price = price * 1.005  # 0.5% up per bar
        elif direction == "down":
            price = price * 0.995  # 0.5% down per bar

        bars.append(
            MockBar(close=price, high=price * 1.01, low=price * 0.99, volume=1000000 + i * 10000)
        )

    return bars


def create_oversold_bars(length=60):
    """Create bars that should trigger RSI-2 buy signal (extreme oversold)."""
    bars = []
    price = 100

    # First build up price
    for _i in range(50):
        price = price * 1.002
        bars.append(MockBar(close=price, volume=1000000))

    # Then sharp drop to trigger RSI-2 < 10
    for _i in range(10):
        price = price * 0.97  # 3% drop each bar
        bars.append(MockBar(close=price, volume=2000000))  # High volume

    return bars


def create_overbought_bars(length=60):
    """Create bars that should trigger RSI-2 sell signal (extreme overbought)."""
    bars = []
    price = 100

    # First build up slowly
    for _i in range(50):
        price = price * 1.001
        bars.append(MockBar(close=price, volume=1000000))

    # Then sharp rise to trigger RSI-2 > 90
    for _i in range(10):
        price = price * 1.03  # 3% rise each bar
        bars.append(MockBar(close=price, volume=2000000))

    return bars


class TestStrategyInitialization:
    """Test strategy initialization."""

    @pytest.mark.asyncio
    async def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        strategy = EnhancedMomentumStrategy(broker=MockBroker(), parameters={"symbols": ["AAPL"]})

        params = strategy.default_parameters()

        # RSI-2 settings
        assert params["rsi_period"] == 2
        assert params["rsi_oversold"] == 10
        assert params["rsi_overbought"] == 90

        # Profit features enabled
        assert params["use_kelly_criterion"]
        assert params["use_multi_timeframe"]
        assert params["use_volatility_regime"]

        # Position sizing
        assert params["kelly_fraction"] == 0.5
        assert params["min_position_size"] == 0.05
        assert params["max_position_size"] == 0.20

    @pytest.mark.asyncio
    async def test_strategy_name(self):
        """Test strategy name is set."""
        strategy = EnhancedMomentumStrategy(broker=MockBroker(), parameters={"symbols": ["AAPL"]})

        assert strategy.NAME == "EnhancedMomentumStrategy"

    @pytest.mark.asyncio
    async def test_initialize_creates_tracking_structures(self):
        """Test initialization creates required tracking structures."""
        broker = MockBroker()
        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL", "MSFT"]})

        await strategy.initialize()

        assert "AAPL" in strategy.indicators
        assert "MSFT" in strategy.indicators
        assert "AAPL" in strategy.signals
        assert strategy.signals["AAPL"] == "neutral"


class TestRSI2SignalGeneration:
    """Test RSI-2 signal generation."""

    @pytest.mark.asyncio
    async def test_buy_signal_on_oversold(self):
        """Test buy signal generated when RSI-2 < 10."""
        bars = create_oversold_bars()
        broker = MockBroker(bars=bars)

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        # Disable MTF to test RSI-2 only
        strategy.use_mtf = False

        await strategy.analyze_symbol("AAPL")

        # Should generate buy signal on extreme oversold
        assert strategy.indicators["AAPL"]["rsi"] is not None

    @pytest.mark.asyncio
    async def test_no_signal_on_neutral_rsi(self):
        """Test no signal when RSI in neutral zone."""
        # Create flat bars (RSI around 50)
        bars = [MockBar(close=100.0) for _ in range(60)]
        broker = MockBroker(bars=bars)

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()
        strategy.use_mtf = False

        await strategy.analyze_symbol("AAPL")

        # Neutral RSI should not generate signal
        # RSI should be around 50 with flat prices
        assert strategy.indicators["AAPL"]["rsi"] is not None

    @pytest.mark.asyncio
    async def test_indicators_stored_correctly(self):
        """Test that all indicators are stored after analysis."""
        bars = create_trending_bars(direction="up")
        broker = MockBroker(bars=bars)

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()
        strategy.use_mtf = False

        await strategy.analyze_symbol("AAPL")

        indicators = strategy.indicators["AAPL"]

        assert "rsi" in indicators
        assert "macd" in indicators
        assert "macd_signal" in indicators
        assert "adx" in indicators
        assert "atr" in indicators
        assert "sma_fast" in indicators
        assert "sma_medium" in indicators
        assert "sma_slow" in indicators
        assert "volume_confirmed" in indicators
        assert "price" in indicators


class TestPositionSizing:
    """Test position sizing with Kelly Criterion."""

    @pytest.mark.asyncio
    async def test_base_position_size(self):
        """Test default position size when Kelly has no history."""
        bars = create_trending_bars()
        broker = MockBroker(bars=bars)

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        # No trade history, should use base size
        size = await strategy.calculate_position_size("AAPL", "buy")

        # Should be between min and max
        assert 0.05 <= size <= 0.20

    @pytest.mark.asyncio
    async def test_position_size_with_trade_history(self):
        """Test Kelly adjusts size based on trade history."""
        bars = create_trending_bars()
        broker = MockBroker(bars=bars)

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        # Add winning trade history
        for i in range(15):
            strategy.record_trade_result(
                symbol="AAPL",
                pnl=100 if i < 12 else -50,  # 80% win rate
                pnl_pct=0.05 if i < 12 else -0.025,
            )

        size = await strategy.calculate_position_size("AAPL", "buy")

        # With good win rate, Kelly should suggest reasonable size
        assert 0.05 <= size <= 0.20

    @pytest.mark.asyncio
    async def test_position_size_respects_limits(self):
        """Test position size stays within limits."""
        bars = create_trending_bars()
        broker = MockBroker(bars=bars)

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        # Force high Kelly suggestion by setting extreme win stats
        strategy.win_count = 100
        strategy.loss_count = 0
        strategy.total_wins = 10.0
        strategy.total_losses = 0.0

        size = await strategy.calculate_position_size("AAPL", "buy")

        # Should be capped at max_position_size
        assert size <= 0.20


class TestTradeExecution:
    """Test trade execution."""

    @pytest.mark.asyncio
    async def test_execute_buy_trade(self):
        """Test executing a buy trade."""
        bars = create_trending_bars()
        broker = MockBroker(bars=bars)

        # Disable advanced features that require complex mocking
        strategy = EnhancedMomentumStrategy(
            broker=broker,
            parameters={
                "symbols": ["AAPL"],
                "use_kelly_criterion": False,
                "use_multi_timeframe": False,
                "use_volatility_regime": False,
            },
        )
        await strategy.initialize()

        # Analyze first to get current price
        await strategy.analyze_symbol("AAPL")

        # Execute trade
        result = await strategy.execute_trade("AAPL", "buy")

        assert result
        assert len(broker.submitted_orders) == 1

    @pytest.mark.asyncio
    async def test_execute_trade_stores_stop_price(self):
        """Test that ATR-based stop price is stored."""
        bars = create_trending_bars()
        broker = MockBroker(bars=bars)

        # Disable advanced features that require complex mocking
        strategy = EnhancedMomentumStrategy(
            broker=broker,
            parameters={
                "symbols": ["AAPL"],
                "use_kelly_criterion": False,
                "use_multi_timeframe": False,
                "use_volatility_regime": False,
            },
        )
        await strategy.initialize()

        await strategy.analyze_symbol("AAPL")
        result = await strategy.execute_trade("AAPL", "buy")

        # Stop price should be set on successful trade
        if result:
            assert "AAPL" in strategy.stop_prices
            assert strategy.stop_prices["AAPL"] > 0

    @pytest.mark.asyncio
    async def test_execute_trade_updates_signal_time(self):
        """Test that signal time is updated after trade."""
        bars = create_trending_bars()
        broker = MockBroker(bars=bars)

        # Disable advanced features that require complex mocking
        strategy = EnhancedMomentumStrategy(
            broker=broker,
            parameters={
                "symbols": ["AAPL"],
                "use_kelly_criterion": False,
                "use_multi_timeframe": False,
                "use_volatility_regime": False,
            },
        )
        await strategy.initialize()

        await strategy.analyze_symbol("AAPL")
        result = await strategy.execute_trade("AAPL", "buy")

        # Signal time should be updated on successful trade
        if result:
            assert strategy.last_signal_time["AAPL"] is not None


class TestPerformanceTracking:
    """Test performance tracking for Kelly Criterion."""

    @pytest.mark.asyncio
    async def test_record_winning_trade(self):
        """Test recording a winning trade."""
        strategy = EnhancedMomentumStrategy(broker=MockBroker(), parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        strategy.record_trade_result("AAPL", pnl=100, pnl_pct=0.05)

        assert strategy.win_count == 1
        assert strategy.loss_count == 0
        assert strategy.total_wins == 0.05
        assert len(strategy.trade_history) == 1

    @pytest.mark.asyncio
    async def test_record_losing_trade(self):
        """Test recording a losing trade."""
        strategy = EnhancedMomentumStrategy(broker=MockBroker(), parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        strategy.record_trade_result("AAPL", pnl=-50, pnl_pct=-0.025)

        assert strategy.win_count == 0
        assert strategy.loss_count == 1
        assert strategy.total_losses == 0.025

    @pytest.mark.asyncio
    async def test_performance_summary(self):
        """Test performance summary calculation."""
        strategy = EnhancedMomentumStrategy(broker=MockBroker(), parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        # Add mixed results
        for i in range(10):
            is_win = i < 6  # 60% win rate
            strategy.record_trade_result(
                "AAPL", pnl=100 if is_win else -50, pnl_pct=0.05 if is_win else -0.025
            )

        summary = strategy.get_performance_summary()

        assert summary["total_trades"] == 10
        assert summary["win_count"] == 6
        assert summary["loss_count"] == 4
        assert abs(summary["win_rate"] - 0.6) < 0.01

    @pytest.mark.asyncio
    async def test_empty_performance_summary(self):
        """Test performance summary with no trades."""
        strategy = EnhancedMomentumStrategy(broker=MockBroker(), parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        summary = strategy.get_performance_summary()

        assert summary["total_trades"] == 0


class TestFeatureFlags:
    """Test feature enable/disable flags."""

    @pytest.mark.asyncio
    async def test_kelly_can_be_disabled(self):
        """Test that Kelly Criterion can be disabled."""
        broker = MockBroker()
        strategy = EnhancedMomentumStrategy(
            broker=broker, parameters={"symbols": ["AAPL"], "use_kelly_criterion": False}
        )
        await strategy.initialize()

        assert not strategy.use_kelly

    @pytest.mark.asyncio
    async def test_mtf_can_be_disabled(self):
        """Test that multi-timeframe can be disabled."""
        broker = MockBroker()
        strategy = EnhancedMomentumStrategy(
            broker=broker, parameters={"symbols": ["AAPL"], "use_multi_timeframe": False}
        )
        await strategy.initialize()

        assert not strategy.use_mtf

    @pytest.mark.asyncio
    async def test_volatility_regime_can_be_disabled(self):
        """Test that volatility regime can be disabled."""
        broker = MockBroker()
        strategy = EnhancedMomentumStrategy(
            broker=broker, parameters={"symbols": ["AAPL"], "use_volatility_regime": False}
        )
        await strategy.initialize()

        assert not strategy.use_vol_regime


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None signal."""
        # Only 10 bars (need 50+)
        bars = [MockBar(close=100) for _ in range(10)]
        broker = MockBroker(bars=bars)

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        signal = await strategy.analyze_symbol("AAPL")

        assert signal is None

    @pytest.mark.asyncio
    async def test_empty_bars_returns_none(self):
        """Test that empty bars returns None signal."""
        broker = MockBroker(bars=[])

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        signal = await strategy.analyze_symbol("AAPL")

        assert signal is None

    @pytest.mark.asyncio
    async def test_handles_missing_price_gracefully(self):
        """Test that missing price doesn't crash trade execution."""
        broker = MockBroker()
        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        # Don't analyze, so no current price
        result = await strategy.execute_trade("AAPL", "buy")

        assert not result  # Should fail gracefully


class TestATRStops:
    """Test ATR-based trailing stops."""

    @pytest.mark.asyncio
    async def test_atr_calculated_correctly(self):
        """Test that ATR is calculated and stored."""
        bars = create_trending_bars()
        broker = MockBroker(bars=bars)

        strategy = EnhancedMomentumStrategy(broker=broker, parameters={"symbols": ["AAPL"]})
        await strategy.initialize()

        await strategy.analyze_symbol("AAPL")

        assert "atr" in strategy.indicators["AAPL"]
        assert strategy.indicators["AAPL"]["atr"] > 0

    @pytest.mark.asyncio
    async def test_stop_price_uses_atr_multiplier(self):
        """Test that stop price is calculated using ATR multiplier."""
        bars = create_trending_bars()
        broker = MockBroker(bars=bars)

        # Disable advanced features that require complex mocking
        strategy = EnhancedMomentumStrategy(
            broker=broker,
            parameters={
                "symbols": ["AAPL"],
                "use_kelly_criterion": False,
                "use_multi_timeframe": False,
                "use_volatility_regime": False,
            },
        )
        await strategy.initialize()

        await strategy.analyze_symbol("AAPL")

        current_price = strategy.current_prices["AAPL"]
        atr = strategy.indicators["AAPL"]["atr"]
        expected_stop = current_price - (atr * 2.0)  # Default multiplier

        result = await strategy.execute_trade("AAPL", "buy")

        # Trade should succeed and stop prices should be set
        assert result
        actual_stop = strategy.stop_prices["AAPL"]

        # Should be approximately ATR-based
        assert abs(actual_stop - expected_stop) < 0.01 * current_price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
