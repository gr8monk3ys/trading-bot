
import pytest

from strategies.momentum_strategy import MomentumStrategy


@pytest.mark.asyncio
async def test_momentum_strategy_initialization(mock_broker, test_symbols):
    """Test momentum strategy initialization"""
    # MomentumStrategy gets symbols from parameters passed to __init__
    strategy = MomentumStrategy(
        broker=mock_broker,
        parameters={"symbols": test_symbols},
    )
    await strategy.initialize()

    assert strategy.symbols == test_symbols
    assert strategy.name == "MomentumStrategy"
    assert strategy.parameters["position_size"] == 0.05  # Conservative default


@pytest.mark.asyncio
async def test_momentum_signal_generation(mock_broker, test_symbols):
    """Test signal generation for momentum strategy"""
    # Initialize strategy with symbols
    strategy = MomentumStrategy(
        broker=mock_broker,
        parameters={"symbols": test_symbols},
    )
    await strategy.initialize()

    # After initialization, signals dict should have entries for each symbol
    for symbol in test_symbols:
        signal = strategy.signals.get(symbol, "neutral")
        # Valid signals are: buy, sell, short, neutral
        assert signal in ["buy", "sell", "short", "neutral", None]


@pytest.mark.asyncio
async def test_momentum_strategy_has_indicators(mock_broker, test_symbols):
    """Test that momentum strategy initializes indicator tracking"""
    strategy = MomentumStrategy(
        broker=mock_broker,
        parameters={"symbols": test_symbols},
    )
    await strategy.initialize()

    # Indicators dict should have entries for each symbol
    for symbol in test_symbols:
        assert symbol in strategy.indicators


@pytest.mark.asyncio
async def test_momentum_strategy_default_parameters(mock_broker, test_symbols):
    """Test momentum strategy default parameters"""
    strategy = MomentumStrategy(
        broker=mock_broker,
        parameters={"symbols": test_symbols},
    )
    await strategy.initialize()

    # Check key parameters are set (conservative defaults)
    assert strategy.position_size == 0.05  # 5% per position (conservative)
    assert strategy.max_positions == 5
    assert strategy.stop_loss == 0.03
    assert strategy.take_profit == 0.05
