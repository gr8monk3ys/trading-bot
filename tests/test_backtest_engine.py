import pytest

from engine.backtest_engine import BacktestEngine


@pytest.mark.asyncio
async def test_backtest_engine_initialization(mock_broker):
    """Test backtest engine initialization"""
    # BacktestEngine only takes an optional broker argument
    backtest_engine = BacktestEngine(broker=mock_broker)

    # Check initialization
    assert backtest_engine.broker == mock_broker
    assert backtest_engine.strategies == []
    assert backtest_engine.results == {}
    assert backtest_engine.current_date is None


@pytest.mark.asyncio
async def test_backtest_engine_default_initialization():
    """Test backtest engine initialization with no arguments"""
    backtest_engine = BacktestEngine()

    assert backtest_engine.broker is None
    assert backtest_engine.strategies == []
    assert backtest_engine.results == {}
