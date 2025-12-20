import pandas as pd
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


@pytest.mark.asyncio
async def test_backtest_run(momentum_strategy, test_period, mock_broker):
    """Test running a backtest using the run() method"""
    start_date, end_date = test_period

    # Initialize backtest engine with broker
    backtest_engine = BacktestEngine(broker=mock_broker)

    # run() takes strategies list, start_date, end_date
    results = await backtest_engine.run(
        strategies=[momentum_strategy],
        start_date=start_date,
        end_date=end_date,
    )

    # run() returns a list of DataFrames, one per strategy
    assert isinstance(results, list)
    assert len(results) == 1

    # Each result is a DataFrame with equity, cash, holdings, returns, trades columns
    result_df = results[0]
    assert isinstance(result_df, pd.DataFrame)
    assert "equity" in result_df.columns
    assert "returns" in result_df.columns
    assert "trades" in result_df.columns


@pytest.mark.asyncio
async def test_strategy_comparison(
    momentum_strategy, mean_reversion_strategy, test_period, mock_broker
):
    """Test comparing multiple strategies via run()"""
    start_date, end_date = test_period

    # Initialize backtest engine
    backtest_engine = BacktestEngine(broker=mock_broker)

    # Run both strategies together
    results = await backtest_engine.run(
        strategies=[momentum_strategy, mean_reversion_strategy],
        start_date=start_date,
        end_date=end_date,
    )

    # Should get one result DataFrame per strategy
    assert isinstance(results, list)
    assert len(results) == 2

    # Both results should be DataFrames
    for result_df in results:
        assert isinstance(result_df, pd.DataFrame)
        assert "equity" in result_df.columns
