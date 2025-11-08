import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.backtest_engine import BacktestEngine
from engine.performance_metrics import PerformanceMetrics

@pytest.mark.asyncio
async def test_backtest_engine_initialization(momentum_strategy, test_symbols, test_period):
    """Test backtest engine initialization"""
    start_date, end_date = test_period
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        strategy=momentum_strategy,
        start_date=start_date,
        end_date=end_date,
        symbols=test_symbols
    )
    
    # Check initialization
    assert backtest_engine.strategy == momentum_strategy
    assert backtest_engine.start_date == start_date
    assert backtest_engine.end_date == end_date
    assert backtest_engine.symbols == test_symbols

@pytest.mark.asyncio
async def test_backtest_run(momentum_strategy, test_symbols, test_period, mock_broker):
    """Test running a backtest"""
    start_date, end_date = test_period
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        strategy=momentum_strategy,
        start_date=start_date,
        end_date=end_date,
        symbols=test_symbols
    )
    
    # Run backtest
    results = await backtest_engine.run()
    
    # Check results structure
    assert isinstance(results, dict)
    assert 'trades' in results
    assert 'equity_curve' in results
    assert 'stats' in results
    
    # Check equity curve
    equity_curve = results['equity_curve']
    assert isinstance(equity_curve, pd.DataFrame)
    assert not equity_curve.empty
    assert 'equity' in equity_curve.columns
    
    # Trades should be a list
    assert isinstance(results['trades'], list)
    
    # Stats should include basic metrics
    assert 'total_trades' in results['stats']
    assert 'win_rate' in results['stats']
    assert 'profit_factor' in results['stats']

@pytest.mark.asyncio
async def test_performance_metrics(momentum_strategy, test_symbols, test_period, mock_broker):
    """Test performance metrics calculation"""
    start_date, end_date = test_period
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        strategy=momentum_strategy,
        start_date=start_date,
        end_date=end_date,
        symbols=test_symbols
    )
    
    # Run backtest
    results = await backtest_engine.run()
    
    # Calculate performance metrics
    performance = PerformanceMetrics(results)
    metrics = performance.calculate_metrics()
    
    # Check metrics
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics
    assert 'average_win' in metrics
    assert 'average_loss' in metrics

@pytest.mark.asyncio
async def test_strategy_comparison(momentum_strategy, mean_reversion_strategy, test_symbols, test_period, mock_broker):
    """Test comparing multiple strategies"""
    start_date, end_date = test_period
    
    # Initialize backtest engines for both strategies
    momentum_backtest = BacktestEngine(
        strategy=momentum_strategy,
        start_date=start_date,
        end_date=end_date,
        symbols=test_symbols
    )
    
    mean_reversion_backtest = BacktestEngine(
        strategy=mean_reversion_strategy,
        start_date=start_date,
        end_date=end_date,
        symbols=test_symbols
    )
    
    # Run backtests
    momentum_results = await momentum_backtest.run()
    mean_reversion_results = await mean_reversion_backtest.run()
    
    # Calculate performance metrics
    momentum_performance = PerformanceMetrics(momentum_results)
    mean_reversion_performance = PerformanceMetrics(mean_reversion_results)
    
    momentum_metrics = momentum_performance.calculate_metrics()
    mean_reversion_metrics = mean_reversion_performance.calculate_metrics()
    
    # Check that we can compare metrics
    assert isinstance(momentum_metrics['sharpe_ratio'], (int, float))
    assert isinstance(mean_reversion_metrics['sharpe_ratio'], (int, float))
    
    # Both strategies should produce valid metrics (not testing which is better)
    assert momentum_metrics['total_return'] is not None
    assert mean_reversion_metrics['total_return'] is not None
