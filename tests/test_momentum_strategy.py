import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.momentum_strategy import MomentumStrategy
from engine.backtest_engine import BacktestEngine

@pytest.mark.asyncio
async def test_momentum_strategy_initialization(momentum_strategy, test_symbols):
    """Test momentum strategy initialization"""
    await momentum_strategy.initialize(test_symbols)
    assert momentum_strategy.symbols == test_symbols
    assert momentum_strategy.name == "MomentumStrategy"
    assert momentum_strategy.parameters['position_size'] == 0.1

@pytest.mark.asyncio
async def test_momentum_signal_generation(momentum_strategy, test_symbols, mock_broker):
    """Test signal generation for momentum strategy"""
    # Initialize strategy
    await momentum_strategy.initialize(test_symbols)
    
    # Get mock price data for analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    price_data = await mock_broker.get_bars(
        test_symbols, 
        'day', 
        start_date, 
        end_date
    )
    
    # Process price data for each symbol
    for symbol in test_symbols:
        # Convert to DataFrame for processing
        bars = price_data[symbol]
        df = pd.DataFrame([{
            'date': bar['t'],
            'open': bar['o'],
            'high': bar['h'],
            'low': bar['l'],
            'close': bar['c'],
            'volume': bar['v']
        } for bar in bars])
        
        # Check signal generation
        signal, signal_data = await momentum_strategy.generate_signal(symbol, df)
        
        # Verify signal is valid
        assert signal in [-1, 0, 1]
        
        # Check signal metadata
        assert 'rsi' in signal_data
        assert 'macd' in signal_data
        assert 'macd_signal' in signal_data
        assert 'histogram' in signal_data

@pytest.mark.asyncio
async def test_position_sizing(momentum_strategy, test_symbols, mock_broker):
    """Test position sizing logic"""
    # Initialize strategy
    await momentum_strategy.initialize(test_symbols)
    
    # Set account value
    mock_broker.account["portfolio_value"] = 100000.0
    
    # Get position size for a symbol with signal 1 (buy)
    position_size = await momentum_strategy.calculate_position_size("AAPL", 1)
    
    # Test position sizing logic
    expected_size = round(mock_broker.account["portfolio_value"] * momentum_strategy.parameters['position_size'])
    assert position_size > 0
    assert abs(position_size - expected_size) < 0.01 * expected_size  # Allow 1% margin of error

@pytest.mark.asyncio
async def test_risk_management(momentum_strategy, test_symbols, mock_broker):
    """Test risk management logic"""
    # Initialize strategy
    await momentum_strategy.initialize(test_symbols)
    
    # Create mock position
    symbol = test_symbols[0]
    entry_price = 100.0
    position_size = 10
    
    # Test stop loss calculation
    stop_price = momentum_strategy.calculate_stop_loss(symbol, entry_price, "long")
    expected_stop = entry_price * (1 - momentum_strategy.parameters['stop_loss'])
    assert abs(stop_price - expected_stop) < 0.001
    
    # Test take profit calculation
    take_profit_price = momentum_strategy.calculate_take_profit(symbol, entry_price, "long")
    expected_tp = entry_price * (1 + momentum_strategy.parameters['take_profit'])
    assert abs(take_profit_price - expected_tp) < 0.001
