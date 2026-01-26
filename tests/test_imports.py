#!/usr/bin/env python3
"""
Import Tests for Trading Bot

This test suite verifies that all critical components can be imported
without errors. These tests are designed to catch import issues early
before running more complex integration tests.

Run with: pytest tests/test_imports.py -v
"""

import os
import sys

import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBrokerImports:
    """Test broker-related imports"""

    def test_alpaca_broker_import(self):
        """Test that AlpacaBroker can be imported"""
        from brokers.alpaca_broker import AlpacaBroker

        assert AlpacaBroker is not None
        assert callable(AlpacaBroker)

    def test_order_builder_import(self):
        """Test that OrderBuilder can be imported"""
        from brokers.order_builder import OrderBuilder

        assert OrderBuilder is not None
        assert callable(OrderBuilder)

    def test_order_builder_convenience_functions(self):
        """Test that convenience functions can be imported"""
        from brokers.order_builder import bracket_order, limit_order, market_order

        assert market_order is not None
        assert limit_order is not None
        assert bracket_order is not None
        assert callable(market_order)
        assert callable(limit_order)
        assert callable(bracket_order)

    def test_backtest_broker_import(self):
        """Test that BacktestBroker can be imported"""
        from brokers.backtest_broker import BacktestBroker

        assert BacktestBroker is not None
        assert callable(BacktestBroker)


class TestStrategyImports:
    """Test strategy-related imports"""

    def test_base_strategy_import(self):
        """Test that BaseStrategy can be imported"""
        from strategies.base_strategy import BaseStrategy

        assert BaseStrategy is not None
        assert callable(BaseStrategy)

    def test_bracket_momentum_strategy_import(self):
        """Test that BracketMomentumStrategy can be imported"""
        from strategies.bracket_momentum_strategy import BracketMomentumStrategy

        assert BracketMomentumStrategy is not None
        assert callable(BracketMomentumStrategy)

    def test_momentum_strategy_import(self):
        """Test that MomentumStrategy can be imported"""
        from strategies.momentum_strategy import MomentumStrategy

        assert MomentumStrategy is not None
        assert callable(MomentumStrategy)

    def test_mean_reversion_strategy_import(self):
        """Test that MeanReversionStrategy can be imported"""
        from strategies.mean_reversion_strategy import MeanReversionStrategy

        assert MeanReversionStrategy is not None
        assert callable(MeanReversionStrategy)

    def test_risk_manager_import(self):
        """Test that RiskManager can be imported"""
        from strategies.risk_manager import RiskManager

        assert RiskManager is not None
        assert callable(RiskManager)


class TestConfigImports:
    """Test configuration imports"""

    def test_config_import(self):
        """Test that config module can be imported"""
        import config

        assert config is not None

    def test_config_constants(self):
        """Test that config constants are defined"""
        from config import ALPACA_CREDS, SYMBOLS, TRADING_PARAMS

        assert SYMBOLS is not None
        assert isinstance(SYMBOLS, list)
        assert len(SYMBOLS) > 0
        assert ALPACA_CREDS is not None
        assert isinstance(ALPACA_CREDS, dict)
        assert TRADING_PARAMS is not None
        assert isinstance(TRADING_PARAMS, dict)


class TestDependencyImports:
    """Test that critical external dependencies can be imported"""

    def test_alpaca_trade_api_import(self):
        """Test that alpaca-py can be imported"""
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        assert TradingClient is not None
        assert MarketOrderRequest is not None
        assert OrderSide is not None
        assert TimeInForce is not None

    def test_pandas_import(self):
        """Test that pandas can be imported"""
        import pandas as pd

        assert pd is not None

    def test_numpy_import(self):
        """Test that numpy can be imported"""
        import numpy as np

        assert np is not None

    def test_talib_import(self):
        """Test that TA-Lib can be imported"""
        try:
            import talib

            assert talib is not None
        except ImportError:
            pytest.skip("TA-Lib not installed - optional dependency")

    def test_dotenv_import(self):
        """Test that python-dotenv can be imported"""
        from dotenv import load_dotenv

        assert load_dotenv is not None

    def test_asyncio_import(self):
        """Test that asyncio can be imported"""
        import asyncio

        assert asyncio is not None


class TestOrderBuilderInstantiation:
    """Test that OrderBuilder can be instantiated and used"""

    def test_create_order_builder(self):
        """Test that OrderBuilder can be instantiated"""
        from brokers.order_builder import OrderBuilder

        builder = OrderBuilder("AAPL", "buy", 1)
        assert builder is not None
        assert builder.symbol == "AAPL"
        assert builder.qty == 1.0

    def test_build_market_order(self):
        """Test that a market order can be built"""
        from brokers.order_builder import OrderBuilder

        order = OrderBuilder("AAPL", "buy", 1).market().day().build()
        assert order is not None
        assert hasattr(order, "symbol")
        assert hasattr(order, "qty")
        assert hasattr(order, "side")
        assert hasattr(order, "time_in_force")

    def test_build_limit_order(self):
        """Test that a limit order can be built"""
        from brokers.order_builder import OrderBuilder

        order = OrderBuilder("AAPL", "buy", 1).limit(150.00).gtc().build()
        assert order is not None
        assert hasattr(order, "limit_price")
        assert order.limit_price == 150.00

    def test_build_bracket_order(self):
        """Test that a bracket order can be built"""
        from brokers.order_builder import OrderBuilder

        order = (
            OrderBuilder("AAPL", "buy", 1)
            .market()
            .bracket(take_profit=200.00, stop_loss=140.00)
            .gtc()
            .build()
        )
        assert order is not None
        assert hasattr(order, "order_class")
        assert hasattr(order, "take_profit")
        assert hasattr(order, "stop_loss")


class TestEnumImports:
    """Test that Alpaca enums can be imported"""

    def test_order_side_enum(self):
        """Test that OrderSide enum can be imported"""
        from alpaca.trading.enums import OrderSide

        assert OrderSide.BUY is not None
        assert OrderSide.SELL is not None

    def test_time_in_force_enum(self):
        """Test that TimeInForce enum can be imported"""
        from alpaca.trading.enums import TimeInForce

        assert TimeInForce.DAY is not None
        assert TimeInForce.GTC is not None
        assert TimeInForce.IOC is not None
        assert TimeInForce.FOK is not None

    def test_order_class_enum(self):
        """Test that OrderClass enum can be imported"""
        from alpaca.trading.enums import OrderClass

        assert OrderClass.SIMPLE is not None
        assert OrderClass.BRACKET is not None
        assert OrderClass.OCO is not None
        assert OrderClass.OTO is not None

    def test_order_type_enum(self):
        """Test that OrderType enum can be imported"""
        from alpaca.trading.enums import OrderType

        assert OrderType.MARKET is not None
        assert OrderType.LIMIT is not None
        assert OrderType.STOP is not None
        assert OrderType.STOP_LIMIT is not None
        assert OrderType.TRAILING_STOP is not None


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
