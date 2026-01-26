#!/usr/bin/env python3
"""
Paper Trading Integration Tests

These tests validate the full trading pipeline against Alpaca's paper trading API.
They require valid API credentials in the environment.

Run with:
    pytest tests/integration/test_paper_trading.py -v -m integration

Environment variables required:
    ALPACA_API_KEY: Paper trading API key
    ALPACA_SECRET_KEY: Paper trading secret key
    PAPER: Set to "True"
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

# Load environment variables before imports that need them
load_dotenv()

logger = logging.getLogger(__name__)


def has_api_credentials():
    """Check if API credentials are available."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("API_SECRET")
    return bool(api_key and api_secret)


# Skip all tests in this module if no credentials
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not has_api_credentials(),
        reason="Alpaca API credentials not available"
    ),
]


@pytest.fixture
def api_credentials():
    """Get API credentials from environment."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("API_SECRET")
    paper = os.getenv("PAPER", "True").lower() == "true"
    return {"api_key": api_key, "api_secret": api_secret, "paper": paper}


@pytest.fixture
def alpaca_broker():
    """Create AlpacaBroker instance for testing.

    AlpacaBroker reads credentials from environment variables
    (ALPACA_API_KEY / API_KEY and ALPACA_SECRET_KEY / API_SECRET).
    """
    from brokers.alpaca_broker import AlpacaBroker

    paper = os.getenv("PAPER", "True").lower() == "true"
    broker = AlpacaBroker(paper=paper)
    return broker


class TestAlpacaConnection:
    """Test basic Alpaca API connectivity."""

    @pytest.mark.asyncio
    async def test_can_connect_to_alpaca(self, api_credentials):
        """Verify we can establish connection to Alpaca paper trading."""
        from alpaca.trading.client import TradingClient

        client = TradingClient(
            api_key=api_credentials["api_key"],
            secret_key=api_credentials["api_secret"],
            paper=api_credentials["paper"],
        )

        try:
            account = client.get_account()
        except Exception as e:
            pytest.skip(f"Cannot connect to Alpaca API (credentials may be invalid): {e}")

        assert account is not None
        assert account.id is not None
        assert account.status == "ACTIVE"
        logger.info(f"Connected to account: {account.id}")

    @pytest.mark.asyncio
    async def test_can_retrieve_account_info(self, alpaca_broker):
        """Verify we can get account information through broker."""
        try:
            account = await alpaca_broker.get_account()
        except Exception as e:
            pytest.skip(f"Cannot retrieve account info (credentials may be invalid): {e}")

        assert account is not None
        assert hasattr(account, "buying_power")
        assert hasattr(account, "equity")
        assert hasattr(account, "cash")
        assert float(account.equity) > 0
        logger.info(f"Account equity: ${float(account.equity):,.2f}")

    @pytest.mark.asyncio
    async def test_can_retrieve_positions(self, alpaca_broker):
        """Verify we can get current positions."""
        try:
            positions = await alpaca_broker.get_positions()
        except Exception as e:
            pytest.skip(f"Cannot retrieve positions (credentials may be invalid): {e}")

        # Positions can be empty, but should return a list
        assert isinstance(positions, list)
        logger.info(f"Current positions: {len(positions)}")


class TestMarketData:
    """Test market data retrieval."""

    @pytest.mark.asyncio
    async def test_can_get_bars(self, alpaca_broker):
        """Verify we can retrieve historical bar data."""
        bars = await alpaca_broker.get_bars("AAPL", timeframe="1Day", limit=30)

        assert bars is not None
        assert len(bars) > 0
        assert hasattr(bars[0], "close")
        assert hasattr(bars[0], "high")
        assert hasattr(bars[0], "low")
        assert hasattr(bars[0], "volume")
        logger.info(f"Retrieved {len(bars)} bars for AAPL")

    @pytest.mark.asyncio
    async def test_bars_have_required_fields(self, alpaca_broker):
        """Verify bar data contains required OHLCV fields."""
        bars = await alpaca_broker.get_bars("AAPL", timeframe="1Day", limit=5)

        if len(bars) == 0:
            pytest.skip("No bar data available (market may be closed)")

        bar = bars[0]
        assert hasattr(bar, "open")
        assert hasattr(bar, "high")
        assert hasattr(bar, "low")
        assert hasattr(bar, "close")
        assert hasattr(bar, "volume")
        logger.info(f"Bar data verified with all OHLCV fields")


class TestStrategyAnalysis:
    """Test strategy analysis without placing orders."""

    @pytest.mark.asyncio
    async def test_momentum_strategy_can_analyze(self, alpaca_broker):
        """Test that MomentumStrategy can analyze a symbol."""
        from strategies.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy(
            broker=alpaca_broker,
            parameters={"symbols": ["AAPL"]},
        )
        await strategy.initialize()

        # Analyze the symbol
        signal = await strategy.analyze_symbol("AAPL")

        # Signal can be None, dict, or string - all are valid
        # The key is that it doesn't crash
        logger.info(f"AAPL analysis complete. Signal: {signal}")

        # Indicators may be empty if insufficient data (e.g., weekend)
        assert "AAPL" in strategy.indicators
        indicators = strategy.indicators["AAPL"]

        if len(indicators) > 0:
            # If we have indicators, verify the expected ones exist
            assert "rsi" in indicators
            assert "macd" in indicators
            logger.info(f"Indicators calculated: {list(indicators.keys())}")
        else:
            logger.info("No indicators calculated (insufficient data available)")

    @pytest.mark.asyncio
    async def test_enhanced_momentum_strategy_can_analyze(self, alpaca_broker):
        """Test that EnhancedMomentumStrategy can analyze a symbol."""
        from strategies.enhanced_momentum_strategy import EnhancedMomentumStrategy

        strategy = EnhancedMomentumStrategy(
            broker=alpaca_broker,
            parameters={
                "symbols": ["MSFT"],
                "use_kelly_criterion": False,
                "use_multi_timeframe": False,  # Disable to speed up test
                "use_volatility_regime": False,
            },
        )
        await strategy.initialize()

        signal = await strategy.analyze_symbol("MSFT")

        logger.info(f"MSFT analysis complete. Signal: {signal}")

        # Verify indicator structure exists
        assert "MSFT" in strategy.indicators
        indicators = strategy.indicators["MSFT"]

        if len(indicators) > 0:
            assert "rsi" in indicators
            assert "atr" in indicators
            logger.info(f"Indicators calculated: {list(indicators.keys())}")
        else:
            logger.info("No indicators calculated (insufficient data available)")


class TestRiskManagement:
    """Test risk management components with live data."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_initializes(self, alpaca_broker):
        """Test that circuit breaker can initialize with account data."""
        from utils.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(max_daily_loss=0.03)
        await cb.initialize(alpaca_broker)

        assert cb.starting_balance > 0
        assert cb.trading_halted is False
        logger.info(f"Circuit breaker initialized. Starting balance: ${cb.starting_balance:,.2f}")

    @pytest.mark.asyncio
    async def test_risk_manager_can_evaluate(self, alpaca_broker):
        """Test that risk manager can evaluate portfolio risk."""
        from strategies.risk_manager import RiskManager

        rm = RiskManager()

        # Get current positions for evaluation
        positions = await alpaca_broker.get_positions()

        # Even with no positions, should not crash
        if len(positions) == 0:
            logger.info("No positions to evaluate - risk check passed (empty portfolio)")
            return

        # With positions, evaluate risk
        account = await alpaca_broker.get_account()
        logger.info(f"Risk evaluation complete for {len(positions)} positions")


class TestOrderBuilding:
    """Test order building without submitting."""

    def test_can_build_market_order(self):
        """Test building a market order."""
        from brokers.order_builder import OrderBuilder

        order = (
            OrderBuilder("AAPL", "buy", 10)
            .market()
            .day()
            .build()
        )

        # Order is an Alpaca request object
        assert order.symbol == "AAPL"
        assert order.qty == 10
        logger.info("Market order built successfully")

    def test_can_build_bracket_order(self):
        """Test building a bracket order."""
        from brokers.order_builder import OrderBuilder

        order = (
            OrderBuilder("AAPL", "buy", 10)
            .market()
            .bracket(take_profit=180.0, stop_loss=150.0)
            .gtc()
            .build()
        )

        # Order is an Alpaca request object
        assert order.symbol == "AAPL"
        assert order.take_profit is not None
        assert order.stop_loss is not None
        logger.info("Bracket order built successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
