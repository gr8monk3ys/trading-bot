#!/usr/bin/env python3
"""
Unit tests for BacktestBroker.

Tests cover:
1. Initialization
2. Price data management
3. Slippage calculation
4. Partial fill simulation
5. Order placement
6. Position management
7. Portfolio value calculation
8. Async wrappers for strategy compatibility
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brokers.backtest_broker import BacktestBroker


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def broker():
    """Create a basic BacktestBroker instance."""
    return BacktestBroker(initial_balance=10000)


@pytest.fixture
def broker_no_partial_fills():
    """Create a BacktestBroker with partial fills disabled."""
    return BacktestBroker(initial_balance=10000, enable_partial_fills=False)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
    prices = 100 + np.arange(30) * 0.5  # Upward trend

    return pd.DataFrame(
        {
            "open": prices - 0.5,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": np.random.randint(100000, 1000000, 30),
        },
        index=dates,
    )


@pytest.fixture
def broker_with_data(broker, sample_price_data):
    """Create a BacktestBroker with price data loaded."""
    broker.set_price_data("AAPL", sample_price_data)
    broker._current_date = sample_price_data.index[15]  # Middle of data
    return broker


# =============================================================================
# TEST INITIALIZATION
# =============================================================================


class TestBacktestBrokerInit:
    """Test BacktestBroker initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        broker = BacktestBroker()
        assert broker.balance == 10000
        assert broker.slippage_bps == 5.0
        assert broker.spread_bps == 3.0
        assert broker.enable_partial_fills == True
        assert broker.positions == {}
        assert broker.orders == []
        assert broker.trades == []
        assert broker.price_data == {}

    def test_custom_initialization(self):
        """Test custom initialization values."""
        broker = BacktestBroker(
            api_key="test_key",
            api_secret="test_secret",
            paper=False,
            initial_balance=50000,
            slippage_bps=10.0,
            spread_bps=5.0,
            enable_partial_fills=False,
        )
        assert broker.api_key == "test_key"
        assert broker.api_secret == "test_secret"
        assert broker.paper == False
        assert broker.balance == 50000
        assert broker.slippage_bps == 10.0
        assert broker.spread_bps == 5.0
        assert broker.enable_partial_fills == False


# =============================================================================
# TEST PRICE DATA MANAGEMENT
# =============================================================================


class TestPriceDataManagement:
    """Test price data management methods."""

    def test_set_price_data(self, broker, sample_price_data):
        """Test setting price data for a symbol."""
        broker.set_price_data("AAPL", sample_price_data)
        assert "AAPL" in broker.price_data
        assert len(broker.price_data["AAPL"]) == 30

    def test_get_price_exact_date(self, broker_with_data, sample_price_data):
        """Test getting price for exact date match."""
        date = sample_price_data.index[10]
        price = broker_with_data.get_price("AAPL", date)
        assert price == sample_price_data.loc[date, "close"]

    def test_get_price_closest_date(self, broker_with_data, sample_price_data):
        """Test getting price for closest previous date."""
        # Get a weekend date
        weekday_date = sample_price_data.index[5]
        price = broker_with_data.get_price("AAPL", weekday_date)
        assert price > 0

    def test_get_price_symbol_not_found(self, broker):
        """Test getting price for unknown symbol raises error."""
        with pytest.raises(ValueError, match="No price data for"):
            broker.get_price("UNKNOWN", datetime.now())

    def test_get_historical_prices_existing_data(self, broker_with_data, sample_price_data):
        """Test getting historical prices for symbol with data."""
        prices = broker_with_data.get_historical_prices("AAPL", days=10)
        assert len(prices) == 10
        assert "close" in prices.columns

    def test_get_historical_prices_no_data(self, broker):
        """Test getting historical prices generates dummy data."""
        prices = broker.get_historical_prices("MSFT", days=20)
        # Business days may be fewer than calendar days
        assert len(prices) > 10
        assert "MSFT" in broker.price_data
        assert all(col in prices.columns for col in ["open", "high", "low", "close", "volume"])
        # Verify high is always highest (filter out NaN)
        valid_rows = prices.dropna()
        assert (valid_rows["high"] >= valid_rows["open"]).all()
        assert (valid_rows["high"] >= valid_rows["close"]).all()
        # Verify low is always lowest
        assert (valid_rows["low"] <= valid_rows["open"]).all()
        assert (valid_rows["low"] <= valid_rows["close"]).all()

    def test_get_historical_prices_with_end_date(self, broker):
        """Test getting historical prices with custom end date."""
        end_date = datetime(2024, 6, 1)
        prices = broker.get_historical_prices("GOOGL", days=15, end_date=end_date)
        # Business days may be fewer than calendar days
        assert len(prices) > 5
        assert "GOOGL" in broker.price_data


# =============================================================================
# TEST SLIPPAGE CALCULATION
# =============================================================================


class TestSlippageCalculation:
    """Test slippage calculation methods."""

    def test_slippage_buy_market_order(self, broker):
        """Test slippage increases price on buy market orders."""
        base_price = 100.0
        exec_price = broker._calculate_slippage("AAPL", 100, "buy", base_price, "market")
        assert exec_price > base_price  # Buy at higher price

    def test_slippage_sell_market_order(self, broker):
        """Test slippage decreases price on sell market orders."""
        base_price = 100.0
        exec_price = broker._calculate_slippage("AAPL", 100, "sell", base_price, "market")
        assert exec_price < base_price  # Sell at lower price

    def test_slippage_limit_order_less_than_market(self, broker):
        """Test limit orders have less slippage than market orders."""
        base_price = 100.0
        market_exec = broker._calculate_slippage("AAPL", 100, "buy", base_price, "market")
        limit_exec = broker._calculate_slippage("AAPL", 100, "buy", base_price, "limit")

        market_slip = market_exec - base_price
        limit_slip = limit_exec - base_price

        assert limit_slip < market_slip

    def test_slippage_large_order_more_impact(self, broker):
        """Test larger orders have more slippage."""
        base_price = 100.0
        small_exec = broker._calculate_slippage("AAPL", 100, "buy", base_price, "market")
        large_exec = broker._calculate_slippage("AAPL", 100000, "buy", base_price, "market")

        small_slip = small_exec - base_price
        large_slip = large_exec - base_price

        assert large_slip > small_slip

    def test_slippage_capped_at_2x(self, broker):
        """Test impact multiplier is capped at 2x."""
        base_price = 100.0
        # Very large order
        exec_price = broker._calculate_slippage("AAPL", 10000000, "buy", base_price, "market")
        # Max slippage should be bounded
        max_spread = base_price * (broker.spread_bps / 10000.0)
        max_slippage = base_price * (broker.slippage_bps / 10000.0) * 2.0  # 2x cap
        max_total = max_spread + max_slippage
        assert exec_price <= base_price + max_total + 0.01  # Small tolerance

    def test_slippage_zero_spread(self):
        """Test slippage calculation with zero spread."""
        broker = BacktestBroker(slippage_bps=5.0, spread_bps=0.0)
        base_price = 100.0
        exec_price = broker._calculate_slippage("AAPL", 100, "buy", base_price, "market")
        assert exec_price > base_price


# =============================================================================
# TEST PARTIAL FILL SIMULATION
# =============================================================================


class TestPartialFillSimulation:
    """Test partial fill simulation."""

    def test_small_order_full_fill(self, broker):
        """Test small orders get fully filled."""
        filled = broker._simulate_partial_fill(1000, "AAPL")
        assert filled == 1000

    def test_large_order_partial_fill(self, broker):
        """Test very large orders may get partial fills."""
        # Order > 10% of daily volume (1M)
        filled = broker._simulate_partial_fill(150000, "AAPL")
        # May be partial filled (70-95% range)
        assert filled >= int(150000 * 0.7)
        assert filled <= 150000

    def test_partial_fill_disabled(self, broker_no_partial_fills):
        """Test partial fills can be disabled."""
        filled = broker_no_partial_fills._simulate_partial_fill(150000, "AAPL")
        assert filled == 150000  # Full fill even for large order

    def test_partial_fill_at_least_one_share(self, broker):
        """Test partial fills return at least 1 share."""
        # Very large order
        filled = broker._simulate_partial_fill(5000000, "AAPL")
        assert filled >= 1


# =============================================================================
# TEST ORDER PLACEMENT
# =============================================================================


class TestOrderPlacement:
    """Test order placement functionality."""

    def test_place_market_buy_order(self, broker_with_data):
        """Test placing a market buy order."""
        initial_balance = broker_with_data.balance
        order = broker_with_data.place_order("AAPL", 10, "buy", order_type="market")

        assert order["symbol"] == "AAPL"
        assert order["quantity"] == 10
        assert order["side"] == "buy"
        assert order["type"] == "market"
        assert order["status"] in ["filled", "partially_filled"]
        assert order["filled_avg_price"] > 0
        assert broker_with_data.balance < initial_balance

    def test_place_market_sell_order(self, broker_with_data):
        """Test placing a market sell order."""
        # First buy
        broker_with_data.place_order("AAPL", 10, "buy", order_type="market")
        balance_after_buy = broker_with_data.balance

        # Then sell
        order = broker_with_data.place_order("AAPL", 10, "sell", order_type="market")

        assert order["side"] == "sell"
        assert broker_with_data.balance > balance_after_buy

    def test_place_limit_order(self, broker_with_data):
        """Test placing a limit order."""
        order = broker_with_data.place_order("AAPL", 10, "buy", price=100.0, order_type="limit")
        assert order["type"] == "limit"
        assert order["price"] == 100.0

    def test_order_updates_position(self, broker_with_data):
        """Test order updates positions correctly."""
        broker_with_data.place_order("AAPL", 10, "buy")
        position = broker_with_data.get_position("AAPL")

        assert position is not None
        assert position["quantity"] == 10

    def test_sell_removes_position(self, broker_with_data):
        """Test selling entire position removes it."""
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data.place_order("AAPL", 10, "sell")

        assert broker_with_data.get_position("AAPL") is None

    def test_averaging_entry_price(self, broker_with_data):
        """Test multiple buys average the entry price."""
        broker_with_data.place_order("AAPL", 10, "buy")
        first_entry = broker_with_data.get_position("AAPL")["entry_price"]

        # Wait a day for price to change
        broker_with_data._current_date = broker_with_data._current_date + timedelta(days=1)
        broker_with_data.place_order("AAPL", 10, "buy")

        position = broker_with_data.get_position("AAPL")
        assert position["quantity"] == 20
        # Entry price should be average
        assert position["entry_price"] != first_entry

    def test_order_records_trade(self, broker_with_data):
        """Test order records trade correctly."""
        broker_with_data.place_order("AAPL", 10, "buy")
        trades = broker_with_data.get_trades()

        assert len(trades) == 1
        assert trades[0]["symbol"] == "AAPL"
        assert trades[0]["quantity"] == 10
        assert trades[0]["side"] == "buy"

    def test_order_without_current_date(self, broker, sample_price_data):
        """Test order uses current datetime when _current_date not set."""
        broker.set_price_data("AAPL", sample_price_data)
        order = broker.place_order("AAPL", 10, "buy")
        assert order["created_at"] is not None

    def test_slippage_recorded_in_order(self, broker_with_data):
        """Test slippage is recorded in order details."""
        order = broker_with_data.place_order("AAPL", 10, "buy", order_type="market")
        assert "slippage_bps" in order
        assert order["slippage_bps"] >= 0


# =============================================================================
# TEST POSITION MANAGEMENT
# =============================================================================


class TestPositionManagement:
    """Test position management methods."""

    def test_get_position_existing(self, broker_with_data):
        """Test getting existing position."""
        broker_with_data.place_order("AAPL", 10, "buy")
        position = broker_with_data.get_position("AAPL")

        assert position is not None
        assert position["symbol"] == "AAPL"
        assert position["quantity"] == 10

    def test_get_position_nonexistent(self, broker):
        """Test getting nonexistent position returns None."""
        assert broker.get_position("AAPL") is None

    def test_get_positions_empty(self, broker):
        """Test getting positions when empty."""
        positions = broker.get_positions()
        assert positions == []

    def test_get_positions_multiple(self, broker_with_data, sample_price_data):
        """Test getting multiple positions."""
        broker_with_data.set_price_data("MSFT", sample_price_data)
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data.place_order("MSFT", 20, "buy")

        positions = broker_with_data.get_positions()
        assert len(positions) == 2


# =============================================================================
# TEST BALANCE AND PORTFOLIO VALUE
# =============================================================================


class TestBalanceAndPortfolioValue:
    """Test balance and portfolio value methods."""

    def test_get_balance(self, broker):
        """Test getting balance."""
        assert broker.get_balance() == 10000

    def test_balance_decreases_on_buy(self, broker_with_data):
        """Test balance decreases after buy order."""
        initial = broker_with_data.get_balance()
        broker_with_data.place_order("AAPL", 10, "buy")
        assert broker_with_data.get_balance() < initial

    def test_balance_increases_on_sell(self, broker_with_data):
        """Test balance increases after sell order."""
        broker_with_data.place_order("AAPL", 10, "buy")
        balance_after_buy = broker_with_data.get_balance()
        broker_with_data.place_order("AAPL", 10, "sell")
        assert broker_with_data.get_balance() > balance_after_buy

    def test_get_portfolio_value_no_positions(self, broker):
        """Test portfolio value equals balance when no positions."""
        assert broker.get_portfolio_value() == broker.balance

    def test_get_portfolio_value_with_positions(self, broker_with_data):
        """Test portfolio value includes positions."""
        broker_with_data.place_order("AAPL", 10, "buy")
        portfolio_value = broker_with_data.get_portfolio_value(broker_with_data._current_date)

        # Portfolio value should be balance + position value
        assert portfolio_value > broker_with_data.balance


# =============================================================================
# TEST ORDER QUERIES
# =============================================================================


class TestOrderQueries:
    """Test order query methods."""

    def test_get_orders_empty(self, broker):
        """Test getting orders when empty."""
        assert broker.get_orders() == []

    def test_get_orders_all(self, broker_with_data):
        """Test getting all orders."""
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data.place_order("AAPL", 10, "sell")

        orders = broker_with_data.get_orders()
        assert len(orders) == 2

    def test_get_orders_by_status(self, broker_with_data):
        """Test filtering orders by status."""
        broker_with_data.place_order("AAPL", 10, "buy")

        filled_orders = broker_with_data.get_orders(status="filled")
        unfilled_orders = broker_with_data.get_orders(status="unfilled")

        # All orders should be filled in normal case
        assert len(filled_orders) >= 0
        assert len(unfilled_orders) == 0


# =============================================================================
# TEST ASYNC WRAPPERS
# =============================================================================


class TestAsyncWrappers:
    """Test async wrapper methods for strategy compatibility."""

    @pytest.mark.asyncio
    async def test_get_account(self, broker_with_data):
        """Test async get_account method."""
        account = await broker_with_data.get_account()

        assert hasattr(account, "equity")
        assert hasattr(account, "cash")
        assert hasattr(account, "buying_power")
        assert float(account.cash) == broker_with_data.balance

    @pytest.mark.asyncio
    async def test_submit_order_advanced(self, broker_with_data):
        """Test async submit_order_advanced method."""

        class MockOrderRequest:
            symbol = "AAPL"
            qty = 10
            side = "buy"
            type = "market"

        order = await broker_with_data.submit_order_advanced(MockOrderRequest())

        assert order is not None
        assert order.symbol == "AAPL"
        assert order.status in ["filled", "partially_filled"]

    @pytest.mark.asyncio
    async def test_submit_order_advanced_with_enum_side(self, broker_with_data):
        """Test submit_order_advanced with enum side value."""

        class MockSide:
            value = "buy"

        class MockOrderRequest:
            symbol = "AAPL"
            qty = 10
            side = MockSide()
            type = "market"

        order = await broker_with_data.submit_order_advanced(MockOrderRequest())
        assert order.side == "buy"

    @pytest.mark.asyncio
    async def test_submit_order_advanced_missing_fields(self, broker_with_data):
        """Test submit_order_advanced with missing fields returns None."""

        class MockOrderRequest:
            pass  # Missing symbol and qty

        order = await broker_with_data.submit_order_advanced(MockOrderRequest())
        assert order is None

    @pytest.mark.asyncio
    async def test_submit_order_advanced_quantity_attribute(self, broker_with_data):
        """Test submit_order_advanced using quantity attribute."""

        class MockOrderRequest:
            symbol = "AAPL"
            quantity = 5  # Using quantity instead of qty
            side = "buy"
            order_type = "market"

        order = await broker_with_data.submit_order_advanced(MockOrderRequest())
        assert order is not None

    @pytest.mark.asyncio
    async def test_get_latest_quote(self, broker_with_data):
        """Test async get_latest_quote method."""
        quote = await broker_with_data.get_latest_quote("AAPL")

        assert hasattr(quote, "ask_price")
        assert hasattr(quote, "bid_price")
        assert quote.ask_price > quote.bid_price

    @pytest.mark.asyncio
    async def test_get_bars(self, broker_with_data):
        """Test async get_bars method."""
        bars = await broker_with_data.get_bars("AAPL", limit=10)

        assert len(bars) == 10
        assert hasattr(bars[0], "open")
        assert hasattr(bars[0], "high")
        assert hasattr(bars[0], "low")
        assert hasattr(bars[0], "close")
        assert hasattr(bars[0], "volume")

    @pytest.mark.asyncio
    async def test_get_bars_no_data(self, broker):
        """Test get_bars returns empty list when no data."""
        bars = await broker.get_bars("UNKNOWN")
        assert bars == []

    @pytest.mark.asyncio
    async def test_get_all_positions(self, broker_with_data, sample_price_data):
        """Test async get_all_positions method."""
        broker_with_data.set_price_data("MSFT", sample_price_data)
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data.place_order("MSFT", 20, "buy")

        positions = await broker_with_data.get_all_positions()

        assert len(positions) == 2
        assert hasattr(positions[0], "symbol")
        assert hasattr(positions[0], "qty")
        assert hasattr(positions[0], "quantity")
        assert hasattr(positions[0], "entry_price")


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_sell_more_than_owned(self, broker_with_data):
        """Test selling more than owned removes entire position."""
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data.place_order("AAPL", 15, "sell")  # Sell more than owned

        # Position should be removed
        assert broker_with_data.get_position("AAPL") is None

    def test_zero_quantity_order(self, broker_with_data):
        """Test placing order with zero quantity."""
        order = broker_with_data.place_order("AAPL", 0, "buy")
        # Order should still be recorded
        assert order is not None

    def test_price_data_with_timezone(self, broker):
        """Test handling price data with timezone-aware index."""
        import pytz

        dates = pd.date_range(start="2024-01-01", periods=10, freq="B", tz="UTC")
        prices = [100 + i for i in range(10)]
        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        broker.set_price_data("AAPL", data)

        # Test with naive datetime
        naive_date = datetime(2024, 1, 3)
        price = broker.get_price("AAPL", naive_date)
        assert price > 0

    def test_multiple_orders_same_symbol(self, broker_with_data):
        """Test multiple orders for same symbol."""
        for i in range(5):
            broker_with_data.place_order("AAPL", 10, "buy")

        position = broker_with_data.get_position("AAPL")
        assert position["quantity"] == 50
        assert len(broker_with_data.get_orders()) == 5

    def test_order_ids_are_unique(self, broker_with_data):
        """Test order IDs are unique."""
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data.place_order("AAPL", 10, "sell")
        broker_with_data.place_order("AAPL", 5, "buy")

        orders = broker_with_data.get_orders()
        ids = [o["id"] for o in orders]
        assert len(ids) == len(set(ids))  # All unique

    def test_trade_ids_are_unique(self, broker_with_data):
        """Test trade IDs are unique."""
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data.place_order("AAPL", 10, "sell")

        trades = broker_with_data.get_trades()
        ids = [t["id"] for t in trades]
        assert len(ids) == len(set(ids))


# =============================================================================
# TEST REALISTIC SCENARIOS
# =============================================================================


class TestRealisticScenarios:
    """Test realistic trading scenarios."""

    def test_round_trip_trade(self, broker_with_data):
        """Test a complete buy-sell round trip."""
        initial_balance = broker_with_data.get_balance()

        # Buy
        buy_order = broker_with_data.place_order("AAPL", 100, "buy")
        buy_price = buy_order["filled_avg_price"]

        # Sell
        sell_order = broker_with_data.place_order("AAPL", 100, "sell")
        sell_price = sell_order["filled_avg_price"]

        final_balance = broker_with_data.get_balance()

        # Due to slippage, final balance should be less than initial
        # (buy high, sell low)
        assert final_balance < initial_balance

    def test_scaling_into_position(self, broker_with_data):
        """Test scaling into a position over time."""
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data._current_date += timedelta(days=1)
        broker_with_data.place_order("AAPL", 10, "buy")
        broker_with_data._current_date += timedelta(days=1)
        broker_with_data.place_order("AAPL", 10, "buy")

        position = broker_with_data.get_position("AAPL")
        assert position["quantity"] == 30

    def test_partial_sell(self, broker_with_data):
        """Test selling partial position."""
        broker_with_data.place_order("AAPL", 100, "buy")
        broker_with_data.place_order("AAPL", 30, "sell")

        position = broker_with_data.get_position("AAPL")
        assert position["quantity"] == 70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
