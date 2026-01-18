#!/usr/bin/env python3
"""
Comprehensive unit tests for OrderBuilder module.

Tests cover:
1. Constructor validation (symbol, side, quantity)
2. Order types (market, limit, stop, stop-limit, trailing stop)
3. Time in force options (DAY, GTC, IOC, FOK, OPG, CLS)
4. Advanced order classes (bracket, OCO, OTO)
5. Additional options (extended hours, client order ID)
6. Build method and validation
7. Convenience functions
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brokers.order_builder import (
    OrderBuilder,
    market_order,
    limit_order,
    bracket_order
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType


class TestOrderBuilderConstructor:
    """Test OrderBuilder constructor and validation."""

    def test_valid_construction(self):
        """Test valid order builder creation."""
        builder = OrderBuilder('AAPL', 'buy', 100)
        assert builder.symbol == 'AAPL'
        assert builder.side == OrderSide.BUY
        assert builder.qty == 100.0

    def test_symbol_uppercase_conversion(self):
        """Test that symbol is converted to uppercase."""
        builder = OrderBuilder('aapl', 'buy', 100)
        assert builder.symbol == 'AAPL'

    def test_symbol_whitespace_strip(self):
        """Test that symbol whitespace is stripped."""
        builder = OrderBuilder('  AAPL  ', 'buy', 100)
        assert builder.symbol == 'AAPL'

    def test_sell_side(self):
        """Test sell side is correctly parsed."""
        builder = OrderBuilder('AAPL', 'sell', 100)
        assert builder.side == OrderSide.SELL

    def test_fractional_quantity(self):
        """Test fractional shares are accepted."""
        builder = OrderBuilder('AAPL', 'buy', 0.5)
        assert builder.qty == 0.5

    def test_invalid_empty_symbol(self):
        """Test that empty symbol raises error."""
        with pytest.raises(ValueError, match="non-empty string"):
            OrderBuilder('', 'buy', 100)

    def test_invalid_none_symbol(self):
        """Test that None symbol raises error."""
        with pytest.raises(ValueError, match="non-empty string"):
            OrderBuilder(None, 'buy', 100)

    def test_invalid_symbol_too_long(self):
        """Test that symbol longer than 5 chars raises error."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            OrderBuilder('TOOLONG', 'buy', 100)

    def test_invalid_symbol_with_numbers(self):
        """Test that symbol with numbers raises error."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            OrderBuilder('AAP1', 'buy', 100)

    def test_invalid_side(self):
        """Test that invalid side raises error."""
        with pytest.raises(ValueError, match="must be 'buy' or 'sell'"):
            OrderBuilder('AAPL', 'hold', 100)

    def test_invalid_none_side(self):
        """Test that None side raises error."""
        with pytest.raises(ValueError, match="must be 'buy' or 'sell'"):
            OrderBuilder('AAPL', None, 100)

    def test_invalid_negative_quantity(self):
        """Test that negative quantity raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            OrderBuilder('AAPL', 'buy', -100)

    def test_invalid_zero_quantity(self):
        """Test that zero quantity raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            OrderBuilder('AAPL', 'buy', 0)

    def test_invalid_quantity_exceeds_max(self):
        """Test that quantity exceeding max raises error."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            OrderBuilder('AAPL', 'buy', 2_000_000)

    def test_invalid_non_numeric_quantity(self):
        """Test that non-numeric quantity raises error."""
        with pytest.raises(ValueError, match="must be numeric"):
            OrderBuilder('AAPL', 'buy', 'one hundred')


class TestOrderTypes:
    """Test order type methods."""

    def test_market_order(self):
        """Test market order type."""
        builder = OrderBuilder('AAPL', 'buy', 100).market()
        assert builder._order_type == OrderType.MARKET

    def test_limit_order(self):
        """Test limit order type."""
        builder = OrderBuilder('AAPL', 'buy', 100).limit(150.00)
        assert builder._order_type == OrderType.LIMIT
        assert builder._limit_price == 150.00

    def test_stop_order(self):
        """Test stop order type."""
        builder = OrderBuilder('AAPL', 'sell', 100).stop(140.00)
        assert builder._order_type == OrderType.STOP
        assert builder._stop_price == 140.00

    def test_stop_limit_order(self):
        """Test stop-limit order type."""
        builder = OrderBuilder('AAPL', 'sell', 100).stop_limit(140.00, 139.50)
        assert builder._order_type == OrderType.STOP_LIMIT
        assert builder._stop_price == 140.00
        assert builder._limit_price == 139.50

    def test_trailing_stop_with_price(self):
        """Test trailing stop with dollar amount."""
        builder = OrderBuilder('AAPL', 'sell', 100).trailing_stop(trail_price=5.00)
        assert builder._order_type == OrderType.TRAILING_STOP
        assert builder._trail_price == 5.00
        assert builder._trail_percent is None

    def test_trailing_stop_with_percent(self):
        """Test trailing stop with percentage."""
        builder = OrderBuilder('AAPL', 'sell', 100).trailing_stop(trail_percent=2.5)
        assert builder._order_type == OrderType.TRAILING_STOP
        assert builder._trail_percent == 2.5
        assert builder._trail_price is None

    def test_trailing_stop_both_raises_error(self):
        """Test that providing both trail options raises error."""
        with pytest.raises(ValueError, match="not both"):
            OrderBuilder('AAPL', 'sell', 100).trailing_stop(trail_price=5.00, trail_percent=2.5)

    def test_trailing_stop_neither_raises_error(self):
        """Test that providing neither trail option raises error."""
        with pytest.raises(ValueError, match="Must provide"):
            OrderBuilder('AAPL', 'sell', 100).trailing_stop()


class TestTimeInForce:
    """Test time in force methods."""

    def test_day_tif(self):
        """Test DAY time in force."""
        builder = OrderBuilder('AAPL', 'buy', 100).market().day()
        assert builder._time_in_force == TimeInForce.DAY

    def test_gtc_tif(self):
        """Test GTC time in force."""
        builder = OrderBuilder('AAPL', 'buy', 100).market().gtc()
        assert builder._time_in_force == TimeInForce.GTC

    def test_ioc_tif(self):
        """Test IOC time in force."""
        builder = OrderBuilder('AAPL', 'buy', 100).market().ioc()
        assert builder._time_in_force == TimeInForce.IOC

    def test_fok_tif(self):
        """Test FOK time in force."""
        builder = OrderBuilder('AAPL', 'buy', 100).market().fok()
        assert builder._time_in_force == TimeInForce.FOK

    def test_opg_tif(self):
        """Test OPG time in force."""
        builder = OrderBuilder('AAPL', 'buy', 100).market().opg()
        assert builder._time_in_force == TimeInForce.OPG

    def test_cls_tif(self):
        """Test CLS time in force."""
        builder = OrderBuilder('AAPL', 'buy', 100).market().cls()
        assert builder._time_in_force == TimeInForce.CLS

    def test_default_tif_is_day(self):
        """Test default time in force is DAY."""
        builder = OrderBuilder('AAPL', 'buy', 100)
        assert builder._time_in_force == TimeInForce.DAY


class TestAdvancedOrderClasses:
    """Test bracket, OCO, and OTO order classes."""

    def test_bracket_order(self):
        """Test bracket order configuration."""
        builder = (OrderBuilder('AAPL', 'buy', 100)
                  .market()
                  .bracket(take_profit=160.00, stop_loss=140.00))

        assert builder._order_class == OrderClass.BRACKET
        assert builder._take_profit == {'limit_price': 160.00}
        assert builder._stop_loss == {'stop_price': 140.00}

    def test_bracket_order_with_stop_limit(self):
        """Test bracket order with stop-limit."""
        builder = (OrderBuilder('AAPL', 'buy', 100)
                  .market()
                  .bracket(take_profit=160.00, stop_loss=140.00, stop_limit=139.50))

        assert builder._stop_loss == {'stop_price': 140.00, 'limit_price': 139.50}

    def test_bracket_changes_invalid_tif_to_gtc(self):
        """Test that bracket order changes invalid TIF to GTC."""
        builder = (OrderBuilder('AAPL', 'buy', 100)
                  .market()
                  .ioc()  # Invalid for bracket
                  .bracket(take_profit=160.00, stop_loss=140.00))

        assert builder._time_in_force == TimeInForce.GTC

    def test_oco_order(self):
        """Test OCO order configuration."""
        builder = (OrderBuilder('AAPL', 'sell', 100)
                  .limit(155.00)
                  .oco(take_profit=160.00, stop_loss=140.00))

        assert builder._order_class == OrderClass.OCO
        assert builder._take_profit == {'limit_price': 160.00}
        assert builder._stop_loss == {'stop_price': 140.00}

    def test_oco_sets_limit_type(self):
        """Test that OCO sets order type to LIMIT if not already."""
        builder = (OrderBuilder('AAPL', 'sell', 100)
                  .market()  # Wrong type
                  .oco(take_profit=160.00, stop_loss=140.00))

        assert builder._order_type == OrderType.LIMIT

    def test_oto_with_take_profit(self):
        """Test OTO order with take profit."""
        builder = (OrderBuilder('AAPL', 'buy', 100)
                  .market()
                  .oto(take_profit=160.00))

        assert builder._order_class == OrderClass.OTO
        assert builder._take_profit == {'limit_price': 160.00}
        assert builder._stop_loss is None

    def test_oto_with_stop_loss(self):
        """Test OTO order with stop loss."""
        builder = (OrderBuilder('AAPL', 'buy', 100)
                  .market()
                  .oto(stop_loss=140.00))

        assert builder._order_class == OrderClass.OTO
        assert builder._stop_loss == {'stop_price': 140.00}
        assert builder._take_profit is None

    def test_oto_with_stop_limit(self):
        """Test OTO order with stop limit."""
        builder = (OrderBuilder('AAPL', 'buy', 100)
                  .market()
                  .oto(stop_loss=140.00, stop_limit=139.50))

        assert builder._stop_loss == {'stop_price': 140.00, 'limit_price': 139.50}

    def test_oto_both_raises_error(self):
        """Test that OTO with both take_profit and stop_loss raises error."""
        with pytest.raises(ValueError, match="not both"):
            (OrderBuilder('AAPL', 'buy', 100)
             .market()
             .oto(take_profit=160.00, stop_loss=140.00))

    def test_oto_neither_raises_error(self):
        """Test that OTO with neither option raises error."""
        with pytest.raises(ValueError, match="requires either"):
            OrderBuilder('AAPL', 'buy', 100).market().oto()


class TestAdditionalOptions:
    """Test additional order options."""

    def test_extended_hours_enabled(self):
        """Test extended hours can be enabled."""
        builder = OrderBuilder('AAPL', 'buy', 100).limit(150.00).day().extended_hours()
        assert builder._extended_hours is True

    def test_extended_hours_disabled(self):
        """Test extended hours can be disabled."""
        builder = OrderBuilder('AAPL', 'buy', 100).limit(150.00).extended_hours(False)
        assert builder._extended_hours is False

    def test_client_order_id(self):
        """Test client order ID can be set."""
        builder = OrderBuilder('AAPL', 'buy', 100).market().client_order_id('my-order-123')
        assert builder._client_order_id == 'my-order-123'

    def test_client_order_id_too_long(self):
        """Test that client order ID longer than 48 chars raises error."""
        with pytest.raises(ValueError, match="48 characters"):
            OrderBuilder('AAPL', 'buy', 100).market().client_order_id('x' * 50)


class TestBuildMethod:
    """Test the build method."""

    def test_build_market_order(self):
        """Test building a market order."""
        order = OrderBuilder('AAPL', 'buy', 100).market().day().build()

        assert order.symbol == 'AAPL'
        assert order.qty == 100
        assert order.side == OrderSide.BUY
        assert order.time_in_force == TimeInForce.DAY

    def test_build_limit_order(self):
        """Test building a limit order."""
        order = OrderBuilder('AAPL', 'buy', 100).limit(150.00).gtc().build()

        assert order.symbol == 'AAPL'
        assert order.limit_price == 150.00
        assert order.time_in_force == TimeInForce.GTC

    def test_build_stop_order(self):
        """Test building a stop order."""
        order = OrderBuilder('AAPL', 'sell', 100).stop(140.00).day().build()

        assert order.stop_price == 140.00

    def test_build_stop_limit_order(self):
        """Test building a stop-limit order."""
        order = OrderBuilder('AAPL', 'sell', 100).stop_limit(140.00, 139.50).day().build()

        assert order.stop_price == 140.00
        assert order.limit_price == 139.50

    def test_build_trailing_stop_price(self):
        """Test building a trailing stop with price."""
        order = OrderBuilder('AAPL', 'sell', 100).trailing_stop(trail_price=5.00).gtc().build()

        assert order.trail_price == 5.00

    def test_build_trailing_stop_percent(self):
        """Test building a trailing stop with percent."""
        order = OrderBuilder('AAPL', 'sell', 100).trailing_stop(trail_percent=2.5).gtc().build()

        assert order.trail_percent == 2.5

    def test_build_bracket_order(self):
        """Test building a bracket order."""
        order = (OrderBuilder('AAPL', 'buy', 100)
                .market()
                .bracket(take_profit=160.00, stop_loss=140.00)
                .gtc()
                .build())

        assert order.order_class == OrderClass.BRACKET
        assert order.take_profit.limit_price == 160.00
        assert order.stop_loss.stop_price == 140.00

    def test_build_with_client_order_id(self):
        """Test building order with client order ID."""
        order = (OrderBuilder('AAPL', 'buy', 100)
                .market()
                .client_order_id('test-123')
                .build())

        assert order.client_order_id == 'test-123'

    def test_build_with_extended_hours(self):
        """Test building order with extended hours."""
        order = (OrderBuilder('AAPL', 'buy', 100)
                .limit(150.00)
                .day()
                .extended_hours()
                .build())

        assert order.extended_hours is True

    def test_build_without_order_type_raises_error(self):
        """Test that building without order type raises error."""
        with pytest.raises(ValueError, match="Order type not set"):
            OrderBuilder('AAPL', 'buy', 100).build()

    def test_build_extended_hours_requires_limit(self):
        """Test that extended hours with non-limit raises error."""
        with pytest.raises(ValueError, match="LIMIT order type"):
            (OrderBuilder('AAPL', 'buy', 100)
             .market()
             .day()
             .extended_hours()
             .build())

    def test_build_extended_hours_requires_day(self):
        """Test that extended hours with non-DAY TIF raises error."""
        with pytest.raises(ValueError, match="DAY time in force"):
            (OrderBuilder('AAPL', 'buy', 100)
             .limit(150.00)
             .gtc()
             .extended_hours()
             .build())


class TestPriceValidation:
    """Test price validation helper."""

    def test_price_validation_rounds_high_price(self):
        """Test that prices >= $1 are rounded to 2 decimals."""
        builder = OrderBuilder('AAPL', 'buy', 100)
        validated = builder._validate_price(150.12345)
        assert validated == 150.12

    def test_price_validation_rounds_low_price(self):
        """Test that prices < $1 are rounded to 4 decimals."""
        builder = OrderBuilder('AAPL', 'buy', 100)
        validated = builder._validate_price(0.123456)
        assert validated == 0.1235

    def test_price_validation_negative_raises_error(self):
        """Test that negative price raises error."""
        builder = OrderBuilder('AAPL', 'buy', 100)
        with pytest.raises(ValueError, match="must be positive"):
            builder._validate_price(-5.00)

    def test_price_validation_zero_raises_error(self):
        """Test that zero price raises error."""
        builder = OrderBuilder('AAPL', 'buy', 100)
        with pytest.raises(ValueError, match="must be positive"):
            builder._validate_price(0)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_market_order_basic(self):
        """Test basic market order creation."""
        order = market_order('AAPL', 'buy', 100)

        assert order.symbol == 'AAPL'
        assert order.qty == 100
        assert order.side == OrderSide.BUY

    def test_market_order_with_gtc(self):
        """Test market order with GTC."""
        order = market_order('AAPL', 'buy', 100, gtc=True)

        assert order.time_in_force == TimeInForce.GTC

    def test_limit_order_basic(self):
        """Test basic limit order creation."""
        order = limit_order('AAPL', 'buy', 100, 150.00)

        assert order.symbol == 'AAPL'
        assert order.limit_price == 150.00

    def test_limit_order_with_gtc(self):
        """Test limit order with GTC."""
        order = limit_order('AAPL', 'buy', 100, 150.00, gtc=True)

        assert order.time_in_force == TimeInForce.GTC

    def test_limit_order_with_extended_hours(self):
        """Test limit order with extended hours."""
        order = limit_order('AAPL', 'buy', 100, 150.00, extended_hours=True)

        assert order.extended_hours is True
        assert order.time_in_force == TimeInForce.DAY

    def test_bracket_order_market_entry(self):
        """Test bracket order with market entry."""
        order = bracket_order('AAPL', 'buy', 100, take_profit=160.00, stop_loss=140.00)

        assert order.symbol == 'AAPL'
        assert order.order_class == OrderClass.BRACKET
        assert order.take_profit.limit_price == 160.00
        assert order.stop_loss.stop_price == 140.00

    def test_bracket_order_limit_entry(self):
        """Test bracket order with limit entry."""
        order = bracket_order('AAPL', 'buy', 100, entry_price=150.00,
                             take_profit=160.00, stop_loss=140.00)

        assert order.limit_price == 150.00

    def test_bracket_order_with_stop_limit(self):
        """Test bracket order with stop limit."""
        order = bracket_order('AAPL', 'buy', 100, take_profit=160.00,
                             stop_loss=140.00, stop_limit=139.50)

        assert order.stop_loss.stop_price == 140.00
        assert order.stop_loss.limit_price == 139.50


class TestFluentInterface:
    """Test that methods can be chained fluently."""

    def test_method_chaining(self):
        """Test that all methods return self for chaining."""
        order = (OrderBuilder('AAPL', 'buy', 100)
                .market()
                .day()
                .client_order_id('test')
                .build())

        assert order is not None

    def test_complex_chain(self):
        """Test complex method chain."""
        order = (OrderBuilder('AAPL', 'buy', 100)
                .limit(150.00)
                .gtc()
                .bracket(take_profit=160.00, stop_loss=140.00, stop_limit=139.50)
                .client_order_id('complex-order')
                .build())

        assert order.symbol == 'AAPL'
        assert order.limit_price == 150.00
        assert order.order_class == OrderClass.BRACKET


class TestReprMethod:
    """Test the __repr__ method."""

    def test_repr_output(self):
        """Test repr returns useful string."""
        builder = OrderBuilder('AAPL', 'buy', 100).market().gtc()
        repr_str = repr(builder)

        assert 'AAPL' in repr_str
        assert 'buy' in repr_str.lower()
        assert '100' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
