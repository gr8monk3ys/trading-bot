#!/usr/bin/env python3
"""
Test Advanced Order Types with Alpaca Paper Trading

This script demonstrates all the new order types including:
- Market orders
- Limit orders
- Stop orders
- Stop-limit orders
- Trailing stop orders
- Bracket orders (entry + take-profit + stop-loss)
- OCO orders (One-Cancels-Other)
- OTO orders (One-Triggers-Other)
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brokers.alpaca_broker import AlpacaBroker
from brokers.order_builder import OrderBuilder, market_order, limit_order, bracket_order

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_account_info(broker):
    """Test getting account information."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Account Information")
    logger.info("="*80)

    try:
        account = await broker.get_account()
        logger.info(f"✅ Account ID: {account.id}")
        logger.info(f"✅ Status: {account.status}")
        logger.info(f"✅ Cash: ${float(account.cash):,.2f}")
        logger.info(f"✅ Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"✅ Portfolio Value: ${float(account.portfolio_value):,.2f}")
        return True
    except Exception as e:
        logger.error(f"❌ Error getting account info: {e}")
        return False


async def test_market_data(broker, symbol='AAPL'):
    """Test getting market data."""
    logger.info("\n" + "="*80)
    logger.info(f"TEST: Market Data for {symbol}")
    logger.info("="*80)

    try:
        # Get latest price
        price = await broker.get_last_price(symbol)
        logger.info(f"✅ Latest price for {symbol}: ${price:.2f}")

        # Get historical bars
        bars = await broker.get_bars(symbol, limit=5)
        logger.info(f"✅ Retrieved {len(bars)} historical bars for {symbol}")

        return price
    except Exception as e:
        logger.error(f"❌ Error getting market data: {e}")
        return None


async def test_simple_market_order(broker, symbol='AAPL'):
    """Test simple market order using OrderBuilder."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Simple Market Order")
    logger.info("="*80)

    try:
        # Build market order
        order = OrderBuilder(symbol, 'buy', 1).market().day().build()

        logger.info(f"📝 Market order created: Buy 1 share of {symbol}")
        logger.info(f"   Would submit order (commented out for safety)")

        # Uncomment to actually submit:
        # result = await broker.submit_order_advanced(order)
        # logger.info(f"✅ Order submitted: {result.id}")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating market order: {e}")
        return False


async def test_limit_order(broker, symbol='AAPL', limit_price=None):
    """Test limit order."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Limit Order")
    logger.info("="*80)

    try:
        # Get current price
        if not limit_price:
            current_price = await broker.get_last_price(symbol)
            # Set limit 2% below current price for buy
            limit_price = current_price * 0.98

        # Build limit order
        order = (
            OrderBuilder(symbol, 'buy', 1)
            .limit(limit_price)
            .gtc()  # Good-Till-Canceled
            .build()
        )

        logger.info(f"📝 Limit order created: Buy 1 share of {symbol} at ${limit_price:.2f}")
        logger.info(f"   Time in force: GTC (Good-Till-Canceled)")
        logger.info(f"   Would submit order (commented out for safety)")

        # Uncomment to actually submit:
        # result = await broker.submit_order_advanced(order)
        # logger.info(f"✅ Order submitted: {result.id}")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating limit order: {e}")
        return False


async def test_stop_order(broker, symbol='AAPL'):
    """Test stop order."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Stop Order")
    logger.info("="*80)

    try:
        current_price = await broker.get_last_price(symbol)
        # Set stop price 5% above current price
        stop_price = current_price * 1.05

        # Build stop order
        order = (
            OrderBuilder(symbol, 'buy', 1)
            .stop(stop_price)
            .gtc()
            .build()
        )

        logger.info(f"📝 Stop order created: Buy 1 share of {symbol} when price hits ${stop_price:.2f}")
        logger.info(f"   Current price: ${current_price:.2f}")
        logger.info(f"   Would submit order (commented out for safety)")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating stop order: {e}")
        return False


async def test_trailing_stop_order(broker, symbol='AAPL'):
    """Test trailing stop order."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Trailing Stop Order")
    logger.info("="*80)

    try:
        # Trailing stop with 2.5% trail
        order = (
            OrderBuilder(symbol, 'sell', 1)
            .trailing_stop(trail_percent=2.5)
            .gtc()
            .build()
        )

        logger.info(f"📝 Trailing stop order created: Sell 1 share of {symbol}")
        logger.info(f"   Trail: 2.5% below highest price")
        logger.info(f"   Would submit order (commented out for safety)")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating trailing stop order: {e}")
        return False


async def test_bracket_order(broker, symbol='AAPL'):
    """Test bracket order (entry + take-profit + stop-loss)."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Bracket Order")
    logger.info("="*80)

    try:
        current_price = await broker.get_last_price(symbol)

        # Calculate levels
        entry_price = current_price  # Market entry
        take_profit_price = current_price * 1.05  # 5% profit
        stop_loss_price = current_price * 0.97     # 3% loss
        stop_limit_price = current_price * 0.965   # 0.5% below stop

        # Build bracket order
        order = (
            OrderBuilder(symbol, 'buy', 1)
            .market()
            .bracket(
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
                stop_limit=stop_limit_price
            )
            .gtc()
            .build()
        )

        logger.info(f"📝 Bracket order created: Buy 1 share of {symbol}")
        logger.info(f"   Entry: Market order (~${entry_price:.2f})")
        logger.info(f"   Take-Profit: ${take_profit_price:.2f} (+5%)")
        logger.info(f"   Stop-Loss: ${stop_loss_price:.2f} (-3%)")
        logger.info(f"   Stop-Limit: ${stop_limit_price:.2f}")
        logger.info(f"   Risk/Reward: 3% / 5% = 1:1.67")
        logger.info(f"   Would submit order (commented out for safety)")

        # Uncomment to actually submit:
        # result = await broker.submit_order_advanced(order)
        # logger.info(f"✅ Bracket order submitted: {result.id}")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating bracket order: {e}")
        return False


async def test_oco_order(broker, symbol='AAPL'):
    """Test OCO (One-Cancels-Other) order."""
    logger.info("\n" + "="*80)
    logger.info("TEST: OCO Order (One-Cancels-Other)")
    logger.info("="*80)

    try:
        current_price = await broker.get_last_price(symbol)

        # Sell limit above current price (take profit)
        take_profit = current_price * 1.05
        # Stop loss below current price
        stop_loss = current_price * 0.97

        # Build OCO order
        order = (
            OrderBuilder(symbol, 'sell', 1)
            .limit(current_price)  # Required for OCO
            .oco(
                take_profit=take_profit,
                stop_loss=stop_loss
            )
            .gtc()
            .build()
        )

        logger.info(f"📝 OCO order created: Sell 1 share of {symbol}")
        logger.info(f"   Take-Profit: ${take_profit:.2f}")
        logger.info(f"   Stop-Loss: ${stop_loss:.2f}")
        logger.info(f"   Note: Assumes you already have a position")
        logger.info(f"   Would submit order (commented out for safety)")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating OCO order: {e}")
        return False


async def test_order_management(broker):
    """Test order management features."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Order Management")
    logger.info("="*80)

    try:
        # Get open orders
        from alpaca.trading.enums import QueryOrderStatus
        open_orders = await broker.get_orders(status=QueryOrderStatus.OPEN, limit=10)
        logger.info(f"✅ Open orders: {len(open_orders)}")

        for order in open_orders[:3]:  # Show first 3
            logger.info(f"   Order {order.id}: {order.side} {order.qty} {order.symbol} @ {order.type}")

        # Get all orders
        all_orders = await broker.get_orders(status=QueryOrderStatus.ALL, limit=5)
        logger.info(f"✅ Recent orders (all statuses): {len(all_orders)}")

        return True
    except Exception as e:
        logger.error(f"❌ Error with order management: {e}")
        return False


async def test_convenience_functions(symbol='AAPL'):
    """Test convenience functions."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Convenience Functions")
    logger.info("="*80)

    try:
        # Simple market order
        order1 = market_order(symbol, 'buy', 1, gtc=True)
        logger.info(f"✅ market_order() created successfully")

        # Simple limit order
        order2 = limit_order(symbol, 'sell', 1, 200.00, gtc=True)
        logger.info(f"✅ limit_order() created successfully")

        # Bracket order
        order3 = bracket_order(
            symbol, 'buy', 1,
            entry_price=None,  # Market entry
            take_profit=210.00,
            stop_loss=190.00,
            stop_limit=189.50
        )
        logger.info(f"✅ bracket_order() created successfully")

        return True
    except Exception as e:
        logger.error(f"❌ Error with convenience functions: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("ALPACA ADVANCED ORDER TYPES - TEST SUITE")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Mode: PAPER TRADING")
    logger.info("=" * 80)

    # Initialize broker
    try:
        broker = AlpacaBroker(paper=True)
        logger.info("✅ Broker initialized (paper trading mode)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize broker: {e}")
        return

    # Run tests
    results = {}

    results['account'] = await test_account_info(broker)
    results['market_data'] = await test_market_data(broker)
    results['simple_market'] = await test_simple_market_order(broker)
    results['limit'] = await test_limit_order(broker)
    results['stop'] = await test_stop_order(broker)
    results['trailing_stop'] = await test_trailing_stop_order(broker)
    results['bracket'] = await test_bracket_order(broker)
    results['oco'] = await test_oco_order(broker)
    results['management'] = await test_order_management(broker)
    results['convenience'] = await test_convenience_functions()

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20s}: {status}")

    logger.info("="*80)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    logger.info("="*80)

    logger.info("\n⚠️  NOTE: All actual order submissions are commented out for safety.")
    logger.info("   To enable real paper trading orders, uncomment the submit lines in each test.")


if __name__ == "__main__":
    asyncio.run(main())
