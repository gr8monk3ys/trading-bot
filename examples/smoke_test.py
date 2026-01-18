#!/usr/bin/env python3
"""
Smoke Test for Trading Bot - Basic Functionality Verification

This script verifies that the core trading bot components can be imported
and used without errors. It does NOT submit actual orders, but tests that
the order building logic works correctly.

Usage:
    python examples/smoke_test.py

Prerequisites:
    - .env file with Alpaca credentials (API_KEY, API_SECRET, PAPER=True)
    - All required dependencies installed (see requirements.txt)
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("TRADING BOT SMOKE TEST")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Test 1: Environment Variables
print("Test 1: Checking environment variables...")
try:
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")
    paper_mode = os.getenv("PAPER", "True")

    if not api_key or not api_secret:
        print("❌ FAILED: Missing Alpaca credentials in .env file")
        print("   Please create a .env file with:")
        print("   ALPACA_API_KEY=your_api_key_here")
        print("   ALPACA_SECRET_KEY=your_secret_key_here")
        print("   PAPER=True")
        sys.exit(1)

    print("✅ PASSED: Environment loaded")
    print(f"   API Key: {api_key[:8]}... (hidden)")
    print(f"   Paper Trading: {paper_mode}")
    print()
except Exception as e:
    print(f"❌ FAILED: Error loading environment: {e}")
    sys.exit(1)

# Test 2: Import Core Modules
print("Test 2: Importing core modules...")
try:
    from brokers.alpaca_broker import AlpacaBroker
    from brokers.order_builder import OrderBuilder, market_order, limit_order, bracket_order
    from config import SYMBOLS

    print("✅ PASSED: All core modules imported successfully")
    print("   - AlpacaBroker")
    print("   - OrderBuilder")
    print("   - Convenience functions (market_order, limit_order, bracket_order)")
    print(f"   - Config (SYMBOLS: {SYMBOLS[:3]}...)")
    print()
except ImportError as e:
    print(f"❌ FAILED: Import error: {e}")
    print("   Please ensure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Create Broker Instance (No Connection)
print("Test 3: Creating broker instance...")
try:
    broker = AlpacaBroker(paper=True)
    print("✅ PASSED: Broker instance created (paper trading mode)")
    print(f"   Broker type: {type(broker).__name__}")
    print()
except Exception as e:
    print(f"❌ FAILED: Error creating broker: {e}")
    print("   Check your Alpaca credentials in .env file")
    sys.exit(1)

# Test 4: Build Simple Market Order
print("Test 4: Building simple market order...")
try:
    symbol = "AAPL"
    qty = 1

    # Using OrderBuilder
    order = OrderBuilder(symbol, 'buy', qty).market().day().build()

    print("✅ PASSED: Market order created")
    print(f"   Symbol: {order.symbol}")
    print(f"   Side: {order.side}")
    print(f"   Quantity: {order.qty}")
    print(f"   Type: {order.type if hasattr(order, 'type') else 'MARKET'}")
    print(f"   Time in Force: {order.time_in_force}")
    print()
except Exception as e:
    print(f"❌ FAILED: Error building market order: {e}")
    sys.exit(1)

# Test 5: Build Limit Order
print("Test 5: Building limit order...")
try:
    symbol = "MSFT"
    qty = 2
    limit_price = 350.00

    order = OrderBuilder(symbol, 'buy', qty).limit(limit_price).gtc().build()

    print("✅ PASSED: Limit order created")
    print(f"   Symbol: {order.symbol}")
    print(f"   Side: {order.side}")
    print(f"   Quantity: {order.qty}")
    print(f"   Limit Price: ${order.limit_price}")
    print(f"   Time in Force: {order.time_in_force}")
    print()
except Exception as e:
    print(f"❌ FAILED: Error building limit order: {e}")
    sys.exit(1)

# Test 6: Build Bracket Order
print("Test 6: Building bracket order...")
try:
    symbol = "TSLA"
    qty = 1
    current_price = 250.00
    take_profit = current_price * 1.05  # 5% profit
    stop_loss = current_price * 0.97     # 3% loss
    stop_limit = current_price * 0.965   # 0.5% below stop

    order = (
        OrderBuilder(symbol, 'buy', qty)
        .market()
        .bracket(
            take_profit=take_profit,
            stop_loss=stop_loss,
            stop_limit=stop_limit
        )
        .gtc()
        .build()
    )

    print("✅ PASSED: Bracket order created")
    print(f"   Symbol: {order.symbol}")
    print(f"   Side: {order.side}")
    print(f"   Quantity: {order.qty}")
    print(f"   Order Class: {order.order_class}")
    print(f"   Take-Profit: ${order.take_profit['limit_price']:.2f}")
    print(f"   Stop-Loss: ${order.stop_loss['stop_price']:.2f}")
    if 'limit_price' in order.stop_loss:
        print(f"   Stop-Limit: ${order.stop_loss['limit_price']:.2f}")
    print()
except Exception as e:
    print(f"❌ FAILED: Error building bracket order: {e}")
    sys.exit(1)

# Test 7: Test Convenience Functions
print("Test 7: Testing convenience functions...")
try:
    # market_order convenience function
    order1 = market_order("AAPL", "buy", 1, gtc=True)
    print("✅ PASSED: market_order() function works")

    # limit_order convenience function
    order2 = limit_order("MSFT", "sell", 2, 400.00, gtc=True)
    print("✅ PASSED: limit_order() function works")

    # bracket_order convenience function
    order3 = bracket_order(
        "NVDA", "buy", 1,
        entry_price=None,  # Market entry
        take_profit=500.00,
        stop_loss=450.00,
        stop_limit=448.00
    )
    print("✅ PASSED: bracket_order() function works")
    print()
except Exception as e:
    print(f"❌ FAILED: Error with convenience functions: {e}")
    sys.exit(1)

# Test 8: Verify Order Object Attributes
print("Test 8: Verifying order object attributes...")
try:
    test_order = OrderBuilder("TEST", "buy", 10).market().day().build()

    # Check required attributes
    required_attrs = ['symbol', 'qty', 'side', 'time_in_force']
    missing_attrs = [attr for attr in required_attrs if not hasattr(test_order, attr)]

    if missing_attrs:
        print(f"❌ FAILED: Missing attributes: {missing_attrs}")
        sys.exit(1)

    print("✅ PASSED: All required order attributes present")
    print(f"   Order has: {', '.join(required_attrs)}")
    print()
except Exception as e:
    print(f"❌ FAILED: Error verifying order attributes: {e}")
    sys.exit(1)

# Summary
print("=" * 80)
print("SMOKE TEST SUMMARY")
print("=" * 80)
print("✅ All tests passed!")
print()
print("What was tested:")
print("  1. Environment variables loaded correctly")
print("  2. Core modules can be imported")
print("  3. Broker instance can be created")
print("  4. Simple market orders can be built")
print("  5. Limit orders can be built")
print("  6. Bracket orders can be built")
print("  7. Convenience functions work")
print("  8. Order objects have correct attributes")
print()
print("⚠️  NOTE: This test does NOT submit actual orders to Alpaca")
print("   To test real order submission, use examples/test_advanced_orders.py")
print("   and uncomment the submit lines")
print()
print("Next steps:")
print("  1. Run: python examples/test_advanced_orders.py")
print("  2. Check Alpaca paper trading dashboard for orders")
print("  3. Run: pytest tests/ for unit tests")
print("=" * 80)
