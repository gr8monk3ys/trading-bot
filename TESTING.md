# Testing Guide for Trading Bot

This guide provides comprehensive instructions for testing the trading bot components.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Test Types](#test-types)
- [Running Tests](#running-tests)
- [Expected Output](#expected-output)
- [Troubleshooting](#troubleshooting)
- [Verifying in Alpaca Dashboard](#verifying-in-alpaca-dashboard)

---

## Prerequisites

### 1. Conda Environment
Ensure you have a conda environment activated with Python 3.10+:

```bash
conda activate trading-bot  # or your environment name
python --version  # Should be 3.10 or higher
```

### 2. Dependencies
All required packages must be installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `alpaca-py>=0.8.0` - Alpaca trading API
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.3` - Numerical computing
- `ta-lib` - Technical analysis (optional but recommended)
- `python-dotenv>=1.0.0` - Environment variable management
- `pytest` - Testing framework

### 3. Alpaca Paper Trading Account
You need an Alpaca paper trading account:
1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Navigate to Paper Trading section
3. Generate API keys

### 4. Environment Configuration
Create a `.env` file in the project root with your Alpaca credentials:

```bash
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here
PAPER=True
```

**Important:** Never commit the `.env` file to version control. It's already in `.gitignore`.

You can reference `.env.example` for the template:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

---

## Quick Start

Run the smoke test to verify everything is working:

```bash
python examples/smoke_test.py
```

If this passes, you're ready to run more comprehensive tests!

---

## Test Types

### 1. Smoke Test (`examples/smoke_test.py`)
**Purpose:** Quick validation that core components work without errors.

**What it tests:**
- Environment variables are loaded correctly
- All core modules can be imported
- Broker instance can be created
- Order objects can be built (market, limit, bracket)
- Convenience functions work
- Order objects have correct attributes

**Does NOT test:**
- Actual API connections to Alpaca
- Real order submissions
- WebSocket connections

**Usage:**
```bash
python examples/smoke_test.py
```

**Expected runtime:** < 5 seconds

---

### 2. Import Tests (`tests/test_imports.py`)
**Purpose:** Verify all critical imports work (pytest-based).

**What it tests:**
- Broker imports (AlpacaBroker, OrderBuilder, BacktestBroker)
- Strategy imports (all strategy classes)
- Configuration imports
- External dependency imports (pandas, numpy, alpaca-py)
- Enum imports (OrderSide, TimeInForce, etc.)
- Order builder instantiation

**Usage:**
```bash
# Run all import tests
pytest tests/test_imports.py -v

# Run specific test class
pytest tests/test_imports.py::TestBrokerImports -v

# Run specific test
pytest tests/test_imports.py::TestBrokerImports::test_alpaca_broker_import -v
```

**Expected runtime:** < 10 seconds

---

### 3. Advanced Order Tests (`examples/test_advanced_orders.py`)
**Purpose:** Comprehensive testing of all order types with Alpaca API.

**What it tests:**
- Account information retrieval
- Market data fetching
- All order types (market, limit, stop, trailing stop, bracket, OCO, OTO)
- Order management features
- Convenience functions

**Important:** By default, order submissions are **commented out** for safety.

**Usage:**
```bash
# Run without submitting orders (safe)
python examples/test_advanced_orders.py

# To enable real paper trading orders:
# 1. Edit examples/test_advanced_orders.py
# 2. Uncomment the submit lines in each test function
# 3. Run the script
python examples/test_advanced_orders.py
```

**Expected runtime:** 30-60 seconds (with API calls)

---

### 4. Unit Tests (`tests/`)
**Purpose:** Test individual components in isolation.

**Available tests:**
- `tests/test_imports.py` - Import validation
- `tests/test_connection.py` - Alpaca connection tests
- `tests/test_momentum_strategy.py` - Momentum strategy tests
- `tests/test_backtest_engine.py` - Backtesting engine tests

**Usage:**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_connection.py -v

# Run and show print statements
pytest tests/ -v -s
```

---

## Running Tests

### Step 1: Verify Environment
```bash
# Check Python version
python --version

# Verify dependencies
pip list | grep -E "alpaca-py|pandas|numpy|pytest"

# Check .env file exists
ls -la .env
```

### Step 2: Run Smoke Test
```bash
python examples/smoke_test.py
```

**Expected output:**
```
================================================================================
TRADING BOT SMOKE TEST
================================================================================
Started at: 2025-11-07 10:30:45

Test 1: Checking environment variables...
✅ PASSED: Environment loaded
   API Key: PKxxxxxx... (hidden)
   Paper Trading: True

Test 2: Importing core modules...
✅ PASSED: All core modules imported successfully
   - AlpacaBroker
   - OrderBuilder
   - Convenience functions (market_order, limit_order, bracket_order)
   - Config (SYMBOLS: ['AAPL', 'MSFT', 'AMZN']...)

Test 3: Creating broker instance...
✅ PASSED: Broker instance created (paper trading mode)
   Broker type: AlpacaBroker

Test 4: Building simple market order...
✅ PASSED: Market order created
   Symbol: AAPL
   Side: OrderSide.BUY
   Quantity: 1.0
   Type: MARKET
   Time in Force: TimeInForce.DAY

Test 5: Building limit order...
✅ PASSED: Limit order created
   Symbol: MSFT
   Side: OrderSide.BUY
   Quantity: 2.0
   Limit Price: $350.0
   Time in Force: TimeInForce.GTC

Test 6: Building bracket order...
✅ PASSED: Bracket order created
   Symbol: TSLA
   Side: OrderSide.BUY
   Quantity: 1.0
   Order Class: OrderClass.BRACKET
   Take-Profit: $262.50
   Stop-Loss: $242.50
   Stop-Limit: $241.25

Test 7: Testing convenience functions...
✅ PASSED: market_order() function works
✅ PASSED: limit_order() function works
✅ PASSED: bracket_order() function works

Test 8: Verifying order object attributes...
✅ PASSED: All required order attributes present
   Order has: symbol, qty, side, time_in_force

================================================================================
SMOKE TEST SUMMARY
================================================================================
✅ All tests passed!

What was tested:
  1. Environment variables loaded correctly
  2. Core modules can be imported
  3. Broker instance can be created
  4. Simple market orders can be built
  5. Limit orders can be built
  6. Bracket orders can be built
  7. Convenience functions work
  8. Order objects have correct attributes

⚠️  NOTE: This test does NOT submit actual orders to Alpaca
   To test real order submission, use examples/test_advanced_orders.py
   and uncomment the submit lines

Next steps:
  1. Run: python examples/test_advanced_orders.py
  2. Check Alpaca paper trading dashboard for orders
  3. Run: pytest tests/ for unit tests
================================================================================
```

### Step 3: Run Import Tests
```bash
pytest tests/test_imports.py -v
```

**Expected output:**
```
========================== test session starts ===========================
collected 25 items

tests/test_imports.py::TestBrokerImports::test_alpaca_broker_import PASSED    [  4%]
tests/test_imports.py::TestBrokerImports::test_order_builder_import PASSED    [  8%]
tests/test_imports.py::TestBrokerImports::test_order_builder_convenience_functions PASSED [ 12%]
tests/test_imports.py::TestBrokerImports::test_backtest_broker_import PASSED  [ 16%]
tests/test_imports.py::TestStrategyImports::test_base_strategy_import PASSED  [ 20%]
tests/test_imports.py::TestStrategyImports::test_bracket_momentum_strategy_import PASSED [ 24%]
...
========================== 25 passed in 2.35s ============================
```

### Step 4: Run Advanced Order Tests (Optional)
```bash
python examples/test_advanced_orders.py
```

This will test API connectivity and order building without submitting orders (by default).

---

## Expected Output

### Success Indicators
- ✅ Green checkmarks for passed tests
- Clear descriptions of what was tested
- Relevant details (symbols, prices, quantities)
- Final summary showing all tests passed

### Failure Indicators
- ❌ Red X marks for failed tests
- Error messages with details
- Exit code 1 (for smoke test)

---

## Troubleshooting

### Problem: Missing .env file
**Error:** `❌ FAILED: Missing Alpaca credentials in .env file`

**Solution:**
1. Copy the example: `cp .env.example .env`
2. Edit `.env` with your actual Alpaca credentials
3. Ensure `PAPER=True` for paper trading

---

### Problem: Import errors
**Error:** `ImportError: No module named 'alpaca'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# If specific package is missing
pip install alpaca-py pandas numpy python-dotenv
```

---

### Problem: TA-Lib import error
**Error:** `ImportError: No module named 'talib'`

**Solution:**
TA-Lib requires system-level installation:

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**Note:** The import tests will skip TA-Lib tests if not installed.

---

### Problem: Alpaca API connection errors
**Error:** `401 Unauthorized` or `403 Forbidden`

**Solution:**
1. Verify API keys in `.env` are correct
2. Ensure you're using **Paper Trading** keys (not live)
3. Check keys haven't expired
4. Regenerate keys in Alpaca dashboard if needed

---

### Problem: Broker initialization fails
**Error:** `❌ FAILED: Error creating broker: ...`

**Solution:**
1. Check internet connection
2. Verify Alpaca service status: https://status.alpaca.markets/
3. Ensure `PAPER=True` in `.env` file
4. Try regenerating API keys

---

## Verifying in Alpaca Dashboard

After running tests that submit orders (with submit lines uncommented):

### Step 1: Log in to Alpaca
Visit: https://app.alpaca.markets/paper/dashboard/overview

### Step 2: Navigate to Orders
Click on "Orders" in the left sidebar

### Step 3: Verify Orders
You should see:
- **Open Orders:** Pending limit orders, bracket orders
- **Filled Orders:** Executed market orders
- **Canceled Orders:** Orders that were canceled

### Step 4: Check Account
- **Portfolio Value:** Should reflect any filled orders
- **Positions:** Any open positions from filled buy orders
- **Buying Power:** Reduced if orders were filled

### Step 5: Clean Up (Optional)
To reset your paper account:
1. Cancel all open orders
2. Close all positions
3. Or reset the entire paper account in settings

---

## Best Practices

### 1. Always Test in Paper Mode First
- Never use live credentials for testing
- Always set `PAPER=True` in `.env`
- Verify you're connected to paper environment

### 2. Start with Smoke Tests
- Run `smoke_test.py` before more complex tests
- Fix any import or configuration issues first
- Progress to API tests only after smoke tests pass

### 3. Use Small Quantities
- Test with 1-2 shares
- Verify order logic before scaling up
- Paper trading accounts have limited buying power

### 4. Monitor Orders
- Check Alpaca dashboard after submitting orders
- Verify orders are created correctly
- Cancel test orders when done

### 5. Clean Up Test Orders
- Don't leave stale open orders
- Close test positions regularly
- Reset paper account if needed

---

## Test Coverage

Current test coverage:

| Component | Coverage |
|-----------|----------|
| Order Building | ✅ Full |
| Broker Imports | ✅ Full |
| Strategy Imports | ✅ Full |
| API Connection | ⚠️ Partial |
| Order Submission | ⚠️ Manual |
| WebSocket | ❌ Not covered |
| Backtesting | ⚠️ Partial |

---

## Next Steps

After successful testing:

1. **Run Backtests**
   ```bash
   python simple_backtest.py
   python smart_backtest.py
   ```

2. **Test Strategies**
   ```bash
   python tests/strategy_tester.py
   ```

3. **Live Paper Trading**
   ```bash
   python main.py
   ```

4. **Review Results**
   - Check `results/` directory for backtest results
   - Analyze performance metrics
   - Adjust strategy parameters

---

## Additional Resources

- **Alpaca API Docs:** https://docs.alpaca.markets/
- **Alpaca Python SDK:** https://github.com/alpacahq/alpaca-py
- **Project README:** See `README.md` for overall architecture
- **Strategy Docs:** See `docs/` directory for strategy details

---

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review error messages carefully
3. Verify all prerequisites are met
4. Check Alpaca service status
5. Consult Alpaca API documentation

---

**Last Updated:** 2025-11-07
**Version:** 1.0.0
