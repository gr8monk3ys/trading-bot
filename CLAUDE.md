# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Professional-grade algorithmic trading bot** with institutional-level features:
- **6 Production Strategies**: Momentum, Mean Reversion, Bracket, Ensemble, Pairs Trading, Extended Hours
- **30+ Technical Indicators**: Complete TA-Lib integration with advanced analysis
- **Extended Hours Trading**: Pre-market (4AM-9:30AM) and after-hours (4PM-8PM)
- **Market-Neutral Strategies**: Pairs trading with cointegration testing
- **Regime Detection**: Automatic market condition analysis (trending/ranging/volatile)
- **Advanced Risk Management**: Circuit breakers, position limits, Kelly Criterion

Built on Alpaca Trading API with full paper and live trading support.

## Key Commands

### Environment Setup
```bash
# Create and activate virtual environment
conda create -n trader python=3.10
conda activate trader

# Install dependencies
pip install -r requirements.txt
```

### Running the Bot

**Live Trading Mode:**
```bash
# Auto-select best strategies (paper trading)
python main.py live --strategy auto --max-strategies 3

# Run specific strategy (paper trading)
python main.py live --strategy MomentumStrategy

# Live trading (real money - use with caution)
python main.py live --strategy auto --real

# Force run even if market is closed
python main.py live --strategy auto --force
```

**Backtesting Mode:**
```bash
# Backtest all strategies
python main.py backtest --strategy all --start-date 2024-01-01 --end-date 2024-03-01

# Backtest specific strategy
python main.py backtest --strategy MomentumStrategy --start-date 2024-01-01

# With plots
python main.py backtest --strategy all --plot
```

**Parameter Optimization:**
```bash
# Optimize strategy parameters
python main.py optimize --strategy MomentumStrategy --optimize-for sharpe

# With custom date range
python main.py optimize --strategy MomentumStrategy --start-date 2024-01-01 --end-date 2024-03-01
```

### Testing
```bash
# Test Alpaca API connection
python tests/test_connection.py

# Test advanced order types (paper trading)
python examples/test_advanced_orders.py

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_momentum_strategy.py

# Run with verbose output
pytest -v tests/
```

### Development Scripts
```bash
# Simple backtesting script
python simple_backtest.py

# Advanced backtesting with multiple strategies
python smart_backtest.py

# Direct execution (alternative to main.py)
python run.py
```

## Architecture

### Core Components

**Strategy System (strategies/):**
- `BaseStrategy`: Abstract base class all strategies inherit from. Provides common functionality including:
  - Position sizing using Kelly Criterion and volatility adjustment
  - Risk limit checking (max drawdown, position limits)
  - Dynamic stop-loss updates based on volatility
  - Performance metric tracking
  - Async execution support
- Strategy implementations: Each strategy implements `analyze_symbol()` and `execute_trade()` methods
- Available strategies: MomentumStrategy, MeanReversionStrategy, SentimentStockStrategy, OptionsStrategy
- `RiskManager`: Centralized risk management including VaR, portfolio correlation, and position limits

**Engine System (engine/):**
- `StrategyManager`: Orchestrates multiple strategies simultaneously
  - Auto-discovers strategies in strategies/ directory
  - Evaluates and scores strategies based on backtested performance
  - Optimizes capital allocation across strategies (Sharpe ratio weighted)
  - Handles starting/stopping strategies and rebalancing
- `BacktestEngine`: Simulates strategy performance on historical data
  - Day-by-day execution simulation
  - Tracks equity curves, returns, and trades
  - Calculates performance metrics (Sharpe, drawdown, etc.)
- `PerformanceMetrics`: Calculates comprehensive performance statistics
- `StrategyEvaluator`: Scores and ranks strategies for selection

**Broker Layer (brokers/):**
- `AlpacaBroker`: Implements Lumibot Broker interface for Alpaca API
  - Retry logic with exponential backoff for API calls (decorator: `@retry_with_backoff`)
  - WebSocket support for real-time market data and trade updates
  - Handles both paper and live trading modes
  - Methods: `get_account()`, `get_positions()`, `submit_order()`, `get_bars()`, `get_last_price()`
- `BacktestBroker`: Mock broker for backtesting (in development)

**Utilities (utils/):**
- `sentiment_analysis.py`: FinBERT-based sentiment analysis for news and market data
- `stock_scanner.py`: Market scanning and symbol filtering utilities

### Configuration (config.py)

Centralized configuration with three main parameter groups:
- `TRADING_PARAMS`: Position sizing, stop-loss, take-profit, sentiment thresholds
- `RISK_PARAMS`: Portfolio-wide risk limits, VaR confidence, correlation limits
- `TECHNICAL_PARAMS`: SMA periods, RSI thresholds, price history windows

## Advanced Order Types (NEW!)

The repository now supports all Alpaca order types through the `OrderBuilder` utility:

### Order Types Available
- **Market Orders**: Execute at current price
- **Limit Orders**: Execute at specified price or better
- **Stop Orders**: Trigger at stop price, then become market orders
- **Stop-Limit Orders**: Trigger at stop, execute as limit order
- **Trailing Stop Orders**: Auto-adjust stop price as market moves favorably
- **Bracket Orders**: Entry + automatic take-profit + stop-loss in one order
- **OCO Orders** (One-Cancels-Other): Exit with both profit target and stop-loss
- **OTO Orders** (One-Triggers-Other): Entry with single exit order

### Time-In-Force Options
- **DAY**: Valid only during trading day
- **GTC**: Good-Till-Canceled (90-day expiry)
- **IOC**: Immediate-Or-Cancel
- **FOK**: Fill-Or-Kill (entire order or nothing)
- **OPG**: Execute in opening auction
- **CLS**: Execute in closing auction

### Using OrderBuilder
```python
from brokers.order_builder import OrderBuilder

# Simple market order
order = OrderBuilder('AAPL', 'buy', 100).market().day().build()
result = await broker.submit_order_advanced(order)

# Limit order with GTC
order = OrderBuilder('TSLA', 'sell', 50).limit(250.00).gtc().build()

# Trailing stop (2.5% trail)
order = OrderBuilder('SPY', 'sell', 100).trailing_stop(trail_percent=2.5).gtc().build()

# Bracket order (entry + take-profit + stop-loss)
order = (OrderBuilder('NVDA', 'buy', 10)
         .market()
         .bracket(take_profit=120.00, stop_loss=95.00, stop_limit=94.50)
         .gtc()
         .build())

# Extended hours limit order
order = OrderBuilder('AAPL', 'buy', 10).limit(150.00).extended_hours().day().build()
```

### Convenience Functions
```python
from brokers.order_builder import market_order, limit_order, bracket_order

# Quick market order
order = market_order('AAPL', 'buy', 100, gtc=True)

# Quick limit order
order = limit_order('TSLA', 'sell', 50, 250.00, gtc=True)

# Quick bracket order
order = bracket_order('NVDA', 'buy', 10,
                     take_profit=120.00,
                     stop_loss=95.00,
                     stop_limit=94.50)
```

### Enhanced Broker Methods
```python
# Submit advanced orders
result = await broker.submit_order_advanced(order_builder)

# Cancel orders
await broker.cancel_order(order_id)
await broker.cancel_all_orders()

# Replace/modify orders
await broker.replace_order(order_id, qty=150, limit_price=155.00)

# Get orders
open_orders = await broker.get_orders(status=QueryOrderStatus.OPEN)
order = await broker.get_order_by_id(order_id)
order = await broker.get_order_by_client_id(client_order_id)
```

### Example Strategy with Bracket Orders
See `strategies/bracket_momentum_strategy.py` for a complete example that:
- Identifies momentum using RSI, MACD, and moving averages
- Automatically sets profit targets and stop losses
- Uses bracket orders with GTC time-in-force
- Supports ATR-based dynamic stops

```bash
# Run bracket momentum strategy
python strategies/bracket_momentum_strategy.py
```

## Important Implementation Details

### Strategy Development

When creating a new strategy:
1. Inherit from `BaseStrategy` in `strategies/base_strategy.py`
2. Implement required methods:
   - `analyze_symbol(symbol)`: Returns trading signal
   - `execute_trade(symbol, signal)`: Executes the trade
3. Override `_initialize_parameters()` for custom parameters
4. Set `NAME` class attribute for identification
5. Place in `strategies/` directory - StrategyManager will auto-discover it

### Async/Await Pattern

The codebase uses async/await extensively:
- All broker operations are async
- Strategy execution is async (`on_trading_iteration()`)
- Use `await` for broker calls: `await self.broker.get_positions()`
- Main entry point uses `asyncio.run()` for the event loop

### Retry Logic for API Calls

Alpaca API calls in `AlpacaBroker` use `@retry_with_backoff` decorator:
- 3 retry attempts by default
- Exponential backoff (1s → 2s → 4s, max 10s)
- Applied to all critical broker operations
- Prevents transient network failures from crashing strategies

### Price History Management

Strategies maintain `self.price_history` dict for volatility calculations:
- Keys are symbols, values are price arrays
- Limited by `price_history_window` parameter (default: 30)
- Used for dynamic stop-loss calculation via `_calculate_volatility()`
- Must be populated by strategy during `analyze_symbol()`

### Multi-Strategy Execution

`StrategyManager` handles concurrent strategy execution:
- Each strategy gets allocated a % of capital (`allocation` parameter)
- Strategies evaluated via backtesting, then scored
- Top N strategies selected based on score threshold
- Capital optimally allocated based on Sharpe ratios
- Periodic rebalancing supported (set via `--evaluation-interval`)

### Paper vs Live Trading

Controlled by `PAPER` environment variable in `.env`:
- `PAPER=True`: Uses paper trading endpoint (safe for testing)
- `PAPER=False`: Uses live trading endpoint (real money)
- Default is paper trading for safety
- AlpacaBroker constructor: `AlpacaBroker(paper=True/False)`

## Coding Style Notes

Following `.windsurfrules` guidance:
- Functional programming preferred over classes where appropriate
- Vectorized operations (pandas/numpy) over explicit loops
- Use method chaining for data transformations
- PEP 8 style guidelines
- Descriptive variable names reflecting data content

## Common Gotchas

1. **Async Context**: Remember to use `await` for all broker operations - forgetting will cause runtime errors
2. **Price History**: Strategies must populate `self.price_history` before calling `_calculate_volatility()`
3. **Strategy Discovery**: New strategies must be in `strategies/` directory and inherit from `BaseStrategy` to be auto-discovered
4. **API Rate Limits**: The retry decorator helps but be mindful of Alpaca API rate limits during development
5. **Market Hours**: By default, bot won't run if market is closed unless `--force` flag is used
6. **NumPy Version**: Pinned to <2.0.0 in requirements.txt for compatibility - don't upgrade

## Data Storage

- `data/`: Historical price data and cached market information
- `results/`: Backtest results and performance reports
- `docs/`: Additional documentation (installation.md, strategy_guide.md)
- `examples/`: Example strategy implementations and usage patterns

## Environment Variables Required

In `.env` file:
```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
PAPER=True
```

---

## Troubleshooting

### Circular Import Issues

**Problem:** Cannot import AlpacaBroker due to circular import between `brokers/__init__.py` and `brokers/alpaca_broker.py`.

**Symptoms:**
```python
from brokers.alpaca_broker import AlpacaBroker
# ImportError or circular import error
```

**Solution:**
The OrderBuilder import in `alpaca_broker.py` should be inside methods, not at module level:
```python
# In brokers/alpaca_broker.py
# WRONG (at top of file):
from brokers.order_builder import OrderBuilder

# CORRECT (inside method):
async def submit_order_advanced(self, order_request):
    from brokers.order_builder import OrderBuilder
    # ... rest of method
```

**Test Fix:**
```bash
python -c "from brokers.alpaca_broker import AlpacaBroker; print('Import successful')"
```

### Missing Dependencies

**Problem:** ImportError for ta-lib, torch, or other dependencies.

**Solution:**
```bash
# Ensure you're in the right environment
conda activate trader

# Reinstall dependencies
pip install -r requirements.txt

# For TA-Lib specifically (macOS):
brew install ta-lib
pip install ta-lib

# For TA-Lib (Linux):
# Download from https://ta-lib.org/ and compile
# Then: pip install ta-lib
```

### .env File Issues

**Problem:** KeyError for ALPACA_API_KEY or missing credentials.

**Solution:**
```bash
# Create .env from example
cp .env.example .env

# Edit with your credentials (use nano, vim, or your editor)
nano .env

# Verify .env exists and has content
cat .env

# Should show:
# ALPACA_API_KEY=your_actual_key
# ALPACA_SECRET_KEY=your_actual_secret
# PAPER=True
```

**Test:**
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key:', os.getenv('ALPACA_API_KEY'))"
```

### Alpaca Connection Problems

**Problem:** Cannot connect to Alpaca API or authentication fails.

**Diagnosis:**
```bash
# Test connection
python tests/test_connection.py
```

**Common Issues:**
1. **Wrong credentials:** Verify API key and secret in .env
2. **Paper vs Live mismatch:** Ensure PAPER=True for paper trading credentials
3. **API endpoint:** Paper trading uses different endpoint than live
4. **Network issues:** Check firewall/proxy settings
5. **SSL certificate issues:** See README.md for certificate installation

**Solution:**
```bash
# Verify credentials at https://app.alpaca.markets/paper/dashboard/overview
# Make sure PAPER=True if using paper trading keys

# Test import and connection
python -c "
from brokers.alpaca_broker import AlpacaBroker
import asyncio

async def test():
    broker = AlpacaBroker(paper=True)
    account = await broker.get_account()
    print(f'Account: {account.id}')
    print(f'Buying Power: {account.buying_power}')

asyncio.run(test())
"
```

### Import Path Issues

**Problem:** Module not found errors.

**Solution:**
```bash
# Make sure you're in the project root
cd /Users/gr8monk3ys/code/trading-bot

# Verify project structure
ls -la brokers/
ls -la strategies/
ls -la engine/

# Python path should include current directory
python -c "import sys; print(sys.path)"
```

### Strategy Not Found

**Problem:** main.py can't find a strategy.

**Solution:**
```bash
# List available strategies
python -c "
from engine.strategy_manager import StrategyManager
import asyncio

async def list_strategies():
    manager = StrategyManager(None)
    strategies = manager.get_available_strategy_names()
    print('Available strategies:', strategies)

asyncio.run(list_strategies())
"

# Verify strategy file exists
ls -la strategies/*.py
```

---

## Current Limitations

**As of 2025-11-07, the following limitations exist:**

### Advanced Order Support

**Status:** Partial implementation

- **Working:** `BracketMomentumStrategy` fully implements advanced orders (bracket, trailing stops, etc.)
- **In Progress:** Other strategies still use basic market/limit orders
- **Not Integrated:** OrderBuilder not yet integrated into `MomentumStrategy`, `MeanReversionStrategy`, `SentimentStockStrategy`

**Impact:**
- Only `BracketMomentumStrategy` benefits from automatic stop-loss and take-profit via bracket orders
- Other strategies require manual risk management
- `main.py live --strategy auto` may select strategies without advanced order support

**Recommendation:**
Use `BracketMomentumStrategy` for best risk management:
```bash
python main.py live --strategy BracketMomentumStrategy
```

### Testing Status

**Current State:**
- Core functionality tested manually
- Integration tests in progress
- Advanced order testing requires paper trading account
- No automated test coverage for order execution

**What's Been Tested:**
- AlpacaBroker basic operations (get_account, get_positions, get_bars)
- OrderBuilder order construction
- Strategy evaluation and backtesting
- StrategyManager auto-discovery

**What Needs More Testing:**
- Bracket order execution end-to-end
- OCO and OTO orders in live trading
- Extended hours trading
- Order replacement and cancellation
- WebSocket streaming reliability

**Recommendation:**
- Always test in paper trading first
- Monitor closely for the first week
- Start with small position sizes
- Review logs daily

### Known Bugs

See [TODO.md](TODO.md) for complete list. Critical items:

1. **Circular Import:** May occur if OrderBuilder imported at module level in brokers
2. **Strategy Discovery:** New strategies must be in strategies/ directory and properly inherit from BaseStrategy
3. **Price History:** Strategies must populate price_history before calculating volatility
4. **Market Hours:** Bot won't run if market is closed unless --force flag used

### Feature Gaps

**Not Yet Implemented:**
- Real-time trade execution notifications
- Portfolio rebalancing automation
- Multi-asset class support (only stocks currently)
- Options trading (infrastructure exists but untested)
- Risk alerts and notifications
- Performance dashboards

**Planned:**
- Complete OrderBuilder integration across all strategies
- Automated parameter optimization
- Strategy A/B testing framework
- Real-time performance monitoring
- Slack/email notifications

---

## Development Notes

### When Adding New Strategies

**Checklist:**
1. Inherit from `BaseStrategy`
2. Implement `analyze_symbol()` and `execute_trade()`
3. Set `NAME` class attribute
4. Define `default_parameters()` static method
5. Place in `strategies/` directory
6. Test imports before running
7. Backtest before live trading

**OrderBuilder Integration:**
```python
from brokers.order_builder import OrderBuilder

# In execute_trade():
order = (OrderBuilder(symbol, 'buy', quantity)
         .market()
         .bracket(take_profit=tp_price, stop_loss=sl_price)
         .gtc()
         .build())

result = await self.broker.submit_order_advanced(order)
```

### When Modifying Broker Code

**Be Careful With:**
- Circular imports (don't import OrderBuilder at module level)
- Async/await patterns (all broker methods are async)
- Error handling (API calls can fail)
- Retry logic (exponential backoff is implemented)

**Testing:**
```bash
# Always test imports after broker changes
python -c "from brokers.alpaca_broker import AlpacaBroker"
python -c "from brokers.order_builder import OrderBuilder"
```

---

## Quick Diagnosis Commands

```bash
# Test everything at once
echo "Testing imports..."
python -c "from brokers.alpaca_broker import AlpacaBroker; print('✓ AlpacaBroker')"
python -c "from brokers.order_builder import OrderBuilder; print('✓ OrderBuilder')"
python -c "from strategies.base_strategy import BaseStrategy; print('✓ BaseStrategy')"
python -c "from engine.strategy_manager import StrategyManager; print('✓ StrategyManager')"

echo -e "\nTesting connection..."
python tests/test_connection.py

echo -e "\nTesting environment..."
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key exists:', bool(os.getenv('ALPACA_API_KEY')))"

echo -e "\nAll tests passed!"
```
