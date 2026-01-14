# Testing Documentation

**Comprehensive testing infrastructure for the trading bot**

This document explains the testing strategy, infrastructure, and how to run tests.

---

## Overview

**Testing Philosophy:**
- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: End-to-end tests for complete workflows
- **Performance Tests**: Ensure strategies meet performance criteria
- **Mock Infrastructure**: Test without external API dependencies

**Coverage Target:** >80% for critical modules (strategies/, brokers/, engine/, utils/)

---

## Quick Start

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/ -m unit -v
pytest tests/ -m integration -v
pytest tests/ -m performance -v

# Run specific test file
pytest tests/unit/test_momentum_strategy.py -v

# Run with coverage
pytest tests/ --cov=strategies --cov=brokers --cov=engine --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Markers

Tests are organized using pytest markers:

| Marker | Description | Speed | Example |
|--------|-------------|-------|---------|
| `unit` | Unit tests (isolated) | Fast (<1s) | Test indicator calculation |
| `integration` | Integration tests | Medium (1-10s) | Test complete trading workflow |
| `performance` | Performance tests | Slow (10s+) | Test backtest performance |
| `strategy` | Strategy-specific | Varies | Test momentum strategy signals |
| `broker` | Broker-specific | Fast | Test order submission |
| `engine` | Engine tests | Medium | Test strategy manager |
| `risk` | Risk management | Fast | Test position sizing |
| `requires_api` | Needs external API | Slow | Test Alpaca connection |
| `slow` | Slow running tests | Slow | Optimization tests |

**Running by marker:**
```bash
# Run only fast unit tests
pytest -m unit -v

# Run strategy tests but skip slow ones
pytest -m "strategy and not slow" -v

# Run integration tests excluding API tests
pytest -m "integration and not requires_api" -v
```

---

## Test Infrastructure

### Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── fixtures/                # Mock objects and test data
│   ├── __init__.py
│   ├── mock_broker.py       # MockAlpacaBroker for testing
│   ├── mock_data.py         # Price data generators
│   └── test_helpers.py      # Helper functions and assertions
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_momentum_strategy.py
│   ├── test_mean_reversion_strategy.py
│   ├── test_alpaca_broker.py
│   ├── test_strategy_manager.py
│   └── test_backtest_engine.py
├── integration/             # Integration tests
│   ├── __init__.py
│   ├── test_live_trading.py
│   ├── test_backtest_workflow.py
│   └── test_multi_strategy.py
└── performance/             # Performance tests
    ├── __init__.py
    ├── test_strategy_performance.py
    └── test_optimization.py
```

### Configuration Files

**pytest.ini** - Main pytest configuration:
```ini
[pytest]
# Test discovery
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = tests

# Coverage
addopts =
    --cov=strategies
    --cov=brokers
    --cov=engine
    --cov=utils
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=70

# Markers
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests
    performance: Performance tests
    strategy: Strategy-specific tests
    broker: Broker-specific tests
```

**.coveragerc** - Coverage configuration:
```ini
[run]
source = .
omit =
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*

[report]
precision = 2
show_missing = True
skip_covered = False
```

---

## Mock Infrastructure

### MockAlpacaBroker

**Purpose**: Simulate Alpaca API without external dependencies

**Features:**
- Realistic market simulation
- Configurable slippage (default: 0.1%)
- Multiple market regimes (bull, bear, sideways, volatile)
- Order execution and fill simulation
- Position and cash tracking
- Historical price data generation

**Usage:**
```python
from tests.fixtures.mock_broker import MockAlpacaBroker

# Create mock broker
broker = MockAlpacaBroker(paper=True, initial_capital=100000.0)

# Get account
account = await broker.get_account()
print(f"Equity: ${account.equity}")

# Submit order
order = await broker.submit_order(
    symbol='AAPL',
    qty=10,
    side='buy',
    type='market'
)

# Check positions
positions = await broker.get_positions()
```

**Market Regime Simulation:**
```python
# Set different market conditions for testing
broker.set_market_regime('AAPL', 'bull')     # Uptrend
broker.set_market_regime('TSLA', 'bear')     # Downtrend
broker.set_market_regime('SPY', 'sideways')  # Ranging
broker.set_market_regime('NVDA', 'volatile') # High volatility
```

### Mock Data Generators

**Purpose**: Generate realistic test data for various scenarios

**Available Functions:**

**1. Price Series Generation:**
```python
from tests.fixtures.mock_data import generate_price_series

# Generate uptrend
prices = generate_price_series(
    start_price=100.0,
    days=100,
    trend="up",
    volatility=0.02
)
```

**2. OHLCV Data:**
```python
from tests.fixtures.mock_data import generate_ohlcv_data

# Generate full OHLCV data
df = generate_ohlcv_data(
    symbol='TEST',
    days=100,
    start_price=100.0,
    trend='up',
    volatility=0.02
)
```

**3. Scenario-Specific Data:**
```python
from tests.fixtures.mock_data import (
    generate_momentum_scenario,      # Strong uptrend
    generate_mean_reversion_scenario, # Oscillating pattern
    generate_volatile_scenario,       # High volatility
    generate_sideways_scenario        # Ranging market
)

# Create momentum test data
df = generate_momentum_scenario(days=100)
```

**4. Multi-Symbol Data:**
```python
from tests.fixtures.mock_data import generate_multi_symbol_data

# Generate data for multiple symbols
data = generate_multi_symbol_data(
    symbols=['AAPL', 'TSLA', 'SPY'],
    days=100
)
```

### Test Helpers

**Purpose**: Common assertions and utilities for tests

**Assertion Functions:**
```python
from tests.fixtures.test_helpers import (
    assert_approximately_equal,
    assert_in_range,
    assert_valid_signal,
    assert_valid_order,
    assert_valid_position
)

# Test float equality with tolerance
assert_approximately_equal(actual=101.5, expected=100.0, tolerance=0.02)

# Test value in range
assert_in_range(value=50, min_val=0, max_val=100)

# Validate trading signal
assert_valid_signal({'action': 'buy', 'confidence': 0.8})

# Validate order structure
assert_valid_order({
    'symbol': 'AAPL',
    'qty': 10,
    'side': 'buy',
    'type': 'market'
})
```

**Calculation Helpers:**
```python
from tests.fixtures.test_helpers import (
    calculate_expected_pnl,
    calculate_expected_return,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)

# Calculate expected P&L
pnl = calculate_expected_pnl(
    qty=10,
    entry_price=100.0,
    exit_price=105.0,
    side='long'
)  # Returns 50.0

# Calculate Sharpe ratio
sharpe = calculate_sharpe_ratio(returns_series)
```

---

## Writing Tests

### Unit Test Example

```python
import pytest
from tests.fixtures.mock_broker import MockAlpacaBroker
from tests.fixtures.mock_data import generate_momentum_scenario
from strategies.momentum_strategy import MomentumStrategy


@pytest.mark.unit
@pytest.mark.strategy
@pytest.mark.asyncio
class TestMomentumStrategy:
    """Unit tests for MomentumStrategy"""

    @pytest.fixture
    async def broker(self):
        """Create mock broker"""
        return MockAlpacaBroker(paper=True, initial_capital=100000)

    @pytest.fixture
    async def strategy(self, broker):
        """Create strategy instance"""
        strategy = MomentumStrategy(
            broker=broker,
            symbols=['TEST']
        )
        await strategy.initialize()
        return strategy

    async def test_indicator_calculation(self, strategy):
        """Test technical indicators are calculated correctly"""
        symbol = 'TEST'

        # Generate test data
        df = generate_momentum_scenario(days=100)

        # Populate price history
        for _, row in df.iterrows():
            strategy.price_history[symbol].append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Update indicators
        await strategy._update_indicators(symbol)

        # Verify indicators
        assert strategy.indicators[symbol]['rsi'] is not None
        assert 0 <= strategy.indicators[symbol]['rsi'] <= 100
        assert strategy.indicators[symbol]['macd'] is not None
        assert strategy.indicators[symbol]['adx'] is not None

    async def test_buy_signal_generation(self, strategy):
        """Test buy signal is generated correctly"""
        symbol = 'TEST'

        # Set up indicators for buy signal
        strategy.indicators[symbol] = {
            'rsi': 25,              # Oversold
            'macd': 0.5,
            'macd_signal': 0.3,     # Bullish MACD
            'macd_hist': 0.2,
            'adx': 30,              # Strong trend
            'fast_ma': 105,
            'medium_ma': 103,
            'slow_ma': 100,         # Bullish MA alignment
            'volume': 5_000_000,
            'volume_ma': 2_000_000  # High volume
        }

        # Generate signal
        signal = await strategy._generate_signal(symbol)

        # Assert buy signal
        assert signal == 'buy'
```

### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_trading_workflow():
    """Test complete buy -> hold -> sell workflow"""
    # Create broker and strategy
    broker = MockAlpacaBroker(paper=True, initial_capital=100000)
    strategy = MomentumStrategy(broker=broker, symbols=['TEST'])
    await strategy.initialize()

    # Generate bullish data
    df = generate_momentum_scenario(days=100)

    # Simulate trading day
    for _, row in df.iterrows():
        await strategy.on_bar(
            symbol='TEST',
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume'],
            timestamp=row['timestamp']
        )

    # Verify position was opened
    positions = await broker.get_positions()
    assert len(positions) > 0
    assert positions[0].symbol == 'TEST'
```

### Performance Test Example

```python
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.asyncio
async def test_strategy_performance_criteria():
    """Test strategy meets performance criteria"""
    from engine.backtest_engine import BacktestEngine

    # Create backtest engine
    engine = BacktestEngine(
        strategy_class=MomentumStrategy,
        symbols=['AAPL', 'TSLA'],
        start_date='2024-01-01',
        end_date='2024-03-01',
        initial_capital=100000
    )

    # Run backtest
    results = await engine.run()

    # Verify performance criteria
    assert results['sharpe_ratio'] > 1.0, "Sharpe ratio should exceed 1.0"
    assert results['total_return'] > 0.0, "Should have positive returns"
    assert results['max_drawdown'] < 0.20, "Max drawdown should be < 20%"
    assert results['win_rate'] > 0.50, "Win rate should exceed 50%"
```

---

## Best Practices

### 1. Use Fixtures

**Good:**
```python
@pytest.fixture
async def strategy(self):
    broker = MockAlpacaBroker(paper=True)
    strategy = MomentumStrategy(broker=broker, symbols=['TEST'])
    await strategy.initialize()
    return strategy

async def test_something(self, strategy):
    # Use strategy fixture
    result = await strategy.analyze_symbol('TEST')
```

**Bad:**
```python
async def test_something(self):
    # Recreate everything in each test
    broker = MockAlpacaBroker(paper=True)
    strategy = MomentumStrategy(broker=broker, symbols=['TEST'])
    await strategy.initialize()
```

### 2. Test One Thing Per Test

**Good:**
```python
async def test_rsi_calculation(self, strategy):
    """Test RSI is calculated correctly"""
    # ... test only RSI

async def test_macd_calculation(self, strategy):
    """Test MACD is calculated correctly"""
    # ... test only MACD
```

**Bad:**
```python
async def test_all_indicators(self, strategy):
    """Test RSI, MACD, ADX, MAs, volume..."""
    # Tests too many things - hard to debug
```

### 3. Use Descriptive Names

**Good:**
```python
async def test_buy_signal_generated_when_rsi_oversold_and_macd_bullish(self):
    """Test buy signal is generated when RSI oversold and MACD bullish"""
```

**Bad:**
```python
async def test_signal(self):
    """Test signal"""
```

### 4. Test Edge Cases

```python
async def test_handles_none_indicators_gracefully(self, strategy):
    """Test strategy handles None indicators without crashing"""
    symbol = 'TEST'
    signal = await strategy._generate_signal(symbol)
    assert signal == 'neutral'  # Should return neutral, not crash

async def test_handles_insufficient_data(self, strategy):
    """Test strategy handles insufficient price history"""
    symbol = 'TEST'
    # Add only 2 bars (not enough for indicators)
    strategy.price_history[symbol] = [
        {'close': 100.0, 'volume': 1000000},
        {'close': 101.0, 'volume': 1000000}
    ]
    await strategy._update_indicators(symbol)
    # Should not crash, indicators should be None
    assert strategy.indicators[symbol].get('rsi') is None
```

### 5. Use Mocks for External Dependencies

```python
# Mock Alpaca API calls
@patch('brokers.alpaca_broker.AlpacaBroker.get_account')
async def test_with_mocked_api(self, mock_get_account):
    mock_get_account.return_value = MockAccount(
        equity=100000,
        cash=50000,
        buying_power=200000
    )
    # ... test code
```

---

## Coverage

### Generating Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=strategies --cov=brokers --cov=engine --cov=utils --cov-report=html

# View report
open htmlcov/index.html

# Generate XML for CI/CD
pytest tests/ --cov=strategies --cov=brokers --cov-report=xml

# Show missing lines in terminal
pytest tests/ --cov=strategies --cov-report=term-missing
```

### Coverage Targets

| Module | Target | Current | Status |
|--------|--------|---------|--------|
| strategies/ | 80% | - | 🟡 In Progress |
| brokers/ | 75% | - | 🟡 Planned |
| engine/ | 80% | - | 🟡 Planned |
| utils/ | 70% | - | 🟡 Planned |

---

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**Workflow:** `.github/workflows/ci.yml`

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=strategies --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Local CI Simulation

```bash
# Run same commands as CI
pytest tests/ -v --cov=strategies --cov=brokers --cov=engine --cov-report=xml

# Check coverage threshold
pytest tests/ --cov-fail-under=70
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure Python path includes project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or add to pytest.ini (already configured)
pythonpath = .
```

**2. Async Test Failures**
```python
# Ensure test is marked as async
@pytest.mark.asyncio
async def test_something(self):
    result = await some_async_function()
```

**3. Fixture Not Found**
```python
# Ensure fixture is defined before test
@pytest.fixture
async def my_fixture(self):
    return something

async def test_something(self, my_fixture):
    # Use fixture
```

**4. Mock Broker Issues**
```python
# Ensure mock broker is awaited
broker = MockAlpacaBroker(paper=True)
account = await broker.get_account()  # Must use await
```

### Debug Mode

```bash
# Run with verbose output
pytest tests/ -vv

# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x

# Run specific test with full traceback
pytest tests/unit/test_momentum_strategy.py::TestMomentumStrategy::test_initialization -vv --tb=long
```

---

## Future Enhancements

**Planned:**
1. Property-based testing with Hypothesis
2. Mutation testing for code quality
3. Performance benchmarking suite
4. Visual regression testing for plots
5. Automated test generation
6. Contract testing for API interactions
7. Chaos engineering tests for resilience

---

## Resources

**Pytest Documentation:**
- https://docs.pytest.org/
- https://pytest-asyncio.readthedocs.io/

**Coverage Documentation:**
- https://coverage.readthedocs.io/

**Testing Best Practices:**
- [Testing Strategies for Algorithmic Trading](https://www.quantstart.com/articles/testing-trading-strategies)
- [Python Testing Best Practices](https://realpython.com/python-testing/)

---

**Updated:** 2025-11-10
**Status:** Testing infrastructure complete and documented
**Next:** Run full test suite and achieve coverage targets

