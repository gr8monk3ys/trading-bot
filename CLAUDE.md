# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Algorithmic trading bot built on Alpaca Trading API with async Python architecture.

**Core Stack:** Python 3.10, asyncio, pandas, numpy, TA-Lib, pytest-asyncio

**Validated Components:**
- MomentumStrategy: RSI/MACD trend following (backtested, paper trading validated)
- CircuitBreaker: Daily loss protection (98.67% test coverage)
- RiskManager: VaR, correlation limits, position sizing (91.21% test coverage)
- OrderBuilder: All Alpaca order types (bracket, OCO, trailing stop)

**Untested/Broken:**
- MeanReversionStrategy, BracketMomentumStrategy, EnsembleStrategy, ExtendedHoursStrategy: Need validation
- PairsTradingStrategy: Requires statsmodels dependency
- Deleted: MLPredictionStrategy, SentimentStockStrategy, OptionsStrategy

## Key Commands

### Environment Setup
```bash
conda create -n trader python=3.10 && conda activate trader
pip install -r requirements.txt
# TA-Lib (macOS): brew install ta-lib && pip install ta-lib
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_risk_manager.py -v

# Run single test by name
pytest -k "test_check_halts_at_daily_loss_limit" -v

# Run with coverage
pytest tests/ --cov=strategies --cov=utils --cov-report=html

# Test API connection
python tests/test_connection.py
```

### Running the Bot
```bash
# Paper trading (recommended)
python main.py live --strategy MomentumStrategy --force

# Background with logging
nohup python3 main.py live --strategy MomentumStrategy --force > paper_trading.log 2>&1 &

# Backtest
python main.py backtest --strategy MomentumStrategy --start-date 2024-01-01 --plot

# Auto-select strategies
python main.py live --strategy auto --max-strategies 3
```

### Docker
```bash
docker-compose build trading-bot-paper
docker-compose up -d trading-bot-paper
docker-compose logs -f trading-bot-paper
```

## Architecture

### Data Flow
```
main.py → StrategyManager → [Strategy] → AlpacaBroker → Alpaca API
                ↓                ↓
         BacktestEngine    RiskManager
                ↓                ↓
        PerformanceMetrics  CircuitBreaker
```

### Core Components

**strategies/base_strategy.py** - Abstract base all strategies inherit:
- `analyze_symbol(symbol)` → returns signal
- `execute_trade(symbol, signal)` → executes via broker
- Kelly Criterion position sizing, volatility-based stop-loss
- Price history management in `self.price_history` dict

**brokers/alpaca_broker.py** - Async Alpaca API wrapper:
- All methods are async (must use `await`)
- `@retry_with_backoff` decorator for transient failures
- Supports paper (`PAPER=True`) and live trading

**brokers/order_builder.py** - Fluent order construction:
```python
order = (OrderBuilder('AAPL', 'buy', 100)
         .market()
         .bracket(take_profit=180, stop_loss=150)
         .gtc()
         .build())
await broker.submit_order_advanced(order)
```

**utils/circuit_breaker.py** - Daily loss protection:
- Halts trading at configurable loss threshold (default 3%)
- Rapid drawdown detection (2% from intraday peak)
- Auto-closes positions on trigger

**strategies/risk_manager.py** - Position/portfolio risk:
- VaR and Expected Shortfall calculations
- Correlation-based position rejection
- Volatility-adjusted sizing

**engine/strategy_manager.py** - Multi-strategy orchestration:
- Auto-discovers strategies in `strategies/` directory
- Sharpe-ratio weighted capital allocation
- Periodic rebalancing

### Configuration (config.py)

Three parameter groups:
- `TRADING_PARAMS`: Position sizing, stop-loss, take-profit
- `RISK_PARAMS`: VaR confidence, correlation limits, drawdown threshold
- `TECHNICAL_PARAMS`: SMA periods, RSI thresholds

### Environment Variables (.env)
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
PAPER=True
```

## Implementation Patterns

### Async/Await
All broker operations are async:
```python
# CORRECT
positions = await self.broker.get_positions()
account = await self.broker.get_account()

# WRONG - will cause runtime errors
positions = self.broker.get_positions()
```

### Creating New Strategies
1. Inherit from `BaseStrategy`
2. Implement `analyze_symbol(symbol)` and `execute_trade(symbol, signal)`
3. Set `NAME` class attribute
4. Place in `strategies/` directory (auto-discovered)

```python
class MyStrategy(BaseStrategy):
    NAME = "MyStrategy"

    async def analyze_symbol(self, symbol: str) -> dict:
        # Return signal dict with 'action', 'confidence', etc.
        pass

    async def execute_trade(self, symbol: str, signal: dict):
        # Build and submit order
        pass
```

### Price History
Strategies must populate `self.price_history[symbol]` before calling `_calculate_volatility()`:
```python
self.price_history[symbol] = prices[-self.price_history_window:]
volatility = self._calculate_volatility(symbol)
```

### Circular Import Avoidance
OrderBuilder must be imported inside methods, not at module level:
```python
# In brokers/alpaca_broker.py
async def submit_order_advanced(self, order_request):
    from brokers.order_builder import OrderBuilder  # Inside method
    # ...
```

## Code Style

From `.windsurfrules`:
- Functional programming preferred; avoid unnecessary classes
- Vectorized pandas/numpy operations over explicit loops
- Method chaining for data transformations
- PEP 8 style guidelines
- Descriptive variable names reflecting data content

## Critical Gotchas

1. **Async context**: All broker operations need `await`
2. **NumPy version**: Pinned to <2.0.0 for compatibility
3. **Market hours**: Bot won't run if market closed unless `--force` flag used
4. **Paper vs Live**: Controlled by `PAPER` env var; paper is default
5. **Strategy discovery**: Must inherit `BaseStrategy` and be in `strategies/` directory

## Test Organization

```
tests/
├── unit/
│   ├── conftest.py          # Shared fixtures
│   ├── test_risk_manager.py # 64 tests, 91% coverage
│   └── test_circuit_breaker.py # 43 tests, 99% coverage
└── test_connection.py        # API connectivity test
```

Fixtures in `conftest.py`:
- `mock_broker`: AsyncMock with default account values
- `sample_price_history`: 30-point price series
- `generate_correlated_price_histories()`: For correlation tests
