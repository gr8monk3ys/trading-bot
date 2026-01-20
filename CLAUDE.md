# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Algorithmic trading bot built on Alpaca Trading API with async Python architecture.

**Core Stack:** Python 3.10+, asyncio, pandas, numpy, TA-Lib, pytest-asyncio

**Version:** 3.0.0

**Validated Components:**
- MomentumStrategy: RSI/MACD trend following with trailing stops (backtested, paper trading validated)
- MomentumStrategyBacktest: Daily-data optimized variant (+42.68% return in 2024 backtest)
- AdaptiveStrategy: Regime-switching coordinator (auto-selects momentum vs mean reversion)
- BacktestEngine: Full backtesting with `run_backtest()` method, slippage modeling
- BacktestBroker: Mock broker with async wrappers for strategy compatibility
- CircuitBreaker: Daily loss protection (98.67% test coverage)
- RiskManager: VaR, correlation limits, position sizing (91.21% test coverage)
- OrderBuilder: All Alpaca order types (bracket, OCO, trailing stop)
- MarketRegimeDetector: Bull/bear/sideways/volatile detection
- PerformanceMetrics: Sharpe, Sortino, Calmar ratios, win rate, profit factor

**Backtest Results (2024):**
- MomentumStrategyBacktest: +42.68% return, 2.0 Sharpe, 2.44% max drawdown
- SPY Benchmark: +24.45% (strategy outperformed by +18%)

**Untested/Needs Validation:**
- BracketMomentumStrategy, EnsembleStrategy, ExtendedHoursStrategy: Need validation
- PairsTradingStrategy: Market-neutral stat arb (statsmodels included)
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

### Linting & Formatting
```bash
# Format code with black
black strategies/ brokers/ engine/ utils/

# Lint with ruff
ruff check strategies/ brokers/ engine/ utils/

# Type checking (lenient mode configured)
mypy strategies/ brokers/ engine/ utils/
```

### Running the Bot
```bash
# Adaptive strategy (recommended - auto-switches based on market regime)
python run_adaptive.py

# Adaptive with custom symbols
python run_adaptive.py --symbols AAPL,MSFT,GOOGL,NVDA,TSLA

# Check market regime only (no trading)
python run_adaptive.py --regime-only

# Traditional momentum strategy
python main.py live --strategy MomentumStrategy --force

# Backtest-optimized momentum (for daily data backtesting)
python main.py live --strategy MomentumStrategyBacktest --force

# Background with logging
nohup python3 run_adaptive.py > adaptive_trading.log 2>&1 &
```

### Backtesting
```bash
# Run backtest with any strategy
python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-12-31

# Quick 6-month backtest
python main.py backtest --strategy SimpleMACrossover --start-date 2024-01-01 --end-date 2024-06-30

# Realistic backtest with adaptive strategy
python run_adaptive.py --backtest --start 2024-01-01 --end 2024-12-31
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

**strategies/adaptive_strategy.py** - Regime-switching coordinator:
- Detects market regime (bull/bear/sideways/volatile)
- Routes to MomentumStrategy in trending markets
- Routes to MeanReversionStrategy in ranging markets
- Adjusts position sizes based on volatility

**utils/market_regime.py** - Market regime detection:
- Uses SMA50/SMA200 crossover for trend direction
- ADX for trend strength (>25 = trending, <20 = ranging)
- Returns recommended strategy and position multiplier

**utils/realistic_backtest.py** - Backtest with realistic costs:
- Slippage modeling (0.4% per trade)
- Bid-ask spread (0.1%)
- Shows gross vs net returns
- Critical for honest performance expectations

**engine/backtest_engine.py** - Backtesting engine:
- `run_backtest(strategy_class, symbols, start_date, end_date)` - Main backtest method
- Uses BacktestBroker for simulated trading
- Day-by-day simulation with realistic slippage
- Returns equity curve, trades, and performance metrics

**brokers/backtest_broker.py** - Mock broker for backtesting:
- Slippage modeling (5 bps + market impact)
- Async wrappers for strategy compatibility (`get_account()`, `submit_order_advanced()`)
- Position tracking and P&L calculation
- Partial fill simulation for large orders

**engine/performance_metrics.py** - Performance analysis:
- Sharpe, Sortino, Calmar ratios
- Max drawdown and recovery factor
- Win rate, profit factor, average win/loss
- Generates insights based on metrics

**engine/walk_forward.py** - Walk-forward validation:
- Detects overfitting (compares in-sample vs out-of-sample)
- Rolling train/test windows across market conditions
- Industry standard: OOS < 50% of IS performance = overfit

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

## Profitability Features

### Trailing Stops (MomentumStrategy)
Instead of fixed 5% take-profit, trails winners:
- Activates after 2% profit
- Trails peak price by 2%
- Captures 10%+ moves instead of always exiting at 5%

### Market Regime Detection
Automatically detects market conditions (see `utils/market_regime.py`):
| Regime | Detection | Strategy Used | Position Mult |
|--------|-----------|---------------|---------------|
| BULL | SMA50 > SMA200, ADX > 25 | Momentum (long) | 1.2x |
| BEAR | SMA50 < SMA200, ADX > 25 | Momentum (short) | 0.8x |
| SIDEWAYS | ADX < 20 | Mean Reversion | 1.0x |
| VOLATILE | ATR > 3% of price | Defensive | 0.5x |

### Realistic Backtesting
Always use `RealisticBacktester` for honest results:
```python
from utils.realistic_backtest import RealisticBacktester, print_backtest_report

backtester = RealisticBacktester(broker, strategy)
results = await backtester.run(start_date, end_date)
print_backtest_report(results)
# Shows: Gross return: +8%, Net return: +5%, Cost drag: 3%
```

### Expected Impact
| Feature | Estimated Benefit |
|---------|-------------------|
| Market Regime Detection | +10-15% by not fighting trends |
| Trailing Stops | +15-25% on winning trades |
| Kelly Criterion Sizing | +4-6% from optimal leverage |
| Volatility Regime | +5-8% from adaptive risk |

## Code Style

From `.windsurfrules`:
- Functional programming preferred; avoid unnecessary classes
- Vectorized pandas/numpy operations over explicit loops
- Method chaining for data transformations
- PEP 8 style guidelines
- Descriptive variable names reflecting data content

## Critical Gotchas

1. **Async context**: All broker operations need `await`
2. **NumPy version**: Pinned to `>=1.24.0,<3.0.0` for compatibility
3. **Market hours**: Bot won't run if market closed unless `--force` flag used
4. **Paper vs Live**: Controlled by `PAPER` env var; paper is default
5. **Strategy discovery**: Must inherit `BaseStrategy` and be in `strategies/` directory
6. **pytest asyncio**: Uses `asyncio_mode = "auto"` - no need for `@pytest.mark.asyncio` decorator

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
