# 🤖 Professional Trading Bot

**Production-ready algorithmic trading system with advanced strategies, real-time monitoring, and comprehensive risk management.**

Built for serious traders who want systematic, data-driven trading with institutional-grade features.

---

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Test connection
python tests/test_connection.py

# 3. Launch interactive setup
python quickstart.py

# 4. Start trading!
# (The quickstart script will guide you through strategy selection)
```

**Status**: ✅ Fully functional with $100,000 paper trading account

---

## 📈 Backtest Results (2024)

| Strategy | Total Return | Annualized | Sharpe Ratio | Max Drawdown | Trades |
|----------|-------------|------------|--------------|--------------|--------|
| MomentumStrategyBacktest | **+42.68%** | 42.71% | 2.00 | 2.44% | 9 |
| SimpleMACrossover | +2.73% | 5.58% | 0.59 | 3.62% | 11 |
| SPY Benchmark | +24.45% | 24.45% | ~0.80 | ~5-6% | - |

**Key Finding**: MomentumStrategyBacktest outperformed SPY by **+18.23%** with lower drawdown.

---

## 🚀 Features

### Core Trading System
- ✅ **6 Production Strategies**: Momentum, Mean Reversion, Bracket, Ensemble, Pairs Trading, Extended Hours
- ✅ **Real-Time Execution**: WebSocket streaming for instant market data
- ✅ **Fractional Shares**: Trade expensive stocks with any capital
- ✅ **Advanced Orders**: Bracket orders with automatic stop-loss & take-profit
- ✅ **Short Selling**: Profit from declining stocks
- ✅ **Multi-Timeframe Analysis**: Confirm signals across 1min, 5min, 1hour
- ✅ **Extended Hours Trading**: Pre-market (4AM-9:30AM) & After-hours (4PM-8PM)
- ✅ **24/5 Overnight Trading**: Blue Ocean ATS overnight sessions (8PM-4AM ET)
- ✅ **Market-Neutral Strategies**: Pairs trading with cointegration testing
- ✅ **Cryptocurrency Trading**: 20 crypto pairs with 24/7 support
- ✅ **Options Trading**: Covered calls, cash-secured puts, Greeks analysis
- ✅ **Dollar-Based Orders**: Notional orders for precise position sizing

### Risk Management (Institutional Grade)
- ✅ **Circuit Breaker**: Auto-halt trading at 3% daily loss
- ✅ **Position Limits**: Maximum 5% per position (prevents over-concentration)
- ✅ **Realistic Slippage**: Backtest with bid-ask spread & market impact
- ✅ **Trailing Stops**: Lock in profits as positions move in your favor
- ✅ **Kelly Criterion**: Mathematically optimal position sizing

###Portfolio Management
- ✅ **Auto-Rebalancing**: Maintain target allocations (equal weight or custom)
- ✅ **Correlation Tracking**: Avoid over-concentration in correlated assets
- ✅ **Real-Time Monitoring**: Live dashboard with positions & P/L
- ✅ **Performance Analytics**: Sharpe ratio, max drawdown, win rate, profit factor

### Monitoring & Notifications
- ✅ **Live Dashboard**: Real-time positions, trades, and metrics
- ✅ **Trade Notifications**: Slack/Email/Discord/Telegram alerts
- ✅ **Performance Tracking**: SQLite database with full trade history
- ✅ **Portfolio History API**: Equity curves and drawdown tracking
- ✅ **Daily Summaries**: Automated performance reports

### Machine Learning (Optional)
- ✅ **LSTM Price Prediction**: Neural network forecasting with PyTorch
- ✅ **FinBERT Sentiment**: News sentiment analysis for alpha generation
- ✅ **DQN Reinforcement Learning**: Self-improving trading agent
- ✅ **Lazy Loading**: ML dependencies only loaded when used

---

## 📊 Strategies

### 1. Momentum Strategy
**Best for**: Trending markets
**Logic**: Buy stocks showing strong upward momentum (RSI + MACD confirmation)
**Risk**: Medium
**Holding**: 1-3 days

```bash
python live_trader.py --strategy momentum --symbols AAPL MSFT GOOGL
```

### 2. Mean Reversion Strategy
**Best for**: Range-bound markets
**Logic**: Buy oversold stocks (RSI < 30), sell when back to mean
**Risk**: Medium-High
**Holding**: 1-5 days

```bash
python live_trader.py --strategy mean_reversion --symbols SPY QQQ
```

### 3. Bracket Momentum Strategy
**Best for**: Active trading
**Logic**: Momentum entries with automatic bracket orders (TP + SL)
**Risk**: Medium
**Holding**: Hours to days

```bash
python live_trader.py --strategy bracket_momentum --symbols TSLA NVDA
```

### 4. Ensemble Strategy ⭐ NEW
**Best for**: All market conditions
**Logic**: Combines mean reversion, momentum, and trend following with regime detection
**Sharpe Ratio**: 0.95-1.25 (highest combined performance)
**Risk**: Medium
**Holding**: Varies based on market regime

```bash
python examples/ensemble_strategy_example.py
```

**Features**:
- Automatic market regime detection (trending/ranging/volatile)
- Intelligent strategy weighting based on conditions
- Requires 60% agreement across sub-strategies
- Adapts to changing market conditions

### 5. Pairs Trading Strategy ⭐ NEW
**Best for**: Market-neutral, all conditions
**Logic**: Statistical arbitrage on cointegrated stock pairs
**Sharpe Ratio**: 0.80-1.20 (highest potential from research)
**Risk**: Medium (market-neutral = hedged)
**Holding**: Days to weeks

```bash
python examples/pairs_trading_example.py
```

**Features**:
- Cointegration testing (Engle-Granger method)
- Long one stock, short the other (market-neutral)
- Z-score based entry/exit signals
- Lower correlation to market movements

**Common Pairs**:
- KO/PEP (Coca-Cola / PepsiCo)
- JPM/BAC (JPMorgan / Bank of America)
- WMT/TGT (Walmart / Target)

### 6. Extended Hours Trading ⭐ NEW
**Sessions**: Pre-market (4AM-9:30AM), After-hours (4PM-8PM)
**Best for**: Gap trading, earnings reactions
**Logic**: News-driven opportunities with conservative risk management

```bash
python examples/extended_hours_trading_example.py
```

**Features**:
- Pre-market gap trading on overnight news
- After-hours earnings reaction trading
- Automatic position size reduction (50%)
- Limit orders only (safer for low liquidity)
- Spread validation and slippage protection

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Alpaca paper trading account (free)
- 10 minutes

### Setup

```bash
# 1. Clone repository
git clone <your-repo>
cd trading-bot

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Alpaca credentials
# Edit .env file with your API keys:
# ALPACA_API_KEY="your_key_here"
# ALPACA_SECRET_KEY="your_secret_here"

# 5. Test connection
python tests/test_connection.py
```

**Get Alpaca API Keys**:
1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Create a paper trading account (free)
3. Generate API keys from dashboard
4. Add to `.env` file

---

## 💰 Usage

### Option 1: Interactive Setup (Recommended)

```bash
python quickstart.py
```

Walks you through:
1. Strategy selection
2. Stock selection
3. Parameter configuration
4. Starts trading automatically

### Option 2: Command Line

```bash
# Basic usage
python live_trader.py --strategy momentum --symbols AAPL MSFT

# Advanced configuration
python live_trader.py \
    --strategy mean_reversion \
    --symbols AAPL MSFT GOOGL AMZN \
    --position-size 0.15 \
    --stop-loss 0.03 \
    --take-profit 0.06
```

### Option 3: View Dashboard (Monitor Running Bot)

```bash
# In a separate terminal
python dashboard.py
```

Shows real-time:
- Open positions & P/L
- Recent trades
- Performance metrics
- Account status

---

## 📈 Performance Tracking

### View Performance Report

```python
from utils.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
print(tracker.get_performance_report(starting_equity=100000))
```

**Metrics Calculated**:
- Total return & annualized return
- Sharpe ratio, Sortino ratio, Calmar ratio
- Maximum drawdown & recovery factor
- Win rate & profit factor
- Average win/loss
- Trade statistics

### Database

All trades stored in SQLite: `data/trading_history.db`

```bash
# Query trades
sqlite3 data/trading_history.db "SELECT * FROM trades ORDER BY exit_time DESC LIMIT 10;"
```

---

## 🔔 Notifications

### Slack Setup (Recommended)

1. Create Slack webhook: https://api.slack.com/messaging/webhooks
2. Add to `.env`:
   ```
   SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ```

3. Restart bot - you'll get alerts for:
   - Trade executions
   - Circuit breaker triggers
   - Daily summaries
   - Position alerts

### Email Setup

Add to `.env`:
```
EMAIL_NOTIFICATIONS=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_TO=your-email@gmail.com
```

---

## 🧪 Examples

### Multi-Timeframe Strategy

```python
python examples/multi_timeframe_strategy_example.py
```

Analyzes 1min, 5min, and 1hour timeframes simultaneously for high-confidence signals.

### Short Selling

```python
python examples/short_selling_strategy_example.py
```

Profit from declining stocks with proper risk management.

### Portfolio Rebalancing

```python
python examples/portfolio_rebalancing_example.py
```

Automatically maintain 25% allocation across 4 stocks.

### Kelly Criterion Position Sizing

```python
python examples/kelly_criterion_example.py
```

Mathematically optimal position sizing based on your edge.

---

## 📊 Advanced Indicators Library ⭐ NEW

Comprehensive technical analysis library with 30+ indicators organized by category.

### Usage

```python
from utils.indicators import TechnicalIndicators

# Initialize with price data
ind = TechnicalIndicators(
    high=high_prices,
    low=low_prices,
    close=close_prices,
    volume=volumes,
    timestamps=timestamps
)

# Calculate indicators
rsi = ind.rsi(period=14)
vwap = ind.vwap()
adx, plus_di, minus_di = ind.adx_di(period=14)
bb_upper, bb_middle, bb_lower = ind.bollinger_bands(period=20, std=2.0)
```

### Available Indicators

**Trend Indicators**:
- SMA/EMA (Simple/Exponential Moving Averages)
- MACD (Moving Average Convergence Divergence)
- ADX with Directional Indicators (+DI, -DI)
- Parabolic SAR (Stop and Reverse)

**Momentum Indicators**:
- RSI (Relative Strength Index)
- Stochastic Oscillator (%K, %D)
- Stochastic RSI (faster signals)
- CCI (Commodity Channel Index)
- Williams %R
- ROC (Rate of Change)

**Volatility Indicators**:
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channels
- Standard Deviation

**Volume Indicators**:
- VWAP (Volume Weighted Average Price) - institutional benchmark
- OBV (On-Balance Volume)
- Volume SMA
- MFI (Money Flow Index)

**Support/Resistance**:
- Pivot Points (PP, R1-R3, S1-S3)
- Fibonacci Retracements

### Quick Analysis Functions

```python
from utils.indicators import analyze_trend, analyze_momentum, analyze_volatility

# Trend analysis
trend = analyze_trend(close, high, low)
# Returns: direction, strength, ADX, SMAs, etc.

# Momentum analysis
momentum = analyze_momentum(close, high, low)
# Returns: condition (overbought/oversold), RSI, Stochastic

# Volatility analysis
volatility = analyze_volatility(close, high, low)
# Returns: state (squeeze/expansion), ATR, Bollinger Bands
```

---

## 📁 Project Structure

```
trading-bot/
├── brokers/
│   ├── alpaca_broker.py          # Alpaca API integration + crypto + portfolio history
│   ├── backtest_broker.py        # Backtesting with slippage
│   ├── order_builder.py          # Advanced order types + notional orders
│   └── options_broker.py         # ⭐ Options trading with Greeks
├── strategies/
│   ├── base_strategy.py          # Base class with safety features
│   ├── momentum_strategy.py      # Momentum following
│   ├── mean_reversion_strategy.py # Buy oversold, sell overbought
│   ├── bracket_momentum_strategy.py # Bracket orders
│   ├── ensemble_strategy.py      # Multi-strategy combination
│   └── pairs_trading_strategy.py # Market-neutral stat arb
├── ml/                           # ⭐ Machine Learning
│   ├── lstm_predictor.py         # LSTM price prediction
│   └── rl_agent.py               # DQN reinforcement learning
├── utils/
│   ├── indicators.py             # 30+ technical indicators library
│   ├── extended_hours.py         # Pre-market, after-hours & overnight trading
│   ├── circuit_breaker.py        # Daily loss protection
│   ├── multi_timeframe.py        # Multi-timeframe analysis
│   ├── portfolio_rebalancer.py   # Auto-rebalancing
│   ├── kelly_criterion.py        # Optimal position sizing
│   ├── performance_tracker.py    # Trade logging & metrics
│   ├── notifier.py               # Slack/Email/Discord/Telegram alerts
│   ├── websocket_manager.py      # ⭐ Real-time WebSocket streaming
│   ├── database.py               # ⭐ SQLite trade storage
│   ├── news_sentiment.py         # ⭐ FinBERT sentiment analysis
│   ├── crypto_utils.py           # ⭐ Crypto symbol utilities
│   └── market_regime.py          # Market regime detection
├── examples/                     # Example strategies
├── tests/                        # Test suites (2400+ tests)
├── .planning/                    # Feature roadmap & planning docs
├── live_trader.py                # Main trading launcher
├── dashboard.py                  # Real-time monitoring
├── quickstart.py                 # Interactive setup
└── config.py                     # Configuration
```

---

## 🛡️ Safety Features

### Circuit Breaker
- **Triggers at**: 3% daily loss
- **Action**: Immediately closes all positions & halts trading
- **Reset**: Automatic at market open next day

### Position Size Limits
- **Max per position**: 5% of portfolio
- **Max positions**: Configurable (default: 3)
- **Fractional shares**: Enabled for precise sizing

### Risk Monitoring
- **Real-time P/L tracking**: Every position monitored
- **Correlation checks**: Avoid concentrated bets
- **VaR calculations**: Understand portfolio risk
- **Drawdown alerts**: Notified of significant drops

---

## 📊 Backtesting

### Run Backtest

```python
from strategies.momentum_strategy import MomentumStrategy
from brokers.backtest_broker import BacktestBroker

# Initialize backtest broker with realistic slippage
broker = BacktestBroker(
    initial_balance=100000,
    slippage_bps=5.0,  # 5 basis points
    spread_bps=3.0     # 3 basis point bid-ask spread
)

# Run strategy
strategy = MomentumStrategy(broker=broker, symbols=['AAPL', 'MSFT'])
await strategy.backtest(start_date='2024-01-01', end_date='2024-12-31')
```

**Realistic Features**:
- Bid-ask spread simulation
- Market impact modeling
- Partial fill simulation
- Prevents look-ahead bias

---

## ⚙️ Configuration

Edit `config.py` or `.env` for:

### Trading Parameters
```python
POSITION_SIZE = 0.10        # 10% of capital per trade
MAX_POSITION_SIZE = 0.05    # 5% max per position
STOP_LOSS = 0.02           # 2% stop loss
TAKE_PROFIT = 0.05         # 5% take profit
```

### Risk Parameters
```python
MAX_DAILY_LOSS = 0.03      # 3% max daily loss (circuit breaker)
MAX_PORTFOLIO_RISK = 0.02  # 2% max portfolio risk
MAX_CORRELATION = 0.7      # Max position correlation
```

### Symbols
```python
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
```

---

## 🐛 Troubleshooting

### Connection Issues
```bash
# Test your Alpaca connection
python tests/test_connection.py

# Check .env file has correct keys
cat .env
```

### Module Not Found
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### No Trades Executing
- Check if market is open (dashboard shows status)
- Verify symbols are tradeable
- Check logs in `logs/` directory
- Ensure buying power available

### Performance Issues
- Reduce number of symbols
- Increase update interval
- Check internet connection
- Review Alpaca API rate limits

---

## 📚 Advanced Topics

### Custom Strategy Development

```python
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    async def on_bar(self, symbol, open, high, low, close, volume, timestamp):
        # Your logic here
        pass
```

See `strategies/base_strategy.py` for full API.

### Parameter Optimization

```python
# Grid search over parameters
for position_size in [0.05, 0.10, 0.15]:
    for stop_loss in [0.01, 0.02, 0.03]:
        # Run backtest with these parameters
        # Track performance
        # Find optimal combination
```

### Machine Learning Integration

Add ML signals to any strategy:

```python
# In your strategy
ml_signal = self.ml_model.predict(features)
if ml_signal > 0.7 and technical_signal == 'buy':
    await self._execute_buy(symbol, price)
```

---

## 📝 Logs

All activity logged to:
- **Console**: Real-time output
- **File**: `logs/trading_YYYYMMDD_HHMMSS.log`
- **Database**: `data/trading_history.db`

---

## 🤝 Contributing

This is a personal trading system, but ideas welcome!

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

## ⚠️ Disclaimer

**FOR EDUCATIONAL AND PAPER TRADING USE ONLY**

- This software is provided "as is" without warranty
- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always paper trade first (months minimum)
- Never risk money you can't afford to lose
- Consult a financial advisor before live trading

**The authors are not responsible for any financial losses incurred through use of this software.**

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- **Alpaca Markets**: API & paper trading platform
- **TA-Lib**: Technical analysis library
- **NumPy/Pandas**: Data analysis

---

## 📞 Support

**Issues**: Open a GitHub issue
**Improvements**: Submit a pull request
**Questions**: Check examples/ directory first

---

**Built with ❤️ for systematic traders**

**Remember**: Trade smart, manage risk, stay disciplined. 🎯
