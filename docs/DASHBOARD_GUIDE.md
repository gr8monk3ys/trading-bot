# Trading Dashboard Guide

**Last Updated:** 2025-11-08

## Overview

The trading bot includes **two dashboard options** for real-time monitoring:

1. **Basic Dashboard** (`scripts/dashboard.py`) - Simple text-based display
2. **Enhanced Dashboard** (`scripts/enhanced_dashboard.py`) - Beautiful Rich UI ✨ **RECOMMENDED**

---

## 🎨 Enhanced Dashboard (NEW!)

### Features

**Real-Time Monitoring:**
- 📊 **Account Summary** - Equity, cash, buying power, day P/L
- 💼 **Open Positions** - All positions with live P/L (both longs and shorts)
- ⚡ **Risk Status** - Circuit breaker, daily loss %, position concentration
- 🎯 **Strategy Status** - Active strategies and their settings
- 🏛️  **Market Status** - Open/closed indicator
- 📈 **Win/Loss Tracking** - Today's trade statistics

**Visual Highlights:**
- Color-coded P/L (green for profits, red for losses)
- Real-time updates every 5 seconds
- Clean, organized panels
- Professional terminal UI using Rich library

### Usage

```bash
# Run the enhanced dashboard
python scripts/enhanced_dashboard.py

# Or from project root
cd /Users/gr8monk3ys/code/trading-bot
python scripts/enhanced_dashboard.py
```

**Controls:**
- `q` - Quit dashboard
- `r` - Force refresh
- Auto-refreshes every 5 seconds

### Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│         🤖 LIVE TRADING DASHBOARD 🤖                │
│              2025-11-08 10:00:00                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────┬─────────────────────────┐
│  Account Summary        │    Risk Status          │
│                         │                         │
│  💰 Equity: $102,450    │  ⚡ Circuit Breaker     │
│  📊 Day P/L: +$2,450    │     ✓ Armed             │
│  💵 Cash: $45,230       │  📊 Daily Loss: 0.5%    │
│  ⚡ Buying Power: $...  │  💼 Positions: 3/10     │
│  📉 Drawdown: 2.3%      │  🎯 Max Position: 8.2%  │
│  🏛️ Market: 🟢 OPEN    │  📈 Win Rate: 65%       │
│                         │                         │
├─────────────────────────┼─────────────────────────┤
│  Open Positions (3)     │  Active Strategies      │
│                         │                         │
│  Symbol  Qty    P/L     │  🎯 Momentum: Active    │
│  📈 AAPL  10  +$245.50  │  📊 Mean Rev: Active    │
│  📈 MSFT  5   +$128.30  │  🔻 Shorts: Enabled     │
│  📉 TSLA -3   +$89.20   │  ⏱️ Multi-TF: Enabled   │
│                         │  ⚖️ Rebalance: 4h       │
└─────────────────────────┴─────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Press 'q' to quit  •  'r' to refresh  •  Auto: 5s  │
└─────────────────────────────────────────────────────┘
```

### Key Metrics Explained

**Account Summary Panel:**
- **Equity** - Total account value (cash + positions)
- **Day P/L** - Profit/loss since market open today
- **Cash** - Available cash balance
- **Buying Power** - Available capital for trading
- **Drawdown** - Percentage decline from peak equity
- **Market** - Current market status (open/closed)

**Risk Status Panel:**
- **Circuit Breaker** - Safety feature status (armed/triggered)
- **Daily Loss** - Current daily loss vs 3% max limit
- **Positions** - Number of open positions vs max (10)
- **Max Position** - Largest position as % of equity
- **Win Rate** - Percentage of winning trades today

**Open Positions Panel:**
- 📈 = Long position (profit from price rising)
- 📉 = Short position (profit from price falling)
- **Qty** - Positive = long, Negative = short
- **P/L** - Unrealized profit/loss on position
- **%** - Return percentage on position

**Active Strategies Panel:**
- Shows which strategies are currently enabled
- Indicates key features (short selling, multi-timeframe, etc.)
- Shows rebalancing frequency

---

## 📟 Basic Dashboard

### Features

Simple text-based display with:
- Current account status
- Open positions list
- Recent trades (last 10)
- Basic performance metrics

### Usage

```bash
# Run the basic dashboard
python scripts/dashboard.py
```

**Note:** The enhanced dashboard is recommended for better visualization.

---

## 🔧 Configuration

### Refresh Rate

To change the refresh interval in enhanced dashboard:

```python
# In scripts/enhanced_dashboard.py, line 54
self.refresh_interval = 5  # Change to desired seconds
```

### Paper vs Live Trading

Both dashboards connect to **paper trading** by default. To monitor live trading:

```python
# In __init__ method
self.broker = AlpacaBroker(paper=False)  # Change to False for live
```

⚠️ **Warning:** Only use live trading after extensive paper trading validation!

---

## 🚨 Circuit Breaker Alerts

The enhanced dashboard shows circuit breaker status in real-time:

- **✓ Armed** (green) - Normal operation, monitoring active
- **🚨 TRIGGERED** (red) - Daily loss limit exceeded, trading halted

When triggered:
1. Dashboard will show RED border on Risk Status panel
2. All trading stops automatically
3. Open positions are liquidated (if `auto_close_positions=True`)
4. Trading resumes next market day

---

## 📊 Understanding Position Types

### Long Positions (📈)
- **What:** You own shares, profit when price rises
- **Example:** Buy AAPL at $150, sell at $160 = $10 profit per share
- **Display:** Green P/L when profitable

### Short Positions (📉)
- **What:** You sell borrowed shares, profit when price drops
- **Example:** Short TSLA at $200, cover at $190 = $10 profit per share
- **Display:** Green P/L when price has dropped

---

## 💡 Tips for Using Dashboard

### Best Practices

1. **Monitor During Active Trading Hours**
   - Most useful when market is open (9:30 AM - 4:00 PM ET)
   - Check before/after market for pre/post-market activity

2. **Watch Risk Metrics**
   - Keep Daily Loss below 2% ideally
   - Max Position should stay below 10% of equity
   - Position count should stay manageable (5-8 max)

3. **Track Win Rate**
   - Target 50%+ win rate for profitability
   - Below 45% indicates strategy needs tuning
   - Above 60% is excellent

4. **Monitor Drawdown**
   - < 5%: Excellent
   - 5-10%: Normal
   - 10-15%: Concerning, review strategies
   - > 15%: Circuit breaker should trigger

### Common Issues

**"No open positions"**
- Normal if strategies haven't found entry signals
- Check Market status - might be closed
- Strategies may be in cooldown period (1 hour between signals)

**"Error connecting to broker"**
- Check `.env` file has correct API keys
- Verify internet connection
- Ensure Alpaca API is accessible

**Dashboard freezes**
- Press `r` to force refresh
- Restart dashboard if issue persists
- Check broker connection status

---

## 🎯 Running Dashboard Alongside Trading Bot

### Recommended Setup

**Terminal 1 - Trading Bot:**
```bash
python main.py live --strategy auto --max-strategies 3
```

**Terminal 2 - Dashboard:**
```bash
python scripts/enhanced_dashboard.py
```

This gives you:
- Live trading execution in Terminal 1
- Real-time monitoring in Terminal 2
- Ability to watch trades execute

### Screen/Tmux Setup

For long-running sessions:

```bash
# Using tmux
tmux new -s trading
# Split pane: Ctrl+B then "
# Top pane: Run trading bot
python main.py live --strategy auto

# Bottom pane: Ctrl+B then arrow down
# Run dashboard
python scripts/enhanced_dashboard.py

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t trading
```

---

## 📈 Future Enhancements

Planned features for dashboard:

- [ ] Trade history with entry/exit details
- [ ] Real-time chart integration
- [ ] Performance charts (equity curve, drawdown)
- [ ] Strategy-specific metrics
- [ ] Alert notifications (sound/desktop)
- [ ] Export data to CSV/JSON
- [ ] Web-based dashboard (HTML)

---

## 🛠️ Technical Details

### Dependencies

- **rich** - Terminal UI library
- **alpaca-py** - Broker API
- **asyncio** - Async execution

### Architecture

```
EnhancedTradingDashboard
├── initialize()          # Connect to broker, circuit breaker
├── create_layout()       # Build UI layout
│   ├── create_header()
│   ├── create_account_panel()
│   ├── create_positions_panel()
│   ├── create_risk_panel()
│   ├── create_strategies_panel()
│   └── create_footer()
└── run()                 # Main loop with auto-refresh
```

### Performance

- Minimal CPU usage (< 1%)
- Low memory footprint (~ 50MB)
- Network: ~10KB per refresh (Alpaca API calls)
- Refresh rate: 5 seconds (configurable)

---

## 📝 Example Output

### Profitable Day Example

```
💰 Equity: $103,245.50  (Green)
📊 Day P/L: +$3,245.50 (+3.25%)  (Green)
📉 Drawdown: 0.0%  (Green)

Open Positions (4):
📈 AAPL   10 shares    +$245.50
📈 MSFT   5 shares     +$128.30
📉 TSLA   -3 shares    +$89.20  (Short position)
📈 GOOGL  2 shares     +$56.80
```

### Drawdown Example

```
💰 Equity: $97,850.00  (Red)
📊 Day P/L: -$2,150.00 (-2.15%)  (Red)
📉 Drawdown: 4.8%  (Yellow)

⚠️ Circuit Breaker: 2.15% loss (approaching 3% limit)
```

### Circuit Breaker Triggered

```
🚨 CIRCUIT BREAKER TRIGGERED
Daily loss limit exceeded (-3.02%)
Trading HALTED for remainder of day
All positions liquidated
```

---

**For questions or issues, check logs in `logs/` directory or consult TODO.md for known issues.**
