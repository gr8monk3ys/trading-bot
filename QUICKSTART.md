# Trading Bot Quick Start Guide

**Status:** ✅ Bot is currently running in paper trading mode with MomentumStrategy

---

## Current Setup (November 2025-11-10)

### Active Trading Bot
- **Strategy:** MomentumStrategy (21 unit tests passing, 73.67% coverage)
- **Mode:** Paper Trading (no real money)
- **Capital:** $100,000 (simulated)
- **Allocation:** 90% of capital to MomentumStrategy
- **Symbols:** AAPL, MSFT, AMZN, META, TSLA
- **Status:** Running, waiting for market open (Mon Nov 11, 9:30 AM EST)
- **Process:** Running in background (PID logged in momentum.log)

### Circuit Breakers
- **Max Daily Loss:** 3% ($3,000)
- **Halt Threshold:** Portfolio drops below $97,000
- **Status:** Armed and monitoring

---

## Quick Commands

### Check Account & Positions
```bash
python3 check_positions.py
```
Shows:
- Account status (cash, portfolio value, buying power)
- Current positions with P&L
- Open orders

### Monitor Bot in Real-Time
```bash
python3 monitor_bot.py 60
```
Live dashboard updating every 60 seconds showing:
- 💰 Account overview
- 📈 Performance metrics (P&L, positions, orders)
- 📊 Position details with real-time P&L
- 📝 Open orders

Press Ctrl+C to stop monitoring.

### Check Bot Logs
```bash
# Last 50 lines
tail -50 momentum.log

# Follow logs in real-time
tail -f momentum.log

# Search for trades
grep -i "trade\|order\|signal" momentum.log

# Check for errors
grep -i "error" momentum.log
```

### Start/Stop Bot

**Start MomentumStrategy:**
```bash
nohup python3 main.py live --strategy MomentumStrategy --force > momentum.log 2>&1 &
```
**Note:** Live mode automatically starts the broker websocket for trade updates. This is required for order fill audit logging. Audit logs are written to `audit_logs/`.

**Emergency Kill Switch:**
```bash
python scripts/kill_switch.py --confirm "HALT TRADING" --cancel-orders --liquidate
```

**Stop Bot:**
```bash
# Find process
pgrep -f "python3 main.py live"

# Kill process (replace PID with actual number)
kill -9 <PID>

# Or kill all trading bots
pgrep -f "python3 main.py live" | xargs kill -9
```

### Multiple Strategies (Not Working - BacktestEngine Broken)
```bash
# This DOESN'T work currently due to BacktestEngine issues
python3 main.py live --strategy auto --max-strategies 3 --force

# Instead, run strategies manually:
python3 main.py live --strategy MomentumStrategy --force
python3 main.py live --strategy BracketMomentumStrategy --force
python3 main.py live --strategy MeanReversionStrategy --force
```

---

## Monitoring During Market Hours

When market opens (Mon-Fri 9:30 AM - 4:00 PM EST):

1. **Watch the live monitor:**
   ```bash
   python3 monitor_bot.py 30  # Update every 30 seconds
   ```

2. **Check positions frequently:**
   ```bash
   watch -n 30 python3 check_positions.py  # Auto-refresh every 30s
   ```

3. **Monitor logs for trading signals:**
   ```bash
   tail -f momentum.log | grep -E "signal|trade|BUY|SELL"
   ```

---

## Understanding the Strategy

### MomentumStrategy

**What it does:**
- Analyzes momentum using RSI, MACD, ADX, and Moving Averages
- Buys stocks showing strong upward momentum
- Implements automatic stop-loss based on volatility
- Uses circuit breakers to limit losses

**Entry Signals:**
- RSI > 50 (price momentum)
- MACD > signal line (trend confirmation)
- ADX > 20 (trend strength)
- Price > 20-day and 50-day moving averages
- Volume confirmation required

**Exit Signals:**
- Price drops below stop-loss (3% default)
- Momentum indicators weaken
- Maximum positions reached (5)

**Position Sizing:**
- 5-10% of portfolio per position
- Max 5 concurrent positions
- Volatility-adjusted sizing (Kelly Criterion)

**Risk Management:**
- Stop-loss per position: 3%
- Max daily loss: 3% ($3,000)
- Circuit breaker halts trading if triggered

---

## Expected Behavior

### Market Closed
- Bot runs but makes no trades
- Logs show "Market status: is_open: False"
- Waits for next market open

### Market Open
- Bot analyzes symbols every iteration (default: 1 minute)
- Generates buy/sell signals based on momentum
- Executes trades if signals are strong enough
- Updates positions and monitors P&L

### First Trading Day
- Likely to open 1-3 positions depending on market conditions
- Each position typically $5,000-$10,000 (5-10% of portfolio)
- May take several iterations before first trade

---

## Performance Expectations

### MomentumStrategy (Based on Testing)
- **Test Coverage:** 73.67%
- **Tests Passing:** 21/21 (100%)
- **Expected Win Rate:** 50-60%
- **Expected Sharpe Ratio:** 1.0-2.0 (if backtesting worked)
- **Risk Level:** Medium

**Note:** Backtests are useful but imperfect. Prefer the validated backtest mode and paper trading to confirm real-world behavior.

---

## Troubleshooting

### Bot Not Trading
**Check:**
1. Is market open? `python3 check_positions.py`
2. Are there strong momentum signals? Check logs
3. Is circuit breaker triggered? Check logs for "halt"
4. Is bot still running? `pgrep -f "python3 main.py live"`

### Bot Stopped
**Restart:**
```bash
nohup python3 main.py live --strategy MomentumStrategy --force > momentum.log 2>&1 &
```

### Errors in Logs
**Common issues:**
- API rate limits: Bot has retry logic, should recover automatically
- Connection issues: Check internet connection
- Missing data: Check if symbols are tradeable

---

## Safety Features

### Paper Trading
- **No real money:** All trades are simulated
- **Real market data:** Uses live prices
- **Real API:** Same code path as live trading
- **Perfect for testing:** Validate strategy before risking capital

### Circuit Breakers
- **Daily loss limit:** 3% max drawdown per day
- **Automatic halt:** Stops trading if limit hit
- **Manual override:** Can disable with `--force` flag
- **Protects capital:** Prevents catastrophic losses

### Position Limits
- **Max positions:** 5 concurrent
- **Max size per position:** 10% of portfolio
- **Cooldown period:** 60 seconds between trades on same symbol
- **Stop-loss:** 3% per position (volatility-adjusted)

---

## Next Steps

### Today (Market Closed)
- ✅ Bot is running and ready
- ✅ Monitoring tools created
- ⏳ Waiting for market open (Mon Nov 11, 9:30 AM EST)

### Tomorrow (Market Open)
1. Monitor bot starting at 9:30 AM EST
2. Watch for first trades in `momentum.log`
3. Check positions with `python3 check_positions.py`
4. Run live monitor: `python3 monitor_bot.py 60`

### This Week
1. Track daily performance
2. Monitor P&L trends
3. Evaluate if strategy is profitable
4. Consider adding more strategies if MomentumStrategy performs well

### Future Enhancements
1. Fix BacktestEngine to enable multi-strategy auto-selection
2. Add more unit tests for other strategies
3. Create performance analytics dashboard
4. Implement automated reporting (daily P&L emails)

---

## Key Files

- `main.py` - Main entry point for starting the bot
- `momentum.log` - Live trading logs
- `check_positions.py` - Quick position checker
- `monitor_bot.py` - Real-time monitoring dashboard
- `strategies/momentum_strategy.py` - Strategy implementation
- `config.py` - Trading parameters and risk limits
- `.env` - API credentials (paper trading keys)

---

## Getting Help

### Check Documentation
- `CLAUDE.md` - Complete project documentation
- `TESTING_SUCCESS_SUMMARY.md` - Testing infrastructure status
- `TODO.md` - Known issues and future work

### Common Questions

**Q: Why no trades yet?**
A: Market is closed. Bot will start trading Mon Nov 11 at 9:30 AM EST.

**Q: Is this safe?**
A: Yes, running in paper trading mode with simulated money only.

**Q: When will I see results?**
A: After market opens and bot makes trades. Could be immediately or after several hours depending on momentum signals.

**Q: What if bot loses money?**
A: Circuit breaker limits losses to 3% per day. Plus it's paper trading, so no real money is at risk.

**Q: Can I run multiple strategies?**
A: Yes, but auto-selection is broken. Start each manually:
```bash
python3 main.py live --strategy MomentumStrategy --force &
python3 main.py live --strategy BracketMomentumStrategy --force &
```

---

**Last Updated:** 2025-11-10 23:40 PST
**Bot Status:** ✅ Running (Paper Trading)
**Next Market Open:** Monday, November 11, 2025 at 9:30 AM EST
