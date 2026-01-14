# Paper Trading Guide

**Status:** ACTIVE - Paper trading started 2025-11-10
**Strategy:** MomentumStrategy (Simplified)
**Mode:** Paper Trading (Alpaca)

---

## Current Setup

**Bot Configuration:**
- Strategy: MomentumStrategy
- Symbols: AAPL, MSFT, AMZN, META, TSLA
- Position Size: 10% per trade
- Max Positions: 3 concurrent
- Stop Loss: 2%
- Take Profit: 5%
- Circuit Breaker: 3% max daily loss ($97,000 floor)

**Advanced Features (DISABLED for initial testing):**
- ❌ Kelly Criterion position sizing
- ❌ Multi-timeframe analysis
- ❌ Volatility regime detection
- ❌ Streak-based sizing
- ❌ Short selling

**Paper Account:**
- Starting Capital: $100,000
- Broker: Alpaca (Paper Trading)
- API: https://paper-api.alpaca.markets

---

## Bot Control Commands

### Check if bot is running
```bash
ps aux | grep "python3 main.py" | grep -v grep
```

### View live logs
```bash
tail -f paper_trading.log
```

### View recent activity (last 50 lines)
```bash
tail -50 paper_trading.log
```

### Check bot status
```bash
cat bot.pid && echo "Bot PID: $(cat bot.pid)"
ps -p $(cat bot.pid)
```

### Stop the bot
```bash
kill $(cat bot.pid)
echo "Bot stopped"
```

### Restart the bot
```bash
# Stop
kill $(cat bot.pid) 2>/dev/null

# Start
nohup python3 main.py live --strategy MomentumStrategy --force > paper_trading.log 2>&1 &
echo $! > bot.pid
echo "Bot restarted"
```

---

## Daily Monitoring Checklist

### Morning (Before Market Open)

1. **Verify bot is running:**
   ```bash
   ps -p $(cat bot.pid) && echo "✅ Bot is running" || echo "❌ Bot is NOT running"
   ```

2. **Check overnight logs:**
   ```bash
   tail -100 paper_trading.log | grep -E "(ERROR|WARNING|Circuit)"
   ```

3. **Check account status:**
   ```bash
   python3 -c "
   from brokers.alpaca_broker import AlpacaBroker
   import asyncio

   async def check():
       broker = AlpacaBroker(paper=True)
       account = await broker.get_account()
       print(f'Equity: \${float(account.equity):,.2f}')
       print(f'Buying Power: \${float(account.buying_power):,.2f}')
       print(f'Cash: \${float(account.cash):,.2f}')

   asyncio.run(check())
   "
   ```

4. **Check open positions:**
   ```bash
   python3 -c "
   from brokers.alpaca_broker import AlpacaBroker
   import asyncio

   async def check():
       broker = AlpacaBroker(paper=True)
       positions = await broker.get_positions()
       if not positions:
           print('No open positions')
       else:
           for p in positions:
               pnl = float(p.unrealized_pl)
               pnl_pct = float(p.unrealized_plpc)
               print(f'{p.symbol}: {p.qty} shares @ \${float(p.avg_entry_price):.2f} | P/L: \${pnl:+,.2f} ({pnl_pct:+.2%})')

   asyncio.run(check())
   "
   ```

### During Market Hours

5. **Monitor for trades:**
   ```bash
   tail -f paper_trading.log | grep -E "(ENTRY|EXIT|SIGNAL)"
   ```

6. **Check for errors:**
   ```bash
   tail -f paper_trading.log | grep ERROR
   ```

7. **Monitor circuit breaker:**
   ```bash
   tail -f paper_trading.log | grep -i circuit
   ```

### End of Day

8. **Check daily performance:**
   ```bash
   python3 -c "
   from brokers.alpaca_broker import AlpacaBroker
   import asyncio

   async def check():
       broker = AlpacaBroker(paper=True)
       account = await broker.get_account()
       equity = float(account.equity)
       pnl = equity - 100000
       pnl_pct = pnl / 100000
       print(f'Equity: \${equity:,.2f}')
       print(f'Daily P/L: \${pnl:+,.2f} ({pnl_pct:+.2%})')

   asyncio.run(check())
   "
   ```

9. **Review closed trades:**
   ```bash
   grep -E "(EXIT|closed)" paper_trading.log | tail -20
   ```

10. **Archive logs:**
    ```bash
    cp paper_trading.log logs/paper_trading_$(date +%Y%m%d).log
    ```

---

## Performance Tracking

### Week 1 Goals (Days 1-7)

**Target Metrics:**
- Total Return: ≥ 0% (don't lose money)
- Win Rate: ≥ 50%
- Max Drawdown: ≤ 5%
- Number of Trades: ≥ 5

**Success Criteria:**
- ✅ Bot runs without crashes
- ✅ No circuit breaker triggers
- ✅ Trades execute as expected
- ✅ Stop-loss and take-profit work correctly

### Week 2 Goals (Days 8-14)

**Target Metrics:**
- Total Return: ≥ 1%
- Win Rate: ≥ 60%
- Max Drawdown: ≤ 5%
- Sharpe Ratio: ≥ 1.0

**Success Criteria:**
- ✅ Consistent with backtest results (~4% over 3 months = ~0.5% per week)
- ✅ No critical bugs discovered
- ✅ Risk management functioning properly

---

## Expected Behavior (Based on Backtests)

**From BACKTEST_RESULTS.md:**
- 3-month return: ~4% (16% annualized)
- Sharpe ratio: 2.81-3.53
- Max drawdown: 0.60-1.32%
- Win rate: 67-86%
- Trades per month: 2-5

**Weekly expectations:**
- Return: ~0.3-0.5% per week
- Trades: 0-2 per week
- Max drawdown: ~1-2%

**Important:** Paper trading may differ from backtests due to:
- Real-time execution delays
- Market conditions (2025 vs 2024)
- Slippage and spread
- Different entry/exit timing

---

## Red Flags (Stop Bot Immediately If:)

1. **Circuit Breaker Triggers**
   - Equity drops below $97,000
   - Bot should auto-halt, but verify manually

2. **Equity Drop > 5%**
   - Something is wrong with the strategy
   - Backtest showed max 1.32% drawdown

3. **Continuous Losses**
   - 5+ losing trades in a row
   - Indicates market regime change or strategy failure

4. **API Errors**
   - Repeated "RemoteDisconnected" errors
   - Order submission failures
   - Position sync issues

5. **Unexpected Behavior**
   - Bot trading outside configured symbols
   - Position sizes > 10% of portfolio
   - More than 3 concurrent positions

---

## Troubleshooting

### Bot Won't Start
```bash
# Check for errors
python3 main.py live --strategy MomentumStrategy --force

# Check API credentials
python3 tests/test_connection.py

# Check .env file
cat .env
```

### Bot Crashes
```bash
# Check logs for error
tail -100 paper_trading.log

# Look for traceback
grep -A 20 "Traceback" paper_trading.log

# Check if process is running
ps aux | grep python3
```

### No Trades Executing
```bash
# Market might be closed
python3 -c "
from brokers.alpaca_broker import AlpacaBroker
import asyncio

async def check():
    broker = AlpacaBroker(paper=True)
    status = await broker.get_market_status()
    print(f'Market Open: {status[\"is_open\"]}')
    print(f'Next Open: {status[\"next_open\"]}')

asyncio.run(check())
"

# Check for signals
tail -100 paper_trading.log | grep -E "(SIGNAL|analyze)"
```

### API Connection Issues
```bash
# Test connection
python3 tests/test_connection.py

# Check credentials
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('ALPACA_API_KEY')[:10] + '...')"

# Restart bot
kill $(cat bot.pid)
sleep 2
nohup python3 main.py live --strategy MomentumStrategy --force > paper_trading.log 2>&1 &
echo $! > bot.pid
```

---

## Weekly Review Process

### End of Week 1 (Day 7)

1. **Calculate Performance:**
   ```bash
   python3 -c "
   from brokers.alpaca_broker import AlpacaBroker
   import asyncio

   async def review():
       broker = AlpacaBroker(paper=True)
       account = await broker.get_account()
       equity = float(account.equity)
       total_return = (equity - 100000) / 100000
       print(f'Week 1 Performance:')
       print(f'  Starting: \$100,000.00')
       print(f'  Ending: \${equity:,.2f}')
       print(f'  Total Return: {total_return:+.2%}')

   asyncio.run(review())
   "
   ```

2. **Analyze Trades:**
   ```bash
   # Count trades
   grep "EXIT" paper_trading.log | wc -l

   # Show all trades
   grep -E "(ENTRY|EXIT)" paper_trading.log
   ```

3. **Calculate Win Rate:**
   - Manual review of logs
   - Count winning vs losing trades
   - Compare to backtest expectations

4. **Decision:**
   - Continue if performance ≥ 0% and no critical bugs
   - Stop if major issues discovered
   - Investigate if results differ significantly from backtests

---

## Next Steps After 2 Weeks

**If successful (total return ≥ 1%, no bugs):**
1. Continue for 2 more weeks (total 1 month)
2. Consider enabling ONE advanced feature
3. Document lessons learned

**If unsuccessful (losses or critical bugs):**
1. Stop paper trading
2. Investigate issues
3. Run additional backtests
4. Fix bugs and restart

**If results differ from backtests:**
1. Analyze why (market conditions? execution timing?)
2. Run backtest on recent data (Nov 2024 - Jan 2025)
3. Adjust expectations or strategy

---

## Log Analysis Commands

### Find all trades
```bash
grep -E "(ENTRY|EXIT)" paper_trading.log
```

### Find errors
```bash
grep "ERROR" paper_trading.log
```

### Find circuit breaker events
```bash
grep -i "circuit" paper_trading.log
```

### Count trades per symbol
```bash
grep "ENTRY" paper_trading.log | awk '{print $NF}' | sort | uniq -c
```

### Show recent activity (last hour)
```bash
tail -1000 paper_trading.log | grep -E "$(date +%Y-%m-%d)" | tail -50
```

---

## Contact / Support

If critical issues occur:
1. Stop the bot immediately
2. Save logs: `cp paper_trading.log logs/error_$(date +%Y%m%d_%H%M%S).log`
3. Document the issue
4. Review BACKTEST_RESULTS.md and TODO.md for context

**Remember:** This is PAPER TRADING. No real money at risk. Goal is validation and learning.
