# Runbook

> Operational procedures for running, monitoring, and troubleshooting the trading bot. Future-you panicking at 10am on a Tuesday will need this.
>
> Many sections are placeholders for now (project is pre-deployment). Fill them in as procedures are established.

## Critical Contacts and Resources

- **GitHub repo (your fork):** `https://github.com/AchyuthPilly/trading-bot`
- **Upstream repo:** `https://github.com/gr8monk3ys/trading-bot`
- **Alpaca paper dashboard:** `https://app.alpaca.markets/paper/dashboard/overview`
- **Alpaca live dashboard:** N/A (Canada — not available)
- **Webull Canada dashboard:** TBD (Phase 3+)
- **Railway project:** TBD
- **Discord/Telegram alert channel:** TBD

## Environment Variables Reference

The bot is configured via `.env` in the project root. Reference (do NOT put real values here):

Good. Next file: RUNBOOK.md. This is operational procedures — what to do when running, monitoring, or troubleshooting the bot. Mostly placeholders for now since we're pre-deployment, but the structure is in place for when you need it.
Open the File
code docs/personal/RUNBOOK.md
Content for RUNBOOK.md
Copy everything between the dashed lines into VS Code:

markdown# Runbook

> Operational procedures for running, monitoring, and troubleshooting the trading bot. Future-you panicking at 10am on a Tuesday will need this.
>
> Many sections are placeholders for now (project is pre-deployment). Fill them in as procedures are established.

## Critical Contacts and Resources

- **GitHub repo (your fork):** `https://github.com/AchyuthPilly/trading-bot`
- **Upstream repo:** `https://github.com/gr8monk3ys/trading-bot`
- **Alpaca paper dashboard:** `https://app.alpaca.markets/paper/dashboard/overview`
- **Alpaca live dashboard:** N/A (Canada — not available)
- **Webull Canada dashboard:** TBD (Phase 3+)
- **Railway project:** TBD
- **Discord/Telegram alert channel:** TBD

## Environment Variables Reference

The bot is configured via `.env` in the project root. Reference (do NOT put real values here):
Alpaca (paper or live)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
PAPER=True
Webull (Phase 3+)
WEBULL_APP_KEY=
WEBULL_APP_SECRET=
WEBULL_ACCOUNT_ID=
LLM features (optional, disabled by default)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
Notifications
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
Database
DATABASE_URL=sqlite:///trading_bot.db

`.env` is in `.gitignore`. Verify with `git status` after editing — it should never appear in tracked files.

## Common Operations

### Start the bot in paper trading mode (local)

```bash
cd /path/to/trading-bot
uv run python main.py live --strategy MomentumStrategy --force
```

Stop with Ctrl+C. The bot should gracefully cancel pending orders and exit.

### Run a backtest

```bash
uv run python main.py backtest --strategy MomentumStrategy --start 2024-01-01 --end 2024-12-31
```

Results print to stdout. For detailed analysis: `scripts/validated_backtest_report.py`.

### Check current positions and account state

While the bot is running, the FastAPI dashboard is available at `http://localhost:8000` (or whatever port is configured). For programmatic access:

```bash
# (TBD — document specific commands once dashboard is verified)
```

### Trigger the kill switch

If you need to halt all trading immediately and liquidate positions:

```bash
uv run python scripts/kill_switch.py
```

This should:
1. Cancel all open orders
2. Close all positions at market
3. Stop the bot from placing new orders

**Test this in paper mode at least once before relying on it in live trading.**

### Rotate API keys

(Procedure to be documented after first key rotation in Phase 2.)

Rough plan:
1. Generate new key in broker dashboard
2. Update `.env` (local) and Railway env vars (cloud)
3. Restart the bot
4. Verify new key works (test trade in paper)
5. Revoke old key in broker dashboard

### Pull updates from upstream repo

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

Resolve conflicts as needed. Personal docs in `docs/personal/` won't conflict with upstream.

## Monitoring

### What to check daily during market hours

- Alpaca/Webull dashboard: are positions and orders as expected?
- Discord/Telegram alerts: any errors or circuit breaker triggers?
- Railway logs (if deployed there): bot still running?
- Bot's FastAPI dashboard: P&L, position summary, strategy state

### What to check weekly

- Total P&L for the week
- Number of trades, win rate
- Any unexpected behavior to log in `INCIDENTS.md`
- GitHub issues — close completed ones, prioritize next week

### What to check monthly

- Strategy performance vs. backtest expectations (drift?)
- Dependency updates (run `uv sync` and check for issues)
- Key rotation if approaching expiration
- Review and update this runbook for anything that's gotten stale

## Troubleshooting

### Bot won't start

- Check `.env` is present and correctly populated
- Run `uv sync` to ensure dependencies are installed
- Check Python version: `python3 --version` (need 3.10+)
- Look at the error message — most issues are config problems

### Bot starts but doesn't place trades

- Is the market open? (9:30 AM – 4:00 PM ET, weekdays)
- Are there enough symbols configured to analyze?
- Check the circuit breaker — it may have tripped (look for messages in logs)
- Check broker API status (Alpaca status page, Webull status page)

### Orders rejected by broker

- Insufficient buying power
- Invalid symbol (delisted, halted, etc.)
- Pattern day trader rules (if applicable)
- API key permissions too restrictive

### Bot crashes overnight

- Check Railway/system logs for the error
- Add the incident to `INCIDENTS.md`
- Verify auto-restart is configured (Docker restart policy or Railway equivalent)
- Open a GitHub issue to investigate root cause

### Suspicious activity in account

- Immediately run kill switch
- Check broker dashboard for unauthorized trades
- Rotate API keys
- Investigate logs for unusual patterns
- Document in `INCIDENTS.md`

## Recovery Procedures

### If your laptop dies

You can re-deploy from any machine:
1. Clone your fork: `git clone https://github.com/AchyuthPilly/trading-bot.git`
2. Restore `.env` from password manager
3. `uv sync`
4. Verify with a paper backtest
5. Resume from Railway dashboard or restart locally

### If Railway has an outage

- Check Railway status page
- If extended outage and bot has open positions, manually monitor via broker dashboard
- Consider having a fallback deployment ready (Phase 7 hardware migration helps here)

### If Webull/broker has an outage

- Bot will fail to place new orders, log errors
- Existing orders submitted to the broker may still execute
- Manual intervention via broker dashboard if needed
- Document the incident

## Deployment Procedures

(To be filled in once deployments are established.)

### Railway deployment

(TBD — document after first successful Railway deployment.)

### Hardware deployment (Beelink/Docker)

(TBD — Phase 7.)

## Compliance and Records

- Trading records: kept by broker, also logged locally in SQLite database
- Tax records: export annual trade history from broker for accountant
- API access agreements: review broker TOS annually for any changes affecting automated trading

## Update Log

- 2026-05-03: Initial runbook created (Phase 0). Many sections are placeholders.