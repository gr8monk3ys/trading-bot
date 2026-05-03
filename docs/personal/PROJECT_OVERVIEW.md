# Project Overview — Trading Bot Canada

## What This Project Is

A personal algorithmic trading bot adapted for Canadian markets. Forked from [gr8monk3ys/trading-bot](https://github.com/gr8monk3ys/trading-bot), modified for use with a Canadian-accessible broker (Webull Canada), and hardened for personal production use.

This is a **single-user** project. It exists to automate my own trading. It is not a product, not a service, and not advice for others.

## Why It Exists

Three motivations:

1. **Learning:** Hands-on exposure to algorithmic trading, broker APIs, and quantitative strategy validation.
2. **Personal trading:** If the strategy validates, automate it instead of manually trading.
3. **DevOps practice:** Apply production engineering discipline (CI/CD, monitoring, incident response, runbooks) to a system where mistakes have real financial consequences.

## What It Does

The bot trades US equities through a brokerage API. It runs strategies that analyze market data, generate buy/sell signals based on technical indicators (RSI, MACD, momentum), and submit orders during market hours. Risk management components (circuit breaker, position limits, kill switch) protect against catastrophic losses.

Key strategies inherited from the upstream repo:
- **MomentumStrategy:** RSI/MACD trend following with trailing stops
- **AdaptiveStrategy:** Regime-switching coordinator (auto-selects momentum vs mean reversion based on detected market conditions)
- **MeanReversionStrategy:** Counter-trend trading on extreme price moves

## What It Doesn't Do

- **Not a trading advisor.** It executes pre-programmed strategies, doesn't make recommendations.
- **Not high-frequency.** Operates on daily/intraday bars, not microsecond ticks.
- **Not multi-asset.** US equities only. No crypto, options, futures, forex (yet).
- **Not multi-user.** Single broker account, single owner.
- **Not regulated as a service.** Personal use only — not offered to others.

## Key Constraints

- **Owner is in Canada.** Limits broker choice (Alpaca paper only, Webull/IBKR/Moomoo for live).
- **Personal capital scale.** Initial live capital target: $500–1000 CAD. Not a hedge fund.
- **One developer.** No team, no on-call rotation, no 24/7 monitoring beyond automated alerts.
- **Limited time.** Maintenance happens evenings and weekends.

## Success Criteria

For Phase 1 (paper trading):
- Bot runs continuously for 4+ weeks without crashes
- Strategies generate trades that match backtest behavior
- Alerts fire correctly on errors and circuit breaker triggers
- No security incidents (key leaks, dashboard exposure)

For Phase 5 (live trading):
- 3+ months live with no critical bugs
- Drawdowns stay within configured risk limits
- P&L is reasonable relative to risk taken (not necessarily profitable, but not catastrophic)

## Failure Criteria (When to Stop)

- Strategy is consistently unprofitable after slippage and fees in paper trading
- Critical bugs found that the architecture can't easily fix
- Time investment exceeds value (this is a personal project, not a job)
- Regulatory or broker TOS issues that can't be resolved

## Disclaimer

This project involves real money and real risk. Algorithmic trading can lose money. Past backtest performance does not guarantee future results. The owner is not a financial advisor and this project should not be used by others without independent due diligence.

## Quick Links

- Roadmap: `docs/personal/ROADMAP.md`
- Decisions log: `docs/personal/DECISIONS.md`
- Operational runbook: `docs/personal/RUNBOOK.md`
- Incident log: `docs/personal/INCIDENTS.md`
- AI assistant context: `docs/personal/AI_CONTEXT.md`
- Original repo docs: `docs/` (inherited from upstream)