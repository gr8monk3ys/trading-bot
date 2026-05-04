# AI Context — Trading Bot Canada

> **Purpose:** This document is the canonical context for AI assistants working on this project. Paste it into any new Claude conversation to bring the assistant up to speed instantly. Keep it updated as the project evolves.

## Project Summary

This is a personal algorithmic trading bot project, forked from [gr8monk3ys/trading-bot](https://github.com/gr8monk3ys/trading-bot). The owner is a DevOps engineer based in Canada. The goal is to validate, harden, and eventually deploy the bot for personal trading using a Canadian-accessible broker.

This is **not** a SaaS product, not a service for others, and not an investment advisory tool. It is a single-user automated trading system.

## Owner Profile

- Background: DevOps / Backend Engineering
- Location: Canada
- Comfortable with: Python, Docker, CI/CD, cloud deployment, terminal workflows
- New to: algorithmic trading specifics, broker APIs, financial regulation

## Current Phase

**Phase 1: Paper trading validation on Alpaca** (active, just started)

Phase 0 setup is complete:
- ✅ Repo forked, remotes configured, SSH dual-account auth
- ✅ Documentation scaffolding in `docs/personal/`
- ✅ Branch protection on `main`
- ✅ Claude.ai Project with this file as project knowledge
- ✅ Claude Code working inside the repo
- ✅ Alpaca paper credentials configured and verified
- ✅ First real bug found and fixed (PR #3): backtest mode was silently dropping all orders due to missing OrderGateway wiring

Phase 1 status:
- ✅ First successful backtest: `MomentumStrategyBacktest` on AAPL/MSFT/GOOGL/NVDA/META/AMZN for 2024 → +21.26% return, Sharpe 1.62, 6 trades. **Important caveat:** subsequent parity audit (see below) revealed this measures `MomentumStrategyBacktest`, which has materially different parameters and sizing logic from the live `MomentumStrategy`. The number is not predictive of live behavior.
- ✅ Parity audit: `MomentumStrategy` vs `MomentumStrategyBacktest` — found `MomentumStrategy.execute_trade()` was a `pass`, so the live class produced zero orders when run through `BacktestEngine`. Same family of bug as PR #3, inverted direction.
- 🟡 PR open (not yet merged): `fix: make MomentumStrategy runnable in BacktestEngine` — branch `fix/momentum-strategy-backtest-parity`. Replaces the empty `execute_trade` with dispatch to `_execute_signal` + `_check_exit_conditions`, fixes three corollary issues (async `get_positions`, simulated-time cooldown, `price_history` shape parity), adds regression + cooldown tests. 4343/4377 passing, 3 verified-pre-existing failures.
- 🔜 Bracket-qty follow-up PR (Phase 1 blocker — see Known Issues). Must merge before any further backtest work.
- 🔜 Re-run 2024 backtest with live `MomentumStrategy` (post-fix) to compare against the +21.26% number from the Backtest variant. Expect divergence; honest divergence is the goal.
- 🔜 Run more backtest variations (different strategies, time windows, symbols)
- 🔜 Run live paper trading during market hours
- 🔜 Set up Discord/Telegram notifications
- 🔜 Deploy to Railway for 24/7 paper trading
- 🔜 Run for 2+ weeks before declaring Phase 1 complete

Known issues identified during Phase 1:
- **Phase 1 blocker:** `BacktestBroker.submit_order_advanced` does not extract qty from bracket-order envelopes. Surfaced (not caused) by the parity PR — once `MomentumStrategy` flows orders through the gateway, bracket orders reach the broker but get logged with `quantity=0`. Real-data backtests will show trades happening with zero equity impact. Follow-up PR required before resuming Phase 1 backtest work.
- BacktestEngine creates its own OrderGateway separately from StrategyManager's — should thread the same instance through (Phase 3 task)
- ~~`BacktestBroker.get_positions()` async warning in OrderGateway logs~~ — fixed by parity PR (pulled forward from Phase 3)
- ~~No CI smoke test for backtest order placement~~ — added by parity PR (`tests/unit/test_momentum_strategy_backtest_parity.py`)

Lessons learned (add to process notes):
- Coordinated-API-change surveys must grep for both class stubs *and* inline mock patches (`mock_x.method.return_value = ...`). Class-only greps miss the latter — bit us during the parity PR (5 test files needed updates, two were missed by the first survey).
- Phase 1 / 2 / 3 stop points in risk-critical PR work caught real issues (Blockers A and B during Phase 1 analysis, fixture mismatch during Phase 2). The ~3-file scope-expansion threshold for "stop and check" worked.
- `MomentumStrategyBacktest` exists primarily as a workaround for the empty `execute_trade` in the live class. Once the parity PR lands, evaluate deleting it or reducing it to a pure parameter override (no logic divergence). Open question for a separate `DECISIONS.md` entry.

## Key Decisions Made

1. **Broker for live trading: Webull Canada** (not Alpaca, not IBKR)
   - Alpaca doesn't allow live trading from Canadian residents (paper only)
   - IBKR works but requires persistent gateway connection, complicating deployment
   - Moomoo has same gateway problem as IBKR
   - Webull Canada has a stateless REST API similar to Alpaca, commission-free, and is available to Canadians

2. **Hosting strategy: Railway short-term, Beelink N100 long-term**
   - Railway for paper trading validation and early development (~$10–15/month)
   - Migrate to dedicated hardware (Beelink Mini S12 Pro, ~$220 CAD one-time) once stable
   - Cloud hosting breaks even with hardware around month 15–20

3. **Documentation lives in the repo, not in chat**
   - This file (`AI_CONTEXT.md`) is the entry point for AI assistants
   - Personal docs in `docs/personal/`, original repo docs in `docs/`

4. **Webull account creation is deferred to Phase 3**
   - Don't open until paper trading validates the bot is worth pursuing
   - API access application takes 1–2 business days when needed

## Roadmap (Summary)

- **Phase 1:** Fork, set up tooling, paper trade on Alpaca (current)
- **Phase 2:** Build Webull Canada broker adapter (`brokers/webull_broker.py`)
- **Phase 3:** Open Webull account, get API access, validate adapter against Webull paper
- **Phase 4:** Extended paper trading on Webull (1 month minimum)
- **Phase 5:** Go live with minimal capital ($500–1000 CAD initially)
- **Phase 6:** Migrate to dedicated hardware (Beelink N100)

See `docs/personal/ROADMAP.md` for details.

## Critical Production Gaps Identified

The original repo's `TODO.md` and our analysis identified these as must-fix before live trading:

1. **Strategies bypass the gateway/risk layer** in some code paths — every order must pass through circuit breakers and risk manager
2. **No regression tests for circuit-breaker emergency liquidation** with gateway enforcement
3. **Walk-forward / out-of-sample strategy validation missing** — backtest claims +42.68% for 2024 but only on a single year
4. **Engine test coverage below 98% target**
5. **Multiple overlapping runtime entrypoints** (`main.py`, `live_trader.py`, `run_adaptive.py`, `start.py`) — architecture is fragmented
6. **FastAPI dashboard has no authentication** — would expose positions, P&L, and account data if deployed publicly
7. **Incident webhook integration unvalidated outside test environments**

## Architecture Overview

main.py → StrategyManager → [Strategy] → Broker → Broker API
↓             ↓
BacktestEngine  RiskManager
↓             ↓
PerformanceMetrics CircuitBreaker

Key components:
- `brokers/alpaca_broker.py` — current production broker (Alpaca)
- `brokers/backtest_broker.py` — mock for backtesting
- `brokers/webull_broker.py` — to be built in Phase 2
- `strategies/` — MomentumStrategy, AdaptiveStrategy, MeanReversionStrategy
- `risk/` — RiskManager, CircuitBreaker, position sizing
- `web/app.py` — FastAPI monitoring dashboard

## Tooling Stack

- **Claude.ai Project ("Trading Bot Canada")**: planning, decisions, research
- **Claude Code**: actual development, file edits, running tests
- **Regular Claude.ai chats**: one-off questions

When working on code, prefer Claude Code over chat. When planning or researching, use the Project. Always update this file (`AI_CONTEXT.md`) when significant decisions are made.

## What Future AI Assistants Should Know

- **The owner is in Canada.** Do not suggest US-only services without flagging the constraint.
- **This is for the owner's personal trading.** Do not suggest scaling, monetization, or multi-user features unless asked.
- **Risk-critical code (orders, kill switch, circuit breaker) requires human review.** Don't auto-merge changes to those files.
- **The owner is not a financial advisor.** Always include appropriate caveats when discussing strategy performance or returns.
- **The chat history is ephemeral.** Important decisions belong in `DECISIONS.md`, not in conversation.

## How to Update This File

When something significant changes:
1. Add the decision to `docs/personal/DECISIONS.md` with a date
2. Update the relevant section in this file
3. Commit both changes together with a message like `docs: update AI context after [decision]`
4. If using Claude.ai Project knowledge, re-upload this file to keep it synced

Last updated: [May 4, 2026]