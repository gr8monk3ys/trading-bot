# Roadmap

> The phased plan for taking this project from fork to live trading. Each phase has clear entry criteria, deliverables, and exit criteria. Update as work progresses.

## Phase Overview

| Phase | Goal | Status | Target Duration |
|-------|------|--------|-----------------|
| 0 | Setup and tooling | In progress | 1 week |
| 1 | Paper trading validation on Alpaca | Not started | 2 weeks |
| 2 | Security hardening | Not started | 2 weeks |
| 3 | Fix critical production gaps | Not started | 2–3 weeks |
| 4 | Build Webull Canada adapter | Not started | 1–2 weeks |
| 5 | Webull paper trading validation | Not started | 4 weeks |
| 6 | Go live with minimal capital | Not started | 2–3 weeks |
| 7 | Migrate to dedicated hardware | Optional | 1 week |

---

## Phase 0: Setup and Tooling (current)

**Goal:** Establish the working environment, documentation scaffolding, and AI tooling that the rest of the project depends on.

### Tasks
- [x] Fork the repo to personal GitHub account
- [x] Clone locally and configure git remotes (origin + upstream)
- [x] Install Python, uv, and verify project builds (`uv sync`)
- [x] Create `docs/personal/` documentation scaffolding
- [ ] Fill in all `docs/personal/*.md` files with initial content
- [ ] Set up Claude.ai Project ("Trading Bot Canada") with `AI_CONTEXT.md` as project knowledge
- [ ] Install and verify Claude Code works in the repo
- [ ] Set up branch protection on `main` (require PRs)
- [ ] Verify CI runs on PRs (GitHub Actions in `.github/`)
- [ ] Create accounts: Alpaca (paper), Railway, GitHub (already have)
- [ ] Enable MFA on all accounts

### Exit criteria
- All docs in `docs/personal/` have meaningful content
- Can open VS Code and Claude Code in the repo and have them work
- First PR (the docs PR) merged successfully

---

## Phase 1: Paper Trading Validation on Alpaca

**Goal:** Confirm the bot actually works end-to-end against a real (paper) brokerage API before investing more time in changes.

### Tasks
- [ ] Generate Alpaca paper trading API keys
- [ ] Create local `.env` with paper credentials, verify it's gitignored
- [ ] Run a backtest on 2024 data to confirm strategies execute
- [ ] Run live paper trading during market hours for 1+ session
- [ ] Confirm orders appear in Alpaca paper dashboard
- [ ] Set up Discord or Telegram notifications for alerts
- [ ] Run paper trading on Railway for 2+ weeks
- [ ] Track: number of trades, P&L, crashes, missed sessions
- [ ] Compare paper P&L to backtest expectations

### Exit criteria
- Bot has run on Railway for at least 2 weeks of market sessions
- Notifications work for trades and errors
- Paper P&L is in a reasonable range (not necessarily profitable)
- No unexplained crashes or strategy failures

---

## Phase 2: Security Hardening

**Goal:** Lock down the deployment before any real money is involved.

### Tasks
- [ ] Audit `.env` handling — no keys in logs, no keys in commits
- [ ] Add authentication to FastAPI dashboard (`web/app.py`)
- [ ] Disable LLM alpha features unless explicitly needed (saves cost and reduces attack surface)
- [ ] Review all third-party dependencies for security issues
- [ ] Pin dependency versions in `pyproject.toml`
- [ ] Set Alpaca API key to minimum required permissions, with expiration
- [ ] Document key rotation procedure in `RUNBOOK.md`
- [ ] Verify Railway env vars are correctly set, not exposed in build logs

### Exit criteria
- No credentials anywhere in the repo or logs
- Dashboard cannot be accessed without authentication
- Documented and tested key rotation procedure

---

## Phase 3: Fix Critical Production Gaps

**Goal:** Address the highest-risk issues from upstream `TODO.md` and our own analysis.

### Tasks (priority order)
- [ ] **HIGHEST:** Enforce gateway-only order submission — no strategy can bypass the risk manager
- [ ] Run validated backtest across multiple time periods (not just 2024)
- [ ] Implement walk-forward validation (train on N-2, test on N-1, validate on N)
- [ ] Test circuit breaker emergency liquidation paths
- [ ] Test kill switch (`scripts/kill_switch.py`) end-to-end
- [ ] Improve engine test coverage to ≥98%
- [ ] Document the multiple runtime entrypoints — pick one canonical path

### Exit criteria
- Strategy validated across multiple years of out-of-sample data
- All order paths go through risk manager (verified by tests)
- Kill switch works in <30 seconds from invocation to all positions closed
- Test coverage targets met

---

## Phase 4: Build Webull Canada Adapter

**Goal:** Add support for live trading via Webull Canada (the actual production broker target).

### Tasks
- [ ] Open Webull Canada account
- [ ] Apply for API access (1–2 business day approval)
- [ ] Generate Webull paper trading credentials
- [ ] Create `brokers/webull_broker.py` parallel to `alpaca_broker.py`
- [ ] Implement core methods: `get_positions()`, `get_account()`, `submit_order_advanced()`, etc.
- [ ] Map order types: market, limit, stop, trailing stop, bracket
- [ ] Handle Webull-specific quirks (rate limits, error codes, contract specs)
- [ ] Write tests for the adapter (parallel to existing Alpaca tests)
- [ ] Document the adapter in `docs/personal/RUNBOOK.md`

### Exit criteria
- All adapter methods implemented and tested
- Strategy can be run unchanged with `--broker webull` flag
- Webull paper trading works end-to-end
- Test parity: anywhere there's an Alpaca test, there's a Webull equivalent

---

## Phase 5: Webull Paper Trading Validation

**Goal:** Run the full system on Webull paper trading for an extended period before risking real money.

### Tasks
- [ ] Switch production deployment to Webull paper credentials
- [ ] Run for 4 weeks of market sessions
- [ ] Daily monitoring: trades, P&L, anomalies
- [ ] Compare Webull paper results to Alpaca paper results (sanity check)
- [ ] Investigate any divergences in fill behavior, slippage, timing
- [ ] Document findings in `INCIDENTS.md`

### Exit criteria
- 4 weeks of stable operation
- P&L tracking matches expectations
- No undocumented Webull API quirks remaining

---

## Phase 6: Go Live with Minimal Capital

**Goal:** First real trades. Confirm live execution matches paper.

### Tasks
- [ ] Switch to Webull live credentials
- [ ] Fund account with $500–1000 CAD
- [ ] Configure conservative risk parameters (small position sizes)
- [ ] Run only the most-validated strategy
- [ ] Manual monitoring during first week
- [ ] Compare live fills to paper fills (slippage, timing)
- [ ] Run for 2–3 weeks before scaling up

### Exit criteria
- 2–3 weeks live with no critical bugs
- Live performance is in expected range vs paper
- Confidence to scale up capital (or stop and reassess)

---

## Phase 7: Migrate to Dedicated Hardware (Optional)

**Goal:** Cut ongoing Railway costs, improve reliability with self-hosted setup.

### Tasks
- [ ] Buy Beelink Mini S12 Pro (or equivalent N100 mini PC)
- [ ] Install Ubuntu Server, Docker, Docker Compose
- [ ] Replicate Railway deployment locally with `docker-compose.yml`
- [ ] Set up monitoring (UptimeRobot, healthchecks.io, or similar)
- [ ] Optional: UPS for power outage protection
- [ ] Run hardware deployment in parallel with Railway for 1 week (verification)
- [ ] Cut over to hardware, decommission Railway

### Exit criteria
- Bot running on hardware for 1+ week without issues
- Monitoring alerts work
- Recovery procedures documented (what if hardware dies?)

---

## Working Notes

- Phases can run in parallel where possible (e.g., Phase 4 adapter work while Phase 1 paper trading continues).
- Phase 0 must finish before any other phase starts.
- Phase 6 (going live) requires Phases 1–5 complete. Don't skip.
- This roadmap is a guide, not a contract. Update it as we learn.