# Decision Log

> An Architecture Decision Record (ADR) log for this project. Every meaningful decision gets a dated entry. When future-you wonders "why did I do it this way?" — this answers it.

## Format

Each decision follows this template:

YYYY-MM-DD — [Short title]
Context: What was the situation that required a decision?
Decision: What did we decide to do?
Alternatives considered: What other options were on the table?
Rationale: Why this option over the others?
Consequences: What does this commit us to? What might we regret?
Status: Active / Superseded by [link] / Reversed

---

## 2026-05-03 — Fork gr8monk3ys/trading-bot as the foundation

**Context:** Wanted to start a personal algorithmic trading project. Building from scratch would take months. Several open-source bots exist on GitHub.

**Decision:** Fork `gr8monk3ys/trading-bot` and build on top of it rather than starting from scratch.

**Alternatives considered:**
- Build from scratch using Lumibot or another framework
- Use a no-code platform like Composer or Cryptohopper
- Fork a different repo (e.g., one of the more popular Alpaca bots)

**Rationale:** The repo has a reasonable architecture (async Python, broker abstraction, risk management primitives, backtesting engine), supports paper trading out of the box, and explicitly documents production gaps. The author is also building it AI-assisted, so the codebase is structured for AI-friendly iteration.

**Consequences:** Inherit the original author's architectural choices, including some I may want to change later. The repo has 0 stars and is actively in development by one person — community support is essentially nonexistent. We're on our own for issues.

**Status:** Active

---

## 2026-05-03 — Use Webull Canada (not Alpaca, not IBKR) for live trading

**Context:** The repo is built around Alpaca's API. Alpaca doesn't allow live trading from Canadian residents (paper only). Need to choose an alternative broker that's accessible from Canada and works with the bot's architecture.

**Decision:** Target Webull Canada for live trading. Build a `brokers/webull_broker.py` adapter parallel to the existing `alpaca_broker.py`. Continue using Alpaca paper trading for development and validation.

**Alternatives considered:**
- **Interactive Brokers Canada:** Most mature API, widest market access, but requires persistent TWS/Gateway connection (complicates Railway deployment) and charges per-trade commissions
- **Moomoo Canada:** Commission-free, but uses OpenD gateway program (same persistent connection problem as IBKR)
- **Questrade:** Read-only API for non-partner customers, can't place trades programmatically
- **OANDA:** No US equities — only forex and CFDs
- **Wait for Alpaca Canada:** No timeline, could be 1–2+ years away

**Rationale:** Webull's REST API model matches Alpaca's stateless architecture, making the adapter relatively straightforward (~5–7 days of work vs ~2 weeks for IBKR). It's commission-free, has a Python SDK, and is available to Canadian residents. Deployment stays simple — no gateway process required.

**Consequences:** Webull's API is newer and less battle-tested than Alpaca's or IBKR's — expect undocumented quirks. Community resources are thinner. We're betting on Webull's API stability for live trading.

**Status:** Active (pending Webull account approval in Phase 3)

---

## 2026-05-03 — Defer Webull account creation to Phase 3

**Context:** Considered opening Webull Canada account immediately during Phase 1 setup.

**Decision:** Don't open Webull account until Phase 3 (after paper trading validates the bot on Alpaca).

**Alternatives considered:** Open the account now, get API approval started in parallel.

**Rationale:** API access approval timer starts when applied for, not when the account is created. Better to apply when actually ready to use it. Also: we may discover the strategy doesn't work in paper trading and abandon the project, in which case the Webull account is wasted. Keep that decision open until needed.

**Consequences:** Slight delay (1–2 business days) when transitioning to Phase 3. Acceptable.

**Status:** Active

---

## 2026-05-03 — Hardware: Railway for development, Beelink N100 for production

**Context:** The bot needs 24/7 always-on hosting during market hours. Several options at different price points.

**Decision:** Use Railway (~$10–15/month) for Phase 1–4 development and validation. Migrate to a Beelink Mini S12 Pro (N100, ~$220 CAD one-time) for long-term production hosting in Phase 6.

**Alternatives considered:**
- **Stay on Railway forever:** Simple but expensive long-term ($120–180/year ongoing)
- **Raspberry Pi 5:** Cheaper hardware (~$120–150 CAD all-in) but ARM architecture causes occasional Python package issues, and 4GB RAM is tight
- **Used mini PC:** $50–100 CAD on Kijiji, but reliability unknown and higher power draw
- **Always run on personal laptop:** Free but laptop can't be reliably 24/7

**Rationale:** Railway gets us moving quickly without hardware purchase. Beelink N100 has 16GB RAM, NVMe SSD, x86 (no ARM compatibility issues), draws ~6–10W idle, and breaks even with Railway around month 15–20. Reliability and headroom matter for live trading.

**Consequences:** Two-stage deployment migration in our future. Need to write deployment docs that cover both Railway and Beelink/Docker setups. Some duplicated configuration work.

**Status:** Active (Railway deployment in Phase 1, Beelink migration in Phase 6)

---

## 2026-05-03 — Documentation strategy: personal subfolder

**Context:** The forked repo already has a `docs/` folder full of the original author's documentation. Need to add our own decisions, roadmap, and notes without conflicting.

**Decision:** Personal docs go in `docs/personal/`. Original repo docs stay untouched in `docs/`.

**Alternatives considered:**
- Overwrite the original files with our own content
- Add prefixes like `MY_ROADMAP.md` to distinguish ours
- Use a separate top-level folder like `mydocs/`

**Rationale:** Subfolder cleanly separates "inherited context" from "our decisions" without polluting either. When pulling upstream changes, our subfolder won't conflict. Clear bus-factor: anyone reading the repo immediately understands which docs are inherited vs ours.

**Consequences:** Two doc directories to maintain. The original author's docs are out of date for our purposes (e.g., his roadmap targets Alpaca, ours targets Webull) but kept for reference.

**Status:** Active

---

## 2026-05-03 — AI tooling: Claude.ai Project + Claude Code + repo as source of truth

**Context:** Project will span months. Chat history in Claude.ai is ephemeral. Need persistent context for AI-assisted work.

**Decision:** Use three tools in combination:
- **Claude.ai Project ("Trading Bot Canada")** for planning conversations with persistent context (uploaded `AI_CONTEXT.md` as project knowledge)
- **Claude Code** for actual development inside the repo
- **Repo itself (especially `docs/personal/AI_CONTEXT.md`)** as the source of truth that both tools sync to

**Alternatives considered:**
- Just use regular Claude.ai chat (loses context every session)
- Just use Claude Code (no good for high-level planning conversations)
- Use Cowork (wrong fit — designed for non-developer workflow automation)

**Rationale:** Each tool plays to its strength. Project keeps planning context durable. Claude Code makes real changes to real files. Repo docs survive any tool changes. Update flow: when something significant changes, update `AI_CONTEXT.md` in the repo, commit it, re-upload to Claude.ai Project.

**Consequences:** Need discipline to keep `AI_CONTEXT.md` current. If we don't, future sessions will get stale context. Worth occasional time investment to keep synced.

**Status:** Active

---

<!-- Add new decisions below this line, newest at top -->