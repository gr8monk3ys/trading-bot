# Honest Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip unjustified narrative and unvalidated modules from the trading-bot repo, fix the broken install, prune operational over-engineering, and publish one honest baseline backtest.

**Architecture:** Nine independent commits, each revertible. First commit touches only docs. Second fixes install so subsequent commits can verify with pytest. Commits 3–7 do the structural cuts (delete unvalidated phases, quarantine plausible-but-unvalidated work, purge ops scripts/CI, trim config). Commit 8 produces the honest reference backtest. Commit 9 repopulates `TODO.md` with real follow-ups.

**Tech Stack:** Python 3.10+, pytest, pandas/numpy, Alpaca SDK (paper). No torch, no transformers, no LLM SDKs after cleanup.

**Spec:** `docs/superpowers/specs/2026-05-11-honest-cleanup-design.md`

---

## Conventions for every Task

- After every code-touching step that could break things, run `pytest tests/unit/ -x --no-cov -q` and confirm it does not error on collection or fail. If tests fail because they tested a deleted module, delete those test files in the same commit.
- Use `git rm <path>` (not just `rm`) so deletions are staged.
- Every commit message ends with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (use HEREDOC form).
- Discovery before deletion: before deleting a module, run `grep -rn "from <module>\|import <module>" --include="*.py" .` to find all references. Strip references before the file delete so tests don't collect dead imports.
- If a step says "verify tests pass" and they don't because of an issue not anticipated by the plan, **stop and surface the issue** rather than patching around it.

---

## Task 1: Doc rewrite

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `backtest_report_2024.md`
- Delete: `docs/PRODUCTION_READINESS.md`, `docs/OPERATIONS_RUNBOOK.md`, `docs/RUNBOOK.md`, `docs/RUNTIME_ARCHITECTURE.md`, `docs/COMPLIANCE_GOVERNANCE.md`, `docs/INCIDENT_ESCALATION_ROSTER.md`, `docs/INCIDENT_RESPONSE_OWNERSHIP.md`, `docs/MULTI_AGENT_EXECUTION_PLAN.md`, `docs/AGENT_REPORT.md`, `docs/PHASE_3_ENHANCEMENTS.md`, `docs/IMPLEMENTATION_SUMMARY.md`, `docs/ADVANCED_FEATURES.md`, `docs/CLAUDE_ADVANCED.md`, `docs/LOW_HARDWARE_PROFILE.md`, `docs/STATUS.md`, `docs/RELEASE_NOTES_2026-02-22.md`, `docs/QUANT_TRADING_RESEARCH_2025-11-08.md`, `docs/SECRETS_ROTATION_INVENTORY.json`, `docs/FILE_STRUCTURE.md`, `docs/KELLY_CRITERION_INTEGRATION.md`, `DOCKER_CICD_SUMMARY.md`

- [ ] **Step 1: Rewrite `CLAUDE.md`**

Replace the entire file with:

```markdown
# CLAUDE.md

Guidance for Claude Code working in this repository.

## Status: experimental

This repository is a personal algorithmic-trading sandbox. It is **paper-only** and has no proven edge. Do not deploy real capital. Previous versions of this document claimed an "institutional-grade" rating and a +42.68% backtest; both claims were unsupported by the evidence in the repo (see `PROFITABILITY_RESEARCH.md` for the analysis) and have been removed.

The only validated baseline lives at `results/honest_backtest_2020-2024.md` — when that file exists, that is the single performance number to cite.

## Project overview

Algorithmic trading bot on the Alpaca Trading API, async Python.

**Stack:** Python 3.10+, asyncio, pandas, numpy, TA-Lib, pytest-asyncio.

## Core code path (production)

- `strategies/momentum_strategy.py` — RSI/MACD/ADX momentum with trailing stops, Kelly gated off by default.
- `strategies/momentum_strategy_backtest.py` — daily-data-friendly variant of the above.
- `strategies/mean_reversion_strategy.py` — pair to momentum for sideways regimes.
- `strategies/adaptive_strategy.py` — regime-switching coordinator that picks between momentum and mean-reversion. Imports only those two strategies plus `MarketRegimeDetector`; all ensemble/ML/cross-asset branches were removed during the 2026-05 cleanup.
- `strategies/simple_ma_strategy.py` — minimal reference strategy.
- `strategies/risk_manager.py` — position sizing, VaR, correlation rejection.
- `strategies/base_strategy.py` — abstract base for all strategies.
- `brokers/alpaca_broker.py`, `brokers/backtest_broker.py`, `brokers/order_builder.py`.
- `engine/backtest_engine.py`, `engine/performance_metrics.py`, `engine/strategy_manager.py`.
- `utils/circuit_breaker.py`, `utils/market_regime.py`, `utils/realistic_backtest.py`, `utils/websocket_manager.py`, `utils/database.py`, `utils/notifier.py`, `utils/audit_log.py`, `utils/multi_timeframe.py`.

## Quarantined (unvalidated)

Under `research/`. Not imported by the production path, excluded from default `pytest`. Includes: factor models, factor portfolios, cross-asset signals, pairs trading, walk-forward / validated backtest, alpha-decay monitoring, IC tracker, point-in-time data, historical universe, crypto and extended-hours support. These modules have no evidence of edge in this codebase; treat them as ideas, not products.

## Commands

```bash
# Install
pip install -r requirements.txt

# Tests
pytest tests/                            # default: excludes research/
pytest tests/unit/test_risk_manager.py -v
pytest tests/ --cov=strategies --cov=utils --cov-report=html

# Backtests
python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-12-31
python run_adaptive.py --backtest --start 2024-01-01 --end 2024-12-31

# Paper trading (requires .env with ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER=True)
python run_adaptive.py
python main.py live --strategy MomentumStrategy --force

# Lint / format
black strategies/ brokers/ engine/ utils/
ruff check strategies/ brokers/ engine/ utils/
mypy strategies/ brokers/ engine/ utils/
```

## Implementation patterns

- All broker operations are async — use `await`.
- New strategies inherit `BaseStrategy`, set `NAME` class attribute, live in `strategies/`.
- Strategies populate `self.price_history[symbol]` before calling `_calculate_volatility(symbol)`.
- `OrderBuilder` is imported inside methods, not at module top, to avoid circular imports.

## Configuration

`config.py` exposes:
- `TRADING_PARAMS`, `RISK_PARAMS`, `TECHNICAL_PARAMS`.

Parameter blocks for deleted features (`ML_PARAMS`, `RL_PARAMS`, `OPTIONS_PARAMS`, `SENTIMENT_PARAMS`, `LLM_PARAMS`, `CRYPTO_PARAMS`, `OVERNIGHT_PARAMS`) were removed in the 2026-05 cleanup.

## Environment variables (`.env`)

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
PAPER=True

# Optional
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DATABASE_URL=sqlite:///trading_bot.db
```

## Critical gotchas

1. All broker operations need `await`.
2. NumPy pinned `>=1.24.0,<3.0.0` for compatibility.
3. Market hours: bot won't run if market closed unless `--force`.
4. `PAPER` env defaults true; live mode requires explicit opt-in and is **not recommended**.
5. Strategy discovery is import-based — strategies must be importable from `strategies/`.
6. `pytest` `asyncio_mode = auto` — don't add `@pytest.mark.asyncio` decorators.

## Test layout

```
tests/
├── unit/         # default test target
├── integration/  # slower, may hit APIs
├── fixtures/     # mock_broker, sample_price_history
└── ...
research/tests/   # quarantined; excluded from default pytest
```

## Style

From `.windsurfrules`:
- Functional preferred; avoid classes that exist only to namespace.
- Vectorized pandas/numpy over explicit loops.
- PEP 8.
- Descriptive variable names.

## When working in this repo

- Don't add features without evidence. If a feature can't be backed by a real backtest or A/B test, don't ship it.
- Don't reintroduce the "phases" framing. Phases are how the repo got into trouble.
- If you delete a module, delete its tests and its config in the same commit.
- Prefer editing existing files; only create new ones when necessary.
```

Run:

```bash
wc -l CLAUDE.md
```

Expected: file is now under 200 lines (down from ~700).

- [ ] **Step 2: Rewrite `README.md`**

Replace the entire file with:

```markdown
# trading-bot

Personal algorithmic-trading sandbox on Alpaca, async Python.

**Status:** experimental, paper-only, no proven edge. Do not deploy real capital.

## What's in here

- A momentum strategy (RSI/MACD/ADX with trailing stops).
- A mean-reversion strategy.
- An adaptive coordinator that switches between them based on market regime.
- A backtest engine with realistic slippage and spread.
- A risk manager (VaR, correlation, position sizing).
- A circuit breaker (daily-loss halts).
- An Alpaca broker wrapper (paper + live).

Plausible-but-unvalidated quant work — factor models, pairs trading, cross-asset signals, walk-forward validation, alpha-decay monitoring — is in `research/`, excluded from the production path and the default test run.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env  # add ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER=True

pytest tests/
python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-12-31
python run_adaptive.py
```

## Performance

When `results/honest_backtest_2020-2024.md` exists, that is the only performance number to cite. Earlier reports (notably `backtest_report_2024.md`) used 9 trades and are below the statistical-significance bar this repo's own `PROFITABILITY_RESEARCH.md` calls out. Don't quote them.

## License

MIT. See `LICENSE`.
```

- [ ] **Step 3: Prepend caveat header to `backtest_report_2024.md`**

Insert at the very top of the file (before the existing `# Backtest Performance Report` line):

```markdown
> **⚠️ Caveat — read first.** This report covers 9 trades over a calendar year. The repo's own `PROFITABILITY_RESEARCH.md` states that any backtest under 50 trades is statistically meaningless (~40% probability the result is luck). 8 of 12 months in this report show exactly 0% return — the strategy was in cash most of the year. The Sharpe of 2.0 is an artifact of low time-in-market, not edge. This file is retained as a deprecated artifact. The current reference baseline is `results/honest_backtest_2020-2024.md` once it exists. Do not quote the +42.68% number.

---

```

- [ ] **Step 4: Delete the obsolete docs**

```bash
git rm docs/PRODUCTION_READINESS.md docs/OPERATIONS_RUNBOOK.md docs/RUNBOOK.md docs/RUNTIME_ARCHITECTURE.md docs/COMPLIANCE_GOVERNANCE.md docs/INCIDENT_ESCALATION_ROSTER.md docs/INCIDENT_RESPONSE_OWNERSHIP.md docs/MULTI_AGENT_EXECUTION_PLAN.md docs/AGENT_REPORT.md docs/PHASE_3_ENHANCEMENTS.md docs/IMPLEMENTATION_SUMMARY.md docs/ADVANCED_FEATURES.md docs/CLAUDE_ADVANCED.md docs/LOW_HARDWARE_PROFILE.md docs/STATUS.md docs/RELEASE_NOTES_2026-02-22.md docs/QUANT_TRADING_RESEARCH_2025-11-08.md docs/SECRETS_ROTATION_INVENTORY.json docs/FILE_STRUCTURE.md docs/KELLY_CRITERION_INTEGRATION.md DOCKER_CICD_SUMMARY.md
```

Run:

```bash
ls docs/
```

Expected: only `SETUP.md`, `installation.md`, `strategy_guide.md`, `TESTING.md`, `advanced_orders_guide.md`, `DASHBOARD_GUIDE.md`, plus the `superpowers/` directory. ≤7 items.

- [ ] **Step 4b: Review remaining root markdown for relevance**

Read `AGENTS.md` and `PAPER_TRADING_GUIDE.md`. If either references deleted modules (LSTM, options, LLM, factor models, incident escalation, etc.), make one of two choices and commit it:

- **Delete** if the file is dominated by deleted-feature content (`git rm <file>`).
- **Edit** if it mostly describes still-valid workflow — strip the dead references inline.

Don't agonize: when in doubt, delete.

- [ ] **Step 5: Verify no other docs reference deleted files**

Run:

```bash
grep -rln "PRODUCTION_READINESS\|OPERATIONS_RUNBOOK\|INCIDENT_ESCALATION_ROSTER\|INCIDENT_RESPONSE_OWNERSHIP\|MULTI_AGENT_EXECUTION_PLAN\|PHASE_3_ENHANCEMENTS\|CLAUDE_ADVANCED\|RELEASE_NOTES_2026" --include="*.md" --include="*.py" --include="*.sh" --include="*.yml" .
```

Expected: empty output, or only the design spec at `docs/superpowers/specs/2026-05-11-honest-cleanup-design.md` and this plan. If other files reference the deleted docs, open them and remove the references in this same commit.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
docs: rewrite CLAUDE.md/README, caveat 2024 backtest, purge ops docs

Replaces the "10/10 institutional-grade" framing and the +42.68%
flagship claim with an honest "experimental, paper-only" status.
Prepends a statistical-significance caveat to backtest_report_2024.md.
Removes 20 docs whose subject matter (incident escalation, production
readiness, multi-agent execution, phase enhancements) was operational
cosplay for a solo paper bot.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fix install (matplotlib + lazy load)

**Files:**
- Modify: `utils/__init__.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

- [ ] **Step 1: Make `utils/__init__.py` lazy on visualization**

Replace the entire contents of `utils/__init__.py` with:

```python
"""
Utilities package for the trading bot.

Visualization helpers are not re-exported at package level because they
pull in matplotlib, which is an optional dependency for headless runs.
Import them explicitly from `utils.visualization` when needed.
"""

from utils.database import (
    DailyMetrics,
    DatabaseError,
    Position,
    Trade,
    TradingDatabase,
    create_database,
)

__all__ = [
    "TradingDatabase",
    "Trade",
    "DailyMetrics",
    "Position",
    "DatabaseError",
    "create_database",
]
```

- [ ] **Step 2: Add matplotlib to `requirements.txt`**

Edit `requirements.txt` to include matplotlib. The current file lacks it. Add this line in the data-science block (after `numexpr`/`bottleneck`):

```
matplotlib>=3.7.0  # Used by utils.visualization for performance reports
```

- [ ] **Step 3: Confirm pyproject.toml matches**

Read `pyproject.toml`. If it declares dependencies, ensure `matplotlib>=3.7.0` is also there. If `pyproject.toml` declares no dependencies (deferring to `requirements.txt`), skip.

- [ ] **Step 4: Verify clean-env install + test collection**

```bash
pip install matplotlib>=3.7.0
pytest tests/unit/ --collect-only -q 2>&1 | tail -20
```

Expected: pytest collects tests without `ModuleNotFoundError: No module named 'matplotlib'`. Other collection errors (e.g. missing torch) are expected at this point — they will be fixed by deletions in Task 3.

- [ ] **Step 5: Find any other `from utils import` callers that relied on visualization re-export**

```bash
grep -rn "from utils import.*plot_\|from utils import.*create_performance_report" --include="*.py" .
```

For any hits, change them to `from utils.visualization import ...`.

- [ ] **Step 6: Commit**

```bash
git add utils/__init__.py requirements.txt pyproject.toml
# Also any callers updated in Step 5
git commit -m "$(cat <<'EOF'
fix: add matplotlib to requirements and lazy-load visualization

The package __init__ eagerly imported utils.visualization, which
imports matplotlib at module load. matplotlib wasn't in
requirements.txt, so `pip install -r requirements.txt && pytest` failed
collection on a fresh environment. Drop the eager re-export and add
matplotlib explicitly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Delete batch 1 — ML / LLM / alt-data

**Files (delete):**
- `ml/lstm_predictor.py`, `ml/rl_agent.py`, `ml/ensemble_predictor.py`, `ml/hyperparameter_optimizer.py`, `ml/feature_importance.py`, `ml/ml_pipeline.py`, `ml/torch_utils.py`
- `llm/` (entire directory)
- `data/data_fetchers/` (entire directory)
- `data/llm_providers/` (entire directory)
- `data/social_sentiment_advanced.py`, `data/order_flow_analyzer.py`, `data/web_scraper.py`, `data/alternative_data_provider.py`, `data/alt_data_types.py`
- `utils/news_sentiment.py`
- `strategies/lstm_enhanced_strategy.py`
- Their tests under `tests/unit/` and `tests/integration/`

**Files (modify):**
- `strategies/adaptive_strategy.py` — strip imports/branches for ensemble, ML, alt-data, LLM, cross-asset.
- `requirements.txt` — remove torch, torchvision, torchaudio, transformers, openai, anthropic, praw if present.
- `pytest.ini` — remove `-W error::RuntimeWarning:strategies.ensemble_strategy` line (ensemble_strategy delete is in Task 4 but its pytest reference can go now).

- [ ] **Step 1: Discover all references to the modules being deleted**

```bash
for mod in ml.lstm_predictor ml.rl_agent ml.ensemble_predictor ml.hyperparameter_optimizer ml.feature_importance ml.ml_pipeline ml.torch_utils llm.llm_client llm.llm_types data.data_fetchers data.llm_providers data.social_sentiment_advanced data.order_flow_analyzer data.web_scraper data.alternative_data_provider data.alt_data_types utils.news_sentiment strategies.lstm_enhanced_strategy; do
  echo "=== $mod ==="
  grep -rn "from $mod\|import $mod" --include="*.py" . 2>/dev/null | grep -v "^./.git" | grep -v "/research/"
done
```

Write the list of caller files to a scratch note. Each one will need its imports stripped before the module file is removed.

- [ ] **Step 2: Strip imports from `strategies/adaptive_strategy.py`**

Open `strategies/adaptive_strategy.py`. Identify all blocks gated by `enable_ensemble`, `enable_ml_signals`, `enable_signal_aggregator`, `enable_cross_asset`, `enable_portfolio_optimizer`, `enable_alt_data`, `enable_llm`, or any code path importing from `ml.*`, `llm.*`, `data.cross_asset*`, `data.alternative_data_provider`, `data.llm_providers`, `data.data_fetchers`, `data.social_sentiment_advanced`, `data.order_flow_analyzer`, `data.web_scraper`. Remove:
- The imports themselves.
- The constructor parameters (`enable_ensemble=False`, etc.) — strip these and any default-argument references.
- The conditional code blocks that use them.
- Any `self.*` assignments seeded from those parameters.

The simplified `adaptive_strategy.py` should only:
1. Detect regime via `MarketRegimeDetector`.
2. Route to `MomentumStrategy` or `MeanReversionStrategy` based on regime.
3. Apply position-multiplier from regime detection.

Run after edits:

```bash
python -c "from strategies.adaptive_strategy import AdaptiveStrategy; print('ok')"
```

Expected: prints `ok` with no import errors.

- [ ] **Step 3: Strip the same imports from `main.py` and `live_trader.py`**

Run again the grep from Step 1 against the caller files. For each remaining caller (likely `main.py`, `live_trader.py`, possibly `run_adaptive.py`, possibly `engine/strategy_manager.py`), remove the imports and any code that uses them.

If a CLI flag references a deleted strategy (e.g., `--strategy LSTMEnhancedStrategy`), delete that flag's handling.

Run after edits:

```bash
python -c "import main; print('ok')"
python -c "import live_trader; print('ok')"
python -c "import run_adaptive; print('ok')"
```

Expected: all three print `ok`. If any fails, the error names the next caller to fix.

- [ ] **Step 4: Delete the module files**

```bash
git rm ml/lstm_predictor.py ml/rl_agent.py ml/ensemble_predictor.py ml/hyperparameter_optimizer.py ml/feature_importance.py ml/ml_pipeline.py ml/torch_utils.py
git rm strategies/lstm_enhanced_strategy.py
git rm utils/news_sentiment.py
git rm -r llm/
git rm -r data/data_fetchers/ data/llm_providers/
git rm data/social_sentiment_advanced.py data/order_flow_analyzer.py data/web_scraper.py data/alternative_data_provider.py data/alt_data_types.py
```

Run:

```bash
ls ml/
ls data/
```

Expected: `ml/` contains only `__init__.py` (if anything). `data/` contains `__init__.py`, `cross_asset_provider.py`, `cross_asset_types.py`, `feature_store.py`, `point_in_time.py`, `tick_data.py`, `README.md` (those go to research/ in Task 5).

If `ml/` is now empty except `__init__.py`, `git rm ml/__init__.py` too and remove the directory.

- [ ] **Step 5: Delete the corresponding test files**

```bash
ls tests/unit/ | grep -E "lstm|rl_agent|ensemble_predictor|hyperparameter|feature_importance|news_sentiment|social_sentiment|order_flow|web_scraper|alt_data|alternative_data|llm|earnings|fed_speech|sec_edgar|news_theme"
```

For each matching file, `git rm tests/unit/<file>`. Same sweep against `tests/integration/`.

- [ ] **Step 6: Strip deleted-feature heavy dependencies from `requirements.txt`**

Open `requirements.txt`. Remove the lines for `torch`, `torchvision`, `torchaudio`, `transformers`. Also remove `openai`, `anthropic`, `praw` if they're present (they may not be — check first).

- [ ] **Step 7: Clean pytest.ini warning filters**

Open `pytest.ini`. In the `addopts` block, remove the lines that reference deleted modules:

```
    -W error::RuntimeWarning:strategies.ensemble_strategy
```

(More cleanup of pytest.ini happens in Tasks 4 and 5; only the ensemble_strategy filter goes here as a placeholder.)

- [ ] **Step 8: Run tests, fix any remaining import errors**

```bash
pytest tests/unit/ --no-cov -x -q 2>&1 | tail -30
```

If a test fails to collect because it imported a deleted module, delete that test file (it tests deleted functionality). If a non-test source file fails to import because of a leftover reference, fix the reference.

Loop this step until `pytest tests/unit/ --no-cov -q` either passes fully or only fails on assertions in tests of modules that still exist (those are bugs that pre-date this work — note them but don't fix in this commit).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: delete unvalidated ML/LLM/alt-data modules

Removes LSTM, DQN, ensemble predictor, hyperparameter optimizer,
feature-importance scoring, the entire LLM analysis pipeline
(client + earnings/Fed/SEC/news analyzers + data fetchers), the
alternative-data framework (Reddit/Glassdoor/jobs/dark-pool/options-
flow), and the FinBERT news sentiment module. None of these had any
validation evidence of edge. Also drops torch / transformers from
requirements (saves ~3GB install footprint) and strips the
corresponding branches from AdaptiveStrategy, main.py, and
live_trader.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Delete batch 2 — strategy variants + options

**Files (delete):**
- `strategies/bracket_momentum_strategy.py`
- `strategies/extended_hours_strategy.py`
- `strategies/gap_trading_strategy.py`
- `strategies/enhanced_momentum_strategy.py`
- `strategies/ensemble_strategy.py`
- `strategies/ensemble_voting_strategy.py`
- `brokers/options_broker.py`
- Their tests

**Files (modify):**
- `live_trader.py` — strip `from strategies.bracket_momentum_strategy import BracketMomentumStrategy` and any code that uses it. If `live_trader.py` is the only consumer of bracket variant, that's expected.
- `main.py` — strip any references to these strategy classes.
- `pytest.ini` — remove `-W error::RuntimeWarning:strategies.bracket_momentum_strategy`.

- [ ] **Step 1: Discover callers**

```bash
for cls in BracketMomentumStrategy ExtendedHoursStrategy GapTradingStrategy EnhancedMomentumStrategy EnsembleStrategy EnsembleVotingStrategy OptionsBroker; do
  echo "=== $cls ==="
  grep -rn "$cls" --include="*.py" . 2>/dev/null | grep -v "^./.git" | grep -v "/research/" | grep -v "test_"
done
```

- [ ] **Step 2: Strip imports + usages from callers**

For each caller surfaced in Step 1: open, remove the import, remove the code that uses the class. If the class was registered in a strategy-discovery dict, remove the dict entry.

Verify:

```bash
python -c "import main; import live_trader; import run_adaptive; print('ok')"
```

- [ ] **Step 3: Delete the module files**

```bash
git rm strategies/bracket_momentum_strategy.py strategies/extended_hours_strategy.py strategies/gap_trading_strategy.py strategies/enhanced_momentum_strategy.py strategies/ensemble_strategy.py strategies/ensemble_voting_strategy.py
git rm brokers/options_broker.py
```

- [ ] **Step 4: Delete the corresponding tests**

```bash
ls tests/unit/ | grep -E "bracket_momentum|extended_hours|gap_trading|enhanced_momentum|ensemble_strategy|ensemble_voting|options_broker"
```

`git rm` each match. Same against `tests/integration/`.

- [ ] **Step 5: Clean pytest.ini**

Remove from `addopts`:

```
    -W error::RuntimeWarning:strategies.bracket_momentum_strategy
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/unit/ --no-cov -x -q 2>&1 | tail -20
```

Expected: collection succeeds, no import errors from deleted modules.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: delete unvalidated strategy variants and options module

Removes BracketMomentumStrategy, ExtendedHoursStrategy,
GapTradingStrategy, EnhancedMomentumStrategy, EnsembleStrategy,
EnsembleVotingStrategy, and the OptionsBroker. The four momentum
variants were unvalidated forks of MomentumStrategy; the ensembles
combined components that themselves had no proven edge; options
trading is a separate specialty that doesn't belong as a bolt-on.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Quarantine to `research/`

**Files (move into `research/`):**
- `strategies/factor_models.py`, `strategies/factor_portfolio.py`, `strategies/factor_screener.py`, `strategies/pairs_trading_strategy.py`
- `engine/factor_attribution.py`, `engine/walk_forward.py`, `engine/statistical_tests.py`, `engine/validated_backtest.py`
- `utils/factor_data.py`, `utils/alpha_decay_monitor.py`, `utils/ic_tracker.py`, `utils/historical_universe.py`, `utils/extended_hours.py`, `utils/market_impact.py`
- `factors/` (entire directory)
- `data/cross_asset_provider.py`, `data/cross_asset_types.py`, `data/feature_store.py`, `data/point_in_time.py`, `data/tick_data.py`
- Their tests → `research/tests/`

**Files (modify):**
- `pytest.ini` — remove warning filters referencing moved modules; add `research/` to `norecursedirs` or set `testpaths = tests` to exclude `research/` (already is `testpaths = tests`, so just remove the filters).
- Any caller in `main.py` / `live_trader.py` / `run_adaptive.py` / `strategies/__init__.py` referencing moved strategies — strip.

- [ ] **Step 1: Create the `research/` tree**

```bash
mkdir -p research/strategies research/engine research/utils research/data research/factors research/tests/unit
```

- [ ] **Step 2: Write `research/README.md`**

Create `research/README.md`:

```markdown
# research/

This directory holds plausible-but-unvalidated quant work that was moved out of the production code path during the 2026-05 cleanup. **Nothing in here is imported by the production path, run by default tests, or trusted to produce signal.** Code here is preserved as ideas, not products.

## Contents

- `strategies/` — factor models, factor portfolios, factor screener, pairs trading.
- `engine/` — factor attribution, walk-forward validation, statistical tests, validated backtest.
- `utils/` — factor data pipeline, alpha-decay monitor, IC tracker, historical universe (point-in-time), extended-hours helpers, market-impact model.
- `factors/` — individual factor implementations (value, quality, momentum, low-vol, size, sentiment, growth, earnings, reversal, volatility, orthogonalization).
- `data/` — cross-asset (VIX/yield-curve/FX), feature store, point-in-time, tick data.
- `tests/` — tests that targeted the moved modules.

## Bringing a module back to production

Don't, unless you have:

1. A backtest with ≥50 trades, real slippage, and an out-of-sample period.
2. A statistical-significance check (e.g. permutation test, FDR-corrected).
3. A written hypothesis about why the signal should work.
4. Evidence that the signal isn't already priced into the symbols you trade.

Without those, the module stays here.
```

- [ ] **Step 3: Move strategy files**

```bash
git mv strategies/factor_models.py research/strategies/factor_models.py
git mv strategies/factor_portfolio.py research/strategies/factor_portfolio.py
git mv strategies/factor_screener.py research/strategies/factor_screener.py
git mv strategies/pairs_trading_strategy.py research/strategies/pairs_trading_strategy.py
```

- [ ] **Step 4: Move engine files**

```bash
git mv engine/factor_attribution.py research/engine/factor_attribution.py
git mv engine/walk_forward.py research/engine/walk_forward.py
git mv engine/statistical_tests.py research/engine/statistical_tests.py
git mv engine/validated_backtest.py research/engine/validated_backtest.py
```

- [ ] **Step 5: Move utils files**

```bash
git mv utils/factor_data.py research/utils/factor_data.py
git mv utils/alpha_decay_monitor.py research/utils/alpha_decay_monitor.py
git mv utils/ic_tracker.py research/utils/ic_tracker.py
git mv utils/historical_universe.py research/utils/historical_universe.py
git mv utils/extended_hours.py research/utils/extended_hours.py
git mv utils/market_impact.py research/utils/market_impact.py
```

- [ ] **Step 6: Move data files**

```bash
git mv data/cross_asset_provider.py research/data/cross_asset_provider.py
git mv data/cross_asset_types.py research/data/cross_asset_types.py
git mv data/feature_store.py research/data/feature_store.py
git mv data/point_in_time.py research/data/point_in_time.py
git mv data/tick_data.py research/data/tick_data.py
```

- [ ] **Step 7: Move `factors/` whole**

```bash
git mv factors research/factors
```

- [ ] **Step 8: Move test files for everything quarantined**

```bash
for pattern in factor_models factor_portfolio factor_screener factor_attribution pairs_trading walk_forward statistical_tests validated_backtest factor_data alpha_decay ic_tracker historical_universe extended_hours market_impact cross_asset feature_store point_in_time tick_data; do
  for f in tests/unit/test_*${pattern}*.py; do
    [ -e "$f" ] && git mv "$f" "research/tests/unit/$(basename $f)"
  done
done
ls tests/unit/ | grep -E "factor|pairs_trading|walk_forward|validated|cross_asset|alpha_decay|ic_tracker|historical_universe|market_impact|tick_data|feature_store|point_in_time" || echo "all quarantine-related tests moved"
```

- [ ] **Step 9: Strip imports of moved modules from production callers**

```bash
for mod in strategies.factor_models strategies.factor_portfolio strategies.factor_screener strategies.pairs_trading_strategy engine.factor_attribution engine.walk_forward engine.statistical_tests engine.validated_backtest utils.factor_data utils.alpha_decay_monitor utils.ic_tracker utils.historical_universe utils.extended_hours utils.market_impact data.cross_asset_provider data.cross_asset_types data.feature_store data.point_in_time data.tick_data factors; do
  echo "=== $mod ==="
  grep -rn "from $mod\|import $mod" --include="*.py" . 2>/dev/null | grep -v "^./research/" | grep -v "^./.git"
done
```

For each non-research hit, open the file, strip the import and any code that uses the symbol. Most likely callers: `main.py`, `live_trader.py`, `run_adaptive.py`, `strategies/__init__.py`, `engine/__init__.py`, `utils/__init__.py`, `data/__init__.py`.

For `strategies/adaptive_strategy.py`, the cross-asset / pairs / factor branches should already have been stripped in Task 3 Step 2; verify they're gone.

- [ ] **Step 10: Clean pytest.ini warning filters for moved modules**

In `pytest.ini` `addopts`, remove:

```
    -W error::RuntimeWarning:strategies.factor_models
    -W error::RuntimeWarning:engine.factor_attribution
    -W error::RuntimeWarning:utils.market_impact
    -W error::FutureWarning:utils.factor_data
```

Also remove from the `--cov` invocations any modules that moved (`engine.validated_backtest`, `engine.walk_forward`). Update:

```
    --cov=engine.backtest_engine
    --cov=engine.performance_metrics
```

(drop the validated_backtest and walk_forward lines)

And in `[coverage:run]`:

```
source = engine.backtest_engine,engine.performance_metrics
```

- [ ] **Step 11: Run tests**

```bash
pytest tests/ --no-cov -q 2>&1 | tail -20
```

Expected: all collection succeeds, all tests pass or fail only on assertions (not imports). Tests for quarantined modules should be gone from `tests/`.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: quarantine unvalidated quant modules to research/

Moves factor models, factor portfolios, pairs trading, walk-forward
validation, statistical tests, validated backtest, alpha-decay monitor,
IC tracker, point-in-time data, historical universe, market impact,
cross-asset (VIX/yield-curve/FX) providers, tick data, feature store,
and the factors/ directory into research/. None of these are imported
by the production path; pytest excludes research/ from default runs.
research/README.md documents the bar for promoting anything back.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Scripts + CI + ops-utils purge

**Files (delete in `scripts/`):**
chaos_drill.py, deploy_canary.py, rollback_drill.py, runtime_industrial_gate.py, runtime_watchdog.py, staging_incident_ticket_drill.py, validate_incident_contacts.py, validate_incident_ticket_drill_evidence.py, validate_incident_ticketing.py, incident_response_automation.py, incident_ack.py, replay_notification_dead_letters.py, push_ops_metrics.py, export_ops_metrics.py, governance_gate.py, secrets_audit.py, shadow_drift_dashboard.py, strategy_promotion_gate.py, paper_burn_in_scorecard.py, fault_injection_matrix.py, feature_comparison_backtest.py, generate_validation_artifacts.py, deployment_preflight.py, validated_backtest_report.py, validate_alternative_data.py, ops_status_report.py, mock_strategies.py, run_low_resource_profile.py, run_now.py, mcp_server.py, mcp.json, enhanced_dashboard.py.

**Files (delete at root):**
`incident_contacts.py`.

**Files (delete in `utils/`) — operational infra:**
`utils/governance_gate.py`, `utils/incident_tracker.py`, `utils/slo_monitor.py`, `utils/slo_alerting.py`, `utils/order_reconciliation.py`, `utils/reconciliation.py`, `utils/run_artifacts.py`, `utils/live_broker_factory.py`, `utils/data_quality.py`, `utils/simple_symbol_selector.py`, `utils/symbol_scope.py`, `utils/runtime_state.py`, `utils/order_gateway.py`, `utils/position_manager.py`.

(Verify each before deleting — some may already be needed by kept code. Use the discovery step.)

**Files (delete in `.github/workflows/`):**
`nightly-incident-ticket-drill.yml`, `org-osv.yml`, `org-gitleaks.yml`, `org-trivy.yml`, `org-trufflehog.yml`, `org-precommit.yml`, `org-release-please.yml`, `org-codeql.yml`, `org-ci-tests.yml`, `security-baseline.yml`, `semgrep.yml`. (Keep `ci.yml`, `docker-build.yml`, `trading_bot.yml`.)

**Files (modify):**
- `main.py` and `live_trader.py` — strip imports + code paths for deleted ops utils.
- Possibly the kept workflows if they reference deleted scripts.

- [ ] **Step 1: Delete the scripts**

```bash
cd scripts
git rm chaos_drill.py deploy_canary.py rollback_drill.py runtime_industrial_gate.py runtime_watchdog.py staging_incident_ticket_drill.py validate_incident_contacts.py validate_incident_ticket_drill_evidence.py validate_incident_ticketing.py incident_response_automation.py incident_ack.py replay_notification_dead_letters.py push_ops_metrics.py export_ops_metrics.py governance_gate.py secrets_audit.py shadow_drift_dashboard.py strategy_promotion_gate.py paper_burn_in_scorecard.py fault_injection_matrix.py feature_comparison_backtest.py generate_validation_artifacts.py deployment_preflight.py validated_backtest_report.py validate_alternative_data.py ops_status_report.py mock_strategies.py run_low_resource_profile.py run_now.py mcp_server.py mcp.json enhanced_dashboard.py
cd ..
git rm incident_contacts.py
ls scripts/
```

Expected: `scripts/` contains README.md, kill_switch.py, dashboard.py, simple_backtest.py, quickstart.py, run.py, simple_trader.py.

- [ ] **Step 2: Delete corresponding test files**

```bash
ls tests/unit/ | grep -E "secrets_audit|push_ops_metrics|shadow_drift|deploy_canary|incident_response|incident_ticket|incident_tracker|incident_ack|governance|go_live_precheck|paper_burn_in|fault_injection|strategy_promotion|validate_incident|runtime_industrial|runtime_watchdog|run_artifacts|order_reconciliation|order_gateway|data_quality|slo_monitor|slo_alerting|live_broker_factory|live_broker|reconciliation|deployment_hardening|chaos|live_validation|main_runtime"
```

`git rm` each match. Same against `tests/integration/`.

- [ ] **Step 3: Discover all callers of operational utils to be deleted**

```bash
for mod in utils.governance_gate utils.incident_tracker utils.slo_monitor utils.slo_alerting utils.order_reconciliation utils.reconciliation utils.run_artifacts utils.live_broker_factory utils.data_quality utils.simple_symbol_selector utils.symbol_scope utils.runtime_state utils.order_gateway utils.position_manager; do
  echo "=== $mod ==="
  grep -rn "from $mod\|import $mod" --include="*.py" . 2>/dev/null | grep -v "^./research/" | grep -v "^./.git" | grep -v "test_"
done
```

Expected callers: `main.py`, `live_trader.py`, possibly other utils that reference each other.

- [ ] **Step 4: Strip operational code from `main.py` and `live_trader.py`**

For both files:

1. Remove imports of the operational utils.
2. Remove constructor calls, e.g. `SLOMonitor(...)`, `IncidentTracker(...)`, `OrderGateway(...)`, `PositionReconciler(...)`, `OrderReconciler(...)`, `RuntimeStateStore(...)`, `build_slo_alert_notifier(...)`, `build_incident_ticket_notifier(...)`, `create_live_broker(...)`, `shutdown_live_broker_failover(...)`.
3. Remove the code that uses them — SLO event emission, incident-ack tracking, reconciliation loops, runtime-state save/load, multi-broker failover loops, governance gate calls.
4. Where `create_live_broker` was used, replace with direct `AlpacaBroker(...)` instantiation:

```python
from brokers.alpaca_broker import AlpacaBroker
broker = AlpacaBroker(paper=True)
```

5. Where `OrderGateway` wrapped a broker, replace the wrapping with the broker itself — `OrderGateway` was only there for the deleted gateway-enforcement layer.

After edits, both files should be substantially shorter (target: `main.py` < 1000 LOC, `live_trader.py` < 600 LOC).

Verify imports:

```bash
python -c "import main; import live_trader; import run_adaptive; print('ok')"
```

- [ ] **Step 5: Delete the operational util modules**

```bash
git rm utils/governance_gate.py utils/incident_tracker.py utils/slo_monitor.py utils/slo_alerting.py utils/order_reconciliation.py utils/reconciliation.py utils/run_artifacts.py utils/live_broker_factory.py utils/data_quality.py utils/simple_symbol_selector.py utils/symbol_scope.py utils/runtime_state.py utils/order_gateway.py utils/position_manager.py
```

If any of those weren't found in Step 1 inventory (e.g. `utils/position_manager.py` might not exist), skip them.

- [ ] **Step 6: Delete the GitHub workflows for deleted ops**

```bash
cd .github/workflows
git rm nightly-incident-ticket-drill.yml org-osv.yml org-gitleaks.yml org-trivy.yml org-trufflehog.yml org-precommit.yml org-release-please.yml org-codeql.yml org-ci-tests.yml security-baseline.yml semgrep.yml
cd ../..
ls .github/workflows/
```

Expected: `ci.yml`, `docker-build.yml`, `trading_bot.yml`.

- [ ] **Step 6b: Decide fate of `infra/`**

`infra/` contains `README.md` and a `systemd/` directory. Open both and read. If the systemd units start any of the scripts deleted in Step 1 (chaos_drill, runtime_watchdog, ops_metrics push, incident response automation, etc.), delete the whole `infra/` tree:

```bash
git rm -r infra/
```

If a unit still references a kept script (e.g. `kill_switch.py` or `run.py`), keep just that unit and delete the rest.

- [ ] **Step 7: Trim remaining workflows**

Open `ci.yml`, `docker-build.yml`, `trading_bot.yml`. Remove any step that calls a deleted script (e.g. `python scripts/deployment_preflight.py`, `python scripts/runtime_industrial_gate.py`, `python scripts/staging_incident_ticket_drill.py`, etc.).

Each workflow should ultimately just:
- Install deps,
- Run `pytest tests/`,
- Run lint (black, ruff),
- (For `docker-build.yml`) build the Docker image.

Anything else — delete the step.

- [ ] **Step 8: Run tests**

```bash
pytest tests/unit/ --no-cov -q 2>&1 | tail -20
```

Expected: all tests pass or fail only on assertions, not imports.

- [ ] **Step 9: Smoke test the entry points**

```bash
python main.py --help
python live_trader.py --help 2>&1 | head -10
python run_adaptive.py --help
python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-03-31 2>&1 | tail -30
```

Expected: `--help` works for all three. The 3-month backtest runs to completion (it may produce poor returns — that's fine, we're testing the pipeline, not the strategy).

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: purge operational infrastructure and CI workflows

Removes 32 scripts (chaos drills, canary deploy, incident response,
SLO export, governance gate, secrets audit, runtime watchdogs, etc.)
and 14 utils modules that supported them. Trims main.py and
live_trader.py to use AlpacaBroker directly without the deleted
OrderGateway/SLOMonitor/IncidentTracker/multi-broker-failover layers.
Deletes the GitHub workflows that ran those scripts. Kept workflows
(ci.yml, docker-build.yml, trading_bot.yml) now only run tests + lint
+ docker build.

This was infrastructure scoped for a fund-grade operation; the actual
product is a solo paper bot. Smaller surface area, fewer false
production-readiness signals.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Config trim

**Files (modify):**
- `config.py`
- `.env.example`

- [ ] **Step 1: Identify the parameter blocks to remove**

Open `config.py`. The block names to remove are: `CRYPTO_PARAMS`, `OVERNIGHT_PARAMS`, `ML_PARAMS`, `RL_PARAMS`, `OPTIONS_PARAMS`, `SENTIMENT_PARAMS`, `LLM_PARAMS`. Also any helper functions or constants used only by them.

- [ ] **Step 2: Verify no kept code reads from these blocks**

```bash
for var in CRYPTO_PARAMS OVERNIGHT_PARAMS ML_PARAMS RL_PARAMS OPTIONS_PARAMS SENTIMENT_PARAMS LLM_PARAMS; do
  echo "=== $var ==="
  grep -rn "$var" --include="*.py" . 2>/dev/null | grep -v "^./research/" | grep -v "^./.git" | grep -v "config.py:"
done
```

If anything kept references these blocks (it shouldn't, after Tasks 3–6), strip the reference now.

- [ ] **Step 3: Remove the blocks from `config.py`**

Edit `config.py`, deleting each named block and any imports / helpers only used by them. Run:

```bash
python -c "from config import TRADING_PARAMS, RISK_PARAMS, TECHNICAL_PARAMS, SYMBOLS; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Trim `.env.example`**

Open `.env.example`. Remove environment variables that only the deleted features used:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `ALPHA_VANTAGE_API_KEY`, `FMP_API_KEY`
- Any `LLM_*`, `INCIDENT_*`, `SLO_PAGING_*`, `MULTI_BROKER_*`, `IB_*`, `NOTIFICATION_DEAD_LETTER_*`, `INCIDENT_TICKETING_*`, `INCIDENT_ACK_*`, `INCIDENT_RESPONSE_*`, `INCIDENT_DRILL_*`, `PROMOTION_*`
- Anything else only the deleted features consumed

Keep: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `PAPER`, `DISCORD_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DATABASE_URL`.

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/ --no-cov -q 2>&1 | tail -10
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add config.py .env.example
git commit -m "$(cat <<'EOF'
chore: trim config blocks for deleted features

Removes CRYPTO_PARAMS, OVERNIGHT_PARAMS, ML_PARAMS, RL_PARAMS,
OPTIONS_PARAMS, SENTIMENT_PARAMS, LLM_PARAMS from config.py and the
corresponding entries from .env.example. config.py shrinks from
~800 lines to the trading/risk/technical core.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Honest baseline backtest

**Files (create):**
- `scripts/run_honest_baseline.py`
- `results/honest_backtest_2020-2024.md`
- `results/honest_backtest_2020-2024.json` (raw artifact)

- [ ] **Step 1: Write `scripts/run_honest_baseline.py`**

Create the file with the following content:

```python
"""Run the honest baseline backtest defined by the 2026-05 cleanup spec.

Output:
    results/honest_backtest_2020-2024.json   - raw metrics + trade log
    results/honest_backtest_2020-2024.md     - human-readable report

Usage:
    python scripts/run_honest_baseline.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from brokers.backtest_broker import BacktestBroker
from engine.backtest_engine import BacktestEngine
from engine.performance_metrics import PerformanceMetrics
from strategies.momentum_strategy import MomentumStrategy

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM"]
START = "2020-01-01"
END = "2024-12-31"
INITIAL_CAPITAL = 100_000
SLIPPAGE_BPS = 40   # 0.40% per trade
SPREAD_BPS = 10     # 0.10%
MIN_TRADES_FOR_SIGNIFICANCE = 50


async def main() -> None:
    broker = BacktestBroker(
        initial_capital=INITIAL_CAPITAL,
        slippage_bps=SLIPPAGE_BPS,
        spread_bps=SPREAD_BPS,
    )
    engine = BacktestEngine(broker=broker)

    result = await engine.run_backtest(
        strategy_class=MomentumStrategy,
        symbols=SYMBOLS,
        start_date=START,
        end_date=END,
    )

    metrics = PerformanceMetrics.from_equity_curve(
        equity_curve=result.equity_curve,
        trades=result.trades,
    )

    n_trades = len(result.trades)
    inconclusive = n_trades < MIN_TRADES_FOR_SIGNIFICANCE

    artifact = {
        "spec_ref": "docs/superpowers/specs/2026-05-11-honest-cleanup-design.md",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "initial_capital": INITIAL_CAPITAL,
            "slippage_bps": SLIPPAGE_BPS,
            "spread_bps": SPREAD_BPS,
            "min_trades_for_significance": MIN_TRADES_FOR_SIGNIFICANCE,
        },
        "n_trades": n_trades,
        "inconclusive": inconclusive,
        "metrics": metrics.to_dict(),
        "trades": [t.to_dict() for t in result.trades],
        "equity_curve_summary": {
            "start_equity": result.equity_curve[0],
            "end_equity": result.equity_curve[-1],
            "n_days": len(result.equity_curve),
        },
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "honest_backtest_2020-2024.json").write_text(
        json.dumps(artifact, indent=2, default=str)
    )

    md = _format_markdown(artifact)
    (results_dir / "honest_backtest_2020-2024.md").write_text(md)
    print(f"Wrote results/honest_backtest_2020-2024.{{json,md}}")
    print(f"Trades: {n_trades}  Inconclusive: {inconclusive}")


def _format_markdown(artifact: dict) -> str:
    cfg = artifact["config"]
    m = artifact["metrics"]
    inconclusive = artifact["inconclusive"]
    n_trades = artifact["n_trades"]

    header = (
        "# Honest baseline backtest 2020-2024\n\n"
        f"Generated: {artifact['generated_at']}\n"
        f"Spec: `{artifact['spec_ref']}`\n\n"
    )

    if inconclusive:
        header += (
            f"> **Status: INCONCLUSIVE.** Strategy produced {n_trades} trades, "
            f"below the {cfg['min_trades_for_significance']}-trade significance bar set by "
            "this repo's `PROFITABILITY_RESEARCH.md`. The numbers below are reported "
            "for transparency but should not be cited as evidence of strategy edge.\n\n"
        )
    else:
        header += f"Trades: **{n_trades}** (meets the {cfg['min_trades_for_significance']}-trade significance bar).\n\n"

    config_block = (
        "## Configuration\n\n"
        f"- **Strategy:** `MomentumStrategy` (default parameters)\n"
        f"- **Symbols:** {', '.join(cfg['symbols'])}\n"
        f"- **Period:** {cfg['start']} → {cfg['end']}\n"
        f"- **Initial capital:** ${cfg['initial_capital']:,}\n"
        f"- **Slippage:** {cfg['slippage_bps']} bps per trade\n"
        f"- **Spread:** {cfg['spread_bps']} bps\n\n"
    )

    metrics_block = (
        "## Headline metrics\n\n"
        f"- **Total return:** {m.get('total_return', float('nan')):.2%}\n"
        f"- **Annualized return:** {m.get('annualized_return', float('nan')):.2%}\n"
        f"- **Sharpe ratio:** {m.get('sharpe_ratio', float('nan')):.2f}\n"
        f"- **Sortino ratio:** {m.get('sortino_ratio', float('nan')):.2f}\n"
        f"- **Calmar ratio:** {m.get('calmar_ratio', float('nan')):.2f}\n"
        f"- **Max drawdown:** {m.get('max_drawdown', float('nan')):.2%}\n"
        f"- **Win rate:** {m.get('win_rate', float('nan')):.2%}\n"
        f"- **Profit factor:** {m.get('profit_factor', float('nan')):.2f}\n\n"
    )

    trades_block = "## Trade log\n\n| # | Symbol | Side | Entry | Exit | P&L | Days held |\n|---|--------|------|-------|------|-----|-----------|\n"
    for i, t in enumerate(artifact["trades"], 1):
        trades_block += (
            f"| {i} | {t.get('symbol','')} | {t.get('side','')} | "
            f"{t.get('entry_price','')} | {t.get('exit_price','')} | "
            f"{t.get('pnl','')} | {t.get('days_held','')} |\n"
        )

    interpretation = (
        "\n## Interpretation\n\n"
        "This is the single performance number cited by `README.md` and "
        "`CLAUDE.md`. It supersedes `backtest_report_2024.md` (9 trades) and "
        "any earlier in-CLAUDE.md claims.\n\n"
        "Do not extrapolate beyond what the trade count supports. Sharpe values "
        "computed on small samples have wide confidence intervals; "
        "`PROFITABILITY_RESEARCH.md` documents realistic expectations for "
        "this strategy family (Sharpe 0.5–1.2 net of costs).\n"
    )

    return header + config_block + metrics_block + trades_block + interpretation


if __name__ == "__main__":
    asyncio.run(main())
```

**Important compatibility notes for the executor:**

If `BacktestBroker.__init__` does not accept `slippage_bps` / `spread_bps` keyword arguments, inspect its signature and adapt to whatever the broker exposes (e.g. `slippage`, `slippage_pct`). Same for `BacktestEngine.run_backtest` — adapt the call to match the real signature. The script's *behavior* is the requirement, not the exact API. If `PerformanceMetrics.from_equity_curve` doesn't exist, use whichever constructor the kept code provides.

- [ ] **Step 2: Sanity-check the script against the actual APIs**

```bash
python -c "from brokers.backtest_broker import BacktestBroker; import inspect; print(inspect.signature(BacktestBroker.__init__))"
python -c "from engine.backtest_engine import BacktestEngine; import inspect; print(inspect.signature(BacktestEngine.run_backtest))"
python -c "from engine.performance_metrics import PerformanceMetrics; print(dir(PerformanceMetrics))"
```

If the signatures don't match the script, edit the script to use the actual APIs.

- [ ] **Step 3: Run the backtest**

```bash
python scripts/run_honest_baseline.py
```

Expected: completes without exception, writes both files. Stdout reports trade count and inconclusive flag.

- [ ] **Step 4: Inspect the artifact**

```bash
head -40 results/honest_backtest_2020-2024.md
jq '.n_trades, .inconclusive, .metrics.sharpe_ratio, .metrics.total_return' results/honest_backtest_2020-2024.json
```

Whatever the result, accept it. If `n_trades < 50`, the markdown will say "INCONCLUSIVE" — that is the honest finding.

- [ ] **Step 5: Update CLAUDE.md and README.md to reference the new artifact**

Both files were already written in Task 1 with a pointer ("when `results/honest_backtest_2020-2024.md` exists, that is the single performance number to cite"). Verify both files still contain that line. If the actual result is dramatic (e.g. negative return, Sharpe < 0), add a one-line summary to both files:

In `README.md`, append at the end of the "Performance" section:

```markdown
**Current baseline (2020-2024, 10 symbols, MomentumStrategy defaults):** Sharpe {X.XX}, total return {Y.YY%} over 5 years, {N} trades. See `results/honest_backtest_2020-2024.md` for the full report.
```

Use the actual numbers from the run.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_honest_baseline.py results/honest_backtest_2020-2024.md results/honest_backtest_2020-2024.json README.md CLAUDE.md
git commit -m "$(cat <<'EOF'
chore: publish honest baseline backtest 2020-2024

Single canonical performance number for this repo: MomentumStrategy
defaults on 10 large-caps, 2020-2024, 40 bps slippage + 10 bps spread.
Result is reported as INCONCLUSIVE if trade count is below the
50-trade significance bar set by PROFITABILITY_RESEARCH.md. This
artifact supersedes backtest_report_2024.md and any earlier in-doc
performance claims.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Repopulate `TODO.md`

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Replace `TODO.md` with the real follow-up list**

Replace the entire contents of `TODO.md` with:

```markdown
# TODO

Follow-ups after the 2026-05 honest cleanup (`docs/superpowers/specs/2026-05-11-honest-cleanup-design.md`).

## Direction (decide before doing more work)

- [ ] Decide the actual goal: paper-only learning sandbox, path to live capital, public showcase, or something else. Different goals produce different next steps. Do not add features before deciding this.

## Code organization (deferred from cleanup)

- [ ] Measure file sizes after the cleanup. If `main.py`, `live_trader.py`, or `adaptive_strategy.py` are still over 800 LOC, split them — one module per responsibility.
- [ ] Audit the kept `utils/` modules. Some (e.g. `multi_timeframe.py`) may be vestigial after the deletions.

## Validation (if continuing toward live)

- [ ] Run 6+ months of paper trading on the kept core. Stop pretending shorter samples are meaningful.
- [ ] Produce at least 50 real trades before claiming Sharpe or win rate.
- [ ] Re-run `scripts/run_honest_baseline.py` quarterly; track drift in `results/`.

## Research-tree promotion (only if a research/ module proves itself)

- [ ] For any `research/` module being considered for promotion, require: (1) ≥50-trade out-of-sample backtest, (2) statistical-significance check (permutation or FDR), (3) written hypothesis, (4) evidence the signal isn't already priced. Document in `research/<module>/PROMOTION.md`.

## Operational (only if scaling beyond solo paper)

- [ ] If running unattended for extended periods, re-evaluate which of the deleted operational scripts (kill switch is already kept) actually need to come back. Don't restore wholesale.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "$(cat <<'EOF'
docs: repopulate TODO.md with real cleanup follow-ups

Replaces the all-checked production-hardening list (which described
work that's now deleted) with the actual outstanding items: decide
the repo's direction, audit kept-file sizes, do real paper-trading
validation before claiming performance, and gate research/ promotion
on real evidence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final acceptance check (do this after Task 9)

- [ ] **Acceptance: CLAUDE.md no longer contains the discredited claims**

```bash
grep -E "10/10|institutional-grade|\+42\.68%|Phase [1-9]|nine phases|suitable for live capital" CLAUDE.md README.md
```

Expected: no output.

- [ ] **Acceptance: clean-env install works**

```bash
python -m venv /tmp/cleanenv && /tmp/cleanenv/bin/pip install -r requirements.txt && /tmp/cleanenv/bin/pytest tests/ --co -q | tail -5
```

Expected: pip install completes; pytest collects without errors.

- [ ] **Acceptance: no torch / transformers / openai / anthropic in the production path**

```bash
grep -rn "import torch\|import transformers\|import openai\|import anthropic\|from torch\|from transformers\|from openai\|from anthropic" --include="*.py" . | grep -v "^./research/" | grep -v "^./.git" | grep -v "test_"
```

Expected: no output.

- [ ] **Acceptance: production LOC materially smaller**

```bash
find . -name "*.py" -not -path "./.git/*" -not -path "./research/*" -not -path "./tests/*" | xargs wc -l | tail -1
```

Expected: total < 80,000 (down from ~193,000).

- [ ] **Acceptance: `docs/` is small**

```bash
ls docs/ | grep -v superpowers | wc -l
```

Expected: ≤7.

- [ ] **Acceptance: `scripts/` is small**

```bash
ls scripts/ | wc -l
```

Expected: ≤8.

- [ ] **Acceptance: honest baseline exists and is the cited reference**

```bash
test -f results/honest_backtest_2020-2024.md && echo "exists"
grep -l "honest_backtest_2020-2024" README.md CLAUDE.md
```

Expected: `exists` printed; both `README.md` and `CLAUDE.md` reference the baseline.

If any acceptance check fails, open an issue or amend the relevant commit; do not declare the cleanup done.
