# Honest Cleanup — Design Spec

**Date:** 2026-05-11
**Author:** Lorenzo + Claude (audit)
**Status:** Approved, ready for implementation plan

---

## Problem statement

The repo ships ~193K lines of Python organized into "9 phases of institutional-grade features", a self-rating of "10/10 — suitable for live capital deployment", and a flagship backtest claiming +42.68% / Sharpe 2.00.

Audit findings contradict this framing:

1. **The flagship backtest has 9 trades.** Eight of twelve months show exactly 0% return. The repo's own `PROFITABILITY_RESEARCH.md` states that any backtest under 50 trades is "statistically meaningless" with a ~40% probability of being pure luck.
2. **No validation evidence exists for any "phase".** None of LSTM, DQN, factor models, LLM alpha, alternative data, cross-asset signals, options trading, or news sentiment has a stored result in `results/` proving it improves out-of-sample Sharpe or even calibrates.
3. **The test suite does not run from a clean checkout.** `pytest` fails at collection because `matplotlib` is imported at module load (`utils/__init__.py → utils/visualization`) but not in `requirements.txt`. The "98% coverage" claim was measured in a non-reproducible environment.
4. **Operational scope is fund-grade; product scope is solo-paper.** 41 scripts include chaos drills, canary deploy, incident escalation rosters, dead-letter replay, Pushgateway export, multi-broker failover probes — for a single-developer Alpaca paper account.
5. **Code-organization principles in `.windsurfrules` and `CLAUDE.md` are not followed.** `main.py` is 1,991 lines; `adaptive_strategy.py` 1,629; `live_trader.py` 1,382; `momentum_strategy.py` 1,229; `base_strategy.py` 1,105; `risk_manager.py` 1,062.

Recent commits (#20–#22: "harden runtime surfaces", "use market-session calendars in backtests", "enforce live validation evidence gates") show course-correction is underway. This spec accelerates that correction by removing the unjustified narrative and the unvalidated surface area.

## Goal

Reduce the repo to its honest, validated core. Replace marketing claims with accurate status. Make the install reproducible. Defer all new direction (paper-only, live, learning, showcase) until after the cleanup commits a clean foundation.

## Non-goals

- Add features.
- Restructure the large core files (`main.py`, `live_trader.py`, `adaptive_strategy.py`) — let deletions shrink them first.
- Try to salvage the +42.68% / Sharpe 2.0 number.
- Decide the long-term direction.

## Scope decisions

### Modules: delete vs quarantine vs keep

**Delete outright** — modules with no plausible near-term value and significant maintenance cost:

| Module / area | Files |
|---|---|
| LSTM predictor | `ml/lstm_predictor.py`, `strategies/lstm_enhanced_strategy.py` |
| RL / DQN agent | `ml/rl_agent.py` |
| LLM alpha | `llm/` (entire), `data/data_fetchers/`, `data/llm_providers/` |
| Alt-data scrapers | `data/social_sentiment_advanced.py`, `data/order_flow_analyzer.py`, `data/web_scraper.py`, `data/alternative_data_provider.py` |
| Options trading | `brokers/options_broker.py` |
| News sentiment | `utils/news_sentiment.py` |
| Ensemble strategies | `strategies/ensemble_strategy.py`, `strategies/ensemble_voting_strategy.py`, `ml/ensemble_predictor.py` |
| Bracket/extended/gap variants | `strategies/bracket_momentum_strategy.py`, `strategies/extended_hours_strategy.py`, `strategies/gap_trading_strategy.py`, `strategies/enhanced_momentum_strategy.py` |
| Hyperparameter optimizer + feature importance | `ml/hyperparameter_optimizer.py`, `ml/feature_importance.py` |

**Rationale per category:**

- *LSTM/DQN*: deep RL on daily financial data with a solo-dev sample size is well-documented to fail. The literature is unambiguous.
- *LLM alpha*: $20–50/day cost for unproven edge; rate-priced markets already process Fed/earnings text faster than any LLM pipeline can. Trivial to rebuild from API docs if the user ever wants to revisit.
- *Alt-data*: Reddit / Glassdoor / job-postings / dark-pool / options-flow either have alpha that requires panel data far beyond a solo bot's reach, or come from data sources retail APIs don't actually expose reliably.
- *Options*: separate specialty; a bolt-on options module is a risk surface, not a feature.
- *News sentiment*: 500MB FinBERT for marginal signal on the same news Alpaca already shows.
- *Ensemble*: ensembling unvalidated components doesn't validate the ensemble.
- *Bracket/extended/gap/enhanced momentum variants*: four un-validated forks of the momentum idea. Pick one (the base `MomentumStrategy`) and delete the rest.
- *Hyperparameter optimizer + SHAP*: serve only the deleted ML modules.

**Quarantine to `research/` (move, add `research/README.md` marking as unvalidated, exclude from default pytest):**

| Module / area | Files |
|---|---|
| Factor models | `strategies/factor_models.py`, `strategies/factor_portfolio.py`, `strategies/factor_screener.py`, `engine/factor_attribution.py`, `utils/factor_data.py`, `factors/` |
| Cross-asset signals | `data/cross_asset_types.py`, `data/cross_asset_provider.py` |
| Pairs trading | `strategies/pairs_trading_strategy.py` |
| Walk-forward + statistical tests | `engine/walk_forward.py`, `engine/statistical_tests.py`, `engine/validated_backtest.py` |
| Alpha decay + IC tracker | `utils/alpha_decay_monitor.py`, `utils/ic_tracker.py` |
| Crypto + extended hours / overnight | `utils/crypto_utils.py` crypto branches, `utils/extended_hours.py`, `CRYPTO_PARAMS` / `OVERNIGHT_PARAMS` in `config.py` |
| Historical universe (point-in-time) | `utils/historical_universe.py` |

**Rationale:** these are legitimate quant ideas — factor models, cross-asset regime detection, walk-forward validation, pairs trading — but none have been calibrated or validated against this codebase's data. They might be brought back later. Quarantining preserves the work without letting it execute in the live path.

**Keep — the honest core:**

- `strategies/base_strategy.py`
- `strategies/momentum_strategy.py`
- `strategies/momentum_strategy_backtest.py`
- `strategies/mean_reversion_strategy.py`
- `strategies/simple_ma_strategy.py`
- `strategies/adaptive_strategy.py` (simplified — see below)
- `strategies/risk_manager.py`
- `brokers/alpaca_broker.py`, `brokers/backtest_broker.py`, `brokers/order_builder.py`
- `engine/backtest_engine.py`, `engine/performance_metrics.py`, `engine/strategy_manager.py`
- `utils/circuit_breaker.py`, `utils/market_regime.py`, `utils/realistic_backtest.py`
- `utils/websocket_manager.py`, `utils/database.py`, `utils/notifier.py`, `utils/audit_log.py`
- `utils/multi_timeframe.py` (used by momentum_strategy)
- `config.py` (with deleted-feature param sections removed)

**Simplify `AdaptiveStrategy`:** remove imports of `enable_ensemble`, `enable_ml_signals`, `enable_signal_aggregator`, `enable_cross_asset`, `enable_portfolio_optimizer`. Strategy should only switch between `MomentumStrategy` and `MeanReversionStrategy` based on regime, as the file's docstring originally describes.

### Docs

**Rewrite:**

- `CLAUDE.md` — remove "Institutional-Grade Features (Rating: 10/10)" section, "9 phases" enumeration, "+42.68% return / Sharpe 2.0" flagship claim. Replace with:
  - A short, honest **Status: experimental** header.
  - Inventory of the core path (the "Keep" list above).
  - Pointer to `research/` for quarantined work.
  - Pointer to `PROFITABILITY_RESEARCH.md` for realistic performance expectations.
- `README.md` — same treatment, plus a "Do not deploy real capital" disclaimer.
- `backtest_report_2024.md` — prepend a caveat header stating: 9 trades is below the 50-trade significance bar (per this repo's own `PROFITABILITY_RESEARCH.md`); 8 of 12 months show 0% return; results should not be cited as evidence of strategy edge.

**Delete from `docs/`:**

- `PRODUCTION_READINESS.md`, `OPERATIONS_RUNBOOK.md`, `RUNBOOK.md`, `RUNTIME_ARCHITECTURE.md`, `COMPLIANCE_GOVERNANCE.md`
- `INCIDENT_ESCALATION_ROSTER.md`, `INCIDENT_RESPONSE_OWNERSHIP.md`
- `MULTI_AGENT_EXECUTION_PLAN.md`, `AGENT_REPORT.md`
- `PHASE_3_ENHANCEMENTS.md`, `IMPLEMENTATION_SUMMARY.md`, `ADVANCED_FEATURES.md`, `CLAUDE_ADVANCED.md`
- `LOW_HARDWARE_PROFILE.md`, `STATUS.md`, `RELEASE_NOTES_2026-02-22.md`
- `QUANT_TRADING_RESEARCH_2025-11-08.md`, `SECRETS_ROTATION_INVENTORY.json`
- `FILE_STRUCTURE.md` (will be stale after deletions; regenerate later if needed)
- `KELLY_CRITERION_INTEGRATION.md` — delete (Kelly Criterion is already gated off by default in `momentum_strategy.py` config; the integration doc adds no current value)
- Root: `DOCKER_CICD_SUMMARY.md`, `CICD.md` reduced to essentials; `AGENTS.md`, `PAPER_TRADING_GUIDE.md` reviewed for relevance

**Keep:**

- `docs/SETUP.md`, `docs/installation.md`, `docs/strategy_guide.md`, `docs/TESTING.md`, `docs/advanced_orders_guide.md`, `docs/DASHBOARD_GUIDE.md` (dashboard script is kept)
- `PROFITABILITY_RESEARCH.md` (the most honest file in the repo)
- `README.md` (rewritten), `CLAUDE.md` (rewritten)
- `QUICKSTART.md`, `SECURITY.md`, `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- `Dockerfile`, `docker-compose.yml`, `DOCKER.md`
- `TODO.md` — repopulated with real items from this spec

### Scripts

**Keep (~6):**

- `scripts/kill_switch.py`
- `scripts/dashboard.py` (delete `enhanced_dashboard.py`)
- `scripts/simple_backtest.py`
- `scripts/quickstart.py`
- `scripts/run.py`
- `scripts/simple_trader.py`

**Delete:**

`chaos_drill.py`, `deploy_canary.py`, `rollback_drill.py`, `runtime_industrial_gate.py`, `runtime_watchdog.py`, `staging_incident_ticket_drill.py`, `validate_incident_contacts.py`, `validate_incident_ticket_drill_evidence.py`, `validate_incident_ticketing.py`, `incident_response_automation.py`, `incident_ack.py`, `replay_notification_dead_letters.py`, `push_ops_metrics.py`, `export_ops_metrics.py`, `governance_gate.py`, `secrets_audit.py`, `shadow_drift_dashboard.py`, `strategy_promotion_gate.py`, `paper_burn_in_scorecard.py`, `fault_injection_matrix.py`, `feature_comparison_backtest.py`, `generate_validation_artifacts.py`, `deployment_preflight.py`, `validated_backtest_report.py`, `validate_alternative_data.py`, `ops_status_report.py`, `mock_strategies.py`, `run_low_resource_profile.py`, `run_now.py`, `mcp_server.py`, `mcp.json`, `enhanced_dashboard.py`.

**Also delete at repo root:**

- `incident_contacts.py`
- `infra/` if its contents only support deleted scripts (verify)

### Install / dependencies

- Add `matplotlib` to `requirements.txt` (transitive requirement at import time).
- Make `utils/__init__.py` lazy on `visualization` — convert the eager import to a function-level import inside whatever uses it — so a broker import doesn't force matplotlib.
- After module deletions, remove from `requirements.txt`: `torch`, `torchvision`, `torchaudio`, `transformers`.
- Acceptance: in a fresh container, `pip install -r requirements.txt && pytest tests/` collects and runs without import errors.

### Tests

- Delete tests for every deleted module.
- For each quarantined module, move its test file to `research/tests/`.
- Configure `pytest.ini` to default-exclude `research/`.
- Acceptance: `pytest tests/` runs in under 30 seconds (or under 2 minutes; just bounded), no skips for missing optional dependencies that are part of the core.

### CI

- Strip the GitHub Actions workflows of the deleted-feature gates: incident-drill nightly, chaos-drill workflow, secrets-audit workflow, scorecard burn-rate, etc. Keep:
  - Tests
  - Lint (black, ruff)
  - Type-check (mypy lenient)
- Verify dependabot config is sane (5 PRs in recent history were dependabot; that's fine, just review the configured ecosystems).

### Config

In `config.py`, remove the parameter blocks for deleted features: `CRYPTO_PARAMS` (or move to `research/`), `OVERNIGHT_PARAMS`, `ML_PARAMS`, `RL_PARAMS`, `OPTIONS_PARAMS`, `SENTIMENT_PARAMS`, `LLM_PARAMS`. Result: `config.py` shrinks from 813 lines toward a few hundred.

### Code-organization debt

**Deferred.** After deletions, re-measure file sizes. If `main.py` and `adaptive_strategy.py` are still >800 LOC, open a follow-up issue. Do not refactor as part of this cleanup; deletions may make the right split obvious.

## Validation: the honest baseline

After all cleanup commits land, run **one** backtest and publish it as the new reference:

- Strategy: `MomentumStrategy` with default parameters.
- Symbols: SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM.
- Period: 2020-01-01 to 2024-12-31 (includes COVID crash, 2022 bear market, 2023 recovery, 2024 rally — multi-regime).
- Slippage: 0.4% per trade (matches `PROFITABILITY_RESEARCH.md`).
- Spread cost: 0.1%.
- Initial capital: $100,000.
- Required minimum trade count: 50. If the strategy produces fewer than 50 trades, the result is reported as "inconclusive" and that is the headline finding.

Output: `results/honest_backtest_2020-2024.md` with full trade log, monthly return table, Sharpe / Sortino / Calmar / max-drawdown / win-rate / profit-factor, and a one-paragraph honest interpretation that does not extrapolate beyond what the trade count supports.

This number — whatever it is — becomes the only performance claim in `README.md` and `CLAUDE.md`. The 2024 9-trade report is retained only as a deprecated artifact with its caveat header.

## Order of operations (commit-by-commit)

1. **Doc rewrite, no code touched.** `CLAUDE.md`, `README.md`, `backtest_report_2024.md` caveat. Delete the unwanted `docs/*.md` files.
2. **Install fix.** Add matplotlib; convert `utils/__init__.py` visualization import to lazy. Verify clean-env install + test collection.
3. **Delete-list, batch 1: ML/LLM/alt-data.** Delete `ml/lstm_predictor.py`, `ml/rl_agent.py`, `ml/ensemble_predictor.py`, `ml/hyperparameter_optimizer.py`, `ml/feature_importance.py`, `llm/`, `data/data_fetchers/`, `data/llm_providers/`, `data/social_sentiment_advanced.py`, `data/order_flow_analyzer.py`, `data/web_scraper.py`, `data/alternative_data_provider.py`, `utils/news_sentiment.py`, `strategies/lstm_enhanced_strategy.py`, and their tests. Strip imports from `adaptive_strategy.py` and `live_trader.py` and `main.py`. Remove `torch`/`transformers` from requirements. Tests pass.
4. **Delete-list, batch 2: strategy variants + options.** Delete `bracket_momentum_strategy.py`, `extended_hours_strategy.py`, `gap_trading_strategy.py`, `enhanced_momentum_strategy.py`, `ensemble_strategy.py`, `ensemble_voting_strategy.py`, `brokers/options_broker.py`, and their tests. Tests pass.
5. **Quarantine.** Create `research/` tree, move quarantined modules + their tests, add `research/README.md` describing the contents as unvalidated. Update `pytest.ini` to exclude `research/`. Tests pass.
6. **Script + infra purge.** Delete the long list of operational scripts. Trim the GitHub Actions workflows. Delete `incident_contacts.py`. Tests pass.
7. **Config trim.** Remove parameter blocks for deleted features from `config.py`. Tests pass.
8. **Honest backtest.** Run the validation procedure above; commit `results/honest_backtest_2020-2024.md`.
9. **TODO.md repopulate.** Replace contents with the real follow-ups identified during cleanup (e.g. "decide direction: paper-only / live / learning / showcase").

Each step a separate commit so any single step can be reverted independently. Each step's commit message ends with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

## Risks

- **Deleting modules that turn out to be used by `live_trader.py` / `main.py` / `adaptive_strategy.py` via dynamic import or string-based strategy discovery.** Mitigation: grep for each module name (and its class names) before deletion; run the full test suite after each commit.
- **`config.py` parameter blocks referenced by code we keep.** Mitigation: same — grep before removing each block.
- **Test collection masks runtime errors.** Mitigation: after cleanup, run `python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-03-31` end-to-end as a smoke test before declaring cleanup done.
- **CI workflows referencing deleted scripts will fail.** Mitigation: prune workflows in the same commit as the scripts they call.
- **Honest baseline backtest produces an embarrassing result (Sharpe < 0.5 or negative).** That is acceptable — the *point* of this cleanup is to be willing to publish honest numbers. If the result is bad, that informs the next direction decision.

## Acceptance criteria

- `CLAUDE.md` no longer contains "10/10", "+42.68%", "institutional-grade", or "Phase 1–9" language.
- `pip install -r requirements.txt && pytest tests/` succeeds in a fresh container.
- `python -c "import strategies.adaptive_strategy"` succeeds with no torch/transformers/openai/anthropic imports.
- `find . -name "*.py" -not -path "./research/*" -not -path "./.git/*" -not -path "./tests/*" | xargs wc -l | tail -1` shows the core code is materially smaller than 193K lines (target: <80K).
- `ls docs/` shows ≤10 files.
- `ls scripts/` shows ≤8 files.
- `results/honest_backtest_2020-2024.md` exists, with ≥50 trades or an explicit "inconclusive" finding, and is the only performance number cited from `README.md` and `CLAUDE.md`.

## Out of scope (explicitly deferred)

- Splitting `main.py`, `live_trader.py`, `adaptive_strategy.py` into smaller modules.
- Deciding the long-term direction (paper / live / learning / showcase).
- Improving strategy performance.
- Adding new validation rigor beyond running the one honest backtest.
