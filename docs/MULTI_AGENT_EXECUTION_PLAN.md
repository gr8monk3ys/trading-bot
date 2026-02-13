# Multi-Agent Execution Plan (Industrial Quant Bot)

## Objective
Build a production-grade quant trading system that is robust in paper trading and ready for controlled live rollout.

## Phase 1 (Completed)
- Agent A (Core Stability): fixed strategy manager initialization/order-of-operations bugs.
- Agent B (Backtest Integrity): fixed synthetic data generation and hardened backtest gap-stat handling.
- Agent C (Execution Path): hardened base strategy order submission fallbacks and simple MA sell path.
- Agent D (Validation Logic): fixed walk-forward pass/fail logic for high overfit ratios.
- Agent E (Model Ensemble): restored expected default weights and added a no-sklearn meta-learner fallback.
- Agent F (Runtime Ops): fixed live trader log directory bootstrapping.
- Agent G (Test Platform): removed coverage/import side effects with lazy engine exports and safer conftest imports.

Validation snapshot:
- `uv run pytest`: 4003 passed, 40 skipped.
- Coverage-gated modules: 81.94% total.

## Phase 2 (Iteration 1 Completed)
- Agent 1 (Data Quality + PIT):
  - Added `/Users/natalyscaturchio/code/trading-bot/utils/data_quality.py`.
  - Added OHLCV validation + PIT announcement-date lookahead checks.
  - Integrated data quality reporting into `BacktestEngine.run_backtest()` output under `data_quality`.
- Agent 2 (Paper Trading Reality Layer):
  - Added execution realism profiles to `BacktestBroker`:
    - `idealistic`, `realistic`, `stressed`
  - Added profile-driven slippage multiplier, partial-fill multiplier, simulated latency, optional liquidity rejects.
  - Added CLI flag `--execution-profile` in `/Users/natalyscaturchio/code/trading-bot/main.py`.
- Agent 3 (Risk Guardrails):
  - Added portfolio intraday drawdown kill-switch guardrail to `OrderGateway`.
  - Added cooldown-based trading halt with audit logging.
  - Added guardrail status metrics in `OrderGateway.get_statistics()`.

Validation snapshot:
- `uv run pytest`: 4014 passed, 40 skipped, coverage 81.79%.

## Phase 2 (Iteration 2 Completed)
- Agent 4 (Observability + Replay):
  - Added `/Users/natalyscaturchio/code/trading-bot/utils/run_artifacts.py` for run IDs and JSON/JSONL artifact IO.
  - Added `/Users/natalyscaturchio/code/trading-bot/utils/run_replay.py` for replay loading, filtering, and formatting.
  - Extended `/Users/natalyscaturchio/code/trading-bot/engine/backtest_engine.py`:
    - run-scoped metadata (`run_metadata`) on backtest results
    - optional artifact persistence under `results/runs/<run_id>/`
    - generated files: `summary.json`, `decision_events.jsonl`, `trades.jsonl`, `manifest.json`
  - Added replay mode to `/Users/natalyscaturchio/code/trading-bot/main.py`:
    - `python main.py replay --run-id <id> [--symbol ...] [--replay-date ...] [--errors-only] [--limit ...]`

Validation snapshot:
- Targeted observability/replay tests:
  - `uv run pytest tests/unit/test_run_artifacts.py tests/unit/test_run_replay.py tests/unit/test_backtest_engine_run_backtest.py -q`
  - 11 passed
- Broader impacted suite:
  - `uv run pytest tests/unit/test_backtest_engine.py tests/unit/test_backtest_execution_profile.py tests/unit/test_data_quality.py tests/unit/test_order_gateway_guardrails.py -q`
  - 42 passed

## Phase 2 (Iteration 3 Completed)
- Agent 5 (Research-to-Prod Pipeline):
  - Extended `/Users/natalyscaturchio/code/trading-bot/research/research_registry.py` with:
    - parameter snapshot registry (`record_parameter_snapshot`)
    - walk-forward artifact storage (`store_walk_forward_artifacts`)
    - strict promotion checklist generation (`generate_promotion_checklist`)
    - strict gating options on readiness/blockers (`is_promotion_ready(..., strict=True)`)
  - Added CI-facing gate script:
    - `/Users/natalyscaturchio/code/trading-bot/scripts/strategy_promotion_gate.py`
  - Added optional strict promotion gate in CI:
    - `/Users/natalyscaturchio/code/trading-bot/.github/workflows/ci.yml`
    - Runs only when `PROMOTION_EXPERIMENT_ID` repo variable is set.

Validation snapshot:
- New pipeline tests:
  - `uv run pytest tests/unit/test_research_registry.py tests/unit/test_strategy_promotion_gate.py --no-cov -q`
  - 31 passed
- Full regression suite:
  - `uv run pytest -q`
  - 4029 passed, 40 skipped, coverage 81.48%

## Phase 2 (Iteration 4 Completed)
- Agent 5 (Pipeline UX + Automation):
  - Added unified research operations to `/Users/natalyscaturchio/code/trading-bot/main.py`:
    - `mode=research` with actions:
      - `create`, `snapshot`, `record-backtest`, `record-validation`, `record-paper`
      - `store-walk-forward`, `approve-review`, `summary`, `blockers`, `check`, `promote`
    - Supports strict promotion checks and JSON/file payload ingestion.
  - Added tests for one-entrypoint research workflow:
    - `/Users/natalyscaturchio/code/trading-bot/tests/unit/test_main_research_mode.py`

Validation snapshot:
- Focused:
  - `uv run pytest tests/unit/test_main_research_mode.py tests/unit/test_research_registry.py tests/unit/test_strategy_promotion_gate.py --no-cov -q`
  - 33 passed
- Full regression suite:
  - `uv run pytest -q`
  - 4031 passed, 40 skipped, coverage 81.48%

## Phase 2 (Iteration 5 Completed)
- Agent 6 (Quality Gates Hardening):
  - Warning cleanup in core modules:
    - `/Users/natalyscaturchio/code/trading-bot/engine/backtest_engine.py` (typed result frame to avoid pandas `FutureWarning`)
    - `/Users/natalyscaturchio/code/trading-bot/strategies/risk_manager.py` (safe return/division calculations and finite-value guards)
    - `/Users/natalyscaturchio/code/trading-bot/strategies/bracket_momentum_strategy.py` and
      `/Users/natalyscaturchio/code/trading-bot/strategies/ensemble_strategy.py` (await async subscriber hooks when needed)
  - CI enforcement upgrades in `/Users/natalyscaturchio/code/trading-bot/.github/workflows/ci.yml`:
    - Added warning-guard test step with warnings promoted to errors for critical modules.
    - Promotion gate is now mandatory for strategy-impacting pull requests via path filtering + required `PROMOTION_EXPERIMENT_ID`.

Validation snapshot:
- Warning-guard targeted run:
  - `uv run pytest tests/unit/test_backtest_engine.py tests/unit/test_risk_manager.py tests/unit/test_bracket_momentum_strategy.py tests/unit/test_ensemble_strategy.py --no-cov -W error::FutureWarning:engine.backtest_engine -W error::RuntimeWarning:strategies.risk_manager -W error::RuntimeWarning:strategies.bracket_momentum_strategy -W error::RuntimeWarning:strategies.ensemble_strategy -q`
  - 154 passed

## Phase 2 (Iteration 6 Completed)
- Agent 7 (Startup Side-Effect Hardening):
  - Refactored `/Users/natalyscaturchio/code/trading-bot/config.py` credential handling to be non-strict at import time.
  - Added runtime credential helpers:
    - `get_alpaca_creds(refresh=False)`
    - `require_alpaca_credentials(context=...)`
  - Added env alias support for credentials (`API_KEY` / `API_SECRET`) in addition to `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`.
  - Updated `/Users/natalyscaturchio/code/trading-bot/main.py` dynamic symbol selection to validate credentials at runtime via `require_alpaca_credentials(...)`.
  - Added regression coverage in `/Users/natalyscaturchio/code/trading-bot/tests/unit/test_config_credentials_runtime.py`:
    - alias env support
    - runtime credential validation failure behavior
    - `main.py --help` has no startup credential warning noise

Validation snapshot:
- Focused tests:
  - `uv run pytest --no-cov tests/unit/test_config_credentials_runtime.py tests/unit/test_main_research_mode.py -q`
  - 5 passed
- Full regression suite:
  - `uv run pytest -q`
  - 4034 passed, 40 skipped, coverage 81.53%

## Phase 2 (Iteration 7 Completed)
- Agent 8 (Warning Hygiene + CI Guard Expansion):
  - Eliminated runtime warning hotspots in:
    - `/Users/natalyscaturchio/code/trading-bot/strategies/factor_models.py`
      - finite-safe winsorization/z-score paths
      - safe log transform for size factor using valid-mask writes
    - `/Users/natalyscaturchio/code/trading-bot/brokers/backtest_broker.py`
      - zero-crossing-safe position math to prevent divide-by-zero when covering shorts
      - safe slippage-bps calculation for zero-price edge cases
    - `/Users/natalyscaturchio/code/trading-bot/utils/factor_data.py`
      - replaced warning-prone DataFrame concat path with key-based record merge (`symbol`,`date`)
  - Improved test hygiene in `/Users/natalyscaturchio/code/trading-bot/tests/test_connection.py`:
    - removed non-None test returns
    - now uses integration-style skip behavior when credentials are missing/invalid or API is unavailable
  - Expanded warning gates:
    - `/Users/natalyscaturchio/code/trading-bot/pytest.ini`
      - `RuntimeWarning` as errors for `strategies.factor_models` and `brokers.backtest_broker`
      - `FutureWarning` as errors for `utils.factor_data`
      - `PytestReturnNotNoneWarning` as error
    - `/Users/natalyscaturchio/code/trading-bot/.github/workflows/ci.yml`
      - warning-guard step now includes `test_factor_models.py`, `test_gap_risk_modeling.py`, `test_factor_data.py`
      - warning-as-error flags expanded to match new hardened modules

Validation snapshot:
- Warning-focused guard run:
  - `uv run pytest tests/unit/test_factor_data.py tests/unit/test_factor_models.py tests/unit/test_gap_risk_modeling.py tests/unit/test_backtest_broker.py tests/test_connection.py --no-cov -W error::RuntimeWarning:strategies.factor_models -W error::RuntimeWarning:brokers.backtest_broker -W error::FutureWarning:utils.factor_data -W error::pytest.PytestReturnNotNoneWarning -q`
  - 241 passed, 1 skipped
- Full regression suite:
  - `uv run pytest -q`
  - 4033 passed, 41 skipped, coverage 81.53%

## Phase 2 (Iteration 8 Completed)
- Agent 9 (Statistical Warning Elimination):
  - Removed residual runtime-warning hotspots in:
    - `/Users/natalyscaturchio/code/trading-bot/engine/factor_attribution.py`
      - added safe t-test helpers for one-sample and two-sample tests
      - deterministic handling for low-variance/degenerate samples
      - local warning-suppressed scipy calls with finite fallbacks
    - `/Users/natalyscaturchio/code/trading-bot/utils/market_impact.py`
      - added `_safe_correlation(...)` to avoid `np.corrcoef` warnings when variance is zero
  - Expanded warning guards:
    - `/Users/natalyscaturchio/code/trading-bot/pytest.ini`
      - `RuntimeWarning` as errors for `engine.factor_attribution` and `utils.market_impact`
    - `/Users/natalyscaturchio/code/trading-bot/.github/workflows/ci.yml`
      - warning-guard step now includes `test_factor_attribution.py` and `test_market_impact.py`
      - warning-as-error flags expanded for new modules

Validation snapshot:
- New warning-focused run:
  - `uv run pytest tests/unit/test_factor_attribution.py tests/unit/test_market_impact.py --no-cov -W error::RuntimeWarning:engine.factor_attribution -W error::RuntimeWarning:utils.market_impact -q`
  - 92 passed
- Expanded guard parity run:
  - `uv run pytest tests/unit/test_backtest_engine.py tests/unit/test_risk_manager.py tests/unit/test_bracket_momentum_strategy.py tests/unit/test_ensemble_strategy.py tests/unit/test_factor_models.py tests/unit/test_factor_attribution.py tests/unit/test_market_impact.py tests/unit/test_gap_risk_modeling.py tests/unit/test_factor_data.py tests/test_connection.py --no-cov -W error::FutureWarning:engine.backtest_engine -W error::RuntimeWarning:strategies.risk_manager -W error::RuntimeWarning:strategies.bracket_momentum_strategy -W error::RuntimeWarning:strategies.ensemble_strategy -W error::RuntimeWarning:strategies.factor_models -W error::RuntimeWarning:brokers.backtest_broker -W error::RuntimeWarning:engine.factor_attribution -W error::RuntimeWarning:utils.market_impact -W error::FutureWarning:utils.factor_data -W error::pytest.PytestReturnNotNoneWarning -q`
  - 429 passed, 1 skipped
- Full regression suite:
  - `uv run pytest -q`
  - 4033 passed, 41 skipped, coverage 81.53% (no warnings summary)

## Phase 2 (Iteration 9 Completed)
- Agent 10 (Global Warning Gate Rollout):
  - Upgraded CI warning enforcement from subset guard tests to full unit-suite strict mode in:
    - `/Users/natalyscaturchio/code/trading-bot/.github/workflows/ci.yml`
  - `Run tests` step now executes `tests/unit` with global warnings-as-errors (`-W error`).
  - Added minimal allowlist for currently unavoidable warning categories:
    - `DeprecationWarning`
    - `PendingDeprecationWarning`
    - `ResourceWarning`
    - `pytest.PytestUnraisableExceptionWarning`
  - Removed redundant subset-only warning guard step to avoid duplicate CI runtime.

Validation snapshot:
- Strict full-unit warning run (same semantics as CI):
  - `uv run pytest tests/unit/ -W error -W ignore::DeprecationWarning -W ignore::PendingDeprecationWarning -W ignore::ResourceWarning -W ignore::pytest.PytestUnraisableExceptionWarning -q`
  - 3796 passed, 29 skipped, coverage 81.53%

## Phase 2 (Iteration 10 Completed)
- Agent 11 (Warning Allowlist Burn-Down):
  - Eliminated remaining warning allowlist categories by fixing source leaks:
    - `/Users/natalyscaturchio/code/trading-bot/main.py`
      - moved file-handler logging setup out of import path into explicit `configure_logging()`
    - `/Users/natalyscaturchio/code/trading-bot/utils/audit_log.py`
      - added deterministic close lifecycle and best-effort destructor cleanup
    - `/Users/natalyscaturchio/code/trading-bot/utils/database.py`
      - made `initialize()` idempotent without leaking prior `aiosqlite` connections
    - `/Users/natalyscaturchio/code/trading-bot/engine/strategy_manager.py`
      - removed implicit event-loop creation during runtime state load (`get_running_loop` path)
      - added owned-resource cleanup (`close()` + destructor fallback)
    - `/Users/natalyscaturchio/code/trading-bot/tests/conftest.py`
      - added explicit teardown cleanup for lingering asyncio loops during pytest teardown
  - Tightened CI warning allowlist in:
    - `/Users/natalyscaturchio/code/trading-bot/.github/workflows/ci.yml`
    - removed `ResourceWarning` and `pytest.PytestUnraisableExceptionWarning` ignores.
  - Split CI execution into:
    - strict warning gate (`tests/unit --no-cov -W error ...`) for deterministic warning enforcement
    - separate coverage pass (`tests/unit`) to generate coverage artifacts.
  - Added explicit asyncio loop-scope defaults in:
    - `/Users/natalyscaturchio/code/trading-bot/pytest.ini`
    - `asyncio_default_fixture_loop_scope = function`
    - `asyncio_default_test_loop_scope = function`

Validation snapshot:
- Strict full-unit warning run (updated CI semantics):
  - `uv run pytest tests/unit/ --no-cov -W error -W ignore::DeprecationWarning -W ignore::PendingDeprecationWarning -q`
  - 3796 passed, 29 skipped
- Coverage parity run:
  - `uv run pytest tests/unit/`
  - 3796 passed, 29 skipped, coverage 81.53%

## Phase 2 (Iteration 11 Completed)
- Agent 12 (Operational Paging + SLO Alert Routing):
  - Added `/Users/natalyscaturchio/code/trading-bot/utils/slo_alerting.py`:
    - webhook notifier with severity filtering and timeout controls
    - config-driven notifier builder from `RISK_PARAMS`
  - Integrated paging hooks into:
    - `/Users/natalyscaturchio/code/trading-bot/utils/slo_monitor.py`
    - `/Users/natalyscaturchio/code/trading-bot/main.py`
    - `/Users/natalyscaturchio/code/trading-bot/live_trader.py`
  - Added risk/env controls in `/Users/natalyscaturchio/code/trading-bot/config.py`:
    - `SLO_PAGING_ENABLED`
    - `SLO_PAGING_WEBHOOK_URL`
    - `SLO_PAGING_MIN_SEVERITY`
    - `SLO_PAGING_TIMEOUT_SECONDS`
- Agent 13 (Strategy Checkpoint Depth):
  - Hardened `/Users/natalyscaturchio/code/trading-bot/engine/strategy_manager.py` runtime persistence:
    - versioned strategy checkpoints (`exported_state` + bounded `internal_state`)
    - restore path supports legacy and v2 snapshots
    - preserved bounded internals: `price_history`, `signals`, `indicators`, circuit-breaker state
  - Extended `/Users/natalyscaturchio/code/trading-bot/live_trader.py` to persist/restore v2 strategy checkpoints in both periodic and shutdown state saves.
- Agent 14 (Chaos Drill Automation):
  - Added `/Users/natalyscaturchio/code/trading-bot/utils/chaos_drills.py` deterministic drill suite:
    - reconciliation broker-fetch failure resilience
    - data-quality auto-halt trigger
    - SLO alert-path failure tolerance
  - Added CLI runner:
    - `/Users/natalyscaturchio/code/trading-bot/scripts/chaos_drill.py`
  - Added CI automation:
    - `/Users/natalyscaturchio/code/trading-bot/.github/workflows/ci.yml` (`Run chaos drills` step)
- Agent 15 (Execution Quality CI Gate):
  - Added `/Users/natalyscaturchio/code/trading-bot/utils/execution_quality_gate.py` for normalized execution-quality metric extraction.
  - Extended strict promotion checks in `/Users/natalyscaturchio/code/trading-bot/research/research_registry.py`:
    - `paper_execution_quality_score_ci_gate`
    - `paper_execution_avg_slippage_ci_gate`
    - `paper_execution_fill_rate_ci_gate`
  - Extended optional paper validation gates:
    - `paper_execution_quality_score`
    - `paper_avg_slippage_bps`
    - `paper_fill_rate`
  - Updated `/Users/natalyscaturchio/code/trading-bot/scripts/generate_validation_artifacts.py` to attach execution-quality summary fields into paper artifacts.

Validation snapshot:
- Focused regression:
  - `uv run pytest tests/unit/test_slo_alerting.py tests/unit/test_slo_monitor.py tests/unit/test_strategy_checkpointing.py tests/unit/test_chaos_drills.py tests/unit/test_execution_quality_gate.py tests/unit/test_research_registry.py tests/unit/test_strategy_promotion_gate.py tests/unit/test_main_research_mode.py tests/test_config_validation.py --no-cov -q`
  - 87 passed
- Strict warning gate parity:
  - `uv run pytest tests/unit/ --no-cov -W error -W ignore::DeprecationWarning -W ignore::PendingDeprecationWarning -q`
  - 3831 passed, 29 skipped
- Full unit suite:
  - `uv run pytest tests/unit/ -q`
  - 3831 passed, 29 skipped, coverage 81.56%

## Phase 2 (Recommended Next Parallel Tracks)
- Agent 1: Broker session chaos suite
  - Add websocket disconnect, reconnect storm, and partial-fill stall drills.
- Agent 2: Incident workflow automation
  - Add acknowledgment SLA tracking and post-incident ticket generation for critical SLO pages.
- Agent 3: Live shadow drift guard
  - Add CI-gated paper/live shadow drift thresholds for top strategies.

## Exit Criteria for Live Trial
- 30+ consecutive paper days with stable risk metrics.
- No unresolved P1/P2 execution or accounting defects.
- Walk-forward + paper/live shadow drift within agreed thresholds.
