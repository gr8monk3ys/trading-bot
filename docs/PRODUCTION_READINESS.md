# Production Readiness Checklist

## Scope
This checklist summarizes operational readiness for live trading in this repository. It is intended for paper trading and staged promotion to live trading after validation.

## Core Safety
- OrderGateway enabled with circuit breaker enforcement.
- Audit logging enabled and writing to `audit_logs/`.
- Websocket trade updates running for fill visibility.
- Kill switch available: `python scripts/kill_switch.py --confirm "HALT TRADING" --cancel-orders --liquidate`.

## Observability & Resilience
- Runtime state saved and restored from `data/runtime_state.json` or `data/live_trader_state.json`.
- Position reconciliation loop running every 5 minutes.
- Order reconciliation loop running every 2 minutes.
- Reconciliation snapshots persisted for replay in `results/runs/<run_id>/`.
- Data quality gate runs in housekeeping and can trigger gateway kill switch.
- SLO breaches persisted in `ops_slo_events.jsonl` and mirrored to audit log.
- Optional webhook paging for critical SLO breaches via `SLO_PAGING_*` risk/env settings.
- Incident acknowledgment SLA tracking enabled via `incident_events.jsonl` and `INCIDENT_ACK_SLA_MINUTES`.
- Automated chaos drill runner available via `python scripts/chaos_drill.py`.
- Restart recovery verified with open positions.

## Strategy Checkpoint Coverage
- MomentumStrategy: stop/target/entry/peak and last signal time.
- MeanReversionStrategy: position entries, peaks/troughs, last signal time.
- BracketMomentumStrategy: active bracket orders.
- EnsembleStrategy: position entries, peaks, ensemble signals.
- PairsTradingStrategy: open pair positions.
- Common internal caches: bounded `price_history`, `signals`, `indicators`, and circuit-breaker state.

## Validation Artifacts
- Run `python scripts/generate_validation_artifacts.py --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2014-01-01 --end-date 2024-12-31`.
- Ensure `results/validation/<timestamp>/manifest.json` matches current git SHA.
- Review `paper_trading_summary.json` and go‑live blockers before live trading.
- Gate promotion with `python scripts/strategy_promotion_gate.py --experiment-id <id> --strict`.
- Strict promotion gate includes execution-quality attribution checks (score/slippage/fill-rate).
- Strict promotion gate includes paper/live shadow-drift threshold checks.
- Validate deployment hardening with `python scripts/deployment_preflight.py --required-env ALPACA_API_KEY,ALPACA_SECRET_KEY`.
- For strategy-impacting PRs, CI requires `PROMOTION_EXPERIMENT_ID` and enforces strict promotion-gate checks.
- CI warning policy is enforced on the entire `tests/unit` suite via a dedicated strict gate (`--no-cov`):
  - Global warnings-as-errors (`-W error`) in CI.
  - Minimal compatibility allowlist:
    - `DeprecationWarning`
    - `PendingDeprecationWarning`
- CI coverage reporting runs as a separate `tests/unit` pass (to produce `coverage.xml`) after strict warning checks.
  - Existing module-specific warning guards remain in pytest config for local parity.
- CI warning guard blocks new warning regressions in critical modules:
  - `FutureWarning`: `engine.backtest_engine`, `utils.factor_data`
  - `RuntimeWarning`: `strategies.risk_manager`, `strategies.bracket_momentum_strategy`, `strategies.ensemble_strategy`, `strategies.factor_models`, `brokers.backtest_broker`, `engine.factor_attribution`, `utils.market_impact`
  - `PytestReturnNotNoneWarning` (test hygiene)
- Runtime credential checks are deferred to execution paths; non-trading CLI paths (for example `main.py --help`) no longer emit Alpaca credential warnings at import/startup.

## Gaps To Resolve Before Live Capital
- Expand chaos drills to include broker auth-token expiry and quote-staleness recovery.
- Add post-incident auto-ticket creation workflow around SLO paging acknowledgments.

## Go‑Live Gate
- 30+ paper trading days and 30+ paper trades.
- Profitability gates pass in validated backtests.
- No reconciliation mismatches over at least 5 trading days.
- Strict promotion checklist passes (parameter snapshots + walk-forward artifacts + required validation gates).
