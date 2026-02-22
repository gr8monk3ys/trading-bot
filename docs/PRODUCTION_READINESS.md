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
- Optional incident ticket auto-creation for ack-SLA breaches via `INCIDENT_TICKETING_*` risk/env settings.
- Incident ticket payloads can include direct response links:
  - `INCIDENT_RESPONSE_RUNBOOK_URL`
  - `INCIDENT_ESCALATION_ROSTER_URL`
- Optional multi-broker failover manager supports Alpaca primary with IB backup:
  - `MULTI_BROKER_ENABLED=true`
  - `MULTI_BROKER_BACKUP_BROKER=ib`
  - `MULTI_BROKER_HEALTH_CHECK_INTERVAL`, `MULTI_BROKER_FAILURE_THRESHOLD`, `MULTI_BROKER_RECOVERY_THRESHOLD`, `MULTI_BROKER_OPERATION_TIMEOUT_SECONDS`
  - `IB_HOST`, `IB_PAPER_PORT` / `IB_LIVE_PORT`, `IB_CLIENT_ID`
- Webhook integrations support retry/backoff and auth headers:
  - `SLO_PAGING_MAX_RETRIES`, `SLO_PAGING_RETRY_BACKOFF_SECONDS`, `SLO_PAGING_AUTH_*`
  - `INCIDENT_TICKETING_MAX_RETRIES`, `INCIDENT_TICKETING_RETRY_BACKOFF_SECONDS`, `INCIDENT_TICKETING_AUTH_*`
- Failed webhook deliveries are persisted to `results/runs/<run_id>/notification_dead_letters.jsonl`.
- Replay dead-letter notifications with `python scripts/replay_notification_dead_letters.py --dead-letter-path results/runs/<run_id>/notification_dead_letters.jsonl --in-place --output results/notification_replay_report.json`.
- Dead-letter backlog SLO monitoring supports sustained backlog alerting via:
  - `NOTIFICATION_DEAD_LETTER_WARNING_THRESHOLD`
  - `NOTIFICATION_DEAD_LETTER_CRITICAL_THRESHOLD`
  - `NOTIFICATION_DEAD_LETTER_PERSIST_MINUTES`
- Automated chaos drill runner available via `python scripts/chaos_drill.py` (includes auth-token expiry, quote-staleness recovery, and multi-broker failover/failback drills).
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
- Strict promotion gate validates incident ownership/escalation docs have no unresolved placeholders.
- Validate incident-contact readiness explicitly with:
  - `python scripts/validate_incident_contacts.py --ownership-doc docs/INCIDENT_RESPONSE_OWNERSHIP.md --escalation-doc docs/INCIDENT_ESCALATION_ROSTER.md`
- Validate deployment hardening with `python scripts/deployment_preflight.py --required-env ALPACA_API_KEY,ALPACA_SECRET_KEY`.
- Validate runtime connectivity watchdogs with:
  - `python scripts/runtime_watchdog.py --output results/validation/runtime_watchdog.json`
  - Includes hard IB API session readiness handshake (`ib_insync` + managed account check).
  - For local-only dry run: `python scripts/runtime_watchdog.py --no-check-alpaca --no-check-ticket-webhook --no-check-ib-port --no-check-ib-api`
- For strategy-impacting PRs, CI requires `PROMOTION_EXPERIMENT_ID` and enforces strict promotion-gate checks.
- CI warning policy is enforced on the entire `tests/unit` suite via a dedicated strict gate (`--no-cov`):
  - Global warnings-as-errors (`-W error`) in CI.
  - Minimal compatibility allowlist:
    - `DeprecationWarning`
    - `PendingDeprecationWarning`
- CI coverage reporting runs as a separate `tests/unit` pass (to produce `coverage.xml`) after strict warning checks.
- Validate ack-SLA-to-ticket workflow with `python scripts/validate_incident_ticketing.py --tmp-dir results/validation`.
- Validate a real staging incident-ticket webhook and persist evidence with:
  - `python scripts/staging_incident_ticket_drill.py --webhook-url <staging_webhook> --artifact-dir results/validation --require-delivery --require-non-test-target --runbook-url <runbook_url> --escalation-roster-url <roster_url>`
  - `python scripts/validate_incident_ticket_drill_evidence.py --report-path results/validation/incident_ticket_drill_report.json --require-delivery --require-non-test-target --require-response-links`
- Run unified runtime readiness checks with:
  - `python scripts/runtime_industrial_gate.py --run-ticket-drill --ticket-webhook-url <staging_webhook> --run-failover-probe --output results/validation/runtime_industrial_gate.json`
  - Failover probe prerequisites: `MULTI_BROKER_ENABLED=true`, valid `ALPACA_*` credentials, configured `IB_*` settings, and `ib_insync` installed.
- Run one-command go-live precheck orchestrator:
  - `bash scripts/go_live_precheck.sh --output-dir results/validation/precheck --run-ticket-drill --run-failover-probe`
  - Local dry-run mode: `bash scripts/go_live_precheck.sh --local-only --skip-deployment-preflight`
  - IB API readiness gate is enforced by default and blocks `--skip-runtime-watchdog` unless explicitly disabled with `--no-enforce-ib-api-gate` (dry-run only).
- Nightly CI drill workflow: `.github/workflows/nightly-incident-ticket-drill.yml`
  - Requires repository secret: `INCIDENT_TICKETING_WEBHOOK_URL_STAGING`
  - Optional secrets: `INCIDENT_TICKETING_AUTH_TOKEN_STAGING`, `INCIDENT_TICKETING_AUTH_SCHEME_STAGING`
  - Workflow enforces non-test drill evidence gate and automatically replays drill dead letters, failing if unresolved backlog remains.
  - Optional failure email webhook secrets:
    - `INCIDENT_DRILL_FAILURE_EMAIL_WEBHOOK_URL`
    - `INCIDENT_DRILL_FAILURE_EMAIL_WEBHOOK_AUTH_TOKEN`
    - `INCIDENT_DRILL_FAILURE_EMAIL_WEBHOOK_AUTH_SCHEME`
  - Manual `workflow_dispatch` supports `send_test_email_only=true` to send a test email payload and skip drill/replay.
- Generate daily operations summary from run artifacts with `python scripts/ops_status_report.py --run-dir results/runs/<run_id> --json-output results/ops_status.json --md-output results/ops_status.md`.
- Export machine-readable ops metrics for Prometheus/Grafana ingestion:
  - `python scripts/export_ops_metrics.py --run-dir results/runs/<run_id> --runtime-watchdog-json results/validation/runtime_watchdog.json --runtime-gate-json results/validation/runtime_industrial_gate.json --go-live-summary-json results/validation/precheck/go_live_precheck_summary.json --json-output results/metrics/ops_metrics.json --prom-output results/metrics/ops_metrics.prom`
  - Push externally (Pushgateway-compatible):
    - `python scripts/push_ops_metrics.py --run-dir results/runs/<run_id> --runtime-watchdog-json results/validation/runtime_watchdog.json --runtime-gate-json results/validation/runtime_industrial_gate.json --go-live-summary-json results/validation/precheck/go_live_precheck_summary.json --pushgateway-url <https://pushgateway.example.com> --push-job trading_bot --push-instance <host_or_pi>`
- Incident response automation (escalation + optional rollback on critical readiness/SLO failures):
  - `python scripts/incident_response_automation.py --run-dir results/runs/<run_id> --runtime-watchdog-json results/validation/runtime_watchdog.json --runtime-gate-json results/validation/runtime_industrial_gate.json --go-live-summary-json results/validation/precheck/go_live_precheck_summary.json --webhook-url <incident_webhook> --rollback-cmd "<rollback cmd>" --output results/validation/incident_response_automation.json`
- Canary deployment/rollback automation available:
  - `python scripts/deploy_canary.py --candidate-cmd "<candidate cmd>" --health-check-cmd "<health cmd>" --rollback-cmd "<rollback cmd>" --output results/validation/canary_report.json`
- Secrets rotation/leak audit:
  - `python scripts/secrets_audit.py --inventory-path docs/SECRETS_ROTATION_INVENTORY.json --output results/validation/secrets_audit.json`
- Compliance/governance gate for beyond-paper capital:
  - `python scripts/governance_gate.py --mode live --output results/validation/governance_gate.json`
  - `main.py live --real` now enforces governance gate by default; override only for controlled dry runs with:
    - `--no-enforce-governance-gate`
    - Optional paths: `--governance-approval-path`, `--governance-policy-doc-path`
- Existing module-specific warning guards remain in pytest config for local parity.
- CI warning guard blocks new warning regressions in critical modules:
  - `FutureWarning`: `engine.backtest_engine`, `utils.factor_data`
  - `RuntimeWarning`: `strategies.risk_manager`, `strategies.bracket_momentum_strategy`, `strategies.ensemble_strategy`, `strategies.factor_models`, `brokers.backtest_broker`, `engine.factor_attribution`, `utils.market_impact`
  - `PytestReturnNotNoneWarning` (test hygiene)
- Runtime credential checks are deferred to execution paths; non-trading CLI paths (for example `main.py --help`) no longer emit Alpaca credential warnings at import/startup.

## Gaps To Resolve Before Live Capital
- Keep incident ownership contacts and escalation links reviewed quarterly.
- Keep staging/prod webhook endpoints mapped to your real on-call system.

## Go‑Live Gate
- 30+ paper trading days and 30+ paper trades.
- Profitability gates pass in validated backtests.
- No reconciliation mismatches over at least 5 trading days.
- Strict promotion checklist passes (parameter snapshots + walk-forward artifacts + required validation gates).
