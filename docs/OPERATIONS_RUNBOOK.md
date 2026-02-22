# Operations Runbook

## Purpose
This runbook defines on-call actions, escalation, and disaster-recovery drills for paper/live trading operations.

## Ownership & Escalation
- Incident ownership: [INCIDENT_RESPONSE_OWNERSHIP.md](./INCIDENT_RESPONSE_OWNERSHIP.md)
- Escalation roster: [INCIDENT_ESCALATION_ROSTER.md](./INCIDENT_ESCALATION_ROSTER.md)

## Operational SLOs
- Order reconciliation consecutive mismatch runs: `< 3`.
- Data quality errors per quality cycle: `0`.
- Data stale warnings per quality cycle: `0`.
- Critical incident acknowledgment SLA: `<= 15 minutes`.
- Paper trading reconciliation pass rate: `>= 99.5%`.
- Paper trading operational error rate: `<= 2.0%`.

## Alert Sources
- `audit_logs/` immutable audit events (`risk_warning`, `risk_limit_breach`).
- Session artifacts under `results/runs/<run_id>/`:
  - `order_reconciliation_events.jsonl`
  - `position_reconciliation_events.jsonl`
  - `ops_slo_events.jsonl`
  - `data_quality_events.jsonl`
  - `data_quality_latest.json`
  - `notification_dead_letters.jsonl`
- Optional external paging webhooks for critical SLO breaches:
  - `SLO_PAGING_ENABLED=true`
  - `SLO_PAGING_WEBHOOK_URL=<webhook>`
  - `SLO_PAGING_MIN_SEVERITY=critical|warning`
- Prometheus/Grafana export:
  - `python scripts/export_ops_metrics.py --run-dir results/runs/<run_id> --prom-output results/metrics/ops_metrics.prom`
  - External Pushgateway delivery:
    - `python scripts/push_ops_metrics.py --run-dir results/runs/<run_id> --pushgateway-url <https://pushgateway.example.com> --push-job trading_bot --push-instance <host_or_pi>`

## First 5 Minutes (Incident Triage)
1. Confirm kill switch state from gateway stats (`halt_reason`, `trading_halted_until`).
2. Open latest run artifact directory and inspect:
   - last SLO breach
   - incident acknowledgment status (`incident_events.jsonl`)
   - latest reconciliation snapshots
   - latest data quality snapshot
3. Classify incident:
   - `data_quality`: stale/missing/corrupt bars
   - `reconciliation`: order/position drift
   - `broker/connectivity`: API/websocket failures
4. Keep entries halted until root cause is identified.
5. Run automated escalation/rollback policy if critical triggers are present:
   - `python scripts/incident_response_automation.py --run-dir results/runs/<run_id> --runtime-watchdog-json results/validation/runtime_watchdog.json --runtime-gate-json results/validation/runtime_industrial_gate.json --go-live-summary-json results/validation/precheck/go_live_precheck_summary.json --webhook-url <incident_webhook> --rollback-cmd "<rollback cmd>"`

## Recovery Decision Tree
1. If data quality breach:
   - refresh data source connectivity
   - validate latest 30 daily bars for all active symbols
   - resume entries only after zero critical quality breaches for one full cycle
2. If reconciliation breach:
   - inspect mismatched order IDs and broker status
   - confirm lifecycle transitions recovered
   - require two consecutive clean reconciliation cycles before resuming
3. If broker failure:
   - verify API and websocket health
   - restart broker session and rerun reconciliation

## Incident Acknowledgment
1. Identify active incident ID from `incident_events.jsonl` (`event_type=incident_open`).
2. Acknowledge incident:
   - `python scripts/incident_ack.py --run-id <run_id> --incident-id <id> --ack-by <oncall>`
3. Keep notes concise and include mitigation status in `--notes`.

## Disaster Recovery Drill (Weekly)
0. Run deterministic operational chaos drills and attach output:
   - `python scripts/chaos_drill.py --output results/chaos_drill_latest.json`
1. Stop trading process gracefully.
2. Restart process and confirm:
   - runtime state restored
   - reconciliation loops restart
   - SLO monitor resumes event emission
3. Record recovery timing:
   - detection time
   - restart complete time
   - first clean reconciliation time
4. Target:
   - RTO: `<= 15 minutes`
   - data-loss tolerance (RPO): `<= 1 minute` of runtime state

## Promotion Readiness (Operational)
- Run strict promotion gate for experiment.
- Ensure paper KPI criteria pass.
- Attach latest run artifacts for reconciliation + SLO evidence.
- Run secrets and governance gates for live-capital readiness:
  - `python scripts/secrets_audit.py --output results/validation/secrets_audit.json`
  - `python scripts/governance_gate.py --mode live --output results/validation/governance_gate.json`
- Real-capital startup enforces governance gate by default in `main.py`:
  - `python main.py live --real --strategy <strategy>`
  - Use `--no-enforce-governance-gate` only for controlled dry-runs.
- Run go-live precheck before enabling live capital:
  - `bash scripts/go_live_precheck.sh --output-dir results/validation/precheck --run-ticket-drill --run-failover-probe`
  - Add governance/secrets stages when promoting beyond paper:
    - `bash scripts/go_live_precheck.sh --output-dir results/validation/precheck --run-secrets-audit --run-governance-gate --governance-mode live --run-ticket-drill --run-failover-probe`
