# Operations Runbook

## Purpose
This runbook defines on-call actions, escalation, and disaster-recovery drills for paper/live trading operations.

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
- Optional external paging webhooks for critical SLO breaches:
  - `SLO_PAGING_ENABLED=true`
  - `SLO_PAGING_WEBHOOK_URL=<webhook>`
  - `SLO_PAGING_MIN_SEVERITY=critical|warning`

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
