# Release Notes - 2026-02-22

## Overview
This release was shipped directly to `main` in two feature commits:
- `d74d25b` - `feat(trading): harden gateway routing and broker failover behavior`
- `0091a2d` - `feat(ops): add governance gates and incident automation tooling`

## Trading Runtime Hardening
Key improvements:
- Enforced gateway-only strategy order routing in `strategies/base_strategy.py`.
- Strengthened multi-broker failover behavior and runtime compatibility in `brokers/multi_broker.py` and `brokers/alpaca_broker.py`.
- Improved reconciliation and alerting signal quality in:
  - `utils/order_reconciliation.py`
  - `utils/slo_monitor.py`
  - `utils/slo_alerting.py`
- Added resilience fixes in:
  - `utils/order_gateway.py` (safe float handling, stale restored kill-switch cleanup)
  - `utils/data_quality.py` (timezone-safe stale-data age calculation)
- Expanded trading safety regression coverage, including:
  - `tests/unit/test_strategy_gateway_compliance.py`
  - `tests/unit/test_strategy_gateway_regressions.py`
  - `tests/unit/test_multi_broker_runtime_compat.py`

## Ops and Governance Automation
Key improvements:
- Added live-governance enforcement and artifacts:
  - `utils/governance_gate.py`
  - `scripts/governance_gate.py`
  - `docs/COMPLIANCE_GOVERNANCE.md`
- Added runtime readiness/watchdog gates:
  - `utils/runtime_watchdog.py`
  - `utils/runtime_industrial_gate.py`
  - `scripts/runtime_watchdog.py`
  - `scripts/runtime_industrial_gate.py`
  - `scripts/go_live_precheck.sh`
- Added incident automation and evidence tooling:
  - `utils/incident_response_automation.py`
  - `scripts/incident_response_automation.py`
  - `scripts/staging_incident_ticket_drill.py`
  - `scripts/validate_incident_ticket_drill_evidence.py`
  - `scripts/replay_notification_dead_letters.py`
- Added metrics export/push path:
  - `utils/ops_metrics.py`
  - `utils/ops_metrics_push.py`
  - `scripts/export_ops_metrics.py`
  - `scripts/push_ops_metrics.py`
- Added secrets and incident-contact governance checks:
  - `utils/secrets_audit.py`
  - `scripts/secrets_audit.py`
  - `incident_contacts.py`
  - `scripts/validate_incident_contacts.py`
- Added ops documentation and infra baseline:
  - `docs/OPERATIONS_RUNBOOK.md`
  - `docs/PRODUCTION_READINESS.md`
  - `docs/INCIDENT_RESPONSE_OWNERSHIP.md`
  - `docs/INCIDENT_ESCALATION_ROSTER.md`
  - `infra/systemd/*`
  - `.github/workflows/nightly-incident-ticket-drill.yml`

## Validation Run During Ship
Focused test slices run successfully before push:
- `uv run pytest --no-cov tests/unit/test_strategy_gateway_regressions.py tests/unit/test_multi_broker_runtime_compat.py tests/unit/test_slo_monitor.py`
- `uv run pytest --no-cov tests/unit/test_governance_gate.py tests/unit/test_runtime_watchdog.py tests/unit/test_runtime_industrial_gate.py tests/unit/test_incident_response_automation.py tests/unit/test_ops_metrics.py tests/unit/test_symbol_scope.py tests/unit/test_live_trader_risk_profiles.py tests/unit/test_go_live_precheck_script.py`

All tests in both slices passed.
