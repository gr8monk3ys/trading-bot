# Scripts Directory

This directory contains various runner scripts and utilities for the trading bot.

## Main Entry Points

- **`../main.py`** - Primary entry point for production trading (supports live, backtest, optimize, replay, and research registry modes)
- **`../live_trader.py`** - Alternate single-strategy live launcher with extra runtime tuning flags

## Utilities

- **`dashboard.py`** - Real-time monitoring dashboard
- **`quickstart.py`** - Interactive setup wizard
- **`simple_trader.py`** - Simple trading script
- **`run.py`** - Legacy runner script; do not use as the primary entrypoint
- **`run_now.py`** - Quick start script
- **`kill_switch.py`** - Emergency kill switch (cancel orders / liquidate)
- **`deployment_preflight.py`** - Deployment hardening preflight checks (git/env readiness)
- **`chaos_drill.py`** - Deterministic operational chaos drills (reconciliation/data-quality/alerting)
- **`incident_ack.py`** - Acknowledge run incidents and satisfy ack-SLA tracking
- **`validate_incident_ticketing.py`** - Validate ack-SLA breach -> ticket workflow
- **`validate_incident_contacts.py`** - Validate incident ownership/escalation docs have no unresolved placeholders
- **`staging_incident_ticket_drill.py`** - Execute a real staging webhook drill for incident ticketing and write evidence artifacts
- **`validate_incident_ticket_drill_evidence.py`** - Gate drill evidence freshness, non-test target verification, delivery success, and response-link presence
- **`replay_notification_dead_letters.py`** - Replay failed SLO/ticket webhook notifications from dead-letter JSONL
- **`runtime_watchdog.py`** - Runtime connectivity watchdog for Alpaca account access, incident webhook delivery, IB socket reachability, and IB API session handshake readiness
- **`runtime_industrial_gate.py`** - Unified runtime readiness gate (incident docs + chaos drills + optional real webhook drill + optional live failover probe)
- **`go_live_precheck.sh`** - One-command go-live precheck wrapper for incident docs, deployment preflight, runtime watchdog, and runtime industrial gate
- **`ops_status_report.py`** - Generate compact daily ops status from run artifacts
- **`export_ops_metrics.py`** - Export operational metrics snapshot to JSON + Prometheus text format
- **`push_ops_metrics.py`** - Export + optionally push Prometheus metrics to external Pushgateway/webhook endpoint
- **`incident_response_automation.py`** - Automate incident escalation and optional rollback actions from readiness/SLO metrics
- **`deploy_canary.py`** - Command-driven canary rollout with health-check + rollback automation
- **`secrets_audit.py`** - Secrets rotation/leak audit against inventory policy
- **`governance_gate.py`** - Compliance/governance gate for live-capital promotion approval

## Backtesting

- **`simple_backtest.py`** - Basic backtesting script
- **`validated_backtest_report.py`** - Validated backtest with profitability gates and report output
- **`generate_validation_artifacts.py`** - One-shot reproducible validation artifacts (report, manifest, paper summary)
- **`strategy_promotion_gate.py`** - Strict research-to-prod promotion checklist and CI gate
- **`paper_burn_in_scorecard.py`** - Long-horizon paper-trading burn-in scorecard + sign-off gate

## Development

- **`mock_strategies.py`** - Mock strategies for testing
- **`mcp_server.py`** - MCP server for integration
- **`mcp.json`** - MCP configuration

## Usage

Prefer `main.py` for normal runtime control. For script helpers, run them from the project root with `uv`:

```bash
# Canonical runtime entrypoint
uv run python main.py live --strategy MomentumStrategy --force

# Quick start
uv run python -m scripts.quickstart

# Dashboard
uv run python -m scripts.dashboard
```
