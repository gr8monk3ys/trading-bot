# Scripts Directory

This directory contains various runner scripts and utilities for the trading bot.

## Main Entry Points

- **`../main.py`** - Primary entry point for production trading (supports live, backtest, optimize, replay, and research registry modes)
- **`../live_trader.py`** - Simplified live trading launcher

## Utilities

- **`dashboard.py`** - Real-time monitoring dashboard
- **`quickstart.py`** - Interactive setup wizard
- **`simple_trader.py`** - Simple trading script
- **`run.py`** - Alternative runner script
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

Most scripts can be run directly from the project root:

```bash
# Production trading
python main.py live --strategy auto

# Quick start
python scripts/quickstart.py

# Dashboard
python scripts/dashboard.py
```
