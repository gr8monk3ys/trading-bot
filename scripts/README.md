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
