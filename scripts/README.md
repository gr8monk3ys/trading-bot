# Scripts Directory

This directory contains various runner scripts and utilities for the trading bot.

## Main Entry Points

- **`../main.py`** - Primary entry point for production trading (supports live, backtest, optimize modes)
- **`../live_trader.py`** - Simplified live trading launcher

## Utilities

- **`dashboard.py`** - Real-time monitoring dashboard
- **`quickstart.py`** - Interactive setup wizard
- **`simple_trader.py`** - Simple trading script
- **`run.py`** - Alternative runner script
- **`run_now.py`** - Quick start script
- **`kill_switch.py`** - Emergency kill switch (cancel orders / liquidate)

## Backtesting

- **`simple_backtest.py`** - Basic backtesting script
- **`validated_backtest_report.py`** - Validated backtest with profitability gates and report output
- **`generate_validation_artifacts.py`** - One-shot reproducible validation artifacts (report, manifest, paper summary)

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
