# Production Runbook

## Purpose
Operational checklist for running this trading bot safely in paper or live mode.

## Pre-Flight
- Verify API keys are set in `.env` and are **paper** keys for testing.
- Run `uv run pytest` and ensure coverage gate passes.
- Generate validation artifacts and review gates:
  - `python scripts/generate_validation_artifacts.py --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2014-01-01 --end-date 2024-12-31`
- Confirm audit logs are writable: `audit_logs/`.

## Start
- Paper/live mode (main entry):
  - `python main.py live --strategy MomentumStrategy --force`
- Monitoring:
  - `python monitor_bot.py 60`
  - `tail -f momentum.log | grep -E "ORDER|FILL|REJECT|HALT"`

## Health Checks (every 5–10 min)
- Broker connectivity and websocket running.
- Audit log events flowing (`audit_logs/` growing).
- Reconciliation loop has no mismatches.
- Circuit breaker status not triggered.

## Alerts & Thresholds
- Circuit breaker: 3% daily loss (default).
- Reconciliation mismatch: investigate immediately.
- Partial fills below 90% fill rate: review liquidity and slippage.

## Incident Response
1. **Pause trading**: run kill switch if needed.
2. **Cancel orders**: `python scripts/kill_switch.py --confirm "HALT TRADING" --cancel-orders`
3. **Liquidate**: `python scripts/kill_switch.py --confirm "HALT TRADING" --cancel-orders --liquidate`
4. **Collect artifacts**: logs, audit logs, validation results.

## Shutdown
- Ctrl+C in live process.
- Confirm positions are closed or intentionally held.
- Verify audit log closure.
