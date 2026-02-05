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
- Restart recovery verified with open positions.

## Strategy Checkpoint Coverage
- MomentumStrategy: stop/target/entry/peak and last signal time.
- MeanReversionStrategy: position entries, peaks/troughs, last signal time.
- BracketMomentumStrategy: active bracket orders.
- EnsembleStrategy: position entries, peaks, ensemble signals.
- PairsTradingStrategy: open pair positions.

## Validation Artifacts
- Run `python scripts/generate_validation_artifacts.py --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2014-01-01 --end-date 2024-12-31`.
- Ensure `results/validation/<timestamp>/manifest.json` matches current git SHA.
- Review `paper_trading_summary.json` and go‑live blockers before live trading.

## Gaps To Resolve Before Live Capital
- Order lifecycle persistence is in place, but broker order reconciliation does not validate fills vs lifecycle history.
- Strategy state checkpointing is light and does not persist indicators or cached data.
- Alerting pipeline (Slack/Email/Discord) is optional and not enforced.
- No automated post‑trade P&L attribution or execution quality report in CI.

## Go‑Live Gate
- 30+ paper trading days and 30+ paper trades.
- Profitability gates pass in validated backtests.
- No reconciliation mismatches over at least 5 trading days.
