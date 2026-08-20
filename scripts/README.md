# scripts/

Operational scripts. This list is the complete, current inventory — a 2026-08
audit found the previous README documented 28 scripts that no longer existed.

| Script | Purpose |
|---|---|
| `run_etf_baseline.py` | The canonical backtest baseline: SPY/QQQ/IWM/EFA 2020-2024 exposure sweep. Produces `results/etf_baseline_2020-2024_gross{25,50,100}.{md,json}` and `_exposure_sweep.md`. |
| `run_honest_baseline.py` | The older hand-picked-mega-cap baseline (survivor-biased). SUPERSEDED; kept for the audit trail. |
| `run_bollinger_ab.py` | A/B of `use_bollinger_filter` on the same universe/window/exposure. Produces `results/bollinger_filter_ab_2020-2024.{md,json}`. The filter-off run is the only configuration here that clears the 50-trade significance bar. |
| `paper_smoke_test.py` | End-to-end proof of the live order path against the Alpaca paper API: submits a 1-share unfillable limit order through the gateway + circuit breaker, verifies the audit log, cancels. Run after any change to the order path. |
| `check_positions.py` | One-shot paper-account status query. |
| `dashboard.py` | Terminal monitoring dashboard. |
| `monitor_bot.py` | Real-time monitoring dashboard. |
| `kill_switch.py` | Emergency halt: cancels orders and liquidates positions. |
| `simple_trader.py` | Minimal trading-bot runner. |
| `quickstart.py` | Interactive setup helper. |

All scripts assume a `.env` at the repo root (see `.env.example`).
