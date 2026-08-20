# ETF baseline exposure sweep — 2020-2024


Same `MomentumStrategyBacktest` signals on SPY/QQQ/IWM/EFA; only the
sizing target varies (equity-based, per-position = target / 4).
This is the first sweep in which the strategy can EXIT on its own
signal (opposite-signal exits, added 2026-08-18). Every earlier run —
including the 2026-08-17 '+42.9% at 92% gross' — measured
enter-once-and-hold-to-liquidation: the signal generator never emitted
'sell' and the engine never ran the exit path, so those numbers priced
market beta, not the strategy. None of the older reports are
comparable.


**Reading:** compare each row against SPY buy-and-hold at the same
realized gross exposure. Exits reduce realized exposure (positions
spend time flat), so per-unit-of-exposure return is the honest
column: with exits active the signal's timing must add value to beat
just holding — if returns fall relative to the hold-only run, the
timing subtracts value.


| Target gross | Avg gross | Peak gross | Trades | Total return | Sharpe | Max DD | SPY B&H | SPY Sharpe |
|---|---|---|---|---|---|---|---|---|
| 25% | 13.86% | 26.18% | 26 | 4.12% | -0.50 | 3.21% | 95.30% | 0.75 |
| 50% | 28.40% | 52.72% | 26 | 8.37% | -0.06 | 6.42% | 95.30% | 0.75 |
| 100% | 59.07% | 104.57% | 26 | 16.42% | 0.16 | 12.90% | 95.30% | 0.75 |

Per-run artifacts: `results/etf_baseline_2020-2024_gross25.json`, `results/etf_baseline_2020-2024_gross50.json`, `results/etf_baseline_2020-2024_gross100.json`
