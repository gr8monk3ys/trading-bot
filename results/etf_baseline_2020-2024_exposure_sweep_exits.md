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
| 25% | 5.80% | 20.96% | 50 | 2.34% | -1.29 | 1.10% | 95.30% | 0.75 |
| 50% | 11.83% | 44.68% | 50 | 4.72% | -0.43 | 2.20% | 95.30% | 0.75 |
| 100% | 24.09% | 100.49% | 50 | 9.09% | -0.02 | 4.59% | 95.30% | 0.75 |

Per-run artifacts: `results/etf_baseline_2020-2024_gross25_exits.json`, `results/etf_baseline_2020-2024_gross50_exits.json`, `results/etf_baseline_2020-2024_gross100_exits.json`
