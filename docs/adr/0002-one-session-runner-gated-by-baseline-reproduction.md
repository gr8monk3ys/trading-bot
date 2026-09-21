---
status: accepted
---
# One session runner for live and backtest, gated by baseline reproduction

The backtest engine never calls `on_bar`, so the exit rules inside it never ran in a backtest (#84); the fix added a third exit implementation in `momentum_strategy_backtest.py` rather than unifying the loop. We decided there is one session runner, parameterised by a clock (websocket or historical window) and a broker adapter, and `momentum_strategy_backtest.py` is deleted once the single `MomentumStrategy` reproduces it through parameters.

Because the refactor can change what a backtest measures, every step is gated: `scripts/run_etf_baseline.py` must reproduce the committed `results/etf_baseline_2020-2024_gross*.json` numbers before merge. The first unification run keeps trailing stops off in backtest so the numbers match; a second, separately documented run with trailing stops active becomes a new results artifact and, if it differs, supersedes the old verdict in the usual way.

## Considered options

- Keep two loops and document the gotcha (rejected: the gotcha already cost one corrupted baseline).
- Unify and accept whatever the numbers become (rejected: silent verdict drift is the thing the repo exists to avoid).

## Outcome (2026-09-21)

The unified session reproduced the baseline exactly. The second run (`scripts/run_etf_baseline.py --exits`, trailing stops active in daily mode) gave 50 trades, +9.1% / Sharpe -0.02 at gross-100 versus +16.3% / 0.16 without; it did not change the verdict and the canonical artifacts stay the run without trailing stops (see `results/where_we_landed.md`).
