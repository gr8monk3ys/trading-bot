# 0011. A trend candidate, a fixed pass bar and a locked holdout

Date: 2026-11-10. Status: accepted.

## Context

Every strategy on main lost to SPY held at the same exposure. Past results
went wrong in the same way each time: the strategy was changed after looking
at the test period, so the test period became training data.

## Decision

1. A new strategy must state, before any backtest, why its edge should
   exist. The first candidate is time-series trend following
   (`strategies/trend_following_strategy.py`): 10 diversified ETFs, hold one
   when its 12-month return is above zero, monthly, long only, equal slots.
   The parameters come from the published literature and were never tuned.
2. Two windows. Dev: 2008-2019. Holdout: 2020-01-01 to 2026-10-30.
   `scripts/run_candidate_gate.py --candidate <name> --window holdout` writes
   `results/holdout_lock.json` on first use; any second look needs
   `--reopen "<reason>"`, and the reason stays in git.
3. The pass bar, after the engine's costs: beat SPY held at the same average
   gross exposure (total return: the engine credits dividends since
   2026-11-10, and SPY is compared with dividends reinvested); at
   least 100 closed round trips; beat it in both halves of the window.
4. A candidate that fails dev does not open the holdout. Open the holdout
   only for a candidate that passes dev.
5. Paper trading (`scripts/run_trend_paper.py`) checks execution only. It
   cannot show an edge in months. Live money needs a holdout PASS first,
   then at least 3 months of paper with slippage at or below the 50 bps
   the backtest assumes.

## Result so far

Dev window: FAIL (`results/gate_trend_dev.md`), with dividends credited and
equity valued on the simulated day (PR #148). +66.0% vs +111.1% for SPY at
the same 67% exposure; Sharpe 0.55 vs 0.54; max drawdown -13.8% vs -37.5%;
96 round trips. It beat SPY over 2008-2013 and lost over 2014-2019. Alpha
+2.1%/yr with t = 1.06, which is not distinguishable from zero. The holdout
stays unopened.

## Consequences

Dividend credit is now in the backtest broker (opt-in per run, so the
momentum baseline is unchanged by it). The trend result is a lower-risk
portfolio, not an edge over SPY. ADR 0012 runs a second candidate through
the same gate.
