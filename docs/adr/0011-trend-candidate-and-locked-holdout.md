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
   `scripts/run_trend_gate.py --window holdout` writes
   `results/holdout_lock.json` on first use; any second look needs
   `--reopen "<reason>"`, and the reason stays in git.
3. The pass bar, after the engine's costs: beat SPY held at the same average
   gross exposure (price-only, because the engine credits no dividends); at
   least 100 closed round trips; beat it in both halves of the window.
4. A candidate that fails dev does not open the holdout. Open the holdout
   only for a candidate that passes dev.
5. Paper trading (`scripts/run_trend_paper.py`) checks execution only. It
   cannot show an edge in months. Live money needs a holdout PASS first,
   then at least 3 months of paper with slippage at or below the 50 bps
   the backtest assumes.

## Result so far

Dev window: FAIL (`results/trend_gate_dev.md`). The candidate cut the
drawdown from 43% to 16% but returned less than SPY at the same exposure, and
lost to it over 2014-2019. The holdout stays unopened.

## Consequences

The engine's lack of dividends costs this candidate more than SPY, because
it holds bond and REIT ETFs. Adding dividend credit to the backtest broker is
the next fix before judging any multi-asset candidate.
