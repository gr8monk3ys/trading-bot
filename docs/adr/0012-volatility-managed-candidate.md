# 0012. A second candidate: volatility-managed equity

Date: 2026-11-10. Status: accepted (candidate failed dev).

## Context

ADR 0011 set a pass bar and a locked holdout. The trend candidate failed
dev. A second candidate should rest on a different reason for an edge, so
one failure does not predict the other.

## Decision

1. The reason for the edge, written before any backtest: volatility
   clusters, so last month's volatility predicts next month's, but expected
   returns do not rise to match. Holding less after volatile months should
   cut risk more than return (Moreira and Muir 2017). The known weakness:
   Cederburg et al. (2020) find the gain unreliable in real time for many
   factors.
2. Rules (`strategies/vol_managed_strategy.py`, `VolManagedParams`):
   SPY/QQQ/IWM/EFA/EEM; on the first session of each month, weight =
   min(1, 15% / realised volatility of the last 21 sessions); each slot is a
   fifth of equity times its weight; resize only when the target moves more
   than 25% of a slot. Long only, no margin. The values came from the paper
   and were never tuned.
3. The same gate as ADR 0011: `scripts/run_candidate_gate.py --candidate
   volmanaged --window dev`, now generic over candidates. The holdout opens
   only for a dev PASS.

## Result

Dev window: FAIL (`results/gate_volmanaged_dev.md`), with dividends
credited and equity valued on the simulated day. +100.4% vs +135.0% for SPY
at the same 79% exposure; Sharpe 0.50 vs 0.54; max drawdown -26.7% vs
-42.9%; 71 round trips. It lost to SPY in both halves. Alpha +0.6%/yr with
t = 0.30: nothing there. The holdout stays unopened.

## Consequences

Two candidates with different reasons for an edge both fail the same fair
test. Both give a smoother ride than SPY but less return than SPY scaled to
the same exposure. Lowering exposure to SPY directly gives that trade more
cheaply. A third candidate is worth running only with a new reason for an
edge, not by tuning these two.
