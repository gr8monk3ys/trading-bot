# TODO

Follow-ups after the 2026-05 honest cleanup and the 2026-08 slop sweep.
Numbers here are kept current deliberately — a stale TODO is how the last
round of fiction got in.

## Direction

- [x] **"Maximize profit by tuning this strategy" is answered: no.** Three
  rounds of evidence, each one fixing a defect that had been flattering the
  previous round. Final: at gross-100 target the momentum strategy returns
  **+16.4% / Sharpe 0.16** over 2020-2024 vs **SPY buy-and-hold +95.3% /
  Sharpe 0.75** (`results/etf_baseline_2020-2024_exposure_sweep.md`). Its own
  exit timing subtracts value at every exposure level.
- [ ] **Choose what the repo is for now that profit is disconfirmed.** The
  honest options: (a) hold the benchmark and keep this as an execution/
  infrastructure sandbox, (b) validate the pairs-trading sleeve in `research/`
  as a structurally different edge, (c) archive it. Do not add strategy
  features before deciding.

## Code organization

- [x] **Measure file sizes.** Done 2026-08-19: ~25K LOC across the production
  tree. Only two files exceed the 800-LOC soft limit —
  `engine/performance_metrics.py` (971) and `utils/database/core.py` (873).
  `main.py` is 781. Neither outlier is being actively worked on, so neither is
  worth splitting yet; split them if you start touching them.
- [x] **Audit the kept `utils/` modules.** Done 2026-08-18 (PR #83): 23 of 41
  `utils/` modules had zero importers anywhere in the production tree and were
  deleted, along with `brokers/ib_broker.py`, `brokers/multi_broker.py`,
  `engine/parameter_stability.py`, and ~720 tests that only covered dead code.
  18 modules remain. `multi_timeframe.py` is **not** vestigial — seven strategy
  files reference it — though it is disabled by default and cannot be exercised
  by the daily-bar backtest.

## Engine and order-path bugs

- [x] **Wire an `OrderGateway` into `BacktestEngine`** (`engine/backtest_order_gateway.py`).
- [x] **Fix short-trade P&L in `_calculate_trade_pnl`** — signed-qty state machine.
- [x] **Add an end-of-backtest liquidation pass** (`BacktestEngine._liquidate_open_positions`).
- [x] **Restore the live order path** (2026-08-17). The 2026-05 cleanup deleted
  the production OrderGateway while `BaseStrategy` blocks every order without
  one, so the live path had never placed a single order.
  `engine/live_order_gateway.py` restores it and wires the circuit breaker's
  per-order interlock. Proven end-to-end against the paper API on 2026-08-18 —
  see `scripts/paper_smoke_test.py`.
- [x] **Audit-log the order lifecycle** (2026-08-18, PR #81). The hash-chained
  log recorded only `ORDER_MODIFIED`; submissions, rejections, and cancels are
  now recorded too. The first smoke-test order produced a real order ID with a
  0-byte audit file — that is what surfaced this.
- [x] **Unbreak backtest exits** (2026-08-18, PRs #82 and #84). Two independent
  defects: `BaseStrategy.submit_exit_order` awaited `BacktestBroker`'s
  *synchronous* `get_positions` (every exit died in a swallowed `TypeError`),
  and the signal generator never emits `sell` while the engine never calls
  `on_bar` where the exit logic lives. Every backtest before this measured
  enter-once-and-hold.

## Validation (if continuing toward live)

- [ ] Run 6+ months of paper trading. The live path finally works, so this is
  now actually possible; it never was before 2026-08-17.
- [ ] Produce at least 50 real trades before quoting Sharpe or win rate. The
  current sweep has 26 and is marked `STATUS=INCONCLUSIVE`.
- [ ] Re-run `scripts/run_etf_baseline.py` quarterly; track drift in `results/`.
- [x] **Replace the hand-picked universe with a bias-free one.** Done via
  `scripts/run_etf_baseline.py` (SPY/QQQ/IWM/EFA — cannot be delisted or
  selection-biased).
- [ ] **Random S&P 500 sample testing** — parked. Only worth doing if some
  future change flips the ETF result to beating SPY. It currently loses by ~79
  percentage points.

## Untested corners of the production path

Surfaced by the 2026-08-18 strategy audit; each is a real gap, not a feature request.

- [ ] **Production trailing stops and bracket orders have never executed in any
  backtest.** They live in `on_bar`, which the engine never calls, and need
  intraday data the daily backtest cannot supply. The "VALIDATED - proven to
  capture extended moves" comment on those parameters is unsupported.
- [ ] **`MeanReversionStrategy` has never been backtested standalone** — no
  script runs it, no `results/` artifact exists. Its 6-clause AND entry gate is
  stricter than momentum's, so expect very few trades.
- [x] **`AdaptiveStrategy` did not switch strategies at all in live mode.**
  Fixed 2026-08-19: both arms self-subscribed to the broker's bar feed during
  their own `initialize()`, so they traded simultaneously — taking opposing
  positions, since momentum buys breakouts and mean reversion buys dips —
  while the coordinator, never subscribed, never ran `on_bar`, never called
  `_update_regime()`, and left `current_regime` at `None` forever. Adaptive now
  claims the subscription and detaches its arms; verified against a live broker
  (one bar → regime detected → arm switched).
- [ ] **`AdaptiveStrategy` still has no backtest evidence and cannot produce
  orders under the backtest engine** (its `execute_trade` delegates to
  `MomentumStrategy`'s no-op, and the engine never calls `on_bar`). It is no
  longer the CLI default. Its uncited "improves returns by 10-15% annually"
  docstring claim was deleted 2026-08-19 (nothing in the repo supported it), as
  was a tautological `test_research_impact_claims` asserting a hardcoded 5-8%
  figure against itself.
- [x] **The Bollinger filter is not inverted — it is load-bearing.** Measured
  2026-08-19 (`scripts/run_bollinger_ab.py`,
  `results/bollinger_filter_ab_2020-2024.md`): filter ON gives 26 trades and
  +16.44%; filter OFF gives 66 trades and **−1.88%** at *higher* exposure. The
  mean-reversion overlay mostly suppresses trades, and suppressing this
  strategy's trades is what keeps its return positive. The filter-off run is
  also the first configuration in this repo to clear the 50-trade significance
  bar, so "no edge" is now a measured result rather than a directional hint.
- [ ] **Production and backtest still disagree on this flag.** Production
  defaults keep the filter off; `MomentumStrategyBacktest` force-enables it.
  Given the A/B, production's setting is the worse of the two — reconcile them
  deliberately rather than leaving the backtest measuring a config that never
  runs live.

## Research-tree promotion

- [ ] Promotion bar for any `research/` module: (1) ≥50-trade out-of-sample
  backtest, (2) statistical-significance check (permutation or FDR), (3)
  written hypothesis, (4) evidence the signal isn't already priced. Document in
  `research/<module>/PROMOTION.md`.
- [ ] **Pairs trading is the only structurally plausible candidate.**
  `research/strategies/pairs_trading_strategy.py` is genuinely complete
  (Engle-Granger cointegration, Hurst, OU half-life, z-score entries/exits,
  market-neutral leg construction), and the validation harness it needs already
  exists in `research/engine/validated_backtest.py` (walk-forward + permutation
  tests with FDR correction, gated at 50 trades). Blockers, in order: two
  `datetime.now()` calls that make hedge-ratio recalculation and the max-holding
  exit no-ops in a simulated timeline; no engine adapter (all logic is in
  `on_bar`); short-leg accounting needs verifying; pair selection must be re-run
  inside each walk-forward window or it repeats the survivor-bias mistake; and
  `research/tests/unit/test_pairs_trading.py` never runs (`norecursedirs`).

## Operational

- [ ] If running unattended for long periods, re-evaluate which deleted
  operational scripts actually need to come back. Don't restore wholesale.
- [ ] Decide where the bot runs. It is a long-lived asyncio process, so
  serverless (Vercel) cannot host it; the realistic options are a local
  LaunchAgent, Railway, or a small VPS. Not worth paying for uptime until
  there is a strategy worth running continuously.
