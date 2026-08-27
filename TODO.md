# TODO

Open items only. Completed work and the measured verdict ("beat the market?
No: +16.4% vs SPY +95.3%, 2020-2024") live in the README's Results section and
`results/where_we_landed.md`. Numbers here are kept current deliberately — a
stale TODO is how the last round of fiction got in.

## Direction

- [ ] **Choose what the repo is for now that profit is disconfirmed.** The
  honest options: (a) hold the benchmark and keep this as an execution/
  infrastructure sandbox, (b) validate the pairs-trading sleeve in `research/`
  as a structurally different edge, (c) archive it. Do not add strategy
  features before deciding.

## Validation (if continuing toward live)

- [ ] Run 6+ months of paper trading. The live path finally works, so this is
  now actually possible; it never was before 2026-08-17.
- [ ] Produce at least 50 real trades before quoting Sharpe or win rate. The
  current sweep has 26 and is marked `STATUS=INCONCLUSIVE`.
- [ ] Re-run `scripts/run_etf_baseline.py` quarterly; track drift in `results/`.
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
- [ ] **`AdaptiveStrategy` still has no backtest evidence and cannot produce
  orders under the backtest engine** (its `execute_trade` delegates to
  `MomentumStrategy`'s no-op, and the engine never calls `on_bar`). It is no
  longer the CLI default. Its uncited "improves returns by 10-15% annually"
  docstring claim was deleted 2026-08-19 (nothing in the repo supported it), as
  was a tautological `test_research_impact_claims` asserting a hardcoded 5-8%
  figure against itself.
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
