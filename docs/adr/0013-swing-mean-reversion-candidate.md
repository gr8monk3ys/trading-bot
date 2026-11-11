# 0013. A third candidate: swing mean reversion, and honest costs

Date: 2026-11-10. Status: accepted. Written before the candidate's first backtest.

## Context

Two monthly candidates failed dev (ADR 0011, 0012). The owner asked whether
swing trading (holding days, not months) could win. Swing trades 10-20 times
more often than a monthly strategy, so the cost per fill decides the answer.

While preparing this, we found that the stated costs were never applied.
`config.BASELINE` says 40 bps slippage + 10 bps spread per fill, and every
report prints that, but no code passes those numbers to the backtest broker.
The broker charges its own dynamic model: a 3 bps base spread scaled by
volume and volatility, plus a square-root market-impact term. For $100k in
liquid ETFs that is a few bps per fill. So the momentum baseline and both
earlier candidates ran far cheaper than their reports said. They failed
anyway, so their verdicts stand; only the cost labels were wrong.

## Decision

1. `BacktestBroker(fixed_cost_bps=...)` and `run_backtest(fixed_cost_bps=...)`
   charge a flat cost per market fill. Every result now reports
   `mean_fill_cost_bps`, the cost actually charged, so a label can no longer
   drift from the code.
2. `scripts/run_candidate_gate.py --cost-bps` sets the flat cost. The default
   is 50, which is ADR 0011 as written (40 + 10). `--cost-bps engine` keeps
   the dynamic model. Artifacts record both the setting and the measured cost.
3. The reason for the edge, written before any backtest: index ETFs that
   drop sharply for a few days inside a long-term uptrend tend to bounce
   within a week. Forced selling (margin calls, fund outflows, volatility
   targeting) pushes prices below fair value for a short time, and buyers
   are paid to supply liquidity. Connors and Alvarez (2008, "Short Term
   Trading Strategies That Work") published the rules below, tested mostly
   on data before 2008, so our dev window (2008-2019) is largely after
   publication. Known weakness: the effect is widely reported to have
   shrunk after about 2010, and each trade earns well under 1%.
4. Rules (`strategies/swing_rsi_strategy.py`, `SwingRSIParams`), the book's
   values, never tuned: SPY/QQQ/IWM/DIA; buy when the close is above its
   200-day average and the 2-day RSI is below 10; sell when the close is
   above its 5-day average. Each position is a quarter of equity. Long only,
   no stop, no margin.
5. One engine limit makes this test harsher than the book. The backtester
   decides on yesterday's close and fills at today's close, so every entry
   and exit is one day late. The book trades at the signal day's close.
   We keep the engine as it is (no look-ahead) and say so in the result.
6. Verdicts, fixed now:
   - Formal: dev window at 50 bps per fill (ADR 0011 as written).
   - Provisional: dev window at 10 bps per fill. Real SPY/QQQ spreads are
     about 1 bp, so 10 bps is still conservative for retail market orders,
     but it is not measured. It becomes formal only after Alpaca paper fills
     measure real costs and a later ADR adopts the number.
   - Informational: `--cost-bps engine`.
   The holdout opens only for a PASS at a cost level backed by measured fills.

## Multiple testing

This is the fourth idea tested on the 2008-2019 window (momentum, trend,
volatility-managed, swing). Each extra look raises the chance that one
passes by luck. A dev PASS here is a reason to measure costs and open the
holdout, not evidence of an edge on its own.

## Result

Dev window 2008-2019, 367 round trips, average gross exposure 11%, so the
benchmark is SPY total return held at 11% (+14.8%, Sharpe 0.54):

| Cost per fill | Return | Sharpe | Max drawdown | 2008-13 vs SPY | 2014-19 vs SPY | Alpha/yr (t) | Verdict |
|---|---|---|---|---|---|---|---|
| 50 bps (formal) | -43.4% | -0.72 | -45.0% | -16.4% vs +5.9% | -32.3% vs +8.4% | -5.8% (-3.41) | FAIL |
| 10 bps (provisional) | +17.7% | 0.25 | -15.2% | +16.2% vs +5.9% | +1.3% vs +8.4% | +0.3% (0.20) | FAIL |
| engine, 4.6 bps (info) | +29.9% | 0.39 | -15.1% | +21.2% vs +5.9% | +7.1% vs +8.4% | +1.1% (0.69) | FAIL |

Artifacts: `results/gate_swing_dev.md`, `gate_swing_dev_cost10.md`,
`gate_swing_dev_engine.md`.

It fails at every cost level. At realistic costs it beats same-exposure SPY
over the whole window, but only because of 2008-2013; over 2014-2019 it
loses. Its Sharpe is below SPY's at every cost, and its drawdown is twice
that of SPY at the same exposure. This matches the known decay. An
exploratory check outside the engine (not a gate run) found about +0.4%
per trade before costs, and that the engine's one-day fill delay does not
lower it on average, so the delay is not the reason it fails.

The holdout stays unopened. Measuring Alpaca fill costs would not change
this verdict: the strategy fails even at 4.6 bps.

## Consequences

- Trend and vol-managed were rerun at 50 bps; those runs replace their old
  `gate_*_dev` artifacts. Their `_engine` reruns reproduce the earlier
  numbers exactly, which confirms the earlier runs used the dynamic model.
- `scripts/run_etf_baseline.py` and `scripts/run_honest_baseline.py` still
  print "40 bps slippage + 10 bps spread" for the momentum baseline, which
  also ran on the dynamic model. Fixing that changes the canonical baseline
  (ADR 0002), so it is a separate change.
- Four ideas have now failed the same window. Another candidate needs an
  edge that does not come from US index ETF prices alone.
