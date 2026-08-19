# Where we landed

**Date:** 2026-05-11 · **Addenda:** 2026-08-17 and 2026-08-18 (read them first — each supersedes what's below it)
**Branch:** main (cleanup branch merged in commit `784e5d8`; follow-up bug-fix commits through `107c893`)

This is the durable summary of the cleanup + validation work done in May 2026. Read this first if you (or a future Claude session) are returning to this repo.

---

## Addendum 2026-08-18: the strategy could never exit — and with exits, it's worse

A day after the 2026-08-17 fixes below, a strategy audit found the backtest still wasn't testing the strategy: **the signal generator only ever emits `buy`/`short`/`neutral` — never `sell` — and the engine never calls `on_bar`, where all exit logic (trailing stops, `_check_exit_conditions`) lives.** So the "8 trades in 5 years" below were 4 entries plus 4 forced end-of-backtest liquidations; not one exit was ever the strategy's own decision. The 2026-08-17 sweep measured *enter-once-and-hold*, i.e. market beta, not the momentum system. (A second silent killer compounded it: `BaseStrategy.submit_exit_order` awaited `BacktestBroker`'s synchronous `get_positions`, so even an emitted exit died in a swallowed `TypeError`.)

Both fixed 2026-08-18: opposite-signal exits (bearish signal while long closes the long; bullish while short covers) and the async mismatch. The rerun is the first backtest in this repo's history where the strategy can exit on its own signal:

| Target gross | Avg gross | Trades | Total return | Sharpe | vs hold-only (08-17) |
|---|---|---|---|---|---|
| 25% | 14% | 26 | +4.1% | -0.50 | was +9.9% / -0.02 |
| 50% | 28% | 26 | +8.4% | -0.06 | was +20.4% / 0.33 |
| 100% | 59% | 26 | +16.4% | 0.16 | was +42.9% / 0.52 |
| SPY B&H | 100% | — | +95.3% | 0.75 | — |

**The strategy's own exit timing destroys value at every exposure level.** With exits active, realized exposure drops (positions spend time flat) *and* return per unit of exposure falls — +16.4% at 59% avg gross (~0.28x per-exposure) vs the hold-only +42.9% at 92% (~0.47x) vs SPY's 0.95x. The signal's timing is anti-predictive on this universe: every exit it took was, on net, a mistake it re-bought later at a worse price. 26 trades is still under the 50-trade significance bar (STATUS=INCONCLUSIVE), but the direction is consistent across all three exposure targets and both eras.

**The verdict tightens:** the 2026-08-17 "+42.9%" was never the strategy — it was beta wearing the strategy's name. The actual momentum system, exits and all, delivers roughly a sixth of SPY's return. Tuning it remains a dead end; the honest routes are unchanged — hold the benchmark, validate the pairs sleeve in `research/`, or treat this repo as infrastructure practice. Notably untested still: the production trailing stops (they need intraday `on_bar` data the daily backtest can't drive), the Bollinger filter A/B, and the mean-reversion strategy standalone.

---

## Addendum 2026-08-17: the May numbers were corrupted; the verdict got *stronger*

*(Superseded by 2026-08-18 above — the "8 trades" here were enter-and-hold artifacts.)*

Three defects invalidated every number in this document (all fixed on this date):

1. **The backtest strategy could never see its own positions** — an operator-precedence bug in `momentum_strategy_backtest.py` made position lookup always return None for the objects `BacktestBroker` actually returns. Signal exits were unreachable, buys re-entered held symbols, shorts fired against longs.
2. **Naked short sells booked no liability** — `place_order`'s sell path credited the proceeds and created no position, so every short signal was free money.
3. **Sizing was 10% of *remaining cash*** — the strategy averaged ~25% gross exposure, so "-7% max drawdown" measured idle cash, not risk control, and the SPY comparison was 25%-deployed vs 100%-deployed.

Separately, the **live paper path had never placed an order at all**: the 2026-05 cleanup deleted the production OrderGateway while `BaseStrategy` blocks every entry/exit without one. Restored via `engine/live_order_gateway.py` (which also wires the circuit breaker's per-order interlock, previously dead code).

The post-fix rerun (`results/etf_baseline_2020-2024_exposure_sweep.md`) sweeps target gross exposure so the SPY comparison is finally like-for-like:

| Target gross | Avg gross | Trades | Total return | Sharpe | Max DD |
|---|---|---|---|---|---|
| 25% | 23% | 8 | +9.9% | -0.02 | -6.7% |
| 50% | 47% | 8 | +20.4% | 0.33 | -12.8% |
| 100% | 92% | 8 | +42.9% | 0.52 | -23.3% |
| SPY B&H | 100% | — | +95.3% | 0.75 | -33.7% |

**At matched exposure the strategy earns less than half of SPY's return with a worse Sharpe, and max drawdown scales almost linearly with exposure** — the drawdown-control story was mostly an exposure artifact (at 100% target it still beats SPY's DD, but at a Sharpe cost no allocator would pay). 8 trades in 5 years is far below any significance bar; the signal barely fires on broad indices. There is no configuration of this strategy on this universe that beats holding SPY.

**What this means for "maximize profit":** tuning this strategy is a dead end — the honest ceiling is below passive. The live routes that remain are (a) hold the benchmark, (b) research a genuinely different edge (the pairs-trading sleeve in `research/` is the only structurally plausible candidate, unvalidated), or (c) treat the bot as an execution/infrastructure sandbox, which is what it is.

---

## The journey, in one paragraph

The repo was a 193K-LOC trading bot claiming "institutional-grade, 10/10, suitable for live capital deployment" backed by a +42.68% backtest with 9 trades — a number the repo's own `PROFITABILITY_RESEARCH.md` admitted was statistically meaningless. An audit cut the unjustified narrative and unvalidated surface area, reducing the repo to ~50K LOC of honest core. A first backtest on hand-picked mega-caps produced a flattering +646% number that several methodology fixes (gateway routing, short-trade PnL accounting, end-of-period liquidation) didn't change much. A bias-free ETF baseline did. **The strategy underperforms SPY buy-and-hold by ~40 percentage points on total return.** Its real value is drawdown control, not profit maximization.

---

## What the evidence says

### ETF baseline (the honest test)

`results/etf_baseline_2020-2024.md` — strategy on SPY, QQQ, IWM, EFA (zero survivorship bias) over 2020–2024:

|  | Strategy | SPY buy-hold | QQQ buy-hold |
|---|---:|---:|---:|
| Total return | **+53.4%** | +95.3% | +146.0% |
| CAGR | 8.94% | ~14% | 19.75% |
| Sharpe (rf=0) | 0.78 | 0.75 | 0.83 |
| Max drawdown | **-7.0%** | -33.7% | -35.1% |
| Calmar | 1.27 | ~0.42 | ~0.56 |
| Trades | 38 | n/a | n/a |

The 38-trade count is below the 50-trade statistical significance bar. **The directional finding is a strong hint, not a verdict**: more data may shift the exact numbers, but the underperformance vs SPY on total return is too large to be explained by sampling alone.

### Survivor-biased baseline (cautionary contrast)

`results/honest_backtest_2020-2024.md` — same strategy on 10 hand-picked mega-caps: +646% / Sharpe 1.36 / MaxDD -47%. Headline is real bookkeeping but reflects the universe (NVDA, TSLA, AAPL, etc. crushed it 2020–2024), not strategy edge.

### Sanity check

`results/can_this_beat_qqq.md` — pre-execution analysis predicted exactly this outcome: solo-dev momentum bot likely loses to QQQ on total return but matches or slightly beats on Sharpe over a full cycle, with real drawdown-control value. The evidence confirmed the prediction.

---

## The honest interpretation

This strategy has three distinct profiles depending on what you're trying to optimize:

**For total return:** Loses. Underperforms SPY by 40 pp, QQQ by 90 pp over 2020–2024. The "maximize profit" framing this repo was built under is wrong.

**For risk-adjusted return (Sharpe):** Roughly tied with SPY (0.78 vs 0.75), slightly worse than QQQ (0.83). Not a winning edge.

**For drawdown control:** Genuinely interesting. -7.0% max DD vs -33% to -35% for passive. The trend filter sidesteps the worst declines. Calmar ratio is meaningfully better. If you'd otherwise panic-sell during a -30% drawdown, a systematic bot that mechanically gets you out and back in has real psychological value.

---

## Decisions made

1. **The repo will not be developed further toward "maximize profit"** unless the user revisits this finding. The evidence doesn't support that goal.

2. **The repo IS left in a usable state** for someone who wants to deploy a drawdown-control sleeve — but only after:
   - Targeted refactor (consolidate the 3 entry points to 1; split `alpaca_broker.py` at 2,977 LOC; audit the 46-file `utils/` for vestigial modules). See "If you decide to deploy" below.
   - Real paper trading for 6+ months with proper measurement.
   - A statistical-significance check that crosses the 50-trade bar.

3. **The two engine bugs found during validation are fixed:**
   - `OrderGateway` is now wired into `BacktestEngine` automatically (`engine/backtest_order_gateway.py`).
   - `_calculate_trade_pnl` handles short positions correctly (signed-qty state machine).
   - `BacktestEngine` liquidates open positions at end-of-backtest so total-return reflects realized P&L.

4. **The kept core is small and reasonable:** `MomentumStrategy` + `RiskManager` + `CircuitBreaker` + `AlpacaBroker` + `BacktestEngine` + `AdaptiveStrategy` regime switcher. Everything speculative is in `research/`.

---

## What's still on the table (TODO.md)

If you come back to this repo, the open items are:

- **Direction decision (the only real one).** "Maximize profit" is now disconfirmed; either reframe to drawdown control or accept the conclusion.
- **Code organization debt.** `main.py` 495 LOC, `live_trader.py` 332 LOC, `adaptive_strategy.py` 420 LOC are all under the 800 LOC soft limit. The real outliers are `alpaca_broker.py` (2,977 LOC), `backtest_engine.py` (1,244 LOC), `momentum_strategy.py` (1,229 LOC). Worth splitting only if you're actively going to touch them.
- **`utils/` audit.** Probably 10–20 files in there are unused after the cleanup. Quick win.
- **Survivor-bias extension.** If you want a stronger test than 38 ETF trades, add a random S&P 500 sample test.
- **Paper trading validation.** If you decide to deploy.

---

## What was deleted vs quarantined vs kept

Net effect of the cleanup branch (`cleanup/honest-2026-05-11`, merged to main):

| Action | Volume | Categories |
|---|---:|---|
| **Deleted** | ~37,000 LOC | LSTM, DQN, ensemble predictor, LLM analysis pipeline (earnings/Fed/SEC/news), alt-data scrapers, options broker, 4 momentum-strategy variants, ensemble strategies, news sentiment (FinBERT), 30+ operational scripts, 14 ops utils, 11 GitHub workflows, infra/ tree |
| **Quarantined** | ~10,000 LOC | `research/`: factor models, pairs trading, walk-forward, validated backtest, alpha-decay, IC tracker, historical universe, cross-asset signals, feature store, point-in-time, 13 factor files |
| **Kept** | ~50,000 LOC | The honest core (momentum, risk manager, circuit breaker, broker, backtest engine) plus utilities still tied to active code |

Documentation reduced from ~30 markdown files to ~7. The "Phase 1–9" / "10/10 institutional-grade" framing is gone from every file.

---

## If you decide to deploy

The strategy might be deployable as a drawdown-control sleeve, NOT a primary investment vehicle. Minimum work before any real capital touches it:

1. **Cross the 50-trade significance bar.** Run real paper trading for 6+ months. The 38-trade ETF result is suggestive but not conclusive.
2. **Compare to benchmark continuously.** SPY buy-and-hold is the bar. If the bot can't tie or beat SPY on Sharpe over the paper period, retire.
3. **Cap exposure honestly.** Even after validation, treat this as 5–10% of liquid net worth at most. Edge in financial markets is rare and decays. Don't size into it like it's free money.
4. **Have a kill-switch.** If the strategy loses 15% from peak, stop and re-evaluate. Don't override.
5. **Do the refactor.** Consolidate `main.py` / `live_trader.py` / `run_adaptive.py` to a single entry point. Split `alpaca_broker.py`. This makes the code auditable, which matters when real money is involved.

If you're not committing to all five, don't deploy. The realistic outcome is then "interesting artifact, not income source."

---

## The single most important sentence

> **The strategy in this repo, on honest evidence, does not maximize total return. It has real drawdown-control value but underperforms SPY buy-and-hold by ~40 percentage points on a 5-year bias-free test.**

If you remember nothing else from this work, remember that.
