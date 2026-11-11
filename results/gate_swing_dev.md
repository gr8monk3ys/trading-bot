# Candidate `swing` (SwingRSIStrategy), dev window (2008-01-01 .. 2019-12-31)

**Verdict: FAIL** (pass bar in `scripts/run_candidate_gate.py`, ADR 0011)

- [ ] beats_spy_same_exposure
- [x] at_least_100_round_trips
- [ ] beats_in_both_halves

| | Strategy | SPY total return at same exposure (11%) |
|---|---|---|
| Total return | -43.43% | +14.84% |
| Annual return | -4.64% | +1.16% |
| Sharpe (rf=0) | -0.72 | 0.54 |
| Max drawdown | -44.99% | -6.94% |

SPY buy-and-hold, 100% invested, dividends reinvested: +181.71%.
Alpha -5.78%/yr (t = -3.41), beta 0.12.
Closed round trips: 367 (742 fills incl. warm-up and final liquidation).
Dividends credited: $1,681.
Costs: flat 50 bps per fill; charged on average 50.0 bps per fill.
Data: `yfinance`, commit `c60a325`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
| 2008-01-02 .. 2013-12-30 | -16.41% | +5.92% | no |
| 2013-12-31 .. 2019-12-31 | -32.32% | +8.42% | no |
