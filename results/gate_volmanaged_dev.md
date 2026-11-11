# Candidate `volmanaged` (VolManagedStrategy), dev window (2008-01-01 .. 2019-12-31)

**Verdict: FAIL** (pass bar in `scripts/run_candidate_gate.py`, ADR 0011)

- [ ] beats_spy_same_exposure
- [ ] at_least_100_round_trips
- [ ] beats_in_both_halves

| | Strategy | SPY total return at same exposure (79%) |
|---|---|---|
| Total return | +89.62% | +135.12% |
| Annual return | +5.48% | +7.39% |
| Sharpe (rf=0) | 0.47 | 0.54 |
| Max drawdown | -27.17% | -42.90% |

SPY buy-and-hold, 100% invested, dividends reinvested: +181.71%.
Alpha +0.14%/yr (t = 0.07), beta 0.58.
Closed round trips: 72 (174 fills incl. warm-up and final liquidation).
Dividends credited: $26,391.
Costs: flat 50 bps per fill; charged on average 50.0 bps per fill.
Data: `yfinance`, commit `c60a325`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
| 2008-01-02 .. 2013-12-30 | +32.07% | +36.57% | no |
| 2013-12-31 .. 2019-12-31 | +43.57% | +72.16% | no |
