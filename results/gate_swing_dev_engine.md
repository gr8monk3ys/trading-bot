# Candidate `swing` (SwingRSIStrategy), dev window (2008-01-01 .. 2019-12-31)

**Verdict: FAIL** (pass bar in `scripts/run_candidate_gate.py`, ADR 0011)

- [x] beats_spy_same_exposure
- [x] at_least_100_round_trips
- [ ] beats_in_both_halves

| | Strategy | SPY total return at same exposure (11%) |
|---|---|---|
| Total return | +29.87% | +14.83% |
| Annual return | +2.20% | +1.16% |
| Sharpe (rf=0) | 0.39 | 0.54 |
| Max drawdown | -15.07% | -6.94% |

SPY buy-and-hold, 100% invested, dividends reinvested: +181.71%.
Alpha +1.14%/yr (t = 0.69), beta 0.12.
Closed round trips: 367 (742 fills incl. warm-up and final liquidation).
Dividends credited: $2,526.
Costs: broker's dynamic spread + impact model; charged on average 4.6 bps per fill.
Data: `yfinance`, commit `c60a325`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
| 2008-01-02 .. 2013-12-30 | +21.23% | +5.92% | yes |
| 2013-12-31 .. 2019-12-31 | +7.13% | +8.41% | no |
