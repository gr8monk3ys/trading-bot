# Candidate `swing` (SwingRSIStrategy), dev window (2008-01-01 .. 2019-12-31)

**Verdict: FAIL** (pass bar in `scripts/run_candidate_gate.py`, ADR 0011)

- [x] beats_spy_same_exposure
- [x] at_least_100_round_trips
- [ ] beats_in_both_halves

| | Strategy | SPY total return at same exposure (11%) |
|---|---|---|
| Total return | +17.72% | +14.83% |
| Annual return | +1.37% | +1.16% |
| Sharpe (rf=0) | 0.25 | 0.54 |
| Max drawdown | -15.18% | -6.94% |

SPY buy-and-hold, 100% invested, dividends reinvested: +181.71%.
Alpha +0.32%/yr (t = 0.20), beta 0.12.
Closed round trips: 367 (742 fills incl. warm-up and final liquidation).
Dividends credited: $2,401.
Costs: flat 10 bps per fill; charged on average 10.0 bps per fill.
Data: `yfinance`, commit `c60a325`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
| 2008-01-02 .. 2013-12-30 | +16.24% | +5.92% | yes |
| 2013-12-31 .. 2019-12-31 | +1.28% | +8.41% | no |
