# Candidate `trend` (TrendFollowingStrategy), dev window (2008-01-01 .. 2019-12-31)

**Verdict: FAIL** (pass bar in `scripts/run_candidate_gate.py`, ADR 0011)

- [ ] beats_spy_same_exposure
- [ ] at_least_100_round_trips
- [ ] beats_in_both_halves

| | Strategy | SPY total return at same exposure (68%) |
|---|---|---|
| Total return | +54.40% | +111.69% |
| Annual return | +3.69% | +6.46% |
| Sharpe (rf=0) | 0.48 | 0.54 |
| Max drawdown | -14.13% | -37.67% |

SPY buy-and-hold, 100% invested, dividends reinvested: +181.71%.
Alpha +1.51%/yr (t = 0.75), beta 0.23.
Closed round trips: 96 (194 fills incl. warm-up and final liquidation).
Dividends credited: $20,449.
Costs: flat 50 bps per fill; charged on average 50.0 bps per fill.
Data: `yfinance`, commit `c60a325`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
| 2008-01-02 .. 2013-12-30 | +31.27% | +32.39% | no |
| 2013-12-31 .. 2019-12-31 | +17.62% | +59.90% | no |
