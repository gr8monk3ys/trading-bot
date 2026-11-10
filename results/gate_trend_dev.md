# Candidate `trend` (TrendFollowingStrategy), dev window (2008-01-01 .. 2019-12-31)

**Verdict: FAIL** (pass bar in `scripts/run_candidate_gate.py`, ADR 0011)

- [ ] beats_spy_same_exposure
- [ ] at_least_100_round_trips
- [ ] beats_in_both_halves

| | Strategy | SPY total return at same exposure (67%) |
|---|---|---|
| Total return | +65.97% | +111.06% |
| Annual return | +4.32% | +6.43% |
| Sharpe (rf=0) | 0.55 | 0.54 |
| Max drawdown | -13.82% | -37.52% |

SPY buy-and-hold, 100% invested, dividends reinvested: +181.71%.
Alpha +2.13%/yr (t = 1.06), beta 0.23.
Closed round trips: 96 (194 fills incl. warm-up and final liquidation).
Dividends credited: $21,100.
Costs: 40 bps slippage + 10 bps spread per fill.
Data: `yfinance`, commit `ec02e3c`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
| 2008-01-02 .. 2013-12-30 | +35.02% | +32.27% | yes |
| 2013-12-31 .. 2019-12-31 | +22.93% | +59.58% | no |
