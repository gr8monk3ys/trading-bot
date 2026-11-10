# Candidate `volmanaged` (VolManagedStrategy), dev window (2008-01-01 .. 2019-12-31)

**Verdict: FAIL** (pass bar in `scripts/run_candidate_gate.py`, ADR 0011)

- [ ] beats_spy_same_exposure
- [ ] at_least_100_round_trips
- [ ] beats_in_both_halves

| | Strategy | SPY total return at same exposure (79%) |
|---|---|---|
| Total return | +100.42% | +135.02% |
| Annual return | +5.97% | +7.39% |
| Sharpe (rf=0) | 0.50 | 0.54 |
| Max drawdown | -26.72% | -42.88% |

SPY buy-and-hold, 100% invested, dividends reinvested: +181.71%.
Alpha +0.61%/yr (t = 0.30), beta 0.58.
Closed round trips: 71 (172 fills incl. warm-up and final liquidation).
Dividends credited: $27,519.
Costs: 40 bps slippage + 10 bps spread per fill.
Data: `yfinance`, commit `ec02e3c`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
| 2008-01-02 .. 2013-12-30 | +36.00% | +36.55% | no |
| 2013-12-31 .. 2019-12-31 | +47.37% | +72.11% | no |
