# Trend candidate, dev window (2008-01-01 .. 2019-12-31)

**Verdict: FAIL** (pass bar in `scripts/run_trend_gate.py`, ADR 0011)

- [ ] beats_spy_same_exposure
- [ ] at_least_100_round_trips
- [ ] beats_in_both_halves

| | Strategy | SPY at same exposure (78%), price-only |
|---|---|---|
| Total return | +49.04% | +92.13% |
| Annual return | +3.38% | +5.60% |
| Sharpe (rf=0) | 0.39 | 0.43 |
| Max drawdown | -16.13% | -43.44% |

SPY buy-and-hold with dividends, 100% invested: +184.20%.
Alpha +1.63%/yr (t = 0.68), beta 0.25.
Closed round trips: 91 (184 fills incl. warm-up and final liquidation).
Costs: 40 bps slippage + 10 bps spread per fill.
Data: `yfinance`, commit `90d61a7`.

| Half | Strategy | SPY same exposure | Beats |
|---|---|---|---|
| 2008-01-02 .. 2013-12-30 | +31.44% | +23.20% | yes |
| 2013-12-31 .. 2019-12-31 | +13.39% | +55.95% | no |
