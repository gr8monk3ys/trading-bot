# Bollinger filter A/B — ETF baseline 2020-2024


Universe SPY, QQQ, IWM, EFA · 2020-01-01 to 2024-12-31 · target gross 100% · data source `alpaca`.
Only `use_bollinger_filter` differs between the two rows.


| Bollinger filter | Trades | Total return | Sharpe | Max DD | Avg gross |
|---|---|---|---|---|---|
| ON | 26 | 16.44% | 0.16 | 12.90% | 59.06% |
| OFF | 66 | -1.88% | -0.11 | 21.50% | 82.03% |
| _SPY buy-and-hold_ | — | 95.30% | 0.75 | -33.72% | 100% |

## What this settles

The filter was suspected of being inverted — penalising the breakouts a
momentum strategy is supposed to buy. **The opposite is true, and it matters
more than the filter itself.**


Turning the filter off roughly triples trade count (26 to 66), pushes average gross exposure from 59.06% to 82.03%, and takes
total return from 16.44% to -1.88% — from thin positive to outright negative,
with drawdown nearly doubling (12.90% to 21.50%).


So the mean-reversion overlay is not fighting the momentum signal; it is
**carrying** it. What the filter mostly does is suppress trades, and
suppressing this strategy's trades is what keeps its return above zero.


**The filter-off run is also the first configuration in this repo to clear the 50-trade significance bar** (66 trades). Its verdict is therefore not INCONCLUSIVE: the
unfiltered momentum signal loses money over five years while SPY buy-and-hold
returns 95.30%. Every earlier 'no edge' finding was directionally
right but statistically under-powered; this one is not.


Note the two rows are not exposure-matched — the filter-off run carries *more*
exposure and still returns less, so no exposure adjustment would flip the sign.


Reproduce: `python scripts/run_bollinger_ab.py` (deterministic; verified
identical across two consecutive runs).
