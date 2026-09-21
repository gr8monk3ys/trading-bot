---
status: proposed
---
# A failed bar fetch fails the run; it is never an empty backtest

Bar loading is copied into two scripts and the engine, and three stacked catch-all points let a network outage publish as an honest zero-trade backtest with exit code 0. We decided historical bars are one module with Alpaca and yfinance adapters that returns a data outcome per symbol, and the verdict consumes those outcomes: any failed symbol makes the run `DATA_UNAVAILABLE`, writes no results artifact, and exits non-zero. Partial success is not a run. `main.py backtest` follows the same rule.
