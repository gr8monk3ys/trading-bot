# ETF baseline 2020-2024 — survivorship-bias-free test of strategy edge

Generated: 2026-09-21T06:54:57.769068Z
Spec: `docs/superpowers/specs/2026-05-11-honest-cleanup-design.md`
Data source: `yfinance`

> **Status: backtest produced 50 trades** (meets the 50-trade significance bar).

## Purpose

This backtest exists to disambiguate **"the strategy has edge"** from
**"the universe was hand-picked winners"**. The existing
`results/honest_backtest_2020-2024.md` posts +646% / Sharpe 1.36 on
10 mega-caps that any 2026 retrospective would obviously pick. That
number is dominated by survivorship bias.

ETFs cannot be delisted and cannot be selection-biased. SPY/QQQ/IWM/EFA
cover US large-cap, US tech, US small-cap, and developed international
equity — broad market exposure with zero look-ahead. If the strategy
can't beat SPY buy-and-hold on this universe, it has no real edge.

## Configuration

- **Strategy:** `MomentumStrategyBacktest` (daily-bar variant of MomentumStrategy, default parameters)
- **Symbols:** SPY, QQQ, IWM, EFA (US large-cap, US tech, US small-cap, developed-intl)
- **Period:** 2020-01-01 to 2024-12-31
- **Initial capital:** $100,000
- **Slippage:** 40 bps per trade
- **Spread:** 10 bps
- **Significance bar:** 50 trades

## Headline metrics

- **Total return:** 9.09%
- **Annualized return:** 1.76%
- **Sharpe ratio:** -0.02
- **Sortino ratio:** -0.02
- **Calmar ratio:** 0.38
- **Max drawdown:** 4.59%
- **Win rate:** 38.00%
- **Profit factor:** 1.70
- **Trade count:** 50
- **Final equity:** $109,087.47

## Trade log

| # | Symbol | Side | Quantity | Price | P&L | Timestamp |
|---|--------|------|----------|-------|-----|-----------|
| 1 | EFA | buy | 407.0 | 61.36 | 0.00 | 2020-06-12 00:00:00 |
| 2 | SPY | buy | 77.0 | 342.68 | 0.00 | 2020-09-04 00:00:00 |
| 3 | QQQ | buy | 109.0 | 283.71 | 0.00 | 2020-09-04 00:00:00 |
| 4 | EFA | sell | 407.0 | 64.48 | 1269.63 | 2020-09-04 00:00:00 |
| 5 | EFA | sell | 563.0 | 64.47 | 0.00 | 2020-11-04 00:00:00 |
| 6 | EFA | buy | 563.0 | 71.86 | -4160.07 | 2020-12-01 00:00:00 |
| 7 | QQQ | sell | 109.0 | 302.40 | 2036.93 | 2020-12-10 00:00:00 |
| 8 | SPY | sell | 77.0 | 377.53 | 2683.51 | 2021-01-28 00:00:00 |
| 9 | EFA | buy | 324.0 | 78.57 | 0.00 | 2021-04-21 00:00:00 |
| 10 | SPY | buy | 60.0 | 416.83 | 0.00 | 2021-04-23 00:00:00 |
| 11 | EFA | sell | 324.0 | 78.49 | -26.14 | 2021-06-18 00:00:00 |
| 12 | SPY | sell | 60.0 | 433.94 | 1026.57 | 2021-09-20 00:00:00 |
| 13 | QQQ | buy | 65.0 | 390.69 | 0.00 | 2021-11-11 00:00:00 |
| 14 | EFA | sell | 361.0 | 76.60 | 0.00 | 2021-12-17 00:00:00 |
| 15 | SPY | sell | 61.0 | 452.82 | 0.00 | 2022-02-01 00:00:00 |
| 16 | SPY | buy | 61.0 | 436.77 | 979.19 | 2022-02-28 00:00:00 |
| 17 | EFA | buy | 361.0 | 73.26 | 1206.59 | 2022-02-28 00:00:00 |
| 18 | SPY | sell | 68.0 | 413.67 | 0.00 | 2022-05-05 00:00:00 |
| 19 | QQQ | sell | 65.0 | 312.84 | -5060.46 | 2022-05-05 00:00:00 |
| 20 | IWM | sell | 118.0 | 185.57 | 0.00 | 2022-05-05 00:00:00 |
| 21 | EFA | sell | 308.0 | 67.89 | 0.00 | 2022-05-05 00:00:00 |
| 22 | SPY | buy | 68.0 | 400.27 | 910.70 | 2022-05-16 00:00:00 |
| 23 | IWM | buy | 118.0 | 177.34 | 971.31 | 2022-05-16 00:00:00 |
| 24 | IWM | sell | 141.0 | 175.60 | 0.00 | 2022-06-27 00:00:00 |
| 25 | EFA | buy | 308.0 | 63.55 | 1337.80 | 2022-06-27 00:00:00 |
| 26 | IWM | buy | 141.0 | 190.74 | -2134.71 | 2022-08-22 00:00:00 |
| 27 | QQQ | buy | 67.0 | 375.75 | 0.00 | 2023-07-21 00:00:00 |
| 28 | SPY | sell | 63.0 | 432.19 | 0.00 | 2023-10-09 00:00:00 |
| 29 | SPY | buy | 63.0 | 430.86 | 83.47 | 2023-11-02 00:00:00 |
| 30 | EFA | buy | 380.0 | 72.61 | 0.00 | 2023-12-06 00:00:00 |
| 31 | QQQ | sell | 67.0 | 398.22 | 1505.52 | 2024-01-03 00:00:00 |
| 32 | SPY | buy | 53.0 | 489.31 | 0.00 | 2024-02-01 00:00:00 |
| 33 | QQQ | buy | 64.0 | 422.01 | 0.00 | 2024-02-01 00:00:00 |
| 34 | QQQ | sell | 64.0 | 438.64 | 1064.12 | 2024-03-06 00:00:00 |
| 35 | EFA | sell | 380.0 | 78.48 | 2229.22 | 2024-04-11 00:00:00 |
| 36 | SPY | sell | 53.0 | 504.35 | 797.31 | 2024-04-15 00:00:00 |
| 37 | QQQ | buy | 55.0 | 479.51 | 0.00 | 2024-06-25 00:00:00 |
| 38 | QQQ | sell | 55.0 | 494.68 | 834.19 | 2024-07-12 00:00:00 |
| 39 | SPY | buy | 48.0 | 552.77 | 0.00 | 2024-07-18 00:00:00 |
| 40 | QQQ | sell | 57.0 | 473.04 | 0.00 | 2024-09-12 00:00:00 |
| 41 | SPY | sell | 48.0 | 570.92 | 871.46 | 2024-11-01 00:00:00 |
| 42 | EFA | sell | 334.0 | 79.10 | 0.00 | 2024-11-08 00:00:00 |
| 43 | SPY | buy | 45.0 | 588.28 | 0.00 | 2024-11-18 00:00:00 |
| 44 | QQQ | buy | 57.0 | 500.21 | -1548.56 | 2024-11-18 00:00:00 |
| 45 | IWM | buy | 117.0 | 229.06 | 0.00 | 2024-11-18 00:00:00 |
| 46 | IWM | sell | 117.0 | 237.67 | 1007.77 | 2024-12-11 00:00:00 |
| 47 | QQQ | buy | 53.0 | 514.34 | 0.00 | 2024-12-19 00:00:00 |
| 48 | EFA | buy | 334.0 | 75.65 | 1151.93 | 2024-12-30 00:00:00 |
| 49 | SPY | sell | 45.0 | 588.09 | -8.39 | 2024-12-30 00:00:00 |
| 50 | QQQ | sell | 53.0 | 515.44 | 58.56 | 2024-12-30 00:00:00 |

## Comparison: ETF baseline vs hand-picked vs buy-and-hold

| Run | Universe | Total return | Sharpe | Max DD | Trades |
|-----|----------|--------------|--------|--------|--------|
| **ETF baseline (this run)** | SPY, QQQ, IWM, EFA | 9.09% | -0.02 | 4.59% | 50 |
| Hand-picked baseline (survivor-biased) | 10 hand-picked mega-caps (SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM) | 646.00% | 1.36 | 46.96% | 102 |
| SPY buy-and-hold | SPY | 95.30% | 0.75 | 33.72% | 1 |
| QQQ buy-and-hold | QQQ | 145.95% | 0.83 | 35.12% | 1 |

Buy-and-hold numbers are computed in this script via yfinance for the
same period and capital, using daily close-to-close returns and rf=0
for the Sharpe (matching the strategy convention). The hand-picked row
is copied from `results/honest_backtest_2020-2024.md`.

## Interpretation

**Directional finding: the strategy underperformed SPY buy-and-hold on
  a bias-free universe.** This is the most damning bucket the script
  can land in. The +646% on the hand-picked baseline is consistent
  with riding survivors, not with possessing timing edge. Treat the
  hand-picked Sharpe as a number to be explained away, not a number
  to deploy capital on.

**Caveats — read before quoting these numbers:**

- ETFs are not the *only* survivor-bias-free universe. A random sample
  of S&P 500 members at each point in time would be stronger; this run
  is a cheap-to-produce first cut. Tracked as a GitHub issue.

- 5 years of daily data on 4 instruments is a small sample even when
  the in-strategy trade count crosses 50. Don't extrapolate Sharpe
  confidence intervals from this run alone.

- Costs included: 40 bps slippage + 10 bps spread per trade. ETFs trade
  tighter than that in practice, so per-trade cost drag is if anything
  overstated here, not understated.

- Realized P&L only — open positions at end-of-period are liquidated at
  the final bar with the same spread + slippage as any other trade
  (`BacktestEngine._liquidate_open_positions`). Headline equity reflects
  realized cash, not unrealized MTM.
