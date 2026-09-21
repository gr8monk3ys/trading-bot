# ETF baseline 2020-2024 — survivorship-bias-free test of strategy edge

Generated: 2026-09-21T06:54:55.131669Z
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

- **Total return:** 4.72%
- **Annualized return:** 0.93%
- **Sharpe ratio:** -0.43
- **Sortino ratio:** -0.45
- **Calmar ratio:** 0.42
- **Max drawdown:** 2.20%
- **Win rate:** 38.00%
- **Profit factor:** 1.78
- **Trade count:** 50
- **Final equity:** $104,721.87

## Trade log

| # | Symbol | Side | Quantity | Price | P&L | Timestamp |
|---|--------|------|----------|-------|-----|-----------|
| 1 | EFA | buy | 203.0 | 61.35 | 0.00 | 2020-06-12 00:00:00 |
| 2 | SPY | buy | 37.0 | 342.67 | 0.00 | 2020-09-04 00:00:00 |
| 3 | QQQ | buy | 49.0 | 283.69 | 0.00 | 2020-09-04 00:00:00 |
| 4 | EFA | sell | 203.0 | 64.49 | 637.29 | 2020-09-04 00:00:00 |
| 5 | EFA | sell | 234.0 | 64.48 | 0.00 | 2020-11-04 00:00:00 |
| 6 | EFA | buy | 234.0 | 71.84 | -1722.36 | 2020-12-01 00:00:00 |
| 7 | QQQ | sell | 49.0 | 302.42 | 917.78 | 2020-12-10 00:00:00 |
| 8 | SPY | sell | 37.0 | 377.54 | 1290.32 | 2021-01-28 00:00:00 |
| 9 | EFA | buy | 160.0 | 78.56 | 0.00 | 2021-04-21 00:00:00 |
| 10 | SPY | buy | 30.0 | 416.83 | 0.00 | 2021-04-23 00:00:00 |
| 11 | EFA | sell | 160.0 | 78.49 | -10.80 | 2021-06-18 00:00:00 |
| 12 | SPY | sell | 30.0 | 433.95 | 513.70 | 2021-09-20 00:00:00 |
| 13 | QQQ | buy | 32.0 | 390.68 | 0.00 | 2021-11-11 00:00:00 |
| 14 | EFA | sell | 172.0 | 76.61 | 0.00 | 2021-12-17 00:00:00 |
| 15 | SPY | sell | 29.0 | 452.83 | 0.00 | 2022-02-01 00:00:00 |
| 16 | SPY | buy | 29.0 | 436.75 | 466.25 | 2022-02-28 00:00:00 |
| 17 | EFA | buy | 172.0 | 73.25 | 577.88 | 2022-02-28 00:00:00 |
| 18 | SPY | sell | 32.0 | 413.68 | 0.00 | 2022-05-05 00:00:00 |
| 19 | QQQ | sell | 32.0 | 312.86 | -2490.36 | 2022-05-05 00:00:00 |
| 20 | IWM | sell | 63.0 | 185.59 | 0.00 | 2022-05-05 00:00:00 |
| 21 | EFA | sell | 169.0 | 67.90 | 0.00 | 2022-05-05 00:00:00 |
| 22 | SPY | buy | 32.0 | 400.26 | 429.72 | 2022-05-16 00:00:00 |
| 23 | IWM | buy | 63.0 | 177.32 | 520.68 | 2022-05-16 00:00:00 |
| 24 | IWM | sell | 71.0 | 175.63 | 0.00 | 2022-06-27 00:00:00 |
| 25 | EFA | buy | 169.0 | 63.54 | 736.63 | 2022-06-27 00:00:00 |
| 26 | IWM | buy | 71.0 | 190.72 | -1071.95 | 2022-08-22 00:00:00 |
| 27 | QQQ | buy | 33.0 | 375.74 | 0.00 | 2023-07-21 00:00:00 |
| 28 | SPY | sell | 30.0 | 432.20 | 0.00 | 2023-10-09 00:00:00 |
| 29 | SPY | buy | 30.0 | 430.85 | 40.31 | 2023-11-02 00:00:00 |
| 30 | EFA | buy | 181.0 | 72.60 | 0.00 | 2023-12-06 00:00:00 |
| 31 | QQQ | sell | 33.0 | 398.24 | 742.35 | 2024-01-03 00:00:00 |
| 32 | SPY | buy | 26.0 | 489.30 | 0.00 | 2024-02-01 00:00:00 |
| 33 | QQQ | buy | 31.0 | 422.00 | 0.00 | 2024-02-01 00:00:00 |
| 34 | QQQ | sell | 31.0 | 438.66 | 516.40 | 2024-03-06 00:00:00 |
| 35 | EFA | sell | 181.0 | 78.48 | 1064.52 | 2024-04-11 00:00:00 |
| 36 | SPY | sell | 26.0 | 504.36 | 391.57 | 2024-04-15 00:00:00 |
| 37 | QQQ | buy | 27.0 | 479.50 | 0.00 | 2024-06-25 00:00:00 |
| 38 | QQQ | sell | 27.0 | 494.70 | 410.40 | 2024-07-12 00:00:00 |
| 39 | SPY | buy | 23.0 | 552.76 | 0.00 | 2024-07-18 00:00:00 |
| 40 | QQQ | sell | 27.0 | 473.06 | 0.00 | 2024-09-12 00:00:00 |
| 41 | SPY | sell | 23.0 | 570.93 | 418.10 | 2024-11-01 00:00:00 |
| 42 | EFA | sell | 163.0 | 79.11 | 0.00 | 2024-11-08 00:00:00 |
| 43 | SPY | buy | 22.0 | 588.26 | 0.00 | 2024-11-18 00:00:00 |
| 44 | QQQ | buy | 27.0 | 500.18 | -732.15 | 2024-11-18 00:00:00 |
| 45 | IWM | buy | 56.0 | 229.03 | 0.00 | 2024-11-18 00:00:00 |
| 46 | IWM | sell | 56.0 | 237.69 | 484.57 | 2024-12-11 00:00:00 |
| 47 | QQQ | buy | 25.0 | 514.31 | 0.00 | 2024-12-19 00:00:00 |
| 48 | EFA | buy | 163.0 | 75.64 | 565.58 | 2024-12-30 00:00:00 |
| 49 | SPY | sell | 22.0 | 588.11 | -3.45 | 2024-12-30 00:00:00 |
| 50 | QQQ | sell | 25.0 | 515.47 | 28.89 | 2024-12-30 00:00:00 |

## Comparison: ETF baseline vs hand-picked vs buy-and-hold

| Run | Universe | Total return | Sharpe | Max DD | Trades |
|-----|----------|--------------|--------|--------|--------|
| **ETF baseline (this run)** | SPY, QQQ, IWM, EFA | 4.72% | -0.43 | 2.20% | 50 |
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
