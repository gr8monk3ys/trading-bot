# ETF baseline 2020-2024 — survivorship-bias-free test of strategy edge

Generated: 2026-05-11T21:48:21.139765Z
Spec: `docs/superpowers/specs/2026-05-11-honest-cleanup-design.md`
Data source: `yfinance`

> **Status: INCONCLUSIVE.** Strategy produced 38 trades, below the 50-trade significance bar set by this repo's `PROFITABILITY_RESEARCH.md`. The numbers below are reported for transparency but must not be cited as evidence of strategy edge.

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

- **Total return:** 53.42%
- **Annualized return:** 8.94%
- **Sharpe ratio:** 0.78
- **Sortino ratio:** 1.49
- **Calmar ratio:** 1.27
- **Max drawdown:** 7.05%
- **Win rate:** 26.32%
- **Profit factor:** 5.48
- **Trade count:** 38
- **Final equity:** $153,415.97

## Trade log

| # | Symbol | Side | Quantity | Price | P&L | Timestamp |
|---|--------|------|----------|-------|-----|-----------|
| 1 | EFA | buy | 163 | 61.31 | 0.00 | 2020-06-12 00:00:00 |
| 2 | SPY | buy | 26 | 342.58 | 0.00 | 2020-09-04 00:00:00 |
| 3 | QQQ | buy | 28 | 283.59 | 0.00 | 2020-09-04 00:00:00 |
| 4 | EFA | sell | 113 | 64.51 | 362.40 | 2020-11-04 00:00:00 |
| 5 | EFA | sell | 121 | 66.06 | 237.84 | 2020-11-05 00:00:00 |
| 6 | EFA | buy | 123 | 71.81 | -407.64 | 2020-12-01 00:00:00 |
| 7 | EFA | buy | 101 | 78.53 | 0.00 | 2021-04-21 00:00:00 |
| 8 | SPY | buy | 17 | 416.74 | 0.00 | 2021-04-23 00:00:00 |
| 9 | QQQ | buy | 16 | 390.60 | 0.00 | 2021-11-11 00:00:00 |
| 10 | EFA | sell | 76 | 76.65 | 30.27 | 2021-12-17 00:00:00 |
| 11 | SPY | sell | 14 | 452.94 | 1134.65 | 2022-02-01 00:00:00 |
| 12 | EFA | sell | 91 | 77.11 | 66.19 | 2022-02-02 00:00:00 |
| 13 | QQQ | buy | 21 | 354.11 | 0.00 | 2022-04-07 00:00:00 |
| 14 | SPY | sell | 16 | 413.80 | 670.45 | 2022-05-05 00:00:00 |
| 15 | QQQ | sell | 24 | 312.99 | -473.40 | 2022-05-05 00:00:00 |
| 16 | IWM | sell | 45 | 185.66 | 0.00 | 2022-05-05 00:00:00 |
| 17 | EFA | sell | 136 | 67.93 | 0.00 | 2022-05-05 00:00:00 |
| 18 | IWM | sell | 57 | 175.72 | 0.00 | 2022-06-27 00:00:00 |
| 19 | IWM | buy | 58 | 190.65 | -611.58 | 2022-08-22 00:00:00 |
| 20 | QQQ | buy | 26 | 375.64 | 0.00 | 2023-07-21 00:00:00 |
| 21 | SPY | sell | 21 | 432.28 | 785.02 | 2023-10-09 00:00:00 |
| 22 | EFA | buy | 137 | 72.57 | -519.57 | 2023-12-06 00:00:00 |
| 23 | SPY | buy | 18 | 489.20 | -455.36 | 2024-02-01 00:00:00 |
| 24 | QQQ | buy | 19 | 421.89 | 0.00 | 2024-02-01 00:00:00 |
| 25 | SPY | buy | 14 | 498.57 | 0.00 | 2024-02-14 00:00:00 |
| 26 | QQQ | buy | 13 | 479.39 | 0.00 | 2024-06-25 00:00:00 |
| 27 | SPY | buy | 10 | 552.66 | 0.00 | 2024-07-18 00:00:00 |
| 28 | QQQ | sell | 11 | 473.21 | 1021.33 | 2024-09-12 00:00:00 |
| 29 | SPY | buy | 10 | 568.87 | 0.00 | 2024-10-02 00:00:00 |
| 30 | EFA | sell | 68 | 79.15 | 0.00 | 2024-11-08 00:00:00 |
| 31 | SPY | buy | 10 | 588.16 | 0.00 | 2024-11-18 00:00:00 |
| 32 | QQQ | buy | 10 | 500.03 | 0.00 | 2024-11-18 00:00:00 |
| 33 | IWM | buy | 21 | 228.94 | -1025.49 | 2024-11-18 00:00:00 |
| 34 | QQQ | buy | 8 | 514.18 | 0.00 | 2024-12-19 00:00:00 |
| 35 | QQQ | sell | 106 | 515.41 | 12047.57 | 2024-12-30 00:00:00 |
| 36 | IWM | sell | 79 | 220.59 | 0.00 | 2024-12-30 00:00:00 |
| 37 | EFA | sell | 69 | 75.58 | 0.00 | 2024-12-30 00:00:00 |
| 38 | SPY | sell | 62 | 588.09 | 2787.64 | 2024-12-30 00:00:00 |

## Comparison: ETF baseline vs hand-picked vs buy-and-hold

| Run | Universe | Total return | Sharpe | Max DD | Trades |
|-----|----------|--------------|--------|--------|--------|
| **ETF baseline (this run)** | SPY, QQQ, IWM, EFA | 53.42% | 0.78 | 7.05% | 38 |
| Hand-picked baseline (survivor-biased) | 10 hand-picked mega-caps (SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM) | 646.00% | 1.36 | 46.96% | 102 |
| SPY buy-and-hold | SPY | 95.30% | 0.75 | 33.72% | 1 |
| QQQ buy-and-hold | QQQ | 145.95% | 0.83 | 35.12% | 1 |

Buy-and-hold numbers are computed in this script via yfinance for the
same period and capital, using daily close-to-close returns and rf=0
for the Sharpe (matching the strategy convention). The hand-picked row
is copied from `results/honest_backtest_2020-2024.md`.

## Interpretation

**Trade count (38) is below the 50-trade significance bar.**
  The directional comparison below is reported because that is the whole
  point of this script — but treat it as a hint, not as evidence. Sharpe
  confidence intervals at 38 trades are very wide; the strategy could be
  underperforming SPY by chance alone.

**Directional finding: the strategy underperformed SPY buy-and-hold on
  a bias-free universe.** This is the most damning bucket the script
  can land in. The +646% on the hand-picked baseline is consistent
  with riding survivors, not with possessing timing edge. Treat the
  hand-picked Sharpe as a number to be explained away, not a number
  to deploy capital on.

**Caveats — read before quoting these numbers:**

- ETFs are not the *only* survivor-bias-free universe. A random sample
  of S&P 500 members at each point in time would be stronger; this run
  is a cheap-to-produce first cut. Follow-up item in `TODO.md`.

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
