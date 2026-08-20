# ETF baseline 2020-2024 — survivorship-bias-free test of strategy edge

Generated: 2026-08-19T04:55:40.632118Z
Spec: `docs/superpowers/specs/2026-05-11-honest-cleanup-design.md`
Data source: `yfinance`

> **Status: INCONCLUSIVE.** Strategy produced 26 trades, below the 50-trade significance bar set by this repo's `PROFITABILITY_RESEARCH.md`. The numbers below are reported for transparency but must not be cited as evidence of strategy edge.

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

- **Total return:** 16.42%
- **Annualized return:** 3.09%
- **Sharpe ratio:** 0.16
- **Sortino ratio:** 0.20
- **Calmar ratio:** 0.24
- **Max drawdown:** 12.90%
- **Win rate:** 30.77%
- **Profit factor:** 2.40
- **Trade count:** 26
- **Final equity:** $116,420.07

## Trade log

| # | Symbol | Side | Quantity | Price | P&L | Timestamp |
|---|--------|------|----------|-------|-----|-----------|
| 1 | EFA | buy | 407 | 61.31 | 0.00 | 2020-06-12 00:00:00 |
| 2 | SPY | buy | 77 | 342.58 | 0.00 | 2020-09-04 00:00:00 |
| 3 | QQQ | buy | 109 | 283.60 | 0.00 | 2020-09-04 00:00:00 |
| 4 | EFA | sell | 407 | 64.47 | 1286.66 | 2020-11-04 00:00:00 |
| 5 | EFA | sell | 550 | 66.06 | 0.00 | 2020-11-05 00:00:00 |
| 6 | EFA | buy | 550 | 71.86 | -3188.96 | 2020-12-01 00:00:00 |
| 7 | EFA | buy | 453 | 78.54 | 0.00 | 2021-04-21 00:00:00 |
| 8 | EFA | sell | 453 | 76.60 | -878.10 | 2021-12-17 00:00:00 |
| 9 | SPY | sell | 77 | 452.81 | 8487.95 | 2022-02-01 00:00:00 |
| 10 | EFA | sell | 424 | 77.10 | 0.00 | 2022-02-02 00:00:00 |
| 11 | SPY | sell | 79 | 413.79 | 0.00 | 2022-05-05 00:00:00 |
| 12 | QQQ | sell | 109 | 312.82 | 3185.07 | 2022-05-05 00:00:00 |
| 13 | IWM | sell | 128 | 185.65 | 0.00 | 2022-05-05 00:00:00 |
| 14 | IWM | buy | 128 | 190.74 | -651.01 | 2022-08-22 00:00:00 |
| 15 | QQQ | buy | 63 | 375.64 | 0.00 | 2023-07-21 00:00:00 |
| 16 | EFA | buy | 424 | 72.61 | 1904.82 | 2023-12-06 00:00:00 |
| 17 | SPY | buy | 79 | 489.32 | -5966.33 | 2024-02-01 00:00:00 |
| 18 | SPY | buy | 56 | 498.58 | 0.00 | 2024-02-14 00:00:00 |
| 19 | QQQ | sell | 63 | 473.03 | 6135.66 | 2024-09-12 00:00:00 |
| 20 | EFA | sell | 364 | 79.15 | 0.00 | 2024-11-08 00:00:00 |
| 21 | QQQ | buy | 58 | 500.04 | 0.00 | 2024-11-18 00:00:00 |
| 22 | IWM | buy | 128 | 228.95 | 0.00 | 2024-11-18 00:00:00 |
| 23 | SPY | sell | 56 | 588.09 | 5012.48 | 2024-12-30 00:00:00 |
| 24 | EFA | buy | 364 | 75.66 | 1271.38 | 2024-12-30 00:00:00 |
| 25 | QQQ | sell | 58 | 515.44 | 892.76 | 2024-12-30 00:00:00 |
| 26 | IWM | sell | 128 | 220.58 | -1072.31 | 2024-12-30 00:00:00 |

## Comparison: ETF baseline vs hand-picked vs buy-and-hold

| Run | Universe | Total return | Sharpe | Max DD | Trades |
|-----|----------|--------------|--------|--------|--------|
| **ETF baseline (this run)** | SPY, QQQ, IWM, EFA | 16.42% | 0.16 | 12.90% | 26 |
| Hand-picked baseline (survivor-biased) | 10 hand-picked mega-caps (SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM) | 646.00% | 1.36 | 46.96% | 102 |
| SPY buy-and-hold | SPY | 95.30% | 0.75 | 33.72% | 1 |
| QQQ buy-and-hold | QQQ | 145.95% | 0.83 | 35.12% | 1 |

Buy-and-hold numbers are computed in this script via yfinance for the
same period and capital, using daily close-to-close returns and rf=0
for the Sharpe (matching the strategy convention). The hand-picked row
is copied from `results/honest_backtest_2020-2024.md`.

## Interpretation

**Trade count (26) is below the 50-trade significance bar.**
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
