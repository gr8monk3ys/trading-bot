# Honest baseline backtest 2020-2024

Generated: 2026-05-11T18:51:25.529321Z
Spec: `docs/superpowers/specs/2026-05-11-honest-cleanup-design.md`
Data source: `yfinance`

> **Status: backtest produced 93 trades** (meets the 50-trade significance bar).

## Configuration

- **Strategy:** `MomentumStrategyBacktest` (daily-bar variant of MomentumStrategy, default parameters)
- **Symbols:** SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM
- **Period:** 2020-01-01 to 2024-12-31
- **Initial capital:** $100,000
- **Slippage:** 40 bps per trade
- **Spread:** 10 bps
- **Significance bar:** 50 trades

## Headline metrics

- **Total return:** 646.64%
- **Annualized return:** 49.50%
- **Sharpe ratio:** 1.36
- **Sortino ratio:** 2.08
- **Calmar ratio:** 1.05
- **Max drawdown:** 46.96%
- **Win rate:** 20.43%
- **Profit factor:** 0.42
- **Trade count:** 93
- **Final equity:** $746,644.64

## Trade log

| # | Symbol | Side | Quantity | Price | P&L | Timestamp |
|---|--------|------|----------|-------|-----|-----------|
| 1 | AAPL | sell | 165 | 60.54 | 0.00 | 2020-03-16 00:00:00 |
| 2 | MSFT | sell | 81 | 135.40 | 0.00 | 2020-03-16 00:00:00 |
| 3 | AMZN | sell | 128 | 94.04 | 0.00 | 2020-03-19 00:00:00 |
| 4 | AMZN | sell | 144 | 92.29 | 0.00 | 2020-03-20 00:00:00 |
| 5 | AMZN | sell | 153 | 95.13 | 0.00 | 2020-03-23 00:00:00 |
| 6 | NVDA | sell | 2619 | 6.14 | 0.00 | 2020-03-25 00:00:00 |
| 7 | TSLA | buy | 348 | 50.76 | 0.00 | 2020-05-04 00:00:00 |
| 8 | NVDA | buy | 1876 | 8.49 | -4409.57 | 2020-05-28 00:00:00 |
| 9 | MSFT | buy | 76 | 187.75 | -3979.26 | 2020-06-12 00:00:00 |
| 10 | JPM | buy | 129 | 99.89 | 0.00 | 2020-06-12 00:00:00 |
| 11 | MSFT | buy | 61 | 188.95 | -267.79 | 2020-06-15 00:00:00 |
| 12 | AAPL | buy | 115 | 90.45 | -3439.08 | 2020-06-29 00:00:00 |
| 13 | MSFT | buy | 47 | 198.45 | 0.00 | 2020-06-29 00:00:00 |
| 14 | TSLA | buy | 89 | 94.47 | 0.00 | 2020-07-24 00:00:00 |
| 15 | SPY | buy | 22 | 342.58 | 0.00 | 2020-09-04 00:00:00 |
| 16 | QQQ | buy | 24 | 283.59 | 0.00 | 2020-09-04 00:00:00 |
| 17 | MSFT | buy | 29 | 214.26 | 0.00 | 2020-09-04 00:00:00 |
| 18 | AMZN | buy | 33 | 164.74 | -2339.62 | 2020-09-04 00:00:00 |
| 19 | NVDA | buy | 400 | 12.62 | -2594.37 | 2020-09-04 00:00:00 |
| 20 | GOOGL | buy | 59 | 76.19 | 0.00 | 2020-09-08 00:00:00 |
| 21 | JPM | buy | 40 | 100.88 | 0.00 | 2020-09-09 00:00:00 |
| 22 | AAPL | buy | 28 | 131.01 | -1973.13 | 2021-01-05 00:00:00 |
| 23 | META | sell | 12 | 267.47 | 0.00 | 2021-01-20 00:00:00 |
| 24 | META | sell | 13 | 272.86 | 0.00 | 2021-01-21 00:00:00 |
| 25 | GOOGL | buy | 43 | 92.67 | 0.00 | 2021-01-28 00:00:00 |
| 26 | AAPL | buy | 27 | 131.96 | -1571.23 | 2021-01-29 00:00:00 |
| 27 | AAPL | buy | 24 | 134.14 | 0.00 | 2021-02-01 00:00:00 |
| 28 | JPM | buy | 19 | 150.51 | 0.00 | 2021-03-01 00:00:00 |
| 29 | TSLA | sell | 11 | 222.67 | 1793.16 | 2021-03-10 00:00:00 |
| 30 | TSLA | sell | 12 | 233.19 | 2082.33 | 2021-03-11 00:00:00 |
| 31 | SPY | buy | 7 | 416.74 | 0.00 | 2021-04-23 00:00:00 |
| 32 | AAPL | buy | 21 | 133.48 | 0.00 | 2021-04-29 00:00:00 |
| 33 | AMZN | buy | 15 | 165.60 | -1076.35 | 2021-05-04 00:00:00 |
| 34 | AMZN | buy | 13 | 172.20 | -1018.65 | 2021-06-28 00:00:00 |
| 35 | AAPL | buy | 14 | 146.15 | 0.00 | 2021-07-20 00:00:00 |
| 36 | META | buy | 5 | 356.31 | -430.21 | 2021-07-30 00:00:00 |
| 37 | TSLA | buy | 4 | 355.99 | 0.00 | 2021-11-10 00:00:00 |
| 38 | QQQ | buy | 4 | 390.59 | 0.00 | 2021-11-11 00:00:00 |
| 39 | AAPL | buy | 8 | 171.14 | 0.00 | 2021-12-17 00:00:00 |
| 40 | SPY | sell | 2 | 452.95 | 184.94 | 2022-02-01 00:00:00 |
| 41 | MSFT | sell | 4 | 297.30 | 397.63 | 2022-02-25 00:00:00 |
| 42 | TSLA | sell | 5 | 288.11 | 1128.09 | 2022-03-01 00:00:00 |
| 43 | AMZN | sell | 11 | 145.52 | 0.00 | 2022-03-11 00:00:00 |
| 44 | QQQ | buy | 5 | 354.11 | 0.00 | 2022-04-07 00:00:00 |
| 45 | MSFT | buy | 5 | 301.38 | 0.00 | 2022-04-07 00:00:00 |
| 46 | MSFT | sell | 5 | 289.62 | 439.18 | 2022-04-28 00:00:00 |
| 47 | SPY | sell | 3 | 413.81 | 159.98 | 2022-05-05 00:00:00 |
| 48 | QQQ | sell | 5 | 312.99 | 28.75 | 2022-05-05 00:00:00 |
| 49 | GOOGL | sell | 16 | 116.50 | 533.86 | 2022-05-05 00:00:00 |
| 50 | META | sell | 12 | 169.48 | 0.00 | 2022-06-27 00:00:00 |
| 51 | AAPL | sell | 14 | 153.72 | 188.59 | 2022-09-21 00:00:00 |
| 52 | NVDA | sell | 200 | 12.56 | 0.00 | 2022-09-22 00:00:00 |
| 53 | MSFT | sell | 12 | 228.55 | 321.18 | 2022-10-14 00:00:00 |
| 54 | META | sell | 27 | 111.86 | 0.00 | 2022-11-10 00:00:00 |
| 55 | GOOGL | sell | 34 | 96.40 | 451.08 | 2022-11-11 00:00:00 |
| 56 | AMZN | sell | 36 | 100.78 | 0.00 | 2022-11-11 00:00:00 |
| 57 | TSLA | sell | 22 | 182.85 | 2647.79 | 2022-11-25 00:00:00 |
| 58 | AMZN | buy | 44 | 100.06 | -185.95 | 2023-02-08 00:00:00 |
| 59 | GOOGL | buy | 42 | 95.02 | 0.00 | 2023-02-09 00:00:00 |
| 60 | AMZN | buy | 34 | 104.98 | -311.24 | 2023-04-26 00:00:00 |
| 61 | AMZN | buy | 26 | 124.25 | -739.02 | 2023-06-08 00:00:00 |
| 62 | QQQ | buy | 7 | 375.63 | 0.00 | 2023-07-21 00:00:00 |
| 63 | MSFT | buy | 7 | 345.12 | 0.00 | 2023-07-24 00:00:00 |
| 64 | NVDA | buy | 51 | 46.84 | -1954.99 | 2023-08-28 00:00:00 |
| 65 | SPY | sell | 5 | 432.29 | 359.05 | 2023-10-09 00:00:00 |
| 66 | AMZN | buy | 16 | 146.72 | -814.16 | 2023-11-22 00:00:00 |
| 67 | NVDA | buy | 45 | 47.78 | -1767.31 | 2023-11-24 00:00:00 |
| 68 | AAPL | sell | 10 | 191.56 | 513.11 | 2024-01-19 00:00:00 |
| 69 | SPY | buy | 4 | 489.20 | 0.00 | 2024-02-01 00:00:00 |
| 70 | QQQ | buy | 4 | 421.88 | 0.00 | 2024-02-01 00:00:00 |
| 71 | MSFT | buy | 4 | 403.78 | 0.00 | 2024-02-01 00:00:00 |
| 72 | SPY | buy | 3 | 498.57 | 0.00 | 2024-02-14 00:00:00 |
| 73 | AAPL | sell | 8 | 173.72 | 267.78 | 2024-03-18 00:00:00 |
| 74 | AMZN | buy | 8 | 183.32 | -699.94 | 2024-04-16 00:00:00 |
| 75 | TSLA | sell | 8 | 170.17 | 861.43 | 2024-04-25 00:00:00 |
| 76 | JPM | buy | 7 | 199.53 | 0.00 | 2024-05-21 00:00:00 |
| 77 | AAPL | buy | 6 | 208.14 | 0.00 | 2024-06-24 00:00:00 |
| 78 | NVDA | buy | 11 | 118.11 | -1205.69 | 2024-06-24 00:00:00 |
| 79 | QQQ | buy | 2 | 479.38 | 0.00 | 2024-06-25 00:00:00 |
| 80 | MSFT | buy | 2 | 456.73 | 0.00 | 2024-07-01 00:00:00 |
| 81 | SPY | buy | 1 | 552.66 | 0.00 | 2024-07-18 00:00:00 |
| 82 | META | sell | 1 | 497.73 | 0.00 | 2024-08-01 00:00:00 |
| 83 | META | sell | 2 | 488.13 | 0.00 | 2024-08-02 00:00:00 |
| 84 | JPM | buy | 5 | 212.46 | 0.00 | 2024-09-06 00:00:00 |
| 85 | QQQ | sell | 2 | 473.22 | 269.43 | 2024-09-12 00:00:00 |
| 86 | GOOGL | sell | 6 | 158.06 | 417.67 | 2024-09-16 00:00:00 |
| 87 | SPY | buy | 2 | 568.86 | 0.00 | 2024-10-02 00:00:00 |
| 88 | GOOGL | buy | 6 | 162.08 | 0.00 | 2024-10-10 00:00:00 |
| 89 | SPY | buy | 1 | 588.15 | 0.00 | 2024-11-18 00:00:00 |
| 90 | QQQ | buy | 1 | 500.02 | 0.00 | 2024-11-18 00:00:00 |
| 91 | AMZN | buy | 4 | 201.70 | -423.49 | 2024-11-18 00:00:00 |
| 92 | QQQ | buy | 1 | 514.17 | 0.00 | 2024-12-19 00:00:00 |
| 93 | TSLA | buy | 1 | 430.60 | 0.00 | 2024-12-23 00:00:00 |

## Interpretation

This is the single performance number cited by `README.md` and `CLAUDE.md`. It supersedes `backtest_report_2024.md` (9 trades) and any earlier in-doc claims (notably the `+42.68%` figure that lacked a publishable evidence file).

**Caveats — read before quoting these numbers:**

1. **Survivorship-bias correction is off.** The 10-symbol universe is hand-picked mega-caps that survived 2020-2024; survivorship-bias handling was quarantined to `research/` in the 2026-05 cleanup. Numbers above are inflated by selection of known winners.
2. **PnL accounting under-reports per-trade P&L.** The engine's trade-matching logic only assigns PnL to sells that follow buys in the same symbol; trades the strategy opens as shorts get `pnl: 0` and their gains/losses show up in equity but not in `avg_trade` / `profit_factor`. Treat trade-level stats as lower bounds.
3. **Mark-to-market dominates the headline return.** A large fraction of the final equity sits in still-open positions on 2024-12-31, valued at end-of-period prices. The 5-year window happens to end near all-time highs; rerun ending on a different date for a different number.
4. **Costs included: 40 bps slippage + 10 bps spread per trade.** These are realistic for retail at this universe size but do not model gap risk on positions held overnight (gap stats: see engine logs — largest gap in this run was 26%).
5. **No walk-forward validation in this artifact.** This is a single in-sample run; treat the Sharpe as an upper bound on what an out-of-sample trader would have realized. `PROFITABILITY_RESEARCH.md` documents realistic expectations for this strategy family (Sharpe 0.5 to 1.2 net of costs) — anything well above that range warrants suspicion, not celebration.

Do not extrapolate beyond what the trade count supports. Use this artifact as a sanity check that the pipeline runs end-to-end on real market data, not as evidence of strategy edge.
