# trading-bot

<p align="center">
  <img src="docs/assets/hero.png" alt="trading-bot preview" width="640">
</p>

Algorithmic trading research bot on Alpaca (async Python): backtests, strategy validation, and paper trading. No proven edge — not for real capital.

**Status:** experimental, paper-only, no proven edge. Do not deploy real capital.

**Coming back to this repo?** Read [`results/where_we_landed.md`](results/where_we_landed.md) first — the durable summary of the May 2026 cleanup and the August 2026 backtest fixes. The headline numbers are in [Results](#results) below.

## What's in here

- A momentum strategy (RSI/MACD/ADX with trailing stops).
- A mean-reversion strategy.
- An adaptive coordinator that switches between them based on market regime.
- A backtest engine with realistic slippage and spread.
- A risk manager (VaR, correlation, position sizing).
- A circuit breaker (daily-loss halts).
- An Alpaca broker wrapper (paper + live).

Plausible-but-unvalidated quant work — factor models, pairs trading, cross-asset signals, walk-forward validation, alpha-decay monitoring — is in `research/`, excluded from the production path and the default test run.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env  # add ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER=True

pytest tests/
python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-12-31
python main.py live --strategy adaptive
```

## Results

The question this repo set out to answer — "can the momentum strategy be tuned to beat the market?" — is answered: **no.**

| 2020-2024, SPY/QQQ/IWM/EFA universe | Total return | Sharpe | Trades |
|---|---|---|---|
| Momentum strategy, gross-100 target | **+16.4%** | 0.16 | 26 (below the 50-trade significance bar) |
| SPY buy-and-hold | **+95.3%** | 0.75 | — |

Source: `results/etf_baseline_2020-2024_exposure_sweep.md`, regenerated 2026-08-18 after fixing the last structural backtest defect (the strategy could never exit a position on its own signal). The strategy's own exit timing subtracts value at every exposure level, and the Bollinger filter A/B (`results/bollinger_filter_ab_2020-2024.md`) shows that the one configuration clearing 50 trades returns −1.88%.

Every earlier headline is superseded: "+42.9% at 92% gross" measured enter-once-and-hold — beta, not the strategy; `etf_baseline_2020-2024.md` (+53.4%) and `honest_backtest_2020-2024.md` (+646%, survivor-biased) carry SUPERSEDED banners. Don't quote any of them.

What's left to decide is what the repo is for: an execution/infrastructure sandbox, a home for validating the pairs-trading sleeve in `research/`, or an archive. See `TODO.md`.

## License

MIT. See `LICENSE`.
