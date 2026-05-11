# trading-bot

Personal algorithmic-trading sandbox on Alpaca, async Python.

**Status:** experimental, paper-only, no proven edge. Do not deploy real capital.

**Coming back to this repo?** Read [`results/where_we_landed.md`](results/where_we_landed.md) first — durable summary of the May 2026 cleanup + validation. Headline: strategy underperforms SPY on a bias-free test; real value is drawdown control (-7% max DD vs -33% passive), not profit maximization.

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

## Performance

Two baselines exist, and the ETF one is the honest test of strategy edge.

- `results/honest_backtest_2020-2024.md` — 10 hand-picked mega-caps, +646%, Sharpe 1.36. Survivor-biased: every name in the universe (NVDA, TSLA, AAPL, MSFT, etc.) is one a 2026 retrospective would obviously pick. Treat this as a cautionary contrast, not as evidence of edge.
- `results/etf_baseline_2020-2024.md` — SPY/QQQ/IWM/EFA, broad market ETFs that can't be delisted or selection-biased. **The strategy returned +53.4% / Sharpe 0.78 on this universe, underperforming SPY buy-and-hold (+95.3% / Sharpe 0.75).** 38 trades, below the 50-trade significance bar, so this is a hint not a verdict — but the direction is the most damning one: most of the hand-picked baseline's outperformance was selection bias, not strategy alpha.

When citing a single performance number, prefer the ETF baseline. Earlier reports (notably `results/backtest_report_2024.md`) used 9 trades and are below the statistical-significance bar this repo's own `docs/PROFITABILITY_RESEARCH.md` calls out. Don't quote them.

## License

MIT. See `LICENSE`.
