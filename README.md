# trading-bot

Personal algorithmic-trading sandbox on Alpaca, async Python.

**Status:** experimental, paper-only, no proven edge. Do not deploy real capital.

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
python run_adaptive.py
```

## Performance

When `results/honest_backtest_2020-2024.md` exists, that is the only performance number to cite. Earlier reports (notably `backtest_report_2024.md`) used 9 trades and are below the statistical-significance bar this repo's own `PROFITABILITY_RESEARCH.md` calls out. Don't quote them.

## License

MIT. See `LICENSE`.
