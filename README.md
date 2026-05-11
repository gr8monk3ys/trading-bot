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

**Current baseline (2020-2024, 10 large-caps, MomentumStrategyBacktest defaults):** 102 trades, Sharpe 1.36, 47% max drawdown, 25.5% win rate, profit factor 7.27, +646.00% total return — now net of an end-of-period liquidation pass (positions are closed at the final bar with realistic spread + slippage, so equity reflects realized P&L only). The universe is hand-picked mega-caps that survived 2020-2024, so survivorship bias is still uncorrected; treat the Sharpe as an in-sample upper bound, not an out-of-sample expectation. Read the caveats section of `results/honest_backtest_2020-2024.md` before quoting any of this.

## License

MIT. See `LICENSE`.
