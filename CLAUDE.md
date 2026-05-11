# CLAUDE.md

Guidance for Claude Code working in this repository.

## Status: experimental

This repository is a personal algorithmic-trading sandbox. It is **paper-only** and has no proven edge. Do not deploy real capital. Previous versions of this document claimed an "institutional-grade" rating and a +42.68% backtest; both claims were unsupported by the evidence in the repo (see `PROFITABILITY_RESEARCH.md` for the analysis) and have been removed.

**If you are picking up this repo after a break:** read `results/where_we_landed.md` first. It is the durable summary of the May 2026 cleanup + validation work. The TL;DR is that the strategy underperforms SPY buy-and-hold on a bias-free test (+53% vs +95% over 2020-2024) and its real value is drawdown control (-7% max DD vs -33% passive), not profit maximization.

Two baselines exist, both 2020-2024, both `MomentumStrategyBacktest` defaults, both yfinance daily bars:

- `results/honest_backtest_2020-2024.md` — 10 hand-picked mega-caps (NVDA/AAPL/TSLA/MSFT/GOOGL/AMZN/META/JPM/SPY/QQQ). 102 trades, Sharpe 1.36, 47% max drawdown, profit factor 7.27, +646.00% total return. Survivor-biased: every symbol is one a 2026 retrospective would pick. This number reflects selection bias more than strategy edge.
- `results/etf_baseline_2020-2024.md` — 4 broad-market ETFs (SPY/QQQ/IWM/EFA). 38 trades (below the 50-trade significance bar — treated as INCONCLUSIVE), Sharpe 0.78, +53.42% total return. **The strategy underperformed SPY buy-and-hold (+95.3% / Sharpe 0.75) on this bias-free universe.** ETFs can't be delisted or selection-biased, so this is the honest test of strategy edge. The directional finding — most of the hand-picked baseline's outperformance was selection bias, not alpha — is reportable even at 38 trades, but Sharpe confidence intervals at that sample size are wide and the result could be chance.

When citing a performance number, lead with the ETF baseline. Both reports include caveats sections; read them before quoting.

## Project overview

Algorithmic trading bot on the Alpaca Trading API, async Python.

**Stack:** Python 3.10+, asyncio, pandas, numpy, TA-Lib, pytest-asyncio.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full data-flow diagram, per-package descriptions, and "where to start reading" guide. It's the single entry point for understanding the code organization.

## Core code path (production)

- `strategies/momentum_strategy.py` — RSI/MACD/ADX momentum with trailing stops, Kelly gated off by default.
- `strategies/momentum_strategy_backtest.py` — daily-data-friendly variant of the above.
- `strategies/mean_reversion_strategy.py` — pair to momentum for sideways regimes.
- `strategies/adaptive_strategy.py` — regime-switching coordinator that picks between momentum and mean-reversion. Imports only those two strategies plus `MarketRegimeDetector`; all ensemble/ML/cross-asset branches were removed during the 2026-05 cleanup.
- `strategies/simple_ma_strategy.py` — minimal reference strategy.
- `strategies/risk_manager.py` — position sizing, VaR, correlation rejection.
- `strategies/base_strategy.py` — abstract base for all strategies.
- `brokers/alpaca_broker.py`, `brokers/backtest_broker.py`, `brokers/order_builder.py`.
- `engine/backtest_engine.py`, `engine/performance_metrics.py`, `engine/strategy_manager.py`.
- `utils/circuit_breaker.py`, `utils/market_regime.py`, `utils/websocket_manager.py`, `utils/database.py`, `utils/notifier.py`, `utils/audit_log.py`, `utils/multi_timeframe.py`.
- `main.py` — single canonical CLI (`live`, `backtest`, `optimize`). The previous `live_trader.py` and `run_adaptive.py` were merged in by Phase 2 of the form-cleanup refactor; pass `--strategy adaptive --regime-only` / `--scan-only` for the old `run_adaptive.py` inspection modes, and `--risk-profile {conservative,balanced,aggressive}` for the old `live_trader.py` presets.

## Quarantined (unvalidated)

Under `research/`. Not imported by the production path, excluded from default `pytest`. Includes: factor models, factor portfolios, cross-asset signals, pairs trading, walk-forward / validated backtest, alpha-decay monitoring, IC tracker, point-in-time data, historical universe, crypto and extended-hours support. These modules have no evidence of edge in this codebase; treat them as ideas, not products.

## Commands

```bash
# Install
pip install -r requirements.txt

# Tests
pytest tests/                            # default: excludes research/
pytest tests/unit/test_risk_manager.py -v
pytest tests/ --cov=strategies --cov=utils --cov-report=html

# Backtests
python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-12-31
python main.py backtest --strategy adaptive --start-date 2024-01-01 --end-date 2024-12-31

# Paper trading (requires .env with ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER=True)
python main.py live --strategy adaptive
python main.py live --strategy MomentumStrategy --force
python main.py live --strategy adaptive --regime-only   # inspect regime, no trading
python main.py live --strategy adaptive --scan-only     # preview auto-scanned symbols

# Lint / format
black strategies/ brokers/ engine/ utils/
ruff check strategies/ brokers/ engine/ utils/
mypy strategies/ brokers/ engine/ utils/
```

## Implementation patterns

- All broker operations are async — use `await`.
- New strategies inherit `BaseStrategy`, set `NAME` class attribute, live in `strategies/`.
- Strategies populate `self.price_history[symbol]` before calling `_calculate_volatility(symbol)`.
- `OrderBuilder` is imported inside methods, not at module top, to avoid circular imports.

## Configuration

`config.py` exposes:
- `TRADING_PARAMS`, `RISK_PARAMS`, `TECHNICAL_PARAMS`.

Parameter blocks for deleted features (`ML_PARAMS`, `RL_PARAMS`, `OPTIONS_PARAMS`, `SENTIMENT_PARAMS`, `LLM_PARAMS`, `CRYPTO_PARAMS`, `OVERNIGHT_PARAMS`) were removed in the 2026-05 cleanup.

## Environment variables (`.env`)

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
PAPER=True

# Optional
DISCORD_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DATABASE_URL=sqlite:///trading_bot.db
```

## Critical gotchas

1. All broker operations need `await`.
2. NumPy pinned `>=1.24.0,<3.0.0` for compatibility.
3. Market hours: bot won't run if market closed unless `--force`.
4. `PAPER` env defaults true; live mode requires explicit opt-in and is **not recommended**.
5. Strategy discovery is import-based — strategies must be importable from `strategies/`.
6. `pytest` `asyncio_mode = auto` — don't add `@pytest.mark.asyncio` decorators.

## Test layout

```
tests/
├── unit/         # default test target
├── integration/  # slower, may hit APIs
├── fixtures/     # mock_broker, sample_price_history
└── ...
research/tests/   # quarantined; excluded from default pytest
```

## Style

From `.windsurfrules`:
- Functional preferred; avoid classes that exist only to namespace.
- Vectorized pandas/numpy over explicit loops.
- PEP 8.
- Descriptive variable names.

## When working in this repo

- Don't add features without evidence. If a feature can't be backed by a real backtest or A/B test, don't ship it.
- Don't reintroduce the "phases" framing. Phases are how the repo got into trouble.
- If you delete a module, delete its tests and its config in the same commit.
- Prefer editing existing files; only create new ones when necessary.
