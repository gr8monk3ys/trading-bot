# CLAUDE.md

Guidance for Claude Code working in this repository.

## Status: experimental

This repository is a personal algorithmic-trading sandbox. It is **paper-only** and has no proven edge. Do not deploy real capital. Previous versions of this document claimed an "institutional-grade" rating and a +42.68% backtest; both claims were unsupported by the evidence in the repo (see `docs/PROFITABILITY_RESEARCH.md` for the analysis) and have been removed.

**If you are picking up this repo after a break:** read `results/where_we_landed.md` first — especially its **2026-08-18 addendum**. Two rounds of backtest corruption were fixed in August 2026: the 2026-08-17 fixes (position detection, naked-short liability, cash-based sizing) and then the 2026-08-18 discovery that **exits were structurally unreachable** — the signal generator never emits "sell" and the engine never calls `on_bar`, so every prior run measured enter-once-and-hold. With exits working, the verdict is the harshest yet: at gross-100 target the strategy returns **+16.4% (Sharpe 0.16, 26 trades, ~59% realized gross) vs SPY buy-and-hold +95.3% (Sharpe 0.75)** — the strategy's own exit timing destroys value at every exposure level.

When citing a performance number, use `results/etf_baseline_2020-2024_exposure_sweep.md` (regenerated 2026-08-18 with exits active). Every older headline is superseded: the 2026-08-17 "+42.9% at 92% gross" measured a system that couldn't exit (beta, not strategy); `etf_baseline_2020-2024.md` +53% and `honest_backtest_2020-2024.md` +646% carry SUPERSEDED banners, the latter also survivor-biased.

Also fixed 2026-08-17: the live paper path had **never placed an order** — the 2026-05 cleanup deleted the production OrderGateway while `BaseStrategy` blocks all entries/exits without one. `engine/live_order_gateway.py` restores it and wires the circuit breaker's per-order interlock (previously dead code) into the order path.

## Project overview

Algorithmic trading bot on the Alpaca Trading API, async Python.

**Stack:** Python 3.10+, asyncio, pandas, numpy, TA-Lib, pytest-asyncio.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full data-flow diagram, per-package descriptions, and "where to start reading" guide. It's the single entry point for understanding the code organization.

## Core code path (production)

- `strategies/momentum_strategy.py` — RSI/MACD/ADX momentum with trailing stops, Kelly gated off by default.
- `strategies/momentum_strategy_backtest.py` — daily-data-friendly variant of the above.
- `strategies/mean_reversion_strategy.py` — pair to momentum for sideways regimes.
- `strategies/adaptive_strategy.py` — regime-switching coordinator that picks between momentum and mean-reversion. Imports only those two strategies plus `MarketRegimeDetector`; all ensemble/ML/cross-asset branches were removed during the 2026-05 cleanup. **It owns the bar subscription and its arms must not subscribe themselves** — until 2026-08-19 both arms self-subscribed, so they traded simultaneously (taking opposing positions) while the coordinator never ran `on_bar` and never detected a regime at all. Adaptive has no backtest evidence, and its sideways arm has never been backtested; it is not the CLI default for that reason.
- `strategies/simple_ma_strategy.py` — minimal reference strategy.
- `strategies/risk_manager/` — position sizing, VaR, correlation rejection.
- `strategies/base_strategy.py` — abstract base for all strategies.
- `brokers/alpaca_broker.py`, `brokers/backtest_broker.py`, `brokers/order_builder.py`.
- `engine/backtest_engine.py`, `engine/performance_metrics.py`, `engine/strategy_manager.py`, `engine/live_order_gateway.py`.
- `utils/circuit_breaker.py`, `utils/market_regime.py`, `utils/websocket_manager.py`, `utils/database/`, `utils/audit_log.py`, `utils/multi_timeframe.py`.
- `main.py` — single canonical CLI (`live`, `backtest`, `optimize`). The previous `live_trader.py` and `run_adaptive.py` were merged in by Phase 2 of the form-cleanup refactor; pass `--strategy adaptive --regime-only` / `--scan-only` for the old `run_adaptive.py` inspection modes, and `--risk-profile {conservative,balanced,aggressive}` for the old `live_trader.py` presets.

## Quarantined (unvalidated)

Under `research/`. Not imported by the production path, excluded from default `pytest`. Includes: factor models, factor portfolios, cross-asset signals, pairs trading, walk-forward / validated backtest, alpha-decay monitoring, IC tracker, point-in-time data, historical universe, crypto and extended-hours support. These modules have no evidence of edge in this codebase; treat them as ideas, not products.

## Commands

```bash
# Install
pip install -r requirements.txt

# Tests
pytest tests/                            # default: excludes research/
pytest tests/unit/strategies/test_risk_manager.py -v
pytest tests/ --cov=strategies --cov=utils --cov-report=html

# Backtests
python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-12-31
python main.py backtest --strategy adaptive --start-date 2024-01-01 --end-date 2024-12-31

# Paper trading (requires .env with ALPACA_API_KEY, ALPACA_SECRET_KEY, PAPER=True)
python main.py live                                     # defaults to MomentumStrategy
python main.py live --strategy adaptive                 # regime switching; unvalidated
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

`config.py` exposes `RISK_PARAMS` (read by `StrategyManager`), `SYMBOLS`, `ALPACA_CREDS`/`get_alpaca_creds()`, and `BACKTEST_PARAMS` (read only by the quarantined `research/` harness). Strategy parameters live in each strategy's `default_parameters()`, not in config.

`TRADING_PARAMS`, `TECHNICAL_PARAMS`, and `SYMBOL_SELECTION` were deleted in the 2026-08 slop sweep — nothing ever read them (as `ML_PARAMS` etc. were deleted in 2026-05).

## Environment variables (`.env`)

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
PAPER=True

# Required for the web dashboard (fails closed without it)
DASHBOARD_TOKEN=

# Optional
DATABASE_URL=sqlite:///trading_bot.db
```

(Discord/Telegram notification vars were dropped with `utils/notifier.py` in the 2026-08 slop sweep — nothing imported it, so the vars never did anything.)

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
