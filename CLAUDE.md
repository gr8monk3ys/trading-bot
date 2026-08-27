# CLAUDE.md

Alpaca paper-trading bot in async Python with a backtester. Verdict, from
`results/etf_baseline_2020-2024_exposure_sweep.md` (2020-2024, exits active):
momentum +16.4% / Sharpe 0.16 / 26 trades vs SPY buy-and-hold +95.3% / 0.75.
Nothing here beats buy-and-hold. Paper only; never deploy real capital.
Cite only that file for numbers; older `results/*.md` carry SUPERSEDED banners.

## Run / test

- Install: `uv sync --group dev --group test`; TA-Lib C library required.
- Tests: `uv run pytest tests/unit/` (CI also runs it with `-W error`).
  `asyncio_mode = auto`; never add `@pytest.mark.asyncio`.
- Lint: `uv run ruff check . && uv run black --check .` (line length 100).
- Backtest: `uv run python main.py backtest --strategy MomentumStrategyBacktest --start-date 2024-01-01 --end-date 2024-12-31`
- Paper: `uv run python main.py live` (needs `.env`; `--force` outside market hours).
- Regenerate the canonical numbers: `uv run python scripts/run_etf_baseline.py`.

## Where things live

- `strategies/` momentum (+ `_backtest` daily variant), mean reversion, adaptive
  regime switcher, `risk_manager/`, `base_strategy.py`.
- `brokers/` Alpaca wrapper, backtest broker, `order_builder.py`.
- `engine/` backtest engine, performance metrics, strategy manager, live order gateway.
- `utils/` circuit breaker, market regime, websocket, database, audit log.
- `main.py` the only CLI: `live`, `backtest`, `optimize`.
- `results/` committed backtest artifacts; `docs/architecture.md` for the data flow.
- Unvalidated quant code (factor models, pairs trading, walk-forward harness) is
  on branch `archive/research`, not on main.

## Gotchas

- Every broker call is async. `OrderBuilder` is imported inside methods (circular import).
- `AdaptiveStrategy` owns the bar subscription; its arms must not subscribe themselves (#89).
- The backtest engine never calls `on_bar`: trailing stops and bracket orders are
  untested in backtests. Exits come from opposite signals (#84).
- Strategy discovery is import-based; a strategy must be importable from `strategies/`.
- Do not add features without a >=50-trade out-of-sample backtest. Delete a module's
  tests and config in the same commit as the module.
