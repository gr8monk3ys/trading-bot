# Architecture

This document is the single entry point for understanding the code organization of this repo. Read it before reading code.

**One-line context:** this is a paper-only experimental trading bot with **no demonstrated edge** — at matched exposure the momentum strategy underperforms SPY buy-and-hold (see `results/etf_baseline_2020-2024_exposure_sweep.md`; the earlier "drawdown-control sleeve" story was an exposure artifact). See [`results/where_we_landed.md`](../results/where_we_landed.md) for the full validation history and [`PROFITABILITY_RESEARCH.md`](PROFITABILITY_RESEARCH.md) for realistic performance expectations. The May 2026 honest cleanup and form-cleanup refactor reduced the repo from 193K to ~45K LOC and reorganized the remaining code into the structure described here.

## Data flow

The main data-flow paths through the system:

```
                ┌───────────────────────────┐
                │      python main.py        │  single CLI entry
                │   live | backtest | optimize │
                └────────────┬──────────────┘
                             │
                  ┌──────────┴────────────┐
                  │                       │
              [live mode]            [backtest mode]
                  │                       │
        ┌─────────▼──────────┐  ┌─────────▼──────────┐
        │   StrategyManager  │  │   BacktestEngine   │
        │ (engine/strategy_  │  │ (engine/backtest_  │
        │  manager.py)       │  │  engine.py +       │
        │                    │  │  engine/backtest/) │
        └─────────┬──────────┘  └─────────┬──────────┘
                  │                       │
        ┌─────────▼──────────────────────▼──────────┐
        │            BaseStrategy                    │
        │   (strategies/base_strategy.py)            │
        │                                            │
        │  ┌─────────────────────────────────────┐  │
        │  │  MomentumStrategy / MeanReversion / │  │
        │  │  AdaptiveStrategy / SimpleMA        │  │
        │  └─────────────────┬───────────────────┘  │
        └─────────────────────┼─────────────────────┘
                              │
                              │ submit_entry_order / submit_exit_order
                              ▼
              ┌───────────────────────────────────┐
              │       OrderSubmission              │  one path for both modes:
              │   engine/order_submission.py       │  halt gate → build →
              │   OrderIntent → OrderOutcome       │  dispatch → normalise →
              │   (circuit-breaker interlock)      │  protective levels (ADR 0004)
              └───────────────────┬───────────────┘
                                  │
                  ┌───────────────┴─────────────────┐
                  │                                 │
              [live]                          [backtest]
                  │                                 │
        ┌─────────▼──────────┐         ┌────────────▼──────────────┐
        │    AlpacaBroker    │         │   BacktestBroker          │
        │ (brokers/          │         │ (brokers/backtest/)       │
        │  alpaca_broker.py  │         │                           │
        │  facade + brokers/ │         │                           │
        │  alpaca/ package)  │         │                           │
        └─────────┬──────────┘         └────────────┬──────────────┘
                  │                                 │
        ┌─────────▼──────────┐         ┌────────────▼──────────────┐
        │  Alpaca REST/WS    │         │  Historical bars via      │
        │  (paper API)       │         │  yfinance or Alpaca       │
        └────────────────────┘         └───────────────────────────┘
```

## Packages

### `brokers/`
Broker abstractions. `brokers/protocol.py` states the seam both brokers satisfy (`Broker` protocol, one `Position` shape, async throughout); everything else on either class is that adapter's internal seam (ADR 0003). `AlpacaBroker` is the live broker; `BacktestBroker` is the backtest-mode simulator. Both have been split into focused sub-modules:

- `brokers/alpaca_broker.py` — thin facade combining the mixins below.
- `brokers/alpaca/account.py` — connection, auth, account/position/asset queries.
- `brokers/alpaca/orders.py` — order submission, cancel, replace, partial-fill tracking.
- `brokers/alpaca/market_data.py` — stock bars, quotes, news.
- `brokers/alpaca/crypto.py` — crypto-asset bars, quotes, orders.
- `brokers/alpaca/streaming.py` — websocket lifecycle, trade-update handler.
- `brokers/alpaca/portfolio.py` — portfolio history, equity curve, performance.
- `brokers/alpaca/_retry.py` — `retry_with_backoff` decorator.
- `brokers/backtest/core.py` — init, price retrieval, position/balance queries.
- `brokers/backtest/execution.py` — order placement, slippage, partial fills, stop orders.
- `brokers/backtest/gaps.py` — gap events, gap simulation.
- `brokers/order_builder.py` — fluent order construction (bracket, OCO, trailing stop).

### `engine/`
Backtest engine and performance analytics.

- `engine/backtest_engine.py` — `BacktestEngine` facade.
- `engine/backtest/core.py` — session resolution and the signed-position P&L calculator.
- `engine/session.py` — `Session`: one strategy over sessions of bars. `prepare()` once per session, then `decide()` per symbol against a fresh `PortfolioView`, submitting the returned intents (ADR 0001). The backtest runner drives it day by day with `Session`; `StrategyManager` drives `LiveSession` from the websocket bar stream, which is the only bar subscriber (ADR 0002). Strategies have no `on_bar`.
- `engine/historical_bars.py` — the one way to load bars: Alpaca or yfinance adapter behind `load_bars`, a data outcome per symbol (loaded / empty / failed), and `DataUnavailableError` when any symbol is missing. The baseline scripts and the runner all use it (ADR 0008). Also hosts `compute_buy_and_hold`.
- `engine/backtest/runner.py` — comprehensive backtest driver (data loading, broker setup, OrderSubmission wiring, end-of-period liquidation, result assembly).
- `engine/order_submission.py` — `OrderSubmission`: the one path from an `OrderIntent` to an `OrderOutcome` (halt gate, build, dispatch, normalise, protective levels). Live and backtest differ only in the broker behind it (ADR 0004).
- `engine/performance_metrics.py` — `PerformanceMetrics.calculate_metrics` (total return, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor) and `verdict()`, the one judgement of whether a run's numbers are quotable; every script renders that verdict.
- `engine/statistical_testing.py` — Bonferroni / FDR-BH multiple-testing corrections, Cohen's d, Hedge's g effect sizes.
- `engine/strategy_manager.py` — orchestrates multiple strategies in live mode; capital allocation.

### `strategies/`
Trading strategies. Each is a subclass of `BaseStrategy`.

- `strategies/base_strategy.py` — `BaseStrategy`: init, lifecycle, state, order submission scaffolding, Kelly / position-size / volatility / streak sizing helpers.
- `strategies/momentum_strategy.py` — `MomentumStrategy`: state, on-bar dispatch, TA-Lib indicators, entry/exit signals, trailing stops, execute.
- `strategies/momentum_strategy_backtest.py` — daily-bar variant of `MomentumStrategy`.
- `strategies/mean_reversion_strategy.py` — `MeanReversionStrategy`: indicator updates, signal generation, exits, execute.
- `strategies/adaptive_strategy.py` — regime-switching coordinator; routes to momentum or mean-reversion based on `MarketRegimeDetector`.
- `strategies/simple_ma_strategy.py` — minimal reference SMA-crossover strategy.
- `strategies/risk_manager/__init__.py` — `RiskManager` facade.
- `strategies/risk_manager/calculator.py` — volatility, all VaR methods, expected shortfall, max drawdown, position risk, correlation, portfolio risk.
- `strategies/risk_manager/enforcer.py` — adjust position size, limit enforcement, margin, halt decisions.

### `utils/`
Utilities that the production path actually uses. (The 2026-08 slop sweep deleted ~23 modules that nothing imported; this list is now the honest inventory.)

Core utilities:
- `utils/circuit_breaker.py` — daily-loss halts + economic-event blocking.
- `utils/economic_calendar.py` — FOMC/NFP/CPI event calendar (lazily imported by the circuit breaker, on by default).
- `utils/database/core.py` + `analytics.py` — SQLite trade/position/metrics storage with aggregation queries.
- `utils/market_regime.py` — `MarketRegimeDetector`: bull/bear/sideways/volatile detection.
- `utils/multi_timeframe.py` — multi-timeframe analyzer (canonical version).
- `utils/audit_log.py` — hash-chained event logging.
- `utils/websocket_manager.py` — auto-reconnecting websocket abstraction.
- `utils/kelly_criterion.py` — Kelly position-sizing math.
- `utils/streak_sizing.py` — streak-based sizing adjustments.
- `utils/volatility_regime.py` — volatility-regime classifier.
- `utils/order_lifecycle.py`, `utils/partial_fill_tracker.py`, `utils/performance_tracker.py`, `utils/sector_rotation.py`, `utils/portfolio_stress.py` — order/portfolio support used by the broker mixins and scanner.

### `data/`
Data providers (small footprint after the 2026-05 cleanup).

### `scripts/`
Operational scripts (kept minimal after the cleanup):

- `scripts/run_etf_baseline.py` — produces the exposure-sweep baseline (`results/etf_baseline_2020-2024_gross{25,50,100}.{md,json}` + `_exposure_sweep.md`). The bias-free SPY/QQQ/IWM/EFA comparison; **this is the canonical performance reference.**
- `scripts/run_honest_baseline.py` — produces `results/honest_backtest_2020-2024.{md,json}`. The hand-picked-mega-cap (survivor-biased) baseline; SUPERSEDED, kept for the audit trail.
- `scripts/paper_smoke_test.py` — end-to-end live-order-path proof against the Alpaca paper API (1-share unfillable limit, audit-log check, cancel).
- `scripts/dashboard.py` — terminal monitoring dashboard.
- `scripts/kill_switch.py` — emergency halt of all trading + position liquidation.
- `scripts/simple_trader.py` — minimal trading-bot runner.
- `scripts/quickstart.py` — interactive setup helper.
- `scripts/check_positions.py` — paper-trading account status query.
- `scripts/monitor_bot.py` — real-time monitoring dashboard.

### `tests/`
Mirrors the source tree:

- `tests/unit/brokers/` — broker tests.
- `tests/unit/engine/` — engine tests.
- `tests/unit/strategies/` — strategy tests.
- `tests/unit/utils/` — utility tests.
- `tests/unit/misc/` — miscellaneous.
- `tests/unit/conftest.py` — shared fixtures (`mock_broker`, `sample_price_history`, etc.).

### `web/`
Optional FastAPI dashboard for live monitoring.

## Where to start reading

**If you're touching live trading:**
1. `main.py` — the CLI entry point.
2. `engine/strategy_manager.py` — orchestration.
3. `strategies/base_strategy.py` — base class lifecycle (`initialize`, `on_trading_iteration`, `submit_entry_order`).
4. The specific strategy file (`strategies/momentum_strategy.py` etc.).
5. `brokers/alpaca/orders.py` — how orders actually go to Alpaca.

**If you're touching the backtest path:**
1. `scripts/run_etf_baseline.py` — the canonical baseline script (read it as the reference invocation). It writes `results/manifest.json`, which `scripts/audit_results.py` reads.
2. `engine/backtest/runner.py` — `run_backtest` driver.
3. `engine/session.py` — prepare / decide / submit per session; `engine/backtest/core.py` for P&L matching.
4. `brokers/backtest/execution.py` — how simulated orders fill.
5. `engine/order_submission.py` — how intents become outcomes in both modes.

**If you're touching risk management:**
1. `strategies/base_strategy.py` — base-class sizing helpers.
2. `strategies/risk_manager/calculator.py` — risk math.
3. `strategies/risk_manager/enforcer.py` — sizing decisions and halts.
4. `utils/circuit_breaker.py` — daily-loss halts.

**If you're investigating a backtest result:**
1. Read `results/etf_baseline_2020-2024_exposure_sweep.md` (the like-for-like comparison; the older `etf_baseline_2020-2024.md` and `honest_backtest_2020-2024.md` carry SUPERSEDED banners).
2. The trade log lives in the corresponding `.json` file.
3. `engine/backtest/core.py::_calculate_trade_pnl` is the signed-position matcher.

## Where an order gets submitted

In live mode:

```
Strategy.execute_trade()
  → BaseStrategy.submit_entry_order(OrderIntent)
    → OrderSubmission.submit(intent)                 (engine/order_submission.py)
      → AlpacaBroker._internal_submit_order(request, gateway_token)
        → brokers/alpaca/orders.py::AlpacaOrdersMixin._internal_submit_order
          → alpaca-py TradingClient.submit_order
            → Alpaca API
```

In backtest mode:

```
Strategy.execute_trade()
  → BaseStrategy.submit_entry_order(OrderIntent)
    → OrderSubmission.submit(intent)                 (same module, backtest broker behind it)
      → BacktestBroker.submit_order_advanced(...) → place_order(...)
        → brokers/backtest/execution.py::BacktestBrokerExecutionMixin.place_order
          → applies slippage + spread, records the trade in self._trades
```

## What's NOT in this repo

- Tier-3 institutional features (LSTM, RL, factor models, LLM analysis, alt-data scrapers, options trading, news sentiment) — **all deleted** in the May 2026 cleanup; the last unvalidated remainder (`research/`) was moved to the `archive/research` branch in August 2026. See `results/where_we_landed.md` for the deletion list. Do not reintroduce without evidence.
- The "9 phases of institutional features" framing — **also deleted**. If you're tempted to add a "Phase 10," stop and read `results/where_we_landed.md`.

## Conventions

- All broker operations are async. Use `await`.
- New strategies inherit `BaseStrategy`, set `NAME` class attribute, live under `strategies/`.
- Don't add features without evidence: a >=50-trade out-of-sample backtest and a significance check before anything ships.
- Don't refactor unrelated code. The May 2026 form-cleanup refactor is complete; further large structural changes need their own plan.

## See also

- `CLAUDE.md` — guidance for Claude Code working in this repo.
- `README.md` — public-facing project description.
- `results/where_we_landed.md` — durable summary of the May 2026 cleanup + validation outcome.
- `results/can_this_beat_qqq.md` — pre-validation skeptical analysis.
- `results/etf_baseline_2020-2024_exposure_sweep.md` — canonical performance reference.
- GitHub issues — open follow-up items.
