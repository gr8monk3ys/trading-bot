# Architecture

This document is the single entry point for understanding the code organization of this repo. Read it before reading code.

**One-line context:** this is a paper-only experimental trading bot whose strategy has been validated as a drawdown-control sleeve rather than a profit maximizer. See [`results/where_we_landed.md`](../results/where_we_landed.md) for the full validation history and [`PROFITABILITY_RESEARCH.md`](PROFITABILITY_RESEARCH.md) for realistic performance expectations. The May 2026 honest cleanup and form-cleanup refactor reduced the repo from 193K to ~45K LOC and reorganized the remaining code into the structure described here.

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
        │   (strategies/base_strategy.py +           │
        │    strategies/base/)                       │
        │  ┌─────────────────────────────────────┐  │
        │  │  MomentumStrategy / MeanReversion / │  │
        │  │  AdaptiveStrategy / SimpleMA        │  │
        │  └─────────────────┬───────────────────┘  │
        └─────────────────────┼─────────────────────┘
                              │
                              │ submit_entry_order / submit_exit_order
                              ▼
              ┌───────────────────────────────────┐
              │       OrderGateway                 │  (live mode only — backtest
              │   (live: deleted in cleanup;       │   uses BacktestOrderGateway
              │    backtest:                       │   wired in engine/backtest/
              │    engine/backtest_order_gateway.  │   runner.py)
              │    py)                             │
              └───────────────────┬───────────────┘
                                  │
                  ┌───────────────┴─────────────────┐
                  │                                 │
              [live]                          [backtest]
                  │                                 │
        ┌─────────▼──────────┐         ┌────────────▼──────────────┐
        │    AlpacaBroker    │         │   BacktestBroker          │
        │ (brokers/          │         │ (brokers/backtest_broker. │
        │  alpaca_broker.py  │         │  py facade + brokers/     │
        │  facade + brokers/ │         │  backtest/)               │
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
Broker abstractions. `AlpacaBroker` is the live broker; `BacktestBroker` is the backtest-mode simulator. Both have been split into focused sub-modules:

- `brokers/alpaca_broker.py` — thin facade combining the mixins below.
- `brokers/alpaca/account.py` — connection, auth, account/position/asset queries.
- `brokers/alpaca/orders.py` — order submission, cancel, replace, partial-fill tracking.
- `brokers/alpaca/market_data.py` — stock bars, quotes, news.
- `brokers/alpaca/crypto.py` — crypto-asset bars, quotes, orders.
- `brokers/alpaca/streaming.py` — websocket lifecycle, trade-update handler.
- `brokers/alpaca/portfolio.py` — portfolio history, equity curve, performance.
- `brokers/alpaca/_retry.py` — `retry_with_backoff` decorator.
- `brokers/backtest_broker.py` — facade.
- `brokers/backtest/core.py` — init, price retrieval, position/balance queries.
- `brokers/backtest/execution.py` — order placement, slippage, partial fills, stop orders.
- `brokers/backtest/gaps.py` — gap events, gap simulation.
- `brokers/order_builder.py` — fluent order construction (bracket, OCO, trailing stop).

### `engine/`
Backtest engine and performance analytics.

- `engine/backtest_engine.py` — `BacktestEngine` facade.
- `engine/backtest/core.py` — main run loop, session resolution, signed-position P&L calculator.
- `engine/backtest/runner.py` — comprehensive backtest driver (data loading, broker setup, OrderGateway wiring, end-of-period liquidation, result assembly).
- `engine/backtest/walk_forward.py` — walk-forward folds and fold-level metrics.
- `engine/backtest_order_gateway.py` — `BacktestOrderGateway` (`BaseStrategy` requires a gateway since PR #22; this satisfies that requirement in backtest mode).
- `engine/performance_metrics.py` — `PerformanceMetrics` class: total return, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor.
- `engine/statistical_testing.py` — Bonferroni / FDR-BH multiple-testing corrections, Cohen's d, Hedge's g effect sizes.
- `engine/strategy_manager.py` — orchestrates multiple strategies in live mode; capital allocation.

### `strategies/`
Trading strategies. Each is a subclass of `BaseStrategy`.

- `strategies/base_strategy.py` — facade.
- `strategies/base/strategy.py` — abstract class, init, lifecycle, state, order submission scaffolding.
- `strategies/base/position_sizing.py` — Kelly criterion, position-size limits, volatility/streak adjustments.
- `strategies/momentum_strategy.py` — facade.
- `strategies/momentum/strategy.py` — `MomentumStrategy` class: state, on-bar dispatch, execute.
- `strategies/momentum/indicators.py` — TA-Lib RSI/MACD/ADX/SMA calculations.
- `strategies/momentum/signals.py` — entry/exit signal generation, trailing stops.
- `strategies/momentum_strategy_backtest.py` — daily-bar variant of `MomentumStrategy`.
- `strategies/mean_reversion_strategy.py` — facade.
- `strategies/mean_reversion/strategy.py` — `MeanReversionStrategy` class.
- `strategies/mean_reversion/signals.py` — indicator updates, signal generation, exits.
- `strategies/adaptive_strategy.py` — regime-switching coordinator; routes to momentum or mean-reversion based on `MarketRegimeDetector`.
- `strategies/simple_ma_strategy.py` — minimal reference SMA-crossover strategy.
- `strategies/risk_manager/__init__.py` — `RiskManager` facade.
- `strategies/risk_manager/calculator.py` — volatility, all VaR methods, expected shortfall, max drawdown, position risk, correlation, portfolio risk.
- `strategies/risk_manager/enforcer.py` — adjust position size, limit enforcement, margin, halt decisions.

### `utils/`
Utilities that the production path actually uses. (~30 modules. Many speculative utilities were deleted in the cleanup.)

Core utilities:
- `utils/circuit_breaker.py` — daily-loss halts.
- `utils/database/core.py` + `analytics.py` — SQLite trade/position/metrics storage with aggregation queries.
- `utils/market_regime.py` — `MarketRegimeDetector`: bull/bear/sideways/volatile detection.
- `utils/indicators.py` + `indicator_analysis.py` — technical indicator library.
- `utils/multi_timeframe.py` — multi-timeframe analyzer (canonical version).
- `utils/audit_log.py` — structured event logging.
- `utils/websocket_manager.py` — auto-reconnecting websocket abstraction.
- `utils/notifier.py` — Discord/Telegram notifications.
- `utils/kelly_criterion.py` — Kelly position-sizing math.
- `utils/streak_sizing.py` — streak-based sizing adjustments.
- `utils/volatility_regime.py` — volatility-regime classifier.

Other utilities (in active use by the production path; see `git grep` for callers): `correlation_manager`, `earnings_calendar`, `economic_calendar`, `execution_quality_gate`, `execution_tracker`, `factor_exposure_limits`, `fundamental_data`, `greeks_aggregator`, `order_lifecycle`, `paper_trading_monitor`, `partial_fill_tracker`, `performance_tracker`, `pnl_attribution`, `portfolio_rebalancer`, `portfolio_stress`, `position_scaling`, `relative_strength`, `sector_rotation`, `stress_tester`, `support_resistance`, `tax_compliance`, `trading_hours`, `twap_executor`, `universe_provider`, `visualization`, `volume_filter`, `vwap_executor`.

### `data/`
Data providers (small footprint after the 2026-05 cleanup quarantined most of this tree to `research/`).

### `research/`
Plausible-but-unvalidated quant work — factor models, pairs trading, walk-forward validation, cross-asset signals. **Not imported by the production path; excluded from default pytest.** See `research/README.md` for the contents and the bar for promoting anything back.

### `scripts/`
Operational scripts (kept minimal after the cleanup):

- `scripts/run_honest_baseline.py` — produces `results/honest_backtest_2020-2024.{md,json}`. The hand-picked-mega-cap (survivor-biased) baseline.
- `scripts/run_etf_baseline.py` — produces `results/etf_baseline_2020-2024.{md,json}`. The bias-free baseline (SPY/QQQ/IWM/EFA). This is the canonical performance reference.
- `scripts/dashboard.py` — terminal monitoring dashboard.
- `scripts/kill_switch.py` — emergency halt of all trading + position liquidation.
- `scripts/simple_backtest.py` — lightweight backtest runner (separate from the canonical CLI; kept for ad-hoc use).
- `scripts/simple_trader.py` — minimal trading-bot runner.
- `scripts/quickstart.py` — interactive setup helper.
- `scripts/run.py` — generic strategy runner.
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
3. `strategies/base/strategy.py` — base class lifecycle (`initialize`, `on_trading_iteration`, `submit_entry_order`).
4. The specific strategy file (`strategies/momentum/strategy.py` etc.).
5. `brokers/alpaca/orders.py` — how orders actually go to Alpaca.

**If you're touching the backtest path:**
1. `scripts/run_etf_baseline.py` — the canonical baseline script (read it as the reference invocation).
2. `engine/backtest/runner.py` — `run_backtest` driver.
3. `engine/backtest/core.py` — main loop and P&L matching.
4. `brokers/backtest/execution.py` — how simulated orders fill.
5. `engine/backtest_order_gateway.py` — the gateway shim that lets `BaseStrategy` work in backtest mode.

**If you're touching risk management:**
1. `strategies/base/position_sizing.py` — base-class sizing helpers.
2. `strategies/risk_manager/calculator.py` — risk math.
3. `strategies/risk_manager/enforcer.py` — sizing decisions and halts.
4. `utils/circuit_breaker.py` — daily-loss halts.

**If you're investigating a backtest result:**
1. Read the report at `results/etf_baseline_2020-2024.md` (or `honest_backtest_2020-2024.md`).
2. The trade log lives in the corresponding `.json` file.
3. `engine/backtest/core.py::_calculate_trade_pnl` is the signed-position matcher.

## Where an order gets submitted

In live mode:

```
Strategy.execute_trade()
  → BaseStrategy.submit_entry_order()
    → self.order_gateway.submit_order(...)
      → AlpacaBroker.submit_order_advanced(...)
        → brokers/alpaca/orders.py::AlpacaOrdersMixin.submit_order_advanced
          → alpaca-py TradingClient.submit_order
            → Alpaca API
```

In backtest mode:

```
Strategy.execute_trade()
  → BaseStrategy.submit_entry_order()
    → self.order_gateway.submit_order(...)            (BacktestOrderGateway)
      → BacktestBroker.place_order(...)               (or submit_order_advanced for entries)
        → brokers/backtest/execution.py::BacktestBrokerExecutionMixin.place_order
          → applies slippage + spread, records the trade in self._trades
```

## What's NOT in this repo

- Tier-3 institutional features (LSTM, RL, factor models, LLM analysis, alt-data scrapers, options trading, news sentiment) — **all deleted or quarantined** in the May 2026 cleanup. See `results/where_we_landed.md` for the deletion list. Do not reintroduce without evidence.
- The "9 phases of institutional features" framing — **also deleted**. If you're tempted to add a "Phase 10," stop and read `results/where_we_landed.md`.

## Conventions

- All broker operations are async. Use `await`.
- New strategies inherit `BaseStrategy`, set `NAME` class attribute, live under `strategies/`.
- Don't add features without evidence. The bar for promoting a `research/` module is in `research/README.md`.
- Don't refactor unrelated code. The May 2026 form-cleanup refactor (plan in `/home/codespace/.claude/plans/alright-then-lets-keep-starry-beaver.md`) is complete; further large structural changes need their own plan.

## See also

- `CLAUDE.md` — guidance for Claude Code working in this repo.
- `README.md` — public-facing project description.
- [`PROFITABILITY_RESEARCH.md`](PROFITABILITY_RESEARCH.md) — realistic performance expectations and limits.
- `results/where_we_landed.md` — durable summary of the May 2026 cleanup + validation outcome.
- `results/can_this_beat_qqq.md` — pre-validation skeptical analysis.
- `results/etf_baseline_2020-2024.md` — canonical performance reference.
- `TODO.md` — open follow-up items.
