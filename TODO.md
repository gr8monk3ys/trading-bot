# TODO

Follow-ups after the 2026-05 honest cleanup (`docs/superpowers/specs/2026-05-11-honest-cleanup-design.md`).

## Direction (decide before doing more work)

- [ ] Decide the actual goal: paper-only learning sandbox, path to live capital, public showcase, or something else. Different goals produce different next steps. Do not add features before deciding this.

## Code organization (deferred from cleanup)

- [ ] Measure file sizes after the cleanup. If `main.py`, `live_trader.py`, or `adaptive_strategy.py` are still over 800 LOC, split them — one module per responsibility.
- [ ] Audit the kept `utils/` modules. Some (e.g. `multi_timeframe.py`) may be vestigial after the deletions.

## Project (engine/backtest bugs surfaced by the Task 8 baseline)

- [x] **Wire an `OrderGateway` into `BacktestEngine`.** Done in commit wiring `BacktestOrderGateway` into `engine/backtest_engine.py` (see `engine/backtest_order_gateway.py`). The shim inside `scripts/run_honest_baseline.py` has been removed; backtests now route orders through the canonical gateway automatically.
- [x] **Fix the short-trade P&L bug in `engine/backtest_engine._calculate_trade_pnl`.** Fixed by rewriting the matcher as a signed-qty state machine that handles long open/close, short open/cover, partial closes, and immediate reversals. Adds `tests/unit/test_backtest_engine_pnl_accounting.py`. Re-running `scripts/run_honest_baseline.py` produced unchanged headline equity (still +646.64%, Sharpe 1.36 — those come from broker MTM, not trade records) but `profit_factor` collapsed from 3.03 to 0.42 and `avg_win`/`avg_loss` halved, exposing that the pre-fix zeros on short legs were silently inflating per-trade quality metrics.

## Validation (if continuing toward live)

- [ ] Run 6+ months of paper trading on the kept core. Stop pretending shorter samples are meaningful.
- [ ] Produce at least 50 real trades before claiming Sharpe or win rate.
- [ ] Re-run `scripts/run_honest_baseline.py` quarterly; track drift in `results/`.
- [ ] **Replace the hand-picked baseline universe with a point-in-time universe.** The Task 8 universe (SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM) is mega-caps that survived 2020-2024. Any momentum strategy will look heroic on this set. A real baseline needs a point-in-time universe — the `historical_universe` module in `research/` does this but isn't wired up. Either promote it (following the research-tree promotion bar below) or build the equivalent in the production path.
- [x] **Add an end-of-backtest liquidation pass.** Done via `BacktestEngine._liquidate_open_positions`, called after the trading-day loop in `run_backtest`. All remaining positions are closed at the final bar via `BacktestBroker.place_order`, so they pick up the same spread + market-impact slippage as any other trade and land in `broker.get_trades()`. Adds `tests/unit/test_backtest_end_liquidation.py`. Re-running `scripts/run_honest_baseline.py` produced: trades 93→102, total_return 646.64%→646.00% (the engine's existing `get_portfolio_value`-based equity curve already mark-to-marketed those positions, so realizing them at ~0.5% slippage barely moved the number), profit_factor 0.42→7.27, win_rate 20.4%→25.5%. The headline is now realized cash, not unrealized MTM — but the survivor-biased universe is still uncorrected (see point-in-time universe item above).

## Research-tree promotion (only if a research/ module proves itself)

- [ ] For any `research/` module being considered for promotion, require: (1) ≥50-trade out-of-sample backtest, (2) statistical-significance check (permutation or FDR), (3) written hypothesis, (4) evidence the signal isn't already priced. Document in `research/<module>/PROMOTION.md`.

## Operational (only if scaling beyond solo paper)

- [ ] If running unattended for extended periods, re-evaluate which of the deleted operational scripts (kill switch is already kept) actually need to come back. Don't restore wholesale.
