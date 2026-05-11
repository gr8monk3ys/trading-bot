# TODO

Follow-ups after the 2026-05 honest cleanup (`docs/superpowers/specs/2026-05-11-honest-cleanup-design.md`).

## Direction (decide before doing more work)

- [ ] Decide the actual goal: paper-only learning sandbox, path to live capital, public showcase, or something else. Different goals produce different next steps. Do not add features before deciding this.

## Code organization (deferred from cleanup)

- [ ] Measure file sizes after the cleanup. If `main.py`, `live_trader.py`, or `adaptive_strategy.py` are still over 800 LOC, split them — one module per responsibility.
- [ ] Audit the kept `utils/` modules. Some (e.g. `multi_timeframe.py`) may be vestigial after the deletions.

## Project (engine/backtest bugs surfaced by the Task 8 baseline)

- [ ] **Wire an `OrderGateway` into `BacktestEngine`.** PR #22 made `BaseStrategy.submit_entry_order` / `submit_exit_order` require an `OrderGateway`. `engine/backtest_engine.py` doesn't construct one, so every backtest order was being rejected with "No OrderGateway configured" until Task 8 added a `_BacktestOrderGateway` shim inside `scripts/run_honest_baseline.py`. Proper fix: either wire a gateway in `BacktestEngine.initialize()`, or make the gateway requirement conditional on a "live" mode flag. The shim in the baseline script should then be removed.
- [ ] **Fix the short-trade P&L bug in `engine/backtest_engine._calculate_trade_pnl`.** The matcher only pairs sells against prior buys per symbol, so short trades get recorded with `pnl: 0`. The Task 8 backtest had 34 such trades. Zero-P&L exits don't count as losses, which biases `profit_factor` upward and silently inflates win-rate framing. Fix the matching logic to track open shorts and pair covers against them.

## Validation (if continuing toward live)

- [ ] Run 6+ months of paper trading on the kept core. Stop pretending shorter samples are meaningful.
- [ ] Produce at least 50 real trades before claiming Sharpe or win rate.
- [ ] Re-run `scripts/run_honest_baseline.py` quarterly; track drift in `results/`.
- [ ] **Replace the hand-picked baseline universe with a point-in-time universe.** The Task 8 universe (SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM) is mega-caps that survived 2020-2024. Any momentum strategy will look heroic on this set. A real baseline needs a point-in-time universe — the `historical_universe` module in `research/` does this but isn't wired up. Either promote it (following the research-tree promotion bar below) or build the equivalent in the production path.
- [ ] **Add an end-of-backtest liquidation pass.** Most of the +646% headline in the Task 8 run is unrealized mark-to-market on still-open positions at end-of-2024 highs. `BacktestEngine` should close all open positions at the final bar (or report a separate realized-only return) so the headline number reflects what an operator could actually have captured.

## Research-tree promotion (only if a research/ module proves itself)

- [ ] For any `research/` module being considered for promotion, require: (1) ≥50-trade out-of-sample backtest, (2) statistical-significance check (permutation or FDR), (3) written hypothesis, (4) evidence the signal isn't already priced. Document in `research/<module>/PROMOTION.md`.

## Operational (only if scaling beyond solo paper)

- [ ] If running unattended for extended periods, re-evaluate which of the deleted operational scripts (kill switch is already kept) actually need to come back. Don't restore wholesale.
