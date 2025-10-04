# research/

This directory holds plausible-but-unvalidated quant work that was moved out of the production code path during the 2026-05 cleanup. **Nothing in here is imported by the production path, run by default tests, or trusted to produce signal.** Code here is preserved as ideas, not products.

## Contents

- `strategies/` — factor models, factor portfolios, factor screener, pairs trading.
- `engine/` — factor attribution, walk-forward validation, statistical tests, validated backtest.
- `utils/` — factor data pipeline, alpha-decay monitor, IC tracker, historical universe (point-in-time), extended-hours helpers, market-impact model.
- `factors/` — individual factor implementations (value, quality, momentum, low-vol, size, sentiment, growth, earnings, reversal, volatility, orthogonalization).
- `data/` — cross-asset (VIX/yield-curve/FX), feature store, point-in-time, tick data.
- `tests/` — tests that targeted the moved modules.

## Bringing a module back to production

Don't, unless you have:

1. A backtest with ≥50 trades, real slippage, and an out-of-sample period.
2. A statistical-significance check (e.g. permutation test, FDR-corrected).
3. A written hypothesis about why the signal should work.
4. Evidence that the signal isn't already priced into the symbols you trade.

Without those, the module stays here.
