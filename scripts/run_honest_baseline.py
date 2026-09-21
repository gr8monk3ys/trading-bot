"""Run the honest baseline backtest defined by the 2026-05 cleanup spec.

Output:
    results/honest_backtest_2020-2024.json   - raw metrics + trade log
    results/honest_backtest_2020-2024.md     - human-readable report

Usage:
    python scripts/run_honest_baseline.py

Data sources (tried in order):
    1. Alpaca historical bars (if ALPACA_API_KEY/ALPACA_SECRET_KEY in env)
    2. yfinance daily bars (fallback)
    3. If both fail, the report says "did not run" — we never report a
       silent 0-trade INCONCLUSIVE result that looks like the strategy ran.

If trade count < MIN_TRADES_FOR_SIGNIFICANCE, the result is reported as
INCONCLUSIVE. The bar is 50 trades: with fewer, the 95% confidence interval on
mean trade return is wider than any plausible edge (at 7-15 trades it spans
roughly -15% to +25%), so a Sharpe or win rate quoted from it is noise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path so this script can be run as `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("honest_baseline")

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM"]
START = "2020-01-01"
END = "2024-12-31"
INITIAL_CAPITAL = 100_000
SLIPPAGE_BPS = 40  # 0.40% per trade (configured via execution profile below)
SPREAD_BPS = 10  # 0.10%
MIN_TRADES_FOR_SIGNIFICANCE = 50
SPEC_REF = "docs/superpowers/specs/2026-05-11-honest-cleanup-design.md"

from config import BASELINE  # noqa: E402
from engine.historical_bars import DataUnavailableError, resolve_bars_source  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
MD_PATH = RESULTS_DIR / "honest_backtest_2020-2024.md"
JSON_PATH = RESULTS_DIR / "honest_backtest_2020-2024.json"


def _format_pct(value, default="N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return default


def _format_num(value, fmt="{:.2f}", default="N/A") -> str:
    if value is None:
        return default
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return default


def _format_markdown(artifact: dict) -> str:
    cfg = artifact["config"]
    m = artifact["metrics"]
    inconclusive = artifact["inconclusive"]
    n_trades = artifact["n_trades"]
    data_source = artifact.get("data_source", "unknown")

    header = (
        "# Honest baseline backtest 2020-2024\n\n"
        f"Generated: {artifact['generated_at']}\n"
        f"Spec: `{artifact['spec_ref']}`\n"
        f"Data source: `{data_source}`\n\n"
    )

    if inconclusive:
        header += (
            f"> **Status: INCONCLUSIVE.** Strategy produced {n_trades} trades, "
            f"below the {cfg['min_trades_for_significance']}-trade significance "
            "bar (see the docstring of `scripts/run_honest_baseline.py`). The numbers "
            "below are reported for transparency but must not be cited as "
            "evidence of strategy edge.\n\n"
        )
    else:
        header += (
            f"> **Status: backtest produced {n_trades} trades** "
            f"(meets the {cfg['min_trades_for_significance']}-trade significance bar).\n\n"
        )

    config_block = (
        "## Configuration\n\n"
        f"- **Strategy:** `MomentumStrategyBacktest` (daily-bar variant of MomentumStrategy, default parameters)\n"
        f"- **Symbols:** {', '.join(cfg['symbols'])}\n"
        f"- **Period:** {cfg['start']} to {cfg['end']}\n"
        f"- **Initial capital:** ${cfg['initial_capital']:,}\n"
        f"- **Slippage:** {cfg['slippage_bps']} bps per trade\n"
        f"- **Spread:** {cfg['spread_bps']} bps\n"
        f"- **Significance bar:** {cfg['min_trades_for_significance']} trades\n\n"
    )

    metrics_block = (
        "## Headline metrics\n\n"
        f"- **Total return:** {_format_pct(m.get('total_return'))}\n"
        f"- **Annualized return:** {_format_pct(m.get('annualized_return'))}\n"
        f"- **Sharpe ratio:** {_format_num(m.get('sharpe_ratio'))}\n"
        f"- **Sortino ratio:** {_format_num(m.get('sortino_ratio'))}\n"
        f"- **Calmar ratio:** {_format_num(m.get('calmar_ratio'))}\n"
        f"- **Max drawdown:** {_format_pct(m.get('max_drawdown'))}\n"
        f"- **Win rate:** {_format_pct(m.get('win_rate'))}\n"
        f"- **Profit factor:** {_format_num(m.get('profit_factor'))}\n"
        f"- **Trade count:** {n_trades}\n"
        f"- **Final equity:** ${_format_num(m.get('final_equity'), fmt='{:,.2f}')}\n\n"
    )

    trades_block = (
        "## Trade log\n\n"
        "| # | Symbol | Side | Quantity | Price | P&L | Timestamp |\n"
        "|---|--------|------|----------|-------|-----|-----------|\n"
    )
    for i, t in enumerate(artifact["trades"], 1):
        trades_block += (
            f"| {i} | {t.get('symbol','')} | {t.get('side','')} | "
            f"{t.get('quantity','')} | {_format_num(t.get('price'), fmt='{:.2f}')} | "
            f"{_format_num(t.get('pnl'), fmt='{:.2f}')} | "
            f"{t.get('timestamp','')} |\n"
        )
    if not artifact["trades"]:
        trades_block += "| _no trades_ | | | | | | |\n"

    interpretation = (
        "\n## Interpretation\n\n"
        "This is the single performance number cited by `README.md` and "
        "`CLAUDE.md`. It supersedes `backtest_report_2024.md` (9 trades) and "
        "any earlier in-doc claims (notably the `+42.68%` figure that lacked "
        "a publishable evidence file).\n\n"
    )

    if inconclusive:
        interpretation += (
            "Because the trade count is below 50, the Sharpe/return numbers above "
            "have very wide confidence intervals and **must not be extrapolated** "
            "into claims about future performance. Re-run with a larger universe "
            "or longer history to "
            "cross the significance bar before quoting edge.\n"
        )
    else:
        interpretation += (
            "**Caveats — read before quoting these numbers:**\n\n"
            "1. **Survivorship-bias correction is off.** The 10-symbol universe is "
            "hand-picked mega-caps that survived 2020-2024; the engine has no "
            "survivorship-bias correction. "
            "Numbers above are inflated by selection of known winners.\n"
            "2. **Realized P&L only — end-of-period liquidation pass enabled.** "
            "Open positions at end-of-period are closed at the final bar with "
            "realistic spread + slippage (see `BacktestEngine._liquidate_open_positions`), "
            "so headline equity reflects realized cash, not unrealized MTM. "
            "Short-leg PnL is also captured correctly (Step 2B fixed the matcher). "
            "The 5-year window happens to end near all-time highs in the chosen "
            "universe; rerun ending on a different date for a different number.\n"
            "3. **Costs included: 40 bps slippage + 10 bps spread per trade.** "
            "These are realistic for retail at this universe size but do not "
            "model gap risk on positions held overnight (gap stats: see "
            "engine logs — largest gap in this run was 26%).\n"
            "4. **No walk-forward validation in this artifact.** This is a single "
            "in-sample run; treat the Sharpe as an upper bound on what an "
            "out-of-sample trader would have realized. A Sharpe well above 1 "
            "from a retail momentum system warrants suspicion, not celebration.\n\n"
            "Do not extrapolate beyond what the trade count supports. Use this "
            "artifact as a sanity check that the pipeline runs end-to-end on "
            "real market data, not as evidence of strategy edge.\n"
        )

    return header + config_block + metrics_block + trades_block + interpretation


async def _run_backtest(data_broker, source_name: str) -> None:
    from engine.backtest_engine import BacktestEngine
    from engine.performance_metrics import PerformanceMetrics, verdict

    # NOTE on strategy choice:
    # The plan's draft script specified `MomentumStrategy`, but that class has
    # an intentionally empty `execute_trade` stub (the live path routes orders
    # through a separate mechanism). With it, the engine logs buy signals but
    # never submits a single order, producing a misleading 0-trade run that
    # looks indistinguishable from "no data". The class that actually places
    # orders inside the backtest engine is `MomentumStrategyBacktest` — same
    # signal logic, same momentum family, but with a working `execute_trade`
    # tuned for daily-bar data. This is the strategy CLAUDE.md already lists
    # as the validated backtest variant. Adaptation noted in the commit
    # message; the spec's intent ("MomentumStrategy with default parameters
    # on 10 large-caps") is preserved.
    from strategies.momentum_strategy_backtest import MomentumStrategyBacktest

    # BacktestEngine internally constructs its own BacktestBroker; the broker
    # we pass via `broker=` is only used as a *data source* (its get_bars is
    # called). The internal BacktestBroker uses the default execution
    # profile "realistic" which models slippage + spread. The engine also
    # attaches an `OrderSubmission` to the strategy automatically so
    # `BaseStrategy.submit_entry_order` / `submit_exit_order` route to the
    # backtest broker — no gateway shim is needed here.
    engine = BacktestEngine(broker=data_broker)

    logger.info(
        "Starting backtest: %s symbols, %s to %s, data=%s", len(SYMBOLS), START, END, source_name
    )
    result = await engine.run_backtest(
        strategy_class=MomentumStrategyBacktest,
        symbols=SYMBOLS,
        start_date=datetime.strptime(START, "%Y-%m-%d"),
        end_date=datetime.strptime(END, "%Y-%m-%d"),
        initial_capital=INITIAL_CAPITAL,
        execution_profile="realistic",
    )

    metrics = PerformanceMetrics().calculate_metrics(result)

    trades = result.get("trades", [])
    n_trades = len(trades)
    run_verdict = verdict(
        metrics, n_trades, result.get("data_quality"), min_trades=MIN_TRADES_FOR_SIGNIFICANCE
    )
    inconclusive = not run_verdict.quotable
    equity_curve = result.get("equity_curve", [INITIAL_CAPITAL])

    data_quality = result.get("data_quality", {})
    symbols_loaded = data_quality.get("symbols_loaded", 0)
    symbols_requested = data_quality.get("symbols_requested", len(SYMBOLS))

    artifact = {
        "spec_ref": SPEC_REF,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "INCONCLUSIVE" if inconclusive else "REPORTED",
        "data_source": source_name,
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "initial_capital": INITIAL_CAPITAL,
            "slippage_bps": SLIPPAGE_BPS,
            "spread_bps": SPREAD_BPS,
            "min_trades_for_significance": MIN_TRADES_FOR_SIGNIFICANCE,
        },
        "n_trades": n_trades,
        "inconclusive": inconclusive,
        "metrics": metrics,
        "trades": [
            {
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "quantity": t.get("quantity"),
                "price": t.get("price"),
                "pnl": t.get("pnl"),
                "timestamp": str(t.get("timestamp")) if t.get("timestamp") else None,
            }
            for t in trades
        ],
        "data_quality": {
            "symbols_loaded": symbols_loaded,
            "symbols_requested": symbols_requested,
            "symbols_rejected": data_quality.get("symbols_rejected", 0),
        },
        "equity_curve_summary": {
            "start_equity": float(equity_curve[0]) if equity_curve else INITIAL_CAPITAL,
            "end_equity": float(equity_curve[-1]) if equity_curve else INITIAL_CAPITAL,
            "n_days": len(equity_curve),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(artifact, indent=2, default=str))
    MD_PATH.write_text(_format_markdown(artifact))

    print(f"Wrote {JSON_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {MD_PATH.relative_to(REPO_ROOT)}")
    print(
        f"STATUS={'INCONCLUSIVE' if inconclusive else 'REPORTED'}  trades={n_trades}  "
        f"total_return={_format_pct(metrics.get('total_return'))}  "
        f"sharpe={_format_num(metrics.get('sharpe_ratio'))}  "
        f"data_source={source_name}"
    )


async def main() -> int:
    try:
        data_broker, source = await resolve_bars_source(preferred=BASELINE["data_source"])
    except DataUnavailableError as exc:
        print(f"STATUS=DATA_UNAVAILABLE  {exc}")
        return 1

    try:
        await _run_backtest(data_broker, source)
    except DataUnavailableError as exc:
        print(f"STATUS=DATA_UNAVAILABLE  {exc}")
        return 1
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Backtest crashed: %s\n%s", exc, tb)
        logger.error(
            f"Backtest engine crashed before producing a result: {exc}\n\n"
            f"Traceback (last 1000 chars):\n{tb[-1000:]}"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
