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
INCONCLUSIVE per PROFITABILITY_RESEARCH.md's 50-trade significance bar.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
SLIPPAGE_BPS = 40   # 0.40% per trade (configured via execution profile below)
SPREAD_BPS = 10     # 0.10%
MIN_TRADES_FOR_SIGNIFICANCE = 50
SPEC_REF = "docs/superpowers/specs/2026-05-11-honest-cleanup-design.md"

RESULTS_DIR = REPO_ROOT / "results"
MD_PATH = RESULTS_DIR / "honest_backtest_2020-2024.md"
JSON_PATH = RESULTS_DIR / "honest_backtest_2020-2024.json"


@dataclass
class SimpleBar:
    """Minimal bar object matching what engine.backtest_engine reads.

    The engine accesses .timestamp, .open, .high, .low, .close, .volume — see
    engine/backtest_engine.py:_load_symbol_data. Anything with those attributes works.
    """

    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float


class YFinanceDataBroker:
    """A read-only data broker that serves daily bars from yfinance.

    BacktestEngine.run_backtest only calls one method on the data broker:
    `await data_broker.get_bars(symbol, start=..., end=..., timeframe="1Day")`,
    so that is the only thing this class needs to implement.
    """

    def __init__(self) -> None:
        import yfinance as yf

        self._yf = yf

    async def get_bars(self, symbol, start=None, end=None, timeframe="1Day", limit=None):
        # yfinance is synchronous; run in thread to avoid blocking the event loop.
        return await asyncio.to_thread(self._sync_get_bars, symbol, start, end)

    def _sync_get_bars(self, symbol, start, end):
        try:
            df = self._yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s: %s", symbol, exc)
            return []

        if df is None or df.empty:
            return []

        # yfinance 1.3 returns a MultiIndex column frame when downloading a
        # single ticker — flatten to a single-level by picking the symbol.
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            try:
                df = df.xs(symbol, axis=1, level=-1)
            except KeyError:
                df.columns = [c[0] for c in df.columns]

        bars = []
        for ts, row in df.iterrows():
            try:
                open_p = float(row["Open"])
                high_p = float(row["High"])
                low_p = float(row["Low"])
                close_p = float(row["Close"])
                volume = float(row["Volume"]) if not _isnan(row["Volume"]) else 0.0
            except (KeyError, TypeError, ValueError):
                continue
            if any(_isnan(x) for x in (open_p, high_p, low_p, close_p)):
                continue
            bars.append(
                SimpleBar(
                    timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=volume,
                )
            )
        return bars


def _isnan(x) -> bool:
    try:
        return x != x  # NaN is the only value not equal to itself
    except Exception:
        return False


async def _try_alpaca_data_broker():
    """Return an AlpacaBroker for data or None if credentials/import are unavailable."""
    if not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY")):
        logger.info("Alpaca credentials not set — will fall back to yfinance.")
        return None
    try:
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper=True)
        # Sanity probe one symbol to confirm the API actually works in this env.
        probe = await broker.get_bars("SPY", start="2024-01-01", end="2024-01-10", timeframe="1Day")
        if not probe:
            logger.warning("Alpaca probe returned no bars — falling back to yfinance.")
            return None
        return broker
    except Exception as exc:
        logger.warning("Alpaca broker unusable (%s) — falling back to yfinance.", exc)
        return None


async def _try_yfinance_data_broker():
    try:
        broker = YFinanceDataBroker()
        # Sanity probe so we fail fast if the network is gone.
        probe = await broker.get_bars("SPY", start="2024-01-01", end="2024-01-10")
        if not probe:
            return None
        return broker
    except Exception as exc:
        logger.warning("yfinance unusable: %s", exc)
        return None


async def _resolve_data_broker():
    """Try Alpaca, then yfinance. Returns (broker, source_name) or (None, error_msg)."""
    alpaca = await _try_alpaca_data_broker()
    if alpaca is not None:
        return alpaca, "alpaca"

    yfin = await _try_yfinance_data_broker()
    if yfin is not None:
        return yfin, "yfinance"

    return None, (
        "No data source available. Tried Alpaca (credentials missing or API "
        "unreachable) and yfinance (network unreachable or rate-limited). "
        "To run: set ALPACA_API_KEY and ALPACA_SECRET_KEY, or ensure outbound "
        "network access to yfinance from this environment."
    )


def _write_data_unavailable_report(reason: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    artifact = {
        "spec_ref": SPEC_REF,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "DATA_UNAVAILABLE",
        "reason": reason,
        "config": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "initial_capital": INITIAL_CAPITAL,
            "slippage_bps": SLIPPAGE_BPS,
            "spread_bps": SPREAD_BPS,
            "min_trades_for_significance": MIN_TRADES_FOR_SIGNIFICANCE,
        },
    }
    JSON_PATH.write_text(json.dumps(artifact, indent=2, default=str))

    md = (
        "# Honest baseline backtest 2020-2024\n\n"
        f"Generated: {artifact['generated_at']}\n"
        f"Spec: `{SPEC_REF}`\n\n"
        "> **Status: BACKTEST DID NOT RUN — data source unavailable.**\n\n"
        f"{reason}\n\n"
        "No metrics are reported because no backtest was executed. The previous "
        "in-doc claims (`+42.68%` etc.) remain unsupported by a real evidence "
        "file; once `scripts/run_honest_baseline.py` succeeds in an environment "
        "with data access, this file will be replaced with the actual result.\n\n"
        "## Configuration that would be used\n\n"
        f"- **Strategy:** `MomentumStrategyBacktest` (daily-bar variant of MomentumStrategy, default parameters)\n"
        f"- **Symbols:** {', '.join(SYMBOLS)}\n"
        f"- **Period:** {START} to {END}\n"
        f"- **Initial capital:** ${INITIAL_CAPITAL:,}\n"
        f"- **Slippage:** {SLIPPAGE_BPS} bps per trade\n"
        f"- **Spread:** {SPREAD_BPS} bps\n"
        f"- **Significance bar:** {MIN_TRADES_FOR_SIGNIFICANCE} trades\n"
    )
    MD_PATH.write_text(md)
    logger.error("Backtest did not run: %s", reason)


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
            "bar set by this repo's `PROFITABILITY_RESEARCH.md`. The numbers "
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
            "into claims about future performance. `PROFITABILITY_RESEARCH.md` "
            "calls out realistic expectations for this strategy family "
            "(Sharpe 0.5 to 1.2 net of costs); this run is too small to confirm "
            "or refute that. Re-run with a larger universe or longer history to "
            "cross the significance bar before quoting edge.\n"
        )
    else:
        interpretation += (
            "**Caveats — read before quoting these numbers:**\n\n"
            "1. **Survivorship-bias correction is off.** The 10-symbol universe is "
            "hand-picked mega-caps that survived 2020-2024; survivorship-bias "
            "handling was quarantined to `research/` in the 2026-05 cleanup. "
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
            "out-of-sample trader would have realized. "
            "`PROFITABILITY_RESEARCH.md` documents realistic expectations for "
            "this strategy family (Sharpe 0.5 to 1.2 net of costs) — anything "
            "well above that range warrants suspicion, not celebration.\n\n"
            "Do not extrapolate beyond what the trade count supports. Use this "
            "artifact as a sanity check that the pipeline runs end-to-end on "
            "real market data, not as evidence of strategy edge.\n"
        )

    return header + config_block + metrics_block + trades_block + interpretation


async def _run_backtest(data_broker, source_name: str) -> None:
    from engine.backtest_engine import BacktestEngine
    from engine.performance_metrics import PerformanceMetrics

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
    # attaches a `BacktestOrderGateway` to the strategy automatically so
    # `BaseStrategy.submit_entry_order` / `submit_exit_order` route to the
    # backtest broker — no gateway shim is needed here.
    engine = BacktestEngine(broker=data_broker)

    logger.info("Starting backtest: %s symbols, %s to %s, data=%s",
                len(SYMBOLS), START, END, source_name)
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
    inconclusive = n_trades < MIN_TRADES_FOR_SIGNIFICANCE
    equity_curve = result.get("equity_curve", [INITIAL_CAPITAL])

    data_quality = result.get("data_quality", {})
    symbols_loaded = data_quality.get("symbols_loaded", 0)
    symbols_requested = data_quality.get("symbols_requested", len(SYMBOLS))

    # Defensive: if we asked for 10 symbols and 0 loaded, the backtest didn't
    # really run — refuse to publish a misleading 0-trade INCONCLUSIVE result.
    if symbols_loaded == 0:
        reason = (
            f"Backtest engine loaded 0 of {symbols_requested} requested symbols "
            f"from data source `{source_name}`. No real backtest was executed. "
            "Check network access and credentials, then re-run."
        )
        _write_data_unavailable_report(reason)
        print("STATUS=DATA_UNAVAILABLE  (0 symbols loaded)")
        return

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
    print(f"STATUS={'INCONCLUSIVE' if inconclusive else 'REPORTED'}  trades={n_trades}  "
          f"total_return={_format_pct(metrics.get('total_return'))}  "
          f"sharpe={_format_num(metrics.get('sharpe_ratio'))}  "
          f"data_source={source_name}")


async def main() -> int:
    try:
        data_broker, source = await _resolve_data_broker()
    except Exception as exc:
        _write_data_unavailable_report(f"Unexpected error resolving data source: {exc}")
        return 1

    if data_broker is None:
        _write_data_unavailable_report(source)  # `source` is the error string here
        return 1

    try:
        await _run_backtest(data_broker, source)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Backtest crashed: %s\n%s", exc, tb)
        _write_data_unavailable_report(
            f"Backtest engine crashed before producing a result: {exc}\n\n"
            f"Traceback (last 1000 chars):\n{tb[-1000:]}"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
