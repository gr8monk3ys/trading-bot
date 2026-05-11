"""Run a survivorship-bias-free ETF baseline backtest.

Output:
    results/etf_baseline_2020-2024.json   - raw metrics + trade log + benchmarks
    results/etf_baseline_2020-2024.md     - human-readable report

Usage:
    python scripts/run_etf_baseline.py

Why ETFs:
    The existing honest baseline (`scripts/run_honest_baseline.py`) trades
    10 hand-picked mega-caps (NVDA, AAPL, TSLA, MSFT, GOOGL, AMZN, META, JPM,
    SPY, QQQ) — names a 2026 retrospective would obviously pick. The +646% /
    Sharpe 1.36 from that run is dominated by survivorship bias.

    This script runs the same strategy on a broad-market ETF universe:
    SPY, QQQ, IWM, EFA. ETFs cannot be delisted and cannot be selection-biased.
    If the strategy can't beat SPY buy-and-hold on this universe, it has no
    real edge.

Data source: yfinance (with optional Alpaca fallback identical to honest
baseline). If neither is available the script writes a DATA_UNAVAILABLE
report rather than silently producing a 0-trade artifact.

If trade count < MIN_TRADES_FOR_SIGNIFICANCE, the result is reported as
INCONCLUSIVE per `PROFITABILITY_RESEARCH.md`'s 50-trade significance bar.
A 0-trade result on this universe is itself an honest finding ("strategy
filter doesn't fire on broad indices") and is reported as such.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
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
logger = logging.getLogger("etf_baseline")

SYMBOLS = ["SPY", "QQQ", "IWM", "EFA"]
START = "2020-01-01"
END = "2024-12-31"
INITIAL_CAPITAL = 100_000
SLIPPAGE_BPS = 40   # 0.40% per trade
SPREAD_BPS = 10     # 0.10%
MIN_TRADES_FOR_SIGNIFICANCE = 50
SPEC_REF = "docs/superpowers/specs/2026-05-11-honest-cleanup-design.md"

# Benchmarks to compute and compare against the strategy.
BENCHMARK_SYMBOLS = ["SPY", "QQQ"]

# Reference numbers from the existing hand-picked baseline for the comparison
# section in the markdown. These come from `results/honest_backtest_2020-2024.md`.
HAND_PICKED_BASELINE = {
    "universe": "10 hand-picked mega-caps (SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM)",
    "total_return": 6.46,        # +646%
    "sharpe": 1.36,
    # Stored as a positive magnitude to match PerformanceMetrics' convention
    # (strategy `max_drawdown` is the positive fraction of peak-to-trough loss).
    "max_dd": 0.4696,
    "n_trades": 102,
}

RESULTS_DIR = REPO_ROOT / "results"
MD_PATH = RESULTS_DIR / "etf_baseline_2020-2024.md"
JSON_PATH = RESULTS_DIR / "etf_baseline_2020-2024.json"


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
        return x != x
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
        probe = await broker.get_bars("SPY", start="2024-01-01", end="2024-01-10")
        if not probe:
            return None
        return broker
    except Exception as exc:
        logger.warning("yfinance unusable: %s", exc)
        return None


async def _resolve_data_broker():
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


def _compute_buy_and_hold(
    symbol: str,
    start: str,
    end: str,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """Compute buy-and-hold metrics for `symbol` over [start, end] via yfinance.

    Returns a dict with total_return (fraction), cagr, sharpe (rf=0), max_dd
    (negative fraction), final_equity, and n_days. Missing data returns None
    fields rather than raising — benchmarks are nice-to-have, not load-bearing.
    """
    try:
        import yfinance as yf
    except Exception as exc:
        logger.warning("yfinance import failed for benchmark %s: %s", symbol, exc)
        return _empty_benchmark(symbol, start, end, reason=f"yfinance import failed: {exc}")

    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as exc:
        logger.warning("yfinance fetch failed for benchmark %s: %s", symbol, exc)
        return _empty_benchmark(symbol, start, end, reason=f"yfinance fetch failed: {exc}")

    if df is None or df.empty:
        return _empty_benchmark(symbol, start, end, reason="no data")

    # Single ticker download can come back as MultiIndex columns in newer yfinance.
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        try:
            df = df.xs(symbol, axis=1, level=-1)
        except KeyError:
            df.columns = [c[0] for c in df.columns]

    closes = df["Close"].squeeze().dropna()
    if len(closes) < 2:
        return _empty_benchmark(symbol, start, end, reason="<2 closes")

    first = float(closes.iloc[0])
    last = float(closes.iloc[-1])
    if first <= 0:
        return _empty_benchmark(symbol, start, end, reason="non-positive start price")

    total_return = (last / first) - 1.0
    final_equity = initial_capital * (last / first)

    # CAGR from the period spanned by the actual close dates.
    try:
        first_ts = closes.index[0]
        last_ts = closes.index[-1]
        years = max((last_ts - first_ts).days / 365.25, 1e-9)
    except Exception:
        years = max(len(closes) / 252.0, 1e-9)
    cagr = ((last / first) ** (1.0 / years)) - 1.0 if last > 0 else None

    # Daily returns, Sharpe (rf=0, annualized), max drawdown.
    daily_returns = closes.pct_change().dropna()
    if len(daily_returns) >= 2:
        mean = float(daily_returns.mean())
        std = float(daily_returns.std(ddof=1))
        sharpe = (mean / std) * math.sqrt(252) if std > 0 else None
    else:
        sharpe = None

    running_max = closes.cummax()
    drawdowns = (closes / running_max) - 1.0
    max_dd = float(drawdowns.min()) if len(drawdowns) else None

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": final_equity,
        "n_days": int(len(closes)),
        "reason": None,
    }


def _empty_benchmark(symbol: str, start: str, end: str, reason: str) -> dict:
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "total_return": None,
        "cagr": None,
        "sharpe": None,
        "max_drawdown": None,
        "final_equity": None,
        "n_days": 0,
        "reason": reason,
    }


def _abs_or_none(value):
    """abs() that survives None — used to normalize signed drawdown values for display."""
    if value is None:
        return None
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return None


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
        "# ETF baseline 2020-2024 — survivorship-bias-free test of strategy edge\n\n"
        f"Generated: {artifact['generated_at']}\n"
        f"Spec: `{SPEC_REF}`\n\n"
        "> **Status: BACKTEST DID NOT RUN — data source unavailable.**\n\n"
        f"{reason}\n\n"
        "No metrics are reported because no backtest was executed. Once the\n"
        "script succeeds in an environment with data access, this file will be\n"
        "replaced with the actual result.\n\n"
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


def _format_markdown(artifact: dict) -> str:
    cfg = artifact["config"]
    m = artifact["metrics"]
    inconclusive = artifact["inconclusive"]
    n_trades = artifact["n_trades"]
    data_source = artifact.get("data_source", "unknown")
    benchmarks = artifact.get("benchmarks", {})

    header = (
        "# ETF baseline 2020-2024 — survivorship-bias-free test of strategy edge\n\n"
        f"Generated: {artifact['generated_at']}\n"
        f"Spec: `{artifact['spec_ref']}`\n"
        f"Data source: `{data_source}`\n\n"
    )

    # 0-trade is its own honest finding; flag it loudly but distinctly from
    # the generic "below 50 trades" inconclusive case.
    if n_trades == 0:
        header += (
            "> **Status: INCONCLUSIVE — 0 trades.** The momentum filter never fired on the\n"
            "> ETF universe over 2020-2024. This is an honest finding in its own right: the\n"
            "> strategy's entry rules, tuned on single-stock volatility, do not produce\n"
            "> signals on broad indices like SPY/QQQ/IWM/EFA. It does NOT mean the strategy\n"
            "> has edge — it means we cannot tell from this run whether it does.\n\n"
        )
    elif inconclusive:
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

    purpose = (
        "## Purpose\n\n"
        "This backtest exists to disambiguate **\"the strategy has edge\"** from\n"
        "**\"the universe was hand-picked winners\"**. The existing\n"
        "`results/honest_backtest_2020-2024.md` posts +646% / Sharpe 1.36 on\n"
        "10 mega-caps that any 2026 retrospective would obviously pick. That\n"
        "number is dominated by survivorship bias.\n\n"
        "ETFs cannot be delisted and cannot be selection-biased. SPY/QQQ/IWM/EFA\n"
        "cover US large-cap, US tech, US small-cap, and developed international\n"
        "equity — broad market exposure with zero look-ahead. If the strategy\n"
        "can't beat SPY buy-and-hold on this universe, it has no real edge.\n\n"
    )

    config_block = (
        "## Configuration\n\n"
        f"- **Strategy:** `MomentumStrategyBacktest` (daily-bar variant of MomentumStrategy, default parameters)\n"
        f"- **Symbols:** {', '.join(cfg['symbols'])} (US large-cap, US tech, US small-cap, developed-intl)\n"
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

    # Comparison section — the whole reason this script exists.
    spy_bh = benchmarks.get("SPY", {})
    qqq_bh = benchmarks.get("QQQ", {})

    comparison_block = (
        "\n## Comparison: ETF baseline vs hand-picked vs buy-and-hold\n\n"
        "| Run | Universe | Total return | Sharpe | Max DD | Trades |\n"
        "|-----|----------|--------------|--------|--------|--------|\n"
        f"| **ETF baseline (this run)** | {', '.join(cfg['symbols'])} | "
        f"{_format_pct(m.get('total_return'))} | "
        f"{_format_num(m.get('sharpe_ratio'))} | "
        f"{_format_pct(m.get('max_drawdown'))} | "
        f"{n_trades} |\n"
        f"| Hand-picked baseline (survivor-biased) | {HAND_PICKED_BASELINE['universe']} | "
        f"{_format_pct(HAND_PICKED_BASELINE['total_return'])} | "
        f"{_format_num(HAND_PICKED_BASELINE['sharpe'])} | "
        f"{_format_pct(HAND_PICKED_BASELINE['max_dd'])} | "
        f"{HAND_PICKED_BASELINE['n_trades']} |\n"
        f"| SPY buy-and-hold | SPY | "
        f"{_format_pct(spy_bh.get('total_return'))} | "
        f"{_format_num(spy_bh.get('sharpe'))} | "
        f"{_format_pct(_abs_or_none(spy_bh.get('max_drawdown')))} | "
        "1 |\n"
        f"| QQQ buy-and-hold | QQQ | "
        f"{_format_pct(qqq_bh.get('total_return'))} | "
        f"{_format_num(qqq_bh.get('sharpe'))} | "
        f"{_format_pct(_abs_or_none(qqq_bh.get('max_drawdown')))} | "
        "1 |\n\n"
        "Buy-and-hold numbers are computed in this script via yfinance for the\n"
        "same period and capital, using daily close-to-close returns and rf=0\n"
        "for the Sharpe (matching the strategy convention). The hand-picked row\n"
        "is copied from `results/honest_backtest_2020-2024.md`.\n\n"
    )

    interpretation = _interpret(m, benchmarks, n_trades, inconclusive)

    return (
        header
        + purpose
        + config_block
        + metrics_block
        + trades_block
        + comparison_block
        + interpretation
    )


def _interpret(metrics: dict, benchmarks: dict, n_trades: int, inconclusive: bool) -> str:
    """Produce an honest interpretation paragraph based on the actual numbers.

    The directional comparison (strategy vs SPY/QQQ buy-and-hold) is the whole
    point of the script and is reported regardless of trade count. When trades
    are below the significance bar, we report it but lead with the small-sample
    caveat so nobody quotes the directional finding as if it were proven.
    """
    spy_ret = (benchmarks.get("SPY") or {}).get("total_return")
    qqq_ret = (benchmarks.get("QQQ") or {}).get("total_return")
    strat_ret = metrics.get("total_return")

    header = "## Interpretation\n\n"

    if n_trades == 0:
        return header + (
            "**0 trades on the ETF universe is the finding.** The momentum filter\n"
            "(RSI/MACD/ADX thresholds tuned for single-stock volatility) never\n"
            "produced an entry signal on SPY/QQQ/IWM/EFA over five years. This\n"
            "is consistent with — but does not prove — the hypothesis that the\n"
            "strategy's edge on the hand-picked baseline came from picking\n"
            "single-stock winners rather than from any timing skill on broad\n"
            "indices.\n\n"
            "What this run does NOT tell us:\n\n"
            "- Whether the strategy would have edge on a random sample of S&P 500\n"
            "  members (i.e. a true survivor-bias-free single-stock universe).\n"
            "- Whether the strategy's entry filter would fire on more volatile\n"
            "  sector ETFs (XLK, XLF, ARKK) or factor ETFs (MTUM, QUAL).\n\n"
            "Next step (if continuing): retune the entry thresholds for lower-\n"
            "volatility instruments OR widen the universe to random S&P 500 members.\n"
        )

    bullets = []

    if inconclusive:
        bullets.append(
            f"**Trade count ({n_trades}) is below the 50-trade significance bar.**\n"
            "  The directional comparison below is reported because that is the whole\n"
            "  point of this script — but treat it as a hint, not as evidence. Sharpe\n"
            "  confidence intervals at 38 trades are very wide; the strategy could be\n"
            "  underperforming SPY by chance alone."
        )

    if strat_ret is None or spy_ret is None:
        bullets.append(
            "Some headline values are unavailable — the comparison below is partial."
        )
    else:
        if strat_ret < spy_ret:
            bullets.append(
                "**Directional finding: the strategy underperformed SPY buy-and-hold on\n"
                "  a bias-free universe.** This is the most damning bucket the script\n"
                "  can land in. The +646% on the hand-picked baseline is consistent\n"
                "  with riding survivors, not with possessing timing edge. Treat the\n"
                "  hand-picked Sharpe as a number to be explained away, not a number\n"
                "  to deploy capital on."
            )
        elif qqq_ret is not None and strat_ret > qqq_ret:
            bullets.append(
                "**Directional finding: the strategy beat QQQ buy-and-hold on a\n"
                "  bias-free universe.** This is surprising — a vanilla momentum\n"
                "  strategy outperforming a tech-heavy passive index over the post-\n"
                "  COVID rally is not a common result and warrants investigation.\n"
                "  Possibilities: (a) genuine edge from trailing stops avoiding the\n"
                "  2022 drawdown, (b) a bug inflating equity, (c) the cost model is\n"
                "  too generous. Investigate before celebrating."
            )
        else:
            bullets.append(
                "**Directional finding: the strategy landed between SPY buy-and-hold\n"
                "  and the hand-picked baseline.** This is the most informative bucket:\n"
                "  evidence of mild timing edge — beating the broad market without the\n"
                "  survivor-bias tailwind of the hand-picked universe — while showing\n"
                "  that most of the hand-picked baseline's outperformance was selection\n"
                "  bias, not strategy alpha. Reasonable next step: characterize where\n"
                "  the edge comes from (regime filter? trailing stops? specific\n"
                "  symbols?) before scaling up."
            )

    bullets.append(
        "**Caveats — read before quoting these numbers:**"
    )
    bullets.append(
        "- ETFs are not the *only* survivor-bias-free universe. A random sample\n"
        "  of S&P 500 members at each point in time would be stronger; this run\n"
        "  is a cheap-to-produce first cut. Follow-up item in `TODO.md`."
    )
    bullets.append(
        "- 5 years of daily data on 4 instruments is a small sample even when\n"
        "  the in-strategy trade count crosses 50. Don't extrapolate Sharpe\n"
        "  confidence intervals from this run alone."
    )
    bullets.append(
        "- Costs included: 40 bps slippage + 10 bps spread per trade. ETFs trade\n"
        "  tighter than that in practice, so per-trade cost drag is if anything\n"
        "  overstated here, not understated."
    )
    bullets.append(
        "- Realized P&L only — open positions at end-of-period are liquidated at\n"
        "  the final bar with the same spread + slippage as any other trade\n"
        "  (`BacktestEngine._liquidate_open_positions`). Headline equity reflects\n"
        "  realized cash, not unrealized MTM."
    )

    body = "\n\n".join(bullets) + "\n"
    return header + body


async def _run_backtest(data_broker, source_name: str) -> None:
    from engine.backtest_engine import BacktestEngine
    from engine.performance_metrics import PerformanceMetrics
    # See run_honest_baseline.py for why we use the *Backtest variant —
    # MomentumStrategy.execute_trade is a no-op in the engine path.
    from strategies.momentum_strategy_backtest import MomentumStrategyBacktest

    engine = BacktestEngine(broker=data_broker)

    logger.info(
        "Starting ETF backtest: %s symbols, %s to %s, data=%s",
        len(SYMBOLS), START, END, source_name,
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
    inconclusive = n_trades < MIN_TRADES_FOR_SIGNIFICANCE
    equity_curve = result.get("equity_curve", [INITIAL_CAPITAL])

    data_quality = result.get("data_quality", {})
    symbols_loaded = data_quality.get("symbols_loaded", 0)
    symbols_requested = data_quality.get("symbols_requested", len(SYMBOLS))

    if symbols_loaded == 0:
        reason = (
            f"Backtest engine loaded 0 of {symbols_requested} requested symbols "
            f"from data source `{source_name}`. No real backtest was executed. "
            "Check network access and credentials, then re-run."
        )
        _write_data_unavailable_report(reason)
        print("STATUS=DATA_UNAVAILABLE  (0 symbols loaded)")
        return

    # Compute benchmarks in parallel-ish (sequentially is fine — these are
    # fast yfinance calls and the script is one-shot).
    logger.info("Computing buy-and-hold benchmarks: %s", BENCHMARK_SYMBOLS)
    benchmarks = {}
    for bench in BENCHMARK_SYMBOLS:
        benchmarks[bench] = await asyncio.to_thread(
            _compute_buy_and_hold, bench, START, END, INITIAL_CAPITAL,
        )

    artifact = {
        "spec_ref": SPEC_REF,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "INCONCLUSIVE" if (inconclusive or n_trades == 0) else "REPORTED",
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
        "inconclusive": inconclusive or n_trades == 0,
        "metrics": metrics,
        "benchmarks": benchmarks,
        "hand_picked_reference": HAND_PICKED_BASELINE,
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

    status = artifact["status"]
    spy = benchmarks.get("SPY", {}) or {}
    print(
        f"STATUS={status}  trades={n_trades}  "
        f"total_return={_format_pct(metrics.get('total_return'))}  "
        f"sharpe={_format_num(metrics.get('sharpe_ratio'))}  "
        f"spy_bh={_format_pct(spy.get('total_return'))} "
        f"spy_bh_sharpe={_format_num(spy.get('sharpe'))}"
    )


async def main() -> int:
    try:
        data_broker, source = await _resolve_data_broker()
    except Exception as exc:
        _write_data_unavailable_report(f"Unexpected error resolving data source: {exc}")
        return 1

    if data_broker is None:
        _write_data_unavailable_report(source)
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
