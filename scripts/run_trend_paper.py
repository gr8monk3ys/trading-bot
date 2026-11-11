"""Run one monthly rebalance of the trend candidate on Alpaca PAPER.

Usage:
    uv run python scripts/run_trend_paper.py --dry-run   # signals only, no keys, no orders
    uv run python scripts/run_trend_paper.py             # paper orders, first session of a month
    uv run python scripts/run_trend_paper.py --force     # paper orders on any day

Schedule it once a day after the open; it does nothing except on the first
trading session of a month. It always builds ``AlpacaBroker(paper=True)``
and has no switch to live trading.

What paper can and cannot tell you: it checks execution (fills, slippage vs
the 50 bps the backtest assumes, the monthly schedule actually running). It
cannot tell you whether the strategy has an edge: a few months of returns
are noise. The candidate FAILED its dev-window pass bar
(results/gate_trend_dev.md), so expect it to track a lower-risk blend of
stocks and bonds, not to beat SPY.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from strategies.params import TrendParams  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("trend_paper")
LOG_DIR = REPO_ROOT / "logs"


def _histories(symbols: list[str], today: datetime) -> dict[str, pd.DataFrame]:
    """Daily bars strictly before today (same no-look-ahead rule as the backtest)."""
    import yfinance as yf

    start = (today - timedelta(days=420)).strftime("%Y-%m-%d")
    out = {}
    for s in symbols:
        df = yf.download(
            s, start=start, end=today.strftime("%Y-%m-%d"), auto_adjust=False, progress=False
        )
        if df is None or df.empty:
            raise SystemExit(f"no bars for {s}; refusing to trade on partial data")
        df.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in df.columns]
        out[s] = df[["open", "high", "low", "close", "volume"]]
    return out


def _first_session_of_month(histories: dict[str, pd.DataFrame], today: datetime) -> bool:
    last = max(df.index[-1] for df in histories.values())
    return (last.year, last.month) != (today.year, today.month)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebalance even mid-month")
    args = ap.parse_args()

    from strategies.trend_following_strategy import TrendFollowingStrategy

    today = datetime.now(timezone.utc).replace(tzinfo=None)
    symbols = list(TrendParams.defaults()["symbols"])
    histories = _histories(symbols, today)
    if not args.force and not args.dry_run and not _first_session_of_month(histories, today):
        logger.info("Not the first session of the month; nothing to do.")
        return 0

    strategy = TrendFollowingStrategy(parameters={"symbols": symbols})
    await strategy.prepare(today, histories)
    for s in symbols:
        logger.info(
            "%-4s %s", s, {True: "IN trend", False: "out", None: "no data"}[strategy.signals.get(s)]
        )
    if args.dry_run:
        return 0

    from brokers.alpaca_broker import AlpacaBroker
    from engine.order_submission import OrderSubmission
    from engine.session import Session
    from engine.trade_recorder import TradeRecorder
    from utils.circuit_breaker import CircuitBreaker

    broker = AlpacaBroker(paper=True)
    if not broker.paper:
        raise SystemExit("refusing: broker is not in paper mode")
    status = await broker.get_market_status()
    if not status.get("is_open", False):
        logger.warning("Market is closed; market orders would queue. Run after the open.")
        return 1
    breaker = CircuitBreaker(max_daily_loss=0.03, auto_close_positions=False)
    await breaker.initialize(broker)
    submission = OrderSubmission(broker, circuit_breaker=breaker, recorder=TradeRecorder())
    session = Session(strategy, broker, submission, symbols)
    report = await session.run_session(today, histories)

    LOG_DIR.mkdir(exist_ok=True)
    with (LOG_DIR / "trend_paper.csv").open("a", encoding="utf-8") as f:
        for o in report.outcomes:
            f.write(
                f"{today.isoformat()},{o.symbol},{o.side},{o.qty_requested},{o.qty_filled},"
                f"{o.fill_price},{o.status.value},{o.reason}\n"
            )
    account = await broker.get_account()
    logger.info(
        "Rebalance done: %d intents, %d errors; paper equity %s",
        report.intents,
        report.errors,
        getattr(account, "equity", "?"),
    )
    return 0 if report.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
