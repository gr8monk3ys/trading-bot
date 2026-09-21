"""Session: one strategy run against a stream of bars from a clock.

The session updates bar histories, asks the strategy to prepare its signals
for the session, then asks it symbol by symbol for order intents against a
fresh portfolio view and submits them. Strategies decide; the session
submits (ADR 0001). The backtest runner drives it with a historical clock;
the live path is wired in the next step (ADR 0002).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

import pandas as pd

from brokers.protocol import Position
from engine.order_submission import OrderIntent, OrderOutcome

logger = logging.getLogger(__name__)


@dataclass
class PortfolioView:
    """What a strategy may know about the book when it decides: nothing that submits."""

    equity: float
    cash: float
    positions: Dict[str, Position]
    buying_power: float = 0.0
    _ask: Callable[[str], Awaitable[float]] = field(repr=False, default=None)  # type: ignore[assignment]

    def position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    async def ask_price(self, symbol: str) -> float:
        return float(await self._ask(symbol))


@dataclass
class SessionReport:
    when: datetime
    intents: int = 0
    outcomes: List[OrderOutcome] = field(default_factory=list)
    errors: int = 0
    decisions: int = 0  # symbols the strategy was asked about


class Session:
    """Drive one strategy over sessions of bars; submit what it decides."""

    def __init__(self, strategy: Any, broker: Any, order_submission: Any, symbols: List[str]):
        self.strategy = strategy
        self.broker = broker
        self.order_submission = order_submission
        self.symbols = list(symbols)

    async def portfolio(self) -> PortfolioView:
        positions = {p.symbol: p for p in await self.broker.get_positions()}
        account = await self.broker.get_account()
        cash = float(getattr(account, "cash", 0) or 0)
        equity = float(getattr(account, "equity", 0) or 0) or cash

        async def ask(symbol: str) -> float:
            quote = await self.broker.get_latest_quote(symbol)
            return float(quote.ask_price)

        buying_power = float(getattr(account, "buying_power", 0) or 0) or cash
        return PortfolioView(
            equity=equity, cash=cash, positions=positions, buying_power=buying_power, _ask=ask
        )

    async def run_session(
        self, when: datetime, histories: Dict[str, pd.DataFrame]
    ) -> SessionReport:
        """One session: prepare signals from bars strictly before ``when``, then decide per symbol.

        The portfolio view is refreshed before every symbol because each fill
        changes cash and equity, and sizing reads them.
        """
        report = SessionReport(when=when)
        try:
            await self.strategy.prepare(when, histories)
        except Exception as e:
            logger.warning(f"prepare failed on {when.date()}: {e}")
            report.errors += 1
            return report
        for symbol in self.symbols:
            if symbol not in histories:
                continue
            report.decisions += 1
            try:
                view = await self.portfolio()
                intents: List[OrderIntent] = list(await self.strategy.decide(symbol, when, view))
            except Exception as e:
                logger.warning(f"decide failed for {symbol} on {when.date()}: {e}")
                report.errors += 1
                continue
            for intent in intents:
                report.intents += 1
                outcome = await self.order_submission.submit(intent)
                report.outcomes.append(outcome)
                if not outcome.ok:
                    logger.info(
                        f"{symbol}: {intent.side} {intent.qty} {outcome.status.value}: {outcome.reason}"
                    )
        return report


class LiveSession(Session):
    """The websocket clock: one bar at a time, histories kept per symbol.

    Subscribes itself to the broker's bar stream (the strategy never does)
    and runs a session for the symbol each bar arrives for. Marks the
    strategy ``execution_mode = "live"`` so its decide() uses the live
    sizing, brackets and exit rules rather than the daily baseline ones.
    """

    def __init__(
        self,
        strategy: Any,
        broker: Any,
        order_submission: Any,
        symbols: List[str],
        *,
        history: int = 200,
    ):
        super().__init__(strategy, broker, order_submission, symbols)
        self.bars: Dict[str, Deque[dict]] = {s: deque(maxlen=history) for s in self.symbols}
        strategy.execution_mode = "live"

    def subscribe(self) -> None:
        add = getattr(self.broker, "_add_subscriber", None)
        if callable(add):
            add(self)

    def unsubscribe(self) -> None:
        remove = getattr(self.broker, "_remove_subscriber", None)
        if callable(remove):
            remove(self)

    async def on_bar(
        self, symbol, open_price, high_price, low_price, close_price, volume, timestamp
    ):
        if symbol not in self.bars:
            return None
        self.bars[symbol].append(
            {
                "timestamp": timestamp,
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume or 0.0),
            }
        )
        rows = list(self.bars[symbol])
        df = pd.DataFrame(rows).set_index(pd.DatetimeIndex([r["timestamp"] for r in rows]))
        when = timestamp if isinstance(timestamp, datetime) else datetime.now()
        return await self.run_session(when, {symbol: df})
