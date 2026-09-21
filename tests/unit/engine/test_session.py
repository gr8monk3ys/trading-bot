"""The session prepares once, decides per symbol with a fresh view, and submits."""

from datetime import datetime

import pandas as pd

from brokers.backtest import BacktestBroker
from engine.order_submission import OrderIntent, OrderStatus, OrderSubmission
from engine.session import Session

WHEN = datetime(2024, 1, 5)


def _broker():
    broker = BacktestBroker(initial_balance=10_000, execution_profile="idealistic", random_seed=1)
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    for symbol in ("AAA", "BBB"):
        broker.set_price_data(
            symbol,
            pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0}, index=dates),
        )
    broker.advance_to(WHEN)
    return broker


class _Decider:
    """Buys 100 of AAA, then asks for a BBB quantity that depends on remaining cash."""

    def __init__(self):
        self.prepared = []
        self.views = []

    async def prepare(self, when, histories):
        self.prepared.append((when, sorted(histories)))

    async def decide(self, symbol, when, view):
        self.views.append((symbol, view.cash))
        if symbol == "AAA":
            return [OrderIntent("AAA", "buy", 100)]
        return [OrderIntent("BBB", "buy", int(view.cash / 10) // 2)]


async def test_prepare_once_then_decide_per_symbol_with_fresh_cash():
    broker = _broker()
    strategy = _Decider()
    session = Session(strategy, broker, OrderSubmission(broker), ["AAA", "BBB"])
    histories = {"AAA": pd.DataFrame({"close": [1.0]}), "BBB": pd.DataFrame({"close": [1.0]})}

    report = await session.run_session(WHEN, histories)

    assert strategy.prepared == [(WHEN, ["AAA", "BBB"])]
    assert strategy.views[0] == ("AAA", 10_000.0)
    assert strategy.views[1][0] == "BBB" and strategy.views[1][1] < 10_000  # after the AAA fill
    assert report.decisions == 2 and report.intents == 2 and report.errors == 0
    assert all(o.status is OrderStatus.FILLED for o in report.outcomes)
    assert {p.symbol for p in await broker.get_positions()} == {"AAA", "BBB"}


async def test_symbols_without_history_are_skipped_and_errors_are_counted():
    broker = _broker()

    class _Broken:
        async def prepare(self, when, histories):
            pass

        async def decide(self, symbol, when, view):
            raise RuntimeError("boom")

    session = Session(_Broken(), broker, OrderSubmission(broker), ["AAA", "BBB"])
    report = await session.run_session(WHEN, {"AAA": pd.DataFrame({"close": [1.0]})})
    assert report.decisions == 1 and report.errors == 1 and report.intents == 0
