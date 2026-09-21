from datetime import datetime

from engine.order_submission import OrderIntent, OrderOutcome, OrderStatus
from engine.trade_recorder import TradeRecorder

T0, T1 = datetime(2024, 1, 2), datetime(2024, 1, 9)


def _fill(intent, price, qty=None):
    q = qty if qty is not None else intent.qty
    return OrderOutcome(OrderStatus.FILLED, intent.symbol, intent.side, intent.qty, q, "id", price)


def test_round_trip_produces_a_trade_and_feeds_listeners():
    rec = TradeRecorder()
    seen = []
    rec.subscribe("mom", seen.append)
    buy = OrderIntent("SPY", "buy", 10, strategy_name="mom")
    assert rec.on_outcome(buy, _fill(buy, 100.0), T0) is None
    sell = OrderIntent("SPY", "sell", 10, is_exit=True, strategy_name="mom")
    trade = rec.on_outcome(sell, _fill(sell, 110.0), T1)
    assert trade.pnl == 100.0 and trade.pnl_pct == 0.1 and trade.is_winner
    assert trade.entry_time == T0 and trade.exit_time == T1
    assert seen == [trade] and rec.trades == [trade]


def test_short_round_trip_and_partial_close():
    rec = TradeRecorder()
    short = OrderIntent("SPY", "sell", 10, strategy_name="s")
    rec.on_outcome(short, _fill(short, 100.0), T0)
    cover = OrderIntent("SPY", "buy", 4, is_exit=True, strategy_name="s")
    trade = rec.on_outcome(cover, _fill(cover, 90.0), T1)
    assert trade.quantity == 4 and trade.pnl == 40.0
    assert rec._lots[("s", "SPY")].qty == -6


def test_adding_averages_the_entry_and_rejections_are_ignored():
    rec = TradeRecorder()
    a = OrderIntent("SPY", "buy", 10, strategy_name="m")
    rec.on_outcome(a, _fill(a, 100.0), T0)
    b = OrderIntent("SPY", "buy", 10, strategy_name="m")
    rec.on_outcome(b, _fill(b, 120.0), T0)
    assert rec._lots[("m", "SPY")].entry_price == 110.0
    rejected = OrderOutcome(OrderStatus.REJECTED, "SPY", "sell", 5, 0.0, "", None, "nope")
    assert (
        rec.on_outcome(OrderIntent("SPY", "sell", 5, is_exit=True, strategy_name="m"), rejected)
        is None
    )


def test_listener_errors_never_propagate():
    rec = TradeRecorder()
    rec.subscribe("m", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
    a = OrderIntent("SPY", "buy", 1, strategy_name="m")
    rec.on_outcome(a, _fill(a, 1.0), T0)
    s = OrderIntent("SPY", "sell", 1, is_exit=True, strategy_name="m")
    assert rec.on_outcome(s, _fill(s, 2.0), T1) is not None
