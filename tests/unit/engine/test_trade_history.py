from datetime import date, datetime

from engine.order_submission import OrderIntent, OrderOutcome, OrderStatus
from engine.trade_history import MemoryStore, SqliteStore, TradeHistory, TradeRecord
from engine.trade_recorder import TradeRecorder


def _rec(symbol, pnl, day, strategy="mom"):
    t = datetime(2024, 1, day, 15)
    return TradeRecord(symbol, strategy, "long", t, t, 100.0, 100.0 + pnl / 10, 10, pnl, pnl / 1000)


def test_summary_daily_and_ordering_in_memory():
    h = TradeHistory(MemoryStore())
    for r in (_rec("SPY", 100.0, 2), _rec("QQQ", -40.0, 3), _rec("SPY", 60.0, 3, "mr")):
        h.record(r)
    s = h.summary()
    assert s["total_trades"] == 3 and s["winning_trades"] == 2 and s["total_pnl"] == 120.0
    assert s["profit_factor"] == 4.0 and s["unique_symbols"] == 2 and s["unique_strategies"] == 2
    assert [t.symbol for t in h.trades(limit=2)] == ["QQQ", "SPY"]  # newest first
    assert h.daily(date(2024, 1, 3), date(2024, 1, 3)) == [
        {"date": "2024-01-03", "pnl": 20.0, "trades": 2}
    ]
    assert h.daily(date(2024, 2, 1), date(2024, 2, 2)) == []


def test_sqlite_store_round_trips(tmp_path):
    path = tmp_path / "t.db"
    h = TradeHistory(SqliteStore(path))
    h.record(_rec("SPY", 5.0, 2))
    h.close()
    again = TradeHistory(SqliteStore(path))
    (t,) = again.trades()
    assert t.symbol == "SPY" and t.pnl == 5.0 and t.exit_time == datetime(2024, 1, 2, 15)
    assert again.summary()["total_trades"] == 1


def test_recorder_writes_history_with_the_strategy_name():
    h = TradeHistory()
    rec = TradeRecorder()
    rec.subscribe_all(h.record_trade)
    buy = OrderIntent("SPY", "buy", 10, strategy_name="mom")
    rec.on_outcome(
        buy,
        OrderOutcome(OrderStatus.FILLED, "SPY", "buy", 10, 10, "a", 100.0),
        datetime(2024, 1, 2),
    )
    sell = OrderIntent("SPY", "sell", 10, is_exit=True, strategy_name="mom")
    rec.on_outcome(
        sell,
        OrderOutcome(OrderStatus.FILLED, "SPY", "sell", 10, 10, "b", 101.0),
        datetime(2024, 1, 3),
    )
    (t,) = h.trades()
    assert (
        t.strategy == "mom"
        and t.pnl == 10.0
        and t.is_winner
        and t.to_dict()["exit_time"].startswith("2024-01-03")
    )
