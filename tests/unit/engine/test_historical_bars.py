"""Historical bars: every symbol gets an outcome; a failed fetch is never an empty run."""

from datetime import datetime

import pytest

from engine.historical_bars import Bar, BarsResult, DataOutcome, DataUnavailableError, load_bars


class _Source:
    name = "fake"

    def __init__(self, table):
        self.table = table

    async def get_bars(self, symbol, start, end):
        value = self.table[symbol]
        if isinstance(value, Exception):
            raise value
        return value


def _bars(n, start_day=2):
    return [
        Bar(datetime(2024, 1, start_day + i), 100.0, 101.0, 99.0, 100.0 + i, 1_000.0)
        for i in range(n)
    ]


async def test_every_symbol_gets_an_outcome():
    result = await load_bars(
        _Source({"SPY": _bars(3), "QQQ": [], "IWM": ConnectionError("dns")}),
        ["SPY", "QQQ", "IWM"],
        "2024-01-01",
        "2024-01-10",
    )
    assert result.source == "fake"
    assert result.loaded == ["SPY"] and result.empty == ["QQQ"] and result.failed == ["IWM"]
    assert result.outcomes["IWM"].cause == "ConnectionError: dns"
    assert not result.ok
    assert "IWM: ConnectionError: dns" in result.describe_failures()
    assert "QQQ: no rows" in result.describe_failures()


async def test_ok_only_when_everything_loaded():
    result = await load_bars(_Source({"SPY": _bars(2), "QQQ": _bars(2)}), ["SPY", "QQQ"], "a", "b")
    assert result.ok
    frames = result.frames()
    assert set(frames) == {"SPY", "QQQ"} and list(frames["SPY"].columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert frames["SPY"]["volume"].dtype == float
    assert result.sessions() == [datetime(2024, 1, 2), datetime(2024, 1, 3)]
    assert result.report()["SPY"] == {"rows": 2, "loaded": True, "status": "loaded"}


async def test_empty_request_is_not_ok():
    assert not (await load_bars(_Source({}), [], "a", "b")).ok


def test_error_names_the_failed_symbols():
    result = BarsResult("yfinance", {"SPY": DataOutcome("SPY", "failed", cause="timeout")})
    err = DataUnavailableError(result)
    assert "yfinance" in str(err) and "SPY: timeout" in str(err)
    assert err.result is result


async def test_engine_refuses_to_run_on_partial_data():
    from unittest.mock import MagicMock

    from engine.backtest_engine import BacktestEngine

    broker = MagicMock()

    async def get_bars(symbol, start=None, end=None, timeframe="1Day"):
        if symbol == "BBB":
            raise RuntimeError("rate limited")
        return _bars(5)

    broker.get_bars = get_bars
    engine = BacktestEngine(broker=broker)
    with pytest.raises(DataUnavailableError) as info:
        await engine.run_backtest(
            strategy_class=MagicMock,
            symbols=["AAA", "BBB"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            initial_capital=10_000,
        )
    assert info.value.result.failed == ["BBB"]
