from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from engine.backtest_engine import BacktestEngine


@dataclass
class _Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


class _FakeDataBroker:
    def __init__(self, *args, **kwargs):
        pass

    async def get_bars(self, symbol, start, end, timeframe="1Day"):
        # Return minimal bar data
        start_dt = datetime.fromisoformat(start)
        return [
            _Bar(100, 101, 99, 100, 1000, start_dt),
            _Bar(100, 102, 98, 101, 1200, start_dt + timedelta(days=1)),
            _Bar(101, 103, 100, 102, 1100, start_dt + timedelta(days=2)),
        ]


class _FakeBacktestBroker:
    def __init__(self, initial_balance=100000, **_kwargs):
        self.balance = initial_balance
        self.positions = {}
        self.orders = []
        self.trades = []
        self.price_data = {}
        self._current_date = None

    def set_price_data(self, symbol, data):
        self.price_data[symbol] = data

    def get_portfolio_value(self, _date):
        return self.balance

    def get_balance(self):
        return self.balance

    def get_positions(self):
        return self.positions

    def get_trades(self):
        return self.trades

    def update_prev_day_closes(self, _date):
        return None

    def process_day_start_gaps(self, _date):
        return []

    def get_gap_statistics(self):
        return SimpleNamespace(
            total_gaps=0,
            gaps_exceeding_2pct=0,
            stops_gapped_through=0,
            total_gap_slippage=0.0,
            largest_gap_pct=0.0,
            average_gap_pct=0.0,
        )

    def get_gap_events(self):
        return []


class _FakeHistoricalUniverse:
    def __init__(self, broker=None):
        self.broker = broker

    async def initialize(self):
        return None

    def get_statistics(self):
        return {"total_symbols": 1}

    def get_tradeable_symbols(self, _date, symbols):
        return symbols


class _DummyStrategy:
    def __init__(self, broker=None, parameters=None, **_kwargs):
        self.broker = broker
        self.parameters = parameters or {}
        self.price_history = {}
        self.current_data = {}

    async def initialize(self):
        return None

    async def generate_signals(self):
        return None

    async def analyze_symbol(self, _symbol):
        return "hold"

    async def execute_trade(self, _symbol, _signal):
        return None


@pytest.mark.asyncio
async def test_run_backtest_returns_structure(monkeypatch):
    # Patch dependencies inside backtest engine
    monkeypatch.setattr("brokers.alpaca_broker.AlpacaBroker", _FakeDataBroker)
    monkeypatch.setattr("brokers.backtest_broker.BacktestBroker", _FakeBacktestBroker)
    monkeypatch.setattr("engine.backtest_engine.HistoricalUniverse", _FakeHistoricalUniverse)

    engine = BacktestEngine()

    result = await engine.run_backtest(
        strategy_class=_DummyStrategy,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 5),
        initial_capital=100000,
    )

    assert "equity_curve" in result
    assert "gap_statistics" in result
    assert result["total_trades"] == 0


@pytest.mark.asyncio
async def test_run_backtest_handles_no_data(monkeypatch):
    class _EmptyDataBroker(_FakeDataBroker):
        async def get_bars(self, symbol, start, end, timeframe="1Day"):
            return []

    monkeypatch.setattr("brokers.alpaca_broker.AlpacaBroker", _EmptyDataBroker)
    monkeypatch.setattr("brokers.backtest_broker.BacktestBroker", _FakeBacktestBroker)
    monkeypatch.setattr("engine.backtest_engine.HistoricalUniverse", _FakeHistoricalUniverse)

    engine = BacktestEngine()

    result = await engine.run_backtest(
        strategy_class=_DummyStrategy,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 3),
        initial_capital=100000,
    )

    assert "equity_curve" in result
    assert result["total_trades"] == 0


@pytest.mark.asyncio
async def test_run_backtest_persists_observability_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr("brokers.alpaca_broker.AlpacaBroker", _FakeDataBroker)
    monkeypatch.setattr("brokers.backtest_broker.BacktestBroker", _FakeBacktestBroker)
    monkeypatch.setattr("engine.backtest_engine.HistoricalUniverse", _FakeHistoricalUniverse)

    engine = BacktestEngine()
    run_id = "backtest_20240101_000000_testabcd"

    result = await engine.run_backtest(
        strategy_class=_DummyStrategy,
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 3),
        initial_capital=100000,
        run_id=run_id,
        persist_artifacts=True,
        artifacts_dir=str(tmp_path),
    )

    metadata = result["run_metadata"]
    assert metadata["run_id"] == run_id
    assert metadata["persist_artifacts"] is True

    run_dir = tmp_path / run_id
    assert run_dir.exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "decision_events.jsonl").exists()
    assert (run_dir / "trades.jsonl").exists()
