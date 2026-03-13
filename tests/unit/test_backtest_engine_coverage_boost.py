from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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


@dataclass
class _GapEvent:
    symbol: str
    gap_pct: float
    stop_triggered: bool
    slippage_from_stop: float
    date: datetime
    prev_close: float = 100.0
    open_price: float = 94.0
    position_side: str = "long"
    position_qty: int = 10
    stop_price: float = 97.0


class _QualityReport:
    def __init__(self, symbol: str, has_errors: bool):
        self.symbol = symbol
        self.has_errors = has_errors
        self.error_count = 1 if has_errors else 0
        self.warning_count = 0

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "rows": 3,
            "has_errors": self.has_errors,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [],
        }


class _FakeDataBroker:
    def __init__(self, *args, **kwargs):
        pass

    async def get_bars(self, symbol, start, end, timeframe="1Day"):
        if symbol == "ERR":
            raise RuntimeError("data broker down")
        start_dt = datetime.fromisoformat(start)
        return [
            _Bar(100.0, 101.0, 99.0, 100.0, 1000.0, start_dt),
            _Bar(101.0, 102.0, 100.0, 101.0, 1100.0, start_dt + timedelta(days=1)),
            _Bar(102.0, 103.0, 101.0, 102.0, 1200.0, start_dt + timedelta(days=2)),
        ]


class _FakeBacktestBroker:
    def __init__(self, initial_balance=100000, **_kwargs):
        self.balance = initial_balance
        self.positions = {}
        self.price_data = {}
        self._current_date = None
        self._portfolio_calls = 0
        self._trades = [
            {
                "symbol": "GOOD",
                "side": "buy",
                "quantity": 10,
                "price": 100.0,
                "timestamp": "2024-01-02",
            },
            {
                "symbol": "GOOD",
                "side": "sell",
                "quantity": 10,
                "price": 103.0,
                "timestamp": "2024-01-03",
            },
        ]
        self._orders = [{"id": "o1", "created_at": "2024-01-03", "symbol": "GOOD"}]

    def set_price_data(self, symbol, data):
        self.price_data[symbol] = data

    def get_portfolio_value(self, current_date):
        self._portfolio_calls += 1
        if self._portfolio_calls == 2:
            raise RuntimeError("portfolio lookup failed")
        return self.balance + self._portfolio_calls * 10

    def get_balance(self):
        return self.balance

    def get_positions(self):
        return self.positions

    def get_trades(self):
        return self._trades

    def get_orders(self):
        return self._orders

    def update_prev_day_closes(self, current_date):
        return None

    def process_day_start_gaps(self, current_date):
        return [
            _GapEvent(
                symbol="GOOD",
                gap_pct=-0.06,
                stop_triggered=True,
                slippage_from_stop=2.0,
                date=current_date,
            )
        ]

    def get_gap_statistics(self):
        return SimpleNamespace(
            total_gaps=3,
            gaps_exceeding_2pct=2,
            stops_gapped_through=1,
            total_gap_slippage=4.5,
            largest_gap_pct=0.10,
            average_gap_pct=0.03,
        )

    def get_gap_events(self):
        return [
            _GapEvent(
                symbol="GOOD",
                gap_pct=-0.06,
                stop_triggered=True,
                slippage_from_stop=2.0,
                date=datetime(2024, 1, 3),
            )
        ]


class _FakeHistoricalUniverse:
    async def initialize(self):
        return None

    def get_statistics(self):
        return {"total_symbols": 3}

    def get_tradeable_symbols(self, _date, symbols):
        return symbols


class _StrategyNoCurrentData:
    last_instance = None

    def __init__(self, broker=None, parameters=None):
        self.broker = broker
        self.parameters = parameters or {}
        self.price_history = {}
        _StrategyNoCurrentData.last_instance = self

    async def initialize(self):
        return None

    async def generate_signals(self):
        return None

    async def analyze_symbol(self, symbol):
        return {"action": "neutral"}

    async def execute_trade(self, symbol, signal):
        return None


def test_calculate_performance_metrics_zero_days_branch():
    engine = BacktestEngine()
    same_day = datetime(2024, 1, 2)
    result_df = pd.DataFrame(
        {
            "equity": [100000.0, 100500.0],
            "returns": [0.0, 0.005],
            "cum_returns": [0.0, 0.005],
        },
        index=[same_day, same_day],
    )
    engine._calculate_performance_metrics(result_df, "ZeroDays")
    assert result_df.attrs["annualized_return"] == 0


def test_calculate_performance_metrics_builds_cum_returns_when_missing():
    engine = BacktestEngine()
    dates = pd.date_range(start="2024-01-01", periods=3, freq="B")
    result_df = pd.DataFrame(
        {
            "equity": [100000.0, 100500.0, 101000.0],
            "returns": [0.0, 0.005, 0.004975124378109453],
        },
        index=dates,
    )

    engine._calculate_performance_metrics(result_df, "MissingCumReturns")

    assert "cum_returns" in result_df.columns
    assert result_df["cum_returns"].iloc[-1] > 0


def test_build_weekday_sessions_returns_empty_for_inverted_range():
    engine = BacktestEngine()

    sessions = engine._build_weekday_sessions(datetime(2024, 1, 5), datetime(2024, 1, 1))

    assert sessions == []


def test_extract_trading_sessions_handles_missing_invalid_and_out_of_range_data():
    engine = BacktestEngine()

    assert (
        engine._extract_trading_sessions_from_price_data(
            datetime(2024, 1, 1),
            datetime(2024, 1, 5),
            None,
        )
        == []
    )

    price_data = {
        "INVALID": object(),
        "EMPTY": pd.DataFrame(),
        "AAPL": pd.DataFrame(
            {"close": [99.0, 100.0, 101.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
        ),
        "MSFT": pd.DataFrame(
            {"close": [200.0]},
            index=pd.to_datetime(["2024-01-02"]),
        ),
    }

    sessions = engine._extract_trading_sessions_from_price_data(
        datetime(2024, 1, 2),
        datetime(2024, 1, 4),
        price_data,
    )

    assert [session.date().isoformat() for session in sessions] == [
        "2024-01-02",
        "2024-01-04",
    ]


@pytest.mark.asyncio
async def test_fetch_trading_sessions_prefers_cached_price_data():
    engine = BacktestEngine()

    class _Broker:
        def __init__(self):
            self.price_data = {
                "AAPL": pd.DataFrame(
                    {"close": [100.0, 101.0]},
                    index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
                )
            }

        async def get_bars(self, symbol, start, end, timeframe="1Day"):
            raise AssertionError("cached sessions should short-circuit get_bars")

    sessions = await engine._fetch_trading_sessions_from_data_broker(
        _Broker(),
        ["AAPL"],
        datetime(2024, 1, 1),
        datetime(2024, 1, 5),
    )

    assert [session.date().isoformat() for session in sessions] == [
        "2024-01-02",
        "2024-01-04",
    ]


@pytest.mark.asyncio
async def test_fetch_trading_sessions_falls_back_when_broker_has_no_bar_api():
    engine = BacktestEngine()

    class _Broker:
        price_data = {}

    sessions = await engine._fetch_trading_sessions_from_data_broker(
        _Broker(),
        ["AAPL"],
        datetime(2024, 1, 5),
        datetime(2024, 1, 8),
    )

    assert [session.date().isoformat() for session in sessions] == [
        "2024-01-05",
        "2024-01-08",
    ]


@pytest.mark.asyncio
async def test_fetch_trading_sessions_handles_symbol_errors_and_missing_timestamps(caplog):
    engine = BacktestEngine()

    class _Broker:
        price_data = {}

        async def get_bars(self, symbol, start, end, timeframe="1Day"):
            if symbol == "ERR":
                raise RuntimeError("calendar lookup failed")
            return [
                SimpleNamespace(timestamp=None),
                _Bar(100.0, 101.0, 99.0, 100.0, 1000.0, datetime(2024, 1, 3)),
                _Bar(101.0, 102.0, 100.0, 101.0, 1200.0, datetime(2024, 1, 10)),
            ]

    caplog.set_level("WARNING")
    sessions = await engine._fetch_trading_sessions_from_data_broker(
        _Broker(),
        ["ERR", "AAPL"],
        datetime(2024, 1, 1),
        datetime(2024, 1, 5),
    )

    assert [session.date().isoformat() for session in sessions] == ["2024-01-03"]
    assert "Failed to load session calendar for ERR" in caplog.text


@pytest.mark.asyncio
async def test_process_symbol_signal_non_dict_action_and_exception():
    engine = BacktestEngine()

    class _Broker:
        price_data = {"AAPL": pd.DataFrame({"close": [100.0]})}

    class _NeutralStrategy:
        async def analyze_symbol(self, symbol):
            return [1, 2, 3]

        async def execute_trade(self, symbol, signal):
            return None

    neutral_event = await engine._process_symbol_signal("AAPL", _NeutralStrategy(), _Broker(), 1)
    assert neutral_event["action"] == "neutral"
    assert neutral_event["trade_attempted"] is False

    class _FailingStrategy:
        async def analyze_symbol(self, symbol):
            return {"action": "buy"}

        async def execute_trade(self, symbol, signal):
            raise RuntimeError("trade failed")

    failing_event = await engine._process_symbol_signal("AAPL", _FailingStrategy(), _Broker(), 1)
    assert failing_event["trade_attempted"] is True
    assert failing_event["error"] == "trade failed"


@pytest.mark.asyncio
async def test_run_backtest_branch_coverage(monkeypatch, tmp_path):
    engine = BacktestEngine()

    monkeypatch.setattr("brokers.alpaca_broker.AlpacaBroker", _FakeDataBroker)
    monkeypatch.setattr("brokers.backtest_broker.BacktestBroker", _FakeBacktestBroker)
    monkeypatch.setattr(
        "engine.backtest_engine.HistoricalUniverse", lambda broker=None: _FakeHistoricalUniverse()
    )
    monkeypatch.setattr(
        "engine.backtest_engine.validate_ohlcv_frame",
        lambda data, symbol, stale_after_days, reference_time: _QualityReport(
            symbol=symbol, has_errors=(symbol == "BADQ")
        ),
    )

    async def _fake_process_symbol_signal(symbol, strategy, backtest_broker, day_num):
        if symbol == "GOOD":
            raise RuntimeError("decision failure")
        if symbol == "BADQ":
            return "unexpected"
        return {
            "event_type": "decision",
            "symbol": symbol,
            "day_num": day_num,
            "action": "buy",
            "trade_attempted": True,
            "trade_executed": False,
            "error": "symbol failure",
        }

    monkeypatch.setattr(engine, "_process_symbol_signal", _fake_process_symbol_signal)

    result = await engine.run_backtest(
        strategy_class=_StrategyNoCurrentData,
        symbols=["GOOD", "BADQ", "ERR"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 5),
        strategy_params={"custom_param": 42},
        persist_artifacts=True,
        run_id="backtest_coverage_case",
        artifacts_dir=str(tmp_path),
    )

    assert result["data_quality"]["symbols_loaded"] == 1
    assert result["data_quality"]["symbols_rejected"] == 2
    assert result["run_metadata"]["decision_errors"] > 0
    assert result["gap_statistics"]["total_gaps"] == 3
    assert _StrategyNoCurrentData.last_instance is not None
    assert "current_data" in _StrategyNoCurrentData.last_instance.__dict__
    assert _StrategyNoCurrentData.last_instance.parameters["custom_param"] == 42

    run_dir = Path(tmp_path) / "backtest_coverage_case"
    trades_text = (run_dir / "trades.jsonl").read_text()
    assert "event_type" in trades_text and "trade" in trades_text
    assert "event_type" in trades_text and "order" in trades_text


@pytest.mark.asyncio
async def test_run_backtest_skips_bars_without_timestamps(monkeypatch):
    engine = BacktestEngine()

    class _TimestampGapDataBroker(_FakeDataBroker):
        async def get_bars(self, symbol, start, end, timeframe="1Day"):
            return [
                SimpleNamespace(
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1000.0,
                    timestamp=None,
                ),
                _Bar(100.0, 101.0, 99.0, 100.0, 1000.0, datetime(2024, 1, 3)),
            ]

    monkeypatch.setattr("brokers.alpaca_broker.AlpacaBroker", _TimestampGapDataBroker)
    monkeypatch.setattr("brokers.backtest_broker.BacktestBroker", _FakeBacktestBroker)
    monkeypatch.setattr(
        "engine.backtest_engine.HistoricalUniverse", lambda broker=None: _FakeHistoricalUniverse()
    )
    monkeypatch.setattr(
        "engine.backtest_engine.validate_ohlcv_frame",
        lambda data, symbol, stale_after_days, reference_time: _QualityReport(
            symbol=symbol, has_errors=False
        ),
    )

    result = await engine.run_backtest(
        strategy_class=_StrategyNoCurrentData,
        symbols=["GOOD"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 5),
        initial_capital=100000,
    )

    assert [ts.date().isoformat() for ts in result["equity_curve_series"].index] == [
        "2024-01-03"
    ]


@pytest.mark.asyncio
async def test_run_walk_forward_backtest_date_conversion_and_is_exception(monkeypatch):
    engine = BacktestEngine()
    calls = {"count": 0}

    async def _fake_run_backtest(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("IS failure")
        return {"equity_curve": [100000, 101000], "total_trades": 3}

    monkeypatch.setattr(engine, "run_backtest", _fake_run_backtest)

    result = await engine.run_walk_forward_backtest(
        strategy_class=object,
        symbols=["AAPL"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 4, 30),
        n_folds=1,
        embargo_days=2,
        train_pct=0.6,
    )

    assert result["fold_results"]
    assert result["fold_results"][0]["is_return"] == 0
    assert result["fold_results"][0]["is_sharpe"] == 0
