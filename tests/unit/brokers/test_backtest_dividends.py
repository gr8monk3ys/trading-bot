"""BacktestBroker dividends: paid on the ex-date to positions held into it; off by default."""

import pandas as pd
import pytest

from brokers.backtest import BacktestBroker


def _broker_with(qty: float) -> BacktestBroker:
    b = BacktestBroker(initial_balance=1_000)
    b.positions["SPY"] = {"symbol": "SPY", "quantity": qty, "avg_price": 100.0}
    return b


def test_no_dividends_loaded_credits_nothing():
    b = _broker_with(10)
    assert b.credit_dividends(pd.Timestamp("2024-03-15")) == 0.0
    assert b.balance == 1_000
    assert b.dividends_credited == 0.0


def test_long_position_collects_on_ex_date_only():
    b = _broker_with(10)
    b.set_dividends("SPY", pd.Series([1.5], index=[pd.Timestamp("2024-03-15")]))
    assert b.credit_dividends(pd.Timestamp("2024-03-14")) == 0.0
    assert b.credit_dividends(pd.Timestamp("2024-03-15")) == pytest.approx(15.0)
    assert b.balance == pytest.approx(1_015.0)
    assert b.dividends_credited == pytest.approx(15.0)


def test_short_position_pays_the_dividend():
    b = _broker_with(-10)
    b.set_dividends("SPY", pd.Series([1.5], index=[pd.Timestamp("2024-03-15")]))
    assert b.credit_dividends(pd.Timestamp("2024-03-15")) == pytest.approx(-15.0)
    assert b.balance == pytest.approx(985.0)


def test_tz_aware_ex_dates_match_naive_session_dates():
    b = _broker_with(10)
    idx = pd.DatetimeIndex([pd.Timestamp("2024-03-15 00:00", tz="America/New_York")])
    b.set_dividends("SPY", pd.Series([2.0], index=idx))
    assert b.credit_dividends(pd.Timestamp("2024-03-15")) == pytest.approx(20.0)


def test_unheld_symbol_collects_nothing():
    b = _broker_with(10)
    b.set_dividends("QQQ", pd.Series([1.0], index=[pd.Timestamp("2024-03-15")]))
    assert b.credit_dividends(pd.Timestamp("2024-03-15")) == 0.0
