#!/usr/bin/env python3
"""
Unit tests for data quality validators.
"""

from datetime import datetime

import pandas as pd

from utils.data_quality import (
    should_halt_trading_for_data_quality,
    summarize_quality_reports,
    validate_ohlcv_frame,
    validate_pit_announcement_dates,
)


def _sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1200, 900, 1100, 1300],
        },
        index=idx,
    )


def test_validate_ohlcv_valid_frame_has_no_errors():
    report = validate_ohlcv_frame(_sample_ohlcv(), symbol="AAPL")
    assert report.has_errors is False
    assert report.error_count == 0


def test_validate_ohlcv_missing_columns_is_error():
    df = _sample_ohlcv().drop(columns=["volume"])
    report = validate_ohlcv_frame(df, symbol="AAPL")
    assert report.has_errors is True
    assert any(i.code == "missing_columns" for i in report.issues)


def test_validate_ohlcv_invalid_bounds_is_error():
    df = _sample_ohlcv()
    df.loc[df.index[2], "high"] = 90.0
    report = validate_ohlcv_frame(df, symbol="AAPL")
    assert report.has_errors is True
    assert any(i.code == "invalid_ohlc_bounds" for i in report.issues)


def test_validate_ohlcv_stale_data_warning():
    df = _sample_ohlcv()
    report = validate_ohlcv_frame(
        df,
        symbol="AAPL",
        stale_after_days=5,
        reference_time=datetime(2024, 2, 1),
    )
    assert any(i.code == "stale_data" for i in report.issues)


def test_validate_pit_announcement_dates_flags_lookahead():
    points = [
        {"announced_date": "2024-01-10T00:00:00"},
        {"announced_date": "2024-01-20T00:00:00"},
    ]
    report = validate_pit_announcement_dates(points, as_of_date=datetime(2024, 1, 15))
    assert report.has_errors is True
    assert any(i.code == "pit_lookahead_violation" for i in report.issues)


def test_summarize_quality_reports_aggregates_counts():
    valid = validate_ohlcv_frame(_sample_ohlcv(), symbol="AAPL")
    stale = validate_ohlcv_frame(
        _sample_ohlcv(),
        symbol="MSFT",
        stale_after_days=1,
        reference_time=datetime(2024, 2, 1),
    )
    bad = validate_ohlcv_frame(_sample_ohlcv().drop(columns=["volume"]), symbol="NVDA")

    summary = summarize_quality_reports([valid, stale, bad])
    assert summary["symbols_checked"] == 3
    assert summary["symbols_with_errors"] == 1
    assert summary["total_errors"] >= 1
    assert summary["stale_warnings"] >= 1


def test_should_halt_trading_for_data_quality_triggers_on_errors():
    summary = {
        "total_errors": 2,
        "stale_warnings": 0,
    }
    halt, reason = should_halt_trading_for_data_quality(
        summary,
        max_errors=0,
        max_stale_warnings=1,
    )
    assert halt is True
    assert "errors" in (reason or "").lower()
