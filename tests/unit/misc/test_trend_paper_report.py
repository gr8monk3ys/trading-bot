"""`scripts/trend_paper_report.py` prices paper fills against the gate's 50 bps.

The sign convention is the contract: a positive number is always a cost, for
buys and sells alike, so a mean above 50 means the gate was too kind.
"""

from __future__ import annotations

import pytest

from scripts.trend_paper_report import (
    COLUMNS,
    main,
    read_rows,
    render,
    slippage_bps,
    summarize,
)


def test_buy_above_reference_is_a_cost():
    assert slippage_bps("buy", 100.0, 100.5) == pytest.approx(50.0)


def test_sell_below_reference_is_a_cost():
    assert slippage_bps("sell", 100.0, 99.5) == pytest.approx(50.0)


def test_price_improvement_is_negative():
    assert slippage_bps("buy", 100.0, 99.9) == pytest.approx(-10.0)
    assert slippage_bps("SELL", 100.0, 100.1) == pytest.approx(-10.0)


@pytest.mark.parametrize("ref,fill", [(None, 100.0), (100.0, None), (0.0, 100.0)])
def test_missing_price_gives_none(ref, fill):
    assert slippage_bps("buy", ref, fill) is None


def test_old_eight_field_rows_count_as_orders_without_cost(tmp_path):
    log = tmp_path / "trend_paper.csv"
    log.write_text(
        "2026-12-01T14:00:00,SPY,buy,10,10,500.0,filled,\n"
        + ",".join(COLUMNS)
        + "\n"
        + "2027-01-04T14:00:00,SPY,sell,10,10,499.0,filled,,500.0,20.00\n"
        + "2027-01-04T14:00:00,TLT,buy,5,5,90.09,filled,,90.0,10.00\n",
        encoding="utf-8",
    )
    s = summarize(read_rows(log))
    assert (s.runs, s.orders, s.fills_with_ref) == (2, 3, 2)
    assert s.mean_bps == pytest.approx(15.0)
    assert s.worst_bps == pytest.approx(20.0)
    text = render(s)
    assert "under the gate assumption" in text
    assert "anecdote" in text


def test_empty_and_missing_logs(tmp_path, capsys):
    assert render(summarize([])) == "No paper orders logged yet."
    assert main([str(tmp_path / "absent.csv")]) == 1
    assert "No log at" in capsys.readouterr().out
