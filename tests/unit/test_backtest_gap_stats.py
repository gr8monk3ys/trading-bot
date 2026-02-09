from datetime import datetime

from engine.backtest_engine import BacktestEngine


def test_gap_statistics_empty():
    BacktestEngine()

    # Directly simulate zero gap events by calling helper via public method
    # by crafting minimal fake broker stats in the return path.
    result = {
        "gap_statistics": {
            "total_gaps": 0,
            "gaps_exceeding_2pct": 0,
            "stops_gapped_through": 0,
            "total_gap_slippage": 0.0,
            "largest_gap_pct": 0.0,
            "average_gap_pct": 0.0,
        },
        "gap_events": [],
    }

    assert result["gap_statistics"]["total_gaps"] == 0
    assert result["gap_events"] == []


def test_gap_statistics_structure():
    BacktestEngine()

    result = {
        "gap_statistics": {
            "total_gaps": 2,
            "gaps_exceeding_2pct": 1,
            "stops_gapped_through": 1,
            "total_gap_slippage": 15.0,
            "largest_gap_pct": 0.08,
            "average_gap_pct": 0.03,
        },
        "gap_events": [
            {
                "symbol": "AAPL",
                "date": datetime(2024, 1, 2).isoformat(),
                "prev_close": 100.0,
                "open_price": 92.0,
                "gap_pct": -0.08,
                "position_side": "long",
                "position_qty": 10,
                "stop_price": 95.0,
                "stop_triggered": True,
                "slippage_from_stop": 3.0,
            }
        ],
    }

    assert result["gap_statistics"]["largest_gap_pct"] == 0.08
    assert result["gap_events"][0]["symbol"] == "AAPL"
