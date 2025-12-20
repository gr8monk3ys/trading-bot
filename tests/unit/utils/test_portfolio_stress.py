#!/usr/bin/env python3
"""
Unit tests for portfolio stress testing utilities.
"""

from utils.portfolio_stress import (
    StressScenario,
    default_stress_scenarios,
    run_portfolio_stress_test,
)


def test_default_stress_scenarios_present():
    scenarios = default_stress_scenarios()
    names = {s.name for s in scenarios}
    assert "market_down_5" in names
    assert "market_down_10" in names
    assert "risk_off_15" in names
    assert "tech_crash_20" in names


def test_run_portfolio_stress_test_computes_worst_loss():
    positions = [
        {"symbol": "AAPL", "qty": 100, "current_price": 100.0},
        {"symbol": "MSFT", "qty": 50, "current_price": 200.0},
    ]
    result = run_portfolio_stress_test(positions, equity=50000)
    assert result["position_count"] == 2
    assert result["gross_exposure"] == 20000
    assert result["worst_scenario"] is not None
    assert result["max_stress_loss"] >= 0
    assert result["max_stress_loss_pct"] is not None


def test_run_portfolio_stress_test_supports_custom_symbol_shocks():
    positions = [
        {"symbol": "XYZ", "market_value": 10000},
    ]
    scenarios = [
        StressScenario(
            name="custom",
            description="custom",
            market_shock_pct=-0.01,
            symbol_shocks={"XYZ": -0.30},
        )
    ]
    result = run_portfolio_stress_test(positions, equity=20000, scenarios=scenarios)
    worst = result["worst_scenario"]
    assert worst["name"] == "custom"
    assert worst["portfolio_loss"] == 3000
    assert abs(worst["portfolio_loss_pct"] - 0.15) < 1e-9
