"""
Portfolio stress-testing utilities for scenario-based risk analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class StressScenario:
    """Represents one stress scenario."""

    name: str
    description: str
    market_shock_pct: float
    symbol_shocks: Dict[str, float] = field(default_factory=dict)


def default_stress_scenarios() -> List[StressScenario]:
    """Default conservative stress scenarios for equity portfolios."""
    return [
        StressScenario(
            name="market_down_5",
            description="Broad market drawdown of -5%",
            market_shock_pct=-0.05,
        ),
        StressScenario(
            name="market_down_10",
            description="Broad market drawdown of -10%",
            market_shock_pct=-0.10,
        ),
        StressScenario(
            name="risk_off_15",
            description="Risk-off shock of -15% across liquid equities",
            market_shock_pct=-0.15,
        ),
        StressScenario(
            name="tech_crash_20",
            description="Technology-heavy symbols drop -20% with market -8%",
            market_shock_pct=-0.08,
            symbol_shocks={
                "AAPL": -0.20,
                "MSFT": -0.20,
                "NVDA": -0.25,
                "TSLA": -0.25,
                "AMD": -0.22,
                "META": -0.18,
                "GOOGL": -0.18,
            },
        ),
    ]


def _extract_position_exposure(position: Any) -> tuple[str, float]:
    """
    Extract (symbol, exposure) from heterogeneous position payloads.

    Exposure is notional value signed by direction.
    """
    if isinstance(position, dict):
        symbol = str(position.get("symbol", "UNKNOWN"))
        if position.get("market_value") is not None:
            return symbol, float(position["market_value"])

        qty = float(position.get("qty", position.get("quantity", 0)) or 0)
        price = float(
            position.get("current_price", position.get("price", position.get("avg_entry_price", 0)))
            or 0
        )
        return symbol, qty * price

    symbol = str(getattr(position, "symbol", "UNKNOWN"))
    market_value = getattr(position, "market_value", None)
    if market_value is not None:
        return symbol, float(market_value)

    qty = float(getattr(position, "qty", getattr(position, "quantity", 0)) or 0)
    price = float(
        getattr(
            position,
            "current_price",
            getattr(position, "price", getattr(position, "avg_entry_price", 0)),
        )
        or 0
    )
    return symbol, qty * price


def run_portfolio_stress_test(
    positions: Iterable[Any],
    *,
    equity: Optional[float] = None,
    scenarios: Optional[List[StressScenario]] = None,
) -> Dict[str, Any]:
    """
    Run scenario stress tests and return loss profile.
    """
    scenarios = scenarios or default_stress_scenarios()
    normalized_positions: List[Dict[str, Any]] = []
    gross_exposure = 0.0

    for pos in positions or []:
        symbol, exposure = _extract_position_exposure(pos)
        normalized_positions.append({"symbol": symbol, "exposure": exposure})
        gross_exposure += abs(exposure)

    scenario_results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        pnl = 0.0
        symbol_impacts: List[Dict[str, Any]] = []
        for pos in normalized_positions:
            symbol = pos["symbol"]
            exposure = float(pos["exposure"])
            shock = scenario.symbol_shocks.get(symbol, scenario.market_shock_pct)
            impact = exposure * shock
            pnl += impact
            symbol_impacts.append(
                {
                    "symbol": symbol,
                    "exposure": exposure,
                    "shock_pct": shock,
                    "pnl_impact": impact,
                }
            )

        loss = min(0.0, pnl)
        loss_pct = (abs(loss) / equity) if equity and equity > 0 else None
        scenario_results.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "portfolio_pnl": pnl,
                "portfolio_loss": abs(loss),
                "portfolio_loss_pct": loss_pct,
                "symbol_impacts": symbol_impacts,
            }
        )

    worst = min(scenario_results, key=lambda row: row["portfolio_pnl"], default=None)
    return {
        "position_count": len(normalized_positions),
        "gross_exposure": gross_exposure,
        "equity": equity,
        "scenarios": scenario_results,
        "worst_scenario": worst,
        "max_stress_loss": abs(min(0.0, worst["portfolio_pnl"])) if worst else 0.0,
        "max_stress_loss_pct": (
            (abs(min(0.0, worst["portfolio_pnl"])) / equity)
            if worst and equity and equity > 0
            else None
        ),
    }
