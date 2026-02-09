"""
Stress Testing Framework - Portfolio Stress Testing Against Historical Scenarios

Tests portfolio resilience against historical market crisis scenarios:
- 2008 Financial Crisis
- 2020 COVID Crash
- 2022 Bear Market
- Interest Rate Shocks
- Volatility Spikes

Usage:
    from utils.stress_tester import StressTester

    tester = StressTester(broker)
    results = await tester.run_stress_test(positions)

    for scenario, result in results.items():
        print(f"{scenario}: {result['pnl_pct']:.1%} loss, {'PASS' if result['passes'] else 'FAIL'}")
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress testing scenario."""

    name: str
    description: str
    spy_shock: float  # SPY return in scenario
    vix_spike: Optional[float] = None  # VIX level (if applicable)
    bond_shock: Optional[float] = None  # Bond return (if applicable)
    correlation: float = 0.80  # How correlated stocks are in crisis
    sector_adjustments: Optional[Dict[str, float]] = None  # Sector-specific shocks


# Historical stress scenarios
STRESS_SCENARIOS = {
    "2008_financial_crisis": StressScenario(
        name="2008 Financial Crisis",
        description="Lehman Brothers collapse, global financial system near-failure",
        spy_shock=-0.50,
        vix_spike=80,
        correlation=0.95,
        sector_adjustments={
            "Financials": -0.75,  # Banks hit hardest
            "Real Estate": -0.65,
            "Consumer Discretionary": -0.55,
            "Technology": -0.45,
            "Consumer Staples": -0.30,
            "Utilities": -0.25,
            "Healthcare": -0.35,
        },
    ),
    "2020_covid_crash": StressScenario(
        name="2020 COVID Crash",
        description="Pandemic lockdown fears, fastest bear market in history",
        spy_shock=-0.35,
        vix_spike=82,
        correlation=0.90,
        sector_adjustments={
            "Energy": -0.60,
            "Financials": -0.45,
            "Industrials": -0.40,
            "Consumer Discretionary": -0.40,
            "Real Estate": -0.35,
            "Technology": -0.25,
            "Healthcare": -0.20,
            "Consumer Staples": -0.15,
        },
    ),
    "2022_bear_market": StressScenario(
        name="2022 Bear Market",
        description="Fed rate hikes, inflation concerns, tech selloff",
        spy_shock=-0.25,
        vix_spike=35,
        correlation=0.75,
        sector_adjustments={
            "Technology": -0.35,
            "Communication Services": -0.40,
            "Consumer Discretionary": -0.30,
            "Real Estate": -0.25,
            "Energy": 0.20,  # Energy outperformed
            "Utilities": -0.10,
            "Healthcare": -0.15,
        },
    ),
    "rate_shock_200bp": StressScenario(
        name="200bp Rate Shock",
        description="Rapid 200 basis point rate increase",
        spy_shock=-0.15,
        bond_shock=-0.10,
        correlation=0.70,
        sector_adjustments={
            "Real Estate": -0.25,
            "Utilities": -0.20,
            "Technology": -0.20,
            "Financials": -0.05,  # Mixed impact on banks
            "Energy": 0.05,
        },
    ),
    "flash_crash": StressScenario(
        name="Flash Crash",
        description="2010-style flash crash, brief but severe",
        spy_shock=-0.10,
        vix_spike=50,
        correlation=0.95,  # Everything drops together
    ),
    "vol_spike": StressScenario(
        name="Volatility Spike",
        description="VIX doubles without major equity decline",
        spy_shock=-0.08,
        vix_spike=50,
        correlation=0.60,
    ),
    "sector_rotation": StressScenario(
        name="Sector Rotation",
        description="Growth-to-value rotation shock",
        spy_shock=-0.05,
        vix_spike=25,
        correlation=0.40,
        sector_adjustments={
            "Technology": -0.25,
            "Communication Services": -0.20,
            "Consumer Discretionary": -0.15,
            "Financials": 0.10,
            "Energy": 0.15,
            "Materials": 0.08,
            "Industrials": 0.05,
        },
    ),
    "black_swan": StressScenario(
        name="Black Swan Event",
        description="Extreme tail event, 4+ standard deviation move",
        spy_shock=-0.20,
        vix_spike=90,
        correlation=0.98,  # Near-perfect correlation in crisis
    ),
}


class StressTester:
    """
    Portfolio stress testing against historical crisis scenarios.

    Tests resilience by applying scenario shocks to current positions
    and calculating expected P&L under each scenario.
    """

    def __init__(
        self,
        broker=None,
        max_acceptable_loss: float = 0.15,  # Max 15% loss in any scenario
        scenarios: Optional[Dict[str, StressScenario]] = None,
    ):
        """
        Initialize stress tester.

        Args:
            broker: Optional broker for fetching position data
            max_acceptable_loss: Maximum acceptable loss in any scenario (default 15%)
            scenarios: Optional custom scenarios (uses defaults if not provided)
        """
        self.broker = broker
        self.max_acceptable_loss = max_acceptable_loss
        self.scenarios = scenarios or STRESS_SCENARIOS
        self._sector_cache: Dict[str, str] = {}

    async def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol using yfinance."""
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]

        # Default mappings for common ETFs and indices
        etf_sectors = {
            "SPY": "Market",
            "QQQ": "Technology",
            "XLK": "Technology",
            "XLF": "Financials",
            "XLV": "Healthcare",
            "XLE": "Energy",
            "XLI": "Industrials",
            "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples",
            "XLU": "Utilities",
            "XLB": "Materials",
            "XLRE": "Real Estate",
            "XLC": "Communication Services",
        }

        if symbol in etf_sectors:
            self._sector_cache[symbol] = etf_sectors[symbol]
            return etf_sectors[symbol]

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get("sector", "Unknown")
            self._sector_cache[symbol] = sector
            return sector

        except Exception as e:
            logger.debug(f"Could not get sector for {symbol}: {e}")
            self._sector_cache[symbol] = "Unknown"
            return "Unknown"

    def _calculate_scenario_pnl(
        self,
        positions: Dict[str, float],
        scenario: StressScenario,
        sector_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Calculate P&L for a single scenario.

        Args:
            positions: Dict of symbol -> position value
            scenario: StressScenario to apply
            sector_map: Dict of symbol -> sector

        Returns:
            Dict with P&L breakdown
        """
        total_value = sum(positions.values())
        if total_value == 0:
            return {"pnl": 0, "pnl_pct": 0, "position_impacts": {}}

        total_pnl = 0.0
        position_impacts = {}

        for symbol, value in positions.items():
            sector = sector_map.get(symbol, "Unknown")

            # Get sector-specific shock or use base SPY shock
            if scenario.sector_adjustments and sector in scenario.sector_adjustments:
                shock = scenario.sector_adjustments[sector]
            else:
                # Apply correlation-adjusted SPY shock
                # Higher correlation = closer to market shock
                base_shock = scenario.spy_shock
                # Add some randomness based on correlation
                noise = (1 - scenario.correlation) * np.random.uniform(-0.1, 0.1)
                shock = base_shock * scenario.correlation + noise

            # Calculate position P&L
            position_pnl = value * shock
            total_pnl += position_pnl

            position_impacts[symbol] = {
                "value": value,
                "sector": sector,
                "shock": shock,
                "pnl": position_pnl,
                "pnl_pct": shock,
            }

        return {
            "pnl": total_pnl,
            "pnl_pct": total_pnl / total_value if total_value > 0 else 0,
            "position_impacts": position_impacts,
        }

    async def run_stress_test(
        self, positions: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all stress scenarios against portfolio.

        Args:
            positions: Dict of symbol -> position value (in dollars)

        Returns:
            Dict of scenario_name -> results
        """
        if not positions:
            logger.warning("No positions to stress test")
            return {}

        total_value = sum(positions.values())
        logger.info(
            f"Running stress tests on portfolio: ${total_value:,.2f} across {len(positions)} positions"
        )

        # Get sectors for all positions
        sector_map = {}
        for symbol in positions.keys():
            sector_map[symbol] = await self._get_sector(symbol)

        results = {}

        for scenario_name, scenario in self.scenarios.items():
            pnl_result = self._calculate_scenario_pnl(positions, scenario, sector_map)

            passes = abs(pnl_result["pnl_pct"]) <= self.max_acceptable_loss

            results[scenario_name] = {
                "scenario": scenario.name,
                "description": scenario.description,
                "pnl": pnl_result["pnl"],
                "pnl_pct": pnl_result["pnl_pct"],
                "spy_shock": scenario.spy_shock,
                "vix_spike": scenario.vix_spike,
                "passes": passes,
                "max_acceptable_loss": self.max_acceptable_loss,
                "position_impacts": pnl_result["position_impacts"],
            }

            status = "PASS" if passes else "FAIL"
            logger.info(
                f"  {scenario.name}: {pnl_result['pnl_pct']:.1%} loss (${pnl_result['pnl']:,.2f}) - {status}"
            )

        # Summary statistics
        failed_scenarios = [name for name, r in results.items() if not r["passes"]]
        if failed_scenarios:
            logger.warning(
                f"⚠️ STRESS TEST FAILURES: {len(failed_scenarios)}/{len(results)} scenarios exceeded max loss"
            )
            logger.warning(f"   Failed scenarios: {', '.join(failed_scenarios)}")
        else:
            logger.info(f"✓ All {len(results)} stress scenarios passed")

        return results

    async def run_single_scenario(
        self, positions: Dict[str, float], scenario_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single stress scenario.

        Args:
            positions: Dict of symbol -> position value
            scenario_name: Name of scenario to run

        Returns:
            Scenario results or None if scenario not found
        """
        if scenario_name not in self.scenarios:
            logger.error(f"Unknown scenario: {scenario_name}")
            return None

        scenario = self.scenarios[scenario_name]

        # Get sectors
        sector_map = {}
        for symbol in positions.keys():
            sector_map[symbol] = await self._get_sector(symbol)

        pnl_result = self._calculate_scenario_pnl(positions, scenario, sector_map)
        passes = abs(pnl_result["pnl_pct"]) <= self.max_acceptable_loss

        return {
            "scenario": scenario.name,
            "description": scenario.description,
            "pnl": pnl_result["pnl"],
            "pnl_pct": pnl_result["pnl_pct"],
            "passes": passes,
            "position_impacts": pnl_result["position_impacts"],
        }

    def get_worst_case_scenario(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """
        Get the worst-case scenario from stress test results.

        Args:
            results: Results from run_stress_test()

        Returns:
            Name of worst scenario
        """
        if not results:
            return None

        worst = min(results.items(), key=lambda x: x[1]["pnl_pct"])
        return worst[0]

    def get_stress_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report from stress test results.

        Args:
            results: Results from run_stress_test()

        Returns:
            Summary report dict
        """
        if not results:
            return {"error": "No results"}

        pnl_pcts = [r["pnl_pct"] for r in results.values()]
        passed = sum(1 for r in results.values() if r["passes"])
        failed = len(results) - passed

        worst_scenario = self.get_worst_case_scenario(results)
        worst_result = results[worst_scenario] if worst_scenario else None

        return {
            "timestamp": datetime.now().isoformat(),
            "scenarios_tested": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(results) if results else 0,
            "average_loss": np.mean(pnl_pcts),
            "worst_case": {
                "scenario": worst_scenario,
                "pnl_pct": worst_result["pnl_pct"] if worst_result else 0,
                "description": worst_result["description"] if worst_result else "",
            },
            "max_acceptable_loss": self.max_acceptable_loss,
            "recommendation": "REDUCE RISK" if failed > 0 else "ACCEPTABLE",
        }

    def add_custom_scenario(
        self,
        name: str,
        spy_shock: float,
        description: str = "",
        vix_spike: Optional[float] = None,
        correlation: float = 0.80,
        sector_adjustments: Optional[Dict[str, float]] = None,
    ):
        """
        Add a custom stress scenario.

        Args:
            name: Scenario identifier
            spy_shock: SPY return under scenario
            description: Human-readable description
            vix_spike: VIX level (optional)
            correlation: Cross-asset correlation
            sector_adjustments: Sector-specific shocks
        """
        self.scenarios[name] = StressScenario(
            name=name,
            description=description or f"Custom scenario with {spy_shock:.0%} SPY shock",
            spy_shock=spy_shock,
            vix_spike=vix_spike,
            correlation=correlation,
            sector_adjustments=sector_adjustments,
        )
        logger.info(f"Added custom stress scenario: {name}")


async def run_stress_test_cli(broker=None):
    """
    Command-line interface for running stress tests.

    Usage:
        python -m utils.stress_tester
    """
    # Example positions for testing
    example_positions = {
        "AAPL": 10000,
        "MSFT": 8000,
        "GOOGL": 7000,
        "AMZN": 6000,
        "NVDA": 5000,
        "JPM": 4000,
        "XLE": 3000,
        "XLU": 2000,
    }

    tester = StressTester(broker=broker)
    results = await tester.run_stress_test(example_positions)

    print("\n" + "=" * 60)
    print("STRESS TEST REPORT")
    print("=" * 60)

    report = tester.get_stress_report(results)
    print(f"\nScenarios Tested: {report['scenarios_tested']}")
    print(f"Passed: {report['passed']} | Failed: {report['failed']}")
    print(f"Pass Rate: {report['pass_rate']:.0%}")
    print(f"\nAverage Loss Across Scenarios: {report['average_loss']:.1%}")
    print(
        f"Worst Case: {report['worst_case']['scenario']} ({report['worst_case']['pnl_pct']:.1%})"
    )
    print(f"\nRecommendation: {report['recommendation']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_stress_test_cli())
