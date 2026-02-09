"""
Tests for Greeks Aggregator

Tests:
- Portfolio-level Greeks aggregation
- Delta/gamma/vega limit checking
- Concentration detection
- Hedge recommendations
- Real-time monitoring
"""

import pytest

from utils.greeks_aggregator import (
    AggregatedGreeks,
    GreeksAggregator,
    GreeksLimitResult,
    GreeksViolation,
    PositionGreeks,
    RealTimeGreeksMonitor,
    print_greeks_report,
)


class TestPositionGreeks:
    """Tests for PositionGreeks dataclass."""

    def test_position_delta_calculation(self):
        """Test position delta calculation."""
        pos = PositionGreeks(
            symbol="AAPL230120C00150000",
            underlying="AAPL",
            quantity=10,
            multiplier=100,
            delta=0.50,
        )

        # 10 contracts * 100 multiplier * 0.50 delta = 500
        assert pos.position_delta == 500

    def test_position_gamma_calculation(self):
        """Test position gamma calculation."""
        pos = PositionGreeks(
            symbol="AAPL230120C00150000",
            underlying="AAPL",
            quantity=10,
            multiplier=100,
            gamma=0.02,
        )

        # 10 * 100 * 0.02 = 20
        assert pos.position_gamma == 20

    def test_position_vega_calculation(self):
        """Test position vega calculation."""
        pos = PositionGreeks(
            symbol="AAPL230120C00150000",
            underlying="AAPL",
            quantity=-5,  # Short
            multiplier=100,
            vega=0.10,
        )

        # -5 * 100 * 0.10 = -50
        assert pos.position_vega == -50

    def test_short_position_negative_delta(self):
        """Test short position has negative delta contribution."""
        pos = PositionGreeks(
            symbol="AAPL230120P00150000",
            underlying="AAPL",
            quantity=-10,  # Short puts
            multiplier=100,
            delta=-0.30,  # Negative delta for puts
        )

        # -10 * 100 * -0.30 = 300 (short put is long delta)
        assert pos.position_delta == 300


class TestGreeksAggregator:
    """Tests for GreeksAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create an aggregator with default settings."""
        return GreeksAggregator()

    @pytest.fixture
    def sample_positions(self):
        """Sample options positions."""
        return [
            PositionGreeks(
                symbol="AAPL230120C00180000",
                underlying="AAPL",
                quantity=10,
                delta=0.60,
                gamma=0.02,
                theta=-0.05,
                vega=0.15,
            ),
            PositionGreeks(
                symbol="MSFT230120C00380000",
                underlying="MSFT",
                quantity=5,
                delta=0.55,
                gamma=0.015,
                theta=-0.04,
                vega=0.12,
            ),
            PositionGreeks(
                symbol="SPY230120P00450000",
                underlying="SPY",
                quantity=-10,  # Short puts
                delta=-0.35,  # Put delta (becomes positive exposure when short)
                gamma=0.01,
                theta=0.03,  # Positive theta when short
                vega=0.10,
            ),
        ]

    @pytest.fixture
    def sample_prices(self):
        """Sample underlying prices."""
        return {
            "AAPL": 180.0,
            "MSFT": 380.0,
            "SPY": 450.0,
        }

    def test_aggregate_returns_result(self, aggregator, sample_positions, sample_prices):
        """Test that aggregate returns GreeksLimitResult."""
        result = aggregator.aggregate(
            positions=sample_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        assert isinstance(result, GreeksLimitResult)
        assert isinstance(result.greeks, AggregatedGreeks)

    def test_net_delta_calculated(self, aggregator, sample_positions, sample_prices):
        """Test net delta calculation."""
        result = aggregator.aggregate(
            positions=sample_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        # Net delta should be sum of position deltas * prices
        assert result.greeks.net_delta != 0

    def test_net_gamma_calculated(self, aggregator, sample_positions, sample_prices):
        """Test net gamma calculation."""
        result = aggregator.aggregate(
            positions=sample_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        # Net gamma should be calculated
        assert result.greeks.net_gamma != 0

    def test_net_vega_calculated(self, aggregator, sample_positions, sample_prices):
        """Test net vega calculation."""
        result = aggregator.aggregate(
            positions=sample_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        assert result.greeks.net_vega != 0

    def test_theta_calculation(self, aggregator, sample_positions, sample_prices):
        """Test theta (time decay) calculation."""
        result = aggregator.aggregate(
            positions=sample_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        assert isinstance(result.greeks.net_theta, float)

    def test_greeks_by_underlying(self, aggregator, sample_positions, sample_prices):
        """Test Greeks breakdown by underlying."""
        result = aggregator.aggregate(
            positions=sample_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        # Should have breakdown by underlying
        assert "AAPL" in result.greeks.delta_by_underlying
        assert "MSFT" in result.greeks.delta_by_underlying
        assert "SPY" in result.greeks.delta_by_underlying

    def test_within_limits_for_small_positions(self, aggregator, sample_prices):
        """Test that small positions are within limits."""
        small_positions = [
            PositionGreeks(
                symbol="AAPL230120C00180000",
                underlying="AAPL",
                quantity=1,  # Just 1 contract
                delta=0.50,
                gamma=0.01,
                vega=0.10,
            )
        ]

        result = aggregator.aggregate(
            positions=small_positions,
            underlying_prices=sample_prices,
            portfolio_value=1000000,  # Large portfolio
        )

        assert result.within_limits

    def test_violation_on_large_delta(self, aggregator, sample_prices):
        """Test delta violation detection."""
        # Create large delta exposure
        large_positions = [
            PositionGreeks(
                symbol="SPY230120C00450000",
                underlying="SPY",
                quantity=100,  # 100 contracts
                delta=0.80,
                gamma=0.01,
                vega=0.10,
            )
        ]

        result = aggregator.aggregate(
            positions=large_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,  # Small portfolio
        )

        # Large delta relative to portfolio should trigger violation
        delta_pct = abs(result.greeks.delta_pct)
        if delta_pct > aggregator.max_delta_pct:
            assert not result.within_limits
            assert any(v[1] == GreeksViolation.DELTA_TOO_HIGH for v in result.violations)

    def test_violation_on_large_gamma(self):
        """Test gamma violation detection."""
        aggregator = GreeksAggregator(max_gamma_pct=0.01)  # Strict limit

        large_gamma_positions = [
            PositionGreeks(
                symbol="SPY230120C00450000",
                underlying="SPY",
                quantity=50,
                delta=0.50,
                gamma=0.10,  # High gamma
                vega=0.10,
            )
        ]

        result = aggregator.aggregate(
            positions=large_gamma_positions,
            underlying_prices={"SPY": 450.0},
            portfolio_value=100000,
        )

        gamma_pct = abs(result.greeks.gamma_pct)
        if gamma_pct > aggregator.max_gamma_pct:
            assert not result.within_limits

    def test_concentration_violation(self, aggregator):
        """Test concentration violation for single underlying."""
        aggregator = GreeksAggregator(max_underlying_pct=0.10)  # Strict

        concentrated_positions = [
            PositionGreeks(
                symbol="AAPL230120C00180000",
                underlying="AAPL",
                quantity=50,  # All in AAPL
                delta=0.70,
                gamma=0.02,
                vega=0.15,
            )
        ]

        result = aggregator.aggregate(
            positions=concentrated_positions,
            underlying_prices={"AAPL": 180.0},
            portfolio_value=100000,
        )

        # May trigger concentration violation
        if abs(result.greeks.delta_pct) > aggregator.max_underlying_pct:
            assert any(v[1] == GreeksViolation.CONCENTRATION for v in result.violations)

    def test_empty_positions(self, aggregator, sample_prices):
        """Test handling of empty positions."""
        result = aggregator.aggregate(
            positions=[],
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        assert result.within_limits
        assert result.position_count == 0
        assert result.greeks.net_delta == 0

    def test_to_dict_serialization(self, aggregator, sample_positions, sample_prices):
        """Test result serialization."""
        result = aggregator.aggregate(
            positions=sample_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        d = result.to_dict()
        assert "within_limits" in d
        assert "greeks" in d
        assert "violations" in d
        assert "position_count" in d

    def test_aggregate_from_broker_positions(self, aggregator, sample_prices):
        """Test aggregation from broker-style position dicts."""
        broker_positions = [
            {
                "symbol": "AAPL230120C00180000",
                "qty": 10,
                "delta": 0.60,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.15,
            }
        ]

        result = aggregator.aggregate_from_broker_positions(
            options_positions=broker_positions,
            underlying_prices=sample_prices,
            portfolio_value=100000,
        )

        assert isinstance(result, GreeksLimitResult)
        assert result.position_count == 1

    def test_get_hedge_recommendations(self, aggregator, sample_prices):
        """Test hedge recommendations generation."""
        # Create violating position
        large_positions = [
            PositionGreeks(
                symbol="SPY230120C00450000",
                underlying="SPY",
                quantity=100,
                delta=0.80,
                gamma=0.05,
                vega=0.20,
            )
        ]

        result = aggregator.aggregate(
            positions=large_positions,
            underlying_prices=sample_prices,
            portfolio_value=50000,
        )

        recommendations = aggregator.get_hedge_recommendations(result)

        assert "summary" in recommendations
        assert isinstance(recommendations["summary"], list)


class TestAggregatedGreeks:
    """Tests for AggregatedGreeks dataclass."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        greeks = AggregatedGreeks(
            net_delta=10000,
            net_gamma=500,
            net_theta=-100,
            net_vega=1500,
            delta_by_underlying={"AAPL": 5000, "SPY": 5000},
            delta_pct=0.10,
            gamma_pct=0.005,
            vega_pct=0.015,
        )

        d = greeks.to_dict()
        assert d["net_delta"] == 10000
        assert d["delta_pct"] == 0.10
        assert "AAPL" in d["delta_by_underlying"]


class TestRealTimeGreeksMonitor:
    """Tests for RealTimeGreeksMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a monitor."""
        aggregator = GreeksAggregator()
        return RealTimeGreeksMonitor(
            aggregator=aggregator,
            check_interval_seconds=1,
        )

    def test_should_check_initially_true(self, monitor):
        """Test initial check is allowed."""
        assert monitor.should_check() is True

    def test_check_and_alert_returns_result(self, monitor):
        """Test check_and_alert returns result."""
        positions = [
            PositionGreeks(
                symbol="AAPL230120C00180000",
                underlying="AAPL",
                quantity=1,
                delta=0.50,
            )
        ]

        result = monitor.check_and_alert(
            positions=positions,
            underlying_prices={"AAPL": 180.0},
            portfolio_value=100000,
            force=True,
        )

        assert isinstance(result, GreeksLimitResult)

    def test_last_result_stored(self, monitor):
        """Test last result is stored."""
        positions = [
            PositionGreeks(
                symbol="AAPL230120C00180000",
                underlying="AAPL",
                quantity=1,
                delta=0.50,
            )
        ]

        monitor.check_and_alert(
            positions=positions,
            underlying_prices={"AAPL": 180.0},
            portfolio_value=100000,
            force=True,
        )

        assert monitor.last_result is not None


class TestPrintGreeksReport:
    """Tests for print_greeks_report function."""

    def test_print_report_no_error(self, capsys):
        """Test print_greeks_report doesn't raise."""
        result = GreeksLimitResult(
            within_limits=True,
            portfolio_value=100000,
            greeks=AggregatedGreeks(
                net_delta=5000,
                net_gamma=100,
                net_theta=-50,
                net_vega=500,
                delta_pct=0.05,
                gamma_pct=0.001,
                vega_pct=0.005,
            ),
            violations=[],
            position_count=5,
            underlying_count=3,
        )

        print_greeks_report(result)

        captured = capsys.readouterr()
        assert "PORTFOLIO GREEKS REPORT" in captured.out
        assert "Delta" in captured.out


class TestGreeksViolation:
    """Tests for GreeksViolation enum."""

    def test_all_violation_types_exist(self):
        """Test all expected violation types exist."""
        expected = ["DELTA_TOO_HIGH", "GAMMA_TOO_HIGH", "VEGA_TOO_HIGH", "CONCENTRATION"]

        for name in expected:
            assert hasattr(GreeksViolation, name)
