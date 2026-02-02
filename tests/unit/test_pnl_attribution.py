"""
Tests for P&L Attribution System

Tests:
- Daily attribution calculation
- Component breakdown (beta, sector, factor, alpha)
- Period report aggregation
- Sector weight calculation
- Rolling beta calculation
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from utils.pnl_attribution import (
    PnLAttributor,
    DailyAttribution,
    AttributionReport,
    AttributionComponent,
    SECTOR_MAPPING,
    print_attribution_report,
)


class TestPnLAttributor:
    """Tests for PnLAttributor class."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker."""
        broker = MagicMock()
        broker.get_bars = AsyncMock(return_value=None)
        return broker

    @pytest.fixture
    def attributor(self, mock_broker):
        """Create an attributor instance."""
        return PnLAttributor(broker=mock_broker)

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        return {
            "AAPL": 50_000.0,
            "MSFT": 40_000.0,
            "JPM": 30_000.0,
            "XOM": 20_000.0,
            "PG": 10_000.0,
        }

    @pytest.fixture
    def sample_prices(self):
        """Create sample prices."""
        return {
            "AAPL": 175.0,
            "MSFT": 380.0,
            "JPM": 170.0,
            "XOM": 110.0,
            "PG": 160.0,
        }

    def test_sector_weight_calculation(self, attributor, sample_positions):
        """Test sector weight calculation from positions."""
        total = sum(sample_positions.values())
        weights = {s: v / total for s, v in sample_positions.items()}

        sector_weights = attributor._calculate_sector_weights(weights)

        # Check that sectors are correctly assigned
        assert "Technology" in sector_weights  # AAPL, MSFT
        assert "Financials" in sector_weights  # JPM
        assert "Energy" in sector_weights      # XOM
        assert "Consumer Staples" in sector_weights  # PG

        # Check weights sum to 1
        total_sector_weight = sum(sector_weights.values())
        assert 0.99 <= total_sector_weight <= 1.01

    def test_record_daily(self, attributor):
        """Test recording daily data."""
        date = datetime.now()
        positions = {"AAPL": 10_000}

        attributor.record_daily(
            date=date,
            portfolio_return=0.01,
            positions=positions,
            benchmark_return=0.005,
            costs=10.0,
        )

        assert len(attributor._portfolio_returns) == 1
        assert len(attributor._positions_history) == 1
        assert len(attributor._costs_history) == 1
        assert len(attributor._benchmark_returns) == 1

    @pytest.mark.asyncio
    async def test_attribute_daily(self, attributor, sample_positions, sample_prices):
        """Test daily attribution calculation."""
        date = datetime.now()

        # Mock benchmark return
        with patch.object(attributor, '_get_benchmark_return', return_value=0.01):
            with patch.object(attributor, '_get_factor_returns', return_value={}):
                with patch.object(attributor, '_calculate_sector_contribution', return_value=0.002):
                    with patch.object(attributor, '_estimate_factor_exposures', return_value={}):
                        attribution = await attributor.attribute_daily(
                            date=date,
                            positions=sample_positions,
                            prices=sample_prices,
                            portfolio_return=0.015,
                            costs=50.0,
                        )

        assert isinstance(attribution, DailyAttribution)
        assert attribution.total_return == 0.015

    def test_rolling_beta_calculation(self, attributor):
        """Test rolling beta calculation."""
        # Add some return history
        for i in range(30):
            date = datetime.now() - timedelta(days=30 - i)
            port_ret = 0.001 * i + np.random.randn() * 0.005
            bench_ret = 0.0005 * i + np.random.randn() * 0.003
            attributor._portfolio_returns.append((date, port_ret))
            attributor._benchmark_returns.append((date, bench_ret))

        beta = attributor._calculate_rolling_beta()

        # Beta should be a reasonable value
        assert 0.0 <= beta <= 3.0

    def test_rolling_beta_insufficient_data(self, attributor):
        """Test that beta returns 1.0 with insufficient data."""
        # Only add a few data points
        for i in range(5):
            date = datetime.now() - timedelta(days=5 - i)
            attributor._portfolio_returns.append((date, 0.01))
            attributor._benchmark_returns.append((date, 0.005))

        beta = attributor._calculate_rolling_beta()

        # Should return default 1.0
        assert beta == 1.0

    @pytest.mark.asyncio
    async def test_get_attribution_report(self, attributor, sample_positions, sample_prices):
        """Test period attribution report generation."""
        # Record several days
        for i in range(10):
            date = datetime.now() - timedelta(days=10 - i)
            with patch.object(attributor, '_get_benchmark_return', return_value=0.005):
                with patch.object(attributor, '_get_factor_returns', return_value={}):
                    with patch.object(attributor, '_calculate_sector_contribution', return_value=0.001):
                        with patch.object(attributor, '_estimate_factor_exposures', return_value={}):
                            await attributor.attribute_daily(
                                date=date,
                                positions=sample_positions,
                                prices=sample_prices,
                                portfolio_return=0.008 + np.random.randn() * 0.002,
                                costs=10.0,
                            )

        report = await attributor.get_attribution_report()

        assert isinstance(report, AttributionReport)
        assert report.trading_days == 10
        assert "alpha" in report.to_dict()["attribution"]

    @pytest.mark.asyncio
    async def test_empty_report(self, attributor):
        """Test report generation with no data."""
        report = await attributor.get_attribution_report()

        assert report.trading_days == 0
        assert report.total_return == 0.0

    def test_clear_history(self, attributor):
        """Test clearing historical data."""
        attributor._portfolio_returns.append((datetime.now(), 0.01))
        attributor._benchmark_returns.append((datetime.now(), 0.005))

        attributor.clear_history()

        assert len(attributor._portfolio_returns) == 0
        assert len(attributor._benchmark_returns) == 0


class TestDailyAttribution:
    """Tests for DailyAttribution dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        attribution = DailyAttribution(
            date=datetime.now(),
            total_return=0.015,
            gross_return=0.016,
            beta_contribution=0.008,
            sector_contribution=0.002,
            factor_contributions={"momentum": 0.003},
            alpha=0.002,
            cost_drag=0.001,
            market_beta=1.1,
            sector_weights={"Technology": 0.5},
            factor_exposures={"momentum": 0.2},
            r_squared=0.85,
        )

        d = attribution.to_dict()

        assert "date" in d
        assert d["total_return"] == 0.015
        assert d["beta_contribution"] == 0.008
        assert d["alpha"] == 0.002


class TestAttributionReport:
    """Tests for AttributionReport dataclass."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report."""
        return AttributionReport(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            trading_days=22,
            total_return=0.15,
            gross_return=0.16,
            benchmark_return=0.10,
            beta_contribution=0.08,
            sector_contribution=0.02,
            factor_contributions={"momentum": 0.03, "value": 0.01},
            alpha=0.01,
            cost_drag=0.01,
            avg_beta=1.1,
            avg_r_squared=0.75,
            information_ratio=1.5,
            tracking_error=0.05,
            sector_returns={"Technology": 0.08, "Financials": 0.05},
            sector_weights={"Technology": 0.4, "Financials": 0.2},
        )

    def test_to_dict(self, sample_report):
        """Test conversion to dictionary."""
        d = sample_report.to_dict()

        assert "period" in d
        assert d["trading_days"] == 22
        assert d["total_return"] == 0.15
        assert "attribution" in d
        assert d["attribution"]["alpha"] == 0.01

    def test_print_report(self, sample_report, capsys):
        """Test printing the report."""
        print_attribution_report(sample_report)

        captured = capsys.readouterr()
        assert "P&L ATTRIBUTION REPORT" in captured.out
        assert "15.00%" in captured.out  # Total return
        assert "Pure Alpha:" in captured.out


class TestSectorMapping:
    """Tests for sector mapping."""

    def test_major_stocks_have_sectors(self):
        """Test that major stocks have sector mappings."""
        major_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "JNJ"]

        for stock in major_stocks:
            assert stock in SECTOR_MAPPING

    def test_sectors_are_valid(self):
        """Test that all sectors are valid GICS sectors."""
        valid_sectors = {
            "Technology", "Financials", "Healthcare", "Energy",
            "Consumer Discretionary", "Consumer Staples", "Industrials",
            "Materials", "Utilities", "Real Estate", "Communication Services",
        }

        for symbol, sector in SECTOR_MAPPING.items():
            assert sector in valid_sectors, f"{symbol} has invalid sector {sector}"


class TestAttributionComponent:
    """Tests for AttributionComponent enum."""

    def test_all_components(self):
        """Test that all expected components exist."""
        expected = [
            "TOTAL", "ALPHA", "BETA", "SECTOR",
            "FACTOR_MOMENTUM", "FACTOR_VALUE", "FACTOR_SIZE",
            "FACTOR_QUALITY", "FACTOR_VOLATILITY",
            "COSTS", "RESIDUAL",
        ]

        for component in expected:
            assert hasattr(AttributionComponent, component)

    def test_component_values(self):
        """Test component value strings."""
        assert AttributionComponent.ALPHA.value == "alpha"
        assert AttributionComponent.BETA.value == "beta"
        assert AttributionComponent.COSTS.value == "costs"
