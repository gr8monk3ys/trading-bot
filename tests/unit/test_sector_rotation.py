#!/usr/bin/env python3
"""
Unit tests for utils/sector_rotation.py

Tests SectorRotator class and EconomicPhase enum.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from utils.sector_rotation import EconomicPhase, SectorRotator

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_broker():
    """Mock broker for testing."""
    broker = MagicMock()
    broker.get_bars = AsyncMock()
    return broker


@pytest.fixture
def rotator(mock_broker):
    """Default sector rotator."""
    return SectorRotator(mock_broker)


@pytest.fixture
def rotator_etf(mock_broker):
    """Sector rotator configured for ETF trading."""
    return SectorRotator(mock_broker, use_etfs=True)


@pytest.fixture
def mock_bars():
    """Create mock bars with OHLCV data."""

    def create_bars(prices, volumes=None):
        bars = []
        for i, price in enumerate(prices):
            bar = MagicMock()
            bar.open = price * 0.99
            bar.high = price * 1.01
            bar.low = price * 0.98
            bar.close = price
            bar.volume = volumes[i] if volumes else 1000000
            bars.append(bar)
        return bars

    return create_bars


# ============================================================================
# EconomicPhase Enum Tests
# ============================================================================


class TestEconomicPhaseEnum:
    """Test EconomicPhase enum."""

    def test_early_expansion_value(self):
        """Test early expansion value."""
        assert EconomicPhase.EARLY_EXPANSION.value == "early_expansion"

    def test_late_expansion_value(self):
        """Test late expansion value."""
        assert EconomicPhase.LATE_EXPANSION.value == "late_expansion"

    def test_contraction_value(self):
        """Test contraction value."""
        assert EconomicPhase.CONTRACTION.value == "contraction"

    def test_trough_value(self):
        """Test trough value."""
        assert EconomicPhase.TROUGH.value == "trough"

    def test_all_phases_defined(self):
        """Test all 4 phases are defined."""
        phases = list(EconomicPhase)
        assert len(phases) == 4


# ============================================================================
# SectorRotator Initialization Tests
# ============================================================================


class TestSectorRotatorInit:
    """Test SectorRotator initialization."""

    def test_default_init(self, rotator):
        """Test default initialization."""
        assert rotator.use_etfs is False
        assert rotator._current_phase is None
        assert rotator._last_phase_check is None
        assert rotator._phase_cache_minutes == 60

    def test_etf_mode_init(self, rotator_etf):
        """Test ETF mode initialization."""
        assert rotator_etf.use_etfs is True

    def test_sector_etfs_defined(self, rotator):
        """Test sector ETFs are defined."""
        assert len(rotator.SECTOR_ETFS) == 11
        assert "XLK" in rotator.SECTOR_ETFS
        assert "XLF" in rotator.SECTOR_ETFS
        assert "XLV" in rotator.SECTOR_ETFS

    def test_phase_allocations_defined(self, rotator):
        """Test phase allocations are defined for all phases."""
        for phase in EconomicPhase:
            assert phase in rotator.PHASE_ALLOCATIONS
            assert len(rotator.PHASE_ALLOCATIONS[phase]) == 11

    def test_sector_stocks_defined(self, rotator):
        """Test sector stocks are defined."""
        assert len(rotator.SECTOR_STOCKS) == 11
        assert "AAPL" in rotator.SECTOR_STOCKS["XLK"]
        assert "JPM" in rotator.SECTOR_STOCKS["XLF"]


# ============================================================================
# Detect Economic Phase Tests
# ============================================================================


class TestDetectEconomicPhase:
    """Test detect_economic_phase method."""

    @pytest.mark.asyncio
    async def test_returns_tuple(self, rotator, mock_broker, mock_bars):
        """Test returns tuple of (phase, confidence)."""
        # Create 200 days of bull market data
        prices = [100 + i * 0.1 for i in range(200)]  # Steady uptrend
        mock_broker.get_bars.return_value = mock_bars(prices)

        result = await rotator.detect_economic_phase()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], EconomicPhase)
        assert isinstance(result[1], float)

    @pytest.mark.asyncio
    async def test_early_expansion_detection(self, rotator, mock_broker, mock_bars):
        """Test early expansion phase detection (strong uptrend, low volatility)."""
        # Strong uptrend with low volatility
        prices = [100 + i * 0.2 for i in range(200)]  # Steady strong uptrend
        mock_broker.get_bars.return_value = mock_bars(prices)

        phase, confidence = await rotator.detect_economic_phase()

        assert phase == EconomicPhase.EARLY_EXPANSION
        assert confidence >= 0.5

    @pytest.mark.asyncio
    async def test_contraction_detection(self, rotator, mock_broker, mock_bars):
        """Test contraction phase detection (downtrend, high volatility)."""
        # Downtrend with high volatility
        base = 150
        prices = []
        for i in range(200):
            # Declining with high volatility
            prices.append(base - i * 0.3 + (i % 10) * 2)  # Volatile decline
        mock_broker.get_bars.return_value = mock_bars(prices)

        phase, confidence = await rotator.detect_economic_phase()

        # Should detect contraction or trough
        assert phase in [EconomicPhase.CONTRACTION, EconomicPhase.TROUGH]

    @pytest.mark.asyncio
    async def test_returns_default_on_insufficient_data(self, rotator, mock_broker, mock_bars):
        """Test returns default on insufficient data."""
        mock_broker.get_bars.return_value = mock_bars([100] * 10)  # Only 10 days

        phase, confidence = await rotator.detect_economic_phase()

        assert phase == EconomicPhase.LATE_EXPANSION
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_returns_default_on_none_data(self, rotator, mock_broker):
        """Test returns default when data is None."""
        mock_broker.get_bars.return_value = None

        phase, confidence = await rotator.detect_economic_phase()

        assert phase == EconomicPhase.LATE_EXPANSION
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_caches_result(self, rotator, mock_broker, mock_bars):
        """Test result is cached."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        # First call
        result1 = await rotator.detect_economic_phase()

        # Second call should use cache (broker not called again)
        mock_broker.get_bars.reset_mock()
        result2 = await rotator.detect_economic_phase()

        mock_broker.get_bars.assert_not_called()
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_expires(self, rotator, mock_broker, mock_bars):
        """Test cache expires after timeout."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        # First call
        await rotator.detect_economic_phase()

        # Expire the cache
        rotator._last_phase_check = datetime.now() - timedelta(minutes=120)

        # Second call should fetch fresh data
        mock_broker.get_bars.reset_mock()
        await rotator.detect_economic_phase()

        mock_broker.get_bars.assert_called()

    @pytest.mark.asyncio
    async def test_handles_exception(self, rotator, mock_broker):
        """Test handles exception gracefully."""
        mock_broker.get_bars.side_effect = Exception("API error")

        phase, confidence = await rotator.detect_economic_phase()

        assert phase == EconomicPhase.LATE_EXPANSION
        assert confidence == 0.5


# ============================================================================
# Get Sector Allocations Tests
# ============================================================================


class TestGetSectorAllocations:
    """Test get_sector_allocations method."""

    @pytest.mark.asyncio
    async def test_returns_all_sectors(self, rotator, mock_broker, mock_bars):
        """Test returns allocations for all sectors."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        allocations = await rotator.get_sector_allocations()

        assert len(allocations) == 11
        for etf in rotator.SECTOR_ETFS.keys():
            assert etf in allocations

    @pytest.mark.asyncio
    async def test_allocations_sum_to_base(self, rotator, mock_broker, mock_bars):
        """Test allocations sum to base allocation."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        allocations = await rotator.get_sector_allocations(base_allocation=1.0)

        total = sum(allocations.values())
        assert pytest.approx(total, abs=0.01) == 1.0

    @pytest.mark.asyncio
    async def test_custom_base_allocation(self, rotator, mock_broker, mock_bars):
        """Test custom base allocation."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        allocations = await rotator.get_sector_allocations(base_allocation=0.5)

        total = sum(allocations.values())
        assert pytest.approx(total, abs=0.01) == 0.5

    @pytest.mark.asyncio
    async def test_blends_toward_equal_weight_on_low_confidence(
        self, rotator, mock_broker, mock_bars
    ):
        """Test allocations blend toward equal weight when confidence is low."""
        # Force low confidence by returning insufficient data
        mock_broker.get_bars.return_value = mock_bars([100] * 10)

        allocations = await rotator.get_sector_allocations()

        # With low confidence, allocations should be closer to equal weight
        for alloc in allocations.values():
            # Should be somewhat close to equal weight
            assert 0.05 <= alloc <= 0.15


# ============================================================================
# Get Recommended Stocks Tests
# ============================================================================


class TestGetRecommendedStocks:
    """Test get_recommended_stocks method."""

    @pytest.mark.asyncio
    async def test_returns_list(self, rotator, mock_broker, mock_bars):
        """Test returns list of stock symbols."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        stocks = await rotator.get_recommended_stocks()

        assert isinstance(stocks, list)
        assert len(stocks) > 0
        assert all(isinstance(s, str) for s in stocks)

    @pytest.mark.asyncio
    async def test_respects_top_n(self, rotator, mock_broker, mock_bars):
        """Test respects top_n parameter."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        stocks = await rotator.get_recommended_stocks(top_n=10)

        assert len(stocks) <= 10

    @pytest.mark.asyncio
    async def test_respects_stocks_per_sector(self, rotator, mock_broker, mock_bars):
        """Test respects stocks_per_sector parameter."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        stocks = await rotator.get_recommended_stocks(top_n=50, stocks_per_sector=2)

        # Should have stocks from multiple sectors but max 2 per sector
        assert len(stocks) <= 22  # 11 sectors * 2 stocks max

    @pytest.mark.asyncio
    async def test_includes_stocks_from_top_sectors(self, rotator, mock_broker, mock_bars):
        """Test includes stocks from top sectors."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        stocks = await rotator.get_recommended_stocks(top_n=20)

        # Should include stocks from multiple sectors
        # Check that at least some known stocks are present
        known_stocks = set(sum(rotator.SECTOR_STOCKS.values(), []))
        assert any(s in known_stocks for s in stocks)


# ============================================================================
# Get Sector Momentum Tests
# ============================================================================


class TestGetSectorMomentum:
    """Test get_sector_momentum method."""

    @pytest.mark.asyncio
    async def test_returns_all_sectors(self, rotator, mock_broker, mock_bars):
        """Test returns momentum for all sectors."""
        prices = [100 + i * 0.5 for i in range(30)]  # Uptrend
        mock_broker.get_bars.return_value = mock_bars(prices)

        momentum = await rotator.get_sector_momentum()

        # Should have entries for all 11 sectors
        assert len(momentum) == 11

    @pytest.mark.asyncio
    async def test_positive_momentum_for_uptrend(self, rotator, mock_broker, mock_bars):
        """Test positive momentum for uptrend."""
        # Strong uptrend: 20 days from 100 to 120 = 20% gain
        prices = [100 + i for i in range(30)]  # 100 to 129
        mock_broker.get_bars.return_value = mock_bars(prices)

        momentum = await rotator.get_sector_momentum()

        # All sectors should have positive momentum
        for _etf, mom in momentum.items():
            assert mom > 0

    @pytest.mark.asyncio
    async def test_negative_momentum_for_downtrend(self, rotator, mock_broker, mock_bars):
        """Test negative momentum for downtrend."""
        # Downtrend: 100 to 80
        prices = [100 - i * 0.5 for i in range(30)]  # 100 to 85.5
        mock_broker.get_bars.return_value = mock_bars(prices)

        momentum = await rotator.get_sector_momentum()

        # All sectors should have negative momentum
        for _etf, mom in momentum.items():
            assert mom < 0

    @pytest.mark.asyncio
    async def test_handles_insufficient_data(self, rotator, mock_broker, mock_bars):
        """Test handles insufficient data."""
        mock_broker.get_bars.return_value = mock_bars([100] * 5)  # Only 5 days

        momentum = await rotator.get_sector_momentum()

        # Should return 0 for sectors with insufficient data
        for mom in momentum.values():
            assert mom == 0

    @pytest.mark.asyncio
    async def test_handles_none_data(self, rotator, mock_broker):
        """Test handles None data."""
        mock_broker.get_bars.return_value = None

        momentum = await rotator.get_sector_momentum()

        # Should return 0 for all sectors
        for mom in momentum.values():
            assert mom == 0


# ============================================================================
# Get Price Data Tests
# ============================================================================


class TestGetPriceData:
    """Test _get_price_data helper method."""

    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self, rotator, mock_broker, mock_bars):
        """Test returns list of dicts with OHLCV data."""
        mock_broker.get_bars.return_value = mock_bars([100, 101, 102])

        data = await rotator._get_price_data("SPY", days=3)

        assert isinstance(data, list)
        assert len(data) == 3
        assert all("open" in d for d in data)
        assert all("high" in d for d in data)
        assert all("low" in d for d in data)
        assert all("close" in d for d in data)
        assert all("volume" in d for d in data)

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self, rotator, mock_broker):
        """Test returns None on API error."""
        mock_broker.get_bars.side_effect = Exception("API error")

        data = await rotator._get_price_data("SPY")

        assert data is None

    @pytest.mark.asyncio
    async def test_returns_none_on_none_bars(self, rotator, mock_broker):
        """Test returns None when bars is None."""
        mock_broker.get_bars.return_value = None

        data = await rotator._get_price_data("SPY")

        assert data is None


# ============================================================================
# Get Sector Report Tests
# ============================================================================


class TestGetSectorReport:
    """Test get_sector_report method."""

    @pytest.mark.asyncio
    async def test_returns_complete_report(self, rotator, mock_broker, mock_bars):
        """Test returns complete report structure."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        report = await rotator.get_sector_report()

        assert "phase" in report
        assert "phase_confidence" in report
        assert "allocations" in report
        assert "momentum" in report
        assert "recommended_stocks" in report
        assert "overweight_sectors" in report
        assert "underweight_sectors" in report

    @pytest.mark.asyncio
    async def test_phase_is_string(self, rotator, mock_broker, mock_bars):
        """Test phase is returned as string value."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        report = await rotator.get_sector_report()

        assert isinstance(report["phase"], str)
        assert report["phase"] in [p.value for p in EconomicPhase]

    @pytest.mark.asyncio
    async def test_allocations_are_sorted(self, rotator, mock_broker, mock_bars):
        """Test allocations are sorted by weight."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        report = await rotator.get_sector_report()

        allocs = list(report["allocations"].values())
        # Check allocations are sorted in descending order
        assert allocs == sorted(allocs, reverse=True)

    @pytest.mark.asyncio
    async def test_overweight_sectors_above_threshold(self, rotator, mock_broker, mock_bars):
        """Test overweight sectors are above 10% allocation."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        report = await rotator.get_sector_report()

        for sector in report["overweight_sectors"]:
            assert report["allocations"][sector] > 0.10

    @pytest.mark.asyncio
    async def test_underweight_sectors_below_threshold(self, rotator, mock_broker, mock_bars):
        """Test underweight sectors are below 8% allocation."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        report = await rotator.get_sector_report()

        for sector in report["underweight_sectors"]:
            assert report["allocations"][sector] < 0.08


# ============================================================================
# Phase-Specific Allocation Tests
# ============================================================================


class TestPhaseAllocations:
    """Test phase-specific allocation weights."""

    def test_early_expansion_favors_growth(self, rotator):
        """Test early expansion favors growth sectors."""
        allocs = rotator.PHASE_ALLOCATIONS[EconomicPhase.EARLY_EXPANSION]

        # Tech and consumer discretionary should be overweight
        assert allocs["XLK"] > 1.0
        assert allocs["XLY"] > 1.0

        # Defensive sectors should be underweight
        assert allocs["XLU"] < 1.0
        assert allocs["XLP"] < 1.0

    def test_late_expansion_favors_cyclicals(self, rotator):
        """Test late expansion favors cyclical sectors."""
        allocs = rotator.PHASE_ALLOCATIONS[EconomicPhase.LATE_EXPANSION]

        # Energy and materials should be overweight
        assert allocs["XLE"] > 1.0
        assert allocs["XLB"] > 1.0

    def test_contraction_favors_defensives(self, rotator):
        """Test contraction favors defensive sectors."""
        allocs = rotator.PHASE_ALLOCATIONS[EconomicPhase.CONTRACTION]

        # Healthcare, utilities, staples should be overweight
        assert allocs["XLV"] > 1.0
        assert allocs["XLU"] > 1.0
        assert allocs["XLP"] > 1.0

        # Real estate and discretionary should be underweight
        assert allocs["XLRE"] < 1.0
        assert allocs["XLY"] < 1.0

    def test_trough_favors_recovery_plays(self, rotator):
        """Test trough favors recovery plays."""
        allocs = rotator.PHASE_ALLOCATIONS[EconomicPhase.TROUGH]

        # Tech and financials should be overweight
        assert allocs["XLK"] > 1.0
        assert allocs["XLF"] > 1.0


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_handles_flat_market(self, rotator, mock_broker, mock_bars):
        """Test handles flat market (zero momentum)."""
        prices = [100.0] * 200  # Completely flat
        mock_broker.get_bars.return_value = mock_bars(prices)

        phase, confidence = await rotator.detect_economic_phase()

        # Should return some valid phase
        assert phase in EconomicPhase
        assert 0 <= confidence <= 1

    @pytest.mark.asyncio
    async def test_handles_extreme_volatility(self, rotator, mock_broker, mock_bars):
        """Test handles extreme volatility."""
        # Wild swings
        prices = [100 + (50 if i % 2 == 0 else -50) for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        phase, confidence = await rotator.detect_economic_phase()

        # Should detect contraction due to high volatility
        assert phase in EconomicPhase

    @pytest.mark.asyncio
    async def test_empty_stock_list(self, rotator, mock_broker, mock_bars):
        """Test with empty recommended stocks."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        stocks = await rotator.get_recommended_stocks(top_n=0)

        assert stocks == []

    @pytest.mark.asyncio
    async def test_momentum_with_exactly_20_days(self, rotator, mock_broker, mock_bars):
        """Test momentum calculation with exactly 20 days."""
        prices = [100 + i for i in range(20)]  # Exactly 20 days
        mock_broker.get_bars.return_value = mock_bars(prices)

        momentum = await rotator.get_sector_momentum()

        # Should calculate momentum (all sectors get same data from mock)
        for mom in momentum.values():
            assert isinstance(mom, (int, float))

    @pytest.mark.asyncio
    async def test_allocation_with_zero_base(self, rotator, mock_broker, mock_bars):
        """Test allocation with zero base allocation."""
        prices = [100 + i * 0.1 for i in range(200)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        allocations = await rotator.get_sector_allocations(base_allocation=0.0)

        # All allocations should be zero
        assert all(a == 0.0 for a in allocations.values())
