#!/usr/bin/env python3
"""
Unit tests for utils/relative_strength.py

Tests RelativeStrengthRanker and RSMomentumFilter classes.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from utils.relative_strength import RelativeStrengthRanker, RSMomentumFilter


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
def ranker(mock_broker):
    """Default relative strength ranker."""
    return RelativeStrengthRanker(mock_broker)


@pytest.fixture
def ranker_custom(mock_broker):
    """Custom relative strength ranker."""
    return RelativeStrengthRanker(
        mock_broker,
        benchmark="QQQ",
        lookback_days=30,
        min_rs_for_long=1.10,
        max_rs_for_short=0.90,
    )


@pytest.fixture
def rs_filter(mock_broker):
    """RS momentum filter."""
    return RSMomentumFilter(mock_broker)


@pytest.fixture
def mock_bars():
    """Create mock bars with prices."""
    def create_bars(prices):
        bars = []
        for price in prices:
            bar = MagicMock()
            bar.close = price
            bars.append(bar)
        return bars
    return create_bars


# ============================================================================
# RelativeStrengthRanker Initialization Tests
# ============================================================================


class TestRelativeStrengthRankerInit:
    """Test RelativeStrengthRanker initialization."""

    def test_default_init(self, ranker):
        """Test default initialization values."""
        assert ranker.benchmark == "SPY"
        assert ranker.lookback_days == 20
        assert ranker.min_rs_for_long == 1.05
        assert ranker.max_rs_for_short == 0.95

    def test_custom_init(self, ranker_custom):
        """Test custom initialization values."""
        assert ranker_custom.benchmark == "QQQ"
        assert ranker_custom.lookback_days == 30
        assert ranker_custom.min_rs_for_long == 1.10
        assert ranker_custom.max_rs_for_short == 0.90

    def test_cache_initialized(self, ranker):
        """Test cache is initialized."""
        assert ranker._benchmark_return is None
        assert ranker._last_calc_date is None


# ============================================================================
# Get Benchmark Return Tests
# ============================================================================


class TestGetBenchmarkReturn:
    """Test get_benchmark_return method."""

    @pytest.mark.asyncio
    async def test_returns_cached_value(self, ranker, mock_broker):
        """Test returns cached value if available."""
        ranker._benchmark_return = 0.05
        ranker._last_calc_date = datetime.now().date()

        result = await ranker.get_benchmark_return()
        assert result == 0.05
        mock_broker.get_bars.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetches_when_no_cache(self, ranker, mock_broker, mock_bars):
        """Test fetches data when no cache."""
        # 25 days of data with 10% increase
        prices = [100.0 + i * 0.5 for i in range(25)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        result = await ranker.get_benchmark_return()
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_caches_result(self, ranker, mock_broker, mock_bars):
        """Test result is cached."""
        prices = [100.0 + i * 0.5 for i in range(25)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        await ranker.get_benchmark_return()

        assert ranker._benchmark_return is not None
        assert ranker._last_calc_date is not None

    @pytest.mark.asyncio
    async def test_returns_zero_on_insufficient_data(self, ranker, mock_broker, mock_bars):
        """Test returns 0 when insufficient data."""
        mock_broker.get_bars.return_value = mock_bars([100.0])

        result = await ranker.get_benchmark_return()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_on_none_bars(self, ranker, mock_broker):
        """Test returns 0 when bars is None."""
        mock_broker.get_bars.return_value = None

        result = await ranker.get_benchmark_return()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_on_exception(self, ranker, mock_broker):
        """Test returns 0 on exception."""
        mock_broker.get_bars.side_effect = Exception("API Error")

        result = await ranker.get_benchmark_return()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_force_refresh_ignores_cache(self, ranker, mock_broker, mock_bars):
        """Test force_refresh ignores cache."""
        ranker._benchmark_return = 0.05
        ranker._last_calc_date = datetime.now().date()

        prices = [100.0 + i * 0.5 for i in range(25)]
        mock_broker.get_bars.return_value = mock_bars(prices)

        await ranker.get_benchmark_return(force_refresh=True)
        mock_broker.get_bars.assert_called_once()


# ============================================================================
# Calculate Relative Strength Tests
# ============================================================================


class TestCalculateRelativeStrength:
    """Test calculate_relative_strength method."""

    @pytest.mark.asyncio
    async def test_returns_dict_structure(self, ranker, mock_broker, mock_bars):
        """Test returns dict with expected keys."""
        # Stock prices increasing
        stock_prices = [100.0 + i for i in range(25)]
        benchmark_prices = [100.0 + i * 0.5 for i in range(25)]

        mock_broker.get_bars.side_effect = [
            mock_bars(stock_prices),
            mock_bars(benchmark_prices),
        ]

        result = await ranker.calculate_relative_strength("AAPL")

        assert result is not None
        assert "symbol" in result
        assert "stock_return" in result
        assert "benchmark_return" in result
        assert "rs_ratio" in result
        assert "rs_line" in result
        assert "outperforming" in result
        assert "current_price" in result

    @pytest.mark.asyncio
    async def test_outperformer_has_ratio_above_one(self, ranker, mock_broker, mock_bars):
        """Test outperformer has RS ratio > 1."""
        # Stock outperforms: 20% gain vs 10% gain
        # With lookback_days=20 and 25 elements: closes[-20]=closes[5]=start, closes[-1]=end
        # Need index 5 to be 100, index 24 to be 120
        stock_prices = [100.0] * 6 + [120.0] * 19  # closes[5]=100, closes[24]=120 -> 20% return

        # Mock get_benchmark_return directly to ensure correct benchmark
        ranker._benchmark_return = 0.10  # 10% return
        ranker._last_calc_date = datetime.now().date()

        mock_broker.get_bars.return_value = mock_bars(stock_prices)

        result = await ranker.calculate_relative_strength("AAPL")

        # RS ratio = (1 + 0.20) / (1 + 0.10) = 1.20/1.10 = 1.0909
        assert result is not None
        assert result["rs_ratio"] > 1.0
        assert result["outperforming"] is True

    @pytest.mark.asyncio
    async def test_underperformer_has_ratio_below_one(self, ranker, mock_broker, mock_bars):
        """Test underperformer has RS ratio < 1."""
        # Stock underperforms: 5% gain vs 10% gain
        # With lookback_days=20 and 25 elements: closes[-20]=closes[5]=start
        stock_prices = [100.0] * 6 + [105.0] * 19  # closes[5]=100, closes[24]=105 -> 5% return

        # Mock get_benchmark_return directly to ensure correct benchmark
        ranker._benchmark_return = 0.10  # 10% return
        ranker._last_calc_date = datetime.now().date()

        mock_broker.get_bars.return_value = mock_bars(stock_prices)

        result = await ranker.calculate_relative_strength("AAPL")

        # RS ratio = (1 + 0.05) / (1 + 0.10) = 1.05/1.10 = 0.9545
        assert result is not None
        assert result["rs_ratio"] < 1.0
        assert result["outperforming"] is False

    @pytest.mark.asyncio
    async def test_returns_none_on_insufficient_data(self, ranker, mock_broker, mock_bars):
        """Test returns None on insufficient data."""
        mock_broker.get_bars.return_value = mock_bars([100.0])

        result = await ranker.calculate_relative_strength("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_none_bars(self, ranker, mock_broker):
        """Test returns None when bars is None."""
        mock_broker.get_bars.return_value = None

        result = await ranker.calculate_relative_strength("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_zero_start_price(self, ranker, mock_broker, mock_bars):
        """Test returns None on zero start price."""
        # With lookback_days=20 and 25 elements, closes[-20] = closes[5] is start_price
        # Need index 5 to be 0.0 to trigger the guard
        stock_prices = [100.0] * 5 + [0.0] + [100.0] * 19  # index 5 is 0.0
        mock_broker.get_bars.return_value = mock_bars(stock_prices)

        result = await ranker.calculate_relative_strength("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_zero_benchmark_return(self, ranker, mock_broker, mock_bars):
        """Test handles zero benchmark return."""
        # Stock gains 10%, benchmark flat
        # With lookback_days=20 and 25 elements: closes[-20]=closes[5]=start
        stock_prices = [100.0] * 6 + [110.0] * 19  # closes[5]=100, closes[24]=110 -> 10% return

        # Mock zero benchmark return directly
        ranker._benchmark_return = 0.0  # 0% return (flat)
        ranker._last_calc_date = datetime.now().date()

        mock_broker.get_bars.return_value = mock_bars(stock_prices)

        result = await ranker.calculate_relative_strength("AAPL")

        # When benchmark_return=0, rs_ratio = 1.0 + stock_return = 1.0 + 0.10 = 1.10
        assert result is not None
        assert result["rs_ratio"] > 1.0  # Should use absolute return


# ============================================================================
# Rank Symbols Tests
# ============================================================================


class TestRankSymbols:
    """Test rank_symbols method."""

    @pytest.mark.asyncio
    async def test_returns_sorted_list(self, ranker, mock_broker, mock_bars):
        """Test returns list sorted by RS ratio."""
        # Set up different returns for each symbol
        stock_prices_high = [100.0] * 5 + [130.0] * 20  # +30%
        stock_prices_mid = [100.0] * 5 + [110.0] * 20   # +10%
        stock_prices_low = [100.0] * 5 + [95.0] * 20    # -5%
        benchmark_prices = [100.0] * 5 + [110.0] * 20   # +10%

        mock_broker.get_bars.side_effect = [
            mock_bars(stock_prices_high),
            mock_bars(benchmark_prices),
            mock_bars(stock_prices_mid),
            mock_bars(benchmark_prices),
            mock_bars(stock_prices_low),
            mock_bars(benchmark_prices),
        ]

        rankings = await ranker.rank_symbols(["HIGH", "MID", "LOW"])

        assert len(rankings) == 3
        assert rankings[0]["symbol"] == "HIGH"
        assert rankings[-1]["symbol"] == "LOW"

    @pytest.mark.asyncio
    async def test_adds_rank_and_percentile(self, ranker, mock_broker, mock_bars):
        """Test adds rank and percentile to results."""
        stock_prices = [100.0] * 5 + [110.0] * 20
        benchmark_prices = [100.0] * 5 + [105.0] * 20

        mock_broker.get_bars.side_effect = [
            mock_bars(stock_prices),
            mock_bars(benchmark_prices),
        ]

        rankings = await ranker.rank_symbols(["AAPL"])

        assert rankings[0]["rank"] == 1
        assert rankings[0]["percentile"] == 1.0

    @pytest.mark.asyncio
    async def test_handles_empty_list(self, ranker):
        """Test handles empty symbol list."""
        rankings = await ranker.rank_symbols([])
        assert rankings == []

    @pytest.mark.asyncio
    async def test_filters_out_none_results(self, ranker, mock_broker):
        """Test filters out None results."""
        mock_broker.get_bars.return_value = None

        rankings = await ranker.rank_symbols(["AAPL", "MSFT"])
        assert rankings == []


# ============================================================================
# Get Leaders Tests
# ============================================================================


class TestGetLeaders:
    """Test get_leaders method."""

    def test_returns_top_symbols(self, ranker):
        """Test returns top symbols by RS."""
        rankings = [
            {"symbol": "A", "rs_ratio": 1.20},
            {"symbol": "B", "rs_ratio": 1.10},
            {"symbol": "C", "rs_ratio": 1.05},
            {"symbol": "D", "rs_ratio": 0.95},
            {"symbol": "E", "rs_ratio": 0.85},
        ]

        leaders = ranker.get_leaders(rankings, top_pct=0.40)

        assert "A" in leaders
        assert "B" in leaders
        assert "D" not in leaders
        assert "E" not in leaders

    def test_respects_min_rs_threshold(self, ranker):
        """Test respects min_rs_for_long threshold."""
        rankings = [
            {"symbol": "A", "rs_ratio": 1.20},
            {"symbol": "B", "rs_ratio": 1.02},  # Below min_rs_for_long
            {"symbol": "C", "rs_ratio": 0.95},
        ]

        leaders = ranker.get_leaders(rankings, top_pct=1.0)

        assert "A" in leaders
        assert "B" not in leaders  # Below 1.05 threshold

    def test_returns_empty_for_empty_rankings(self, ranker):
        """Test returns empty for empty rankings."""
        leaders = ranker.get_leaders([])
        assert leaders == []

    def test_respects_min_count(self, ranker):
        """Test respects min_count parameter."""
        rankings = [
            {"symbol": "A", "rs_ratio": 1.20},
            {"symbol": "B", "rs_ratio": 1.15},
            {"symbol": "C", "rs_ratio": 1.10},
            {"symbol": "D", "rs_ratio": 1.08},
            {"symbol": "E", "rs_ratio": 1.06},
        ]

        leaders = ranker.get_leaders(rankings, top_pct=0.01, min_count=3)

        # Should return at least 3 even though 1% would be ~0
        assert len(leaders) >= 3


# ============================================================================
# Get Laggards Tests
# ============================================================================


class TestGetLaggards:
    """Test get_laggards method."""

    def test_returns_bottom_symbols(self, ranker):
        """Test returns bottom symbols by RS."""
        rankings = [
            {"symbol": "A", "rs_ratio": 1.20},
            {"symbol": "B", "rs_ratio": 1.10},
            {"symbol": "C", "rs_ratio": 1.05},
            {"symbol": "D", "rs_ratio": 0.92},
            {"symbol": "E", "rs_ratio": 0.85},
        ]

        laggards = ranker.get_laggards(rankings, bottom_pct=0.40)

        assert "E" in laggards
        assert "D" in laggards
        assert "A" not in laggards

    def test_respects_max_rs_threshold(self, ranker):
        """Test respects max_rs_for_short threshold."""
        rankings = [
            {"symbol": "A", "rs_ratio": 1.20},
            {"symbol": "B", "rs_ratio": 0.98},  # Above max_rs_for_short
            {"symbol": "C", "rs_ratio": 0.85},
        ]

        laggards = ranker.get_laggards(rankings, bottom_pct=1.0)

        assert "C" in laggards
        assert "B" not in laggards  # Above 0.95 threshold

    def test_returns_empty_for_empty_rankings(self, ranker):
        """Test returns empty for empty rankings."""
        laggards = ranker.get_laggards([])
        assert laggards == []


# ============================================================================
# Filter By RS Tests
# ============================================================================


class TestFilterByRS:
    """Test filter_by_rs method."""

    def test_allows_long_for_leader(self, ranker):
        """Test allows long for strong leader."""
        rs_info = {"rs_ratio": 1.10}
        should_trade, reason = ranker.filter_by_rs("long", rs_info)

        assert should_trade is True
        assert "leader" in reason

    def test_allows_long_for_slight_outperformer(self, ranker):
        """Test allows long for slight outperformer."""
        rs_info = {"rs_ratio": 1.02}
        should_trade, reason = ranker.filter_by_rs("long", rs_info)

        assert should_trade is True
        assert "outperformer" in reason

    def test_rejects_long_for_underperformer(self, ranker):
        """Test rejects long for underperformer."""
        rs_info = {"rs_ratio": 0.90}
        should_trade, reason = ranker.filter_by_rs("long", rs_info)

        assert should_trade is False
        assert "underperformer" in reason

    def test_allows_short_for_laggard(self, ranker):
        """Test allows short for strong laggard."""
        rs_info = {"rs_ratio": 0.90}
        should_trade, reason = ranker.filter_by_rs("short", rs_info)

        assert should_trade is True
        assert "laggard" in reason

    def test_rejects_short_for_outperformer(self, ranker):
        """Test rejects short for outperformer."""
        rs_info = {"rs_ratio": 1.10}
        should_trade, reason = ranker.filter_by_rs("short", rs_info)

        assert should_trade is False
        assert "outperformer" in reason

    def test_allows_when_no_rs_data(self, ranker):
        """Test allows trade when no RS data."""
        should_trade, reason = ranker.filter_by_rs("long", None)

        assert should_trade is True
        assert "No RS data" in reason

    def test_neutral_signal(self, ranker):
        """Test neutral signal allows trade."""
        rs_info = {"rs_ratio": 1.0}
        should_trade, reason = ranker.filter_by_rs("neutral", rs_info)

        assert should_trade is True


# ============================================================================
# Get RS Report Tests
# ============================================================================


class TestGetRSReport:
    """Test get_rs_report method."""

    @pytest.mark.asyncio
    async def test_returns_report_structure(self, ranker, mock_broker, mock_bars):
        """Test returns report with expected structure."""
        stock_prices = [100.0] * 5 + [110.0] * 20
        benchmark_prices = [100.0] * 5 + [105.0] * 20

        mock_broker.get_bars.side_effect = [
            mock_bars(stock_prices),
            mock_bars(benchmark_prices),
            mock_bars(benchmark_prices),  # For get_benchmark_return
        ]

        report = await ranker.get_rs_report(["AAPL"])

        assert "benchmark" in report
        assert "benchmark_return" in report
        assert "lookback_days" in report
        assert "symbols_ranked" in report
        assert "outperformers" in report
        assert "underperformers" in report
        assert "avg_rs" in report
        assert "median_rs" in report
        assert "leaders" in report
        assert "laggards" in report
        assert "rankings" in report


# ============================================================================
# RSMomentumFilter Tests
# ============================================================================


class TestRSMomentumFilterInit:
    """Test RSMomentumFilter initialization."""

    def test_default_init(self, rs_filter):
        """Test default initialization."""
        assert rs_filter.ranker is not None
        assert rs_filter._rankings_cache == {}
        assert rs_filter._cache_time is None

    def test_custom_max_cache_size(self, mock_broker):
        """Test custom max cache size."""
        filter_custom = RSMomentumFilter(mock_broker, max_cache_size=100)
        assert filter_custom._max_cache_size == 100


class TestRefreshRankings:
    """Test refresh_rankings method."""

    @pytest.mark.asyncio
    async def test_populates_cache(self, rs_filter, mock_broker, mock_bars):
        """Test populates cache."""
        stock_prices = [100.0] * 5 + [110.0] * 20
        benchmark_prices = [100.0] * 5 + [105.0] * 20

        mock_broker.get_bars.side_effect = [
            mock_bars(stock_prices),
            mock_bars(benchmark_prices),
        ]

        await rs_filter.refresh_rankings(["AAPL"])

        assert "AAPL" in rs_filter._rankings_cache
        assert rs_filter._cache_time is not None

    @pytest.mark.asyncio
    async def test_truncates_to_max_cache(self, mock_broker):
        """Test truncates to max cache size."""
        filter_small = RSMomentumFilter(mock_broker, max_cache_size=1)

        # Mock ranker to return multiple results
        with patch.object(filter_small.ranker, "rank_symbols") as mock_rank:
            mock_rank.return_value = [
                {"symbol": "A", "rs_ratio": 1.2},
                {"symbol": "B", "rs_ratio": 1.1},
            ]

            await filter_small.refresh_rankings(["A", "B"])

            assert len(filter_small._rankings_cache) <= 1


class TestGetRS:
    """Test get_rs method."""

    def test_returns_cached_value(self, rs_filter):
        """Test returns cached value."""
        rs_filter._cache_time = datetime.now()
        rs_filter._rankings_cache = {"AAPL": {"rs_ratio": 1.05}}

        result = rs_filter.get_rs("AAPL")
        assert result["rs_ratio"] == 1.05

    def test_returns_none_when_no_cache(self, rs_filter):
        """Test returns None when no cache."""
        result = rs_filter.get_rs("AAPL")
        assert result is None

    def test_returns_none_when_cache_stale(self, rs_filter):
        """Test returns None when cache is stale."""
        rs_filter._cache_time = datetime.now() - timedelta(hours=3)
        rs_filter._rankings_cache = {"AAPL": {"rs_ratio": 1.05}}

        result = rs_filter.get_rs("AAPL", max_age_minutes=60)
        assert result is None

    def test_returns_none_for_unknown_symbol(self, rs_filter):
        """Test returns None for unknown symbol."""
        rs_filter._cache_time = datetime.now()
        rs_filter._rankings_cache = {"AAPL": {"rs_ratio": 1.05}}

        result = rs_filter.get_rs("UNKNOWN")
        assert result is None


class TestShouldTrade:
    """Test should_trade method."""

    def test_uses_cached_rs(self, rs_filter):
        """Test uses cached RS info."""
        rs_filter._cache_time = datetime.now()
        rs_filter._rankings_cache = {"AAPL": {"rs_ratio": 1.10}}

        should_trade, reason = rs_filter.should_trade("AAPL", "long")
        assert should_trade is True


class TestGetPositionMultiplier:
    """Test get_position_multiplier method."""

    def test_returns_default_when_no_cache(self, rs_filter):
        """Test returns 1.0 when no cache."""
        multiplier = rs_filter.get_position_multiplier("AAPL")
        assert multiplier == 1.0

    def test_top_tier_gets_higher_multiplier(self, rs_filter):
        """Test top tier gets higher multiplier."""
        rs_filter._cache_time = datetime.now()
        rs_filter._rankings_cache = {"AAPL": {"percentile": 0.90}}

        multiplier = rs_filter.get_position_multiplier("AAPL")
        assert multiplier == 1.2

    def test_bottom_tier_gets_lower_multiplier(self, rs_filter):
        """Test bottom tier gets lower multiplier."""
        rs_filter._cache_time = datetime.now()
        rs_filter._rankings_cache = {"AAPL": {"percentile": 0.10}}

        multiplier = rs_filter.get_position_multiplier("AAPL")
        assert multiplier == 0.8

    def test_middle_tier_gets_default(self, rs_filter):
        """Test middle tier gets default multiplier."""
        rs_filter._cache_time = datetime.now()
        rs_filter._rankings_cache = {"AAPL": {"percentile": 0.50}}

        multiplier = rs_filter.get_position_multiplier("AAPL")
        assert multiplier == 1.0


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_exception_in_calculate_rs(self, ranker, mock_broker):
        """Test handles exception in calculate_relative_strength."""
        mock_broker.get_bars.side_effect = Exception("Network Error")

        result = await ranker.calculate_relative_strength("AAPL")
        assert result is None

    def test_filter_by_rs_boundary_values(self, ranker):
        """Test filter_by_rs at boundary values."""
        # Exactly at min_rs_for_long (1.05)
        rs_info = {"rs_ratio": 1.05}
        should_trade, _ = ranker.filter_by_rs("long", rs_info)
        assert should_trade is True

        # Exactly at max_rs_for_short (0.95)
        rs_info = {"rs_ratio": 0.95}
        should_trade, _ = ranker.filter_by_rs("short", rs_info)
        assert should_trade is True

    def test_multiplier_boundary_percentiles(self, rs_filter):
        """Test multiplier at percentile boundaries."""
        rs_filter._cache_time = datetime.now()

        # At exact boundaries
        test_cases = [
            (0.80, 1.2),   # Top tier
            (0.60, 1.1),   # Upper tier
            (0.40, 0.9),   # Lower tier
            (0.20, 0.8),   # Bottom tier
        ]

        for percentile, expected in test_cases:
            rs_filter._rankings_cache = {"TEST": {"percentile": percentile}}
            multiplier = rs_filter.get_position_multiplier("TEST")
            assert multiplier == expected
