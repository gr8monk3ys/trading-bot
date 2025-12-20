"""
Unit tests for CorrelationManager.

Tests the correlation management system including:
- Sector classification
- Assumed correlations
- Sector exposure calculation
- Diversification scoring
- Position size adjustments
- Portfolio reporting
"""


class TestCorrelationManagerInit:
    """Test CorrelationManager initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        assert manager.max_sector_concentration == 0.40
        assert manager.max_cluster_concentration == 0.50
        assert manager.sector_correlation_penalty == 0.60
        assert manager.target_sector_count == 4

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager(
            max_sector_concentration=0.30,
            max_cluster_concentration=0.40,
            sector_correlation_penalty=0.50,
            target_sector_count=5,
        )

        assert manager.max_sector_concentration == 0.30
        assert manager.max_cluster_concentration == 0.40
        assert manager.sector_correlation_penalty == 0.50
        assert manager.target_sector_count == 5


class TestGetSector:
    """Test sector classification."""

    def test_get_sector_known_stock(self):
        """Test getting sector for known stock."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        assert manager.get_sector("AAPL") == "Technology"
        assert manager.get_sector("JPM") == "Financials"
        assert manager.get_sector("XOM") == "Energy"
        assert manager.get_sector("PG") == "ConsumerStaples"

    def test_get_sector_unknown_stock(self):
        """Test getting sector for unknown stock."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        assert manager.get_sector("UNKNOWN") == "Unknown"
        assert manager.get_sector("XYZ123") == "Unknown"

    def test_get_sector_etfs(self):
        """Test getting sector for ETFs."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        assert manager.get_sector("SPY") == "ETF_Broad"
        assert manager.get_sector("QQQ") == "ETF_Tech"
        assert manager.get_sector("GLD") == "ETF_Commodity"


class TestGetAssumedCorrelation:
    """Test assumed correlation calculations."""

    def test_same_symbol(self):
        """Test correlation of symbol with itself is 1.0."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        assert manager.get_assumed_correlation("AAPL", "AAPL") == 1.0

    def test_same_sector_high_correlation(self):
        """Test same sector stocks have high assumed correlation."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        # Both tech
        corr = manager.get_assumed_correlation("AAPL", "MSFT")
        assert corr >= 0.65

        # Both energy
        corr = manager.get_assumed_correlation("XOM", "CVX")
        assert corr >= 0.70

    def test_different_sector_lower_correlation(self):
        """Test different sector stocks have lower assumed correlation."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        # Tech vs Energy
        corr = manager.get_assumed_correlation("AAPL", "XOM")
        assert corr <= 0.40

    def test_unknown_sector_default_correlation(self):
        """Test unknown sector uses default correlation."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        corr = manager.get_assumed_correlation("AAPL", "UNKNOWN")
        assert corr == manager.DEFAULT_CROSS_SECTOR_CORR

    def test_defined_cross_sector_correlation(self):
        """Test defined cross-sector correlations."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        # Tech and Communication have defined correlation
        corr = manager.get_assumed_correlation("AAPL", "META")
        assert corr == 0.65

        # Consumer Disc and Staples
        corr = manager.get_assumed_correlation("AMZN", "PG")
        assert corr == 0.40


class TestGetSectorExposure:
    """Test sector exposure calculation."""

    def test_empty_positions(self):
        """Test sector exposure with no positions."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        assert manager.get_sector_exposure({}) == {}

    def test_single_sector(self):
        """Test exposure with single sector."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()
        positions = {
            "AAPL": {"value": 10000},
            "MSFT": {"value": 10000},
        }

        exposure = manager.get_sector_exposure(positions)

        assert exposure["Technology"] == 1.0

    def test_multiple_sectors(self):
        """Test exposure with multiple sectors."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()
        positions = {
            "AAPL": {"value": 5000},  # Tech
            "JPM": {"value": 5000},  # Financials
        }

        exposure = manager.get_sector_exposure(positions)

        assert exposure["Technology"] == 0.5
        assert exposure["Financials"] == 0.5

    def test_zero_total_value(self):
        """Test exposure with zero total value."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()
        positions = {
            "AAPL": {"value": 0},
        }

        assert manager.get_sector_exposure(positions) == {}


class TestGetDiversificationScore:
    """Test diversification score calculation."""

    def test_empty_positions(self):
        """Test score with no positions."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        assert manager.get_diversification_score({}) == 0.5

    def test_single_position(self):
        """Test score with single position."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()
        positions = {"AAPL": {"value": 10000}}

        assert manager.get_diversification_score(positions) == 0.5

    def test_concentrated_portfolio(self):
        """Test score with concentrated portfolio."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()
        # All tech
        positions = {
            "AAPL": {"value": 10000},
            "MSFT": {"value": 10000},
            "GOOGL": {"value": 10000},
        }

        score = manager.get_diversification_score(positions)
        # Low score due to single sector
        assert score < 0.5

    def test_diversified_portfolio(self):
        """Test score with diversified portfolio."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()
        # Multiple sectors, evenly distributed
        positions = {
            "AAPL": {"value": 2500},  # Tech
            "JPM": {"value": 2500},  # Financials
            "XOM": {"value": 2500},  # Energy
            "PG": {"value": 2500},  # Consumer Staples
        }

        score = manager.get_diversification_score(positions)
        # High score due to multiple sectors
        assert score > 0.6


class TestGetSectorLimitMultiplier:
    """Test sector limit multiplier calculation."""

    def test_empty_positions(self):
        """Test multiplier with no existing positions."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        mult = manager.get_sector_limit_multiplier("AAPL", {})
        assert mult == 1.0

    def test_at_sector_limit(self):
        """Test multiplier when sector is at limit."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager(max_sector_concentration=0.40)

        # Tech at 50% (exceeds 40% limit)
        positions = {
            "AAPL": {"value": 5000},  # Tech
            "JPM": {"value": 5000},  # Financials
        }

        mult = manager.get_sector_limit_multiplier("MSFT", positions)
        # Should be heavily penalized
        assert mult == 0.2

    def test_below_sector_limit(self):
        """Test multiplier when sector below limit."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager(max_sector_concentration=0.40)

        positions = {
            "AAPL": {"value": 2000},  # Tech 20%
            "JPM": {"value": 8000},  # Financials 80%
        }

        mult = manager.get_sector_limit_multiplier("MSFT", positions)
        # Should have some capacity remaining
        assert mult > 0.3


class TestGetCorrelationPenalty:
    """Test correlation penalty calculation."""

    def test_empty_positions(self):
        """Test penalty with no existing positions."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        penalty = manager.get_correlation_penalty("AAPL", {})
        assert penalty == 1.0

    def test_same_sector_penalty(self):
        """Test penalty when adding to same sector."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager(sector_correlation_penalty=0.60)

        positions = {
            "AAPL": {"value": 10000},  # Tech
        }

        penalty = manager.get_correlation_penalty("MSFT", positions)
        assert penalty == 0.60

    def test_different_sector_no_penalty(self):
        """Test no penalty when adding to different sector."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        positions = {
            "AAPL": {"value": 10000},  # Tech
        }

        penalty = manager.get_correlation_penalty("JPM", positions)
        assert penalty == 1.0


class TestGetAdjustedPositionSize:
    """Test position size adjustment."""

    def test_no_adjustment_needed(self):
        """Test no adjustment for well-diversified addition."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        positions = {
            "AAPL": {"value": 10000},  # Tech
        }

        size, info = manager.get_adjusted_position_size(
            "JPM", 10000, positions  # Financials - different sector
        )

        assert size == 10000
        assert info["final_multiplier"] == 1.0
        assert info["reason"] == "None"

    def test_same_sector_adjustment(self):
        """Test adjustment for same sector addition."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager(sector_correlation_penalty=0.60)

        positions = {
            "AAPL": {"value": 10000},  # Tech
        }

        size, info = manager.get_adjusted_position_size(
            "MSFT", 10000, positions  # Tech - same sector
        )

        # Should be penalized
        assert size < 10000
        assert info["correlation_multiplier"] == 0.60

    def test_sector_limit_adjustment(self):
        """Test adjustment at sector concentration limit."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager(max_sector_concentration=0.40)

        # Tech already at 50%
        positions = {
            "AAPL": {"value": 5000},  # Tech
            "JPM": {"value": 5000},  # Financials
        }

        size, info = manager.get_adjusted_position_size("MSFT", 10000, positions)  # Tech - at limit

        # Should be heavily penalized
        assert size < 3000
        assert "limit" in info["reason"].lower() or "Sector" in info["reason"]


class TestGetPortfolioReport:
    """Test portfolio report generation."""

    def test_empty_portfolio(self):
        """Test report with empty portfolio."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        report = manager.get_portfolio_report({})

        assert report["diversification_score"] == 0.5
        assert report["sector_count"] == 0

    def test_diversified_portfolio_report(self):
        """Test report with diversified portfolio."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        positions = {
            "AAPL": {"value": 2500},  # Tech
            "JPM": {"value": 2500},  # Financials
            "XOM": {"value": 2500},  # Energy
            "PG": {"value": 2500},  # Consumer Staples
        }

        report = manager.get_portfolio_report(positions)

        assert report["sector_count"] == 4
        assert report["is_well_diversified"] is True
        assert len(report["sector_exposure"]) == 4

    def test_concentrated_portfolio_report(self):
        """Test report with concentrated portfolio."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager(max_sector_concentration=0.40)

        positions = {
            "AAPL": {"value": 6000},  # Tech
            "MSFT": {"value": 6000},  # Tech - combined 60%
            "JPM": {"value": 4000},  # Financials
        }

        report = manager.get_portfolio_report(positions)

        # Should have concentration warnings
        assert len(report["concentrated_sectors"]) > 0
        assert any("Technology" in sector for sector, _ in report["concentrated_sectors"])


class TestGetRecommendation:
    """Test diversification recommendations."""

    def test_high_score_recommendation(self):
        """Test recommendation for well-diversified portfolio."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        rec = manager._get_recommendation(0.85, [])

        assert "well diversified" in rec.lower()

    def test_medium_score_recommendation(self):
        """Test recommendation for moderately diversified portfolio."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        rec = manager._get_recommendation(0.65, [])

        assert "good" in rec.lower() or "minor" in rec.lower()

    def test_low_score_with_concentration(self):
        """Test recommendation for concentrated portfolio."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        rec = manager._get_recommendation(0.40, [("Technology", 0.55)])

        assert "Technology" in rec
        assert "reducing" in rec.lower() or "reduce" in rec.lower()

    def test_low_score_no_concentration(self):
        """Test recommendation for undiversified portfolio without concentration."""
        from utils.correlation_manager import CorrelationManager

        manager = CorrelationManager()

        rec = manager._get_recommendation(0.40, [])

        assert "different sectors" in rec.lower() or "diversification" in rec.lower()
