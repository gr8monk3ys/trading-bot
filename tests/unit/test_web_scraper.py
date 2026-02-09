"""
Unit tests for web scraper alternative data providers.
"""


import pytest

from data.alt_data_types import AltDataSource
from data.web_scraper import (
    COMPANY_TICKER_MAP,
    TICKER_COMPANY_MAP,
    AppRankingData,
    AppRankingsProvider,
    GlassdoorData,
    GlassdoorSentimentProvider,
    JobPostingData,
    JobPostingsProvider,
    WebScraperAggregator,
    create_web_scraper_provider,
)


class TestJobPostingData:
    """Tests for JobPostingData dataclass."""

    def test_default_values(self):
        """Test default values."""
        data = JobPostingData()
        assert data.total_postings == 0
        assert data.change_pct_30d == 0.0
        assert data.hiring_surge is False
        assert data.contraction_signal is False

    def test_hiring_surge_flag(self):
        """Test hiring surge scenario."""
        data = JobPostingData(
            total_postings=200,
            change_pct_30d=25.0,
            hiring_surge=True,
        )
        assert data.hiring_surge is True
        assert data.total_postings == 200

    def test_contraction_signal(self):
        """Test contraction scenario."""
        data = JobPostingData(
            total_postings=50,
            change_pct_30d=-20.0,
            contraction_signal=True,
            layoff_mentions=3,
        )
        assert data.contraction_signal is True
        assert data.layoff_mentions == 3


class TestGlassdoorData:
    """Tests for GlassdoorData dataclass."""

    def test_default_values(self):
        """Test default values."""
        data = GlassdoorData()
        assert data.overall_rating == 0.0
        assert data.sentiment_score == 0.0

    def test_positive_sentiment(self):
        """Test positive employee sentiment."""
        data = GlassdoorData(
            overall_rating=4.5,
            rating_change_90d=0.2,
            recommend_pct=0.85,
            ceo_approval=0.9,
            sentiment_score=0.6,
        )
        assert data.overall_rating == 4.5
        assert data.recommend_pct == 0.85


class TestAppRankingData:
    """Tests for AppRankingData dataclass."""

    def test_default_values(self):
        """Test default values."""
        data = AppRankingData()
        assert data.ios_rank is None
        assert data.android_rank is None

    def test_ranking_improvement(self):
        """Test ranking improvement (negative change = better)."""
        data = AppRankingData(
            ios_rank=5,
            android_rank=8,
            ios_rank_change_7d=-10,  # Improved from 15 to 5
            android_rank_change_7d=-5,
            combined_rank_score=0.9,
        )
        assert data.ios_rank_change_7d < 0  # Improvement
        assert data.combined_rank_score > 0.5


class TestJobPostingsProvider:
    """Tests for JobPostingsProvider."""

    @pytest.fixture
    def provider(self):
        """Create job postings provider."""
        return JobPostingsProvider()

    async def test_initialization(self, provider):
        """Test provider initializes successfully."""
        result = await provider.initialize()
        assert result is True
        assert provider._initialized is True

    async def test_fetch_signal(self, provider):
        """Test fetching job posting signal."""
        await provider.initialize()
        signal = await provider.fetch_signal("AAPL")

        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.source == AltDataSource.JOB_POSTINGS
        assert -1.0 <= signal.signal_value <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.raw_data is not None
        assert "total_postings" in signal.raw_data
        assert signal.job_posting_count > 0

    async def test_signal_consistency(self, provider):
        """Test signal is consistent for same symbol."""
        await provider.initialize()

        signal1 = await provider.fetch_signal("MSFT")
        signal2 = await provider.fetch_signal("MSFT")

        # Mock data uses seeded random, should be consistent
        assert signal1.signal_value == signal2.signal_value

    async def test_different_symbols_different_signals(self, provider):
        """Test different symbols produce different signals."""
        await provider.initialize()

        signal_aapl = await provider.fetch_signal("AAPL")
        signal_googl = await provider.fetch_signal("GOOGL")

        # Different symbols should have different mock data
        assert signal_aapl.raw_data != signal_googl.raw_data

    def test_signal_calculation_hiring_surge(self, provider):
        """Test signal calculation with hiring surge."""
        data = JobPostingData(
            total_postings=150,
            change_pct_30d=30.0,
            engineering_pct=0.6,
            hiring_surge=True,
        )
        signal = provider._calculate_signal(data)
        assert signal > 0  # Positive signal for hiring surge

    def test_signal_calculation_contraction(self, provider):
        """Test signal calculation with contraction."""
        data = JobPostingData(
            total_postings=50,
            change_pct_30d=-25.0,
            engineering_pct=0.3,
            contraction_signal=True,
        )
        signal = provider._calculate_signal(data)
        assert signal < 0  # Negative signal for contraction

    def test_confidence_calculation(self, provider):
        """Test confidence based on data quality."""
        # High quality data
        high_quality = JobPostingData(total_postings=200, change_pct_30d=20.0)
        high_conf = provider._calculate_confidence(high_quality)

        # Low quality data
        low_quality = JobPostingData(total_postings=20, change_pct_30d=5.0)
        low_conf = provider._calculate_confidence(low_quality)

        assert high_conf > low_conf


class TestGlassdoorSentimentProvider:
    """Tests for GlassdoorSentimentProvider."""

    @pytest.fixture
    def provider(self):
        """Create Glassdoor provider."""
        return GlassdoorSentimentProvider()

    async def test_initialization(self, provider):
        """Test provider initializes successfully."""
        result = await provider.initialize()
        assert result is True

    async def test_fetch_signal(self, provider):
        """Test fetching Glassdoor signal."""
        await provider.initialize()
        signal = await provider.fetch_signal("AAPL")

        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.source == AltDataSource.GLASSDOOR
        assert "overall_rating" in signal.raw_data
        assert "ceo_approval" in signal.raw_data
        assert signal.avg_rating > 0
        assert signal.review_count >= 0

    def test_signal_calculation_positive(self, provider):
        """Test signal calculation with positive sentiment."""
        data = GlassdoorData(
            overall_rating=4.5,
            rating_change_90d=0.3,
            recommend_pct=0.9,
            sentiment_score=0.5,
        )
        signal = provider._calculate_signal(data)
        assert signal > 0  # Positive signal

    def test_signal_calculation_negative(self, provider):
        """Test signal calculation with negative sentiment."""
        data = GlassdoorData(
            overall_rating=2.5,
            rating_change_90d=-0.3,
            recommend_pct=0.4,
            sentiment_score=-0.5,
        )
        signal = provider._calculate_signal(data)
        assert signal < 0  # Negative signal

    def test_confidence_more_reviews(self, provider):
        """Test confidence increases with more reviews."""
        high_reviews = GlassdoorData(review_count_30d=100)
        low_reviews = GlassdoorData(review_count_30d=10)

        high_conf = provider._calculate_confidence(high_reviews)
        low_conf = provider._calculate_confidence(low_reviews)

        assert high_conf > low_conf


class TestAppRankingsProvider:
    """Tests for AppRankingsProvider."""

    @pytest.fixture
    def provider(self):
        """Create app rankings provider."""
        return AppRankingsProvider()

    async def test_initialization(self, provider):
        """Test provider initializes successfully."""
        result = await provider.initialize()
        assert result is True

    async def test_fetch_signal_consumer_company(self, provider):
        """Test fetching signal for consumer app company."""
        await provider.initialize()
        signal = await provider.fetch_signal("META")

        assert signal is not None
        assert signal.symbol == "META"
        assert signal.source == AltDataSource.APP_STORE
        assert "ios_rank" in signal.raw_data
        assert signal.app_rank >= 0

    async def test_fetch_signal_non_consumer_company(self, provider):
        """Test fetching signal for non-consumer company returns None."""
        await provider.initialize()
        signal = await provider.fetch_signal("XOM")  # Exxon - not a consumer app company

        assert signal is None

    def test_signal_calculation_improving_rank(self, provider):
        """Test signal for improving app rank."""
        data = AppRankingData(
            ios_rank=5,
            android_rank=8,
            ios_rank_change_7d=-15,  # Improved
            android_rank_change_7d=-10,
            combined_rank_score=0.9,
        )
        signal = provider._calculate_signal(data)
        assert signal > 0  # Positive signal for improvement

    def test_signal_calculation_declining_rank(self, provider):
        """Test signal for declining app rank."""
        data = AppRankingData(
            ios_rank=80,
            android_rank=90,
            ios_rank_change_7d=20,  # Declined
            android_rank_change_7d=15,
            combined_rank_score=0.3,
        )
        signal = provider._calculate_signal(data)
        assert signal < 0  # Negative signal for decline

    def test_consumer_app_companies_list(self, provider):
        """Test consumer app companies are defined."""
        assert "META" in provider.CONSUMER_APP_COMPANIES
        assert "NFLX" in provider.CONSUMER_APP_COMPANIES
        assert "UBER" in provider.CONSUMER_APP_COMPANIES


class TestWebScraperAggregator:
    """Tests for WebScraperAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create web scraper aggregator."""
        return WebScraperAggregator()

    async def test_initialization(self, aggregator):
        """Test aggregator initializes all providers."""
        result = await aggregator.initialize()
        assert result is True
        assert len(aggregator._providers) == 3

    async def test_composite_signal(self, aggregator):
        """Test getting composite signal."""
        await aggregator.initialize()
        signal = await aggregator.get_composite_signal("AAPL")

        assert signal is not None
        assert signal.symbol == "AAPL"
        assert -1.0 <= signal.signal_value <= 1.0
        assert signal.raw_data is not None

    async def test_composite_signal_consumer_company(self, aggregator):
        """Test composite signal for consumer app company includes all sources."""
        await aggregator.initialize()
        signal = await aggregator.get_composite_signal("META")

        # META should have signals from all 3 sources
        assert signal is not None
        assert signal.raw_data is not None
        assert "source_count" in signal.raw_data

    async def test_auto_initialization(self, aggregator):
        """Test aggregator auto-initializes when getting signal."""
        signal = await aggregator.get_composite_signal("TSLA")

        # Should auto-initialize
        assert aggregator._initialized is True
        assert signal is not None


class TestFactoryFunction:
    """Tests for create_web_scraper_provider factory."""

    def test_create_job_postings_provider(self):
        """Test creating job postings provider."""
        provider = create_web_scraper_provider(AltDataSource.JOB_POSTINGS)
        assert provider is not None
        assert isinstance(provider, JobPostingsProvider)

    def test_create_glassdoor_provider(self):
        """Test creating Glassdoor provider."""
        provider = create_web_scraper_provider(AltDataSource.GLASSDOOR)
        assert provider is not None
        assert isinstance(provider, GlassdoorSentimentProvider)

    def test_create_app_store_provider(self):
        """Test creating app store provider."""
        provider = create_web_scraper_provider(AltDataSource.APP_STORE)
        assert provider is not None
        assert isinstance(provider, AppRankingsProvider)

    def test_create_unknown_provider(self):
        """Test creating unknown provider returns None."""
        provider = create_web_scraper_provider(AltDataSource.TWITTER)
        assert provider is None


class TestCompanyTickerMapping:
    """Tests for company-ticker mapping."""

    def test_company_to_ticker(self):
        """Test company name to ticker mapping."""
        assert COMPANY_TICKER_MAP["apple"] == "AAPL"
        assert COMPANY_TICKER_MAP["microsoft"] == "MSFT"
        assert COMPANY_TICKER_MAP["google"] == "GOOGL"

    def test_ticker_to_company(self):
        """Test ticker to company name mapping."""
        assert TICKER_COMPANY_MAP["AAPL"] == "apple"
        assert TICKER_COMPANY_MAP["MSFT"] == "microsoft"

    def test_mapping_consistency(self):
        """Test mapping is bidirectional."""
        for _company, ticker in COMPANY_TICKER_MAP.items():
            # Some companies share tickers (google, alphabet -> GOOGL)
            # So reverse mapping picks one
            if ticker in TICKER_COMPANY_MAP:
                reverse_company = TICKER_COMPANY_MAP[ticker]
                assert COMPANY_TICKER_MAP[reverse_company] == ticker


class TestWebScraperIntegration:
    """Integration tests for web scraper system."""

    async def test_full_pipeline(self):
        """Test full web scraper pipeline."""
        aggregator = WebScraperAggregator()
        await aggregator.initialize()

        # Fetch signals for multiple symbols
        symbols = ["AAPL", "MSFT", "META", "GOOGL"]
        signals = []

        for symbol in symbols:
            signal = await aggregator.get_composite_signal(symbol)
            if signal:
                signals.append(signal)

        assert len(signals) >= 3  # Should have signals for most symbols

    async def test_provider_independence(self):
        """Test each provider works independently."""
        job_provider = JobPostingsProvider()
        glassdoor_provider = GlassdoorSentimentProvider()
        app_provider = AppRankingsProvider()

        await job_provider.initialize()
        await glassdoor_provider.initialize()
        await app_provider.initialize()

        job_signal = await job_provider.fetch_signal("AAPL")
        glassdoor_signal = await glassdoor_provider.fetch_signal("AAPL")
        app_signal = await app_provider.fetch_signal("META")  # Consumer company

        assert job_signal is not None
        assert glassdoor_signal is not None
        assert app_signal is not None

        # Different sources
        assert job_signal.source == AltDataSource.JOB_POSTINGS
        assert glassdoor_signal.source == AltDataSource.GLASSDOOR
        assert app_signal.source == AltDataSource.APP_STORE
