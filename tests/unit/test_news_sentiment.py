"""
Unit tests for the NewsSentimentAnalyzer.

Tests cover:
- NewsArticle and SentimentResult dataclasses
- News fetching with mocked Alpaca API
- Sentiment analysis with mocked FinBERT
- Caching behavior
- Bulk sentiment analysis
- Error handling
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.news_sentiment import (
    NewsArticle,
    NewsSentimentAnalyzer,
    SentimentResult,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_api_credentials():
    """Provide mock API credentials."""
    return {
        "api_key": "test_api_key",
        "secret_key": "test_secret_key",
    }


@pytest.fixture
def analyzer(mock_api_credentials):
    """Create a NewsSentimentAnalyzer instance with mocked credentials."""
    return NewsSentimentAnalyzer(
        api_key=mock_api_credentials["api_key"],
        secret_key=mock_api_credentials["secret_key"],
        use_gpu=False,
        cache_ttl_minutes=15,
    )


@pytest.fixture
def sample_news_articles():
    """Create sample news articles for testing."""
    now = datetime.now()
    return [
        NewsArticle(
            id="1",
            headline="Apple Reports Record iPhone Sales",
            summary="Apple Inc. announced record-breaking iPhone sales for Q4.",
            author="John Doe",
            source="Reuters",
            url="https://example.com/1",
            symbols=["AAPL"],
            created_at=now - timedelta(hours=2),
            updated_at=now - timedelta(hours=2),
        ),
        NewsArticle(
            id="2",
            headline="Apple Stock Rises on Strong Earnings",
            summary="Shares of Apple jumped 5% following earnings beat.",
            author="Jane Smith",
            source="Bloomberg",
            url="https://example.com/2",
            symbols=["AAPL"],
            created_at=now - timedelta(hours=4),
            updated_at=now - timedelta(hours=4),
        ),
        NewsArticle(
            id="3",
            headline="Tech Sector Shows Weakness",
            summary="Technology stocks faced selling pressure amid rate concerns.",
            author="Bob Wilson",
            source="CNBC",
            url="https://example.com/3",
            symbols=["AAPL", "MSFT", "GOOGL"],
            created_at=now - timedelta(hours=6),
            updated_at=now - timedelta(hours=6),
        ),
    ]


@pytest.fixture
def mock_finbert_positive():
    """Mock FinBERT returning positive sentiment."""
    return [
        {"label": "positive", "score": 0.85},
        {"label": "positive", "score": 0.75},
        {"label": "neutral", "score": 0.60},
    ]


@pytest.fixture
def mock_finbert_negative():
    """Mock FinBERT returning negative sentiment."""
    return [
        {"label": "negative", "score": 0.80},
        {"label": "negative", "score": 0.70},
        {"label": "neutral", "score": 0.55},
    ]


@pytest.fixture
def mock_finbert_mixed():
    """Mock FinBERT returning mixed sentiment."""
    return [
        {"label": "positive", "score": 0.75},
        {"label": "negative", "score": 0.65},
        {"label": "neutral", "score": 0.80},
    ]


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestNewsArticle:
    """Tests for the NewsArticle dataclass."""

    def test_news_article_creation(self):
        """Test creating a NewsArticle instance."""
        now = datetime.now()
        article = NewsArticle(
            id="test-id",
            headline="Test Headline",
            summary="Test Summary",
            author="Test Author",
            source="Test Source",
            url="https://example.com",
            symbols=["AAPL", "MSFT"],
            created_at=now,
            updated_at=now,
        )

        assert article.id == "test-id"
        assert article.headline == "Test Headline"
        assert article.summary == "Test Summary"
        assert article.author == "Test Author"
        assert article.source == "Test Source"
        assert article.url == "https://example.com"
        assert article.symbols == ["AAPL", "MSFT"]
        assert article.created_at == now
        assert article.updated_at == now


class TestSentimentResult:
    """Tests for the SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        """Test creating a SentimentResult instance."""
        now = datetime.now()
        result = SentimentResult(
            symbol="AAPL",
            sentiment="positive",
            confidence=0.85,
            score=0.75,
            news_count=10,
            headlines=["Headline 1", "Headline 2"],
            timestamp=now,
        )

        assert result.symbol == "AAPL"
        assert result.sentiment == "positive"
        assert result.confidence == 0.85
        assert result.score == 0.75
        assert result.news_count == 10
        assert len(result.headlines) == 2
        assert result.timestamp == now

    def test_sentiment_result_neutral(self):
        """Test creating a neutral SentimentResult."""
        result = SentimentResult(
            symbol="TSLA",
            sentiment="neutral",
            confidence=0.0,
            score=0.0,
            news_count=0,
            headlines=[],
            timestamp=datetime.now(),
        )

        assert result.sentiment == "neutral"
        assert result.confidence == 0.0
        assert result.score == 0.0
        assert result.news_count == 0


# =============================================================================
# ANALYZER INITIALIZATION TESTS
# =============================================================================


class TestAnalyzerInitialization:
    """Tests for NewsSentimentAnalyzer initialization."""

    def test_analyzer_creation(self, mock_api_credentials):
        """Test creating an analyzer instance."""
        analyzer = NewsSentimentAnalyzer(
            api_key=mock_api_credentials["api_key"],
            secret_key=mock_api_credentials["secret_key"],
        )

        assert analyzer._api_key == mock_api_credentials["api_key"]
        assert analyzer._secret_key == mock_api_credentials["secret_key"]
        assert analyzer.use_gpu is False
        assert analyzer._news_client is None
        assert analyzer._finbert_loaded is False

    def test_analyzer_with_gpu(self, mock_api_credentials):
        """Test creating an analyzer with GPU enabled."""
        analyzer = NewsSentimentAnalyzer(
            api_key=mock_api_credentials["api_key"],
            secret_key=mock_api_credentials["secret_key"],
            use_gpu=True,
        )

        assert analyzer.use_gpu is True

    def test_analyzer_custom_cache_ttl(self, mock_api_credentials):
        """Test creating an analyzer with custom cache TTL."""
        analyzer = NewsSentimentAnalyzer(
            api_key=mock_api_credentials["api_key"],
            secret_key=mock_api_credentials["secret_key"],
            cache_ttl_minutes=30,
        )

        assert analyzer._cache_ttl == timedelta(minutes=30)


# =============================================================================
# SENTIMENT AGGREGATION TESTS
# =============================================================================


class TestSentimentAggregation:
    """Tests for sentiment aggregation logic."""

    def test_aggregate_positive_sentiment(self, analyzer, mock_finbert_positive):
        """Test aggregating predominantly positive sentiment."""
        sentiment, confidence, score = analyzer._aggregate_sentiment(mock_finbert_positive)

        assert sentiment == "positive"
        assert score > 0.2  # Above positive threshold
        assert confidence > 0

    def test_aggregate_negative_sentiment(self, analyzer, mock_finbert_negative):
        """Test aggregating predominantly negative sentiment."""
        sentiment, confidence, score = analyzer._aggregate_sentiment(mock_finbert_negative)

        assert sentiment == "negative"
        assert score < -0.2  # Below negative threshold
        assert confidence > 0

    def test_aggregate_neutral_sentiment(self, analyzer, mock_finbert_mixed):
        """Test aggregating mixed sentiment resulting in neutral."""
        sentiment, confidence, score = analyzer._aggregate_sentiment(mock_finbert_mixed)

        # Mixed sentiment should average out to near neutral
        assert sentiment in ["neutral", "positive", "negative"]
        assert -1.0 <= score <= 1.0

    def test_aggregate_empty_sentiment(self, analyzer):
        """Test aggregating empty sentiment list."""
        sentiment, confidence, score = analyzer._aggregate_sentiment([])

        assert sentiment == "neutral"
        assert confidence == 0.0
        assert score == 0.0


# =============================================================================
# SENTIMENT ANALYSIS TESTS (MOCKED FINBERT)
# =============================================================================


class TestSentimentAnalysis:
    """Tests for sentiment analysis with mocked FinBERT."""

    def test_analyze_sentiment_with_mock(self, analyzer, mock_finbert_positive):
        """Test analyze_sentiment with mocked FinBERT pipeline."""
        with patch.object(analyzer, "_get_finbert") as mock_get_finbert:
            mock_pipeline = MagicMock()
            mock_pipeline.return_value = mock_finbert_positive
            mock_get_finbert.return_value = mock_pipeline

            texts = ["Good news", "Great earnings", "Stock rises"]
            results = analyzer.analyze_sentiment(texts)

            assert len(results) == 3
            mock_pipeline.assert_called_once()

    def test_analyze_sentiment_empty_texts(self, analyzer):
        """Test analyze_sentiment with empty text list."""
        results = analyzer.analyze_sentiment([])
        assert results == []

    def test_analyze_sentiment_finbert_not_loaded(self, analyzer):
        """Test analyze_sentiment when FinBERT fails to load."""
        with patch.object(analyzer, "_get_finbert", return_value=None):
            texts = ["Some text"]
            results = analyzer.analyze_sentiment(texts)

            assert len(results) == 1
            assert results[0]["label"] == "neutral"
            assert results[0]["score"] == 0.5


# =============================================================================
# NEWS FETCHING TESTS (MOCKED ALPACA)
# =============================================================================


class TestNewsFetching:
    """Tests for news fetching with mocked Alpaca API."""

    @pytest.mark.asyncio
    async def test_get_news_with_mock(self, analyzer, sample_news_articles):
        """Test fetching news with mocked Alpaca client."""
        # Create mock news response
        mock_news_item = MagicMock()
        mock_news_item.id = "1"
        mock_news_item.headline = "Test Headline"
        mock_news_item.summary = "Test Summary"
        mock_news_item.author = "Test Author"
        mock_news_item.source = "Test Source"
        mock_news_item.url = "https://example.com"
        mock_news_item.symbols = ["AAPL"]
        mock_news_item.created_at = datetime.now()
        mock_news_item.updated_at = datetime.now()

        mock_response = MagicMock()
        mock_response.news = [mock_news_item]

        mock_client = MagicMock()
        mock_client.get_news.return_value = mock_response

        # Mock both the news client and the import
        mock_news_request = MagicMock()
        with patch.object(analyzer, "_get_news_client", return_value=mock_client):
            with patch.dict(
                "sys.modules", {"alpaca.data.requests": MagicMock(NewsRequest=mock_news_request)}
            ):
                with patch(
                    "utils.news_sentiment.asyncio.to_thread", new_callable=AsyncMock
                ) as mock_to_thread:
                    mock_to_thread.return_value = mock_response

                    articles = await analyzer.get_news(["AAPL"])

                    assert len(articles) == 1
                    assert articles[0].headline == "Test Headline"

    @pytest.mark.asyncio
    async def test_get_news_empty_response(self, analyzer):
        """Test fetching news when no articles are returned."""
        mock_response = MagicMock()
        mock_response.news = []

        mock_client = MagicMock()
        mock_client.get_news.return_value = mock_response

        mock_news_request = MagicMock()
        with patch.object(analyzer, "_get_news_client", return_value=mock_client):
            with patch.dict(
                "sys.modules", {"alpaca.data.requests": MagicMock(NewsRequest=mock_news_request)}
            ):
                with patch(
                    "utils.news_sentiment.asyncio.to_thread", new_callable=AsyncMock
                ) as mock_to_thread:
                    mock_to_thread.return_value = mock_response

                    articles = await analyzer.get_news(["UNKNOWN"])

                    assert articles == []


# =============================================================================
# SYMBOL SENTIMENT TESTS
# =============================================================================


class TestSymbolSentiment:
    """Tests for get_symbol_sentiment method."""

    @pytest.mark.asyncio
    async def test_get_symbol_sentiment_no_news(self, analyzer):
        """Test getting sentiment when no news is available."""
        with patch.object(analyzer, "get_news", return_value=[]):
            result = await analyzer.get_symbol_sentiment("UNKNOWN")

            assert result.symbol == "UNKNOWN"
            assert result.sentiment == "neutral"
            assert result.confidence == 0.0
            assert result.score == 0.0
            assert result.news_count == 0

    @pytest.mark.asyncio
    async def test_get_symbol_sentiment_with_news(
        self, analyzer, sample_news_articles, mock_finbert_positive
    ):
        """Test getting sentiment with news articles."""
        with patch.object(analyzer, "get_news", return_value=sample_news_articles):
            with patch.object(analyzer, "analyze_sentiment") as mock_analyze:
                mock_analyze.return_value = mock_finbert_positive

                result = await analyzer.get_symbol_sentiment("AAPL")

                assert result.symbol == "AAPL"
                assert result.news_count == 3
                assert len(result.headlines) > 0

    @pytest.mark.asyncio
    async def test_get_symbol_sentiment_caching(self, analyzer):
        """Test that sentiment results are cached."""
        mock_result = SentimentResult(
            symbol="AAPL",
            sentiment="positive",
            confidence=0.8,
            score=0.7,
            news_count=5,
            headlines=["Test"],
            timestamp=datetime.now(),
        )

        # Pre-populate cache
        cache_key = "AAPL_24_True"
        analyzer._sentiment_cache[cache_key] = (mock_result, datetime.now())

        # Should return cached result without calling get_news
        with patch.object(analyzer, "get_news") as mock_get_news:
            result = await analyzer.get_symbol_sentiment("AAPL")

            mock_get_news.assert_not_called()
            assert result.symbol == "AAPL"
            assert result.sentiment == "positive"

    @pytest.mark.asyncio
    async def test_get_symbol_sentiment_cache_expired(self, analyzer):
        """Test that expired cache entries are refreshed."""
        old_result = SentimentResult(
            symbol="AAPL",
            sentiment="positive",
            confidence=0.8,
            score=0.7,
            news_count=5,
            headlines=["Test"],
            timestamp=datetime.now() - timedelta(hours=1),
        )

        # Pre-populate cache with old timestamp
        cache_key = "AAPL_24_True"
        analyzer._sentiment_cache[cache_key] = (
            old_result,
            datetime.now() - timedelta(hours=1),
        )

        # Should call get_news because cache is expired
        with patch.object(analyzer, "get_news", return_value=[]) as mock_get_news:
            result = await analyzer.get_symbol_sentiment("AAPL")

            mock_get_news.assert_called_once()
            assert result.sentiment == "neutral"  # No news = neutral


# =============================================================================
# BULK SENTIMENT TESTS
# =============================================================================


class TestBulkSentiment:
    """Tests for bulk sentiment analysis."""

    @pytest.mark.asyncio
    async def test_get_bulk_sentiment(self, analyzer):
        """Test getting sentiment for multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]

        with patch.object(analyzer, "get_news", return_value=[]):
            results = await analyzer.get_bulk_sentiment(symbols)

            assert len(results) == 3
            assert "AAPL" in results
            assert "MSFT" in results
            assert "GOOGL" in results

            for symbol in symbols:
                assert results[symbol].symbol == symbol
                assert results[symbol].sentiment == "neutral"

    @pytest.mark.asyncio
    async def test_get_bulk_sentiment_with_error(self, analyzer):
        """Test bulk sentiment handles errors for individual symbols."""

        async def mock_sentiment(symbol, *args, **kwargs):
            if symbol == "ERROR":
                raise ValueError("Test error")
            return SentimentResult(
                symbol=symbol,
                sentiment="neutral",
                confidence=0.5,
                score=0.0,
                news_count=1,
                headlines=[],
                timestamp=datetime.now(),
            )

        with patch.object(analyzer, "get_symbol_sentiment", side_effect=mock_sentiment):
            results = await analyzer.get_bulk_sentiment(["AAPL", "ERROR", "MSFT"])

            assert len(results) == 3
            assert results["AAPL"].sentiment == "neutral"
            assert results["ERROR"].sentiment == "neutral"  # Error handled gracefully
            assert results["MSFT"].sentiment == "neutral"


# =============================================================================
# UTILITY TESTS
# =============================================================================


class TestUtilities:
    """Tests for utility methods."""

    def test_clear_cache(self, analyzer):
        """Test clearing the sentiment cache."""
        # Add items to cache
        analyzer._sentiment_cache["test"] = (MagicMock(), datetime.now())

        analyzer.clear_cache()

        assert len(analyzer._sentiment_cache) == 0

    def test_is_finbert_loaded_false(self, analyzer):
        """Test is_finbert_loaded returns False when not loaded."""
        assert analyzer.is_finbert_loaded() is False

    def test_is_finbert_loaded_true(self, analyzer):
        """Test is_finbert_loaded returns True when loaded."""
        analyzer._finbert_loaded = True
        assert analyzer.is_finbert_loaded() is True

    def test_get_finbert_status(self, analyzer):
        """Test get_finbert_status returns correct status."""
        status = analyzer.get_finbert_status()

        assert "loaded" in status
        assert "error" in status
        assert "use_gpu" in status
        assert status["loaded"] is False
        assert status["error"] is None
        assert status["use_gpu"] is False
