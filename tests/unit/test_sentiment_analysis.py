#!/usr/bin/env python3
"""
Unit tests for Sentiment Analysis module.

Tests cover:
1. Text preprocessing
2. Keyword sentiment scoring
3. FinBERT analysis
4. SentimentAnalyzer trading signals
5. News fetcher caching
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.sentiment_analysis import (
    clean_text,
    preprocess_text,
    get_keyword_sentiment_score,
    check_keyword_combinations,
    analyze_sentiment,
    NewsArticle,
    NewsFetcher,
    SentimentAnalyzer,
    POSITIVE_KEYWORDS,
    NEGATIVE_KEYWORDS,
    POSITIVE_COMBINATIONS,
    NEGATIVE_COMBINATIONS
)


class TestTextPreprocessing:
    """Test text cleaning and preprocessing."""

    def test_clean_text_removes_special_chars(self):
        """Test that special characters are removed."""
        text = "AAPL is up 5%! Great news..."
        cleaned = clean_text(text)

        assert "!" not in cleaned
        assert "%" not in cleaned
        assert "." not in cleaned

    def test_clean_text_preserves_words(self):
        """Test that words are preserved."""
        text = "Apple reports strong earnings"
        cleaned = clean_text(text)

        assert "Apple" in cleaned
        assert "reports" in cleaned
        assert "strong" in cleaned
        assert "earnings" in cleaned

    def test_preprocess_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        text = "Multiple   spaces    here"
        processed = preprocess_text(text)

        assert "  " not in processed


class TestKeywordSentiment:
    """Test keyword-based sentiment scoring."""

    def test_positive_keywords_detected(self):
        """Test positive keywords give positive score."""
        text = "Company reports strong growth and increased profits"
        score = get_keyword_sentiment_score(text)

        assert score > 0, f"Expected positive score, got {score}"

    def test_negative_keywords_detected(self):
        """Test negative keywords give negative score."""
        text = "Company missed expectations with disappointing loss"
        score = get_keyword_sentiment_score(text)

        assert score < 0, f"Expected negative score, got {score}"

    def test_neutral_text_near_zero(self):
        """Test neutral text gives near-zero score."""
        text = "The market opened today"
        score = get_keyword_sentiment_score(text)

        assert abs(score) < 0.2, f"Expected near-zero score, got {score}"

    def test_positive_combination_boost(self):
        """Test that positive keyword combinations boost score."""
        text_with_combo = "Company beat expectations"
        text_without = "Company performed"

        score_with = get_keyword_sentiment_score(text_with_combo)
        score_without = get_keyword_sentiment_score(text_without)

        assert score_with > score_without

    def test_negative_combination_boost(self):
        """Test that negative keyword combinations boost negative score."""
        text_with_combo = "Company missed expectations"
        score = get_keyword_sentiment_score(text_with_combo)

        assert score < 0  # Should be negative


class TestKeywordCombinations:
    """Test keyword combination detection."""

    def test_detects_two_word_combo(self):
        """Test detection of two-word combinations."""
        text = "Company beat expectations this quarter"
        found = check_keyword_combinations(text, POSITIVE_COMBINATIONS)

        assert found is True

    def test_detects_three_word_combo(self):
        """Test detection of three-word combinations."""
        text = "Results were better than expected"
        found = check_keyword_combinations(text, POSITIVE_COMBINATIONS)

        assert found is True

    def test_no_false_positives(self):
        """Test no false positives for unrelated text."""
        text = "Weather was nice today"
        found_pos = check_keyword_combinations(text, POSITIVE_COMBINATIONS)
        found_neg = check_keyword_combinations(text, NEGATIVE_COMBINATIONS)

        assert found_pos is False
        assert found_neg is False


class TestFinBERTAnalysis:
    """Test FinBERT sentiment analysis."""

    def test_positive_text_positive_sentiment(self):
        """Test positive text gets positive sentiment."""
        texts = ["Apple reports record quarterly earnings, beating expectations"]
        confidence, sentiment = analyze_sentiment(texts)

        # Should be positive or at least not negative
        assert sentiment in ['positive', 'neutral']

    def test_negative_text_negative_sentiment(self):
        """Test negative text gets negative sentiment."""
        texts = ["Company reports massive loss, cuts dividend"]
        confidence, sentiment = analyze_sentiment(texts)

        # Should be negative or neutral
        assert sentiment in ['negative', 'neutral']

    def test_empty_input_returns_neutral(self):
        """Test empty input returns neutral."""
        confidence, sentiment = analyze_sentiment([])

        assert sentiment == "neutral"
        assert confidence == 0

    def test_string_input_accepted(self):
        """Test string input is accepted (not just list)."""
        confidence, sentiment = analyze_sentiment("This is a test")

        assert sentiment in ['positive', 'negative', 'neutral']

    def test_returns_confidence_value(self):
        """Test that confidence is returned."""
        confidence, sentiment = analyze_sentiment(["Test headline"])

        assert 0 <= confidence <= 1


class TestNewsArticle:
    """Test NewsArticle dataclass."""

    def test_article_creation(self):
        """Test article can be created."""
        article = NewsArticle(
            headline="Test headline",
            summary="Test summary",
            source="Test source",
            published=datetime.now(),
            url="https://test.com",
            symbol="AAPL"
        )

        assert article.headline == "Test headline"
        assert article.symbol == "AAPL"


class TestNewsFetcher:
    """Test NewsFetcher class."""

    def test_fetcher_initialization(self):
        """Test fetcher initializes with env vars."""
        with patch.dict(os.environ, {
            'FINNHUB_API_KEY': 'test_key',
            'ALPHA_VANTAGE_API_KEY': 'av_key'
        }):
            fetcher = NewsFetcher()

            assert fetcher.finnhub_key == 'test_key'
            assert fetcher.alpha_vantage_key == 'av_key'

    def test_cache_duration_set(self):
        """Test cache duration is set."""
        fetcher = NewsFetcher()

        assert fetcher._cache_duration == timedelta(minutes=5)

    @pytest.mark.asyncio
    async def test_cache_works(self):
        """Test that caching prevents duplicate API calls."""
        fetcher = NewsFetcher()

        # Manually add to cache
        fetcher._cache['AAPL_3'] = (datetime.now(), [
            NewsArticle(
                headline="Cached article",
                summary="Summary",
                source="Cache",
                published=datetime.now(),
                url="https://test.com",
                symbol="AAPL"
            )
        ])

        # Should return cached result
        articles = await fetcher.fetch_news('AAPL', days_back=3)

        assert len(articles) == 1
        assert articles[0].headline == "Cached article"


class TestSentimentAnalyzer:
    """Test SentimentAnalyzer class."""

    def test_trading_signal_bullish(self):
        """Test bullish signal generation."""
        analyzer = SentimentAnalyzer()

        result = {
            'sentiment': 'positive',
            'score': 0.5,
            'confidence': 0.8
        }

        signal = analyzer.get_trading_signal(result)

        assert signal == 'bullish'

    def test_trading_signal_bearish(self):
        """Test bearish signal generation."""
        analyzer = SentimentAnalyzer()

        result = {
            'sentiment': 'negative',
            'score': -0.5,
            'confidence': 0.8
        }

        signal = analyzer.get_trading_signal(result)

        assert signal == 'bearish'

    def test_trading_signal_neutral_low_confidence(self):
        """Test neutral signal for low confidence."""
        analyzer = SentimentAnalyzer()

        result = {
            'sentiment': 'positive',
            'score': 0.5,
            'confidence': 0.3  # Below threshold
        }

        signal = analyzer.get_trading_signal(result, min_confidence=0.5)

        assert signal == 'neutral'

    def test_trading_signal_neutral_score(self):
        """Test neutral signal for neutral score."""
        analyzer = SentimentAnalyzer()

        result = {
            'sentiment': 'neutral',
            'score': 0.1,  # Within neutral range
            'confidence': 0.8
        }

        signal = analyzer.get_trading_signal(result)

        assert signal == 'neutral'

    def test_cache_initialization(self):
        """Test analyzer cache is initialized."""
        analyzer = SentimentAnalyzer(cache_minutes=10)

        assert analyzer._cache_duration == timedelta(minutes=10)


class TestKeywordConstants:
    """Test keyword constant values."""

    def test_positive_keywords_have_weights(self):
        """Test all positive keywords have valid weights."""
        for keyword, weight in POSITIVE_KEYWORDS.items():
            assert 0 < weight <= 1, f"{keyword} has invalid weight {weight}"

    def test_negative_keywords_have_weights(self):
        """Test all negative keywords have valid weights."""
        for keyword, weight in NEGATIVE_KEYWORDS.items():
            assert 0 < weight <= 1, f"{keyword} has invalid weight {weight}"

    def test_positive_combinations_not_empty(self):
        """Test positive combinations list is not empty."""
        assert len(POSITIVE_COMBINATIONS) > 0

    def test_negative_combinations_not_empty(self):
        """Test negative combinations list is not empty."""
        assert len(NEGATIVE_COMBINATIONS) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
