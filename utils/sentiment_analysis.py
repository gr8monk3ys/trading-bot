#!/usr/bin/env python3
"""
Sentiment Analysis for Trading

This module provides functions to analyze sentiment from financial news and social media
for use in trading strategies.

Supports multiple news APIs:
- Finnhub (free tier, financial focused)
- Alpha Vantage News (free tier)
- Alpaca News (if you have Alpaca subscription)

Usage:
    from utils.sentiment_analysis import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    sentiment = await analyzer.get_symbol_sentiment('AAPL')
    print(f"Sentiment: {sentiment['sentiment']} ({sentiment['score']:.2f})")
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re
import logging
import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Initialize the model and tokenizer locally
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define financial sentiment keywords with weights
POSITIVE_KEYWORDS = {
    # Strong positive indicators
    'strong': 0.6, 'surges': 0.6, 'beat': 0.6, 'exceeded': 0.6,
    # Growth and improvement indicators
    'increase': 0.5, 'growth': 0.5, 'improved': 0.5, 'higher': 0.5,
    'gains': 0.5, 'increased': 0.5,
    # Financial indicators
    'profit': 0.4, 'dividend': 0.4, 'earnings': 0.4,
    # General positive indicators
    'better': 0.4, 'positive': 0.4, 'reported': 0.3
}

NEGATIVE_KEYWORDS = {
    # Strong negative indicators
    'missed': 0.6, 'plunges': 0.6, 'disappointing': 0.6, 'loss': 0.6,
    # Decline indicators
    'cut': 0.5, 'decline': 0.5, 'falls': 0.5, 'worse': 0.5,
    # Warning indicators
    'concerns': 0.4, 'negative': 0.4, 'lower': 0.4, 'weak': 0.4
}

# Define keyword combinations that strongly indicate sentiment
POSITIVE_COMBINATIONS = [
    ('strong', 'earnings'),
    ('increased', 'dividend'),
    ('beat', 'expectations'),
    ('better', 'than', 'expected')
]

NEGATIVE_COMBINATIONS = [
    ('missed', 'expectations'),
    ('cut', 'dividend'),
    ('lower', 'guidance'),
    ('below', 'expectations')
]

def clean_text(text):
    """Clean text while preserving case."""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def preprocess_text(text):
    """Preprocess the text for sentiment analysis."""
    # Clean text while preserving case for display
    return clean_text(text)

def check_keyword_combinations(text, combinations):
    """Check if any keyword combinations are present in the text."""
    text_lower = text.lower()
    words = text_lower.split()
    
    for combination in combinations:
        if len(combination) == 2:
            if combination[0] in words and combination[1] in words:
                return True
        elif len(combination) == 3:
            for i in range(len(words) - 2):
                if (words[i] == combination[0] and 
                    words[i+1] == combination[1] and 
                    words[i+2] == combination[2]):
                    return True
    return False

def get_keyword_sentiment_score(text):
    """Get sentiment score based on weighted keyword presence and combinations."""
    words = text.lower().split()
    
    # Calculate weighted scores
    positive_score = sum(POSITIVE_KEYWORDS[word] for word in words if word in POSITIVE_KEYWORDS)
    negative_score = sum(NEGATIVE_KEYWORDS[word] for word in words if word in NEGATIVE_KEYWORDS)
    
    # Check for keyword combinations
    if check_keyword_combinations(text, POSITIVE_COMBINATIONS):
        positive_score += 0.4  # Boost score for positive combinations
    if check_keyword_combinations(text, NEGATIVE_COMBINATIONS):
        negative_score += 0.4  # Boost score for negative combinations
        
    # Calculate the difference with positive bias
    score_diff = positive_score - negative_score
    
    # Apply stronger positive bias
    if score_diff > 0:
        return min(0.6, score_diff * 1.3)  # Stronger boost for positive scores
    elif score_diff < 0:
        return -min(0.5, abs(score_diff))  # Keep negative scores lower
    return 0.0

def analyze_sentiment(texts):
    """
    Analyze sentiment of financial texts using FinBERT with ensemble approach.
    
    Args:
        texts (list): List of text strings to analyze
        
    Returns:
        tuple: (probability, sentiment) where probability is the confidence
               and sentiment is 'positive', 'negative', or 'neutral'
    """
    try:
        if not texts:
            return 0, "neutral"
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
            
        # Tokenize texts (use lowercase version for the model)
        inputs = tokenizer([text.lower() for text in processed_texts], 
                         padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Process each text's sentiment
        sentiments = []
        confidences = []
        
        for text, prob in zip(processed_texts, probabilities):
            # Get class probabilities
            class_probs = prob.numpy()
            
            # Get keyword sentiment bias
            keyword_bias = get_keyword_sentiment_score(text)
            
            # Apply asymmetric adjustments
            if keyword_bias > 0:
                # Stronger positive adjustment
                class_probs[2] += keyword_bias  # Increase positive probability
                class_probs[0] -= keyword_bias * 0.8  # Strongly decrease negative probability
            elif keyword_bias < 0:
                # Weaker negative adjustment
                class_probs[0] += abs(keyword_bias) * 0.7  # Increase negative less
                class_probs[2] -= abs(keyword_bias) * 0.3  # Decrease positive less
            
            # Ensure probabilities are non-negative
            class_probs = np.maximum(class_probs, 0)
            
            # Normalize probabilities
            class_probs = class_probs / np.sum(class_probs)
            
            # Calculate sentiment with asymmetric thresholds
            pos_neg_diff = class_probs[2] - class_probs[0]  # positive - negative
            
            # Check for keyword combinations
            has_positive_combo = check_keyword_combinations(text, POSITIVE_COMBINATIONS)
            has_negative_combo = check_keyword_combinations(text, NEGATIVE_COMBINATIONS)
            
            # Determine sentiment with combination-aware thresholds
            if has_positive_combo or pos_neg_diff > 0.01:  # Very low threshold for positive
                sentiment = "positive"
                confidence = class_probs[2]
            elif has_negative_combo or pos_neg_diff < -0.1:  # Higher threshold for negative
                sentiment = "negative"
                confidence = class_probs[0]
            else:
                sentiment = "neutral"
                confidence = class_probs[1]
                
            sentiments.append(sentiment)
            confidences.append(confidence)
        
        # Determine final sentiment
        if len(sentiments) == 1:
            return float(confidences[0]), sentiments[0]
        
        # For multiple texts, use weighted voting with positive bias
        sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for sent, conf in zip(sentiments, confidences):
            if sent == "positive":
                sentiment_scores[sent] += conf * 1.2  # Boost positive scores
            else:
                sentiment_scores[sent] += conf
            
        # Get the dominant sentiment
        final_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
        final_confidence = sentiment_scores[final_sentiment] / len(texts)
        
        return float(final_confidence), final_sentiment
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
        return 0, "neutral"


@dataclass
class NewsArticle:
    """Represents a news article."""
    headline: str
    summary: str
    source: str
    published: datetime
    url: str
    symbol: str


class NewsFetcher:
    """
    Fetches real financial news from multiple APIs.

    Supported APIs (in order of preference):
    1. Finnhub - Free tier, financial focused
    2. Alpha Vantage - Free tier with key
    3. Alpaca - If you have Alpaca subscription
    """

    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        """Initialize news fetcher with API keys from environment."""
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.alpaca_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')

        # Cache for rate limiting
        self._cache: Dict[str, Tuple[datetime, List[NewsArticle]]] = {}
        self._cache_duration = timedelta(minutes=5)

    async def fetch_news(
        self,
        symbol: str,
        days_back: int = 3,
        max_articles: int = 10
    ) -> List[NewsArticle]:
        """
        Fetch news for a symbol from available APIs.

        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            max_articles: Maximum articles to return

        Returns:
            List of NewsArticle objects
        """
        # Check cache first
        cache_key = f"{symbol}_{days_back}"
        if cache_key in self._cache:
            cached_time, cached_articles = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                logger.debug(f"Using cached news for {symbol}")
                return cached_articles[:max_articles]

        articles = []

        # Try Finnhub first (best free tier)
        if self.finnhub_key:
            articles = await self._fetch_finnhub(symbol, days_back)

        # Fallback to Alpha Vantage
        if not articles and self.alpha_vantage_key:
            articles = await self._fetch_alpha_vantage(symbol)

        # Fallback to Alpaca
        if not articles and self.alpaca_key:
            articles = await self._fetch_alpaca(symbol, days_back)

        # Cache results
        if articles:
            self._cache[cache_key] = (datetime.now(), articles)

        return articles[:max_articles]

    async def _fetch_finnhub(
        self,
        symbol: str,
        days_back: int
    ) -> List[NewsArticle]:
        """Fetch news from Finnhub API."""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            url = f"{self.FINNHUB_BASE_URL}/company-news"
            params = {
                'symbol': symbol,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = []
                        for item in data[:20]:  # Limit to 20
                            try:
                                articles.append(NewsArticle(
                                    headline=item.get('headline', ''),
                                    summary=item.get('summary', ''),
                                    source=item.get('source', 'Finnhub'),
                                    published=datetime.fromtimestamp(item.get('datetime', 0)),
                                    url=item.get('url', ''),
                                    symbol=symbol
                                ))
                            except Exception:
                                continue
                        logger.info(f"Fetched {len(articles)} articles from Finnhub for {symbol}")
                        return articles
                    else:
                        logger.warning(f"Finnhub returned {response.status}")

        except Exception as e:
            logger.error(f"Finnhub fetch error: {e}")

        return []

    async def _fetch_alpha_vantage(self, symbol: str) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage News API."""
        try:
            url = self.ALPHA_VANTAGE_BASE_URL
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 20
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = []
                        for item in data.get('feed', [])[:20]:
                            try:
                                # Parse Alpha Vantage date format
                                time_str = item.get('time_published', '')
                                if time_str:
                                    published = datetime.strptime(time_str[:14], '%Y%m%dT%H%M%S')
                                else:
                                    published = datetime.now()

                                articles.append(NewsArticle(
                                    headline=item.get('title', ''),
                                    summary=item.get('summary', ''),
                                    source=item.get('source', 'Alpha Vantage'),
                                    published=published,
                                    url=item.get('url', ''),
                                    symbol=symbol
                                ))
                            except Exception:
                                continue
                        logger.info(f"Fetched {len(articles)} articles from Alpha Vantage for {symbol}")
                        return articles

        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")

        return []

    async def _fetch_alpaca(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from Alpaca News API."""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).isoformat()

            url = "https://data.alpaca.markets/v1beta1/news"
            params = {
                'symbols': symbol,
                'start': from_date,
                'limit': 20
            }
            headers = {
                'APCA-API-KEY-ID': self.alpaca_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = []
                        for item in data.get('news', [])[:20]:
                            try:
                                # Parse Alpaca date format
                                time_str = item.get('created_at', '')
                                published = datetime.fromisoformat(time_str.replace('Z', '+00:00')) if time_str else datetime.now()

                                articles.append(NewsArticle(
                                    headline=item.get('headline', ''),
                                    summary=item.get('summary', ''),
                                    source=item.get('source', 'Alpaca'),
                                    published=published,
                                    url=item.get('url', ''),
                                    symbol=symbol
                                ))
                            except Exception:
                                continue
                        logger.info(f"Fetched {len(articles)} articles from Alpaca for {symbol}")
                        return articles

        except Exception as e:
            logger.error(f"Alpaca fetch error: {e}")

        return []


class SentimentAnalyzer:
    """
    Complete sentiment analysis pipeline for trading.

    Fetches real news and analyzes sentiment using FinBERT.
    """

    def __init__(self, cache_minutes: int = 5):
        """Initialize sentiment analyzer."""
        self.news_fetcher = NewsFetcher()
        self._cache: Dict[str, Tuple[datetime, Dict]] = {}
        self._cache_duration = timedelta(minutes=cache_minutes)

    async def get_symbol_sentiment(
        self,
        symbol: str,
        days_back: int = 3,
        min_articles: int = 2
    ) -> Dict:
        """
        Get sentiment analysis for a symbol based on recent news.

        Args:
            symbol: Stock symbol
            days_back: Days of news to analyze
            min_articles: Minimum articles required for valid sentiment

        Returns:
            Dict with keys: sentiment, score, confidence, article_count, source
        """
        # Check cache
        if symbol in self._cache:
            cached_time, cached_result = self._cache[symbol]
            if datetime.now() - cached_time < self._cache_duration:
                logger.debug(f"Using cached sentiment for {symbol}")
                return cached_result

        # Fetch news
        articles = await self.news_fetcher.fetch_news(symbol, days_back)

        if len(articles) < min_articles:
            result = {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'article_count': len(articles),
                'source': 'insufficient_data',
                'articles': []
            }
            return result

        # Extract headlines and summaries for analysis
        texts = []
        for article in articles:
            if article.headline:
                texts.append(article.headline)
            if article.summary and len(article.summary) > 50:
                texts.append(article.summary[:500])  # Limit summary length

        if not texts:
            result = {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'article_count': len(articles),
                'source': 'no_text',
                'articles': []
            }
            return result

        # Run sentiment analysis
        confidence, sentiment = analyze_sentiment(texts)

        # Calculate numeric score (-1 to 1)
        if sentiment == 'positive':
            score = confidence
        elif sentiment == 'negative':
            score = -confidence
        else:
            score = 0.0

        result = {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'article_count': len(articles),
            'source': articles[0].source if articles else 'unknown',
            'articles': [
                {
                    'headline': a.headline,
                    'source': a.source,
                    'published': a.published.isoformat()
                }
                for a in articles[:5]  # Include top 5 articles
            ]
        }

        # Cache result
        self._cache[symbol] = (datetime.now(), result)

        logger.info(
            f"Sentiment for {symbol}: {sentiment} "
            f"(score={score:.2f}, confidence={confidence:.2f}, "
            f"articles={len(articles)})"
        )

        return result

    async def get_multi_symbol_sentiment(
        self,
        symbols: List[str],
        days_back: int = 3
    ) -> Dict[str, Dict]:
        """
        Get sentiment for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            days_back: Days of news to analyze

        Returns:
            Dict mapping symbols to sentiment results
        """
        tasks = [self.get_symbol_sentiment(symbol, days_back) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sentiment_map = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error getting sentiment for {symbol}: {result}")
                sentiment_map[symbol] = {
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0,
                    'article_count': 0,
                    'source': 'error'
                }
            else:
                sentiment_map[symbol] = result

        return sentiment_map

    def get_trading_signal(
        self,
        sentiment_result: Dict,
        bullish_threshold: float = 0.3,
        bearish_threshold: float = -0.3,
        min_confidence: float = 0.5
    ) -> str:
        """
        Convert sentiment result to trading signal.

        Args:
            sentiment_result: Result from get_symbol_sentiment
            bullish_threshold: Score threshold for bullish signal
            bearish_threshold: Score threshold for bearish signal
            min_confidence: Minimum confidence required

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        score = sentiment_result.get('score', 0)
        confidence = sentiment_result.get('confidence', 0)

        if confidence < min_confidence:
            return 'neutral'

        if score >= bullish_threshold:
            return 'bullish'
        elif score <= bearish_threshold:
            return 'bearish'
        else:
            return 'neutral'


# Convenience function for quick sentiment check
async def get_quick_sentiment(symbol: str) -> Tuple[str, float]:
    """
    Quick sentiment check for a symbol.

    Returns:
        Tuple of (sentiment, score)

    Example:
        sentiment, score = await get_quick_sentiment('AAPL')
    """
    analyzer = SentimentAnalyzer()
    result = await analyzer.get_symbol_sentiment(symbol)
    return result['sentiment'], result['score']
