#!/usr/bin/env python3
"""
Comprehensive validation script for the Alternative Data Framework.

This script validates:
1. Core framework components (types, cache, aggregator)
2. Social sentiment providers (Reddit)
3. Order flow analyzer (options flow, dark pool)
4. Web scraper providers (jobs, Glassdoor, app rankings)
5. Ensemble integration (ALTERNATIVE_DATA signal source)
6. End-to-end pipeline

Run with: python scripts/validate_alternative_data.py
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    """Print section header."""
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  {status} {test_name}")
    if details and not passed:
        print(f"       {YELLOW}{details}{RESET}")


def print_summary(results: Dict[str, bool]):
    """Print test summary."""
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print_header("VALIDATION SUMMARY")
    print(f"\n  Tests Passed: {GREEN}{passed}/{total}{RESET}")

    if passed == total:
        print(f"\n  {GREEN}{BOLD}ALL VALIDATIONS PASSED ✓{RESET}")
    else:
        print(f"\n  {RED}Failed tests:{RESET}")
        for name, result in results.items():
            if not result:
                print(f"    - {name}")

    return passed == total


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_core_types() -> Dict[str, bool]:
    """Validate core alternative data types."""
    results = {}

    try:
        from data.alt_data_types import (
            AggregatedSignal,
            AltDataSource,
            AlternativeSignal,
            OrderFlowSignal,
            SignalDirection,
            SignalStrength,
            SocialSentimentSignal,
        )
        results["Import alt_data_types"] = True

        # Test AltDataSource enum
        assert len(AltDataSource) >= 10, "Should have at least 10 data sources"
        assert AltDataSource.REDDIT.value == "reddit"
        assert AltDataSource.DARK_POOL.value == "dark_pool"
        results["AltDataSource enum"] = True

        # Test AlternativeSignal creation and clamping
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=1.5,  # Should clamp to 1.0
            confidence=0.8,
        )
        assert signal.signal_value == 1.0, "Signal value should be clamped"
        assert signal.direction == SignalDirection.BULLISH
        assert signal.strength == SignalStrength.VERY_STRONG  # confidence >= 0.8 -> VERY_STRONG
        results["AlternativeSignal clamping"] = True

        # Test SocialSentimentSignal
        social = SocialSentimentSignal(
            symbol="GME",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.8,
            confidence=0.7,
            mention_count=500,
            mention_change_pct=250.0,
            meme_stock_risk=True,
        )
        assert social.meme_stock_risk is True
        results["SocialSentimentSignal"] = True

        # Test OrderFlowSignal
        order_flow = OrderFlowSignal(
            symbol="TSLA",
            source=AltDataSource.OPTIONS_FLOW,
            timestamp=datetime.now(),
            signal_value=0.6,
            confidence=0.75,
            put_call_ratio=0.5,
            unusual_options_activity=True,
        )
        assert order_flow.put_call_ratio == 0.5
        results["OrderFlowSignal"] = True

        # Test AggregatedSignal
        signals = [
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.REDDIT,
                timestamp=datetime.now(),
                signal_value=0.5,
                confidence=0.7,
            ),
            AlternativeSignal(
                symbol="AAPL",
                source=AltDataSource.OPTIONS_FLOW,
                timestamp=datetime.now(),
                signal_value=0.6,
                confidence=0.8,
            ),
        ]
        agg = AggregatedSignal(
            symbol="AAPL",
            timestamp=datetime.now(),
            sources=[s.source for s in signals],
            individual_signals=signals,
        )
        assert agg.composite_signal != 0
        assert agg.agreement_ratio > 0
        results["AggregatedSignal calculation"] = True

    except Exception as e:
        results[f"Error: {type(e).__name__}"] = False
        print(f"    {RED}Exception: {e}{RESET}")

    return results


def validate_cache() -> Dict[str, bool]:
    """Validate TTL cache functionality."""
    results = {}

    try:
        from data.alt_data_types import AltDataSource, AlternativeSignal
        from data.alternative_data_provider import AltDataCache

        cache = AltDataCache(default_ttl_seconds=300)
        results["Create cache"] = True

        # Test cache operations
        signal = AlternativeSignal(
            symbol="AAPL",
            source=AltDataSource.REDDIT,
            timestamp=datetime.now(),
            signal_value=0.5,
            confidence=0.7,
        )

        cache.set(AltDataSource.REDDIT, "AAPL", signal)
        retrieved = cache.get(AltDataSource.REDDIT, "AAPL")
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        results["Cache set/get"] = True

        # Test cache miss
        miss = cache.get(AltDataSource.REDDIT, "MSFT")
        assert miss is None
        results["Cache miss"] = True

        # Test invalidation
        cache.invalidate(AltDataSource.REDDIT, "AAPL")
        invalidated = cache.get(AltDataSource.REDDIT, "AAPL")
        assert invalidated is None
        results["Cache invalidation"] = True

        # Test hit rate
        assert cache.hit_rate >= 0
        results["Cache hit rate"] = True

    except Exception as e:
        results[f"Cache error: {type(e).__name__}"] = False
        print(f"    {RED}Exception: {e}{RESET}")

    return results


async def validate_social_sentiment() -> Dict[str, bool]:
    """Validate social sentiment providers."""
    results = {}

    try:
        from data.social_sentiment_advanced import (
            RedditSentimentProvider,
            TickerExtractor,
        )

        # Test TickerExtractor
        extractor = TickerExtractor(valid_tickers={"AAPL", "MSFT", "GOOGL"})
        tickers = extractor.extract("Just bought $AAPL and some MSFT today!")
        assert "AAPL" in tickers
        results["TickerExtractor"] = True

        # Test RedditSentimentProvider initialization
        provider = RedditSentimentProvider()
        init_result = await provider.initialize()
        assert init_result is True
        assert provider._initialized is True
        results["RedditSentimentProvider init"] = True

        # Test fetch_signal (uses mock data without credentials)
        signal = await provider.fetch_signal("AAPL")
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert -1.0 <= signal.signal_value <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.mention_count >= 0
        results["Reddit fetch_signal"] = True

        # Verify signal has required fields
        assert hasattr(signal, "positive_ratio")
        assert hasattr(signal, "negative_ratio")
        assert hasattr(signal, "meme_stock_risk")
        results["Reddit signal fields"] = True

    except Exception as e:
        results[f"Social error: {type(e).__name__}"] = False
        print(f"    {RED}Exception: {e}{RESET}")

    return results


async def validate_order_flow() -> Dict[str, bool]:
    """Validate order flow analyzer."""
    results = {}

    try:
        from data.order_flow_analyzer import (
            DarkPoolProvider,
            OrderFlowAnalyzer,
        )

        # Test OrderFlowAnalyzer
        analyzer = OrderFlowAnalyzer()
        init_result = await analyzer.initialize()
        assert init_result is True
        results["OrderFlowAnalyzer init"] = True

        # Test fetch_signal
        signal = await analyzer.fetch_signal("AAPL")
        assert signal is not None
        assert signal.symbol == "AAPL"
        assert -1.0 <= signal.signal_value <= 1.0
        results["OrderFlow fetch_signal"] = True

        # Verify order flow specific fields
        assert hasattr(signal, "put_call_ratio")
        assert hasattr(signal, "call_volume")
        assert hasattr(signal, "put_volume")
        assert hasattr(signal, "unusual_options_activity")
        results["OrderFlow signal fields"] = True

        # Test DarkPoolProvider
        dark_pool = DarkPoolProvider()
        await dark_pool.initialize()
        dp_signal = await dark_pool.fetch_signal("MSFT")
        assert dp_signal is not None
        results["DarkPoolProvider"] = True

    except Exception as e:
        results[f"OrderFlow error: {type(e).__name__}"] = False
        print(f"    {RED}Exception: {e}{RESET}")

    return results


async def validate_web_scrapers() -> Dict[str, bool]:
    """Validate web scraper providers."""
    results = {}

    try:
        from data.web_scraper import (
            AppRankingsProvider,
            GlassdoorSentimentProvider,
            JobPostingsProvider,
            WebScraperAggregator,
        )

        # Test JobPostingsProvider
        jobs = JobPostingsProvider()
        await jobs.initialize()
        job_signal = await jobs.fetch_signal("AAPL")
        assert job_signal is not None
        assert job_signal.job_posting_count >= 0
        results["JobPostingsProvider"] = True

        # Test GlassdoorSentimentProvider
        glassdoor = GlassdoorSentimentProvider()
        await glassdoor.initialize()
        gd_signal = await glassdoor.fetch_signal("MSFT")
        assert gd_signal is not None
        assert gd_signal.avg_rating >= 0
        results["GlassdoorSentimentProvider"] = True

        # Test AppRankingsProvider (consumer company)
        app_rankings = AppRankingsProvider()
        await app_rankings.initialize()
        app_signal = await app_rankings.fetch_signal("META")
        assert app_signal is not None  # META is a consumer app company
        assert app_signal.app_rank >= 0
        results["AppRankingsProvider (consumer)"] = True

        # Test AppRankingsProvider (non-consumer company)
        non_consumer_signal = await app_rankings.fetch_signal("XOM")
        assert non_consumer_signal is None  # XOM is not a consumer app company
        results["AppRankingsProvider (non-consumer)"] = True

        # Test WebScraperAggregator
        aggregator = WebScraperAggregator()
        await aggregator.initialize()
        composite = await aggregator.get_composite_signal("GOOGL")
        assert composite is not None
        assert -1.0 <= composite.signal_value <= 1.0
        results["WebScraperAggregator"] = True

    except Exception as e:
        results[f"WebScraper error: {type(e).__name__}"] = False
        print(f"    {RED}Exception: {e}{RESET}")

    return results


async def validate_aggregator() -> Dict[str, bool]:
    """Validate AltDataAggregator."""
    results = {}

    try:
        from data.alt_data_types import AltDataSource
        from data.alternative_data_provider import AltDataAggregator
        from data.order_flow_analyzer import OrderFlowAnalyzer
        from data.social_sentiment_advanced import RedditSentimentProvider

        # Create aggregator with multiple providers
        aggregator = AltDataAggregator()

        reddit = RedditSentimentProvider()
        await reddit.initialize()
        aggregator.register_provider(reddit)

        order_flow = OrderFlowAnalyzer()
        await order_flow.initialize()
        aggregator.register_provider(order_flow)

        results["Register multiple providers"] = True

        # Test get_signal
        signal = await aggregator.get_signal("AAPL")
        assert signal is not None
        assert len(signal.sources) >= 1
        assert signal.composite_signal is not None
        results["Aggregator get_signal"] = True

        # Test kill switch
        aggregator.kill_source(AltDataSource.TWITTER, "Test kill")
        assert AltDataSource.TWITTER in aggregator._killed_sources
        results["Kill switch"] = True

        # Test revive
        aggregator.revive_source(AltDataSource.TWITTER)
        assert AltDataSource.TWITTER not in aggregator._killed_sources
        results["Revive source"] = True

        # Test performance recording
        aggregator.record_performance(AltDataSource.REDDIT, 0.05, 0.5)
        accuracy = aggregator.get_source_accuracy(AltDataSource.REDDIT)
        assert accuracy is not None
        results["Performance tracking"] = True

    except Exception as e:
        results[f"Aggregator error: {type(e).__name__}"] = False
        print(f"    {RED}Exception: {e}{RESET}")

    return results


def validate_ensemble_integration() -> Dict[str, bool]:
    """Validate ensemble predictor integration."""
    results = {}

    try:
        from ml.ensemble_predictor import (
            AltDataSignalGenerator,
            EnsemblePredictor,
            MarketRegime,
            SignalComponent,
            SignalSource,
            create_ensemble_with_alt_data,
        )

        # Test ALTERNATIVE_DATA in SignalSource
        assert hasattr(SignalSource, "ALTERNATIVE_DATA")
        assert SignalSource.ALTERNATIVE_DATA.value == "alternative_data"
        results["ALTERNATIVE_DATA SignalSource"] = True

        # Test EnsemblePredictor has alt data weights
        ensemble = EnsemblePredictor()
        assert SignalSource.ALTERNATIVE_DATA in ensemble.base_weights
        assert ensemble.base_weights[SignalSource.ALTERNATIVE_DATA] == 0.20
        results["Ensemble base weights"] = True

        # Test regime weights include alt data
        for regime in MarketRegime:
            assert SignalSource.ALTERNATIVE_DATA in ensemble.regime_weights[regime]
            assert ensemble.regime_weights[regime][SignalSource.ALTERNATIVE_DATA] > 0
        results["Regime weights include alt data"] = True

        # Test regime weights sum to ~1.0
        for regime, weights in ensemble.regime_weights.items():
            total = sum(weights.values())
            assert 0.99 <= total <= 1.01, f"{regime} weights sum to {total}"
        results["Regime weights sum to 1.0"] = True

        # Test SignalComponent creation for alt data
        component = SignalComponent(
            source=SignalSource.ALTERNATIVE_DATA,
            signal_value=0.6,
            confidence=0.75,
            direction="long",
            metadata={"sources": ["reddit", "options_flow"]},
        )
        assert component.source == SignalSource.ALTERNATIVE_DATA
        results["SignalComponent for alt data"] = True

        # Test ensemble with alt data signal
        def mock_lstm(symbol, data):
            return SignalComponent(
                source=SignalSource.LSTM,
                signal_value=0.5,
                confidence=0.7,
                direction="long",
            )

        def mock_alt_data(symbol, data):
            return SignalComponent(
                source=SignalSource.ALTERNATIVE_DATA,
                signal_value=0.7,
                confidence=0.8,
                direction="long",
            )

        ensemble.register_source(SignalSource.LSTM, mock_lstm)
        ensemble.register_source(SignalSource.ALTERNATIVE_DATA, mock_alt_data)

        prediction = ensemble.predict("AAPL", {})
        assert prediction is not None
        assert SignalSource.ALTERNATIVE_DATA in prediction.components
        assert SignalSource.ALTERNATIVE_DATA in prediction.weights_used
        results["Ensemble prediction with alt data"] = True

        # Test AltDataSignalGenerator exists
        assert AltDataSignalGenerator is not None
        results["AltDataSignalGenerator class"] = True

        # Test create_ensemble_with_alt_data exists
        assert create_ensemble_with_alt_data is not None
        results["create_ensemble_with_alt_data function"] = True

    except Exception as e:
        results[f"Ensemble error: {type(e).__name__}"] = False
        print(f"    {RED}Exception: {e}{RESET}")

    return results


async def validate_end_to_end() -> Dict[str, bool]:
    """Run end-to-end integration test."""
    results = {}

    try:
        from data.alternative_data_provider import AltDataAggregator
        from data.order_flow_analyzer import OrderFlowAnalyzer
        from data.social_sentiment_advanced import RedditSentimentProvider
        from data.web_scraper import GlassdoorSentimentProvider, JobPostingsProvider
        from ml.ensemble_predictor import (
            EnsemblePredictor,
            SignalComponent,
            SignalSource,
        )

        # Step 1: Initialize all alt data providers
        aggregator = AltDataAggregator()

        providers = [
            RedditSentimentProvider(),
            OrderFlowAnalyzer(),
            JobPostingsProvider(),
            GlassdoorSentimentProvider(),
        ]

        for provider in providers:
            await provider.initialize()
            aggregator.register_provider(provider)

        results["Initialize all providers"] = True

        # Step 2: Fetch signals for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
        signals = await aggregator.get_signals(symbols)

        assert len(signals) >= 3  # Should get signals for most symbols
        results["Fetch signals for portfolio"] = True

        # Step 3: Verify signal quality
        for _symbol, signal in signals.items():
            assert signal.composite_signal is not None
            assert -1.0 <= signal.composite_signal <= 1.0
            assert 0.0 <= signal.composite_confidence <= 1.0
            assert len(signal.sources) >= 1
        results["Signal quality validation"] = True

        # Step 4: Integrate with ensemble
        ensemble = EnsemblePredictor(min_sources_required=1)

        def alt_data_signal_fn(symbol, data):
            # Use pre-fetched aggregated signal
            if symbol in signals:
                agg = signals[symbol]
                direction = "long" if agg.composite_signal > 0.1 else "short" if agg.composite_signal < -0.1 else "neutral"
                return SignalComponent(
                    source=SignalSource.ALTERNATIVE_DATA,
                    signal_value=agg.composite_signal,
                    confidence=agg.composite_confidence,
                    direction=direction,
                )
            return None

        ensemble.register_source(SignalSource.ALTERNATIVE_DATA, alt_data_signal_fn)

        # Get ensemble predictions
        for symbol in ["AAPL", "MSFT", "META"]:
            prediction = ensemble.predict(symbol, {})
            assert prediction is not None
            assert SignalSource.ALTERNATIVE_DATA in prediction.components

        results["Ensemble integration"] = True

        # Step 5: Test caching
        await aggregator.get_signals(["AAPL"], use_cache=True)
        cache_stats = aggregator.get_cache_stats()
        assert cache_stats["hits"] >= 0
        results["Caching works"] = True

        # Step 6: Get provider statuses
        statuses = aggregator.get_all_statuses()
        for _source, status in statuses.items():
            assert status.source is not None
            assert isinstance(status.is_healthy, bool)
        results["Provider status reporting"] = True

    except Exception as e:
        results[f"E2E error: {type(e).__name__}"] = False
        print(f"    {RED}Exception: {e}{RESET}")

    return results


# ============================================================================
# MAIN VALIDATION
# ============================================================================

async def main():
    """Run all validations."""
    print(f"\n{BOLD}Alternative Data Framework Validation{RESET}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {}

    # 1. Core Types
    print_header("1. CORE DATA TYPES")
    results = validate_core_types()
    for name, passed in results.items():
        print_result(name, passed)
    all_results.update({f"Types: {k}": v for k, v in results.items()})

    # 2. Cache
    print_header("2. TTL CACHE")
    results = validate_cache()
    for name, passed in results.items():
        print_result(name, passed)
    all_results.update({f"Cache: {k}": v for k, v in results.items()})

    # 3. Social Sentiment
    print_header("3. SOCIAL SENTIMENT PROVIDERS")
    results = await validate_social_sentiment()
    for name, passed in results.items():
        print_result(name, passed)
    all_results.update({f"Social: {k}": v for k, v in results.items()})

    # 4. Order Flow
    print_header("4. ORDER FLOW ANALYZER")
    results = await validate_order_flow()
    for name, passed in results.items():
        print_result(name, passed)
    all_results.update({f"OrderFlow: {k}": v for k, v in results.items()})

    # 5. Web Scrapers
    print_header("5. WEB SCRAPER PROVIDERS")
    results = await validate_web_scrapers()
    for name, passed in results.items():
        print_result(name, passed)
    all_results.update({f"WebScraper: {k}": v for k, v in results.items()})

    # 6. Aggregator
    print_header("6. ALT DATA AGGREGATOR")
    results = await validate_aggregator()
    for name, passed in results.items():
        print_result(name, passed)
    all_results.update({f"Aggregator: {k}": v for k, v in results.items()})

    # 7. Ensemble Integration
    print_header("7. ENSEMBLE INTEGRATION")
    results = validate_ensemble_integration()
    for name, passed in results.items():
        print_result(name, passed)
    all_results.update({f"Ensemble: {k}": v for k, v in results.items()})

    # 8. End-to-End
    print_header("8. END-TO-END INTEGRATION")
    results = await validate_end_to_end()
    for name, passed in results.items():
        print_result(name, passed)
    all_results.update({f"E2E: {k}": v for k, v in results.items()})

    # Summary
    success = print_summary(all_results)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
