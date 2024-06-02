"""
AlpacaBroker stock market data mixin.

Contains:
    - get_last_price / get_last_prices (single + batch, TTL-cached)
    - get_bars (historical stock bars, multiple timeframes)
    - get_news (Alpaca News API)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, cast

from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

from config import ALPACA_CREDS

from brokers.alpaca._retry import DEBUG_MODE, retry_with_backoff

logger = logging.getLogger(__name__)


class AlpacaMarketDataMixin:
    """Stock historical bars, latest trades, news."""

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_last_price(self, symbol):
        """Get last price for a symbol with TTL-based caching."""
        try:
            # P2 FIX: Validate symbol before API call
            symbol = self._validate_symbol(symbol)

            # Performance optimization: Check cache first
            now = datetime.now()
            if symbol in self._price_cache:
                cached_price, cached_time = self._price_cache[symbol]
                if now - cached_time < self._price_cache_ttl:
                    logger.debug(f"Price cache hit for {symbol}: ${cached_price:.2f}")
                    return cached_price

            # Cache miss - fetch from API with timeout protection
            request_params = StockLatestTradeRequest(symbol_or_symbols=[symbol])
            response = await self._async_call_with_timeout(
                self.data_client.get_stock_latest_trade,
                request_params,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name=f"get_last_price({symbol})",
            )

            if symbol in response:
                price = float(response[symbol].price)
                # Cache the result before returning
                self._price_cache[symbol] = (price, now)
                return price
            else:
                logger.warning(f"No price data found for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error getting last price for {symbol}: {e}", exc_info=DEBUG_MODE)
            return None

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_last_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Get last trade prices for multiple symbols in a single API call.

        More efficient than calling get_last_price() multiple times.

        Args:
            symbols: List of stock symbols to get prices for

        Returns:
            Dict mapping symbol to price (None if not available)
        """
        if not symbols:
            return {}

        try:
            # Validate and normalize symbols
            validated_symbols = [self._validate_symbol(s) for s in symbols]

            request_params = StockLatestTradeRequest(symbol_or_symbols=validated_symbols)
            response = await self._async_call_with_timeout(
                self.data_client.get_stock_latest_trade,
                request_params,
                timeout=self.DEFAULT_API_TIMEOUT,
                operation_name=f"get_last_prices({len(validated_symbols)} symbols)",
            )

            result: Dict[str, Optional[float]] = {}
            now = datetime.now()
            for symbol in validated_symbols:
                if symbol in response:
                    price = float(response[symbol].price)
                    result[symbol] = price
                    # Update cache for individual symbol lookups
                    self._price_cache[symbol] = (price, now)
                else:
                    result[symbol] = None

            logger.debug(f"Fetched batch prices for {len(result)} symbols")
            return result

        except ValueError as e:
            logger.error(f"Invalid symbol in batch: {e}")
            return cast(Dict[str, Optional[float]], dict.fromkeys(symbols, None))
        except Exception as e:
            logger.error(f"Error fetching batch prices for {symbols}: {e}", exc_info=DEBUG_MODE)
            return cast(Dict[str, Optional[float]], dict.fromkeys(symbols, None))

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_bars(self, symbol, timeframe=TimeFrame.Day, limit=100, start=None, end=None):
        """Get historical bars for a symbol."""
        try:
            # P2 FIX: Validate symbol before API call
            symbol = self._validate_symbol(symbol)

            # Convert string timeframe to TimeFrame object if needed
            if isinstance(timeframe, str):
                timeframe_map = {
                    "1Min": TimeFrame.Minute,
                    "5Min": TimeFrame(5, "Min"),
                    "15Min": TimeFrame(15, "Min"),
                    "1Hour": TimeFrame.Hour,
                    "1Day": TimeFrame.Day,
                    "Day": TimeFrame.Day,
                    "Hour": TimeFrame.Hour,
                    "Minute": TimeFrame.Minute,
                }
                timeframe = timeframe_map.get(timeframe, TimeFrame.Day)
                logger.debug(f"Converted timeframe string to TimeFrame object: {timeframe}")

            # Set default dates if not provided
            if end is None:
                end = datetime.now().date()
            if start is None:
                start = (datetime.now() - timedelta(days=limit)).date()

            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol], timeframe=timeframe, start=start, end=end
            )

            # Use timeout-protected async call (longer timeout for data-heavy operations)
            bars = await self._async_call_with_timeout(
                self.data_client.get_stock_bars,
                request_params,
                timeout=self.DATA_API_TIMEOUT,
                operation_name=f"get_bars({symbol})",
            )

            if symbol in bars.data:
                return bars.data[symbol]
            else:
                logger.warning(f"No bar data found for {symbol}")
                return []

        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}", exc_info=DEBUG_MODE)
            return []

    @retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10)
    async def get_news(self, symbols, start=None, end=None, limit=50):
        """
        Get news for symbols using Alpaca News API.

        Args:
            symbols: Single symbol string or list of symbols
            start: Start datetime (default: 24 hours ago)
            end: End datetime (default: now)
            limit: Maximum number of articles to return

        Returns:
            List of news articles or empty list on error
        """
        try:
            from alpaca.data.historical.news import NewsClient
            from alpaca.data.requests import NewsRequest

            # Normalize symbols to list
            if isinstance(symbols, str):
                symbols = [symbols]

            # P2 FIX: Validate symbols before API call
            symbols = [self._validate_symbol(s) for s in symbols]

            # Set default dates if not provided
            if end is None:
                end = datetime.now()
            if start is None:
                start = end - timedelta(hours=24)
            # Initialize news client (lazy load)
            if not hasattr(self, "_news_client") or self._news_client is None:
                _api_key = ALPACA_CREDS["API_KEY"]
                _api_secret = ALPACA_CREDS["API_SECRET"]
                self._news_client = NewsClient(
                    api_key=_api_key,
                    secret_key=_api_secret,
                )

            request = NewsRequest(
                symbols=symbols,
                start=start,
                end=end,
                limit=limit,
            )

            # Use timeout-protected async call
            news_response = await self._async_call_with_timeout(
                self._news_client.get_news,
                request,
                timeout=self.DATA_API_TIMEOUT,
                operation_name=f"get_news({symbols})",
            )

            # Convert to list of dicts for easier consumption
            articles = []
            for item in news_response.news:
                articles.append(
                    {
                        "id": str(item.id),
                        "headline": item.headline or "",
                        "summary": item.summary or "",
                        "author": item.author or "",
                        "source": item.source or "",
                        "url": item.url or "",
                        "symbols": list(item.symbols) if item.symbols else [],
                        "created_at": item.created_at,
                        "updated_at": item.updated_at,
                    }
                )

            logger.debug(f"Fetched {len(articles)} news articles for {symbols}")
            return articles

        except ImportError as e:
            logger.error(
                f"News API import error: {e}. " "Ensure alpaca-py is installed with news support."
            )
            return []
        except ValueError as e:
            logger.error(f"Invalid symbol: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting news for {symbols}: {e}", exc_info=DEBUG_MODE)
            return []
