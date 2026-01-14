#!/usr/bin/env python3
"""
Simple Symbol Selector - Uses Alpaca API to find liquid, tradable stocks

Instead of complex scanning with buggy sentiment analysis, this uses Alpaca's
built-in APIs to find:
- Most active stocks (high volume = easier to trade)
- Tradable stocks that meet our criteria
- Stocks with reasonable price ranges

KISS principle: Keep it simple, make it work FIRST.
"""

import logging
from typing import List, Dict, Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SimpleSymbolSelector:
    """
    Simple, working symbol selector that focuses on liquid, tradable stocks.

    No complex TA-lib calculations, no sentiment analysis - just find stocks that:
    1. Are actively traded (high volume)
    2. Have reasonable prices ($10-$500)
    3. Are tradable on Alpaca
    4. Have recent price movement (not stagnant)
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """Initialize with Alpaca credentials."""
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

        # Criteria for stock selection
        self.min_price = 10.0
        self.max_price = 500.0
        self.min_volume = 1_000_000  # 1M shares/day minimum

    def get_all_tradable_stocks(self) -> List[str]:
        """Get all stocks that are tradable on Alpaca."""
        try:
            request = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY
            )
            assets = self.trading_client.get_all_assets(request)

            # Filter to tradable stocks only
            tradable = [
                asset.symbol for asset in assets
                if asset.tradable and asset.fractionable and asset.marginable
            ]

            logger.info(f"Found {len(tradable)} tradable stocks on Alpaca")
            return tradable

        except Exception as e:
            logger.error(f"Error getting tradable stocks: {e}")
            return []

    def get_most_active_stocks(self, top_n: int = 50) -> List[str]:
        """
        Get most active stocks by using a curated list of liquid stocks.

        Uses S&P 100 (most liquid large caps) + popular growth stocks.
        This is more reliable than trying to scan all stocks.
        """
        # S&P 100 (most liquid large caps)
        sp100 = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'V', 'UNH',
            'JNJ', 'WMT', 'XOM', 'JPM', 'LLY', 'MA', 'PG', 'AVGO', 'HD', 'CVX',
            'MRK', 'ABBV', 'KO', 'PEP', 'COST', 'ADBE', 'MCD', 'CSCO', 'ACN', 'CRM',
            'NFLX', 'LIN', 'ABT', 'TMO', 'ORCL', 'NKE', 'AMD', 'DHR', 'VZ', 'CMCSA',
            'DIS', 'TXN', 'INTC', 'WFC', 'UPS', 'QCOM', 'PM', 'NEE', 'UNP', 'BA',
            'RTX', 'HON', 'SPGI', 'INTU', 'LOW', 'AMGN', 'IBM', 'CAT', 'GE', 'SBUX',
            'T', 'AMAT', 'BMY', 'DE', 'AXP', 'ELV', 'BLK', 'BKNG', 'GILD', 'ADI',
            'MDLZ', 'LMT', 'VRTX', 'SYK', 'ISRG', 'TJX', 'PLD', 'CI', 'REGN', 'MMC',
            'C', 'ZTS', 'PGR', 'CB', 'ETN', 'SO', 'NOC', 'DUK', 'CME', 'SHW',
            'BSX', 'MU', 'BDX', 'EOG', 'TGT', 'SCHW', 'PNC', 'USB', 'CL', 'GD'
        ]

        # Popular growth/momentum stocks
        growth_stocks = [
            'PLTR', 'SNOW', 'NET', 'DDOG', 'CRWD', 'ZS', 'MDB', 'SHOP', 'SQ', 'COIN',
            'RBLX', 'U', 'RIVN', 'LCID', 'NIO', 'SOFI', 'HOOD', 'ABNB', 'DASH', 'UBER'
        ]

        # Popular ETFs for diversification
        etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EEM', 'GLD', 'SLV', 'TLT']

        # Combine and dedupe
        all_symbols = list(set(sp100 + growth_stocks + etfs))

        logger.info(f"Using {len(all_symbols)} curated liquid stocks")
        return all_symbols[:top_n] if top_n else all_symbols

    def filter_by_criteria(self, symbols: List[str]) -> List[Dict]:
        """
        Filter symbols by our trading criteria.

        Returns list of dicts with symbol and basic stats.
        """
        filtered = []

        for symbol in symbols:
            try:
                # Get recent trade to check price
                latest_request = StockLatestTradeRequest(symbol_or_symbols=[symbol])
                latest_trade = self.data_client.get_stock_latest_trade(latest_request)

                if symbol not in latest_trade:
                    continue

                price = float(latest_trade[symbol].price)

                # Check price range
                if not (self.min_price <= price <= self.max_price):
                    continue

                # Get volume data from recent bars
                end = datetime.now()
                start = end - timedelta(days=5)
                bars_request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end
                )
                bars = self.data_client.get_stock_bars(bars_request)

                if symbol not in bars.data or len(bars.data[symbol]) == 0:
                    continue

                # Calculate average volume
                avg_volume = sum(bar.volume for bar in bars.data[symbol]) / len(bars.data[symbol])

                # Check volume
                if avg_volume < self.min_volume:
                    continue

                # Calculate simple momentum (5-day price change)
                first_close = float(bars.data[symbol][0].close)
                last_close = float(bars.data[symbol][-1].close)
                momentum_pct = ((last_close - first_close) / first_close) * 100

                filtered.append({
                    'symbol': symbol,
                    'price': price,
                    'avg_volume': avg_volume,
                    'momentum_5d': momentum_pct,
                    'score': abs(momentum_pct)  # Simple score: higher momentum = higher score
                })

            except Exception as e:
                logger.debug(f"Error checking {symbol}: {e}")
                continue

        # Sort by momentum (absolute value - we want movement in either direction)
        filtered.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"Filtered to {len(filtered)} stocks meeting criteria")
        return filtered

    def select_top_symbols(self, top_n: int = 10, min_score: float = 0.0) -> List[str]:
        """
        Select top N symbols for trading.

        Returns just the symbol list, ready to use.
        """
        # Get candidate stocks
        candidates = self.get_most_active_stocks(top_n=100)

        # Filter by our criteria
        filtered = self.filter_by_criteria(candidates)

        # Filter by min score and take top N
        top_stocks = [
            stock for stock in filtered
            if stock['score'] >= min_score
        ][:top_n]

        symbols = [stock['symbol'] for stock in top_stocks]

        logger.info(f"Selected {len(symbols)} top symbols:")
        for stock in top_stocks:
            logger.info(
                f"  {stock['symbol']:6s} - "
                f"${stock['price']:7.2f}, "
                f"Vol: {stock['avg_volume']/1e6:5.1f}M, "
                f"5d: {stock['momentum_5d']:+5.1f}%"
            )

        return symbols


if __name__ == "__main__":
    """Test the symbol selector."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    selector = SimpleSymbolSelector(
        api_key=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper=True
    )

    print("\n" + "="*80)
    print("SIMPLE SYMBOL SELECTOR - Finding tradable, liquid stocks")
    print("="*80 + "\n")

    symbols = selector.select_top_symbols(top_n=20, min_score=1.0)

    print(f"\n{'='*80}")
    print(f"SELECTED {len(symbols)} SYMBOLS FOR TRADING:")
    print(f"{'='*80}")
    print(", ".join(symbols))
    print()
