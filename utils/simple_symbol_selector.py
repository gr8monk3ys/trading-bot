#!/usr/bin/env python3
"""
Simple Symbol Selector - Uses Alpaca API to find liquid, tradable stocks

Instead of complex scanning with buggy sentiment analysis, this uses Alpaca's
built-in APIs to find:
- Most active stocks (high volume = easier to trade)
- Tradable stocks that meet our criteria
- Stocks with reasonable price ranges
- Sector rotation for dynamic allocation based on economic phase

KISS principle: Keep it simple, make it work FIRST.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

# Lazy import for sector rotation
SectorRotator = None

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

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        use_sector_rotation: bool = True,
        broker=None,
    ):
        """
        Initialize with Alpaca credentials.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading
            use_sector_rotation: Enable sector rotation for dynamic allocation
            broker: Trading broker instance (for sector rotation)
        """
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.broker = broker

        # Criteria for stock selection
        self.min_price = 10.0
        self.max_price = 500.0
        self.min_volume = 1_000_000  # 1M shares/day minimum

        # Sector rotation
        self.use_sector_rotation = use_sector_rotation and broker is not None
        self.sector_rotator = None

        if self.use_sector_rotation:
            self._init_sector_rotator()

    def _init_sector_rotator(self):
        """Lazy-load and initialize sector rotator."""
        global SectorRotator
        if SectorRotator is None:
            try:
                from utils.sector_rotation import SectorRotator
            except ImportError:
                logger.warning("Sector rotation not available - feature disabled")
                self.use_sector_rotation = False
                return

        self.sector_rotator = SectorRotator(self.broker, use_etfs=False)
        logger.info("Sector rotator initialized for dynamic symbol selection")

    def get_all_tradable_stocks(self) -> List[str]:
        """Get all stocks that are tradable on Alpaca."""
        try:
            request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
            assets = self.trading_client.get_all_assets(request)

            # Filter to tradable stocks only
            tradable = [
                asset.symbol
                for asset in assets
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
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK.B",
            "V",
            "UNH",
            "JNJ",
            "WMT",
            "XOM",
            "JPM",
            "LLY",
            "MA",
            "PG",
            "AVGO",
            "HD",
            "CVX",
            "MRK",
            "ABBV",
            "KO",
            "PEP",
            "COST",
            "ADBE",
            "MCD",
            "CSCO",
            "ACN",
            "CRM",
            "NFLX",
            "LIN",
            "ABT",
            "TMO",
            "ORCL",
            "NKE",
            "AMD",
            "DHR",
            "VZ",
            "CMCSA",
            "DIS",
            "TXN",
            "INTC",
            "WFC",
            "UPS",
            "QCOM",
            "PM",
            "NEE",
            "UNP",
            "BA",
            "RTX",
            "HON",
            "SPGI",
            "INTU",
            "LOW",
            "AMGN",
            "IBM",
            "CAT",
            "GE",
            "SBUX",
            "T",
            "AMAT",
            "BMY",
            "DE",
            "AXP",
            "ELV",
            "BLK",
            "BKNG",
            "GILD",
            "ADI",
            "MDLZ",
            "LMT",
            "VRTX",
            "SYK",
            "ISRG",
            "TJX",
            "PLD",
            "CI",
            "REGN",
            "MMC",
            "C",
            "ZTS",
            "PGR",
            "CB",
            "ETN",
            "SO",
            "NOC",
            "DUK",
            "CME",
            "SHW",
            "BSX",
            "MU",
            "BDX",
            "EOG",
            "TGT",
            "SCHW",
            "PNC",
            "USB",
            "CL",
            "GD",
        ]

        # Popular growth/momentum stocks
        growth_stocks = [
            "PLTR",
            "SNOW",
            "NET",
            "DDOG",
            "CRWD",
            "ZS",
            "MDB",
            "SHOP",
            "SQ",
            "COIN",
            "RBLX",
            "U",
            "RIVN",
            "LCID",
            "NIO",
            "SOFI",
            "HOOD",
            "ABNB",
            "DASH",
            "UBER",
        ]

        # Popular ETFs for diversification
        etfs = ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EEM", "GLD", "SLV", "TLT"]

        # Combine and dedupe (preserving order so sp100 comes first)
        all_symbols = list(dict.fromkeys(sp100 + growth_stocks + etfs))

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
                    symbol_or_symbols=[symbol], timeframe=TimeFrame.Day, start=start, end=end
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

                filtered.append(
                    {
                        "symbol": symbol,
                        "price": price,
                        "avg_volume": avg_volume,
                        "momentum_5d": momentum_pct,
                        "score": abs(momentum_pct),  # Simple score: higher momentum = higher score
                    }
                )

            except Exception as e:
                logger.debug(f"Error checking {symbol}: {e}")
                continue

        # Sort by momentum (absolute value - we want movement in either direction)
        filtered.sort(key=lambda x: x["score"], reverse=True)

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
        top_stocks = [stock for stock in filtered if stock["score"] >= min_score][:top_n]

        symbols = [stock["symbol"] for stock in top_stocks]

        logger.info(f"Selected {len(symbols)} top symbols:")
        for stock in top_stocks:
            logger.info(
                f"  {stock['symbol']:6s} - "
                f"${stock['price']:7.2f}, "
                f"Vol: {stock['avg_volume']/1e6:5.1f}M, "
                f"5d: {stock['momentum_5d']:+5.1f}%"
            )

        return symbols

    async def select_with_sector_rotation(
        self,
        top_n: int = 20,
        min_score: float = 0.0,
        blend_ratio: float = 0.5,
    ) -> Dict[str, float]:
        """
        Select symbols using sector rotation for dynamic allocation.

        Combines:
        1. Sector rotation recommendations (based on economic phase)
        2. Traditional momentum/liquidity filtering

        Args:
            top_n: Maximum number of symbols to return
            min_score: Minimum momentum score
            blend_ratio: How much to weight sector rotation (0 = momentum only, 1 = sector only)

        Returns:
            Dict of symbol -> weight (sum = 1.0)
        """
        result = {}

        # If sector rotation not available, fall back to standard selection
        if not self.sector_rotator:
            symbols = self.select_top_symbols(top_n, min_score)
            equal_weight = 1.0 / len(symbols) if symbols else 0
            return {s: equal_weight for s in symbols}

        try:
            # Get sector rotation recommendations
            sector_stocks = await self.sector_rotator.get_recommended_stocks(
                top_n=int(top_n * blend_ratio) + 5,
                stocks_per_sector=3,
            )

            # Get allocations for weighting
            allocations = await self.sector_rotator.get_sector_allocations()

            # Map stocks to their sector weights
            sector_weights = {}
            for sector_etf, weight in allocations.items():
                if sector_etf in self.sector_rotator.SECTOR_STOCKS:
                    for stock in self.sector_rotator.SECTOR_STOCKS[sector_etf]:
                        sector_weights[stock] = weight

            # Get momentum-filtered stocks
            candidates = self.get_most_active_stocks(top_n=100)
            filtered = self.filter_by_criteria(candidates)
            momentum_stocks = [s["symbol"] for s in filtered if s["score"] >= min_score]

            # Blend: prioritize stocks that appear in both lists
            combined = []
            seen = set()

            # First: stocks in both sector rotation AND momentum (highest priority)
            for symbol in sector_stocks:
                if symbol in momentum_stocks and symbol not in seen:
                    weight = sector_weights.get(symbol, 0.09) * 1.5  # Bonus for appearing in both
                    combined.append((symbol, weight))
                    seen.add(symbol)

            # Second: sector rotation stocks not in momentum
            for symbol in sector_stocks:
                if symbol not in seen:
                    weight = sector_weights.get(symbol, 0.09)
                    combined.append((symbol, weight * blend_ratio))
                    seen.add(symbol)

            # Third: momentum stocks not in sector rotation (fill remaining slots)
            remaining = top_n - len(combined)
            for symbol in momentum_stocks[:remaining]:
                if symbol not in seen:
                    weight = 0.05 * (1 - blend_ratio)  # Lower weight for non-sector stocks
                    combined.append((symbol, weight))
                    seen.add(symbol)

            # Normalize weights
            combined = combined[:top_n]
            total_weight = sum(w for _, w in combined)
            if total_weight > 0:
                result = {s: w / total_weight for s, w in combined}
            else:
                equal_weight = 1.0 / len(combined) if combined else 0
                result = {s: equal_weight for s, _ in combined}

            # Log selection
            logger.info(f"Selected {len(result)} symbols with sector rotation:")
            for symbol, weight in sorted(result.items(), key=lambda x: -x[1])[:10]:
                sector = self._get_stock_sector(symbol)
                logger.info(f"  {symbol:6s} ({sector:25s}): {weight:.1%}")

            return result

        except Exception as e:
            logger.error(f"Error in sector rotation selection: {e}")
            # Fallback to standard selection
            symbols = self.select_top_symbols(top_n, min_score)
            equal_weight = 1.0 / len(symbols) if symbols else 0
            return {s: equal_weight for s in symbols}

    def _get_stock_sector(self, symbol: str) -> str:
        """Get the sector name for a stock symbol."""
        if not self.sector_rotator:
            return "Unknown"

        for sector_etf, stocks in self.sector_rotator.SECTOR_STOCKS.items():
            if symbol in stocks:
                return self.sector_rotator.SECTOR_ETFS.get(sector_etf, sector_etf)

        return "Other"

    async def get_sector_report(self) -> Optional[Dict]:
        """
        Get comprehensive sector rotation report.

        Returns:
            Dict with sector analysis or None if sector rotation disabled
        """
        if not self.sector_rotator:
            return None

        try:
            return await self.sector_rotator.get_sector_report()
        except Exception as e:
            logger.error(f"Error getting sector report: {e}")
            return None


if __name__ == "__main__":
    """Test the symbol selector."""
    import os

    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    selector = SimpleSymbolSelector(
        api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY"), paper=True
    )

    print("\n" + "=" * 80)
    print("SIMPLE SYMBOL SELECTOR - Finding tradable, liquid stocks")
    print("=" * 80 + "\n")

    symbols = selector.select_top_symbols(top_n=20, min_score=1.0)

    print(f"\n{'='*80}")
    print(f"SELECTED {len(symbols)} SYMBOLS FOR TRADING:")
    print(f"{'='*80}")
    print(", ".join(symbols))
    print()
