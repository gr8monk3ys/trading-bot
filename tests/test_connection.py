import os
import logging
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_alpaca_connection():
    """Test connection to Alpaca API with the provided credentials."""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")
        paper = os.getenv("PAPER", "True").lower() == "true"
        
        if not api_key or not api_secret:
            logger.error("API credentials not found in environment")
            return False
            
        logger.info(f"Testing connection with API_KEY: {api_key[:5]}... (Paper: {paper})")
        
        # Initialize the trading client with explicit endpoint
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper,
            url_override="https://paper-api.alpaca.markets" if paper else None
        )
        
        # Get account information
        account = trading_client.get_account()
        logger.info(f"Successfully connected to Alpaca account: {account.id}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Cash: ${float(account.cash):.2f}")
        logger.info(f"Portfolio value: ${float(account.portfolio_value):.2f}")
        
        # Get available assets
        request_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
        assets = trading_client.get_all_assets(request_params)
        logger.info(f"Retrieved {len(assets)} available assets")
        
        # Initialize the data client
        data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )
        
        # Test data retrieval
        symbols = ["AAPL", "MSFT"]
        logger.info(f"Testing data retrieval for: {', '.join(symbols)}")
        
        # Get latest trade data
        latest_trade_request = StockLatestTradeRequest(
            symbol_or_symbols=symbols
        )
        latest_trades = data_client.get_stock_latest_trade(latest_trade_request)
        
        for symbol in symbols:
            if symbol in latest_trades:
                price = latest_trades[symbol].price
                timestamp = latest_trades[symbol].timestamp
                logger.info(f"{symbol} latest price: ${price:.2f} at {timestamp}")
            else:
                logger.warning(f"No data retrieved for {symbol}")
                
        # Get historical bars
        end = datetime.now()
        start = end - timedelta(days=5)
        
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start.date(),
            end=end.date()
        )
        
        try:
            bars = data_client.get_stock_bars(bars_request)
            for symbol in symbols:
                if symbol in bars.data:
                    bar_count = len(bars.data[symbol])
                    logger.info(f"Retrieved {bar_count} bars for {symbol}")
                else:
                    logger.warning(f"No historical data retrieved for {symbol}")
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            
        return True
            
    except Exception as e:
        logger.error(f"Error testing Alpaca connection: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_alpaca_connection()
