import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sentiment_analysis import analyze_sentiment
from concurrent.futures import ThreadPoolExecutor
import talib
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockScanner:
    def __init__(self):
        self.sentiment_threshold = 0.6
        self.volume_threshold = 1000000  # Minimum daily volume
        self.price_min = 10  # Minimum price
        self.price_max = 500  # Maximum price
        self.risk_free_rate = 0.05  # Current risk-free rate (Treasury yield)
        
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols using Wikipedia"""
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = table[0]
            return df['Symbol'].tolist()
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {e}")
            return []

    def calculate_risk_metrics(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate risk metrics including Sharpe ratio, volatility, and max drawdown"""
        try:
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio
            excess_returns = returns - (self.risk_free_rate / 252)
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
            
            # Maximum Drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = cum_returns/rolling_max - 1
            max_drawdown = drawdowns.min()
            
            return sharpe_ratio, volatility, max_drawdown
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return 0, 1, -1

    def calculate_technical_score(self, data: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate technical analysis score (0-1) with detailed metrics"""
        try:
            if len(data) < 50:
                return 0, {}
            
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            # Core Technical Indicators
            rsi = talib.RSI(close, timeperiod=14)
            macd, signal, hist = talib.MACD(close)
            slowk, slowd = talib.STOCH(high, low, close)
            
            # Trend Indicators
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)
            ema_20 = talib.EMA(close, timeperiod=20)
            
            # Volatility Indicators
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # Volume and Momentum
            obv = talib.OBV(close, volume)
            mfi = talib.MFI(high, low, close, volume, timeperiod=14)
            adx = talib.ADX(high, low, close, timeperiod=14)
            aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
            
            # Risk Metrics
            sharpe_ratio, volatility, max_drawdown = self.calculate_risk_metrics(data)
            
            # Current values
            current_close = close[-1]
            current_rsi = rsi[-1]
            current_macd = macd[-1]
            current_signal = signal[-1]
            current_slowk = slowk[-1]
            current_slowd = slowd[-1]
            current_adx = adx[-1]
            current_mfi = mfi[-1]
            current_aroon_up = aroon_up[-1]
            current_aroon_down = aroon_down[-1]
            
            # Trend Analysis
            trend_strength = 0
            if current_close > sma_20[-1] > sma_50[-1] > sma_200[-1]:
                trend_strength = 1.0
            elif current_close > sma_20[-1] and current_close > sma_50[-1]:
                trend_strength = 0.75
            elif current_close > sma_20[-1]:
                trend_strength = 0.5
            
            # Support/Resistance Levels
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            s1 = 2 * pivot - high[-1]
            s2 = pivot - (high[-1] - low[-1])
            r1 = 2 * pivot - low[-1]
            r2 = pivot + (high[-1] - low[-1])
            
            # Score Components
            momentum_score = (
                (0.3 * (1 if 30 <= current_rsi <= 70 else 0)) +
                (0.3 * (1 if current_macd > current_signal else 0)) +
                (0.4 * (1 if current_slowk > current_slowd else 0))
            )
            
            trend_score = (
                (0.4 * trend_strength) +
                (0.3 * (current_adx / 100)) +
                (0.3 * (1 if current_aroon_up > current_aroon_down else 0))
            )
            
            volume_score = (
                (0.5 * (1 if volume[-1] > np.mean(volume[-20:]) else 0)) +
                (0.5 * (1 if current_mfi > 50 else 0))
            )
            
            volatility_score = (
                (0.4 * (1 if volatility < 0.3 else 0.5 if volatility < 0.5 else 0)) +
                (0.3 * (1 if max_drawdown > -0.2 else 0)) +
                (0.3 * (1 if sharpe_ratio > 1 else 0.5 if sharpe_ratio > 0 else 0))
            )
            
            # Risk-Adjusted Position Score
            risk_reward_ratio = abs((r2 - current_close) / (current_close - s2))
            position_score = (
                (0.3 * (1 if s1 < current_close < r1 else 0)) +
                (0.4 * (1 if risk_reward_ratio > 2 else 0.5 if risk_reward_ratio > 1 else 0)) +
                (0.3 * (1 if current_close > sma_200[-1] else 0))
            )
            
            # Final Technical Score
            technical_score = (
                momentum_score * 0.25 +
                trend_score * 0.25 +
                volume_score * 0.15 +
                volatility_score * 0.15 +
                position_score * 0.20
            )
            
            # Detailed metrics for analysis
            metrics = {
                'momentum': {
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'stoch_k': current_slowk,
                    'stoch_d': current_slowd
                },
                'trend': {
                    'adx': current_adx,
                    'aroon_up': current_aroon_up,
                    'aroon_down': current_aroon_down,
                    'trend_strength': trend_strength
                },
                'volume': {
                    'mfi': current_mfi,
                    'volume_change': volume[-1] / np.mean(volume[-20:])
                },
                'risk': {
                    'sharpe_ratio': sharpe_ratio,
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'risk_reward_ratio': risk_reward_ratio
                },
                'support_resistance': {
                    's2': s2,
                    's1': s1,
                    'pivot': pivot,
                    'r1': r1,
                    'r2': r2
                }
            }
            
            return technical_score, metrics
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0, {}

    def get_market_data(self, symbol: str, max_retries: int = 3) -> Tuple[pd.DataFrame, Tuple[float, Dict]]:
        """Get market data and calculate technical score with retry mechanism"""
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3mo")
                if len(data) < 50:
                    return None, (0, {})
                    
                current_price = data['Close'].iloc[-1]
                if not (self.price_min <= current_price <= self.price_max):
                    return None, (0, {})
                
                avg_volume = data['Volume'].tail(20).mean()
                if avg_volume < self.volume_threshold:
                    return None, (0, {})
                
                technical_score, metrics = self.calculate_technical_score(data)
                return data, (technical_score, metrics)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error getting market data for {symbol} after {max_retries} attempts: {e}")
                    return None, (0, {})
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}, retrying...")
                continue

    def get_sentiment_score(self, symbol: str, max_retries: int = 3) -> float:
        """Get sentiment score for a symbol with retry mechanism"""
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                if not news:
                    return 0.5  # Neutral if no news
                
                # Weight recent news more heavily
                scores = []
                weights = []
                
                for i, item in enumerate(news[:5]):
                    headline = item['title']
                    probability, sentiment = analyze_sentiment(headline)
                    
                    # Convert sentiment to score
                    if sentiment == "positive" and probability >= self.sentiment_threshold:
                        score = probability
                    elif sentiment == "negative" and probability >= self.sentiment_threshold:
                        score = 1 - probability
                    else:
                        score = 0.5
                    
                    # More recent news gets higher weight
                    weight = 1.0 / (i + 1)
                    scores.append(score)
                    weights.append(weight)
                
                # Calculate weighted average
                weighted_score = np.average(scores, weights=weights)
                return weighted_score
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error getting sentiment for {symbol} after {max_retries} attempts: {e}")
                    return 0.5
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}, retrying...")
                continue

    def scan_single_stock(self, symbol: str) -> Dict:
        """Analyze a single stock with detailed metrics"""
        try:
            # Get market data and technical score
            data, technical_score_tuple = self.get_market_data(symbol)
            if data is None:
                return None
            
            technical_score, metrics = technical_score_tuple
            
            # Get sentiment score
            sentiment_score = self.get_sentiment_score(symbol)
            
            # Calculate combined score with risk adjustment
            risk_adjustment = (1 + metrics['risk']['sharpe_ratio']) / 2  # Normalize to 0-1
            combined_score = (
                (technical_score * 0.6) + 
                (sentiment_score * 0.4)
            ) * risk_adjustment
            
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1]
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume': volume,
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'combined_score': combined_score,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None

    def scan_market(self, custom_symbols: List[str] = None) -> pd.DataFrame:
        """Scan the market for trading opportunities"""
        symbols = custom_symbols if custom_symbols else self.get_sp500_symbols()
        logger.info(f"Starting market scan for {len(symbols)} symbols...")
        
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.scan_single_stock, symbol) for symbol in symbols]
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        
        # Convert to DataFrame and sort by combined score
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('combined_score', ascending=False)
        
        logger.info(f"Scan completed. Found {len(df)} potential opportunities.")
        return df

    def get_top_opportunities(self, n: int = 10, min_score: float = 0.7) -> pd.DataFrame:
        """Get top N trading opportunities with detailed analysis"""
        df = self.scan_market()
        if df.empty:
            return pd.DataFrame()
        
        # Filter by minimum score and get top N
        opportunities = df[df['combined_score'] >= min_score].head(n)
        
        # Add recommendations and risk levels
        def get_recommendation(row):
            if row['combined_score'] >= 0.9:
                return 'Strong Buy - Low Risk'
            elif row['combined_score'] >= 0.8:
                if row['metrics']['risk']['sharpe_ratio'] > 1:
                    return 'Strong Buy - Moderate Risk'
                else:
                    return 'Buy - High Risk'
            elif row['combined_score'] >= 0.7:
                return 'Buy - Monitor Risk'
            else:
                return 'Hold'
        
        opportunities['recommendation'] = opportunities.apply(get_recommendation, axis=1)
        
        # Add key metrics
        opportunities['risk_reward'] = opportunities.apply(
            lambda x: f"{x['metrics']['risk']['risk_reward_ratio']:.2f}", axis=1)
        opportunities['sharpe_ratio'] = opportunities.apply(
            lambda x: f"{x['metrics']['risk']['sharpe_ratio']:.2f}", axis=1)
        opportunities['volatility'] = opportunities.apply(
            lambda x: f"{x['metrics']['risk']['volatility']*100:.1f}%", axis=1)
        
        return opportunities[['symbol', 'price', 'volume', 'technical_score', 
                            'sentiment_score', 'combined_score', 'recommendation',
                            'risk_reward', 'sharpe_ratio', 'volatility']]

def main():
    scanner = StockScanner()
    
    # Get top opportunities
    opportunities = scanner.get_top_opportunities(n=10, min_score=0.7)
    
    if opportunities.empty:
        logger.info("No trading opportunities found matching criteria.")
    else:
        logger.info("\nTop Trading Opportunities:")
        print(opportunities.to_string(index=False))

if __name__ == "__main__":
    main()
