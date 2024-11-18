import unittest
from stock_scanner import StockScanner
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class TestStockScanner(unittest.TestCase):
    def setUp(self):
        self.scanner = StockScanner()
        # Test with a small set of well-known stocks
        self.test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'META']

    def test_market_data(self):
        """Test market data retrieval"""
        for symbol in self.test_symbols:
            data, score = self.scanner.get_market_data(symbol)
            self.assertIsNotNone(data)
            self.assertIsInstance(score, float)
            self.assertTrue(0 <= score <= 1)

    def test_sentiment_score(self):
        """Test sentiment analysis"""
        for symbol in self.test_symbols:
            score = self.scanner.get_sentiment_score(symbol)
            self.assertIsInstance(score, float)
            self.assertTrue(0 <= score <= 1)

    def test_technical_analysis(self):
        """Test technical analysis calculations"""
        for symbol in self.test_symbols:
            data, _ = self.scanner.get_market_data(symbol)
            if data is not None:
                score = self.scanner.calculate_technical_score(data)
                self.assertIsInstance(score, float)
                self.assertTrue(0 <= score <= 1)

    def test_scan_market(self):
        """Test market scanning functionality"""
        results = self.scanner.scan_market(self.test_symbols)
        self.assertIsInstance(results, pd.DataFrame)
        if not results.empty:
            self.assertTrue(all(col in results.columns for col in [
                'symbol', 'price', 'volume', 'technical_score', 
                'sentiment_score', 'combined_score'
            ]))

    def test_get_top_opportunities(self):
        """Test getting top opportunities"""
        opportunities = self.scanner.get_top_opportunities(n=3, min_score=0.5)
        self.assertIsInstance(opportunities, pd.DataFrame)
        if not opportunities.empty:
            self.assertTrue(all(col in opportunities.columns for col in [
                'symbol', 'price', 'volume', 'technical_score',
                'sentiment_score', 'combined_score', 'recommendation'
            ]))
            self.assertLessEqual(len(opportunities), 3)

if __name__ == '__main__':
    unittest.main()
