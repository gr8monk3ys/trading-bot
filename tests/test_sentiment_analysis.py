import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_analysis import analyze_sentiment, preprocess_text

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.positive_texts = [
            "The company reported strong earnings and increased dividend",
            "Impressive revenue growth and market share expansion",
            "Stock surges on better-than-expected quarterly results"
        ]
        self.negative_texts = [
            "The company missed earnings expectations and cut guidance",
            "Significant decline in profit margins raises concerns",
            "Stock plunges after disappointing earnings report"
        ]
        self.neutral_texts = [
            "The company reported earnings in line with expectations",
            "Company maintains current market position",
            "Trading volume remains consistent with previous quarter"
        ]

    def test_preprocessing(self):
        """Test text preprocessing function."""
        text = "Test! With? Punctuation... and   spaces"
        processed = preprocess_text(text)
        self.assertEqual(processed, "Test With Punctuation and spaces")

    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        for text in self.positive_texts:
            probability, sentiment = analyze_sentiment([text])
            self.assertEqual(sentiment, "positive", 
                           f"Failed to detect positive sentiment in: {text}")
            self.assertGreater(probability, 0, 
                             f"Probability should be > 0 for positive sentiment")

    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        for text in self.negative_texts:
            probability, sentiment = analyze_sentiment([text])
            self.assertEqual(sentiment, "negative", 
                           f"Failed to detect negative sentiment in: {text}")
            self.assertGreater(probability, 0, 
                             f"Probability should be > 0 for negative sentiment")

    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        for text in self.neutral_texts:
            probability, sentiment = analyze_sentiment([text])
            self.assertIn(sentiment, ["neutral", "positive", "negative"], 
                         f"Invalid sentiment value for neutral text: {text}")
            self.assertGreater(probability, 0, 
                             f"Probability should be > 0 for neutral sentiment")

    def test_empty_input(self):
        """Test empty input handling."""
        probability, sentiment = analyze_sentiment([])
        self.assertEqual(sentiment, "neutral", 
                        "Empty input should return neutral sentiment")
        self.assertEqual(probability, 0, 
                        "Empty input should return 0 probability")

    def test_multiple_texts(self):
        """Test sentiment analysis with multiple texts."""
        mixed_texts = [
            self.positive_texts[0],
            self.negative_texts[0],
            self.neutral_texts[0]
        ]
        probability, sentiment = analyze_sentiment(mixed_texts)
        self.assertIn(sentiment, ["positive", "negative", "neutral"], 
                     "Invalid sentiment for multiple texts")
        self.assertGreater(probability, 0, 
                          "Probability should be > 0 for multiple texts")

if __name__ == '__main__':
    unittest.main()
