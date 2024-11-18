from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
import re

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
        print(f"Error in sentiment analysis: {str(e)}")
        return 0, "neutral"
