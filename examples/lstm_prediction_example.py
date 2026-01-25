"""
Example: Using LSTM price prediction with the trading bot.

This example demonstrates:
1. Training an LSTM model on historical data
2. Making predictions for trading signals
3. Integrating predictions with strategy decisions

Requirements:
    pip install torch scikit-learn

Usage:
    python examples/lstm_prediction_example.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set testing environment to avoid credential validation
os.environ["TESTING"] = "1"

import numpy as np

from config import ML_PARAMS


def generate_synthetic_prices(n_bars: int = 500, base_price: float = 100.0) -> list:
    """
    Generate synthetic OHLCV data for demonstration.

    In production, you would fetch this from Alpaca or another data source.
    """
    np.random.seed(42)
    prices = []
    price = base_price

    for i in range(n_bars):
        # Add some trend and mean reversion
        trend = 0.001 * np.sin(i / 50)  # Cyclical trend
        noise = np.random.randn() * 0.005
        returns = trend + noise

        price = price * (1 + returns)

        # Generate OHLCV
        open_price = price * (1 + np.random.randn() * 0.001)
        close_price = price
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.002))
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.002))
        volume = 1000000 + np.random.randint(-100000, 100000)

        prices.append({
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        })

    return prices


def main():
    """Main example demonstrating LSTM prediction."""

    print("=" * 60)
    print("LSTM Price Prediction Example")
    print("=" * 60)

    # Check if PyTorch is available
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS (Apple Silicon) available")
        else:
            print("Using CPU (no GPU acceleration)")
    except ImportError:
        print("\nError: PyTorch is not installed.")
        print("Install with: pip install torch")
        return

    # Import LSTM predictor (after torch check)
    from ml.lstm_predictor import LSTMPredictor

    # Configuration from ML_PARAMS
    print(f"\nConfiguration:")
    print(f"  Sequence Length: {ML_PARAMS['SEQUENCE_LENGTH']} bars")
    print(f"  Prediction Horizon: {ML_PARAMS['PREDICTION_HORIZON']} bars")
    print(f"  Hidden Size: {ML_PARAMS['HIDDEN_SIZE']}")
    print(f"  Minimum Confidence: {ML_PARAMS['MIN_CONFIDENCE']}")

    # Generate synthetic data
    print("\n" + "-" * 60)
    print("Step 1: Generating synthetic price data...")
    prices = generate_synthetic_prices(n_bars=500)
    print(f"  Generated {len(prices)} bars of OHLCV data")
    print(f"  Price range: ${min(p['close'] for p in prices):.2f} - ${max(p['close'] for p in prices):.2f}")

    # Create predictor
    print("\n" + "-" * 60)
    print("Step 2: Creating LSTM predictor...")
    predictor = LSTMPredictor(
        sequence_length=ML_PARAMS['SEQUENCE_LENGTH'],
        prediction_horizon=ML_PARAMS['PREDICTION_HORIZON'],
        hidden_size=ML_PARAMS['HIDDEN_SIZE'],
        num_layers=ML_PARAMS['NUM_LAYERS'],
        use_gpu=ML_PARAMS['USE_GPU'],
        model_dir="models",
    )
    print(f"  Device: {predictor.device}")

    # Train the model
    print("\n" + "-" * 60)
    print("Step 3: Training LSTM model...")
    metrics = predictor.train(
        symbol="SYNTHETIC",
        prices=prices,
        epochs=ML_PARAMS['EPOCHS'],
        batch_size=ML_PARAMS['BATCH_SIZE'],
        learning_rate=ML_PARAMS['LEARNING_RATE'],
        early_stopping_patience=ML_PARAMS['EARLY_STOPPING_PATIENCE'],
    )

    print(f"\nTraining Results:")
    print(f"  Epochs completed: {metrics.epochs}")
    print(f"  Final Train Loss: {metrics.final_train_loss:.6f}")
    print(f"  Final Val Loss: {metrics.final_val_loss:.6f}")
    print(f"  Best Val Loss: {metrics.best_val_loss:.6f}")
    print(f"  Training Time: {metrics.training_time_seconds:.1f}s")
    print(f"  Model saved to: {metrics.model_path}")

    # Make predictions
    print("\n" + "-" * 60)
    print("Step 4: Making predictions...")

    # Use the most recent data for prediction
    recent_prices = prices[-100:]  # Last 100 bars

    result = predictor.predict("SYNTHETIC", recent_prices)

    if result:
        print(f"\nPrediction Results:")
        print(f"  Symbol: {result.symbol}")
        print(f"  Current Price: ${result.current_price:.2f}")
        print(f"  Predicted Price: ${result.predicted_price:.2f}")
        print(f"  Direction: {result.predicted_direction.upper()}")
        print(f"  Price Change: {result.price_change_pct:+.2f}%")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Horizon: {result.horizon} bars")

        # Trading decision based on prediction
        print("\n" + "-" * 60)
        print("Step 5: Trading decision...")

        min_confidence = ML_PARAMS['MIN_CONFIDENCE']

        if result.confidence >= min_confidence:
            if result.predicted_direction == "up":
                print(f"  SIGNAL: BUY (confidence {result.confidence:.2f} >= {min_confidence})")
            elif result.predicted_direction == "down":
                print(f"  SIGNAL: SELL (confidence {result.confidence:.2f} >= {min_confidence})")
            else:
                print(f"  SIGNAL: HOLD (neutral prediction)")
        else:
            print(f"  SIGNAL: NO ACTION (confidence {result.confidence:.2f} < {min_confidence})")
    else:
        print("  Prediction failed!")

    # Demonstrate model persistence
    print("\n" + "-" * 60)
    print("Step 6: Model persistence...")

    # Save model
    saved = predictor.save_model("SYNTHETIC")
    print(f"  Model saved: {saved}")

    # Load model in new predictor
    new_predictor = LSTMPredictor(
        sequence_length=ML_PARAMS['SEQUENCE_LENGTH'],
        prediction_horizon=ML_PARAMS['PREDICTION_HORIZON'],
        hidden_size=ML_PARAMS['HIDDEN_SIZE'],
        num_layers=ML_PARAMS['NUM_LAYERS'],
        use_gpu=ML_PARAMS['USE_GPU'],
        model_dir="models",
    )
    loaded = new_predictor.load_model("SYNTHETIC")
    print(f"  Model loaded: {loaded}")

    # Verify loaded model can predict
    if loaded:
        result2 = new_predictor.predict("SYNTHETIC", recent_prices)
        print(f"  Loaded model prediction: {result2.predicted_direction if result2 else 'failed'}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
