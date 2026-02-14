"""
LSTM-based price prediction model for trading signals.

This module provides an LSTM neural network for predicting future price
movements based on historical OHLCV data. It uses lazy imports for heavy
dependencies (PyTorch, sklearn) to minimize startup time.

Features:
- Sequence-based price prediction with configurable lookback
- GPU acceleration support
- Model persistence (save/load)
- Confidence-based predictions

Usage:
    from ml.lstm_predictor import LSTMPredictor

    predictor = LSTMPredictor(sequence_length=60, prediction_horizon=5)

    # Train on historical data
    result = predictor.train("AAPL", historical_prices, epochs=50)

    # Make predictions
    prediction = predictor.predict("AAPL", recent_prices)
    if prediction and prediction.confidence > 0.6:
        print(f"Predicted direction: {prediction.predicted_direction}")
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.torch_utils import get_torch_device, import_torch

logger = logging.getLogger(__name__)

# Lazy import for sklearn
_MinMaxScaler = None


def _import_sklearn():
    """Lazy import sklearn to reduce startup time."""
    global _MinMaxScaler
    if _MinMaxScaler is None:
        try:
            from sklearn.preprocessing import MinMaxScaler

            _MinMaxScaler = MinMaxScaler
            logger.debug("sklearn imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import sklearn: {e}")
            raise ImportError(
                "scikit-learn is required for LSTM predictions. "
                "Install with: pip install scikit-learn"
            ) from e
    return _MinMaxScaler


@dataclass
class PredictionResult:
    """Result of an LSTM price prediction."""

    symbol: str
    predicted_price: float
    predicted_direction: str  # 'up', 'down', 'neutral'
    confidence: float  # 0.0 to 1.0
    horizon: int  # Prediction horizon in bars
    timestamp: datetime
    current_price: float = 0.0
    price_change_pct: float = 0.0

    def __post_init__(self):
        """Calculate price change percentage after initialization."""
        if self.current_price > 0:
            self.price_change_pct = (
                (self.predicted_price - self.current_price) / self.current_price
            ) * 100


@dataclass
class MCDropoutResult:
    """
    Result of Monte Carlo Dropout uncertainty estimation.

    MC Dropout provides Bayesian uncertainty estimates by keeping
    dropout enabled during inference and running multiple forward passes.
    """

    symbol: str
    mean_prediction: float  # Mean of all forward passes
    std_prediction: float  # Std deviation (uncertainty measure)
    lower_bound: float  # 95% confidence interval lower
    upper_bound: float  # 95% confidence interval upper
    confidence: float  # Calibrated confidence (0-1)
    n_samples: int  # Number of forward passes
    predicted_direction: str
    direction_probability: float  # P(direction is correct)
    timestamp: datetime
    current_price: float = 0.0

    @property
    def coefficient_of_variation(self) -> float:
        """CV = std/mean - normalized uncertainty measure."""
        if abs(self.mean_prediction) < 1e-8:
            return float("inf")
        return abs(self.std_prediction / self.mean_prediction)


@dataclass
class TrainingMetrics:
    """Metrics from model training."""

    symbol: str
    epochs: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    training_time_seconds: float
    model_path: str
    samples_used: int


class LSTMNetwork:
    """
    PyTorch LSTM network for price prediction.

    Architecture:
    - Input: (batch_size, sequence_length, num_features)
    - LSTM layers with dropout
    - Fully connected output layer
    - Output: (batch_size, 1) - predicted normalized price
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        """
        Initialize the LSTM network.

        Args:
            input_size: Number of input features (default 5 for OHLCV)
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of stacked LSTM layers
            output_size: Size of output (1 for single price prediction)
            dropout: Dropout rate between LSTM layers
        """
        torch, nn, _, _ = import_torch()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # Build the model
        self.model = self._build_model()

    def _build_model(self):
        """Build and return the PyTorch model."""
        torch, nn, _, _ = import_torch()

        class _LSTMModel(nn.Module):
            """Internal LSTM model class."""

            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                # LSTM layers
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )

                # Dropout for regularization
                self.dropout = nn.Dropout(dropout)

                # Fully connected output layer
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                """Forward pass through the network."""
                # LSTM forward pass
                # lstm_out shape: (batch_size, seq_length, hidden_size)
                lstm_out, (h_n, c_n) = self.lstm(x)

                # Take the output from the last timestep
                last_output = lstm_out[:, -1, :]

                # Apply dropout
                dropped = self.dropout(last_output)

                # Fully connected layer for prediction
                output = self.fc(dropped)

                return output

        return _LSTMModel(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.output_size,
            self.dropout,
        )


class LSTMPredictor:
    """
    LSTM-based price predictor for trading signals.

    This class handles data preparation, model training, and prediction
    for stock price movements using LSTM neural networks.

    Features used:
    - Open, High, Low, Close, Volume (OHLCV)
    - All features are normalized using MinMaxScaler

    Example:
        predictor = LSTMPredictor(sequence_length=60)

        # Train the model
        metrics = predictor.train("AAPL", price_history, epochs=50)

        # Make predictions
        result = predictor.predict("AAPL", recent_prices)
        if result.confidence > 0.6:
            print(f"Direction: {result.predicted_direction}")
    """

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_gpu: bool = False,
        model_dir: str = "models",
    ):
        """
        Initialize the LSTM predictor.

        Args:
            sequence_length: Number of historical bars for input sequence
            prediction_horizon: Number of bars ahead to predict
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            use_gpu: Whether to use GPU acceleration if available
            model_dir: Directory to save/load trained models
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_dir = model_dir

        # Determine device (GPU or CPU)
        self.device = get_torch_device(use_gpu)

        # Storage for trained models and scalers
        self.models: Dict[str, any] = {}  # symbol -> trained model
        self.scalers: Dict[str, any] = {}  # symbol -> fitted scaler
        self.feature_info: Dict[str, dict] = {}  # symbol -> feature metadata

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def _prepare_features(self, prices: List[dict]) -> np.ndarray:
        """
        Prepare feature matrix from price data.

        Args:
            prices: List of OHLCV dicts with keys: open, high, low, close, volume

        Returns:
            Feature matrix of shape (num_samples, num_features)
        """
        if len(prices) < 2:
            logger.warning("Insufficient price data for feature preparation")
            return np.array([])

        features = []
        for bar in prices:
            # Extract OHLCV features
            row = [
                float(bar.get("open", bar.get("o", 0))),
                float(bar.get("high", bar.get("h", 0))),
                float(bar.get("low", bar.get("l", 0))),
                float(bar.get("close", bar.get("c", 0))),
                float(bar.get("volume", bar.get("v", 0))),
            ]
            features.append(row)

        return np.array(features, dtype=np.float32)

    def _normalize_features(
        self, features: np.ndarray, symbol: str, fit: bool = False
    ) -> np.ndarray:
        """
        Normalize features using MinMaxScaler.

        Args:
            features: Raw feature matrix
            symbol: Stock symbol (used for scaler storage)
            fit: If True, fit a new scaler; if False, use existing

        Returns:
            Normalized feature matrix
        """
        MinMaxScaler = _import_sklearn()

        if fit or symbol not in self.scalers:
            # Create and fit a new scaler
            self.scalers[symbol] = MinMaxScaler(feature_range=(0, 1))
            normalized = self.scalers[symbol].fit_transform(features)
            logger.debug(f"Fitted new scaler for {symbol}")
        else:
            # Use existing scaler
            normalized = self.scalers[symbol].transform(features)

        return normalized

    def _create_sequences(
        self, features: np.ndarray, targets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create input sequences for LSTM training/prediction.

        Args:
            features: Normalized feature matrix
            targets: Optional target values for training

        Returns:
            Tuple of (X sequences, y targets)
        """
        X = []
        y = [] if targets is not None else None

        # Calculate number of valid sequences
        num_sequences = len(features) - self.sequence_length - self.prediction_horizon + 1

        if num_sequences <= 0:
            logger.warning(
                f"Insufficient data for sequences: {len(features)} samples, "
                f"need {self.sequence_length + self.prediction_horizon}"
            )
            return np.array([]), None

        for i in range(num_sequences):
            # Input sequence
            X.append(features[i : i + self.sequence_length])

            # Target value (future close price)
            if targets is not None:
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                y.append(targets[target_idx])

        X_array = np.array(X)
        y_array = np.array(y) if y is not None else None

        return X_array, y_array

    def train(
        self,
        symbol: str,
        prices: List[dict],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
    ) -> TrainingMetrics:
        """
        Train LSTM model for a specific symbol.

        Args:
            symbol: Stock symbol to train for
            prices: Historical OHLCV data (list of dicts)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Adam optimizer learning rate
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs to wait before early stopping

        Returns:
            TrainingMetrics with training results
        """
        import time

        torch, nn, _, _ = import_torch()
        start_time = time.time()

        # Validate input data
        min_required = self.sequence_length + self.prediction_horizon + 20
        if len(prices) < min_required:
            logger.warning(
                f"Insufficient data for {symbol}: {len(prices)} bars, "
                f"need at least {min_required}"
            )
            return TrainingMetrics(
                symbol=symbol,
                epochs=0,
                final_train_loss=float("inf"),
                final_val_loss=float("inf"),
                best_val_loss=float("inf"),
                training_time_seconds=0,
                model_path="",
                samples_used=len(prices),
            )

        # Prepare features
        features = self._prepare_features(prices)
        if len(features) == 0:
            raise ValueError(f"Failed to prepare features for {symbol}")

        # CRITICAL FIX: Split data BEFORE normalization to prevent data leakage
        # The scaler must only see training data, not validation/future data
        # This is a time-series split (not random) to respect temporal order

        # Calculate split point for raw features
        # We need to account for sequence_length and prediction_horizon when splitting
        total_sequences = len(features) - self.sequence_length - self.prediction_horizon + 1
        train_sequences = int(total_sequences * (1 - validation_split))

        # Calculate the raw feature split index
        # train_sequences correspond to features[0:train_split_idx] for the last training sequence
        train_split_idx = train_sequences + self.sequence_length + self.prediction_horizon - 1

        # Split raw features BEFORE normalization
        train_features = features[:train_split_idx]
        val_features = features[
            train_split_idx - self.sequence_length - self.prediction_horizon + 1 :
        ]

        logger.info(
            f"Data leakage prevention: Fitting scaler on {len(train_features)} training samples only "
            f"(validation: {len(val_features)} samples)"
        )

        # Fit scaler ONLY on training data (prevents data leakage)
        MinMaxScaler = _import_sklearn()
        self.scalers[symbol] = MinMaxScaler(feature_range=(0, 1))
        normalized_train = self.scalers[symbol].fit_transform(train_features)

        # Transform validation data using training-fitted scaler
        normalized_val = self.scalers[symbol].transform(val_features)

        # Target is the normalized close price (column index 3)
        train_targets = normalized_train[:, 3]
        val_targets = normalized_val[:, 3]

        # Create sequences for training data
        X_train, y_train = self._create_sequences(normalized_train, train_targets)
        if len(X_train) == 0:
            raise ValueError(f"Could not create training sequences for {symbol}")

        # Create sequences for validation data
        X_val, y_val = self._create_sequences(normalized_val, val_targets)
        if len(X_val) == 0:
            logger.warning(
                f"No validation sequences created for {symbol}, using last 20% of training"
            )
            # Fallback: use last portion of training as validation
            fallback_split = int(len(X_train) * 0.8)
            X_val = X_train[fallback_split:]
            y_val = y_train[fallback_split:]
            X_train = X_train[:fallback_split]
            y_train = y_train[:fallback_split]

        logger.info(
            f"Training {symbol}: {len(X_train)} train samples, "
            f"{len(X_val)} validation samples (scaler fitted on training data only)"
        )

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Create model
        network = LSTMNetwork(
            input_size=features.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        model = network.model.to(self.device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()

            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                batch_X = X_train_t[batch_idx]
                batch_y = y_train_t[batch_idx]

                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} " f"(best val loss: {best_val_loss:.6f})"
                )
                break

        # Restore best model if early stopping occurred
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save model
        self.models[symbol] = model
        model_path = os.path.join(self.model_dir, f"{symbol}_lstm.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_data_min": self.scalers[symbol].data_min_,
                "scaler_data_max": self.scalers[symbol].data_max_,
                "scaler_data_range": self.scalers[symbol].data_range_,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "sequence_length": self.sequence_length,
                "prediction_horizon": self.prediction_horizon,
                "input_size": features.shape[1],
            },
            model_path,
        )

        training_time = time.time() - start_time

        logger.info(
            f"Training complete for {symbol}: "
            f"Best Val Loss: {best_val_loss:.6f}, "
            f"Time: {training_time:.1f}s"
        )

        return TrainingMetrics(
            symbol=symbol,
            epochs=len(train_losses),
            final_train_loss=train_losses[-1],
            final_val_loss=val_losses[-1],
            best_val_loss=best_val_loss,
            training_time_seconds=training_time,
            model_path=model_path,
            samples_used=len(X_train) + len(X_val),
        )

    def predict(self, symbol: str, prices: List[dict]) -> Optional[PredictionResult]:
        """
        Predict future price direction for a symbol.

        Args:
            symbol: Stock symbol
            prices: Recent OHLCV data (at least sequence_length bars)

        Returns:
            PredictionResult or None if prediction fails
        """
        torch, _, _, _ = import_torch()

        # Check if model exists
        if symbol not in self.models:
            logger.warning(f"No trained model for {symbol}. Call train() first.")
            return None

        # Validate input data
        if len(prices) < self.sequence_length:
            logger.warning(
                f"Insufficient data for prediction: {len(prices)} bars, "
                f"need {self.sequence_length}"
            )
            return None

        # Check if scaler exists
        if symbol not in self.scalers:
            logger.warning(f"No scaler for {symbol}. Model may not be properly loaded.")
            return None

        try:
            # Prepare features from recent prices
            recent_prices = prices[-self.sequence_length :]
            features = self._prepare_features(recent_prices)

            if len(features) == 0:
                logger.error(f"Failed to prepare features for {symbol}")
                return None

            # Normalize using existing scaler
            normalized = self.scalers[symbol].transform(features)

            # Create input tensor
            X = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)

            # Get model and make prediction
            model = self.models[symbol]
            model.eval()

            with torch.no_grad():
                predicted_normalized = model(X).item()

            # Denormalize the prediction
            # Close price is at index 3
            scaler = self.scalers[symbol]
            close_min = scaler.data_min_[3]
            close_max = scaler.data_max_[3]
            predicted_price = predicted_normalized * (close_max - close_min) + close_min

            # Get current price for comparison
            current_close = float(prices[-1].get("close", prices[-1].get("c", 0)))

            if current_close <= 0:
                logger.error(f"Invalid current price for {symbol}")
                return None

            # Calculate price change
            price_change_pct = (predicted_price - current_close) / current_close

            # Determine direction and confidence
            # Confidence is based on the magnitude of predicted change
            # with some noise filtering
            if price_change_pct > 0.01:  # > 1% predicted increase
                direction = "up"
                confidence = min(abs(price_change_pct) * 5, 1.0)  # Scale to 0-1
            elif price_change_pct < -0.01:  # > 1% predicted decrease
                direction = "down"
                confidence = min(abs(price_change_pct) * 5, 1.0)
            else:
                direction = "neutral"
                confidence = 0.5 - abs(price_change_pct) * 10  # Lower confidence for small moves

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            return PredictionResult(
                symbol=symbol,
                predicted_price=predicted_price,
                predicted_direction=direction,
                confidence=confidence,
                horizon=self.prediction_horizon,
                timestamp=datetime.now(),
                current_price=current_close,
                price_change_pct=price_change_pct * 100,
            )

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}", exc_info=True)
            return None

    def load_model(self, symbol: str) -> bool:
        """
        Load a previously saved model from disk.

        Args:
            symbol: Stock symbol

        Returns:
            True if model loaded successfully, False otherwise
        """
        torch, _, _, _ = import_torch()
        MinMaxScaler = _import_sklearn()

        model_path = os.path.join(self.model_dir, f"{symbol}_lstm.pt")

        if not os.path.exists(model_path):
            logger.warning(f"No saved model found for {symbol} at {model_path}")
            return False

        try:
            # Load checkpoint
            # Note: weights_only=False needed because checkpoint includes
            # non-tensor metadata (scaler, config, metrics).
            # Only load model files from trusted sources.
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Recreate model architecture
            network = LSTMNetwork(
                input_size=checkpoint.get("input_size", 5),
                hidden_size=checkpoint.get("hidden_size", self.hidden_size),
                num_layers=checkpoint.get("num_layers", self.num_layers),
            )
            model = network.model.to(self.device)

            # Load model weights
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.models[symbol] = model

            # Recreate scaler
            scaler = MinMaxScaler()
            scaler.data_min_ = checkpoint["scaler_data_min"]
            scaler.data_max_ = checkpoint["scaler_data_max"]
            scaler.data_range_ = checkpoint["scaler_data_range"]
            scaler.scale_ = 1.0 / scaler.data_range_
            scaler.min_ = -scaler.data_min_ * scaler.scale_
            scaler.n_features_in_ = len(scaler.data_min_)
            scaler.feature_range = (0, 1)

            self.scalers[symbol] = scaler

            # Update predictor settings if they differ
            if "sequence_length" in checkpoint:
                saved_seq_len = checkpoint["sequence_length"]
                if saved_seq_len != self.sequence_length:
                    logger.warning(
                        f"Loaded model has sequence_length={saved_seq_len}, "
                        f"but predictor has {self.sequence_length}. Using loaded value."
                    )
                    self.sequence_length = saved_seq_len

            if "prediction_horizon" in checkpoint:
                saved_horizon = checkpoint["prediction_horizon"]
                if saved_horizon != self.prediction_horizon:
                    logger.warning(
                        f"Loaded model has prediction_horizon={saved_horizon}, "
                        f"but predictor has {self.prediction_horizon}. Using loaded value."
                    )
                    self.prediction_horizon = saved_horizon

            logger.info(f"Successfully loaded model for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}", exc_info=True)
            return False

    def save_model(self, symbol: str) -> bool:
        """
        Save a trained model to disk.

        Args:
            symbol: Stock symbol

        Returns:
            True if saved successfully, False otherwise
        """
        torch, _, _, _ = import_torch()

        if symbol not in self.models:
            logger.warning(f"No model to save for {symbol}")
            return False

        if symbol not in self.scalers:
            logger.warning(f"No scaler to save for {symbol}")
            return False

        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_lstm.pt")
            model = self.models[symbol]
            scaler = self.scalers[symbol]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_data_min": scaler.data_min_,
                    "scaler_data_max": scaler.data_max_,
                    "scaler_data_range": scaler.data_range_,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "sequence_length": self.sequence_length,
                    "prediction_horizon": self.prediction_horizon,
                    "input_size": scaler.n_features_in_,
                },
                model_path,
            )

            logger.info(f"Saved model for {symbol} to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}", exc_info=True)
            return False

    def has_model(self, symbol: str) -> bool:
        """Check if a trained model exists for a symbol."""
        return symbol in self.models

    def get_model_info(self, symbol: str) -> Optional[dict]:
        """Get information about a trained model."""
        if symbol not in self.models:
            return None

        return {
            "symbol": symbol,
            "has_model": True,
            "has_scaler": symbol in self.scalers,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "device": str(self.device),
        }

    def list_trained_models(self) -> List[str]:
        """Return list of symbols with trained models."""
        return list(self.models.keys())

    def clear_model(self, symbol: str) -> bool:
        """Remove a model from memory."""
        removed = False
        if symbol in self.models:
            del self.models[symbol]
            removed = True
        if symbol in self.scalers:
            del self.scalers[symbol]
        return removed

    def predict_with_uncertainty(
        self,
        symbol: str,
        prices: List[dict],
        n_samples: int = 100,
        confidence_level: float = 0.95,
    ) -> Optional[MCDropoutResult]:
        """
        Make prediction with uncertainty estimation using MC Dropout.

        Monte Carlo Dropout keeps dropout enabled during inference and
        runs multiple forward passes to estimate epistemic uncertainty
        (uncertainty due to model/parameter uncertainty).

        This is CRITICAL for institutional-grade trading:
        - Know when model is uncertain
        - Size positions based on confidence
        - Avoid trading when model is outside its training distribution

        Args:
            symbol: Stock symbol
            prices: Recent OHLCV data
            n_samples: Number of forward passes (more = better estimate)
            confidence_level: Confidence interval level (default 95%)

        Returns:
            MCDropoutResult with uncertainty estimates, or None if fails
        """
        torch, _, _, _ = import_torch()

        # Validate inputs
        if symbol not in self.models:
            logger.warning(f"No trained model for {symbol}")
            return None

        if len(prices) < self.sequence_length:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        if symbol not in self.scalers:
            logger.warning(f"No scaler for {symbol}")
            return None

        try:
            # Prepare features
            recent_prices = prices[-self.sequence_length :]
            features = self._prepare_features(recent_prices)

            if len(features) == 0:
                return None

            # Normalize using existing scaler
            normalized = self.scalers[symbol].transform(features)

            # Create input tensor
            X = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)

            # Get model
            model = self.models[symbol]

            # CRITICAL: Keep model in TRAINING mode to enable dropout
            # This is what makes it "Monte Carlo" dropout
            model.train()

            # Run multiple forward passes
            predictions = []
            for _ in range(n_samples):
                with torch.no_grad():
                    pred = model(X).item()
                    predictions.append(pred)

            # Return to eval mode for normal predictions
            model.eval()

            predictions = np.array(predictions)

            # Calculate statistics
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            # Confidence interval
            1 - confidence_level
            z_score = 1.96  # For 95% CI
            if confidence_level == 0.99:
                z_score = 2.576
            elif confidence_level == 0.90:
                z_score = 1.645

            lower = mean_pred - z_score * std_pred
            upper = mean_pred + z_score * std_pred

            # Denormalize predictions
            scaler = self.scalers[symbol]
            close_min = scaler.data_min_[3]
            close_max = scaler.data_max_[3]
            close_range = close_max - close_min

            mean_price = mean_pred * close_range + close_min
            std_price = std_pred * close_range  # Scale std
            lower_price = lower * close_range + close_min
            upper_price = upper * close_range + close_min

            # Get current price
            current_close = float(prices[-1].get("close", prices[-1].get("c", 0)))

            if current_close <= 0:
                return None

            # Calculate direction and probability
            price_change = mean_price - current_close
            direction = "up" if price_change > 0 else "down" if price_change < 0 else "neutral"

            # Direction probability: what fraction of samples agree?
            if direction == "up":
                agreeing = np.sum((predictions * close_range + close_min) > current_close)
            elif direction == "down":
                agreeing = np.sum((predictions * close_range + close_min) < current_close)
            else:
                agreeing = n_samples // 2

            direction_prob = agreeing / n_samples

            # Calibrated confidence
            # High confidence = low uncertainty + high direction agreement
            cv = abs(std_price / mean_price) if abs(mean_price) > 1e-8 else float("inf")
            uncertainty_penalty = min(1.0, cv * 10)  # Penalize high CV
            confidence = direction_prob * (1 - uncertainty_penalty * 0.5)
            confidence = max(0.0, min(1.0, confidence))

            return MCDropoutResult(
                symbol=symbol,
                mean_prediction=mean_price,
                std_prediction=std_price,
                lower_bound=lower_price,
                upper_bound=upper_price,
                confidence=confidence,
                n_samples=n_samples,
                predicted_direction=direction,
                direction_probability=direction_prob,
                timestamp=datetime.now(),
                current_price=current_close,
            )

        except Exception as e:
            logger.error(f"MC Dropout prediction failed for {symbol}: {e}", exc_info=True)
            return None

    def get_prediction_calibration(
        self,
        symbol: str,
        price_history: List[dict],
        n_test_points: int = 50,
    ) -> Optional[Dict]:
        """
        Evaluate calibration of MC Dropout confidence estimates.

        A well-calibrated model should have:
        - 95% of true values within 95% confidence interval
        - Confidence correlated with actual accuracy

        Args:
            symbol: Stock symbol
            price_history: Historical prices for evaluation
            n_test_points: Number of test predictions to make

        Returns:
            Calibration metrics dictionary
        """
        if len(price_history) < self.sequence_length + self.prediction_horizon + n_test_points:
            logger.warning("Insufficient data for calibration analysis")
            return None

        in_interval_count = 0
        direction_correct_count = 0
        confidence_accuracy_pairs = []

        for i in range(n_test_points):
            # Get data for this test point
            start_idx = i
            end_idx = start_idx + self.sequence_length

            if end_idx + self.prediction_horizon >= len(price_history):
                break

            test_prices = price_history[start_idx:end_idx]

            # Make prediction with uncertainty
            result = self.predict_with_uncertainty(symbol, test_prices)

            if result is None:
                continue

            # Get actual future price
            actual_idx = end_idx + self.prediction_horizon - 1
            actual_price = float(
                price_history[actual_idx].get("close", price_history[actual_idx].get("c", 0))
            )

            # Check if actual is within confidence interval
            if result.lower_bound <= actual_price <= result.upper_bound:
                in_interval_count += 1

            # Check direction accuracy
            actual_direction = "up" if actual_price > result.current_price else "down"
            if result.predicted_direction == actual_direction:
                direction_correct_count += 1
                confidence_accuracy_pairs.append((result.confidence, 1))
            else:
                confidence_accuracy_pairs.append((result.confidence, 0))

        n_valid = len(confidence_accuracy_pairs)
        if n_valid == 0:
            return None

        # Calculate metrics
        coverage = in_interval_count / n_valid
        direction_accuracy = direction_correct_count / n_valid

        # Calibration: bin by confidence and check accuracy
        confidence_bins = {}
        for conf, acc in confidence_accuracy_pairs:
            bin_key = round(conf * 10) / 10  # 0.0, 0.1, 0.2, ... 1.0
            if bin_key not in confidence_bins:
                confidence_bins[bin_key] = []
            confidence_bins[bin_key].append(acc)

        calibration_by_bin = {k: np.mean(v) for k, v in confidence_bins.items() if len(v) >= 3}

        return {
            "coverage_95ci": coverage,
            "expected_coverage": 0.95,
            "is_well_calibrated": 0.90 <= coverage <= 1.0,
            "direction_accuracy": direction_accuracy,
            "calibration_by_confidence": calibration_by_bin,
            "n_test_points": n_valid,
        }
