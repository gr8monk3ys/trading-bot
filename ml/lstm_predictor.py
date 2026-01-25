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

from ml.torch_utils import import_torch, get_torch_device

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

            def __init__(
                self, input_size, hidden_size, num_layers, output_size, dropout
            ):
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

        # Normalize features (fit new scaler)
        normalized = self._normalize_features(features, symbol, fit=True)

        # Target is the normalized close price (column index 3)
        targets = normalized[:, 3]

        # Create sequences
        X, y = self._create_sequences(normalized, targets)
        if len(X) == 0:
            raise ValueError(f"Could not create sequences for {symbol}")

        # Split into train/validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(
            f"Training {symbol}: {len(X_train)} train samples, "
            f"{len(X_val)} validation samples"
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
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best val loss: {best_val_loss:.6f})"
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
            samples_used=len(X),
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
            current_close = float(
                prices[-1].get("close", prices[-1].get("c", 0))
            )

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
