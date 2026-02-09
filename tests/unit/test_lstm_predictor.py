"""
Unit tests for the LSTM price predictor.

Tests cover:
- Feature preparation from OHLCV data
- Data normalization
- Sequence creation
- Model training
- Prediction generation
- Model persistence (save/load)

Note: Tests requiring PyTorch are skipped if torch is not installed.
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

# Set testing environment before importing config
os.environ["TESTING"] = "1"

# Check if PyTorch is available
import importlib.util

HAS_TORCH = importlib.util.find_spec("torch") is not None

# Check if scikit-learn is available
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

# Skip marker for tests requiring PyTorch
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# Skip marker for tests requiring scikit-learn
requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")

# Skip marker for tests requiring both PyTorch and scikit-learn
requires_ml_deps = pytest.mark.skipif(
    not HAS_TORCH or not HAS_SKLEARN,
    reason="PyTorch and/or scikit-learn not installed"
)


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """Test creating a PredictionResult."""
        from ml.lstm_predictor import PredictionResult

        result = PredictionResult(
            symbol="AAPL",
            predicted_price=155.0,
            predicted_direction="up",
            confidence=0.75,
            horizon=5,
            timestamp=datetime.now(),
            current_price=150.0,
        )

        assert result.symbol == "AAPL"
        assert result.predicted_price == 155.0
        assert result.predicted_direction == "up"
        assert result.confidence == 0.75
        assert result.horizon == 5
        assert result.current_price == 150.0

    def test_prediction_result_price_change_calculation(self):
        """Test that price_change_pct is calculated correctly."""
        from ml.lstm_predictor import PredictionResult

        result = PredictionResult(
            symbol="AAPL",
            predicted_price=165.0,
            predicted_direction="up",
            confidence=0.8,
            horizon=5,
            timestamp=datetime.now(),
            current_price=150.0,
        )

        # (165 - 150) / 150 * 100 = 10%
        assert abs(result.price_change_pct - 10.0) < 0.01


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics."""
        from ml.lstm_predictor import TrainingMetrics

        metrics = TrainingMetrics(
            symbol="AAPL",
            epochs=50,
            final_train_loss=0.001,
            final_val_loss=0.002,
            best_val_loss=0.0015,
            training_time_seconds=120.5,
            model_path="/tmp/AAPL_lstm.pt",
            samples_used=1000,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.epochs == 50
        assert metrics.final_train_loss == 0.001
        assert metrics.best_val_loss == 0.0015


@requires_ml_deps
class TestLSTMPredictorFeaturePreparation:
    """Tests for feature preparation methods."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance."""
        from ml.lstm_predictor import LSTMPredictor

        return LSTMPredictor(
            sequence_length=10,
            prediction_horizon=2,
            hidden_size=32,
            num_layers=1,
            use_gpu=False,
            model_dir=tempfile.mkdtemp(),
        )

    @pytest.fixture
    def sample_prices(self):
        """Generate sample OHLCV data."""
        prices = []
        base_price = 100.0
        for _i in range(100):
            open_price = base_price + np.random.randn() * 2
            close_price = open_price + np.random.randn() * 1
            high_price = max(open_price, close_price) + abs(np.random.randn() * 0.5)
            low_price = min(open_price, close_price) - abs(np.random.randn() * 0.5)
            volume = 1000000 + np.random.randint(-100000, 100000)

            prices.append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
            base_price = close_price

        return prices

    def test_prepare_features_basic(self, predictor, sample_prices):
        """Test basic feature preparation."""
        features = predictor._prepare_features(sample_prices)

        assert features.shape == (100, 5)
        assert features.dtype == np.float32

    def test_prepare_features_empty_data(self, predictor):
        """Test feature preparation with empty data."""
        features = predictor._prepare_features([])
        assert len(features) == 0

    def test_prepare_features_single_bar(self, predictor):
        """Test feature preparation with single bar."""
        prices = [{"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}]
        features = predictor._prepare_features(prices)
        assert len(features) == 0  # Need at least 2 bars

    def test_prepare_features_alternate_keys(self, predictor):
        """Test feature preparation with short key names (o, h, l, c, v)."""
        prices = [
            {"o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000},
            {"o": 101, "h": 102, "l": 100, "c": 101.5, "v": 1100},
        ]
        features = predictor._prepare_features(prices)

        assert features.shape == (2, 5)
        assert features[0, 0] == 100  # open
        assert features[0, 3] == 100.5  # close

    def test_normalize_features(self, predictor, sample_prices):
        """Test feature normalization."""
        features = predictor._prepare_features(sample_prices)
        normalized = predictor._normalize_features(features, "AAPL", fit=True)

        # All values should be between 0 and 1
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert normalized.shape == features.shape

    def test_normalize_features_uses_existing_scaler(self, predictor, sample_prices):
        """Test that normalization uses existing scaler when available."""
        features = predictor._prepare_features(sample_prices)

        # Fit scaler first
        predictor._normalize_features(features, "AAPL", fit=True)

        # Normalize again without fitting
        normalized = predictor._normalize_features(features[:50], "AAPL", fit=False)

        assert "AAPL" in predictor.scalers
        assert normalized.shape == (50, 5)

    def test_create_sequences(self, predictor, sample_prices):
        """Test sequence creation for LSTM input."""
        features = predictor._prepare_features(sample_prices)
        normalized = predictor._normalize_features(features, "AAPL", fit=True)
        targets = normalized[:, 3]  # Close prices

        X, y = predictor._create_sequences(normalized, targets)

        # With 100 samples, sequence_length=10, horizon=2:
        # num_sequences = 100 - 10 - 2 + 1 = 89
        expected_sequences = 100 - predictor.sequence_length - predictor.prediction_horizon + 1

        assert X.shape == (expected_sequences, predictor.sequence_length, 5)
        assert y.shape == (expected_sequences,)

    def test_create_sequences_without_targets(self, predictor, sample_prices):
        """Test sequence creation for inference (no targets)."""
        features = predictor._prepare_features(sample_prices)
        normalized = predictor._normalize_features(features, "AAPL", fit=True)

        X, y = predictor._create_sequences(normalized, targets=None)

        assert X.shape[1] == predictor.sequence_length
        assert y is None

    def test_create_sequences_insufficient_data(self, predictor):
        """Test sequence creation with insufficient data."""
        # Only 5 samples, need at least sequence_length + prediction_horizon
        features = np.random.rand(5, 5).astype(np.float32)

        X, y = predictor._create_sequences(features, features[:, 3])

        assert len(X) == 0


@requires_ml_deps
class TestLSTMPredictorTraining:
    """Tests for model training."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance."""
        from ml.lstm_predictor import LSTMPredictor

        return LSTMPredictor(
            sequence_length=10,
            prediction_horizon=2,
            hidden_size=16,  # Small for fast tests
            num_layers=1,
            use_gpu=False,
            model_dir=tempfile.mkdtemp(),
        )

    @pytest.fixture
    def training_prices(self):
        """Generate sufficient training data."""
        np.random.seed(42)
        prices = []
        base_price = 100.0

        for i in range(200):
            # Add some trend to make learning possible
            trend = 0.01 * i
            noise = np.random.randn() * 0.5

            open_price = base_price + trend + noise
            close_price = open_price + np.random.randn() * 0.3
            high_price = max(open_price, close_price) + abs(np.random.randn() * 0.2)
            low_price = min(open_price, close_price) - abs(np.random.randn() * 0.2)
            volume = 1000000 + np.random.randint(-100000, 100000)

            prices.append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        return prices

    def test_train_basic(self, predictor, training_prices):
        """Test basic model training."""
        metrics = predictor.train(
            symbol="AAPL",
            prices=training_prices,
            epochs=5,  # Few epochs for fast test
            batch_size=16,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.epochs > 0
        assert metrics.final_train_loss < float("inf")
        assert metrics.final_val_loss < float("inf")
        assert "AAPL" in predictor.models

    def test_train_insufficient_data(self, predictor):
        """Test training with insufficient data."""
        # Too few bars for training
        short_prices = [
            {"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}
            for _ in range(20)
        ]

        metrics = predictor.train(symbol="AAPL", prices=short_prices, epochs=5)

        assert metrics.epochs == 0
        assert metrics.final_train_loss == float("inf")

    def test_train_saves_model(self, predictor, training_prices):
        """Test that training saves the model to disk."""
        metrics = predictor.train(
            symbol="AAPL", prices=training_prices, epochs=3, batch_size=16
        )

        assert os.path.exists(metrics.model_path)

    def test_train_creates_scaler(self, predictor, training_prices):
        """Test that training creates and stores a scaler."""
        predictor.train(symbol="AAPL", prices=training_prices, epochs=3)

        assert "AAPL" in predictor.scalers
        assert hasattr(predictor.scalers["AAPL"], "data_min_")
        assert hasattr(predictor.scalers["AAPL"], "data_max_")


@requires_ml_deps
class TestLSTMPredictorPrediction:
    """Tests for prediction generation."""

    @pytest.fixture
    def trained_predictor(self):
        """Create and train a predictor."""
        from ml.lstm_predictor import LSTMPredictor

        np.random.seed(42)

        predictor = LSTMPredictor(
            sequence_length=10,
            prediction_horizon=2,
            hidden_size=16,
            num_layers=1,
            use_gpu=False,
            model_dir=tempfile.mkdtemp(),
        )

        # Generate training data
        prices = []
        base_price = 100.0
        for _i in range(150):
            open_price = base_price + np.random.randn() * 0.5
            close_price = open_price + np.random.randn() * 0.3
            high_price = max(open_price, close_price) + 0.1
            low_price = min(open_price, close_price) - 0.1
            volume = 1000000

            prices.append(
                {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
            base_price = close_price

        predictor.train(symbol="AAPL", prices=prices, epochs=3, batch_size=16)

        return predictor, prices

    def test_predict_basic(self, trained_predictor):
        """Test basic prediction."""
        predictor, prices = trained_predictor

        result = predictor.predict("AAPL", prices[-20:])

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.predicted_direction in ["up", "down", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.horizon == predictor.prediction_horizon

    def test_predict_no_model(self, trained_predictor):
        """Test prediction without trained model."""
        predictor, prices = trained_predictor

        result = predictor.predict("MSFT", prices[-20:])

        assert result is None

    def test_predict_insufficient_data(self, trained_predictor):
        """Test prediction with insufficient data."""
        predictor, prices = trained_predictor

        # Only 5 bars, need sequence_length (10)
        result = predictor.predict("AAPL", prices[:5])

        assert result is None

    def test_predict_returns_current_price(self, trained_predictor):
        """Test that prediction includes current price."""
        predictor, prices = trained_predictor

        result = predictor.predict("AAPL", prices[-20:])

        assert result.current_price > 0
        assert result.current_price == prices[-1]["close"]


@requires_ml_deps
class TestLSTMPredictorPersistence:
    """Tests for model save/load functionality."""

    @pytest.fixture
    def predictor_with_model(self):
        """Create and train a predictor."""
        from ml.lstm_predictor import LSTMPredictor

        np.random.seed(42)
        model_dir = tempfile.mkdtemp()

        predictor = LSTMPredictor(
            sequence_length=10,
            prediction_horizon=2,
            hidden_size=16,
            num_layers=1,
            use_gpu=False,
            model_dir=model_dir,
        )

        # Generate training data
        prices = []
        base_price = 100.0
        for _i in range(150):
            prices.append(
                {
                    "open": base_price,
                    "high": base_price + 0.5,
                    "low": base_price - 0.5,
                    "close": base_price + np.random.randn() * 0.1,
                    "volume": 1000000,
                }
            )
            base_price += np.random.randn() * 0.1

        predictor.train(symbol="AAPL", prices=prices, epochs=3, batch_size=16)

        return predictor, prices, model_dir

    def test_save_model(self, predictor_with_model):
        """Test saving a model."""
        predictor, prices, model_dir = predictor_with_model

        success = predictor.save_model("AAPL")

        assert success
        assert os.path.exists(os.path.join(model_dir, "AAPL_lstm.pt"))

    def test_load_model(self, predictor_with_model):
        """Test loading a saved model."""
        from ml.lstm_predictor import LSTMPredictor

        predictor, prices, model_dir = predictor_with_model

        # Create new predictor and load the model
        new_predictor = LSTMPredictor(
            sequence_length=10,
            prediction_horizon=2,
            hidden_size=16,
            num_layers=1,
            use_gpu=False,
            model_dir=model_dir,
        )

        success = new_predictor.load_model("AAPL")

        assert success
        assert "AAPL" in new_predictor.models
        assert "AAPL" in new_predictor.scalers

    def test_load_nonexistent_model(self, predictor_with_model):
        """Test loading a model that doesn't exist."""
        predictor, _, _ = predictor_with_model

        success = predictor.load_model("NONEXISTENT")

        assert not success

    def test_loaded_model_can_predict(self, predictor_with_model):
        """Test that a loaded model can make predictions."""
        from ml.lstm_predictor import LSTMPredictor

        predictor, prices, model_dir = predictor_with_model

        # Create new predictor and load
        new_predictor = LSTMPredictor(
            sequence_length=10,
            prediction_horizon=2,
            hidden_size=16,
            num_layers=1,
            use_gpu=False,
            model_dir=model_dir,
        )
        new_predictor.load_model("AAPL")

        # Should be able to predict
        result = new_predictor.predict("AAPL", prices[-20:])

        assert result is not None
        assert result.predicted_direction in ["up", "down", "neutral"]


@requires_torch
class TestLSTMPredictorUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance."""
        from ml.lstm_predictor import LSTMPredictor

        return LSTMPredictor(
            sequence_length=10,
            prediction_horizon=2,
            hidden_size=16,
            num_layers=1,
            use_gpu=False,
            model_dir=tempfile.mkdtemp(),
        )

    def test_has_model(self, predictor):
        """Test has_model method."""
        assert not predictor.has_model("AAPL")

        # Manually add a model
        predictor.models["AAPL"] = MagicMock()

        assert predictor.has_model("AAPL")

    def test_list_trained_models(self, predictor):
        """Test listing trained models."""
        assert predictor.list_trained_models() == []

        predictor.models["AAPL"] = MagicMock()
        predictor.models["MSFT"] = MagicMock()

        models = predictor.list_trained_models()
        assert "AAPL" in models
        assert "MSFT" in models

    def test_clear_model(self, predictor):
        """Test clearing a model from memory."""
        predictor.models["AAPL"] = MagicMock()
        predictor.scalers["AAPL"] = MagicMock()

        removed = predictor.clear_model("AAPL")

        assert removed
        assert "AAPL" not in predictor.models
        assert "AAPL" not in predictor.scalers

    def test_clear_nonexistent_model(self, predictor):
        """Test clearing a model that doesn't exist."""
        removed = predictor.clear_model("NONEXISTENT")

        assert not removed

    def test_get_model_info(self, predictor):
        """Test getting model info."""
        assert predictor.get_model_info("AAPL") is None

        predictor.models["AAPL"] = MagicMock()
        predictor.scalers["AAPL"] = MagicMock()

        info = predictor.get_model_info("AAPL")

        assert info is not None
        assert info["symbol"] == "AAPL"
        assert info["has_model"]
        assert info["has_scaler"]
        assert info["sequence_length"] == 10
        assert info["prediction_horizon"] == 2


@requires_torch
class TestLSTMNetwork:
    """Tests for the LSTM network architecture."""

    def test_network_creation(self):
        """Test creating an LSTM network."""
        from ml.lstm_predictor import LSTMNetwork

        network = LSTMNetwork(
            input_size=5,
            hidden_size=32,
            num_layers=2,
            output_size=1,
            dropout=0.2,
        )

        assert network.model is not None
        assert network.input_size == 5
        assert network.hidden_size == 32
        assert network.num_layers == 2

    def test_network_forward_pass(self):
        """Test forward pass through the network."""
        import torch

        from ml.lstm_predictor import LSTMNetwork

        network = LSTMNetwork(input_size=5, hidden_size=16, num_layers=1)

        # Create sample input: (batch_size=2, sequence_length=10, features=5)
        x = torch.randn(2, 10, 5)

        output = network.model(x)

        assert output.shape == (2, 1)  # (batch_size, output_size)
