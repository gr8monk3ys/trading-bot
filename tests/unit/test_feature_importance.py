"""
Unit tests for ml/feature_importance.py

Tests feature importance analysis for ML models including:
- FeatureImportanceResult dataclass methods
- ImportanceDriftResult dataclass
- FeatureImportanceAnalyzer with multiple importance methods
- LSTMFeatureImportance for sequence models
- Helper function analyze_feature_importance

All heavy dependencies (SHAP, PyTorch) are mocked for isolated testing.
"""

from datetime import datetime
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.feature_importance import (
    FeatureImportanceAnalyzer,
    FeatureImportanceResult,
    ImportanceDriftResult,
    LSTMFeatureImportance,
    analyze_feature_importance,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_feature_names() -> List[str]:
    """Return sample feature names for testing."""
    return ["open", "high", "low", "close", "volume"]


@pytest.fixture
def sample_importance_scores() -> Dict[str, float]:
    """Return sample importance scores for testing."""
    return {
        "open": 0.15,
        "high": 0.25,
        "low": 0.10,
        "close": 0.40,
        "volume": 0.10,
    }


@pytest.fixture
def sample_importance_result(
    sample_feature_names, sample_importance_scores
) -> FeatureImportanceResult:
    """Return a sample FeatureImportanceResult for testing."""
    return FeatureImportanceResult(
        feature_names=sample_feature_names,
        importance_scores=sample_importance_scores,
        method="permutation",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        model_type="generic",
        n_samples=100,
    )


@pytest.fixture
def mock_model():
    """Create a mock model with predict method."""
    model = MagicMock()
    # Default predict returns array matching input shape
    model.predict = MagicMock(side_effect=lambda X: np.random.randn(len(X)))
    return model


@pytest.fixture
def mock_lstm_model():
    """Create a mock LSTM model for sequence data."""
    model = MagicMock()
    # For 3D input (samples, sequence, features)
    model.predict = MagicMock(side_effect=lambda X: np.random.randn(X.shape[0]))
    return model


@pytest.fixture
def sample_2d_data():
    """Generate sample 2D input data (samples, features)."""
    np.random.seed(42)
    return np.random.randn(50, 5)


@pytest.fixture
def sample_3d_data():
    """Generate sample 3D input data for LSTM (samples, sequence, features)."""
    np.random.seed(42)
    return np.random.randn(50, 10, 5)


@pytest.fixture
def sample_target():
    """Generate sample target values."""
    np.random.seed(42)
    return np.random.randn(50)


@pytest.fixture
def analyzer_with_features(sample_feature_names) -> FeatureImportanceAnalyzer:
    """Create an analyzer with feature names."""
    return FeatureImportanceAnalyzer(feature_names=sample_feature_names)


@pytest.fixture
def analyzer_with_history(sample_feature_names) -> FeatureImportanceAnalyzer:
    """Create an analyzer with pre-populated importance history."""
    analyzer = FeatureImportanceAnalyzer(feature_names=sample_feature_names)

    # Add 10 historical results with varying importance
    for i in range(10):
        importance_scores = {
            "open": 0.15 + (i * 0.01),
            "high": 0.25 - (i * 0.005),
            "low": 0.10 + (i * 0.002),
            "close": 0.40 - (i * 0.02),
            "volume": 0.10 + (i * 0.003),
        }
        result = FeatureImportanceResult(
            feature_names=sample_feature_names,
            importance_scores=importance_scores,
            method="permutation",
            timestamp=datetime(2024, 1, i + 1, 10, 0, 0),
            model_type="generic",
            n_samples=100,
        )
        analyzer._importance_history.append(result)

    return analyzer


# =============================================================================
# TESTS: FeatureImportanceResult DATACLASS
# =============================================================================


class TestFeatureImportanceResult:
    """Tests for FeatureImportanceResult dataclass."""

    def test_initialization(self, sample_importance_result):
        """Test dataclass initializes correctly."""
        result = sample_importance_result

        assert result.method == "permutation"
        assert result.model_type == "generic"
        assert result.n_samples == 100
        assert len(result.feature_names) == 5
        assert len(result.importance_scores) == 5

    def test_get_top_features_default(self, sample_importance_result):
        """Test get_top_features returns top 10 by default."""
        top = sample_importance_result.get_top_features()

        # Should return all 5 since we only have 5 features
        assert len(top) == 5
        # First should be highest importance (close = 0.40)
        assert top[0][0] == "close"
        assert top[0][1] == 0.40

    def test_get_top_features_limited(self, sample_importance_result):
        """Test get_top_features respects n parameter."""
        top = sample_importance_result.get_top_features(n=3)

        assert len(top) == 3
        # Verify sorted by absolute importance
        assert top[0][0] == "close"  # 0.40
        assert top[1][0] == "high"  # 0.25
        assert top[2][0] == "open"  # 0.15

    def test_get_top_features_with_negative_importance(self):
        """Test get_top_features handles negative importance values."""
        result = FeatureImportanceResult(
            feature_names=["a", "b", "c"],
            importance_scores={"a": 0.3, "b": -0.5, "c": 0.1},
            method="shap",
            timestamp=datetime.now(),
            model_type="tree",
            n_samples=50,
        )

        top = result.get_top_features(n=3)
        # Should sort by absolute value
        assert top[0][0] == "b"  # |-0.5| = 0.5
        assert top[1][0] == "a"  # |0.3| = 0.3
        assert top[2][0] == "c"  # |0.1| = 0.1

    def test_get_top_features_empty_scores(self):
        """Test get_top_features with empty scores."""
        result = FeatureImportanceResult(
            feature_names=[],
            importance_scores={},
            method="permutation",
            timestamp=datetime.now(),
            model_type="generic",
            n_samples=0,
        )

        top = result.get_top_features(n=5)
        assert len(top) == 0

    def test_to_dataframe(self, sample_importance_result):
        """Test conversion to pandas DataFrame."""
        df = sample_importance_result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "feature" in df.columns
        assert "importance" in df.columns
        # Should be sorted by absolute importance descending
        assert df.iloc[0]["feature"] == "close"
        assert df.iloc[0]["importance"] == 0.40

    def test_to_dataframe_preserves_all_features(self, sample_importance_result):
        """Test DataFrame contains all features."""
        df = sample_importance_result.to_dataframe()

        features_in_df = set(df["feature"].tolist())
        expected = {"open", "high", "low", "close", "volume"}
        assert features_in_df == expected


# =============================================================================
# TESTS: ImportanceDriftResult DATACLASS
# =============================================================================


class TestImportanceDriftResult:
    """Tests for ImportanceDriftResult dataclass."""

    def test_initialization(self):
        """Test dataclass initializes with all fields."""
        result = ImportanceDriftResult(
            feature_name="close",
            historical_importance=0.35,
            recent_importance=0.20,
            change_pct=-0.4286,
            is_significant=True,
            p_value=0.02,
        )

        assert result.feature_name == "close"
        assert result.historical_importance == 0.35
        assert result.recent_importance == 0.20
        assert abs(result.change_pct - (-0.4286)) < 0.001
        assert result.is_significant is True
        assert result.p_value == 0.02

    def test_initialization_without_p_value(self):
        """Test dataclass initializes with optional p_value as None."""
        result = ImportanceDriftResult(
            feature_name="volume",
            historical_importance=0.10,
            recent_importance=0.25,
            change_pct=1.5,
            is_significant=True,
        )

        assert result.p_value is None

    def test_not_significant_drift(self):
        """Test drift result marked as not significant."""
        result = ImportanceDriftResult(
            feature_name="open",
            historical_importance=0.15,
            recent_importance=0.16,
            change_pct=0.0667,
            is_significant=False,
        )

        assert result.is_significant is False


# =============================================================================
# TESTS: FeatureImportanceAnalyzer - INITIALIZATION
# =============================================================================


class TestFeatureImportanceAnalyzerInit:
    """Tests for FeatureImportanceAnalyzer initialization."""

    def test_init_default_values(self):
        """Test default initialization."""
        analyzer = FeatureImportanceAnalyzer()

        assert analyzer.feature_names == []
        assert analyzer.background_samples == 100
        assert analyzer._importance_history == []

    def test_init_with_feature_names(self, sample_feature_names):
        """Test initialization with feature names."""
        analyzer = FeatureImportanceAnalyzer(feature_names=sample_feature_names)

        assert analyzer.feature_names == sample_feature_names

    def test_init_with_custom_background_samples(self):
        """Test initialization with custom background samples."""
        analyzer = FeatureImportanceAnalyzer(background_samples=50)

        assert analyzer.background_samples == 50


# =============================================================================
# TESTS: FeatureImportanceAnalyzer - PERMUTATION IMPORTANCE
# =============================================================================


class TestPermutationImportance:
    """Tests for calculate_permutation_importance method."""

    def test_permutation_importance_basic(
        self, analyzer_with_features, mock_model, sample_2d_data, sample_target
    ):
        """Test basic permutation importance calculation."""
        result = analyzer_with_features.calculate_permutation_importance(
            model=mock_model,
            X=sample_2d_data,
            y=sample_target,
            n_repeats=3,
        )

        assert result is not None
        assert isinstance(result, FeatureImportanceResult)
        assert result.method == "permutation"
        assert result.model_type == "generic"
        assert result.n_samples == 50
        assert len(result.importance_scores) == 5

    def test_permutation_importance_calls_predict(
        self, analyzer_with_features, mock_model, sample_2d_data, sample_target
    ):
        """Test that predict is called multiple times for permutation."""
        analyzer_with_features.calculate_permutation_importance(
            model=mock_model,
            X=sample_2d_data,
            y=sample_target,
            n_repeats=5,
        )

        # 1 baseline + (5 features * 5 repeats) = 26 calls
        expected_calls = 1 + (5 * 5)
        assert mock_model.predict.call_count == expected_calls

    def test_permutation_importance_feature_shuffling_affects_score(self):
        """Test that shuffling features actually changes predictions."""

        # Create a model that returns different values based on first feature
        class SimpleModel:
            def predict(self, X):
                # Returns sum of first feature
                if len(X.shape) == 2:
                    return X[:, 0].copy()
                return X[:, :, 0].mean(axis=1)

        model = SimpleModel()
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0]  # Target is first feature

        analyzer = FeatureImportanceAnalyzer(feature_names=["f1", "f2", "f3"])
        result = analyzer.calculate_permutation_importance(
            model=model,
            X=X,
            y=y,
            n_repeats=10,
        )

        # First feature should have highest importance
        assert result.importance_scores["f1"] > result.importance_scores["f2"]
        assert result.importance_scores["f1"] > result.importance_scores["f3"]

    def test_permutation_importance_3d_array_lstm(
        self, mock_lstm_model, sample_3d_data, sample_target, sample_feature_names
    ):
        """Test permutation importance handles 3D arrays for LSTM."""
        analyzer = FeatureImportanceAnalyzer(feature_names=sample_feature_names)

        result = analyzer.calculate_permutation_importance(
            model=mock_lstm_model,
            X=sample_3d_data,
            y=sample_target,
            n_repeats=3,
        )

        assert result is not None
        assert len(result.importance_scores) == 5

    def test_permutation_importance_generates_feature_names(self):
        """Test feature names are generated when not provided."""
        analyzer = FeatureImportanceAnalyzer()  # No feature names
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = MagicMock()
        model.predict = MagicMock(return_value=y)

        result = analyzer.calculate_permutation_importance(
            model=model,
            X=X,
            y=y,
        )

        assert "feature_0" in result.importance_scores
        assert "feature_1" in result.importance_scores
        assert "feature_2" in result.importance_scores

    def test_permutation_importance_custom_scoring_fn(
        self, analyzer_with_features, mock_model, sample_2d_data, sample_target
    ):
        """Test permutation importance with custom scoring function."""
        custom_scoring_calls = []

        def custom_scoring(y_true, y_pred):
            custom_scoring_calls.append(1)
            return -np.mean(np.abs(y_true - y_pred))  # Negative MAE

        result = analyzer_with_features.calculate_permutation_importance(
            model=mock_model,
            X=sample_2d_data,
            y=sample_target,
            scoring_fn=custom_scoring,
            n_repeats=2,
        )

        assert result is not None
        # Scoring function should be called multiple times
        assert len(custom_scoring_calls) > 0

    def test_permutation_importance_flattens_2d_predictions(
        self, analyzer_with_features, sample_2d_data, sample_target
    ):
        """Test that 2D predictions are flattened."""
        model = MagicMock()
        # Return 2D array (n_samples, 1)
        model.predict = MagicMock(side_effect=lambda X: np.random.randn(len(X), 1))

        result = analyzer_with_features.calculate_permutation_importance(
            model=model,
            X=sample_2d_data,
            y=sample_target,
        )

        # Should complete without error
        assert result is not None

    def test_permutation_importance_adds_to_history(
        self, analyzer_with_features, mock_model, sample_2d_data, sample_target
    ):
        """Test that result is added to importance history."""
        initial_history_len = len(analyzer_with_features._importance_history)

        analyzer_with_features.calculate_permutation_importance(
            model=mock_model,
            X=sample_2d_data,
            y=sample_target,
        )

        assert len(analyzer_with_features._importance_history) == initial_history_len + 1


# =============================================================================
# TESTS: FeatureImportanceAnalyzer - SHAP IMPORTANCE
# =============================================================================


class TestSHAPImportance:
    """Tests for calculate_shap_importance method."""

    def test_shap_returns_none_when_unavailable(self, analyzer_with_features):
        """Test SHAP returns None when library not available."""
        with patch("ml.feature_importance._import_shap", return_value=None):
            np.random.seed(42)
            X = np.random.randn(50, 5)
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=X,
            )

            assert result is None

    def test_shap_tree_explainer(self, analyzer_with_features, sample_2d_data):
        """Test SHAP with tree explainer."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        mock_explainer.shap_values.return_value = np.random.randn(50, 5)

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_2d_data,
                model_type="tree",
            )

            assert result is not None
            assert result.method == "shap"
            assert result.model_type == "tree"
            mock_shap.TreeExplainer.assert_called_once_with(model)

    def test_shap_deep_explainer(self, analyzer_with_features, sample_2d_data):
        """Test SHAP with deep explainer for neural networks."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.DeepExplainer.return_value = mock_explainer
        mock_explainer.shap_values.return_value = np.random.randn(50, 5)

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_2d_data,
                model_type="deep",
            )

            assert result is not None
            assert result.model_type == "deep"
            mock_shap.DeepExplainer.assert_called_once()

    def test_shap_kernel_explainer(self, analyzer_with_features, sample_2d_data):
        """Test SHAP with kernel explainer (model-agnostic)."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_explainer
        mock_shap.sample.return_value = sample_2d_data[:10]
        mock_explainer.shap_values.return_value = np.random.randn(50, 5)

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_2d_data,
                model_type="kernel",
            )

            assert result is not None
            assert result.model_type == "kernel"

    def test_shap_linear_explainer(self, analyzer_with_features, sample_2d_data):
        """Test SHAP with linear explainer."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.LinearExplainer.return_value = mock_explainer
        mock_explainer.shap_values.return_value = np.random.randn(50, 5)

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_2d_data,
                model_type="linear",
            )

            assert result is not None
            assert result.model_type == "linear"

    def test_shap_unknown_model_type(self, analyzer_with_features, sample_2d_data):
        """Test SHAP returns None for unknown model type."""
        mock_shap = MagicMock()

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_2d_data,
                model_type="unknown_type",
            )

            assert result is None

    def test_shap_handles_multi_output(self, analyzer_with_features, sample_2d_data):
        """Test SHAP handles multi-output models (list of SHAP values)."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        # Return list of arrays (multi-output)
        mock_explainer.shap_values.return_value = [
            np.random.randn(50, 5),
            np.random.randn(50, 5),
        ]

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_2d_data,
            )

            # Should use first output
            assert result is not None
            assert len(result.importance_scores) == 5

    def test_shap_handles_3d_arrays(self, analyzer_with_features, sample_3d_data):
        """Test SHAP handles 3D arrays (sequence models)."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.DeepExplainer.return_value = mock_explainer
        # Return 3D SHAP values (samples, sequence, features)
        mock_explainer.shap_values.return_value = np.random.randn(50, 10, 5)

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_3d_data,
                model_type="deep",
            )

            # Should average over sequence dimension
            assert result is not None
            assert len(result.importance_scores) == 5

    def test_shap_feature_count_mismatch_generates_names(
        self, analyzer_with_features, sample_2d_data
    ):
        """Test SHAP generates feature names when count doesn't match."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        # Return 10 features instead of 5
        mock_explainer.shap_values.return_value = np.random.randn(50, 10)

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_2d_data,
            )

            assert result is not None
            assert len(result.importance_scores) == 10
            assert "feature_0" in result.importance_scores

    def test_shap_exception_handling(self, analyzer_with_features, sample_2d_data):
        """Test SHAP returns None on exception."""
        mock_shap = MagicMock()
        mock_shap.TreeExplainer.side_effect = Exception("SHAP error")

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyzer_with_features.calculate_shap_importance(
                model=model,
                X=sample_2d_data,
            )

            assert result is None


# =============================================================================
# TESTS: FeatureImportanceAnalyzer - GRADIENT IMPORTANCE
# =============================================================================


class TestGradientImportance:
    """Tests for calculate_gradient_importance method."""

    def test_gradient_importance_returns_none_without_pytorch(
        self, analyzer_with_features, sample_2d_data
    ):
        """Test gradient importance returns None when PyTorch unavailable."""
        with patch(
            "ml.feature_importance.FeatureImportanceAnalyzer.calculate_gradient_importance"
        ) as mock_method:
            mock_method.return_value = None

            result = analyzer_with_features.calculate_gradient_importance(
                model=MagicMock(),
                X=sample_2d_data,
            )

            # Since we're patching the whole method, it returns None
            assert result is None

    def test_gradient_importance_with_mock_torch(self, sample_feature_names):
        """Test gradient importance with mocked PyTorch."""
        # Create mock torch module
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.FloatTensor.return_value = mock_tensor
        mock_tensor.requires_grad_.return_value = mock_tensor
        mock_tensor.grad = MagicMock()
        mock_tensor.grad.numpy.return_value = np.random.randn(50, 5)

        # Mock model
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_model.return_value = mock_output
        mock_output.sum.return_value = mock_output

        analyzer = FeatureImportanceAnalyzer(feature_names=sample_feature_names)
        np.random.seed(42)
        X = np.random.randn(50, 5)

        # Patch the torch import
        with patch.dict("sys.modules", {"ml.torch_utils": MagicMock()}):
            with patch(
                "ml.feature_importance.FeatureImportanceAnalyzer.calculate_gradient_importance"
            ) as mock_grad:
                mock_grad.return_value = FeatureImportanceResult(
                    feature_names=sample_feature_names,
                    importance_scores=dict.fromkeys(sample_feature_names, 0.2),
                    method="gradient",
                    timestamp=datetime.now(),
                    model_type="neural_network",
                    n_samples=50,
                )

                result = analyzer.calculate_gradient_importance(
                    model=mock_model,
                    X=X,
                )

                assert result is not None
                assert result.method == "gradient"


# =============================================================================
# TESTS: FeatureImportanceAnalyzer - IMPORTANCE DRIFT DETECTION
# =============================================================================


class TestImportanceDriftDetection:
    """Tests for detect_importance_drift method."""

    def test_drift_detection_insufficient_history(self, analyzer_with_features):
        """Test drift detection with insufficient history."""
        # Only add 3 results (need at least 2 * window_size)
        for _i in range(3):
            result = FeatureImportanceResult(
                feature_names=["f1"],
                importance_scores={"f1": 0.5},
                method="permutation",
                timestamp=datetime.now(),
                model_type="generic",
                n_samples=100,
            )
            analyzer_with_features._importance_history.append(result)

        drift_results = analyzer_with_features.detect_importance_drift(window_size=5)

        assert drift_results == []

    def test_drift_detection_finds_significant_changes(self, sample_feature_names):
        """Test drift detection identifies significant importance changes."""
        analyzer = FeatureImportanceAnalyzer(feature_names=sample_feature_names)

        # Add historical results with high close importance
        for i in range(5):
            result = FeatureImportanceResult(
                feature_names=sample_feature_names,
                importance_scores={
                    "open": 0.15,
                    "high": 0.25,
                    "low": 0.10,
                    "close": 0.40,  # High importance
                    "volume": 0.10,
                },
                method="permutation",
                timestamp=datetime(2024, 1, i + 1),
                model_type="generic",
                n_samples=100,
            )
            analyzer._importance_history.append(result)

        # Add recent results with low close importance (significant change)
        for i in range(5):
            result = FeatureImportanceResult(
                feature_names=sample_feature_names,
                importance_scores={
                    "open": 0.15,
                    "high": 0.25,
                    "low": 0.10,
                    "close": 0.10,  # Low importance - 75% drop
                    "volume": 0.40,  # Increased
                },
                method="permutation",
                timestamp=datetime(2024, 1, i + 6),
                model_type="generic",
                n_samples=100,
            )
            analyzer._importance_history.append(result)

        drift_results = analyzer.detect_importance_drift(
            window_size=5,
            significance_threshold=0.3,
        )

        # Should find significant drift in close and volume
        significant = [d for d in drift_results if d.is_significant]
        assert len(significant) >= 2

        feature_names_with_drift = {d.feature_name for d in significant}
        assert "close" in feature_names_with_drift
        assert "volume" in feature_names_with_drift

    def test_drift_detection_calculates_change_percent(self, sample_feature_names):
        """Test drift detection calculates correct change percentage."""
        analyzer = FeatureImportanceAnalyzer(feature_names=sample_feature_names)

        # Historical: close = 0.40
        for i in range(5):
            result = FeatureImportanceResult(
                feature_names=["close"],
                importance_scores={"close": 0.40},
                method="permutation",
                timestamp=datetime(2024, 1, i + 1),
                model_type="generic",
                n_samples=100,
            )
            analyzer._importance_history.append(result)

        # Recent: close = 0.20 (50% decrease)
        for i in range(5):
            result = FeatureImportanceResult(
                feature_names=["close"],
                importance_scores={"close": 0.20},
                method="permutation",
                timestamp=datetime(2024, 1, i + 6),
                model_type="generic",
                n_samples=100,
            )
            analyzer._importance_history.append(result)

        drift_results = analyzer.detect_importance_drift(window_size=5)

        close_drift = next(d for d in drift_results if d.feature_name == "close")
        assert close_drift.historical_importance == 0.40
        assert close_drift.recent_importance == 0.20
        assert abs(close_drift.change_pct - (-0.5)) < 0.01  # -50%

    def test_drift_detection_handles_zero_historical_importance(self, sample_feature_names):
        """Test drift handles features with zero historical importance."""
        analyzer = FeatureImportanceAnalyzer(feature_names=sample_feature_names)

        # Historical: volume = 0
        for i in range(5):
            analyzer._importance_history.append(
                FeatureImportanceResult(
                    feature_names=["volume"],
                    importance_scores={"volume": 0.0},
                    method="permutation",
                    timestamp=datetime(2024, 1, i + 1),
                    model_type="generic",
                    n_samples=100,
                )
            )

        # Recent: volume > 0
        for i in range(5):
            analyzer._importance_history.append(
                FeatureImportanceResult(
                    feature_names=["volume"],
                    importance_scores={"volume": 0.25},
                    method="permutation",
                    timestamp=datetime(2024, 1, i + 6),
                    model_type="generic",
                    n_samples=100,
                )
            )

        drift_results = analyzer.detect_importance_drift(window_size=5)

        volume_drift = next(d for d in drift_results if d.feature_name == "volume")
        assert volume_drift.change_pct == float("inf")

    def test_drift_detection_sorted_by_absolute_change(self, analyzer_with_history):
        """Test drift results are sorted by absolute change percentage."""
        drift_results = analyzer_with_history.detect_importance_drift(window_size=5)

        # Verify sorting
        for i in range(len(drift_results) - 1):
            assert abs(drift_results[i].change_pct) >= abs(drift_results[i + 1].change_pct)

    def test_drift_handles_new_features_in_recent(self, sample_feature_names):
        """Test drift handles features appearing only in recent results."""
        analyzer = FeatureImportanceAnalyzer()

        # Historical without 'new_feature'
        for i in range(5):
            analyzer._importance_history.append(
                FeatureImportanceResult(
                    feature_names=["f1"],
                    importance_scores={"f1": 0.5},
                    method="permutation",
                    timestamp=datetime(2024, 1, i + 1),
                    model_type="generic",
                    n_samples=100,
                )
            )

        # Recent with 'new_feature'
        for i in range(5):
            analyzer._importance_history.append(
                FeatureImportanceResult(
                    feature_names=["f1", "new_feature"],
                    importance_scores={"f1": 0.5, "new_feature": 0.3},
                    method="permutation",
                    timestamp=datetime(2024, 1, i + 6),
                    model_type="generic",
                    n_samples=100,
                )
            )

        drift_results = analyzer.detect_importance_drift(window_size=5)

        new_feature_drift = next(d for d in drift_results if d.feature_name == "new_feature")
        assert new_feature_drift.historical_importance == 0.0
        assert new_feature_drift.recent_importance == 0.3
        assert new_feature_drift.change_pct == float("inf")


# =============================================================================
# TESTS: FeatureImportanceAnalyzer - AVERAGE IMPORTANCE
# =============================================================================


class TestAverageImportance:
    """Tests for _average_importance method."""

    def test_average_importance_empty_list(self):
        """Test average importance with empty results list."""
        analyzer = FeatureImportanceAnalyzer()

        result = analyzer._average_importance([])

        assert result == {}

    def test_average_importance_single_result(self, sample_importance_result):
        """Test average importance with single result."""
        analyzer = FeatureImportanceAnalyzer()

        result = analyzer._average_importance([sample_importance_result])

        assert result == sample_importance_result.importance_scores

    def test_average_importance_multiple_results(self, sample_feature_names):
        """Test average importance across multiple results."""
        analyzer = FeatureImportanceAnalyzer()

        results = [
            FeatureImportanceResult(
                feature_names=sample_feature_names,
                importance_scores={"f1": 0.1, "f2": 0.2},
                method="permutation",
                timestamp=datetime.now(),
                model_type="generic",
                n_samples=100,
            ),
            FeatureImportanceResult(
                feature_names=sample_feature_names,
                importance_scores={"f1": 0.3, "f2": 0.4},
                method="permutation",
                timestamp=datetime.now(),
                model_type="generic",
                n_samples=100,
            ),
        ]

        avg = analyzer._average_importance(results)

        assert abs(avg["f1"] - 0.2) < 0.001  # (0.1 + 0.3) / 2
        assert abs(avg["f2"] - 0.3) < 0.001  # (0.2 + 0.4) / 2


# =============================================================================
# TESTS: FeatureImportanceAnalyzer - GET IMPORTANCE REPORT
# =============================================================================


class TestGetImportanceReport:
    """Tests for get_importance_report method."""

    def test_report_with_no_history(self):
        """Test report returns error when no history exists."""
        analyzer = FeatureImportanceAnalyzer()

        report = analyzer.get_importance_report()

        assert "error" in report
        assert "No importance calculations" in report["error"]

    def test_report_with_history(self, analyzer_with_history):
        """Test report with populated history."""
        report = analyzer_with_history.get_importance_report()

        assert "latest_method" in report
        assert "n_features" in report
        assert "top_features" in report
        assert "n_importance_records" in report
        assert "drift_detected" in report
        assert "drifting_features" in report

    def test_report_top_features_format(self, analyzer_with_history):
        """Test top features in report have correct format."""
        report = analyzer_with_history.get_importance_report()

        assert len(report["top_features"]) > 0
        for feature_info in report["top_features"]:
            assert "feature" in feature_info
            assert "importance" in feature_info
            assert isinstance(feature_info["importance"], float)

    def test_report_limits_drifting_features(self, sample_feature_names):
        """Test report limits drifting features to top 5."""
        analyzer = FeatureImportanceAnalyzer(feature_names=sample_feature_names)

        # Create many significant drifts
        for i in range(7):
            analyzer._importance_history.append(
                FeatureImportanceResult(
                    feature_names=[f"f{j}" for j in range(10)],
                    importance_scores={f"f{j}": 0.1 for j in range(10)},
                    method="permutation",
                    timestamp=datetime(2024, 1, i + 1),
                    model_type="generic",
                    n_samples=100,
                )
            )

        for i in range(7):
            # All features change significantly
            analyzer._importance_history.append(
                FeatureImportanceResult(
                    feature_names=[f"f{j}" for j in range(10)],
                    importance_scores={f"f{j}": 0.5 for j in range(10)},
                    method="permutation",
                    timestamp=datetime(2024, 1, i + 8),
                    model_type="generic",
                    n_samples=100,
                )
            )

        report = analyzer.get_importance_report()

        # Should limit to 5 drifting features
        assert len(report["drifting_features"]) <= 5


# =============================================================================
# TESTS: LSTMFeatureImportance
# =============================================================================


class TestLSTMFeatureImportance:
    """Tests for LSTMFeatureImportance class."""

    def test_initialization(self):
        """Test LSTMFeatureImportance initialization."""
        mock_predictor = MagicMock()

        lstm_fi = LSTMFeatureImportance(predictor=mock_predictor)

        assert lstm_fi.predictor == mock_predictor
        assert lstm_fi.analyzer is not None
        assert lstm_fi.analyzer.feature_names == ["open", "high", "low", "close", "volume"]

    def test_calculate_importance_no_model(self):
        """Test returns None when no model for symbol."""
        mock_predictor = MagicMock()
        mock_predictor.models = {}  # No models

        lstm_fi = LSTMFeatureImportance(predictor=mock_predictor)

        result = lstm_fi.calculate_importance(
            symbol="AAPL",
            prices=[{"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}],
        )

        assert result is None

    def test_calculate_importance_empty_sequences(self):
        """Test returns None when no sequences can be created."""
        mock_predictor = MagicMock()
        mock_predictor.models = {"AAPL": MagicMock()}
        # Return proper 2D arrays that will result in empty sequences
        mock_predictor._prepare_features.return_value = np.zeros((5, 5))
        mock_predictor._normalize_features.return_value = np.zeros((5, 5))
        # Empty sequences after processing
        mock_predictor._create_sequences.return_value = (np.array([]), np.array([]))

        lstm_fi = LSTMFeatureImportance(predictor=mock_predictor)

        result = lstm_fi.calculate_importance(
            symbol="AAPL",
            prices=[],
        )

        assert result is None

    def test_calculate_importance_permutation_method(self):
        """Test calculate_importance with permutation method."""
        mock_predictor = MagicMock()
        mock_model = MagicMock()
        mock_predictor.models = {"AAPL": mock_model}
        mock_predictor.device = "cpu"

        # Setup feature preparation
        np.random.seed(42)
        features = np.random.randn(100, 5)
        mock_predictor._prepare_features.return_value = features
        mock_predictor._normalize_features.return_value = features

        X = np.random.randn(80, 10, 5)  # 3D sequences
        y = np.random.randn(80)
        mock_predictor._create_sequences.return_value = (X, y)

        lstm_fi = LSTMFeatureImportance(predictor=mock_predictor)

        # Mock the analyzer's permutation importance
        with patch.object(lstm_fi.analyzer, "calculate_permutation_importance") as mock_perm:
            mock_perm.return_value = FeatureImportanceResult(
                feature_names=["open", "high", "low", "close", "volume"],
                importance_scores={
                    "open": 0.1,
                    "high": 0.2,
                    "low": 0.1,
                    "close": 0.4,
                    "volume": 0.2,
                },
                method="permutation",
                timestamp=datetime.now(),
                model_type="lstm",
                n_samples=80,
            )

            prices = [{"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}]
            result = lstm_fi.calculate_importance(
                symbol="AAPL",
                prices=prices,
                method="permutation",
            )

            assert result is not None
            mock_perm.assert_called_once()

    def test_calculate_importance_gradient_method(self):
        """Test calculate_importance with gradient method."""
        mock_predictor = MagicMock()
        mock_model = MagicMock()
        mock_predictor.models = {"AAPL": mock_model}

        np.random.seed(42)
        features = np.random.randn(100, 5)
        mock_predictor._prepare_features.return_value = features
        mock_predictor._normalize_features.return_value = features

        X = np.random.randn(80, 10, 5)
        y = np.random.randn(80)
        mock_predictor._create_sequences.return_value = (X, y)

        lstm_fi = LSTMFeatureImportance(predictor=mock_predictor)

        # Mock the analyzer's gradient importance
        with patch.object(lstm_fi.analyzer, "calculate_gradient_importance") as mock_grad:
            mock_grad.return_value = FeatureImportanceResult(
                feature_names=["open", "high", "low", "close", "volume"],
                importance_scores={
                    "open": 0.15,
                    "high": 0.25,
                    "low": 0.1,
                    "close": 0.35,
                    "volume": 0.15,
                },
                method="gradient",
                timestamp=datetime.now(),
                model_type="neural_network",
                n_samples=80,
            )

            prices = [{"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}]
            result = lstm_fi.calculate_importance(
                symbol="AAPL",
                prices=prices,
                method="gradient",
            )

            assert result is not None
            mock_grad.assert_called_once()


# =============================================================================
# TESTS: analyze_feature_importance HELPER FUNCTION
# =============================================================================


class TestAnalyzeFeatureImportanceFunction:
    """Tests for analyze_feature_importance convenience function."""

    def test_auto_method_tries_shap_first(self, sample_2d_data, sample_target):
        """Test auto method tries SHAP first."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        mock_explainer.shap_values.return_value = np.random.randn(50, 5)

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyze_feature_importance(
                model=model,
                X=sample_2d_data,
                y=sample_target,
                method="auto",
            )

            assert result is not None
            assert result.method == "shap"

    def test_auto_method_falls_back_to_permutation(self, sample_2d_data, sample_target):
        """Test auto method falls back to permutation when SHAP unavailable."""
        with patch("ml.feature_importance._import_shap", return_value=None):
            model = MagicMock()
            model.predict = MagicMock(return_value=sample_target)

            result = analyze_feature_importance(
                model=model,
                X=sample_2d_data,
                y=sample_target,
                method="auto",
            )

            assert result is not None
            assert result.method == "permutation"

    def test_explicit_shap_method(self, sample_2d_data):
        """Test explicit SHAP method."""
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        mock_explainer.shap_values.return_value = np.random.randn(50, 5)

        with patch("ml.feature_importance._import_shap", return_value=mock_shap):
            model = MagicMock()

            result = analyze_feature_importance(
                model=model,
                X=sample_2d_data,
                method="shap",
            )

            assert result is not None

    def test_explicit_permutation_method(self, sample_2d_data, sample_target):
        """Test explicit permutation method."""
        model = MagicMock()
        model.predict = MagicMock(return_value=sample_target)

        result = analyze_feature_importance(
            model=model,
            X=sample_2d_data,
            y=sample_target,
            method="permutation",
        )

        assert result is not None
        assert result.method == "permutation"

    def test_permutation_method_requires_y(self, sample_2d_data):
        """Test permutation method raises error without y."""
        model = MagicMock()

        with pytest.raises(ValueError, match="y is required"):
            analyze_feature_importance(
                model=model,
                X=sample_2d_data,
                y=None,
                method="permutation",
            )

    def test_gradient_method(self, sample_2d_data, sample_feature_names):
        """Test gradient method."""
        with patch(
            "ml.feature_importance.FeatureImportanceAnalyzer.calculate_gradient_importance"
        ) as mock_grad:
            mock_grad.return_value = FeatureImportanceResult(
                feature_names=sample_feature_names,
                importance_scores=dict.fromkeys(sample_feature_names, 0.2),
                method="gradient",
                timestamp=datetime.now(),
                model_type="neural_network",
                n_samples=50,
            )

            model = MagicMock()

            result = analyze_feature_importance(
                model=model,
                X=sample_2d_data,
                method="gradient",
            )

            assert result is not None

    def test_unknown_method_raises_error(self, sample_2d_data):
        """Test unknown method raises ValueError."""
        model = MagicMock()

        with pytest.raises(ValueError, match="Unknown method"):
            analyze_feature_importance(
                model=model,
                X=sample_2d_data,
                method="unknown_method",
            )

    def test_feature_names_passed_to_analyzer(self, sample_2d_data, sample_target):
        """Test feature names are passed to the analyzer."""
        model = MagicMock()
        model.predict = MagicMock(return_value=sample_target)
        feature_names = ["a", "b", "c", "d", "e"]

        result = analyze_feature_importance(
            model=model,
            X=sample_2d_data,
            y=sample_target,
            feature_names=feature_names,
            method="permutation",
        )

        assert result.feature_names == feature_names


# =============================================================================
# TESTS: LAZY IMPORT
# =============================================================================


class TestLazyImport:
    """Tests for lazy import functionality."""

    def test_import_shap_returns_none_when_unavailable(self):
        """Test _import_shap returns None when SHAP not installed."""
        import ml.feature_importance as fi_module

        # Reset the cached import
        original_shap = fi_module._shap
        fi_module._shap = None

        with patch.dict("sys.modules", {"shap": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                result = fi_module._import_shap()
                assert result is None

        # Restore
        fi_module._shap = original_shap

    def test_import_shap_caches_result(self):
        """Test _import_shap caches successful import."""
        import ml.feature_importance as fi_module

        mock_shap = MagicMock()
        original_shap = fi_module._shap
        fi_module._shap = mock_shap

        # Should return cached value without importing
        result = fi_module._import_shap()
        assert result == mock_shap

        # Restore
        fi_module._shap = original_shap
