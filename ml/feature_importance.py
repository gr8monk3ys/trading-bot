"""
Feature Importance Analysis

Provides multiple methods for understanding which features drive model predictions:

1. SHAP Values (SHapley Additive exPlanations)
   - Gold standard for model interpretability
   - Consistent, locally accurate attributions
   - Works for any model type

2. Permutation Importance
   - Model-agnostic, simple to interpret
   - Measures feature importance by shuffling
   - Good for detecting data leakage

3. Importance Drift Detection
   - Track feature importance over time
   - Alert when importance distribution shifts
   - Critical for detecting regime changes

Why This Matters for Trading:
- Understand WHY model makes predictions
- Detect when model relies on spurious correlations
- Identify features that stop being predictive (alpha decay)
- Regulatory compliance (explainability requirements)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_shap = None


def _import_shap():
    """Lazy import SHAP to reduce startup time."""
    global _shap
    if _shap is None:
        try:
            import shap

            _shap = shap
            logger.debug("SHAP imported successfully")
        except ImportError as e:
            logger.warning(f"SHAP not available: {e}")
            return None
    return _shap


@dataclass
class FeatureImportanceResult:
    """Result of feature importance analysis."""

    feature_names: List[str]
    importance_scores: Dict[str, float]  # Feature -> importance
    method: str  # 'shap', 'permutation', etc.
    timestamp: datetime
    model_type: str
    n_samples: int

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(
            self.importance_scores.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return sorted_features[:n]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(
            [
                {"feature": k, "importance": v}
                for k, v in self.importance_scores.items()
            ]
        ).sort_values("importance", key=abs, ascending=False)


@dataclass
class ImportanceDriftResult:
    """Result of feature importance drift analysis."""

    feature_name: str
    historical_importance: float
    recent_importance: float
    change_pct: float
    is_significant: bool
    p_value: Optional[float] = None


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for trading ML models.

    Supports multiple methods:
    - SHAP values for deep explanations
    - Permutation importance for quick analysis
    - Gradient-based importance for neural networks
    """

    def __init__(
        self,
        feature_names: List[str] = None,
        background_samples: int = 100,
    ):
        """
        Initialize the analyzer.

        Args:
            feature_names: Names of input features
            background_samples: Samples for SHAP background dataset
        """
        self.feature_names = feature_names or []
        self.background_samples = background_samples
        self._importance_history: List[FeatureImportanceResult] = []

    def calculate_shap_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str] = None,
        model_type: str = "tree",
    ) -> Optional[FeatureImportanceResult]:
        """
        Calculate SHAP values for feature importance.

        SHAP provides the most rigorous importance measure:
        - Based on game-theoretic Shapley values
        - Consistent: if feature contributes more, it gets higher importance
        - Local: can explain individual predictions

        Args:
            model: Trained model (sklearn, pytorch, etc.)
            X: Input data array
            feature_names: Names of features
            model_type: 'tree', 'deep', 'kernel', or 'linear'

        Returns:
            FeatureImportanceResult or None if SHAP unavailable
        """
        shap = _import_shap()
        if shap is None:
            logger.warning("SHAP not available, falling back to permutation importance")
            return None

        feature_names = feature_names or self.feature_names
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(X.shape[-1])]

        try:
            # Select appropriate explainer
            if model_type == "tree":
                # For tree-based models (XGBoost, LightGBM, RandomForest)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            elif model_type == "deep":
                # For neural networks
                background = X[np.random.choice(len(X), min(self.background_samples, len(X)), replace=False)]
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(X)
            elif model_type == "kernel":
                # Model-agnostic (slower but works for anything)
                background = shap.sample(X, min(self.background_samples, len(X)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X, nsamples=100)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None

            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Handle 3D arrays (sequence models)
            if len(shap_values.shape) == 3:
                # Average over sequence dimension
                shap_values = np.mean(shap_values, axis=1)

            # Calculate mean absolute SHAP value per feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            # Handle case where feature count doesn't match
            if len(mean_abs_shap) != len(feature_names):
                logger.warning(
                    f"Feature count mismatch: {len(mean_abs_shap)} vs {len(feature_names)}"
                )
                feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

            importance_scores = dict(zip(feature_names, mean_abs_shap, strict=False))

            result = FeatureImportanceResult(
                feature_names=feature_names,
                importance_scores=importance_scores,
                method="shap",
                timestamp=datetime.now(),
                model_type=model_type,
                n_samples=len(X),
            )

            self._importance_history.append(result)
            return result

        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            return None

    def calculate_permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        n_repeats: int = 10,
        scoring_fn: Callable = None,
    ) -> FeatureImportanceResult:
        """
        Calculate permutation importance.

        Permutation importance measures how much model performance
        degrades when a feature is randomly shuffled.

        Advantages:
        - Works for ANY model (model-agnostic)
        - Simple to interpret
        - Good at detecting data leakage (leaky features will have high importance)

        Args:
            model: Trained model with predict() method
            X: Input data
            y: True labels/targets
            feature_names: Names of features
            n_repeats: Number of permutation repeats
            scoring_fn: Custom scoring function (default: negative MSE)

        Returns:
            FeatureImportanceResult
        """
        feature_names = feature_names or self.feature_names
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(X.shape[-1])]

        if scoring_fn is None:
            def scoring_fn(y_true, y_pred):
                return -np.mean((y_true - y_pred) ** 2)  # Negative MSE

        # Calculate baseline score
        baseline_pred = model.predict(X)
        if len(baseline_pred.shape) > 1:
            baseline_pred = baseline_pred.flatten()
        baseline_score = scoring_fn(y, baseline_pred)

        importance_scores = {}

        # Handle 3D arrays (sequence models)
        if len(X.shape) == 3:
            # For LSTM: (samples, sequence, features)
            for feature_idx in range(X.shape[-1]):
                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                score_drops = []

                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    # Shuffle this feature across all samples and timesteps
                    perm_idx = np.random.permutation(len(X))
                    X_permuted[:, :, feature_idx] = X_permuted[perm_idx, :, feature_idx]

                    permuted_pred = model.predict(X_permuted)
                    if len(permuted_pred.shape) > 1:
                        permuted_pred = permuted_pred.flatten()
                    permuted_score = scoring_fn(y, permuted_pred)
                    score_drops.append(baseline_score - permuted_score)

                importance_scores[feature_name] = np.mean(score_drops)
        else:
            # For 2D arrays
            for feature_idx in range(X.shape[-1]):
                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                score_drops = []

                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

                    permuted_pred = model.predict(X_permuted)
                    if len(permuted_pred.shape) > 1:
                        permuted_pred = permuted_pred.flatten()
                    permuted_score = scoring_fn(y, permuted_pred)
                    score_drops.append(baseline_score - permuted_score)

                importance_scores[feature_name] = np.mean(score_drops)

        result = FeatureImportanceResult(
            feature_names=list(importance_scores.keys()),
            importance_scores=importance_scores,
            method="permutation",
            timestamp=datetime.now(),
            model_type="generic",
            n_samples=len(X),
        )

        self._importance_history.append(result)
        return result

    def calculate_gradient_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str] = None,
    ) -> Optional[FeatureImportanceResult]:
        """
        Calculate gradient-based importance for neural networks.

        Uses integrated gradients or simple gradient magnitude.

        Args:
            model: PyTorch model
            X: Input data as numpy array
            feature_names: Names of features

        Returns:
            FeatureImportanceResult or None
        """
        try:
            from ml.torch_utils import import_torch
            torch, _, _, _ = import_torch()
        except ImportError:
            logger.warning("PyTorch not available for gradient importance")
            return None

        feature_names = feature_names or self.feature_names

        try:
            # Convert to tensor with gradients
            X_tensor = torch.FloatTensor(X).requires_grad_(True)

            # Forward pass
            model.eval()
            output = model(X_tensor)

            # Backward pass
            output.sum().backward()

            # Get gradients
            gradients = X_tensor.grad.numpy()

            # Handle 3D arrays
            if len(gradients.shape) == 3:
                # Average over samples and sequence
                importance = np.mean(np.abs(gradients), axis=(0, 1))
            else:
                importance = np.mean(np.abs(gradients), axis=0)

            if not feature_names or len(feature_names) != len(importance):
                feature_names = [f"feature_{i}" for i in range(len(importance))]

            importance_scores = dict(zip(feature_names, importance, strict=False))

            return FeatureImportanceResult(
                feature_names=feature_names,
                importance_scores=importance_scores,
                method="gradient",
                timestamp=datetime.now(),
                model_type="neural_network",
                n_samples=len(X),
            )

        except Exception as e:
            logger.error(f"Gradient importance calculation failed: {e}")
            return None

    def detect_importance_drift(
        self,
        window_size: int = 5,
        significance_threshold: float = 0.3,
    ) -> List[ImportanceDriftResult]:
        """
        Detect drift in feature importance over time.

        Importance drift can indicate:
        - Market regime change
        - Feature losing predictive power
        - Model degradation

        Args:
            window_size: Number of recent observations for comparison
            significance_threshold: Change threshold for significance

        Returns:
            List of ImportanceDriftResult for drifting features
        """
        if len(self._importance_history) < window_size * 2:
            logger.warning("Insufficient history for drift detection")
            return []

        # Get historical and recent importance
        historical = self._importance_history[:-window_size]
        recent = self._importance_history[-window_size:]

        # Calculate average importance for each period
        historical_avg = self._average_importance(historical)
        recent_avg = self._average_importance(recent)

        drift_results = []

        all_features = set(historical_avg.keys()) | set(recent_avg.keys())

        for feature in all_features:
            hist_imp = historical_avg.get(feature, 0.0)
            recent_imp = recent_avg.get(feature, 0.0)

            if hist_imp == 0:
                change_pct = float("inf") if recent_imp != 0 else 0
            else:
                change_pct = (recent_imp - hist_imp) / abs(hist_imp)

            is_significant = abs(change_pct) > significance_threshold

            drift_results.append(
                ImportanceDriftResult(
                    feature_name=feature,
                    historical_importance=hist_imp,
                    recent_importance=recent_imp,
                    change_pct=change_pct,
                    is_significant=is_significant,
                )
            )

        # Sort by significance
        drift_results.sort(key=lambda x: abs(x.change_pct), reverse=True)

        return drift_results

    def _average_importance(
        self, results: List[FeatureImportanceResult]
    ) -> Dict[str, float]:
        """Calculate average importance across multiple results."""
        if not results:
            return {}

        feature_totals: Dict[str, List[float]] = {}

        for result in results:
            for feature, importance in result.importance_scores.items():
                if feature not in feature_totals:
                    feature_totals[feature] = []
                feature_totals[feature].append(importance)

        return {f: np.mean(scores) for f, scores in feature_totals.items()}

    def get_importance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive feature importance report.

        Returns:
            Dictionary with importance analysis
        """
        if not self._importance_history:
            return {"error": "No importance calculations performed yet"}

        latest = self._importance_history[-1]
        top_features = latest.get_top_features(10)

        # Drift analysis
        drift = self.detect_importance_drift()
        significant_drifts = [d for d in drift if d.is_significant]

        return {
            "latest_method": latest.method,
            "n_features": len(latest.feature_names),
            "top_features": [
                {"feature": f, "importance": float(i)} for f, i in top_features
            ],
            "n_importance_records": len(self._importance_history),
            "drift_detected": len(significant_drifts) > 0,
            "drifting_features": [
                {
                    "feature": d.feature_name,
                    "change_pct": f"{d.change_pct:.1%}",
                }
                for d in significant_drifts[:5]
            ],
        }


class LSTMFeatureImportance:
    """
    Specialized feature importance for LSTM models.

    Handles sequence data and temporal dependencies.
    """

    def __init__(self, predictor):
        """
        Initialize with an LSTMPredictor instance.

        Args:
            predictor: Trained LSTMPredictor
        """
        self.predictor = predictor
        self.analyzer = FeatureImportanceAnalyzer(
            feature_names=["open", "high", "low", "close", "volume"]
        )

    def calculate_importance(
        self,
        symbol: str,
        prices: List[dict],
        method: str = "permutation",
    ) -> Optional[FeatureImportanceResult]:
        """
        Calculate feature importance for LSTM prediction.

        Args:
            symbol: Stock symbol
            prices: Price history
            method: 'permutation' or 'gradient'

        Returns:
            FeatureImportanceResult
        """
        if symbol not in self.predictor.models:
            logger.warning(f"No model for {symbol}")
            return None

        # Prepare data
        features = self.predictor._prepare_features(prices)
        normalized = self.predictor._normalize_features(features, symbol, fit=False)

        # Create sequences
        X, y = self.predictor._create_sequences(normalized, normalized[:, 3])

        if len(X) == 0:
            return None

        # Get model
        model = self.predictor.models[symbol]

        if method == "gradient":
            return self.analyzer.calculate_gradient_importance(
                model, X, ["open", "high", "low", "close", "volume"]
            )
        else:
            # Permutation importance with wrapper for LSTM
            class ModelWrapper:
                def __init__(self, lstm_model, device):
                    self.model = lstm_model
                    self.device = device

                def predict(self, X):
                    from ml.torch_utils import import_torch
                    torch, _, _, _ = import_torch()
                    self.model.eval()
                    with torch.no_grad():
                        X_t = torch.FloatTensor(X).to(self.device)
                        return self.model(X_t).cpu().numpy()

            wrapper = ModelWrapper(model, self.predictor.device)

            return self.analyzer.calculate_permutation_importance(
                wrapper, X, y, ["open", "high", "low", "close", "volume"]
            )


def analyze_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray = None,
    feature_names: List[str] = None,
    method: str = "auto",
) -> FeatureImportanceResult:
    """
    Convenience function to analyze feature importance.

    Args:
        model: Trained model
        X: Input data
        y: Target values (required for permutation)
        feature_names: Feature names
        method: 'shap', 'permutation', 'gradient', or 'auto'

    Returns:
        FeatureImportanceResult
    """
    analyzer = FeatureImportanceAnalyzer(feature_names=feature_names)

    if method == "auto":
        # Try SHAP first, fall back to permutation
        result = analyzer.calculate_shap_importance(model, X, feature_names)
        if result is None and y is not None:
            result = analyzer.calculate_permutation_importance(model, X, y, feature_names)
        return result

    if method == "shap":
        return analyzer.calculate_shap_importance(model, X, feature_names)
    elif method == "permutation":
        if y is None:
            raise ValueError("y is required for permutation importance")
        return analyzer.calculate_permutation_importance(model, X, y, feature_names)
    elif method == "gradient":
        return analyzer.calculate_gradient_importance(model, X, feature_names)
    else:
        raise ValueError(f"Unknown method: {method}")
