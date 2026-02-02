"""
Hyperparameter Optimization using Optuna

Implements Bayesian optimization for finding optimal hyperparameters
for trading ML models (LSTM, DQN).

Features:
- Bayesian optimization via Tree-structured Parzen Estimator (TPE)
- Walk-forward cross-validation as optimization objective
- Regime-aware hyperparameter selection
- Early stopping for unpromising trials
- Persistent study storage for resumable optimization

Why Bayesian Optimization:
- Grid search is exponentially expensive
- Random search misses optimal regions
- Bayesian methods learn from previous trials
- Much more efficient for expensive objectives (ML training)
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import for optuna
_optuna = None


def _import_optuna():
    """Lazy import optuna to reduce startup time."""
    global _optuna
    if _optuna is None:
        try:
            import optuna

            # Suppress optuna logging noise
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            _optuna = optuna
            logger.debug("Optuna imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import optuna: {e}")
            raise ImportError(
                "optuna is required for hyperparameter optimization. "
                "Install with: pip install optuna"
            ) from e
    return _optuna


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""

    best_params: Dict[str, Any]
    best_value: float  # Best objective value (e.g., OOS Sharpe)
    n_trials: int
    study_name: str
    optimization_time_seconds: float
    all_trials: List[Dict[str, Any]]
    parameter_importance: Dict[str, float]


@dataclass
class HyperparameterSpace:
    """Defines the search space for a hyperparameter."""

    name: str
    param_type: str  # 'int', 'float', 'categorical', 'loguniform'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False  # For loguniform sampling
    step: Optional[float] = None  # For discrete steps


# Default search spaces for different model types
LSTM_SEARCH_SPACE = [
    HyperparameterSpace("hidden_size", "int", low=32, high=256, step=32),
    HyperparameterSpace("num_layers", "int", low=1, high=4),
    HyperparameterSpace("dropout", "float", low=0.1, high=0.5),
    HyperparameterSpace("learning_rate", "float", low=1e-4, high=1e-2, log=True),
    HyperparameterSpace("batch_size", "categorical", choices=[16, 32, 64, 128]),
    HyperparameterSpace("sequence_length", "categorical", choices=[30, 60, 90, 120]),
]

DQN_SEARCH_SPACE = [
    HyperparameterSpace("hidden_size", "int", low=32, high=256, step=32),
    HyperparameterSpace("num_layers", "int", low=1, high=3),
    HyperparameterSpace("learning_rate", "float", low=1e-5, high=1e-3, log=True),
    HyperparameterSpace("gamma", "float", low=0.9, high=0.999),
    HyperparameterSpace("epsilon_decay", "float", low=0.99, high=0.9999),
    HyperparameterSpace("batch_size", "categorical", choices=[32, 64, 128]),
    HyperparameterSpace("buffer_size", "categorical", choices=[5000, 10000, 50000]),
]


class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimizer using Optuna.

    Usage:
        optimizer = HyperparameterOptimizer(
            model_type='lstm',
            objective_fn=train_and_evaluate,
            search_space=LSTM_SEARCH_SPACE
        )

        result = optimizer.optimize(n_trials=100)
        best_params = result.best_params
    """

    def __init__(
        self,
        model_type: str,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: List[HyperparameterSpace] = None,
        direction: str = "maximize",  # 'maximize' for Sharpe, 'minimize' for loss
        study_name: str = None,
        storage_path: str = None,
        n_startup_trials: int = 10,
        n_warmup_steps: int = 5,
    ):
        """
        Initialize the optimizer.

        Args:
            model_type: Type of model ('lstm', 'dqn')
            objective_fn: Function that takes params dict, returns objective value
            search_space: List of HyperparameterSpace definitions
            direction: 'maximize' or 'minimize'
            study_name: Name for the Optuna study
            storage_path: Path to SQLite storage for resumable studies
            n_startup_trials: Random trials before Bayesian optimization kicks in
            n_warmup_steps: Warmup steps for pruner
        """
        self.model_type = model_type
        self.objective_fn = objective_fn
        self.direction = direction
        self.study_name = study_name or f"{model_type}_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_path = storage_path
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

        # Set default search space based on model type
        if search_space is None:
            if model_type == "lstm":
                self.search_space = LSTM_SEARCH_SPACE
            elif model_type == "dqn":
                self.search_space = DQN_SEARCH_SPACE
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            self.search_space = search_space

        self._study = None
        self._best_trial = None

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial based on search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        for hp in self.search_space:
            if hp.param_type == "int":
                params[hp.name] = trial.suggest_int(
                    hp.name, int(hp.low), int(hp.high), step=int(hp.step or 1)
                )
            elif hp.param_type == "float":
                if hp.log:
                    params[hp.name] = trial.suggest_float(
                        hp.name, hp.low, hp.high, log=True
                    )
                else:
                    params[hp.name] = trial.suggest_float(hp.name, hp.low, hp.high)
            elif hp.param_type == "categorical":
                params[hp.name] = trial.suggest_categorical(hp.name, hp.choices)
            elif hp.param_type == "loguniform":
                params[hp.name] = trial.suggest_float(
                    hp.name, hp.low, hp.high, log=True
                )

        return params

    def _create_objective(self) -> Callable:
        """
        Create Optuna objective function.

        Returns:
            Objective function for Optuna
        """

        def objective(trial):
            # Suggest hyperparameters
            params = self._suggest_params(trial)

            try:
                # Evaluate objective
                value = self.objective_fn(params)

                # Report intermediate value for pruning
                trial.report(value, step=0)

                # Check if trial should be pruned
                if trial.should_prune():
                    optuna = _import_optuna()
                    raise optuna.TrialPruned()

                return value

            except Exception as e:
                logger.warning(f"Trial failed with params {params}: {e}")
                # Return worst possible value
                return float("-inf") if self.direction == "maximize" else float("inf")

        return objective

    def optimize(
        self,
        n_trials: int = 100,
        timeout: int = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            show_progress: Show progress bar

        Returns:
            OptimizationResult with best parameters
        """
        optuna = _import_optuna()
        start_time = datetime.now()

        # Create sampler with startup trials
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials, seed=42
        )

        # Create pruner for early stopping
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.n_warmup_steps, n_warmup_steps=self.n_warmup_steps
        )

        # Create or load study
        storage = (
            f"sqlite:///{self.storage_path}" if self.storage_path else None
        )

        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # Run optimization
        objective = self._create_objective()

        self._study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress,
            gc_after_trial=True,
        )

        self._best_trial = self._study.best_trial
        optimization_time = (datetime.now() - start_time).total_seconds()

        # Calculate parameter importance
        try:
            importance = optuna.importance.get_param_importances(self._study)
        except Exception:
            importance = {}

        # Collect all trials info
        all_trials = [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state),
            }
            for t in self._study.trials
            if t.value is not None
        ]

        return OptimizationResult(
            best_params=self._best_trial.params,
            best_value=self._best_trial.value,
            n_trials=len(self._study.trials),
            study_name=self.study_name,
            optimization_time_seconds=optimization_time,
            all_trials=all_trials,
            parameter_importance=importance,
        )

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found so far."""
        if self._best_trial is None:
            raise ValueError("No optimization has been run yet")
        return self._best_trial.params

    def get_param_importance(self) -> Dict[str, float]:
        """Get parameter importance analysis."""
        if self._study is None:
            raise ValueError("No optimization has been run yet")

        optuna = _import_optuna()
        try:
            return optuna.importance.get_param_importances(self._study)
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            return {}

    def plot_optimization_history(self, output_path: str = None):
        """
        Plot optimization history.

        Args:
            output_path: Optional path to save figure
        """
        if self._study is None:
            raise ValueError("No optimization has been run yet")

        optuna = _import_optuna()
        try:
            fig = optuna.visualization.plot_optimization_history(self._study)
            if output_path:
                fig.write_html(output_path)
            return fig
        except ImportError:
            logger.warning("plotly required for visualization")
            return None

    def plot_param_importance(self, output_path: str = None):
        """
        Plot parameter importance.

        Args:
            output_path: Optional path to save figure
        """
        if self._study is None:
            raise ValueError("No optimization has been run yet")

        optuna = _import_optuna()
        try:
            fig = optuna.visualization.plot_param_importances(self._study)
            if output_path:
                fig.write_html(output_path)
            return fig
        except ImportError:
            logger.warning("plotly required for visualization")
            return None


class WalkForwardOptimizer:
    """
    Hyperparameter optimizer with walk-forward cross-validation.

    Uses walk-forward validation as the optimization objective to ensure
    parameters generalize to out-of-sample data.
    """

    def __init__(
        self,
        model_type: str,
        train_fn: Callable,
        evaluate_fn: Callable,
        n_folds: int = 5,
        train_ratio: float = 0.7,
        gap_days: int = 5,
    ):
        """
        Initialize the walk-forward optimizer.

        Args:
            model_type: Type of model ('lstm', 'dqn')
            train_fn: Function to train model: (params, train_data) -> model
            evaluate_fn: Function to evaluate: (model, test_data) -> metric
            n_folds: Number of walk-forward folds
            train_ratio: Train/test split ratio
            gap_days: Gap between train and test (embargo period)
        """
        self.model_type = model_type
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.gap_days = gap_days

        self._data = None
        self._optimizer = None

    def set_data(self, data: Any):
        """Set the data to use for optimization."""
        self._data = data

    def _walk_forward_objective(self, params: Dict[str, Any]) -> float:
        """
        Walk-forward cross-validation objective.

        Returns average OOS metric across folds.
        """
        if self._data is None:
            raise ValueError("Must call set_data first")

        # Split data into folds
        data_length = len(self._data)
        fold_size = data_length // (self.n_folds + 1)

        oos_metrics = []

        for fold in range(self.n_folds):
            # Calculate fold boundaries
            train_end = fold_size * (fold + 1)
            test_start = train_end + self.gap_days
            test_end = min(test_start + fold_size, data_length)

            if test_end <= test_start:
                continue

            # Split data
            train_data = self._data[:train_end]
            test_data = self._data[test_start:test_end]

            try:
                # Train model
                model = self.train_fn(params, train_data)

                # Evaluate on OOS data
                metric = self.evaluate_fn(model, test_data)
                oos_metrics.append(metric)

            except Exception as e:
                logger.warning(f"Fold {fold} failed: {e}")
                continue

        if not oos_metrics:
            return float("-inf")

        # Return average OOS metric
        return np.mean(oos_metrics)

    def optimize(
        self,
        data: Any,
        n_trials: int = 100,
        search_space: List[HyperparameterSpace] = None,
    ) -> OptimizationResult:
        """
        Run walk-forward hyperparameter optimization.

        Args:
            data: Data to use for training/evaluation
            n_trials: Number of optimization trials
            search_space: Optional custom search space

        Returns:
            OptimizationResult with best parameters
        """
        self.set_data(data)

        self._optimizer = HyperparameterOptimizer(
            model_type=self.model_type,
            objective_fn=self._walk_forward_objective,
            search_space=search_space,
            direction="maximize",
        )

        return self._optimizer.optimize(n_trials=n_trials)


@dataclass
class NestedCVResult:
    """Result of nested cross-validation optimization."""

    # Best parameters per outer fold
    fold_params: List[Dict[str, Any]]

    # OOS metrics per outer fold
    fold_oos_metrics: List[float]

    # Final recommended parameters (most stable)
    recommended_params: Dict[str, Any]

    # Parameter stability analysis
    parameter_stability: Dict[str, Dict[str, float]]  # param -> {mean, std, cv}

    # Summary statistics
    mean_oos_metric: float
    std_oos_metric: float

    # Warning if parameters vary too much across folds
    unstable_parameters: List[str]


class NestedCVOptimizer:
    """
    Nested Cross-Validation optimizer for unbiased hyperparameter selection.

    Implements proper nested CV to avoid overfitting during hyperparameter tuning:
    - Outer loop: K walk-forward folds for unbiased performance estimation
    - Inner loop: Hyperparameter optimization within each training fold
    - Tracks parameter stability across folds

    Why Nested CV:
    - Standard CV can overfit hyperparameters to validation set
    - Nested CV provides unbiased estimate of generalization
    - Parameter stability indicates robustness of the strategy

    Usage:
        optimizer = NestedCVOptimizer(
            model_type='lstm',
            train_fn=train_model,
            evaluate_fn=evaluate_model,
            n_outer_folds=5,
            n_inner_trials=50,
        )

        result = optimizer.optimize(data)
        print(f"Recommended params: {result.recommended_params}")
        print(f"OOS Sharpe: {result.mean_oos_metric:.3f} +/- {result.std_oos_metric:.3f}")
        print(f"Unstable parameters: {result.unstable_parameters}")
    """

    def __init__(
        self,
        model_type: str,
        train_fn: Callable,
        evaluate_fn: Callable,
        search_space: List[HyperparameterSpace] = None,
        n_outer_folds: int = 5,
        n_inner_trials: int = 50,
        inner_cv_folds: int = 3,
        gap_days: int = 5,
        stability_threshold: float = 0.2,  # CV > 20% = unstable
    ):
        """
        Initialize the nested CV optimizer.

        Args:
            model_type: Type of model ('lstm', 'dqn')
            train_fn: Function to train model: (params, train_data) -> model
            evaluate_fn: Function to evaluate: (model, test_data) -> metric
            search_space: Hyperparameter search space
            n_outer_folds: Number of outer walk-forward folds
            n_inner_trials: Number of Optuna trials per inner optimization
            inner_cv_folds: Number of folds for inner CV
            gap_days: Embargo period between train and test
            stability_threshold: Coefficient of variation threshold for stable params
        """
        self.model_type = model_type
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.n_outer_folds = n_outer_folds
        self.n_inner_trials = n_inner_trials
        self.inner_cv_folds = inner_cv_folds
        self.gap_days = gap_days
        self.stability_threshold = stability_threshold

        # Set default search space based on model type
        if search_space is None:
            if model_type == "lstm":
                self.search_space = LSTM_SEARCH_SPACE
            elif model_type == "dqn":
                self.search_space = DQN_SEARCH_SPACE
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            self.search_space = search_space

    def _inner_cv_objective(
        self,
        params: Dict[str, Any],
        train_data: Any,
    ) -> float:
        """
        Inner cross-validation objective for hyperparameter tuning.

        Args:
            params: Hyperparameters to evaluate
            train_data: Training data (excluding outer test set)

        Returns:
            Average validation metric across inner folds
        """
        data_length = len(train_data)
        fold_size = data_length // (self.inner_cv_folds + 1)

        val_metrics = []

        for fold in range(self.inner_cv_folds):
            # Split within training data
            inner_train_end = fold_size * (fold + 1)
            inner_val_start = inner_train_end + self.gap_days
            inner_val_end = min(inner_val_start + fold_size, data_length)

            if inner_val_end <= inner_val_start:
                continue

            inner_train = train_data[:inner_train_end]
            inner_val = train_data[inner_val_start:inner_val_end]

            try:
                model = self.train_fn(params, inner_train)
                metric = self.evaluate_fn(model, inner_val)
                val_metrics.append(metric)
            except Exception as e:
                logger.warning(f"Inner fold {fold} failed: {e}")
                continue

        if not val_metrics:
            return float("-inf")

        return np.mean(val_metrics)

    def _optimize_inner(
        self,
        train_data: Any,
        fold_idx: int,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run inner hyperparameter optimization on a training fold.

        Args:
            train_data: Training data for this outer fold
            fold_idx: Index of the outer fold (for naming)

        Returns:
            Tuple of (best_params, best_value)
        """

        def inner_objective(params):
            return self._inner_cv_objective(params, train_data)

        optimizer = HyperparameterOptimizer(
            model_type=self.model_type,
            objective_fn=inner_objective,
            search_space=self.search_space,
            direction="maximize",
            study_name=f"{self.model_type}_nested_outer{fold_idx}",
        )

        result = optimizer.optimize(
            n_trials=self.n_inner_trials, show_progress=False
        )

        return result.best_params, result.best_value

    def _analyze_parameter_stability(
        self,
        fold_params: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """
        Analyze stability of parameters across outer folds.

        Args:
            fold_params: Best parameters from each outer fold

        Returns:
            Tuple of (stability_dict, unstable_params_list)
        """
        stability = {}
        unstable = []

        # Get all parameter names
        if not fold_params:
            return stability, unstable

        param_names = fold_params[0].keys()

        for param in param_names:
            values = []
            for fp in fold_params:
                v = fp.get(param)
                if v is not None:
                    # Handle categorical params
                    if isinstance(v, (int, float)):
                        values.append(float(v))
                    else:
                        # For categorical, we can't compute CV meaningfully
                        # Just check if all values are the same
                        values.append(hash(str(v)))

            if len(values) < 2:
                continue

            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)

            # Coefficient of variation (relative std dev)
            cv = std_val / abs(mean_val) if mean_val != 0 else 0.0

            stability[param] = {
                "mean": float(mean_val) if isinstance(fold_params[0].get(param), (int, float)) else None,
                "std": float(std_val) if isinstance(fold_params[0].get(param), (int, float)) else None,
                "cv": float(cv),
            }

            if cv > self.stability_threshold and isinstance(fold_params[0].get(param), (int, float)):
                unstable.append(param)

        return stability, unstable

    def _recommend_parameters(
        self,
        fold_params: List[Dict[str, Any]],
        stability: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Recommend final parameters based on fold results.

        Strategy:
        - For stable numeric params: use median across folds
        - For unstable numeric params: use most conservative value
        - For categorical params: use mode (most common)

        Args:
            fold_params: Best parameters from each outer fold
            stability: Stability analysis

        Returns:
            Recommended parameter dictionary
        """
        if not fold_params:
            return {}

        recommended = {}
        param_names = fold_params[0].keys()

        for param in param_names:
            values = [fp.get(param) for fp in fold_params if fp.get(param) is not None]

            if not values:
                continue

            if isinstance(values[0], (int, float)):
                # Numeric parameter - use median
                median_val = np.median(values)

                # If integer parameter, round
                for hp in self.search_space:
                    if hp.name == param and hp.param_type == "int":
                        median_val = int(round(median_val))
                        break

                recommended[param] = median_val
            else:
                # Categorical parameter - use mode
                from collections import Counter
                counter = Counter(values)
                recommended[param] = counter.most_common(1)[0][0]

        return recommended

    def optimize(
        self,
        data: Any,
    ) -> NestedCVResult:
        """
        Run nested cross-validation optimization.

        Args:
            data: Full dataset to use

        Returns:
            NestedCVResult with comprehensive results
        """
        data_length = len(data)
        fold_size = data_length // (self.n_outer_folds + 1)

        fold_params = []
        fold_oos_metrics = []

        logger.info(f"Starting nested CV with {self.n_outer_folds} outer folds")

        for outer_fold in range(self.n_outer_folds):
            logger.info(f"Outer fold {outer_fold + 1}/{self.n_outer_folds}")

            # Calculate outer fold boundaries
            train_end = fold_size * (outer_fold + 1)
            test_start = train_end + self.gap_days
            test_end = min(test_start + fold_size, data_length)

            if test_end <= test_start:
                logger.warning(f"Skipping fold {outer_fold}: insufficient data")
                continue

            train_data = data[:train_end]
            test_data = data[test_start:test_end]

            # Inner optimization: find best params using only train_data
            logger.info(f"  Inner optimization ({self.n_inner_trials} trials)...")
            best_params, inner_val = self._optimize_inner(train_data, outer_fold)

            logger.info(
                f"  Best inner params: {best_params} (inner val: {inner_val:.4f})"
            )

            # Outer evaluation: train with best params on full training fold,
            # evaluate on held-out test fold
            try:
                model = self.train_fn(best_params, train_data)
                oos_metric = self.evaluate_fn(model, test_data)

                fold_params.append(best_params)
                fold_oos_metrics.append(oos_metric)

                logger.info(f"  OOS metric: {oos_metric:.4f}")

            except Exception as e:
                logger.warning(f"  Outer evaluation failed: {e}")
                continue

        if not fold_oos_metrics:
            raise ValueError("All outer folds failed")

        # Analyze parameter stability
        stability, unstable = self._analyze_parameter_stability(fold_params)

        if unstable:
            logger.warning(
                f"Unstable parameters (CV > {self.stability_threshold}): {unstable}"
            )

        # Recommend final parameters
        recommended = self._recommend_parameters(fold_params, stability)

        return NestedCVResult(
            fold_params=fold_params,
            fold_oos_metrics=fold_oos_metrics,
            recommended_params=recommended,
            parameter_stability=stability,
            mean_oos_metric=float(np.mean(fold_oos_metrics)),
            std_oos_metric=float(np.std(fold_oos_metrics)),
            unstable_parameters=unstable,
        )


class RegimeAwareOptimizer:
    """
    Hyperparameter optimizer that finds different optimal params per market regime.

    Different market conditions may require different model configurations.
    """

    def __init__(
        self,
        model_type: str,
        objective_fn: Callable,
        regime_classifier: Callable,
    ):
        """
        Initialize the regime-aware optimizer.

        Args:
            model_type: Type of model
            objective_fn: Objective function (params, regime) -> metric
            regime_classifier: Function to classify market regime from data
        """
        self.model_type = model_type
        self.objective_fn = objective_fn
        self.regime_classifier = regime_classifier

        self._regime_params: Dict[str, Dict[str, Any]] = {}

    def optimize_for_regimes(
        self,
        data: Any,
        regimes: List[str] = None,
        n_trials_per_regime: int = 50,
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize hyperparameters separately for each market regime.

        Args:
            data: Historical data
            regimes: List of regimes to optimize (default: all detected)
            n_trials_per_regime: Trials per regime

        Returns:
            Dictionary of regime -> OptimizationResult
        """
        if regimes is None:
            regimes = ["bull", "bear", "sideways", "volatile"]

        results = {}

        for regime in regimes:
            logger.info(f"Optimizing for {regime} regime...")

            def regime_objective(params):
                return self.objective_fn(params, regime)

            optimizer = HyperparameterOptimizer(
                model_type=self.model_type,
                objective_fn=regime_objective,
                direction="maximize",
                study_name=f"{self.model_type}_{regime}",
            )

            result = optimizer.optimize(n_trials=n_trials_per_regime)
            results[regime] = result
            self._regime_params[regime] = result.best_params

            logger.info(
                f"Best params for {regime}: {result.best_params} "
                f"(value: {result.best_value:.4f})"
            )

        return results

    def get_params_for_regime(self, regime: str) -> Dict[str, Any]:
        """Get optimized parameters for a specific regime."""
        if regime not in self._regime_params:
            raise ValueError(f"No optimized params for regime: {regime}")
        return self._regime_params[regime]


def optimize_lstm_hyperparameters(
    symbol: str,
    price_data: List[dict],
    n_trials: int = 100,
    validation_split: float = 0.2,
) -> OptimizationResult:
    """
    Convenience function to optimize LSTM hyperparameters.

    Args:
        symbol: Stock symbol
        price_data: Historical price data
        n_trials: Number of optimization trials
        validation_split: Fraction for validation

    Returns:
        OptimizationResult with best LSTM parameters
    """
    from ml.lstm_predictor import LSTMPredictor

    def objective(params: Dict[str, Any]) -> float:
        """Train and evaluate LSTM with given params."""
        predictor = LSTMPredictor(
            sequence_length=params.get("sequence_length", 60),
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.2),
        )

        # Split data
        split_idx = int(len(price_data) * (1 - validation_split))
        train_data = price_data[:split_idx]
        val_data = price_data[split_idx:]

        try:
            # Train
            metrics = predictor.train(
                symbol,
                train_data,
                epochs=params.get("epochs", 50),
                learning_rate=params.get("learning_rate", 0.001),
                batch_size=params.get("batch_size", 32),
            )

            # Return negative validation loss (we maximize)
            return -metrics.best_val_loss

        except Exception as e:
            logger.warning(f"Training failed: {e}")
            return float("-inf")

    optimizer = HyperparameterOptimizer(
        model_type="lstm",
        objective_fn=objective,
        direction="maximize",  # Maximizing negative loss = minimizing loss
    )

    return optimizer.optimize(n_trials=n_trials)
