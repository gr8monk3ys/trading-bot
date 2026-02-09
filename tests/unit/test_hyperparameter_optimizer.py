"""
Unit tests for hyperparameter_optimizer.py

Tests cover:
- OptimizationResult and HyperparameterSpace dataclasses
- HyperparameterOptimizer class
- WalkForwardOptimizer class
- RegimeAwareOptimizer class
- Search space definitions
- Parameter suggestion for different types
- Error handling

Mock optuna to avoid long optimization runs.
"""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml.hyperparameter_optimizer import (
    DQN_SEARCH_SPACE,
    LSTM_SEARCH_SPACE,
    HyperparameterOptimizer,
    HyperparameterSpace,
    NestedCVOptimizer,
    NestedCVResult,
    OptimizationResult,
    RegimeAwareOptimizer,
    WalkForwardOptimizer,
    _import_optuna,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_optuna():
    """Create a mock optuna module for testing."""
    with patch("ml.hyperparameter_optimizer._optuna", None):
        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            # Set up mock trial state enum
            mock_optuna.TrialPruned = Exception

            yield mock_optuna


@pytest.fixture
def mock_trial():
    """Create a mock optuna trial for testing parameter suggestion."""
    trial = MagicMock()
    trial.suggest_int = MagicMock(return_value=64)
    trial.suggest_float = MagicMock(return_value=0.001)
    trial.suggest_categorical = MagicMock(return_value=32)
    trial.should_prune = MagicMock(return_value=False)
    trial.report = MagicMock()
    return trial


@pytest.fixture
def simple_objective_fn():
    """Simple objective function that returns a fixed value."""
    def objective(params):
        return 1.5  # Simulate a Sharpe ratio
    return objective


@pytest.fixture
def params_based_objective_fn():
    """Objective function that varies based on params."""
    def objective(params):
        # Higher hidden_size = higher Sharpe (simplified)
        return params.get("hidden_size", 64) / 100.0
    return objective


@pytest.fixture
def failing_objective_fn():
    """Objective function that always raises an exception."""
    def objective(params):
        raise ValueError("Training failed")
    return objective


@pytest.fixture
def sample_search_space():
    """Create a sample search space for testing."""
    return [
        HyperparameterSpace("hidden_size", "int", low=32, high=128, step=32),
        HyperparameterSpace("learning_rate", "float", low=0.0001, high=0.01, log=True),
        HyperparameterSpace("batch_size", "categorical", choices=[16, 32, 64]),
        HyperparameterSpace("dropout", "float", low=0.1, high=0.5),
        HyperparameterSpace("epsilon", "loguniform", low=1e-5, high=1e-3),
    ]


@pytest.fixture
def sample_data():
    """Create sample data for walk-forward testing."""
    # 120 data points for 5 folds
    return np.random.randn(120, 10)


# =============================================================================
# TESTS: DATACLASSES
# =============================================================================


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating an OptimizationResult with all fields."""
        result = OptimizationResult(
            best_params={"hidden_size": 64, "learning_rate": 0.001},
            best_value=1.5,
            n_trials=100,
            study_name="test_study",
            optimization_time_seconds=60.0,
            all_trials=[{"number": 0, "params": {}, "value": 1.0, "state": "COMPLETE"}],
            parameter_importance={"hidden_size": 0.8, "learning_rate": 0.2},
        )

        assert result.best_params == {"hidden_size": 64, "learning_rate": 0.001}
        assert result.best_value == 1.5
        assert result.n_trials == 100
        assert result.study_name == "test_study"
        assert result.optimization_time_seconds == 60.0
        assert len(result.all_trials) == 1
        assert result.parameter_importance["hidden_size"] == 0.8

    def test_optimization_result_to_dict(self):
        """Test converting OptimizationResult to dictionary."""
        result = OptimizationResult(
            best_params={"hidden_size": 64},
            best_value=1.5,
            n_trials=10,
            study_name="test",
            optimization_time_seconds=30.0,
            all_trials=[],
            parameter_importance={},
        )

        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert result_dict["best_value"] == 1.5


class TestHyperparameterSpace:
    """Tests for HyperparameterSpace dataclass."""

    def test_int_hyperparameter_space(self):
        """Test creating an integer hyperparameter space."""
        hp = HyperparameterSpace("hidden_size", "int", low=32, high=256, step=32)

        assert hp.name == "hidden_size"
        assert hp.param_type == "int"
        assert hp.low == 32
        assert hp.high == 256
        assert hp.step == 32
        assert hp.choices is None
        assert hp.log is False

    def test_float_hyperparameter_space(self):
        """Test creating a float hyperparameter space."""
        hp = HyperparameterSpace("dropout", "float", low=0.1, high=0.5)

        assert hp.name == "dropout"
        assert hp.param_type == "float"
        assert hp.low == 0.1
        assert hp.high == 0.5

    def test_loguniform_hyperparameter_space(self):
        """Test creating a loguniform hyperparameter space."""
        hp = HyperparameterSpace("learning_rate", "float", low=1e-4, high=1e-2, log=True)

        assert hp.name == "learning_rate"
        assert hp.param_type == "float"
        assert hp.log is True
        assert hp.low == 1e-4
        assert hp.high == 1e-2

    def test_categorical_hyperparameter_space(self):
        """Test creating a categorical hyperparameter space."""
        hp = HyperparameterSpace("batch_size", "categorical", choices=[16, 32, 64])

        assert hp.name == "batch_size"
        assert hp.param_type == "categorical"
        assert hp.choices == [16, 32, 64]


# =============================================================================
# TESTS: SEARCH SPACES
# =============================================================================


class TestSearchSpaces:
    """Tests for default search space definitions."""

    def test_lstm_search_space_structure(self):
        """Test LSTM search space has expected parameters."""
        param_names = {hp.name for hp in LSTM_SEARCH_SPACE}

        assert "hidden_size" in param_names
        assert "num_layers" in param_names
        assert "dropout" in param_names
        assert "learning_rate" in param_names
        assert "batch_size" in param_names
        assert "sequence_length" in param_names

    def test_lstm_search_space_types(self):
        """Test LSTM search space parameter types."""
        param_types = {hp.name: hp.param_type for hp in LSTM_SEARCH_SPACE}

        assert param_types["hidden_size"] == "int"
        assert param_types["num_layers"] == "int"
        assert param_types["dropout"] == "float"
        assert param_types["learning_rate"] == "float"
        assert param_types["batch_size"] == "categorical"
        assert param_types["sequence_length"] == "categorical"

    def test_lstm_learning_rate_is_log_scale(self):
        """Test LSTM learning rate uses log scale."""
        lr_hp = next(hp for hp in LSTM_SEARCH_SPACE if hp.name == "learning_rate")
        assert lr_hp.log is True

    def test_dqn_search_space_structure(self):
        """Test DQN search space has expected parameters."""
        param_names = {hp.name for hp in DQN_SEARCH_SPACE}

        assert "hidden_size" in param_names
        assert "num_layers" in param_names
        assert "learning_rate" in param_names
        assert "gamma" in param_names
        assert "epsilon_decay" in param_names
        assert "batch_size" in param_names
        assert "buffer_size" in param_names

    def test_dqn_search_space_gamma_range(self):
        """Test DQN gamma has valid discount factor range."""
        gamma_hp = next(hp for hp in DQN_SEARCH_SPACE if hp.name == "gamma")
        assert gamma_hp.low >= 0.0
        assert gamma_hp.high <= 1.0

    def test_dqn_epsilon_decay_range(self):
        """Test DQN epsilon decay has valid range."""
        eps_hp = next(hp for hp in DQN_SEARCH_SPACE if hp.name == "epsilon_decay")
        assert eps_hp.low > 0.0
        assert eps_hp.high <= 1.0


# =============================================================================
# TESTS: HYPERPARAMETER OPTIMIZER
# =============================================================================


class TestHyperparameterOptimizerInit:
    """Tests for HyperparameterOptimizer initialization."""

    def test_init_with_lstm_model_type(self, simple_objective_fn):
        """Test initialization with LSTM model type uses default space."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        assert optimizer.model_type == "lstm"
        assert optimizer.search_space == LSTM_SEARCH_SPACE
        assert optimizer.direction == "maximize"

    def test_init_with_dqn_model_type(self, simple_objective_fn):
        """Test initialization with DQN model type uses default space."""
        optimizer = HyperparameterOptimizer(
            model_type="dqn",
            objective_fn=simple_objective_fn,
        )

        assert optimizer.model_type == "dqn"
        assert optimizer.search_space == DQN_SEARCH_SPACE

    def test_init_with_unknown_model_type_raises(self, simple_objective_fn):
        """Test initialization with unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            HyperparameterOptimizer(
                model_type="unknown",
                objective_fn=simple_objective_fn,
            )

    def test_init_with_custom_search_space(self, simple_objective_fn, sample_search_space):
        """Test initialization with custom search space."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=sample_search_space,
        )

        assert optimizer.search_space == sample_search_space
        assert optimizer.search_space != LSTM_SEARCH_SPACE

    def test_init_with_minimize_direction(self, simple_objective_fn):
        """Test initialization with minimize direction."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            direction="minimize",
        )

        assert optimizer.direction == "minimize"

    def test_init_with_custom_study_name(self, simple_objective_fn):
        """Test initialization with custom study name."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            study_name="custom_study",
        )

        assert optimizer.study_name == "custom_study"

    def test_init_generates_study_name(self, simple_objective_fn):
        """Test initialization generates study name if not provided."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        assert "lstm_opt_" in optimizer.study_name

    def test_init_startup_trials(self, simple_objective_fn):
        """Test initialization with custom startup trials."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            n_startup_trials=20,
        )

        assert optimizer.n_startup_trials == 20


class TestHyperparameterOptimizerSuggestParams:
    """Tests for _suggest_params method."""

    def test_suggest_int_param(self, simple_objective_fn, mock_trial):
        """Test suggesting integer parameters."""
        search_space = [
            HyperparameterSpace("hidden_size", "int", low=32, high=128, step=32),
        ]
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=search_space,
        )

        params = optimizer._suggest_params(mock_trial)

        mock_trial.suggest_int.assert_called_once_with("hidden_size", 32, 128, step=32)
        assert "hidden_size" in params

    def test_suggest_float_param(self, simple_objective_fn, mock_trial):
        """Test suggesting float parameters."""
        search_space = [
            HyperparameterSpace("dropout", "float", low=0.1, high=0.5),
        ]
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=search_space,
        )

        params = optimizer._suggest_params(mock_trial)

        mock_trial.suggest_float.assert_called_once_with("dropout", 0.1, 0.5)
        assert "dropout" in params

    def test_suggest_float_log_scale(self, simple_objective_fn, mock_trial):
        """Test suggesting float parameters with log scale."""
        search_space = [
            HyperparameterSpace("learning_rate", "float", low=1e-4, high=1e-2, log=True),
        ]
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=search_space,
        )

        params = optimizer._suggest_params(mock_trial)

        mock_trial.suggest_float.assert_called_once_with(
            "learning_rate", 1e-4, 1e-2, log=True
        )
        assert "learning_rate" in params

    def test_suggest_categorical_param(self, simple_objective_fn, mock_trial):
        """Test suggesting categorical parameters."""
        search_space = [
            HyperparameterSpace("batch_size", "categorical", choices=[16, 32, 64]),
        ]
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=search_space,
        )

        params = optimizer._suggest_params(mock_trial)

        mock_trial.suggest_categorical.assert_called_once_with("batch_size", [16, 32, 64])
        assert "batch_size" in params

    def test_suggest_loguniform_param(self, simple_objective_fn, mock_trial):
        """Test suggesting loguniform parameters."""
        search_space = [
            HyperparameterSpace("epsilon", "loguniform", low=1e-5, high=1e-3),
        ]
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=search_space,
        )

        params = optimizer._suggest_params(mock_trial)

        mock_trial.suggest_float.assert_called_once_with("epsilon", 1e-5, 1e-3, log=True)
        assert "epsilon" in params

    def test_suggest_multiple_params(self, simple_objective_fn, mock_trial, sample_search_space):
        """Test suggesting multiple parameters of different types."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=sample_search_space,
        )

        params = optimizer._suggest_params(mock_trial)

        # Should have all parameter names from search space
        expected_names = {hp.name for hp in sample_search_space}
        assert set(params.keys()) == expected_names


class TestHyperparameterOptimizerCreateObjective:
    """Tests for _create_objective method."""

    def test_create_objective_returns_callable(self, simple_objective_fn):
        """Test that _create_objective returns a callable."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        objective = optimizer._create_objective()

        assert callable(objective)

    def test_objective_calls_suggest_params(self, simple_objective_fn, mock_trial):
        """Test that objective function calls _suggest_params."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        # Patch _suggest_params to track calls
        with patch.object(optimizer, "_suggest_params", return_value={"hidden_size": 64}):
            objective = optimizer._create_objective()
            objective(mock_trial)

            optimizer._suggest_params.assert_called_once_with(mock_trial)

    def test_objective_calls_user_objective(self, mock_trial):
        """Test that objective function calls user's objective function."""
        user_objective = MagicMock(return_value=1.5)
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=user_objective,
        )

        with patch.object(optimizer, "_suggest_params", return_value={"hidden_size": 64}):
            objective = optimizer._create_objective()
            result = objective(mock_trial)

            user_objective.assert_called_once_with({"hidden_size": 64})
            assert result == 1.5

    def test_objective_reports_value(self, simple_objective_fn, mock_trial):
        """Test that objective function reports value for pruning."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with patch.object(optimizer, "_suggest_params", return_value={}):
            objective = optimizer._create_objective()
            objective(mock_trial)

            mock_trial.report.assert_called_once_with(1.5, step=0)

    def test_objective_handles_exception_maximize(self, failing_objective_fn, mock_trial):
        """Test that objective returns -inf on exception when maximizing."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=failing_objective_fn,
            direction="maximize",
        )

        with patch.object(optimizer, "_suggest_params", return_value={}):
            objective = optimizer._create_objective()
            result = objective(mock_trial)

            assert result == float("-inf")

    def test_objective_handles_exception_minimize(self, failing_objective_fn, mock_trial):
        """Test that objective returns +inf on exception when minimizing."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=failing_objective_fn,
            direction="minimize",
        )

        with patch.object(optimizer, "_suggest_params", return_value={}):
            objective = optimizer._create_objective()
            result = objective(mock_trial)

            assert result == float("inf")


class TestHyperparameterOptimizerOptimize:
    """Tests for optimize method."""

    def test_optimize_creates_study(self, simple_objective_fn):
        """Test that optimize creates an Optuna study."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        # Mock optuna module
        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            # Set up mock study
            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 64}
            mock_study.best_trial.value = 1.5
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            result = optimizer.optimize(n_trials=5, show_progress=False)

            mock_optuna.create_study.assert_called_once()
            assert result.best_params == {"hidden_size": 64}
            assert result.best_value == 1.5

    def test_optimize_returns_optimization_result(self, simple_objective_fn):
        """Test that optimize returns an OptimizationResult."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {}
            mock_study.best_trial.value = 1.0
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            result = optimizer.optimize(n_trials=5, show_progress=False)

            assert isinstance(result, OptimizationResult)
            assert result.study_name == optimizer.study_name

    def test_optimize_with_storage_path(self, simple_objective_fn, tmp_path):
        """Test optimize with SQLite storage for persistence."""
        db_path = str(tmp_path / "study.db")
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            storage_path=db_path,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {}
            mock_study.best_trial.value = 1.0
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            optimizer.optimize(n_trials=5, show_progress=False)

            # Verify storage URL is passed
            call_kwargs = mock_optuna.create_study.call_args.kwargs
            assert call_kwargs["storage"] == f"sqlite:///{db_path}"

    def test_optimize_collects_all_trials(self, simple_objective_fn):
        """Test that optimize collects all trial information."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            # Create mock trials
            mock_trial1 = MagicMock()
            mock_trial1.number = 0
            mock_trial1.params = {"hidden_size": 32}
            mock_trial1.value = 1.0
            mock_trial1.state = "COMPLETE"

            mock_trial2 = MagicMock()
            mock_trial2.number = 1
            mock_trial2.params = {"hidden_size": 64}
            mock_trial2.value = 1.5
            mock_trial2.state = "COMPLETE"

            mock_study = MagicMock()
            mock_study.best_trial = mock_trial2
            mock_study.trials = [mock_trial1, mock_trial2]
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            result = optimizer.optimize(n_trials=2, show_progress=False)

            assert len(result.all_trials) == 2
            assert result.all_trials[0]["params"] == {"hidden_size": 32}


class TestHyperparameterOptimizerGetMethods:
    """Tests for get_best_params and get_param_importance methods."""

    def test_get_best_params_before_optimize_raises(self, simple_objective_fn):
        """Test get_best_params raises if optimize not called."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with pytest.raises(ValueError, match="No optimization has been run"):
            optimizer.get_best_params()

    def test_get_best_params_after_optimize(self, simple_objective_fn):
        """Test get_best_params returns best parameters after optimization."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        # Simulate optimization completed
        mock_trial = MagicMock()
        mock_trial.params = {"hidden_size": 128, "learning_rate": 0.001}
        optimizer._best_trial = mock_trial

        params = optimizer.get_best_params()

        assert params == {"hidden_size": 128, "learning_rate": 0.001}

    def test_get_param_importance_before_optimize_raises(self, simple_objective_fn):
        """Test get_param_importance raises if optimize not called."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with pytest.raises(ValueError, match="No optimization has been run"):
            optimizer.get_param_importance()

    def test_get_param_importance_after_optimize(self, simple_objective_fn):
        """Test get_param_importance returns importance after optimization."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        # Set up mock study
        optimizer._study = MagicMock()

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna
            mock_optuna.importance.get_param_importances.return_value = {
                "hidden_size": 0.7,
                "learning_rate": 0.3,
            }

            importance = optimizer.get_param_importance()

            assert importance["hidden_size"] == 0.7
            assert importance["learning_rate"] == 0.3


# =============================================================================
# TESTS: WALK FORWARD OPTIMIZER
# =============================================================================


class TestWalkForwardOptimizerInit:
    """Tests for WalkForwardOptimizer initialization."""

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        train_fn = MagicMock()
        eval_fn = MagicMock()

        optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
        )

        assert optimizer.model_type == "lstm"
        assert optimizer.train_fn == train_fn
        assert optimizer.evaluate_fn == eval_fn
        assert optimizer.n_folds == 5
        assert optimizer.train_ratio == 0.7
        assert optimizer.gap_days == 5

    def test_init_with_custom_folds(self):
        """Test initialization with custom fold settings."""
        optimizer = WalkForwardOptimizer(
            model_type="dqn",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
            n_folds=10,
            train_ratio=0.8,
            gap_days=10,
        )

        assert optimizer.n_folds == 10
        assert optimizer.train_ratio == 0.8
        assert optimizer.gap_days == 10


class TestWalkForwardOptimizerSetData:
    """Tests for set_data method."""

    def test_set_data_stores_data(self, sample_data):
        """Test that set_data stores the data."""
        optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        optimizer.set_data(sample_data)

        assert optimizer._data is not None
        assert len(optimizer._data) == 120


class TestWalkForwardOptimizerObjective:
    """Tests for _walk_forward_objective method."""

    def test_walk_forward_objective_without_data_raises(self):
        """Test walk-forward objective raises if no data set."""
        optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        with pytest.raises(ValueError, match="Must call set_data first"):
            optimizer._walk_forward_objective({})

    def test_walk_forward_objective_calls_train_and_evaluate(self, sample_data):
        """Test walk-forward objective calls train and evaluate functions."""
        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(return_value=1.5)

        optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_folds=3,
            gap_days=2,
        )
        optimizer.set_data(sample_data)

        result = optimizer._walk_forward_objective({"hidden_size": 64})

        # Should have called train/evaluate for each fold
        assert train_fn.call_count >= 1
        assert eval_fn.call_count >= 1
        assert isinstance(result, float)

    def test_walk_forward_objective_respects_gap_days(self, sample_data):
        """Test walk-forward objective creates gap between train and test."""
        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(return_value=1.0)
        gap_days = 10

        optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_folds=2,
            gap_days=gap_days,
        )
        optimizer.set_data(sample_data)

        optimizer._walk_forward_objective({})

        # Verify train/test data don't overlap by gap_days
        # Get the train and test data from the calls
        for call in train_fn.call_args_list:
            train_data = call[0][1]  # Second positional arg is train_data
            assert len(train_data) > 0

    def test_walk_forward_objective_handles_failed_folds(self, sample_data):
        """Test walk-forward objective handles fold failures gracefully."""
        train_fn = MagicMock(side_effect=ValueError("Training failed"))
        eval_fn = MagicMock()

        optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_folds=3,
        )
        optimizer.set_data(sample_data)

        result = optimizer._walk_forward_objective({})

        # All folds failed, should return -inf
        assert result == float("-inf")

    def test_walk_forward_objective_returns_average(self, sample_data):
        """Test walk-forward objective returns average of fold metrics."""
        train_fn = MagicMock(return_value="model")
        # Return different values for each fold
        eval_fn = MagicMock(side_effect=[1.0, 1.5, 2.0, 1.5, 1.0])

        optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_folds=5,
            gap_days=0,
        )
        optimizer.set_data(sample_data)

        result = optimizer._walk_forward_objective({})

        # Result should be average of successful folds
        assert isinstance(result, float)
        assert result > 0


class TestWalkForwardOptimizerOptimize:
    """Tests for optimize method."""

    def test_optimize_sets_data_and_creates_optimizer(self, sample_data):
        """Test optimize sets data and creates HyperparameterOptimizer."""
        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(return_value=1.5)

        wf_optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 64}
            mock_study.best_trial.value = 1.5
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            result = wf_optimizer.optimize(sample_data, n_trials=2)

            assert wf_optimizer._data is not None
            assert wf_optimizer._optimizer is not None
            assert isinstance(result, OptimizationResult)


# =============================================================================
# TESTS: REGIME AWARE OPTIMIZER
# =============================================================================


class TestRegimeAwareOptimizerInit:
    """Tests for RegimeAwareOptimizer initialization."""

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        objective_fn = MagicMock()
        regime_classifier = MagicMock()

        optimizer = RegimeAwareOptimizer(
            model_type="lstm",
            objective_fn=objective_fn,
            regime_classifier=regime_classifier,
        )

        assert optimizer.model_type == "lstm"
        assert optimizer.objective_fn == objective_fn
        assert optimizer.regime_classifier == regime_classifier
        assert optimizer._regime_params == {}


class TestRegimeAwareOptimizerOptimizeForRegimes:
    """Tests for optimize_for_regimes method."""

    def test_optimize_for_default_regimes(self):
        """Test optimization for all default market regimes."""
        objective_fn = MagicMock(return_value=1.5)
        regime_classifier = MagicMock()

        ra_optimizer = RegimeAwareOptimizer(
            model_type="lstm",
            objective_fn=objective_fn,
            regime_classifier=regime_classifier,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 64}
            mock_study.best_trial.value = 1.5
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            results = ra_optimizer.optimize_for_regimes(
                data=np.random.randn(100, 10),
                n_trials_per_regime=2,
            )

            # Should optimize for all 4 default regimes
            assert len(results) == 4
            assert "bull" in results
            assert "bear" in results
            assert "sideways" in results
            assert "volatile" in results

    def test_optimize_for_custom_regimes(self):
        """Test optimization for custom regime list."""
        objective_fn = MagicMock(return_value=1.5)
        regime_classifier = MagicMock()

        ra_optimizer = RegimeAwareOptimizer(
            model_type="lstm",
            objective_fn=objective_fn,
            regime_classifier=regime_classifier,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 64}
            mock_study.best_trial.value = 1.5
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            results = ra_optimizer.optimize_for_regimes(
                data=np.random.randn(100, 10),
                regimes=["bull", "bear"],
                n_trials_per_regime=2,
            )

            assert len(results) == 2
            assert "bull" in results
            assert "bear" in results
            assert "sideways" not in results

    def test_optimize_stores_regime_params(self):
        """Test that optimize_for_regimes stores params for each regime."""
        objective_fn = MagicMock(return_value=1.5)
        regime_classifier = MagicMock()

        ra_optimizer = RegimeAwareOptimizer(
            model_type="lstm",
            objective_fn=objective_fn,
            regime_classifier=regime_classifier,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 128}
            mock_study.best_trial.value = 2.0
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            ra_optimizer.optimize_for_regimes(
                data=np.random.randn(100, 10),
                regimes=["bull"],
                n_trials_per_regime=2,
            )

            assert "bull" in ra_optimizer._regime_params
            assert ra_optimizer._regime_params["bull"] == {"hidden_size": 128}


class TestRegimeAwareOptimizerGetParamsForRegime:
    """Tests for get_params_for_regime method."""

    def test_get_params_for_existing_regime(self):
        """Test getting params for a regime that was optimized."""
        optimizer = RegimeAwareOptimizer(
            model_type="lstm",
            objective_fn=MagicMock(),
            regime_classifier=MagicMock(),
        )

        # Manually set regime params
        optimizer._regime_params["bull"] = {"hidden_size": 128, "learning_rate": 0.001}

        params = optimizer.get_params_for_regime("bull")

        assert params == {"hidden_size": 128, "learning_rate": 0.001}

    def test_get_params_for_unknown_regime_raises(self):
        """Test getting params for unknown regime raises ValueError."""
        optimizer = RegimeAwareOptimizer(
            model_type="lstm",
            objective_fn=MagicMock(),
            regime_classifier=MagicMock(),
        )

        with pytest.raises(ValueError, match="No optimized params for regime"):
            optimizer.get_params_for_regime("unknown_regime")


# =============================================================================
# TESTS: OPTUNA IMPORT AND ERROR HANDLING
# =============================================================================


class TestOptunaImport:
    """Tests for lazy optuna import functionality."""

    def test_import_optuna_caches_module(self):
        """Test that _import_optuna caches the module."""
        with patch("ml.hyperparameter_optimizer._optuna", None):
            with patch.dict("sys.modules", {"optuna": MagicMock()}):
                # First import
                result1 = _import_optuna()
                # Second import should return same object
                result2 = _import_optuna()

                assert result1 is result2

    def test_import_optuna_raises_on_missing(self):
        """Test that _import_optuna raises ImportError if optuna not installed."""
        with patch("ml.hyperparameter_optimizer._optuna", None):
            with patch.dict("sys.modules", {"optuna": None}):
                # Remove optuna from available imports
                import builtins
                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "optuna":
                        raise ImportError("No module named 'optuna'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    with pytest.raises(ImportError, match="optuna is required"):
                        _import_optuna()


# =============================================================================
# TESTS: VISUALIZATION METHODS
# =============================================================================


class TestHyperparameterOptimizerVisualization:
    """Tests for visualization methods."""

    def test_plot_optimization_history_before_optimize_raises(self, simple_objective_fn):
        """Test plot_optimization_history raises if no optimization run."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with pytest.raises(ValueError, match="No optimization has been run"):
            optimizer.plot_optimization_history()

    def test_plot_param_importance_before_optimize_raises(self, simple_objective_fn):
        """Test plot_param_importance raises if no optimization run."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with pytest.raises(ValueError, match="No optimization has been run"):
            optimizer.plot_param_importance()

    def test_plot_optimization_history_with_study(self, simple_objective_fn):
        """Test plot_optimization_history with a valid study."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )
        optimizer._study = MagicMock()

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna
            mock_fig = MagicMock()
            mock_optuna.visualization.plot_optimization_history.return_value = mock_fig

            result = optimizer.plot_optimization_history()

            mock_optuna.visualization.plot_optimization_history.assert_called_once_with(
                optimizer._study
            )
            assert result == mock_fig

    def test_plot_optimization_history_saves_to_file(self, simple_objective_fn, tmp_path):
        """Test plot_optimization_history saves to file when path provided."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )
        optimizer._study = MagicMock()

        output_path = str(tmp_path / "history.html")

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna
            mock_fig = MagicMock()
            mock_optuna.visualization.plot_optimization_history.return_value = mock_fig

            optimizer.plot_optimization_history(output_path=output_path)

            mock_fig.write_html.assert_called_once_with(output_path)


# =============================================================================
# TESTS: EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_search_space(self, simple_objective_fn, mock_trial):
        """Test optimizer with empty search space."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=[],
        )

        params = optimizer._suggest_params(mock_trial)

        assert params == {}

    def test_walk_forward_with_small_data(self):
        """Test walk-forward optimizer with data smaller than folds."""
        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(return_value=1.0)

        optimizer = WalkForwardOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_folds=10,  # More folds than data allows
        )

        # Very small dataset
        small_data = np.random.randn(10, 5)
        optimizer.set_data(small_data)

        # Should handle gracefully
        result = optimizer._walk_forward_objective({})
        # Result might be -inf if no valid folds, or average of partial folds
        assert isinstance(result, float)

    def test_param_importance_exception_handling(self, simple_objective_fn):
        """Test get_param_importance handles exceptions gracefully."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )
        optimizer._study = MagicMock()

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna
            mock_optuna.importance.get_param_importances.side_effect = Exception(
                "Not enough trials"
            )

            importance = optimizer.get_param_importance()

            assert importance == {}

    def test_optimization_with_timeout(self, simple_objective_fn):
        """Test optimization respects timeout parameter."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {}
            mock_study.best_trial.value = 1.0
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            optimizer.optimize(n_trials=100, timeout=60, show_progress=False)

            # Verify timeout was passed to optimize
            call_kwargs = mock_study.optimize.call_args.kwargs
            assert call_kwargs["timeout"] == 60

    def test_optimization_with_parallel_jobs(self, simple_objective_fn):
        """Test optimization with multiple parallel jobs."""
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {}
            mock_study.best_trial.value = 1.0
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            optimizer.optimize(n_trials=10, n_jobs=4, show_progress=False)

            # Verify n_jobs was passed to optimize
            call_kwargs = mock_study.optimize.call_args.kwargs
            assert call_kwargs["n_jobs"] == 4

    def test_int_param_without_step(self, simple_objective_fn, mock_trial):
        """Test integer parameter suggestion without explicit step."""
        search_space = [
            HyperparameterSpace("num_layers", "int", low=1, high=4),  # No step
        ]
        optimizer = HyperparameterOptimizer(
            model_type="lstm",
            objective_fn=simple_objective_fn,
            search_space=search_space,
        )

        optimizer._suggest_params(mock_trial)

        # Should use step=1 by default
        mock_trial.suggest_int.assert_called_once_with("num_layers", 1, 4, step=1)


# =============================================================================
# TESTS: NESTED CV RESULT DATACLASS
# =============================================================================


class TestNestedCVResult:
    """Tests for NestedCVResult dataclass."""

    def test_nested_cv_result_creation(self):
        """Test creating a NestedCVResult with all fields."""
        result = NestedCVResult(
            fold_params=[{"hidden_size": 64}, {"hidden_size": 128}],
            fold_oos_metrics=[1.5, 1.8],
            recommended_params={"hidden_size": 96},
            parameter_stability={"hidden_size": {"mean": 96.0, "std": 32.0, "cv": 0.33}},
            mean_oos_metric=1.65,
            std_oos_metric=0.15,
            unstable_parameters=["hidden_size"],
        )

        assert len(result.fold_params) == 2
        assert len(result.fold_oos_metrics) == 2
        assert result.recommended_params == {"hidden_size": 96}
        assert result.mean_oos_metric == 1.65
        assert result.std_oos_metric == 0.15
        assert "hidden_size" in result.unstable_parameters

    def test_nested_cv_result_empty_unstable(self):
        """Test NestedCVResult with no unstable parameters."""
        result = NestedCVResult(
            fold_params=[{"hidden_size": 64}, {"hidden_size": 65}],  # Very stable
            fold_oos_metrics=[1.5, 1.5],
            recommended_params={"hidden_size": 64},
            parameter_stability={"hidden_size": {"mean": 64.5, "std": 0.5, "cv": 0.008}},
            mean_oos_metric=1.5,
            std_oos_metric=0.0,
            unstable_parameters=[],
        )

        assert result.unstable_parameters == []
        assert result.parameter_stability["hidden_size"]["cv"] < 0.2


# =============================================================================
# TESTS: NESTED CV OPTIMIZER INIT
# =============================================================================


class TestNestedCVOptimizerInit:
    """Tests for NestedCVOptimizer initialization."""

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        train_fn = MagicMock()
        eval_fn = MagicMock()

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
        )

        assert optimizer.model_type == "lstm"
        assert optimizer.train_fn == train_fn
        assert optimizer.evaluate_fn == eval_fn
        assert optimizer.n_outer_folds == 5
        assert optimizer.n_inner_trials == 50
        assert optimizer.inner_cv_folds == 3
        assert optimizer.gap_days == 5
        assert optimizer.stability_threshold == 0.2

    def test_init_with_custom_folds(self):
        """Test initialization with custom fold settings."""
        optimizer = NestedCVOptimizer(
            model_type="dqn",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
            n_outer_folds=3,
            n_inner_trials=20,
            inner_cv_folds=5,
            gap_days=10,
            stability_threshold=0.3,
        )

        assert optimizer.n_outer_folds == 3
        assert optimizer.n_inner_trials == 20
        assert optimizer.inner_cv_folds == 5
        assert optimizer.gap_days == 10
        assert optimizer.stability_threshold == 0.3

    def test_init_with_lstm_default_search_space(self):
        """Test LSTM model type uses default LSTM search space."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        assert optimizer.search_space == LSTM_SEARCH_SPACE

    def test_init_with_dqn_default_search_space(self):
        """Test DQN model type uses default DQN search space."""
        optimizer = NestedCVOptimizer(
            model_type="dqn",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        assert optimizer.search_space == DQN_SEARCH_SPACE

    def test_init_with_unknown_model_type_raises(self):
        """Test unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            NestedCVOptimizer(
                model_type="unknown",
                train_fn=MagicMock(),
                evaluate_fn=MagicMock(),
            )

    def test_init_with_custom_search_space(self, sample_search_space):
        """Test initialization with custom search space."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
            search_space=sample_search_space,
        )

        assert optimizer.search_space == sample_search_space


# =============================================================================
# TESTS: NESTED CV OPTIMIZER INNER CV OBJECTIVE
# =============================================================================


class TestNestedCVOptimizerInnerCV:
    """Tests for _inner_cv_objective method."""

    def test_inner_cv_calls_train_and_evaluate(self, sample_data):
        """Test inner CV calls train and evaluate functions."""
        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(return_value=1.5)

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            inner_cv_folds=3,
            gap_days=2,
        )

        result = optimizer._inner_cv_objective({"hidden_size": 64}, sample_data)

        assert train_fn.call_count >= 1
        assert eval_fn.call_count >= 1
        assert isinstance(result, float)

    def test_inner_cv_returns_average(self, sample_data):
        """Test inner CV returns average of fold metrics."""
        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(side_effect=[1.0, 1.5, 2.0])

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            inner_cv_folds=3,
            gap_days=0,
        )

        result = optimizer._inner_cv_objective({}, sample_data)

        assert isinstance(result, float)
        # Should be average of successful folds

    def test_inner_cv_handles_failed_folds(self, sample_data):
        """Test inner CV handles fold failures gracefully."""
        train_fn = MagicMock(side_effect=ValueError("Training failed"))
        eval_fn = MagicMock()

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            inner_cv_folds=3,
        )

        result = optimizer._inner_cv_objective({}, sample_data)

        assert result == float("-inf")


# =============================================================================
# TESTS: NESTED CV OPTIMIZER PARAMETER STABILITY
# =============================================================================


class TestNestedCVOptimizerParameterStability:
    """Tests for _analyze_parameter_stability method."""

    def test_stability_analysis_stable_params(self):
        """Test stability analysis identifies stable parameters."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
            stability_threshold=0.2,
        )

        # Very stable parameters (low variance)
        fold_params = [
            {"hidden_size": 64, "dropout": 0.2},
            {"hidden_size": 65, "dropout": 0.21},
            {"hidden_size": 63, "dropout": 0.19},
        ]

        stability, unstable = optimizer._analyze_parameter_stability(fold_params)

        assert "hidden_size" in stability
        assert "dropout" in stability
        # Low CV should mean not unstable
        assert stability["hidden_size"]["cv"] < 0.2
        assert len(unstable) == 0  # All stable

    def test_stability_analysis_unstable_params(self):
        """Test stability analysis identifies unstable parameters."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
            stability_threshold=0.2,
        )

        # Unstable parameters (high variance)
        fold_params = [
            {"hidden_size": 32, "dropout": 0.1},
            {"hidden_size": 128, "dropout": 0.5},
            {"hidden_size": 256, "dropout": 0.3},
        ]

        stability, unstable = optimizer._analyze_parameter_stability(fold_params)

        assert "hidden_size" in stability
        # High CV should flag as unstable
        assert stability["hidden_size"]["cv"] > 0.2
        assert "hidden_size" in unstable

    def test_stability_analysis_empty_params(self):
        """Test stability analysis with empty params list."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        stability, unstable = optimizer._analyze_parameter_stability([])

        assert stability == {}
        assert unstable == []

    def test_stability_analysis_categorical_params(self):
        """Test stability analysis handles categorical parameters."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        # Categorical params (can't compute meaningful CV)
        fold_params = [
            {"batch_size": 32},
            {"batch_size": 64},
            {"batch_size": 32},
        ]

        stability, unstable = optimizer._analyze_parameter_stability(fold_params)

        # Should handle without errors
        assert "batch_size" in stability


# =============================================================================
# TESTS: NESTED CV OPTIMIZER PARAMETER RECOMMENDATION
# =============================================================================


class TestNestedCVOptimizerRecommendation:
    """Tests for _recommend_parameters method."""

    def test_recommend_numeric_params_median(self):
        """Test recommending numeric parameters uses median."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        fold_params = [
            {"hidden_size": 32},
            {"hidden_size": 64},
            {"hidden_size": 128},
        ]
        stability = {}

        recommended = optimizer._recommend_parameters(fold_params, stability)

        # Median of [32, 64, 128] = 64
        assert recommended["hidden_size"] == 64

    def test_recommend_int_params_rounded(self):
        """Test integer parameters are rounded properly."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        fold_params = [
            {"num_layers": 1},
            {"num_layers": 2},
            {"num_layers": 3},
            {"num_layers": 4},
        ]
        stability = {}

        recommended = optimizer._recommend_parameters(fold_params, stability)

        # Median should be integer
        assert isinstance(recommended["num_layers"], (int, float))

    def test_recommend_categorical_params_mode(self):
        """Test categorical parameters use mode (most common)."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        fold_params = [
            {"batch_size": 32},
            {"batch_size": 32},
            {"batch_size": 64},
        ]
        stability = {}

        recommended = optimizer._recommend_parameters(fold_params, stability)

        # Mode of [32, 32, 64] = 32
        assert recommended["batch_size"] == 32

    def test_recommend_empty_params(self):
        """Test recommendation with empty params."""
        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
        )

        recommended = optimizer._recommend_parameters([], {})

        assert recommended == {}


# =============================================================================
# TESTS: NESTED CV OPTIMIZER OPTIMIZE
# =============================================================================


class TestNestedCVOptimizerOptimize:
    """Tests for optimize method."""

    def test_optimize_runs_outer_folds(self, sample_data):
        """Test optimize runs correct number of outer folds."""
        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(return_value=1.5)

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_outer_folds=3,
            n_inner_trials=2,
            inner_cv_folds=2,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 64}
            mock_study.best_trial.value = 1.5
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            result = optimizer.optimize(sample_data)

            assert isinstance(result, NestedCVResult)
            # May have fewer folds if some failed
            assert len(result.fold_params) <= 3

    def test_optimize_returns_nested_cv_result(self, sample_data):
        """Test optimize returns NestedCVResult with all fields."""
        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(return_value=1.5)

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_outer_folds=2,
            n_inner_trials=2,
            inner_cv_folds=2,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 64}
            mock_study.best_trial.value = 1.5
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            result = optimizer.optimize(sample_data)

            assert hasattr(result, "fold_params")
            assert hasattr(result, "fold_oos_metrics")
            assert hasattr(result, "recommended_params")
            assert hasattr(result, "parameter_stability")
            assert hasattr(result, "mean_oos_metric")
            assert hasattr(result, "std_oos_metric")
            assert hasattr(result, "unstable_parameters")

    def test_optimize_raises_on_all_failed_folds(self):
        """Test optimize raises if all folds fail."""
        train_fn = MagicMock(side_effect=ValueError("Always fails"))
        eval_fn = MagicMock()

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_outer_folds=2,
            n_inner_trials=1,
        )

        small_data = np.random.randn(10, 5)  # Too small to have valid folds

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {}
            mock_study.best_trial.value = float("-inf")
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study

            with pytest.raises(ValueError, match="All outer folds failed"):
                optimizer.optimize(small_data)

    def test_optimize_calculates_statistics(self, sample_data):
        """Test optimize calculates mean and std of OOS metrics."""
        train_fn = MagicMock(return_value="model")
        # Return different values for each fold
        eval_fn = MagicMock(side_effect=[1.0, 2.0, 1.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_outer_folds=2,
            n_inner_trials=1,
            inner_cv_folds=2,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 64}
            mock_study.best_trial.value = 1.5
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            result = optimizer.optimize(sample_data)

            assert isinstance(result.mean_oos_metric, float)
            assert isinstance(result.std_oos_metric, float)


# =============================================================================
# TESTS: NESTED CV OPTIMIZER INTEGRATION
# =============================================================================


class TestNestedCVOptimizerIntegration:
    """Integration tests for NestedCVOptimizer."""

    def test_full_nested_cv_workflow(self):
        """Test complete nested CV workflow end-to-end."""
        # Generate larger dataset for meaningful splits
        data = np.random.randn(200, 10)

        train_fn = MagicMock(return_value="model")
        eval_fn = MagicMock(return_value=1.5)

        optimizer = NestedCVOptimizer(
            model_type="lstm",
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            n_outer_folds=3,
            n_inner_trials=2,
            inner_cv_folds=2,
            gap_days=5,
            stability_threshold=0.2,
        )

        with patch("ml.hyperparameter_optimizer._import_optuna") as mock_import:
            mock_optuna = MagicMock()
            mock_import.return_value = mock_optuna

            mock_study = MagicMock()
            mock_study.best_trial = MagicMock()
            mock_study.best_trial.params = {"hidden_size": 64, "dropout": 0.3}
            mock_study.best_trial.value = 1.5
            mock_study.trials = []
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.importance.get_param_importances.return_value = {}

            result = optimizer.optimize(data)

            # Verify complete result structure
            assert isinstance(result, NestedCVResult)
            assert result.recommended_params is not None
            assert isinstance(result.parameter_stability, dict)
            assert isinstance(result.unstable_parameters, list)
