"""
Research Registry - Research/Production Code Separation

Prevents overfitting and ensures proper model validation before production deployment.

Key problems this solves:
1. **Data leakage**: Research code might accidentally use future data
2. **Overfitting**: Models tuned to historical data without proper validation
3. **Deployment risks**: Untested models deployed to production
4. **Reproducibility**: Experiments not versioned or documented

Architecture:
    Research Environment                Production Environment
    ┌─────────────────────┐            ┌─────────────────────┐
    │  Experiments        │            │  Deployed Models    │
    │  - Alpha signals    │  ──────>   │  - Validated only   │
    │  - New strategies   │  Promote   │  - Version tracked  │
    │  - Parameter tuning │            │  - Monitored        │
    └─────────────────────┘            └─────────────────────┘
              │                                   │
              ▼                                   ▼
    ┌─────────────────────┐            ┌─────────────────────┐
    │  Validation Gates   │            │  Production Monitor │
    │  - Walk-forward     │            │  - Performance      │
    │  - Out-of-sample    │            │  - Degradation      │
    │  - Paper trading    │            │  - Auto-rollback    │
    └─────────────────────┘            └─────────────────────┘

Usage:
    registry = ResearchRegistry()

    # Register a new experiment
    exp_id = registry.create_experiment(
        name="momentum_v2",
        description="Enhanced momentum with sector-relative component",
        author="quant_team",
    )

    # Record results
    registry.record_backtest_results(exp_id, backtest_results)
    registry.record_paper_results(exp_id, paper_results)

    # Check if ready for production
    if registry.is_promotion_ready(exp_id):
        registry.promote_to_production(exp_id)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.execution_quality_gate import (
    extract_execution_quality_metrics,
    extract_paper_live_shadow_drift,
)
from utils.paper_burn_in import build_paper_burn_in_scorecard

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of a research experiment."""
    DRAFT = "draft"  # Initial development
    BACKTEST = "backtest"  # Backtesting phase
    VALIDATION = "validation"  # Walk-forward validation
    PAPER_TRADING = "paper_trading"  # Live paper trading test
    REVIEW = "review"  # Awaiting promotion review
    PROMOTED = "promoted"  # In production
    REJECTED = "rejected"  # Failed validation
    DEPRECATED = "deprecated"  # Removed from production


class ValidationResult(Enum):
    """Result of validation check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class ValidationGate:
    """A validation gate that must pass before promotion."""

    name: str
    description: str
    required: bool
    result: ValidationResult = ValidationResult.PENDING
    message: str = ""
    checked_at: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "result": self.result.value,
            "message": self.message,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "metrics": self.metrics,
        }


@dataclass
class Experiment:
    """A research experiment."""

    id: str
    name: str
    description: str
    author: str
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime

    # Version tracking
    version: str = "0.1.0"
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_history: List[Dict[str, Any]] = field(default_factory=list)
    artifact_index: Dict[str, Any] = field(default_factory=dict)
    promotion_checklist: Dict[str, Any] = field(default_factory=dict)

    # Results
    backtest_results: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    paper_results: Optional[Dict[str, Any]] = None
    production_results: Optional[Dict[str, Any]] = None

    # Validation gates
    validation_gates: List[ValidationGate] = field(default_factory=list)

    # Production tracking
    promoted_at: Optional[datetime] = None
    production_start: Optional[datetime] = None
    production_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "tags": self.tags,
            "parameters": self.parameters,
            "parameter_history": self.parameter_history,
            "artifact_index": self.artifact_index,
            "promotion_checklist": self.promotion_checklist,
            "backtest_results": self.backtest_results,
            "validation_results": self.validation_results,
            "paper_results": self.paper_results,
            "production_results": self.production_results,
            "validation_gates": [g.to_dict() for g in self.validation_gates],
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
        }


class ResearchRegistry:
    """
    Registry for research experiments and production deployments.

    Enforces validation gates before production promotion:
    1. Backtest performance meets thresholds
    2. Walk-forward validation passes
    3. Paper trading period completed
    4. Manual review approved
    """

    # Default validation gate thresholds
    DEFAULT_GATES = {
        "backtest_sharpe": {
            "description": "Backtest Sharpe ratio >= 1.0",
            "threshold": 1.0,
            "required": True,
        },
        "backtest_drawdown": {
            "description": "Max drawdown <= 20%",
            "threshold": 0.20,
            "required": True,
        },
        "walk_forward_oos": {
            "description": "Out-of-sample Sharpe >= 50% of in-sample",
            "threshold": 0.50,
            "required": True,
        },
        "paper_trading_days": {
            "description": "Minimum 20 days paper trading",
            "threshold": 20,
            "required": True,
        },
        "paper_trading_profit": {
            "description": "Paper trading profitable",
            "threshold": 0.0,
            "required": False,
        },
        "paper_reconciliation_rate": {
            "description": "Paper reconciliation pass rate >= 99.5%",
            "threshold": 0.995,
            "required": True,
        },
        "paper_operational_error_rate": {
            "description": "Paper operational error rate <= 2%",
            "threshold": 0.02,
            "required": True,
        },
        "paper_execution_quality_score": {
            "description": "Paper execution quality score >= 70",
            "threshold": 70.0,
            "required": False,
        },
        "paper_avg_slippage_bps": {
            "description": "Paper average slippage <= 25 bps",
            "threshold": 25.0,
            "required": False,
        },
        "paper_fill_rate": {
            "description": "Paper fill rate >= 95%",
            "threshold": 0.95,
            "required": False,
        },
        "paper_live_shadow_drift": {
            "description": "Paper/live shadow drift <= 15%",
            "threshold": 0.15,
            "required": False,
        },
        "statistical_significance": {
            "description": "T-stat >= 2.0 for alpha",
            "threshold": 2.0,
            "required": True,
        },
        "manual_review": {
            "description": "Manual review approved",
            "threshold": None,
            "required": True,
        },
    }

    def __init__(
        self,
        registry_path: str = ".research/experiments",
        production_path: str = ".research/production",
        parameter_registry_path: str = ".research/parameters",
        artifacts_path: str = ".research/artifacts",
    ):
        """
        Initialize research registry.

        Args:
            registry_path: Path for experiment storage
            production_path: Path for production deployments
            parameter_registry_path: Path for parameter snapshot history
            artifacts_path: Path for walk-forward and validation artifacts
        """
        self.registry_path = Path(registry_path)
        self.production_path = Path(production_path)
        self.parameter_registry_path = Path(parameter_registry_path)
        self.artifacts_path = Path(artifacts_path)

        # Create directories
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.production_path.mkdir(parents=True, exist_ok=True)
        self.parameter_registry_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)

        # Load existing experiments
        self.experiments: Dict[str, Experiment] = {}
        self._load_experiments()

        logger.info(
            f"ResearchRegistry initialized: {len(self.experiments)} experiments, "
            f"path={self.registry_path}"
        )

    def _load_experiments(self):
        """Load experiments from disk."""
        for exp_file in self.registry_path.glob("*.json"):
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                    exp = self._dict_to_experiment(data)
                    self.experiments[exp.id] = exp
            except Exception as e:
                logger.warning(f"Error loading experiment {exp_file}: {e}")

    def _dict_to_experiment(self, data: Dict[str, Any]) -> Experiment:
        """Convert dictionary to Experiment."""
        # Parse validation gates
        gates = []
        for g in data.get("validation_gates", []):
            gates.append(ValidationGate(
                name=g["name"],
                description=g["description"],
                required=g["required"],
                result=ValidationResult(g.get("result", "pending")),
                message=g.get("message", ""),
                checked_at=datetime.fromisoformat(g["checked_at"]) if g.get("checked_at") else None,
                metrics=g.get("metrics", {}),
            ))

        return Experiment(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            author=data["author"],
            status=ExperimentStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", "0.1.0"),
            git_commit=data.get("git_commit"),
            config_hash=data.get("config_hash"),
            tags=data.get("tags", []),
            parameters=data.get("parameters", {}),
            parameter_history=data.get("parameter_history", []),
            artifact_index=data.get("artifact_index", {}),
            promotion_checklist=data.get("promotion_checklist", {}),
            backtest_results=data.get("backtest_results"),
            validation_results=data.get("validation_results"),
            paper_results=data.get("paper_results"),
            production_results=data.get("production_results"),
            validation_gates=gates,
            promoted_at=datetime.fromisoformat(data["promoted_at"]) if data.get("promoted_at") else None,
        )

    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk."""
        filepath = self.registry_path / f"{experiment.id}.json"
        with open(filepath, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)

    def create_experiment(
        self,
        name: str,
        description: str,
        author: str,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new research experiment.

        Args:
            name: Experiment name
            description: What this experiment tests
            author: Author/team name
            parameters: Configuration parameters
            tags: Labels for categorization

        Returns:
            Experiment ID
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_slug = name.lower().replace(" ", "_")[:20]
        exp_id = f"{name_slug}_{timestamp}"

        # Create default validation gates
        gates = [
            ValidationGate(
                name=gate_name,
                description=gate_config["description"],
                required=gate_config["required"],
            )
            for gate_name, gate_config in self.DEFAULT_GATES.items()
        ]

        # Get git commit if available
        git_commit = self._get_git_commit()

        # Hash config for reproducibility
        config_hash = self._hash_config(parameters or {})

        experiment = Experiment(
            id=exp_id,
            name=name,
            description=description,
            author=author,
            status=ExperimentStatus.DRAFT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            git_commit=git_commit,
            config_hash=config_hash,
            parameters=parameters or {},
            tags=tags or [],
            validation_gates=gates,
        )

        self.experiments[exp_id] = experiment
        self._save_experiment(experiment)
        self.record_parameter_snapshot(
            exp_id,
            parameters or {},
            source="create_experiment",
            notes="Initial parameter set at experiment creation",
            make_active=True,
        )

        logger.info(f"Created experiment: {exp_id} - {name}")
        return exp_id

    def record_parameter_snapshot(
        self,
        experiment_id: str,
        parameters: Dict[str, Any],
        source: str = "manual",
        notes: str = "",
        make_active: bool = True,
    ) -> Dict[str, Any]:
        """
        Record a versioned parameter snapshot for an experiment.

        Args:
            experiment_id: Experiment ID
            parameters: Parameter dictionary to snapshot
            source: Snapshot source (manual/optimizer/backtest/etc.)
            notes: Optional notes
            make_active: Whether to set this as active experiment parameters

        Returns:
            Snapshot dictionary
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        now = datetime.now()
        param_hash = self._hash_config(parameters)
        snapshot = {
            "snapshot_id": f"p{len(exp.parameter_history) + 1:04d}",
            "created_at": now.isoformat(),
            "source": source,
            "notes": notes,
            "hash": param_hash,
            "parameters": parameters,
        }

        exp.parameter_history.append(snapshot)
        if make_active:
            exp.parameters = parameters
            exp.config_hash = param_hash

        exp.updated_at = now
        self._save_experiment(exp)
        self._save_parameter_registry(exp)

        return snapshot

    def _save_parameter_registry(self, experiment: Experiment) -> None:
        """Persist parameter history as a dedicated registry file."""
        out_path = self.parameter_registry_path / f"{experiment.id}.json"
        payload = {
            "experiment_id": experiment.id,
            "active_config_hash": experiment.config_hash,
            "updated_at": experiment.updated_at.isoformat(),
            "snapshots": experiment.parameter_history,
        }
        with open(out_path, "w") as handle:
            json.dump(payload, handle, indent=2)

    def store_walk_forward_artifacts(
        self,
        experiment_id: str,
        walk_forward_results: Dict[str, Any],
        source_run_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Store walk-forward artifacts and update validation gates.

        Args:
            experiment_id: Experiment ID
            walk_forward_results: Walk-forward output dictionary
            source_run_id: Optional linked backtest run ID

        Returns:
            Dictionary with saved artifact paths
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        now = datetime.now()
        artifact_dir = self.artifacts_path / experiment_id / now.strftime("%Y%m%d_%H%M%S")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        raw_path = artifact_dir / "walk_forward_results.json"
        summary_path = artifact_dir / "walk_forward_summary.json"

        normalized = self._normalize_walk_forward_results(
            walk_forward_results,
            existing_validation=exp.validation_results or {},
        )

        with open(raw_path, "w") as handle:
            json.dump(walk_forward_results, handle, indent=2, default=str)

        summary = {
            "experiment_id": experiment_id,
            "stored_at": now.isoformat(),
            "source_run_id": source_run_id,
            "in_sample_sharpe": normalized.get("in_sample_sharpe"),
            "out_of_sample_sharpe": normalized.get("out_of_sample_sharpe"),
            "alpha_t_stat": normalized.get("alpha_t_stat"),
            "passes_validation": bool(
                walk_forward_results.get("passes_validation")
                if isinstance(walk_forward_results, dict)
                else False
            ),
            "avg_overfit_ratio": (
                walk_forward_results.get("avg_overfit_ratio")
                if isinstance(walk_forward_results, dict)
                else None
            ),
        }
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)

        merged_validation = dict(walk_forward_results)
        merged_validation.update(normalized)

        exp.validation_results = merged_validation
        exp.status = ExperimentStatus.VALIDATION
        exp.updated_at = now

        wf_history = exp.artifact_index.get("walk_forward_history", [])
        wf_history.append(
            {
                "stored_at": now.isoformat(),
                "source_run_id": source_run_id,
                "results_path": str(raw_path),
                "summary_path": str(summary_path),
            }
        )
        exp.artifact_index["walk_forward_history"] = wf_history
        exp.artifact_index["walk_forward_latest"] = wf_history[-1]

        self._check_walkforward_gates(exp, merged_validation)
        self._save_experiment(exp)

        return {
            "results_path": str(raw_path),
            "summary_path": str(summary_path),
            "artifact_dir": str(artifact_dir),
        }

    def _normalize_walk_forward_results(
        self,
        walk_forward_results: Dict[str, Any],
        existing_validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Normalize walk-forward result keys for validation gates.
        """
        existing_validation = existing_validation or {}

        in_sample_sharpe = (
            walk_forward_results.get("in_sample_sharpe")
            if isinstance(walk_forward_results, dict)
            else None
        )
        if in_sample_sharpe is None and isinstance(walk_forward_results, dict):
            in_sample_sharpe = walk_forward_results.get("is_avg_sharpe")
        if in_sample_sharpe is None and isinstance(walk_forward_results, dict):
            in_sample_sharpe = walk_forward_results.get("is_sharpe")
        if in_sample_sharpe is None:
            in_sample_sharpe = existing_validation.get("in_sample_sharpe", 0.0)

        out_of_sample_sharpe = (
            walk_forward_results.get("out_of_sample_sharpe")
            if isinstance(walk_forward_results, dict)
            else None
        )
        if out_of_sample_sharpe is None and isinstance(walk_forward_results, dict):
            out_of_sample_sharpe = walk_forward_results.get("oos_avg_sharpe")
        if out_of_sample_sharpe is None and isinstance(walk_forward_results, dict):
            out_of_sample_sharpe = walk_forward_results.get("oos_sharpe")
        if out_of_sample_sharpe is None:
            out_of_sample_sharpe = existing_validation.get("out_of_sample_sharpe", 0.0)

        alpha_t_stat = (
            walk_forward_results.get("alpha_t_stat")
            if isinstance(walk_forward_results, dict)
            else None
        )
        if alpha_t_stat is None:
            alpha_t_stat = existing_validation.get("alpha_t_stat", 0.0)

        return {
            "in_sample_sharpe": float(in_sample_sharpe or 0.0),
            "out_of_sample_sharpe": float(out_of_sample_sharpe or 0.0),
            "alpha_t_stat": float(alpha_t_stat or 0.0),
        }

    def record_backtest_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
    ):
        """
        Record backtest results for an experiment.

        Args:
            experiment_id: Experiment ID
            results: Backtest results dict
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        exp.backtest_results = results
        exp.status = ExperimentStatus.BACKTEST
        exp.updated_at = datetime.now()

        # Check backtest validation gates
        self._check_backtest_gates(exp, results)

        self._save_experiment(exp)
        logger.info(f"Recorded backtest results for {experiment_id}")

    def record_validation_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
    ):
        """
        Record walk-forward validation results.

        Args:
            experiment_id: Experiment ID
            results: Walk-forward results dict
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        exp.validation_results = results
        exp.status = ExperimentStatus.VALIDATION
        exp.updated_at = datetime.now()

        # Check walk-forward gates
        self._check_walkforward_gates(exp, results)

        self._save_experiment(exp)
        logger.info(f"Recorded validation results for {experiment_id}")

    def record_paper_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
    ):
        """
        Record paper trading results.

        Args:
            experiment_id: Experiment ID
            results: Paper trading results dict
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        exp.paper_results = results
        exp.status = ExperimentStatus.PAPER_TRADING
        exp.updated_at = datetime.now()

        # Check paper trading gates
        self._check_paper_gates(exp, results)

        self._save_experiment(exp)
        logger.info(f"Recorded paper trading results for {experiment_id}")

    def approve_manual_review(
        self,
        experiment_id: str,
        reviewer: str,
        notes: str = "",
    ):
        """
        Approve manual review gate.

        Args:
            experiment_id: Experiment ID
            reviewer: Reviewer name
            notes: Review notes
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        for gate in exp.validation_gates:
            if gate.name == "manual_review":
                gate.result = ValidationResult.PASS
                gate.message = f"Approved by {reviewer}. {notes}"
                gate.checked_at = datetime.now()

        exp.status = ExperimentStatus.REVIEW
        exp.updated_at = datetime.now()
        self._save_experiment(exp)

        logger.info(f"Manual review approved for {experiment_id} by {reviewer}")

    def is_promotion_ready(self, experiment_id: str, strict: bool = False) -> bool:
        """
        Check if experiment is ready for production promotion.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if all required gates pass. In strict mode, also enforce
            parameter registry and artifact checklist requirements.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return False

        for gate in exp.validation_gates:
            if gate.required and gate.result != ValidationResult.PASS:
                return False

        if strict:
            checklist = self.generate_promotion_checklist(experiment_id)
            return bool(checklist.get("ready_for_promotion", False))

        return True

    def get_promotion_blockers(self, experiment_id: str, strict: bool = False) -> List[str]:
        """
        Get list of gates blocking promotion.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of blocking gate descriptions
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return ["Experiment not found"]

        blockers = []
        for gate in exp.validation_gates:
            if gate.required and gate.result != ValidationResult.PASS:
                status = gate.result.value if gate.result else "pending"
                blockers.append(f"{gate.name}: {gate.description} ({status})")

        if strict:
            checklist = self.generate_promotion_checklist(experiment_id)
            for criterion in checklist.get("criteria", []):
                if criterion.get("required") and not criterion.get("passed"):
                    blockers.append(
                        f"{criterion['name']}: {criterion.get('details', 'required criterion not met')}"
                    )

        return blockers

    def generate_promotion_checklist(self, experiment_id: str) -> Dict[str, Any]:
        """
        Generate a strict promotion checklist for CI or review.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {
                "experiment_id": experiment_id,
                "ready_for_promotion": False,
                "criteria": [],
                "blockers": ["Experiment not found"],
            }

        latest_wf = exp.artifact_index.get("walk_forward_latest", {})
        wf_summary_path = latest_wf.get("summary_path")
        wf_summary_exists = bool(wf_summary_path and Path(wf_summary_path).exists())
        burn_in_scorecard: Optional[Dict[str, Any]] = None

        param_file = self.parameter_registry_path / f"{experiment_id}.json"
        criteria = [
            {
                "name": "required_validation_gates",
                "required": True,
                "passed": all(
                    gate.result == ValidationResult.PASS
                    for gate in exp.validation_gates
                    if gate.required
                ),
                "details": "All required validation gates must pass",
            },
            {
                "name": "parameter_snapshot_recorded",
                "required": True,
                "passed": len(exp.parameter_history) > 0 and param_file.exists(),
                "details": f"Parameter history entries: {len(exp.parameter_history)}",
            },
            {
                "name": "walk_forward_artifacts_stored",
                "required": True,
                "passed": wf_summary_exists,
                "details": wf_summary_path or "No walk-forward summary artifact found",
            },
            {
                "name": "config_hash_pinned",
                "required": True,
                "passed": bool(exp.config_hash),
                "details": exp.config_hash or "Missing config hash",
            },
            {
                "name": "git_commit_pinned",
                "required": True,
                "passed": bool(exp.git_commit),
                "details": exp.git_commit or "Missing git commit",
            },
            {
                "name": "paper_results_recorded",
                "required": False,
                "passed": exp.paper_results is not None,
                "details": "Recommended before live rollout",
            },
        ]

        if exp.paper_results is not None:
            paper = exp.paper_results or {}
            trading_days = float(paper.get("trading_days", 0) or 0)
            recon_rate = paper.get("reconciliation_pass_rate")
            if recon_rate is None:
                runs = float(paper.get("reconciliation_runs", 0) or 0)
                mismatches = float(paper.get("reconciliation_mismatch_count", 0) or 0)
                recon_rate = 1.0 if runs <= 0 else max(0.0, 1.0 - (mismatches / runs))
            recon_rate = float(recon_rate)

            clean_days = paper.get("clean_reconciliation_days")
            if clean_days is None:
                clean_days = trading_days if recon_rate >= 0.999 else 0
            clean_days = float(clean_days)

            error_rate = paper.get("operational_error_rate")
            if error_rate is None:
                decisions = float(paper.get("decision_events", 0) or 0)
                errors = float(paper.get("decision_errors", 0) or 0)
                error_rate = 0.0 if decisions <= 0 else max(0.0, errors / decisions)
            error_rate = float(error_rate)

            execution_metrics = extract_execution_quality_metrics(paper)
            execution_score = execution_metrics.get("execution_quality_score")
            avg_slippage_bps = execution_metrics.get("avg_actual_slippage_bps")
            fill_rate = execution_metrics.get("fill_rate")
            shadow_drift = extract_paper_live_shadow_drift(paper)

            min_recon = self.DEFAULT_GATES["paper_reconciliation_rate"]["threshold"]
            min_clean_days = 5
            max_error = self.DEFAULT_GATES["paper_operational_error_rate"]["threshold"]
            min_execution_score = self.DEFAULT_GATES["paper_execution_quality_score"]["threshold"]
            max_slippage_bps = self.DEFAULT_GATES["paper_avg_slippage_bps"]["threshold"]
            min_fill_rate = self.DEFAULT_GATES["paper_fill_rate"]["threshold"]
            max_shadow_drift = self.DEFAULT_GATES["paper_live_shadow_drift"]["threshold"]
            criteria.append(
                {
                    "name": "paper_kpi_reconciliation_rate",
                    "required": True,
                    "passed": recon_rate >= min_recon,
                    "details": (
                        f"reconciliation_pass_rate={recon_rate:.3f}, "
                        f"required>={min_recon:.3f}"
                    ),
                }
            )
            criteria.append(
                {
                    "name": "paper_kpi_clean_reconciliation_days",
                    "required": True,
                    "passed": clean_days >= min_clean_days,
                    "details": (
                        f"clean_reconciliation_days={clean_days:.0f}, "
                        f"required>={min_clean_days}"
                    ),
                }
            )
            criteria.append(
                {
                    "name": "paper_kpi_operational_error_rate",
                    "required": True,
                    "passed": error_rate <= max_error,
                    "details": (
                        f"operational_error_rate={error_rate:.4f}, "
                        f"required<={max_error:.4f}"
                    ),
                }
            )
            criteria.append(
                {
                    "name": "paper_execution_quality_score_ci_gate",
                    "required": True,
                    "passed": (
                        execution_score is not None and execution_score >= min_execution_score
                    ),
                    "details": (
                        f"execution_quality_score={execution_score:.1f}, "
                        f"required>={min_execution_score:.1f}"
                        if execution_score is not None
                        else "execution_quality_score missing in paper results"
                    ),
                }
            )
            criteria.append(
                {
                    "name": "paper_execution_avg_slippage_ci_gate",
                    "required": True,
                    "passed": (
                        avg_slippage_bps is not None and avg_slippage_bps <= max_slippage_bps
                    ),
                    "details": (
                        f"avg_actual_slippage_bps={avg_slippage_bps:.1f}, "
                        f"required<={max_slippage_bps:.1f}"
                        if avg_slippage_bps is not None
                        else "avg_actual_slippage_bps missing in paper results"
                    ),
                }
            )
            criteria.append(
                {
                    "name": "paper_execution_fill_rate_ci_gate",
                    "required": True,
                    "passed": fill_rate is not None and fill_rate >= min_fill_rate,
                    "details": (
                        f"fill_rate={fill_rate:.3f}, required>={min_fill_rate:.3f}"
                        if fill_rate is not None
                        else "fill_rate missing in paper results"
                    ),
                }
            )
            criteria.append(
                {
                    "name": "paper_live_shadow_drift_ci_gate",
                    "required": True,
                    "passed": shadow_drift is not None and shadow_drift <= max_shadow_drift,
                    "details": (
                        f"paper_live_shadow_drift={shadow_drift:.3f}, "
                        f"required<={max_shadow_drift:.3f}"
                        if shadow_drift is not None
                        else "paper_live_shadow_drift missing in paper results"
                    ),
                }
            )
            burn_in_scorecard = build_paper_burn_in_scorecard(paper)
            burn_in_blockers = burn_in_scorecard.get("blockers", [])
            burn_in_details = (
                f"score={float(burn_in_scorecard.get('score', 0.0)):.2%}; "
                f"ready={bool(burn_in_scorecard.get('ready_for_signoff', False))}"
            )
            if burn_in_blockers:
                burn_in_details = (
                    f"{burn_in_details}; blockers={'; '.join(str(item) for item in burn_in_blockers[:3])}"
                )
            criteria.append(
                {
                    "name": "paper_burn_in_signoff",
                    "required": True,
                    "passed": bool(burn_in_scorecard.get("ready_for_signoff", False)),
                    "details": burn_in_details,
                }
            )

        blockers = [
            f"{c['name']}: {c.get('details', '')}"
            for c in criteria
            if c.get("required") and not c.get("passed")
        ]
        ready = len(blockers) == 0

        checklist = {
            "experiment_id": experiment_id,
            "generated_at": datetime.now().isoformat(),
            "ready_for_promotion": ready,
            "criteria": criteria,
            "blockers": blockers,
            "burn_in_scorecard": burn_in_scorecard,
        }
        exp.promotion_checklist = checklist
        exp.updated_at = datetime.now()
        self._save_experiment(exp)
        return checklist

    def promote_to_production(
        self,
        experiment_id: str,
        force: bool = False,
    ) -> bool:
        """
        Promote experiment to production.

        Args:
            experiment_id: Experiment ID
            force: Force promotion even if gates fail

        Returns:
            True if promotion successful
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if not force and not self.is_promotion_ready(experiment_id):
            blockers = self.get_promotion_blockers(experiment_id)
            logger.warning(
                f"Cannot promote {experiment_id}: {len(blockers)} blocking gates: "
                f"{', '.join(blockers)}"
            )
            return False

        # Update status
        exp.status = ExperimentStatus.PROMOTED
        exp.promoted_at = datetime.now()
        exp.production_start = datetime.now()
        exp.updated_at = datetime.now()

        # Save to production registry
        prod_file = self.production_path / f"{experiment_id}.json"
        with open(prod_file, "w") as f:
            json.dump(exp.to_dict(), f, indent=2)

        self._save_experiment(exp)

        logger.info(
            f"Promoted {experiment_id} to production. "
            f"Force={force}, Commit={exp.git_commit}"
        )

        return True

    def deprecate_production(
        self,
        experiment_id: str,
        reason: str,
    ):
        """
        Remove experiment from production.

        Args:
            experiment_id: Experiment ID
            reason: Reason for deprecation
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        exp.status = ExperimentStatus.DEPRECATED
        exp.production_end = datetime.now()
        exp.updated_at = datetime.now()

        # Add deprecation note
        if not exp.production_results:
            exp.production_results = {}
        exp.production_results["deprecation_reason"] = reason
        exp.production_results["deprecated_at"] = datetime.now().isoformat()

        # Remove from production folder
        prod_file = self.production_path / f"{experiment_id}.json"
        if prod_file.exists():
            prod_file.rename(self.production_path / f"{experiment_id}.deprecated.json")

        self._save_experiment(exp)

        logger.info(f"Deprecated {experiment_id}: {reason}")

    def get_production_experiments(self) -> List[Experiment]:
        """Get all experiments currently in production."""
        return [
            exp for exp in self.experiments.values()
            if exp.status == ExperimentStatus.PROMOTED
        ]

    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get summary of an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Summary dict
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}

        gates_summary = {
            "total": len(exp.validation_gates),
            "passed": sum(1 for g in exp.validation_gates if g.result == ValidationResult.PASS),
            "failed": sum(1 for g in exp.validation_gates if g.result == ValidationResult.FAIL),
            "pending": sum(1 for g in exp.validation_gates if g.result == ValidationResult.PENDING),
        }

        return {
            "id": exp.id,
            "name": exp.name,
            "status": exp.status.value,
            "author": exp.author,
            "created": exp.created_at.isoformat(),
            "updated": exp.updated_at.isoformat(),
            "version": exp.version,
            "git_commit": exp.git_commit,
            "validation_gates": gates_summary,
            "promotion_ready": self.is_promotion_ready(experiment_id),
            "blockers": self.get_promotion_blockers(experiment_id),
        }

    def _check_backtest_gates(self, exp: Experiment, results: Dict[str, Any]):
        """Check backtest-related validation gates."""
        for gate in exp.validation_gates:
            if gate.name == "backtest_sharpe":
                sharpe = results.get("sharpe_ratio", 0)
                threshold = self.DEFAULT_GATES["backtest_sharpe"]["threshold"]
                if sharpe >= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Sharpe {sharpe:.2f} >= {threshold}"
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Sharpe {sharpe:.2f} < {threshold}"
                gate.checked_at = datetime.now()
                gate.metrics = {"sharpe_ratio": sharpe}

            elif gate.name == "backtest_drawdown":
                drawdown = abs(results.get("max_drawdown", 1.0))
                threshold = self.DEFAULT_GATES["backtest_drawdown"]["threshold"]
                if drawdown <= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Drawdown {drawdown:.1%} <= {threshold:.0%}"
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Drawdown {drawdown:.1%} > {threshold:.0%}"
                gate.checked_at = datetime.now()
                gate.metrics = {"max_drawdown": drawdown}

    def _check_walkforward_gates(self, exp: Experiment, results: Dict[str, Any]):
        """Check walk-forward validation gates."""
        for gate in exp.validation_gates:
            if gate.name == "walk_forward_oos":
                is_sharpe = results.get("in_sample_sharpe", 0)
                oos_sharpe = results.get("out_of_sample_sharpe", 0)
                threshold = self.DEFAULT_GATES["walk_forward_oos"]["threshold"]

                if is_sharpe > 0:
                    ratio = oos_sharpe / is_sharpe
                    if ratio >= threshold:
                        gate.result = ValidationResult.PASS
                        gate.message = f"OOS/IS ratio {ratio:.2f} >= {threshold}"
                    else:
                        gate.result = ValidationResult.FAIL
                        gate.message = f"OOS/IS ratio {ratio:.2f} < {threshold} (overfit)"
                else:
                    gate.result = ValidationResult.WARNING
                    gate.message = "In-sample Sharpe <= 0"

                gate.checked_at = datetime.now()
                gate.metrics = {
                    "in_sample_sharpe": is_sharpe,
                    "out_of_sample_sharpe": oos_sharpe,
                }

            elif gate.name == "statistical_significance":
                t_stat = results.get("alpha_t_stat", 0)
                threshold = self.DEFAULT_GATES["statistical_significance"]["threshold"]
                if t_stat >= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"T-stat {t_stat:.2f} >= {threshold}"
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"T-stat {t_stat:.2f} < {threshold}"
                gate.checked_at = datetime.now()
                gate.metrics = {"alpha_t_stat": t_stat}

    def _check_paper_gates(self, exp: Experiment, results: Dict[str, Any]):
        """Check paper trading validation gates."""
        execution_metrics = extract_execution_quality_metrics(results)
        for gate in exp.validation_gates:
            if gate.name == "paper_trading_days":
                days = results.get("trading_days", 0)
                threshold = self.DEFAULT_GATES["paper_trading_days"]["threshold"]
                if days >= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Paper traded {days} days >= {threshold}"
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Paper traded {days} days < {threshold}"
                gate.checked_at = datetime.now()
                gate.metrics = {"trading_days": days}

            elif gate.name == "paper_trading_profit":
                net_return = results.get("net_return", 0)
                if net_return >= 0:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Paper trading profitable: {net_return:.2%}"
                else:
                    gate.result = ValidationResult.WARNING
                    gate.message = f"Paper trading loss: {net_return:.2%}"
                gate.checked_at = datetime.now()
                gate.metrics = {"net_return": net_return}

            elif gate.name == "paper_reconciliation_rate":
                recon_rate = results.get("reconciliation_pass_rate")
                if recon_rate is None:
                    runs = float(results.get("reconciliation_runs", 0) or 0)
                    mismatches = float(results.get("reconciliation_mismatch_count", 0) or 0)
                    recon_rate = 1.0 if runs <= 0 else max(0.0, 1.0 - (mismatches / runs))
                recon_rate = float(recon_rate)
                threshold = self.DEFAULT_GATES["paper_reconciliation_rate"]["threshold"]
                if recon_rate >= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Reconciliation pass rate {recon_rate:.3f} >= {threshold:.3f}"
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Reconciliation pass rate {recon_rate:.3f} < {threshold:.3f}"
                gate.checked_at = datetime.now()
                gate.metrics = {"reconciliation_pass_rate": recon_rate}

            elif gate.name == "paper_operational_error_rate":
                error_rate = results.get("operational_error_rate")
                if error_rate is None:
                    decisions = float(results.get("decision_events", 0) or 0)
                    errors = float(results.get("decision_errors", 0) or 0)
                    error_rate = 0.0 if decisions <= 0 else max(0.0, errors / decisions)
                error_rate = float(error_rate)
                threshold = self.DEFAULT_GATES["paper_operational_error_rate"]["threshold"]
                if error_rate <= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Operational error rate {error_rate:.4f} <= {threshold:.4f}"
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Operational error rate {error_rate:.4f} > {threshold:.4f}"
                gate.checked_at = datetime.now()
                gate.metrics = {"operational_error_rate": error_rate}

            elif gate.name == "paper_execution_quality_score":
                score = execution_metrics.get("execution_quality_score")
                threshold = self.DEFAULT_GATES["paper_execution_quality_score"]["threshold"]
                if score is None:
                    gate.result = ValidationResult.WARNING
                    gate.message = "Execution quality score not provided"
                    gate.metrics = {}
                elif score >= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Execution quality score {score:.1f} >= {threshold:.1f}"
                    gate.metrics = {"execution_quality_score": score}
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Execution quality score {score:.1f} < {threshold:.1f}"
                    gate.metrics = {"execution_quality_score": score}
                gate.checked_at = datetime.now()

            elif gate.name == "paper_avg_slippage_bps":
                slippage_bps = execution_metrics.get("avg_actual_slippage_bps")
                threshold = self.DEFAULT_GATES["paper_avg_slippage_bps"]["threshold"]
                if slippage_bps is None:
                    gate.result = ValidationResult.WARNING
                    gate.message = "Average slippage (bps) not provided"
                    gate.metrics = {}
                elif slippage_bps <= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Average slippage {slippage_bps:.1f}bps <= {threshold:.1f}bps"
                    gate.metrics = {"avg_actual_slippage_bps": slippage_bps}
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Average slippage {slippage_bps:.1f}bps > {threshold:.1f}bps"
                    gate.metrics = {"avg_actual_slippage_bps": slippage_bps}
                gate.checked_at = datetime.now()

            elif gate.name == "paper_fill_rate":
                fill_rate = execution_metrics.get("fill_rate")
                threshold = self.DEFAULT_GATES["paper_fill_rate"]["threshold"]
                if fill_rate is None:
                    gate.result = ValidationResult.WARNING
                    gate.message = "Fill rate not provided"
                    gate.metrics = {}
                elif fill_rate >= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Fill rate {fill_rate:.3f} >= {threshold:.3f}"
                    gate.metrics = {"fill_rate": fill_rate}
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Fill rate {fill_rate:.3f} < {threshold:.3f}"
                    gate.metrics = {"fill_rate": fill_rate}
                gate.checked_at = datetime.now()

            elif gate.name == "paper_live_shadow_drift":
                drift = extract_paper_live_shadow_drift(results)
                threshold = self.DEFAULT_GATES["paper_live_shadow_drift"]["threshold"]
                if drift is None:
                    gate.result = ValidationResult.WARNING
                    gate.message = "Paper/live shadow drift metric not provided"
                    gate.metrics = {}
                elif drift <= threshold:
                    gate.result = ValidationResult.PASS
                    gate.message = f"Paper/live shadow drift {drift:.3f} <= {threshold:.3f}"
                    gate.metrics = {"paper_live_shadow_drift": drift}
                else:
                    gate.result = ValidationResult.FAIL
                    gate.message = f"Paper/live shadow drift {drift:.3f} > {threshold:.3f}"
                    gate.metrics = {"paper_live_shadow_drift": drift}
                gate.checked_at = datetime.now()

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash of configuration for reproducibility."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


def print_experiment_summary(registry: ResearchRegistry, experiment_id: str):
    """Print formatted experiment summary."""
    summary = registry.get_experiment_summary(experiment_id)

    if "error" in summary:
        print(f"Error: {summary['error']}")
        return

    exp = registry.experiments.get(experiment_id)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\nID: {summary['id']}")
    print(f"Name: {summary['name']}")
    print(f"Status: {summary['status'].upper()}")
    print(f"Author: {summary['author']}")
    print(f"Version: {summary['version']}")
    print(f"Git Commit: {summary['git_commit'] or 'N/A'}")

    print("\n--- VALIDATION GATES ---")
    gates = summary['validation_gates']
    print(f"Passed: {gates['passed']}/{gates['total']}")
    print(f"Failed: {gates['failed']}")
    print(f"Pending: {gates['pending']}")

    if exp:
        for gate in exp.validation_gates:
            status_icon = {
                ValidationResult.PASS: "[PASS]",
                ValidationResult.FAIL: "[FAIL]",
                ValidationResult.WARNING: "[WARN]",
                ValidationResult.PENDING: "[----]",
            }[gate.result]
            req = "*" if gate.required else " "
            print(f"  {status_icon}{req} {gate.name}: {gate.message or gate.description}")

    print("\n--- PROMOTION STATUS ---")
    if summary['promotion_ready']:
        print("Ready for production promotion")
    else:
        print("Blocked by:")
        for blocker in summary['blockers']:
            print(f"  - {blocker}")

    print("\n" + "=" * 60)
