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
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    ):
        """
        Initialize research registry.

        Args:
            registry_path: Path for experiment storage
            production_path: Path for production deployments
        """
        self.registry_path = Path(registry_path)
        self.production_path = Path(production_path)

        # Create directories
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.production_path.mkdir(parents=True, exist_ok=True)

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

        logger.info(f"Created experiment: {exp_id} - {name}")
        return exp_id

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

    def is_promotion_ready(self, experiment_id: str) -> bool:
        """
        Check if experiment is ready for production promotion.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if all required gates pass
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return False

        for gate in exp.validation_gates:
            if gate.required and gate.result != ValidationResult.PASS:
                return False

        return True

    def get_promotion_blockers(self, experiment_id: str) -> List[str]:
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

        return blockers

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
