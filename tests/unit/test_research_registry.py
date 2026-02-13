"""
Tests for Research Registry

Tests:
- Experiment creation and lifecycle
- Validation gate checking
- Promotion workflow
- Result recording
"""

import os
import shutil
import tempfile
from datetime import datetime

import pytest

from research.research_registry import (
    Experiment,
    ExperimentStatus,
    ResearchRegistry,
    ValidationGate,
    ValidationResult,
    print_experiment_summary,
)


class TestResearchRegistry:
    """Tests for ResearchRegistry class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def registry(self, temp_dir):
        """Create a registry instance with temp directory."""
        return ResearchRegistry(
            registry_path=os.path.join(temp_dir, "experiments"),
            production_path=os.path.join(temp_dir, "production"),
        )

    def test_create_experiment(self, registry):
        """Test creating a new experiment."""
        exp_id = registry.create_experiment(
            name="test_momentum",
            description="Testing momentum factor",
            author="test_user",
            parameters={"lookback": 12},
            tags=["momentum", "test"],
        )

        assert exp_id is not None
        assert exp_id in registry.experiments

        exp = registry.experiments[exp_id]
        assert exp.name == "test_momentum"
        assert exp.status == ExperimentStatus.DRAFT
        assert exp.author == "test_user"
        assert exp.parameters == {"lookback": 12}
        assert "momentum" in exp.tags
        assert len(exp.parameter_history) == 1
        assert exp.parameter_history[0]["source"] == "create_experiment"

    def test_experiment_has_validation_gates(self, registry):
        """Test that new experiments have validation gates."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        exp = registry.experiments[exp_id]
        assert len(exp.validation_gates) > 0

        # Check for expected gates
        gate_names = [g.name for g in exp.validation_gates]
        assert "backtest_sharpe" in gate_names
        assert "walk_forward_oos" in gate_names
        assert "paper_trading_days" in gate_names
        assert "manual_review" in gate_names

    def test_record_backtest_results(self, registry):
        """Test recording backtest results."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        results = {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.15,
            "total_return": 0.25,
        }

        registry.record_backtest_results(exp_id, results)

        exp = registry.experiments[exp_id]
        assert exp.backtest_results == results
        assert exp.status == ExperimentStatus.BACKTEST

        # Check that gates were evaluated
        sharpe_gate = next(g for g in exp.validation_gates if g.name == "backtest_sharpe")
        assert sharpe_gate.result == ValidationResult.PASS

    def test_record_parameter_snapshot(self, registry):
        """Test adding parameter snapshots after creation."""
        exp_id = registry.create_experiment(
            name="snapshot_exp",
            description="Test snapshots",
            author="test_user",
            parameters={"lookback": 20},
        )
        snapshot = registry.record_parameter_snapshot(
            exp_id,
            parameters={"lookback": 30, "threshold": 1.2},
            source="optimizer",
            notes="Grid search winner",
            make_active=True,
        )

        exp = registry.experiments[exp_id]
        assert snapshot["source"] == "optimizer"
        assert snapshot["parameters"]["lookback"] == 30
        assert exp.parameters["lookback"] == 30
        assert len(exp.parameter_history) == 2

        param_file = registry.parameter_registry_path / f"{exp_id}.json"
        assert param_file.exists()

    def test_store_walk_forward_artifacts(self, registry):
        """Test walk-forward artifact storage and gate normalization."""
        exp_id = registry.create_experiment(
            name="wf_exp",
            description="Walk-forward artifact test",
            author="test_user",
        )

        paths = registry.store_walk_forward_artifacts(
            exp_id,
            walk_forward_results={
                "is_avg_sharpe": 2.0,
                "oos_avg_sharpe": 1.4,
                "alpha_t_stat": 2.3,
                "avg_overfit_ratio": 1.2,
                "passes_validation": True,
            },
            source_run_id="backtest_123",
        )

        assert os.path.exists(paths["results_path"])
        assert os.path.exists(paths["summary_path"])

        exp = registry.experiments[exp_id]
        assert "walk_forward_latest" in exp.artifact_index
        gate = next(g for g in exp.validation_gates if g.name == "walk_forward_oos")
        assert gate.result == ValidationResult.PASS

    def test_backtest_sharpe_gate_fail(self, registry):
        """Test that low Sharpe fails the gate."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        results = {
            "sharpe_ratio": 0.5,  # Below threshold of 1.0
            "max_drawdown": -0.10,
        }

        registry.record_backtest_results(exp_id, results)

        exp = registry.experiments[exp_id]
        sharpe_gate = next(g for g in exp.validation_gates if g.name == "backtest_sharpe")
        assert sharpe_gate.result == ValidationResult.FAIL

    def test_drawdown_gate(self, registry):
        """Test drawdown validation gate."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        results = {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.25,  # Above threshold of 0.20
        }

        registry.record_backtest_results(exp_id, results)

        exp = registry.experiments[exp_id]
        dd_gate = next(g for g in exp.validation_gates if g.name == "backtest_drawdown")
        assert dd_gate.result == ValidationResult.FAIL

    def test_record_validation_results(self, registry):
        """Test recording walk-forward validation results."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        results = {
            "in_sample_sharpe": 2.0,
            "out_of_sample_sharpe": 1.5,  # 75% of IS = good
            "alpha_t_stat": 2.5,
        }

        registry.record_validation_results(exp_id, results)

        exp = registry.experiments[exp_id]
        assert exp.validation_results == results
        assert exp.status == ExperimentStatus.VALIDATION

        wf_gate = next(g for g in exp.validation_gates if g.name == "walk_forward_oos")
        assert wf_gate.result == ValidationResult.PASS

    def test_walk_forward_overfit_detection(self, registry):
        """Test that overfitting is detected."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        results = {
            "in_sample_sharpe": 2.0,
            "out_of_sample_sharpe": 0.5,  # Only 25% of IS = overfit
            "alpha_t_stat": 2.5,
        }

        registry.record_validation_results(exp_id, results)

        exp = registry.experiments[exp_id]
        wf_gate = next(g for g in exp.validation_gates if g.name == "walk_forward_oos")
        assert wf_gate.result == ValidationResult.FAIL
        assert "overfit" in wf_gate.message.lower()

    def test_record_paper_results(self, registry):
        """Test recording paper trading results."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        results = {
            "trading_days": 25,
            "net_return": 0.05,
            "execution_quality_score": 80.0,
            "avg_actual_slippage_bps": 12.0,
            "fill_rate": 0.97,
        }

        registry.record_paper_results(exp_id, results)

        exp = registry.experiments[exp_id]
        assert exp.paper_results == results
        assert exp.status == ExperimentStatus.PAPER_TRADING

        days_gate = next(g for g in exp.validation_gates if g.name == "paper_trading_days")
        assert days_gate.result == ValidationResult.PASS
        recon_gate = next(g for g in exp.validation_gates if g.name == "paper_reconciliation_rate")
        assert recon_gate.result == ValidationResult.PASS
        err_gate = next(g for g in exp.validation_gates if g.name == "paper_operational_error_rate")
        exec_score_gate = next(g for g in exp.validation_gates if g.name == "paper_execution_quality_score")
        slippage_gate = next(g for g in exp.validation_gates if g.name == "paper_avg_slippage_bps")
        fill_gate = next(g for g in exp.validation_gates if g.name == "paper_fill_rate")
        assert err_gate.result == ValidationResult.PASS
        assert exec_score_gate.result == ValidationResult.PASS
        assert slippage_gate.result == ValidationResult.PASS
        assert fill_gate.result == ValidationResult.PASS

    def test_insufficient_paper_days(self, registry):
        """Test that insufficient paper trading days fails."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        results = {
            "trading_days": 10,  # Below threshold of 20
            "net_return": 0.05,
        }

        registry.record_paper_results(exp_id, results)

        exp = registry.experiments[exp_id]
        days_gate = next(g for g in exp.validation_gates if g.name == "paper_trading_days")
        assert days_gate.result == ValidationResult.FAIL

    def test_paper_kpi_gates_fail_for_poor_reconciliation_and_error_rate(self, registry):
        exp_id = registry.create_experiment(
            name="paper_kpi_fail",
            description="Test paper KPI failures",
            author="test",
        )

        registry.record_paper_results(
            exp_id,
            {
                "trading_days": 30,
                "net_return": 0.01,
                "reconciliation_pass_rate": 0.90,
                "operational_error_rate": 0.10,
            },
        )

        exp = registry.experiments[exp_id]
        recon_gate = next(g for g in exp.validation_gates if g.name == "paper_reconciliation_rate")
        err_gate = next(g for g in exp.validation_gates if g.name == "paper_operational_error_rate")
        assert recon_gate.result == ValidationResult.FAIL
        assert err_gate.result == ValidationResult.FAIL

    def test_approve_manual_review(self, registry):
        """Test manual review approval."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        registry.approve_manual_review(
            exp_id,
            reviewer="senior_quant",
            notes="Looks good",
        )

        exp = registry.experiments[exp_id]
        review_gate = next(g for g in exp.validation_gates if g.name == "manual_review")
        assert review_gate.result == ValidationResult.PASS
        assert "senior_quant" in review_gate.message

    def test_is_promotion_ready_false(self, registry):
        """Test that new experiment is not promotion ready."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        assert not registry.is_promotion_ready(exp_id)

    def test_is_promotion_ready_true(self, registry):
        """Test that fully validated experiment is promotion ready."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        # Pass all gates
        registry.record_backtest_results(exp_id, {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
        })

        registry.record_validation_results(exp_id, {
            "in_sample_sharpe": 2.0,
            "out_of_sample_sharpe": 1.5,
            "alpha_t_stat": 2.5,
        })

        registry.record_paper_results(exp_id, {
            "trading_days": 25,
            "net_return": 0.05,
            "execution_quality_score": 78.0,
            "avg_actual_slippage_bps": 16.0,
            "fill_rate": 0.96,
        })

        registry.approve_manual_review(exp_id, "reviewer")

        assert registry.is_promotion_ready(exp_id)

    def test_is_promotion_ready_strict_requires_artifacts(self, registry):
        """Strict promotion requires artifacts and checklist criteria."""
        exp_id = registry.create_experiment(
            name="strict_exp",
            description="Strict readiness test",
            author="test",
        )

        registry.record_backtest_results(exp_id, {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
        })
        registry.record_validation_results(exp_id, {
            "in_sample_sharpe": 2.0,
            "out_of_sample_sharpe": 1.5,
            "alpha_t_stat": 2.5,
        })
        registry.record_paper_results(exp_id, {
            "trading_days": 70,
            "total_trades": 140,
            "net_return": 0.05,
            "max_drawdown": -0.12,
            "reconciliation_pass_rate": 0.999,
            "operational_error_rate": 0.006,
            "execution_quality_score": 81.0,
            "avg_actual_slippage_bps": 13.0,
            "fill_rate": 0.97,
            "paper_live_shadow_drift": 0.08,
            "critical_slo_breaches": 0,
        })
        registry.approve_manual_review(exp_id, "reviewer")

        assert registry.is_promotion_ready(exp_id) is True
        assert registry.is_promotion_ready(exp_id, strict=True) is False

        registry.store_walk_forward_artifacts(
            exp_id,
            walk_forward_results={
                "is_avg_sharpe": 2.0,
                "oos_avg_sharpe": 1.5,
                "alpha_t_stat": 2.5,
                "passes_validation": True,
            },
        )

        assert registry.is_promotion_ready(exp_id, strict=True) is True

    def test_strict_checklist_requires_burn_in_signoff(self, registry):
        exp_id = registry.create_experiment(
            name="strict_burn_in",
            description="Strict checklist burn-in coverage",
            author="test",
        )
        registry.record_backtest_results(exp_id, {"sharpe_ratio": 1.5, "max_drawdown": -0.1})
        registry.record_validation_results(
            exp_id,
            {"in_sample_sharpe": 2.0, "out_of_sample_sharpe": 1.4, "alpha_t_stat": 2.3},
        )
        registry.record_paper_results(
            exp_id,
            {
                "trading_days": 30,
                "total_trades": 40,
                "net_return": 0.03,
                "execution_quality_score": 80.0,
                "avg_actual_slippage_bps": 12.0,
                "fill_rate": 0.97,
                "paper_live_shadow_drift": 0.08,
                "reconciliation_pass_rate": 0.998,
                "operational_error_rate": 0.004,
            },
        )
        registry.approve_manual_review(exp_id, "reviewer")
        registry.store_walk_forward_artifacts(
            exp_id,
            {
                "is_avg_sharpe": 2.0,
                "oos_avg_sharpe": 1.4,
                "alpha_t_stat": 2.3,
                "passes_validation": True,
            },
        )

        checklist = registry.generate_promotion_checklist(exp_id)

        assert checklist["ready_for_promotion"] is False
        assert any(
            "paper_burn_in_signoff" in blocker
            for blocker in checklist.get("blockers", [])
        )

    def test_generate_promotion_checklist(self, registry):
        """Checklist should include required criteria and blockers."""
        exp_id = registry.create_experiment(
            name="checklist_exp",
            description="Checklist test",
            author="test",
        )
        checklist = registry.generate_promotion_checklist(exp_id)
        assert checklist["experiment_id"] == exp_id
        assert "criteria" in checklist
        assert "blockers" in checklist
        assert checklist["ready_for_promotion"] is False

    def test_strict_checklist_requires_execution_quality_metrics(self, registry):
        exp_id = registry.create_experiment(
            name="strict_exec_quality",
            description="Strict checklist execution quality coverage",
            author="test",
        )
        registry.record_backtest_results(exp_id, {"sharpe_ratio": 1.5, "max_drawdown": -0.1})
        registry.record_validation_results(
            exp_id,
            {"in_sample_sharpe": 2.0, "out_of_sample_sharpe": 1.4, "alpha_t_stat": 2.3},
        )
        registry.record_paper_results(exp_id, {"trading_days": 25, "net_return": 0.03})
        registry.approve_manual_review(exp_id, "reviewer")
        registry.store_walk_forward_artifacts(
            exp_id,
            {
                "is_avg_sharpe": 2.0,
                "oos_avg_sharpe": 1.4,
                "alpha_t_stat": 2.3,
                "passes_validation": True,
            },
        )

        checklist = registry.generate_promotion_checklist(exp_id)
        blockers = checklist.get("blockers", [])

        assert checklist["ready_for_promotion"] is False
        assert any("paper_execution_quality_score_ci_gate" in blocker for blocker in blockers)

    def test_strict_checklist_requires_shadow_drift_metric(self, registry):
        exp_id = registry.create_experiment(
            name="strict_shadow_drift",
            description="Strict checklist shadow drift coverage",
            author="test",
        )
        registry.record_backtest_results(exp_id, {"sharpe_ratio": 1.5, "max_drawdown": -0.1})
        registry.record_validation_results(
            exp_id,
            {"in_sample_sharpe": 2.0, "out_of_sample_sharpe": 1.4, "alpha_t_stat": 2.3},
        )
        registry.record_paper_results(
            exp_id,
            {
                "trading_days": 25,
                "net_return": 0.03,
                "execution_quality_score": 81.0,
                "avg_actual_slippage_bps": 13.0,
                "fill_rate": 0.97,
            },
        )
        registry.approve_manual_review(exp_id, "reviewer")
        registry.store_walk_forward_artifacts(
            exp_id,
            {
                "is_avg_sharpe": 2.0,
                "oos_avg_sharpe": 1.4,
                "alpha_t_stat": 2.3,
                "passes_validation": True,
            },
        )

        checklist = registry.generate_promotion_checklist(exp_id)
        blockers = checklist.get("blockers", [])

        assert checklist["ready_for_promotion"] is False
        assert any("paper_live_shadow_drift_ci_gate" in blocker for blocker in blockers)

    def test_get_promotion_blockers(self, registry):
        """Test getting list of blocking gates."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        blockers = registry.get_promotion_blockers(exp_id)

        # Should have multiple blockers
        assert len(blockers) > 0
        assert any("backtest_sharpe" in b for b in blockers)

    def test_promote_to_production(self, registry):
        """Test promoting experiment to production."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        # Force promote
        result = registry.promote_to_production(exp_id, force=True)

        assert result is True
        exp = registry.experiments[exp_id]
        assert exp.status == ExperimentStatus.PROMOTED
        assert exp.promoted_at is not None

        # Check production file exists
        prod_file = registry.production_path / f"{exp_id}.json"
        assert prod_file.exists()

    def test_promote_blocked_without_force(self, registry):
        """Test that promotion fails without force if gates fail."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        result = registry.promote_to_production(exp_id, force=False)

        assert result is False
        exp = registry.experiments[exp_id]
        assert exp.status == ExperimentStatus.DRAFT

    def test_deprecate_production(self, registry):
        """Test deprecating a production experiment."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test",
            author="test",
        )

        # First promote
        registry.promote_to_production(exp_id, force=True)

        # Then deprecate
        registry.deprecate_production(exp_id, reason="Performance degraded")

        exp = registry.experiments[exp_id]
        assert exp.status == ExperimentStatus.DEPRECATED
        assert exp.production_end is not None
        assert exp.production_results["deprecation_reason"] == "Performance degraded"

    def test_get_production_experiments(self, registry):
        """Test getting list of production experiments."""
        # Create and promote two experiments
        exp1 = registry.create_experiment(name="exp1", description="1", author="test")
        exp2 = registry.create_experiment(name="exp2", description="2", author="test")
        registry.create_experiment(name="exp3", description="3", author="test")

        registry.promote_to_production(exp1, force=True)
        registry.promote_to_production(exp2, force=True)
        # exp3 not promoted

        production = registry.get_production_experiments()

        assert len(production) == 2
        assert any(e.id == exp1 for e in production)
        assert any(e.id == exp2 for e in production)

    def test_get_experiment_summary(self, registry):
        """Test getting experiment summary."""
        exp_id = registry.create_experiment(
            name="test_exp",
            description="Test experiment",
            author="test_user",
        )

        summary = registry.get_experiment_summary(exp_id)

        assert summary["id"] == exp_id
        assert summary["name"] == "test_exp"
        assert summary["status"] == "draft"
        assert "validation_gates" in summary
        assert summary["promotion_ready"] is False

    def test_persistence(self, temp_dir):
        """Test that experiments persist across registry instances."""
        # Create first registry and add experiment
        registry1 = ResearchRegistry(
            registry_path=os.path.join(temp_dir, "experiments"),
            production_path=os.path.join(temp_dir, "production"),
        )

        exp_id = registry1.create_experiment(
            name="persistent_exp",
            description="Should persist",
            author="test",
        )

        # Create second registry (should load saved experiment)
        registry2 = ResearchRegistry(
            registry_path=os.path.join(temp_dir, "experiments"),
            production_path=os.path.join(temp_dir, "production"),
        )

        assert exp_id in registry2.experiments
        assert registry2.experiments[exp_id].name == "persistent_exp"


class TestExperimentDataclass:
    """Tests for Experiment dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        exp = Experiment(
            id="test_123",
            name="Test Experiment",
            description="Description",
            author="author",
            status=ExperimentStatus.DRAFT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        d = exp.to_dict()

        assert d["id"] == "test_123"
        assert d["name"] == "Test Experiment"
        assert d["status"] == "draft"
        assert "created_at" in d
        assert "validation_gates" in d


class TestValidationGate:
    """Tests for ValidationGate dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        gate = ValidationGate(
            name="test_gate",
            description="Test gate",
            required=True,
            result=ValidationResult.PASS,
            message="Passed",
            checked_at=datetime.now(),
            metrics={"value": 1.5},
        )

        d = gate.to_dict()

        assert d["name"] == "test_gate"
        assert d["result"] == "pass"
        assert d["required"] is True
        assert d["metrics"] == {"value": 1.5}


class TestExperimentStatus:
    """Tests for ExperimentStatus enum."""

    def test_all_statuses(self):
        """Test that all expected statuses exist."""
        expected = [
            "DRAFT", "BACKTEST", "VALIDATION", "PAPER_TRADING",
            "REVIEW", "PROMOTED", "REJECTED", "DEPRECATED",
        ]

        for status in expected:
            assert hasattr(ExperimentStatus, status)


class TestValidationResult:
    """Tests for ValidationResult enum."""

    def test_all_results(self):
        """Test that all expected results exist."""
        expected = ["PASS", "FAIL", "WARNING", "PENDING"]

        for result in expected:
            assert hasattr(ValidationResult, result)


class TestPrintExperimentSummary:
    """Tests for print_experiment_summary function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_print_not_found(self, capsys, temp_dir):
        """Test printing summary for non-existent experiment."""
        registry = ResearchRegistry(
            registry_path=os.path.join(temp_dir, "experiments"),
            production_path=os.path.join(temp_dir, "production"),
        )

        print_experiment_summary(registry, "nonexistent_id")

        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_print_valid_experiment(self, capsys, temp_dir):
        """Test printing summary for valid experiment."""
        registry = ResearchRegistry(
            registry_path=os.path.join(temp_dir, "experiments"),
            production_path=os.path.join(temp_dir, "production"),
        )

        exp_id = registry.create_experiment(
            name="print_test",
            description="Test print",
            author="test",
        )

        print_experiment_summary(registry, exp_id)

        captured = capsys.readouterr()
        assert "EXPERIMENT SUMMARY" in captured.out
        assert "print_test" in captured.out
        assert "VALIDATION GATES" in captured.out
