import json
import subprocess
import sys
from pathlib import Path

from research.research_registry import ResearchRegistry


def _registry_paths(tmp_path: Path):
    return {
        "registry_path": str(tmp_path / "experiments"),
        "production_path": str(tmp_path / "production"),
        "parameter_registry_path": str(tmp_path / "parameters"),
        "artifacts_path": str(tmp_path / "artifacts"),
    }


def test_promotion_gate_fails_when_not_ready(tmp_path):
    paths = _registry_paths(tmp_path)
    registry = ResearchRegistry(**paths)
    exp_id = registry.create_experiment(
        name="gate_fail",
        description="Should fail gate",
        author="test_user",
    )

    cmd = [
        sys.executable,
        "scripts/strategy_promotion_gate.py",
        "--experiment-id",
        exp_id,
        "--strict",
        "--registry-path",
        paths["registry_path"],
        "--production-path",
        paths["production_path"],
        "--parameter-registry-path",
        paths["parameter_registry_path"],
        "--artifacts-path",
        paths["artifacts_path"],
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "Ready: NO" in proc.stdout


def test_promotion_gate_passes_when_strict_ready(tmp_path):
    paths = _registry_paths(tmp_path)
    registry = ResearchRegistry(**paths)
    exp_id = registry.create_experiment(
        name="gate_pass",
        description="Should pass gate",
        author="test_user",
        parameters={"lookback": 20},
    )

    registry.record_backtest_results(exp_id, {"sharpe_ratio": 1.5, "max_drawdown": -0.1})
    registry.record_validation_results(
        exp_id,
        {
            "in_sample_sharpe": 2.0,
            "out_of_sample_sharpe": 1.4,
            "alpha_t_stat": 2.4,
        },
    )
    registry.record_paper_results(
        exp_id,
        {
            "trading_days": 70,
            "total_trades": 150,
            "net_return": 0.02,
            "max_drawdown": -0.10,
            "reconciliation_pass_rate": 0.999,
            "operational_error_rate": 0.005,
            "execution_quality_score": 79.0,
            "avg_actual_slippage_bps": 15.0,
            "fill_rate": 0.96,
            "paper_live_shadow_drift": 0.07,
            "critical_slo_breaches": 0,
        },
    )
    registry.approve_manual_review(exp_id, "reviewer")
    registry.store_walk_forward_artifacts(
        exp_id,
        {
            "is_avg_sharpe": 2.0,
            "oos_avg_sharpe": 1.4,
            "alpha_t_stat": 2.4,
            "passes_validation": True,
        },
    )

    output_path = tmp_path / "promotion_checklist.json"
    cmd = [
        sys.executable,
        "scripts/strategy_promotion_gate.py",
        "--experiment-id",
        exp_id,
        "--strict",
        "--registry-path",
        paths["registry_path"],
        "--production-path",
        paths["production_path"],
        "--parameter-registry-path",
        paths["parameter_registry_path"],
        "--artifacts-path",
        paths["artifacts_path"],
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Ready: YES" in proc.stdout
    assert "Burn-In Signoff: READY" in proc.stdout
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    burn_in_scorecard = payload.get("burn_in_scorecard")
    assert isinstance(burn_in_scorecard, dict)
    assert burn_in_scorecard["ready_for_signoff"] is True
