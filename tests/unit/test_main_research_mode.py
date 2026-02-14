import json
from argparse import Namespace
from pathlib import Path

from main import _load_json_object, run_research


def _base_args(tmp_path: Path, **overrides):
    values = {
        "research_action": "check",
        "experiment_id": None,
        "name": None,
        "description": None,
        "author": None,
        "tags": None,
        "parameters_json": None,
        "source": "manual",
        "notes": "",
        "reviewer": None,
        "backtest_json": None,
        "validation_json": None,
        "paper_json": None,
        "walk_forward_json": None,
        "source_run_id": None,
        "research_registry_path": str(tmp_path / "experiments"),
        "research_production_path": str(tmp_path / "production"),
        "research_parameter_registry_path": str(tmp_path / "parameters"),
        "research_artifacts_path": str(tmp_path / "artifacts"),
        "no_make_active": False,
        "strict": False,
        "output": None,
        "force": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _first_experiment_id(tmp_path: Path) -> str:
    registry_dir = tmp_path / "experiments"
    files = sorted(registry_dir.glob("*.json"))
    assert files
    return files[0].stem


def test_main_research_create_and_strict_check_flow(tmp_path):
    rc = run_research(
        _base_args(
            tmp_path,
            research_action="create",
            name="main_research_test",
            description="Main mode research flow",
            author="qa",
            parameters_json='{"lookback": 20}',
            tags="momentum,test",
        )
    )
    assert rc == 0

    exp_id = _first_experiment_id(tmp_path)

    # Strict check should fail until all criteria are satisfied.
    output_path = tmp_path / "check_initial.json"
    rc = run_research(
        _base_args(
            tmp_path,
            research_action="check",
            experiment_id=exp_id,
            strict=True,
            output=str(output_path),
        )
    )
    assert rc == 1
    assert output_path.exists()

    # Populate validation inputs.
    assert (
        run_research(
            _base_args(
                tmp_path,
                research_action="record-backtest",
                experiment_id=exp_id,
                backtest_json='{"sharpe_ratio": 1.5, "max_drawdown": -0.1}',
            )
        )
        == 0
    )
    assert (
        run_research(
            _base_args(
                tmp_path,
                research_action="record-validation",
                experiment_id=exp_id,
                validation_json='{"in_sample_sharpe": 2.0, "out_of_sample_sharpe": 1.4, "alpha_t_stat": 2.4}',
            )
        )
        == 0
    )
    assert (
        run_research(
            _base_args(
                tmp_path,
                research_action="record-paper",
                experiment_id=exp_id,
                paper_json=(
                    '{"trading_days": 70, "total_trades": 160, "net_return": 0.03, '
                    '"max_drawdown": -0.11, "reconciliation_pass_rate": 0.999, '
                    '"operational_error_rate": 0.005, "execution_quality_score": 82.0, '
                    '"avg_actual_slippage_bps": 14.0, "fill_rate": 0.97, '
                    '"paper_live_shadow_drift": 0.09, "critical_slo_breaches": 0}'
                ),
            )
        )
        == 0
    )
    assert (
        run_research(
            _base_args(
                tmp_path,
                research_action="approve-review",
                experiment_id=exp_id,
                reviewer="reviewer_1",
            )
        )
        == 0
    )
    assert (
        run_research(
            _base_args(
                tmp_path,
                research_action="store-walk-forward",
                experiment_id=exp_id,
                walk_forward_json='{"is_avg_sharpe": 2.0, "oos_avg_sharpe": 1.4, "alpha_t_stat": 2.4, "passes_validation": true}',
                source_run_id="backtest_demo_001",
            )
        )
        == 0
    )

    output_ready = tmp_path / "check_ready.json"
    rc = run_research(
        _base_args(
            tmp_path,
            research_action="check",
            experiment_id=exp_id,
            strict=True,
            output=str(output_ready),
        )
    )
    assert rc == 0
    assert output_ready.exists()


def test_main_research_promote_strict_blocks_without_requirements(tmp_path):
    run_research(
        _base_args(
            tmp_path,
            research_action="create",
            name="strict_block",
            description="Strict promote block test",
            author="qa",
        )
    )
    exp_id = _first_experiment_id(tmp_path)
    rc = run_research(
        _base_args(
            tmp_path,
            research_action="promote",
            experiment_id=exp_id,
            strict=True,
            force=False,
        )
    )
    assert rc == 1


def test_load_json_object_accepts_long_inline_json_payload():
    payload = {
        "trading_days": 70,
        "total_trades": 160,
        "notes": "x" * 400,
    }
    raw = json.dumps(payload)

    assert _load_json_object(raw, "paper_json") == payload
