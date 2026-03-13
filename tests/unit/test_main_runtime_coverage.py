from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import main


def _live_args(**overrides) -> Namespace:
    values = {
        "strategy": "MomentumStrategy",
        "run_id": "live_test_run",
        "artifacts_dir": "results/runs",
        "skip_validation": False,
        "validation_artifacts_dir": "results/validation",
        "go_live_precheck_summary_path": "results/validation/precheck/go_live_precheck_summary.json",
        "validation_max_age_hours": 168,
        "real": False,
        "enforce_governance_gate": True,
        "governance_approval_path": None,
        "governance_policy_doc_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _write_validation_bundle(
    base_dir: Path,
    *,
    git_sha: str,
    strategy: str = "MomentumStrategy",
    eligible_for_trading: bool = True,
    include_paper_summary: bool = False,
    paper_ready: bool = True,
    bundle_name: str = "20260313_010203",
    generated_at: str | None = None,
) -> Path:
    bundle_dir = base_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "git_sha": git_sha,
                "strategy": strategy,
                "eligible_for_trading": eligible_for_trading,
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "validated_backtest_result.json").write_text(
        json.dumps(
            {
                "eligible_for_trading": eligible_for_trading,
                "validation_gates": {"blockers": [] if eligible_for_trading else ["min_sharpe"]},
            }
        ),
        encoding="utf-8",
    )
    if include_paper_summary:
        (bundle_dir / "paper_trading_summary.json").write_text(
            json.dumps(
                {
                    "ready": paper_ready,
                    "blockers": [] if paper_ready else ["Insufficient paper trading time: 5 days"],
                }
            ),
            encoding="utf-8",
        )
    return bundle_dir


def test_validation_helper_functions_cover_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main.subprocess, "check_output", lambda *args, **kwargs: "abc123\n")
    assert main._current_git_sha(tmp_path) == "abc123"

    def _raise_git_error(*_args, **_kwargs):
        raise OSError("git metadata unavailable")

    monkeypatch.setattr(main.subprocess, "check_output", _raise_git_error)
    assert main._current_git_sha(tmp_path) is None

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    payload, error = main._load_json_dict(invalid_json)
    assert payload is None
    assert "Expecting property name" in error

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[]", encoding="utf-8")
    assert main._load_json_dict(non_dict) == (None, "expected a JSON object")

    timestamp_file = tmp_path / "timestamp.json"
    timestamp_file.write_text("{}", encoding="utf-8")
    explicit_timestamp = main._parse_report_timestamp(
        {"generated_at": "2026-03-13T12:34:56"},
        timestamp_file,
    )
    assert explicit_timestamp == datetime(2026, 3, 13, 12, 34, 56, tzinfo=timezone.utc)

    fallback_dt = datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc)
    fallback_epoch = fallback_dt.timestamp()
    os.utime(timestamp_file, (fallback_epoch, fallback_epoch))
    fallback_timestamp = main._parse_report_timestamp(
        {"generated_at": "not-a-timestamp"}, timestamp_file
    )
    assert fallback_timestamp == fallback_dt


def test_find_latest_validation_artifact_dir_prefers_latest_bundle(tmp_path: Path) -> None:
    assert main._find_latest_validation_artifact_dir(tmp_path / "missing") is None
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert main._find_latest_validation_artifact_dir(empty_root) is None

    root_bundle = _write_validation_bundle(tmp_path, git_sha="abc123", bundle_name="")
    assert main._find_latest_validation_artifact_dir(root_bundle) == root_bundle

    _write_validation_bundle(tmp_path, git_sha="abc123", bundle_name="20260310_010203")
    newer_bundle = _write_validation_bundle(
        tmp_path, git_sha="abc123", bundle_name="20260313_010203"
    )

    assert main._find_latest_validation_artifact_dir(tmp_path) == newer_bundle


@pytest.mark.asyncio
async def test_run_live_warns_when_skip_validation_flag_is_used(monkeypatch) -> None:
    warning = MagicMock()

    monkeypatch.setattr(main.logger, "warning", warning)
    monkeypatch.setattr(main, "generate_run_id", lambda prefix: f"{prefix}_run")
    monkeypatch.setattr(main, "ensure_run_directory", lambda *_args: Path("results/runs/live_run"))
    monkeypatch.setattr(
        main,
        "_evaluate_live_validation_gate",
        lambda **_kwargs: (True, {"ready": True, "skipped": True, "checks": []}),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "cancel")

    await main.run_live(_live_args(skip_validation=True, real=True))

    warning.assert_any_call("Live pre-trade validation skipped via --skip-validation")


def test_evaluate_live_validation_gate_skips_when_disabled(tmp_path: Path) -> None:
    ready, report = main._evaluate_live_validation_gate(
        enforce=False,
        validation_artifacts_dir=tmp_path / "missing",
        require_go_live_evidence=True,
    )

    assert ready is True
    assert report["skipped"] is True
    assert report["mode"] == "real"


def test_evaluate_live_validation_gate_passes_for_current_paper_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = _write_validation_bundle(tmp_path, git_sha="abc123")

    monkeypatch.setattr(main, "_current_git_sha", lambda _repo_root=".": "abc123")

    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        requested_strategy="MomentumStrategy",
        require_go_live_evidence=False,
        max_age_hours=168,
    )

    assert ready is True
    assert report["ready"] is True
    assert report["selected_validation_artifact_dir"] == str(bundle_dir)


def test_evaluate_live_validation_gate_allows_missing_git_sha_when_artifacts_are_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = _write_validation_bundle(tmp_path, git_sha="abc123")

    monkeypatch.setattr(main, "_current_git_sha", lambda _repo_root=".": None)

    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        requested_strategy="MomentumStrategy",
        require_go_live_evidence=False,
        max_age_hours=168,
    )

    assert ready is True
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["validation_git_sha_match"]["passed"] is True
    assert checks["validation_git_sha_match"]["severity"] == "warning"


def test_evaluate_live_validation_gate_rejects_missing_or_invalid_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "_current_git_sha", lambda _repo_root=".": "abc123")

    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=tmp_path / "missing",
        requested_strategy="MomentumStrategy",
    )
    assert ready is False
    assert any(check["name"] == "validated_backtest_artifacts" for check in report["checks"])

    bundle_dir = _write_validation_bundle(tmp_path, git_sha="abc123")
    (bundle_dir / "manifest.json").write_text("{", encoding="utf-8")
    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        requested_strategy="MomentumStrategy",
    )
    assert ready is False
    assert any(check["name"] == "validation_manifest" for check in report["checks"])

    bundle_dir = _write_validation_bundle(tmp_path, git_sha="abc123", bundle_name="20260313_020304")
    (bundle_dir / "validated_backtest_result.json").write_text("{", encoding="utf-8")
    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        requested_strategy="MomentumStrategy",
    )
    assert ready is False
    assert any(check["name"] == "validated_backtest_result" for check in report["checks"])


def test_evaluate_live_validation_gate_surfaces_stale_mismatch_and_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale_generated_at = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    bundle_dir = _write_validation_bundle(
        tmp_path,
        git_sha="abc123",
        strategy="MeanReversionStrategy",
        eligible_for_trading=False,
        generated_at=stale_generated_at,
    )

    monkeypatch.setattr(main, "_current_git_sha", lambda _repo_root=".": "different-sha")

    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        requested_strategy="MomentumStrategy",
        require_go_live_evidence=False,
        max_age_hours=1,
    )

    assert ready is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["validation_artifact_freshness"]["passed"] is False
    assert checks["validation_git_sha_match"]["passed"] is False
    assert checks["validated_backtest_eligibility"]["passed"] is False
    assert checks["validation_strategy_match"]["passed"] is False


def test_evaluate_live_validation_gate_blocks_real_startup_without_precheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = _write_validation_bundle(
        tmp_path,
        git_sha="abc123",
        include_paper_summary=True,
        paper_ready=True,
    )

    monkeypatch.setattr(main, "_current_git_sha", lambda _repo_root=".": "abc123")

    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        go_live_summary_path=tmp_path / "missing_precheck.json",
        requested_strategy="MomentumStrategy",
        require_go_live_evidence=True,
        max_age_hours=168,
    )

    assert ready is False
    assert report["ready"] is False
    assert any(
        check["name"] == "go_live_precheck" and check["passed"] is False
        for check in report["checks"]
    )


def test_evaluate_live_validation_gate_checks_real_readiness_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = _write_validation_bundle(
        tmp_path,
        git_sha="abc123",
        include_paper_summary=True,
        paper_ready=False,
    )
    go_live_summary_path = tmp_path / "go_live_precheck_summary.json"
    go_live_summary_path.write_text(
        json.dumps(
            {
                "generated_at": (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(),
                "ready": False,
                "steps": [{"name": "credentials", "passed": False}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "_current_git_sha", lambda _repo_root=".": "abc123")

    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        go_live_summary_path=go_live_summary_path,
        requested_strategy="MomentumStrategy",
        require_go_live_evidence=True,
        max_age_hours=1,
    )

    assert ready is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["paper_trading_readiness"]["passed"] is False
    assert checks["go_live_precheck_freshness"]["passed"] is False
    assert checks["go_live_precheck"]["passed"] is False


def test_evaluate_live_validation_gate_handles_real_summary_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    go_live_summary_path = tmp_path / "go_live_precheck_summary.json"
    go_live_summary_path.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "ready": True}),
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "_current_git_sha", lambda _repo_root=".": "abc123")

    bundle_dir = _write_validation_bundle(tmp_path, git_sha="abc123")
    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        go_live_summary_path=go_live_summary_path,
        requested_strategy="MomentumStrategy",
        require_go_live_evidence=True,
        max_age_hours=168,
    )
    assert ready is False
    assert any(
        check["name"] == "paper_trading_readiness" and check["passed"] is False
        for check in report["checks"]
    )

    bundle_dir = _write_validation_bundle(
        tmp_path,
        git_sha="abc123",
        include_paper_summary=True,
        bundle_name="20260313_020304",
    )
    (bundle_dir / "paper_trading_summary.json").write_text("{", encoding="utf-8")
    invalid_go_live_summary_path = tmp_path / "invalid_go_live_precheck_summary.json"
    invalid_go_live_summary_path.write_text("{", encoding="utf-8")

    ready, report = main._evaluate_live_validation_gate(
        enforce=True,
        repo_root=".",
        validation_artifacts_dir=bundle_dir,
        go_live_summary_path=invalid_go_live_summary_path,
        requested_strategy="MomentumStrategy",
        require_go_live_evidence=True,
        max_age_hours=168,
    )

    assert ready is False
    assert any(
        check["name"] == "paper_trading_readiness" and check["passed"] is False
        for check in report["checks"]
    )
    assert any(
        check["name"] == "go_live_precheck" and check["passed"] is False
        for check in report["checks"]
    )


@pytest.mark.asyncio
async def test_run_live_blocks_when_validation_gate_fails(monkeypatch) -> None:
    error = MagicMock()

    monkeypatch.setattr(main.logger, "error", error)
    monkeypatch.setattr(main, "generate_run_id", lambda prefix: f"{prefix}_run")
    monkeypatch.setattr(main, "ensure_run_directory", lambda *_args: Path("results/runs/live_run"))
    monkeypatch.setattr(
        main,
        "_evaluate_live_validation_gate",
        lambda **_kwargs: (
            False,
            {
                "ready": False,
                "skipped": False,
                "checks": [
                    {
                        "name": "validated_backtest_artifacts",
                        "passed": False,
                        "severity": "critical",
                        "message": "No validation artifact bundle found.",
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (_ for _ in ()).throw(
            AssertionError("confirmation should not be requested")
        ),
    )
    monkeypatch.setattr(
        main,
        "create_live_broker",
        AsyncMock(
            side_effect=AssertionError("broker creation should not happen when validation fails")
        ),
    )

    await main.run_live(_live_args(real=True))

    error.assert_any_call(
        "Live validation gate failed: %s",
        {
            "ready": False,
            "skipped": False,
            "checks": [
                {
                    "name": "validated_backtest_artifacts",
                    "passed": False,
                    "severity": "critical",
                    "message": "No validation artifact bundle found.",
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_optimize_parameters_logs_combination_counter(monkeypatch) -> None:
    info = MagicMock()

    class _FakeStrategy:
        def __init__(self, broker=None, symbols=None):
            self.broker = broker
            self.symbols = symbols or []

        def default_parameters(self):
            return {"lookback": 1}

    class _FakeStrategyManager:
        def __init__(self, broker=None):
            self.broker = broker
            self.available_strategies = {"momentum": _FakeStrategy}
            self.backtest_engine = SimpleNamespace(
                run_backtest=AsyncMock(return_value={"equity_curve": [100000.0, 101000.0]})
            )
            self.perf_metrics = SimpleNamespace(
                calculate_metrics=lambda _result: {
                    "sharpe_ratio": 1.0,
                    "total_return": 0.01,
                    "max_drawdown": -0.01,
                }
            )

        def get_available_strategy_names(self):
            return list(self.available_strategies.keys())

        def close(self):
            return None

    monkeypatch.setattr(main.logger, "info", info)
    monkeypatch.setattr(main, "AlpacaBroker", lambda paper=True: object())
    monkeypatch.setattr(main, "StrategyManager", _FakeStrategyManager)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    args = Namespace(
        strategy="momentum",
        start_date="2024-01-01",
        end_date="2024-01-31",
        symbols="AAPL",
        param_ranges='{"lookback": {"min": 1, "max": 1, "step": 1}}',
        capital=100000,
        execution_profile="realistic",
        optimize_for="sharpe",
    )

    await main.optimize_parameters(args)

    info.assert_any_call("Testing combination 1/1: {'lookback': 1}")
