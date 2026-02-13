#!/usr/bin/env python3
"""
Integration-style tests for paper burn-in scorecard CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys

from research.research_registry import ResearchRegistry


def test_paper_burn_in_scorecard_script_fails_when_not_ready(tmp_path):
    paper_results_path = tmp_path / "paper_results.json"
    paper_results_path.write_text(
        json.dumps(
            {
                "trading_days": 10,
                "total_trades": 25,
                "execution_quality_score": 62.0,
                "paper_live_shadow_drift": 0.22,
            }
        ),
        encoding="utf-8",
    )
    json_output = tmp_path / "burn_in_scorecard.json"
    md_output = tmp_path / "burn_in_scorecard.md"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/paper_burn_in_scorecard.py",
            "--paper-results-json",
            str(paper_results_path),
            "--json-output",
            str(json_output),
            "--md-output",
            str(md_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "Paper Burn-In Scorecard" in proc.stdout
    assert json_output.exists()
    assert md_output.exists()

    scorecard = json.loads(json_output.read_text(encoding="utf-8"))
    assert scorecard["ready_for_signoff"] is False
    assert scorecard["blockers"]


def test_paper_burn_in_scorecard_script_supports_experiment_lookup(tmp_path):
    paths = {
        "registry_path": str(tmp_path / "experiments"),
        "production_path": str(tmp_path / "production"),
        "parameter_registry_path": str(tmp_path / "parameters"),
        "artifacts_path": str(tmp_path / "artifacts"),
    }
    registry = ResearchRegistry(**paths)
    exp_id = registry.create_experiment(
        name="burn_in_cli",
        description="burn-in cli test",
        author="test_user",
    )
    registry.record_paper_results(
        exp_id,
        {
            "trading_days": 90,
            "total_trades": 180,
            "max_drawdown": -0.08,
            "reconciliation_pass_rate": 0.999,
            "operational_error_rate": 0.004,
            "execution_quality_score": 85.0,
            "avg_actual_slippage_bps": 11.0,
            "fill_rate": 0.98,
            "paper_live_shadow_drift": 0.05,
            "critical_slo_breaches": 0,
            "manual_signoff_approved": True,
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/paper_burn_in_scorecard.py",
            "--experiment-id",
            exp_id,
            "--registry-path",
            paths["registry_path"],
            "--production-path",
            paths["production_path"],
            "--parameter-registry-path",
            paths["parameter_registry_path"],
            "--artifacts-path",
            paths["artifacts_path"],
            "--require-manual-signoff",
            "--fail-on",
            "not_ready",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Status: **READY**" in proc.stdout
