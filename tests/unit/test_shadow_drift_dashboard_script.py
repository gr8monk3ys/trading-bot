#!/usr/bin/env python3
"""
Integration-style tests for shadow drift dashboard CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys


def test_shadow_drift_dashboard_script_outputs_files_and_fails_on_critical(tmp_path):
    paper_results_path = tmp_path / "paper_results.json"
    paper_results_path.write_text(
        json.dumps(
            {
                "paper_live_shadow_drift": 0.21,
                "execution_quality_score": 66.0,
                "avg_actual_slippage_bps": 18.0,
                "fill_rate": 0.96,
            }
        ),
        encoding="utf-8",
    )
    json_output = tmp_path / "shadow_dashboard.json"
    md_output = tmp_path / "shadow_dashboard.md"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/shadow_drift_dashboard.py",
            "--paper-results-json",
            str(paper_results_path),
            "--critical-threshold",
            "0.15",
            "--warning-threshold",
            "0.10",
            "--fail-on",
            "critical",
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
    assert "Shadow Drift Dashboard" in proc.stdout
    assert json_output.exists()
    assert md_output.exists()

    dashboard = json.loads(json_output.read_text(encoding="utf-8"))
    assert dashboard["status"] == "critical"
    assert dashboard["alert"] is True
