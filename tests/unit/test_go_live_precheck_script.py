from __future__ import annotations

import json
import subprocess


def test_go_live_precheck_local_only_passes(tmp_path):
    output_dir = tmp_path / "precheck"
    proc = subprocess.run(
        [
            "bash",
            "scripts/go_live_precheck.sh",
            "--repo-root",
            ".",
            "--output-dir",
            str(output_dir),
            "--local-only",
            "--skip-deployment-preflight",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "GO-LIVE PRECHECK SUMMARY" in proc.stdout
    summary = json.loads((output_dir / "go_live_precheck_summary.json").read_text(encoding="utf-8"))
    assert summary["ready"] is True
    steps = {step["name"]: step for step in summary["steps"]}
    assert steps["deployment_preflight"]["skipped"] is True
    assert steps["incident_contacts"]["passed"] is True
    assert steps["runtime_watchdog"]["passed"] is True
    assert steps["runtime_industrial_gate"]["passed"] is True


def test_go_live_precheck_returns_nonzero_on_invalid_repo_root(tmp_path):
    fake_repo = tmp_path / "missing-repo-shape"
    fake_repo.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "precheck"
    proc = subprocess.run(
        [
            "bash",
            "scripts/go_live_precheck.sh",
            "--repo-root",
            str(fake_repo),
            "--output-dir",
            str(output_dir),
            "--skip-deployment-preflight",
            "--skip-runtime-watchdog",
            "--skip-runtime-gate",
            "--no-enforce-ib-api-gate",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    summary = json.loads((output_dir / "go_live_precheck_summary.json").read_text(encoding="utf-8"))
    assert summary["ready"] is False
    steps = {step["name"]: step for step in summary["steps"]}
    assert steps["incident_contacts"]["passed"] is False


def test_go_live_precheck_blocks_skipping_watchdog_when_ib_gate_enforced(tmp_path):
    output_dir = tmp_path / "precheck"
    proc = subprocess.run(
        [
            "bash",
            "scripts/go_live_precheck.sh",
            "--repo-root",
            ".",
            "--output-dir",
            str(output_dir),
            "--skip-runtime-watchdog",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "runtime_watchdog cannot be skipped" in proc.stderr
