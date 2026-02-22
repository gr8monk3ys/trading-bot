from __future__ import annotations

import json
import subprocess
import sys


def test_push_ops_metrics_script_writes_outputs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    watchdog = tmp_path / "runtime_watchdog.json"
    watchdog.write_text(json.dumps({"ready": True}), encoding="utf-8")
    json_output = tmp_path / "ops_metrics.json"
    prom_output = tmp_path / "ops_metrics.prom"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/push_ops_metrics.py",
            "--run-dir",
            str(run_dir),
            "--runtime-watchdog-json",
            str(watchdog),
            "--json-output",
            str(json_output),
            "--prom-output",
            str(prom_output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert json_output.exists()
    assert prom_output.exists()
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["metrics"]["runtime_watchdog_ready"] == 1
    prom_text = prom_output.read_text(encoding="utf-8")
    assert "trading_bot_runtime_watchdog_ready 1" in prom_text


def test_push_ops_metrics_script_fail_on_push_error(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/push_ops_metrics.py",
            "--run-dir",
            str(run_dir),
            "--pushgateway-url",
            "http://127.0.0.1:1",
            "--fail-on-push-error",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "Push Status: FAILED" in proc.stdout
