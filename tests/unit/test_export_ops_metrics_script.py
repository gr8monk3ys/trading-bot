from __future__ import annotations

import json
import subprocess
import sys


def test_export_ops_metrics_script_writes_json_and_prom(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "incident_events.jsonl").write_text('{"event_type":"incident_open"}\n', encoding="utf-8")
    (run_dir / "ops_slo_events.jsonl").write_text(
        '{"event_type":"slo_breach","severity":"critical"}\n',
        encoding="utf-8",
    )

    json_output = tmp_path / "ops_metrics.json"
    prom_output = tmp_path / "ops_metrics.prom"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/export_ops_metrics.py",
            "--run-dir",
            str(run_dir),
            "--json-output",
            str(json_output),
            "--prom-output",
            str(prom_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "OPS METRICS EXPORT" in proc.stdout

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["metrics"]["incidents_opened_total"] == 1
    assert payload["metrics"]["slo_breaches_critical_total"] == 1

    prom_text = prom_output.read_text(encoding="utf-8")
    assert "trading_bot_incidents_opened_total 1" in prom_text
    assert "trading_bot_slo_breaches_critical_total 1" in prom_text
