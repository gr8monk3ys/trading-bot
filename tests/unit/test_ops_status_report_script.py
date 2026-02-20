from __future__ import annotations

import json
import subprocess
import sys


def test_ops_status_report_script_writes_outputs(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "incident_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event_type": "incident_open"}),
                json.dumps({"event_type": "incident_sla_breach"}),
                json.dumps({"event_type": "incident_ack"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "ops_slo_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event_type": "slo_breach", "severity": "critical"}),
                json.dumps({"event_type": "slo_breach", "severity": "warning"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "position_reconciliation_events.jsonl").write_text(
        json.dumps({"has_mismatch": True}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "order_reconciliation_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"mismatch_count": 0}),
                json.dumps({"mismatch_count": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_output = tmp_path / "ops_report.json"
    md_output = tmp_path / "ops_report.md"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/ops_status_report.py",
            "--run-dir",
            str(run_dir),
            "--json-output",
            str(json_output),
            "--md-output",
            str(md_output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Ops Status Report" in proc.stdout
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["incidents"]["opened"] == 1
    assert payload["slo"]["critical"] == 1
    assert payload["reconciliation"]["order_mismatch_runs"] == 1
    assert md_output.exists()
