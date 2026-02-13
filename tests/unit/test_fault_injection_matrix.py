#!/usr/bin/env python3
"""
Unit tests for broker/API fault-injection matrix.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from utils.fault_injection_matrix import run_fault_injection_matrix


@pytest.mark.asyncio
async def test_fault_injection_matrix_returns_passing_report(tmp_path):
    report = await run_fault_injection_matrix(tmp_dir=str(tmp_path))
    assert report["passed"] is True
    names = [check["name"] for check in report["checks"]]
    assert "broker_submit_timeout_rejected" in names
    assert "data_quality_breach_opens_incident" in names
    assert "incident_ack_timeout_critical_slo" in names
    assert "shadow_drift_critical_threshold" in names
    assert "order_reconciliation_fetch_failure_health_signal" in names
    assert all(check["passed"] is True for check in report["checks"])


def test_fault_injection_matrix_script_writes_report(tmp_path):
    output = tmp_path / "fault_matrix.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/fault_injection_matrix.py",
            "--tmp-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "FAULT INJECTION MATRIX" in proc.stdout
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
