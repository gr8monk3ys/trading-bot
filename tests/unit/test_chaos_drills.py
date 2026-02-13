#!/usr/bin/env python3
"""
Unit tests for operational chaos drill runner.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from utils.chaos_drills import run_chaos_drills


@pytest.mark.asyncio
async def test_run_chaos_drills_returns_passing_report():
    report = await run_chaos_drills()

    assert report["passed"] is True
    names = [c["name"] for c in report["checks"]]
    assert "order_reconciliation_fetch_failure" in names
    assert "data_quality_halt_on_critical_errors" in names
    assert "slo_alert_failure_tolerance" in names
    assert "websocket_reconnect_storm_tolerance" in names
    assert "partial_fill_stall_detection" in names
    assert "crash_recovery_idempotent_replay" in names
    assert all(c["passed"] is True for c in report["checks"])


def test_chaos_drill_script_writes_report(tmp_path):
    output_path = tmp_path / "chaos_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/chaos_drill.py",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "CHAOS DRILL REPORT" in proc.stdout
    assert output_path.exists()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
