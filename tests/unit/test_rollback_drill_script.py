#!/usr/bin/env python3
"""
Integration-style tests for rollback drill CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys


def test_rollback_drill_script_writes_report(tmp_path):
    output = tmp_path / "rollback_drill.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/rollback_drill.py",
            "--workdir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "RUNTIME ROLLBACK DRILL" in proc.stdout
    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
