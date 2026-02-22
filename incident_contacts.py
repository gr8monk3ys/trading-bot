"""
Incident ownership/escalation contact validation helpers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_PLACEHOLDER_RE = re.compile(r"REPLACE_WITH_[A-Z0-9_]+")


def _scan_file(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not path.exists():
        findings.append(
            {
                "file": str(path),
                "line": 0,
                "token": "MISSING_FILE",
                "context": "File does not exist",
            }
        )
        return findings

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        for match in _PLACEHOLDER_RE.finditer(line):
            findings.append(
                {
                    "file": str(path),
                    "line": line_no,
                    "token": match.group(0),
                    "context": line,
                }
            )
    return findings


def validate_incident_contacts(
    ownership_doc: str | Path,
    escalation_doc: str | Path,
) -> dict[str, Any]:
    """
    Validate that incident ownership/escalation docs do not contain placeholders.
    """
    ownership_path = Path(ownership_doc)
    escalation_path = Path(escalation_doc)

    findings = _scan_file(ownership_path) + _scan_file(escalation_path)

    return {
        "ownership_doc": str(ownership_path),
        "escalation_doc": str(escalation_path),
        "valid": len(findings) == 0,
        "placeholder_count": len(findings),
        "findings": findings,
    }
