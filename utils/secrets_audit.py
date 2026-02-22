"""
Secrets rotation and repository leak audit helpers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SecretsAuditCheck:
    name: str
    passed: bool
    message: str
    severity: str = "critical"
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "details": self.details or {},
        }


_SECRET_VALUE_PATTERN = re.compile(r"""(?ix)
    \b
    (?:alpaca[_-]?(?:api[_-]?key|secret[_-]?key)|api[_-]?key|api[_-]?secret|secret[_-]?key)
    \b
    \s*[:=]\s*
    ["']?
    ([A-Za-z0-9][A-Za-z0-9_\-]{15,})
    ["']?
    """)


def _looks_like_placeholder(line: str) -> bool:
    lower = line.lower()
    tokens = ("replace_with", "example", "<", "placeholder", "dummy", "sample")
    return any(token in lower for token in tokens)


def _scan_file_for_secret_hits(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []

    hits: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _looks_like_placeholder(line):
            continue
        match = _SECRET_VALUE_PATTERN.search(line)
        if not match:
            continue
        value = match.group(1).strip()
        if len(value) < 16:
            continue
        if value.isupper():
            continue
        hits.append(
            {
                "file": str(path),
                "line": lineno,
                "excerpt": line[:180],
            }
        )
    return hits


def _iter_repo_files(repo_root: Path) -> list[Path]:
    ignored = {
        ".git",
        ".venv",
        ".pytest_cache",
        "__pycache__",
        "htmlcov",
        ".mypy_cache",
        ".ruff_cache",
    }
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored for part in path.parts):
            continue
        if path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".pdf",
            ".zip",
            ".gz",
            ".pyc",
            ".so",
        }:
            continue
        if path.stat().st_size > 512_000:
            continue
        files.append(path)
    return files


def _parse_inventory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict):
        secrets = payload.get("secrets", [])
        if isinstance(secrets, list):
            return [item for item in secrets if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _days_since_iso_date(value: str) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = date.fromisoformat(raw)
    except ValueError:
        return None
    today = datetime.now(timezone.utc).date()
    return (today - parsed).days


def run_secrets_audit(
    *,
    repo_root: str | Path = ".",
    inventory_path: str | Path = "docs/SECRETS_ROTATION_INVENTORY.json",
    default_max_age_days: int = 90,
) -> dict[str, Any]:
    """
    Audit repository for leaked secrets and stale rotation inventory entries.
    """
    root = Path(repo_root).resolve()
    inventory_file = Path(inventory_path)
    if not inventory_file.is_absolute():
        inventory_file = (root / inventory_file).resolve()

    checks: list[SecretsAuditCheck] = []

    inventory = _parse_inventory(inventory_file)
    checks.append(
        SecretsAuditCheck(
            name="secrets_inventory_present",
            passed=inventory_file.exists(),
            message=(
                f"Secrets inventory found: {inventory_file}"
                if inventory_file.exists()
                else f"Missing secrets inventory: {inventory_file}"
            ),
        )
    )
    checks.append(
        SecretsAuditCheck(
            name="secrets_inventory_entries_present",
            passed=len(inventory) > 0,
            message=(
                f"Secrets inventory contains {len(inventory)} entries"
                if inventory
                else "Secrets inventory is empty or invalid"
            ),
        )
    )

    stale_entries: list[dict[str, Any]] = []
    for item in inventory:
        secret_name = str(item.get("name", "")).strip()
        max_age_days = int(item.get("max_age_days", default_max_age_days) or default_max_age_days)
        age_days = _days_since_iso_date(str(item.get("last_rotated", "")).strip())
        if not secret_name:
            stale_entries.append(
                {
                    "name": "",
                    "reason": "missing_name",
                }
            )
            continue
        if age_days is None:
            stale_entries.append(
                {
                    "name": secret_name,
                    "reason": "invalid_or_missing_last_rotated",
                }
            )
            continue
        if age_days > max(1, max_age_days):
            stale_entries.append(
                {
                    "name": secret_name,
                    "reason": "rotation_overdue",
                    "age_days": age_days,
                    "max_age_days": max_age_days,
                }
            )

    checks.append(
        SecretsAuditCheck(
            name="secrets_rotation_fresh",
            passed=len(stale_entries) == 0,
            message=(
                "All tracked secrets are within rotation windows"
                if not stale_entries
                else f"{len(stale_entries)} secrets are stale or invalid in inventory"
            ),
            details={"stale_entries": stale_entries},
        )
    )

    leak_hits: list[dict[str, Any]] = []
    for file_path in _iter_repo_files(root):
        rel_path = file_path.relative_to(root)
        if rel_path.as_posix().startswith("docs/") and rel_path.name.endswith(".md"):
            continue
        if rel_path.as_posix().startswith("tests/"):
            continue
        if rel_path.as_posix() == ".env.example":
            continue
        for hit in _scan_file_for_secret_hits(file_path):
            hit["file"] = str(rel_path)
            leak_hits.append(hit)

    checks.append(
        SecretsAuditCheck(
            name="repository_secret_leak_scan",
            passed=len(leak_hits) == 0,
            message=(
                "No probable inline secrets found in repository files"
                if not leak_hits
                else f"Detected {len(leak_hits)} potential inline secret leaks"
            ),
            details={"hits": leak_hits[:50]},
        )
    )

    ready = all(check.passed for check in checks if check.severity == "critical")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "checks": [check.to_dict() for check in checks],
        "inventory_path": str(inventory_file),
        "repo_root": str(root),
    }
