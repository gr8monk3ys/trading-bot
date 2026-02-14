"""
Helpers for run-scoped observability artifacts.

This module provides lightweight utilities to:
- generate stable run IDs
- write/read JSON and JSONL files
- normalize Python objects for JSON serialization
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Literal, Mapping, Optional
from uuid import uuid4


def generate_run_id(prefix: str = "run", now: Optional[datetime] = None) -> str:
    """Generate a unique run ID."""
    dt = now or datetime.utcnow()
    return f"{prefix}_{dt.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def ensure_run_directory(base_dir: str | Path, run_id: str) -> Path:
    """Create and return a run directory under the base directory."""
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def to_jsonable(value: Any) -> Any:
    """Convert values to JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))

    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return to_jsonable(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON file with normalized content."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(to_jsonable(data), indent=2, sort_keys=True),
        encoding="utf-8",
    )


class JsonlWriter:
    """Append-only JSONL writer for structured event streams."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(to_jsonable(dict(record)), separators=(",", ":"))
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle and not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        self.close()
        return False


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON file, returning default if file is missing."""
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL file into a list of dicts."""
    p = Path(path)
    if not p.exists():
        return []

    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
