"""
Runtime State Persistence for Live Trading.

Stores minimal, recoverable state needed across restarts:
- Position ownership/reservations (PositionManager)
- Active strategies/allocations (optional)

Durability guarantees:
- Atomic snapshot writes via write+fsync+rename
- Snapshot checksum validation on load
- Recovery fallback through replay journal and backup snapshots
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    """Serializable runtime state snapshot."""

    timestamp: datetime = field(default_factory=datetime.now)
    position_manager: Dict[str, Any] = field(default_factory=dict)
    active_strategies: Dict[str, Any] = field(default_factory=dict)
    allocations: Dict[str, float] = field(default_factory=dict)
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    gateway_state: Dict[str, Any] = field(default_factory=dict)
    strategy_states: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "position_manager": self.position_manager,
            "active_strategies": self.active_strategies,
            "allocations": self.allocations,
            "lifecycle": self.lifecycle,
            "gateway_state": self.gateway_state,
            "strategy_states": self.strategy_states,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeState":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            position_manager=data.get("position_manager", {}),
            active_strategies=data.get("active_strategies", {}),
            allocations=data.get("allocations", {}),
            lifecycle=data.get("lifecycle", {}),
            gateway_state=data.get("gateway_state", {}),
            strategy_states=data.get("strategy_states", {}),
        )


class RuntimeStateStore:
    """Persist and restore runtime state to a JSON file."""

    def __init__(self, path: str = "data/runtime_state.json"):
        self.path = Path(path)
        self._backup_path = self.path.with_suffix(f"{self.path.suffix}.bak")
        self._replay_path = self.path.with_suffix(".replay.jsonl")

    def exists(self) -> bool:
        return (
            self.path.exists()
            or self._backup_path.exists()
            or self._replay_path.exists()
        )

    @staticmethod
    def _payload_checksum(payload: Dict[str, Any]) -> str:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _build_envelope(self, state: RuntimeState) -> Dict[str, Any]:
        payload = state.to_dict()
        return {
            "format_version": 2,
            "checksum_algo": "sha256",
            "checksum": self._payload_checksum(payload),
            "payload": payload,
        }

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            self._fsync_directory(path.parent)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def _copy_with_fsync(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        with open(dst, "rb") as handle:
            os.fsync(handle.fileno())
        self._fsync_directory(dst.parent)

    def _append_replay_envelope(self, envelope: Dict[str, Any]) -> None:
        self._replay_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._replay_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(envelope, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _parse_blob(data: Any) -> RuntimeState:
        if not isinstance(data, dict):
            raise ValueError("Runtime state payload must be an object")

        # v2 envelope with checksum verification.
        if "payload" in data and "checksum" in data:
            payload = data.get("payload")
            checksum = str(data.get("checksum", "")).strip().lower()
            algo = str(data.get("checksum_algo", "sha256")).strip().lower()
            if algo != "sha256":
                raise ValueError(f"Unsupported checksum algorithm: {algo}")
            if not isinstance(payload, dict):
                raise ValueError("Envelope payload is missing or invalid")
            expected = RuntimeStateStore._payload_checksum(payload)
            if expected != checksum:
                raise ValueError("Runtime state checksum mismatch")
            return RuntimeState.from_dict(payload)

        # Backward compatibility: legacy direct RuntimeState JSON.
        return RuntimeState.from_dict(data)

    def _load_from_json_snapshot(self, path: Path) -> RuntimeState:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return self._parse_blob(data)

    def _load_from_replay_journal(self) -> RuntimeState:
        with open(self._replay_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        for line in reversed(lines):
            try:
                data = json.loads(line)
                return self._parse_blob(data)
            except Exception:
                continue
        raise ValueError("Replay journal has no valid runtime snapshot")

    async def save(
        self,
        position_manager,
        active_strategies: Optional[Dict[str, Any]] = None,
        allocations: Optional[Dict[str, float]] = None,
        lifecycle: Optional[Dict[str, Any]] = None,
        gateway_state: Optional[Dict[str, Any]] = None,
        strategy_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            state = RuntimeState(
                position_manager=await position_manager.export_state(),
                active_strategies=active_strategies or {},
                allocations=allocations or {},
                lifecycle=lifecycle or {},
                gateway_state=gateway_state or {},
                strategy_states=strategy_states or {},
            )
            envelope = self._build_envelope(state)
            if self.path.exists():
                self._copy_with_fsync(self.path, self._backup_path)

            self._atomic_write_json(self.path, envelope)
            self._append_replay_envelope(envelope)
            logger.info(f"Runtime state saved: {self.path}")
        except Exception as e:
            logger.error(f"Failed to save runtime state: {e}")

    async def load(self) -> Optional[RuntimeState]:
        if not self.exists():
            return None

        state: Optional[RuntimeState] = None
        recovered_from = None
        loaders = [
            ("primary", self.path, self._load_from_json_snapshot),
            ("replay", self._replay_path, lambda _: self._load_from_replay_journal()),
            ("backup", self._backup_path, self._load_from_json_snapshot),
        ]
        for source_name, source_path, loader in loaders:
            if not source_path.exists():
                continue
            try:
                state = loader(source_path)
                recovered_from = source_name
                break
            except Exception as e:
                logger.warning(f"Runtime state {source_name} unreadable: {e}")

        if state is None:
            logger.error("Failed to load runtime state from primary/replay/backup sources")
            return None

        if recovered_from and recovered_from != "primary":
            try:
                self._atomic_write_json(self.path, self._build_envelope(state))
                logger.warning(
                    "Recovered runtime state from %s and restored primary snapshot",
                    recovered_from,
                )
            except Exception as e:
                logger.error(f"Failed to restore primary runtime state from {recovered_from}: {e}")

        logger.info(f"Runtime state loaded: {self.path}")
        return state
