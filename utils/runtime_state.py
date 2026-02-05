"""
Runtime State Persistence for Live Trading.

Stores minimal, recoverable state needed across restarts:
- Position ownership/reservations (PositionManager)
- Active strategies/allocations (optional)
"""

import json
import logging
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
    strategy_states: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "position_manager": self.position_manager,
            "active_strategies": self.active_strategies,
            "allocations": self.allocations,
            "lifecycle": self.lifecycle,
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
            strategy_states=data.get("strategy_states", {}),
        )


class RuntimeStateStore:
    """Persist and restore runtime state to a JSON file."""

    def __init__(self, path: str = "data/runtime_state.json"):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    async def save(
        self,
        position_manager,
        active_strategies: Optional[Dict[str, Any]] = None,
        allocations: Optional[Dict[str, float]] = None,
        lifecycle: Optional[Dict[str, Any]] = None,
        strategy_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            state = RuntimeState(
                position_manager=await position_manager.export_state(),
                active_strategies=active_strategies or {},
                allocations=allocations or {},
                lifecycle=lifecycle or {},
                strategy_states=strategy_states or {},
            )
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.info(f"Runtime state saved: {self.path}")
        except Exception as e:
            logger.error(f"Failed to save runtime state: {e}")

    async def load(self) -> Optional[RuntimeState]:
        if not self.path.exists():
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = RuntimeState.from_dict(data)
            logger.info(f"Runtime state loaded: {self.path}")
            return state
        except Exception as e:
            logger.error(f"Failed to load runtime state: {e}")
            return None
