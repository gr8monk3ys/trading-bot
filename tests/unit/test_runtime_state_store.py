#!/usr/bin/env python3
"""
Unit tests for atomic runtime state persistence and recovery fallback.
"""

from __future__ import annotations

import json

import pytest

from utils.runtime_state import RuntimeStateStore


class _DummyPositionManager:
    def __init__(self, state):
        self._state = state

    async def export_state(self):
        return self._state


@pytest.mark.asyncio
async def test_runtime_state_store_roundtrip_writes_checksum_envelope(tmp_path):
    path = tmp_path / "runtime_state.json"
    store = RuntimeStateStore(str(path))
    pm = _DummyPositionManager({"ownership": {"AAPL": {"strategy": "S1"}}})

    await store.save(
        position_manager=pm,
        active_strategies={"S1": "running"},
        allocations={"S1": 0.5},
        lifecycle={"ord-1": {"state": "submitted"}},
        gateway_state={"halt_reason": "test_halt", "trading_halted_until": "2026-01-01T00:00:00"},
        strategy_states={"S1": {"version": 2}},
    )

    assert path.exists()
    replay_path = path.with_suffix(".replay.jsonl")
    assert replay_path.exists()

    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["format_version"] == 2
    assert envelope["checksum_algo"] == "sha256"
    assert "payload" in envelope
    assert envelope["checksum"] == RuntimeStateStore._payload_checksum(envelope["payload"])

    loaded = await store.load()
    assert loaded is not None
    assert loaded.active_strategies["S1"] == "running"
    assert loaded.allocations["S1"] == 0.5
    assert loaded.lifecycle["ord-1"]["state"] == "submitted"
    assert loaded.gateway_state["halt_reason"] == "test_halt"
    assert loaded.strategy_states["S1"]["version"] == 2
    assert loaded.position_manager["ownership"]["AAPL"]["strategy"] == "S1"


@pytest.mark.asyncio
async def test_runtime_state_store_checksum_mismatch_falls_back_to_replay_and_heals_primary(
    tmp_path,
):
    path = tmp_path / "runtime_state.json"
    store = RuntimeStateStore(str(path))
    pm = _DummyPositionManager({"ownership": {"AAPL": {"strategy": "S2"}}})

    await store.save(
        position_manager=pm,
        active_strategies={"S2": "running"},
    )

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["checksum"] = "deadbeef"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    loaded = await store.load()
    assert loaded is not None
    assert loaded.active_strategies["S2"] == "running"

    healed = json.loads(path.read_text(encoding="utf-8"))
    assert healed["checksum"] == RuntimeStateStore._payload_checksum(healed["payload"])


@pytest.mark.asyncio
async def test_runtime_state_store_backup_fallback_when_primary_and_replay_are_corrupt(tmp_path):
    path = tmp_path / "runtime_state.json"
    store = RuntimeStateStore(str(path))

    await store.save(
        position_manager=_DummyPositionManager({"ownership": {"AAPL": {"strategy": "old"}}}),
        active_strategies={"Old": "running"},
    )
    await store.save(
        position_manager=_DummyPositionManager({"ownership": {"AAPL": {"strategy": "new"}}}),
        active_strategies={"New": "running"},
    )

    path.write_text("{corrupt", encoding="utf-8")
    path.with_suffix(".replay.jsonl").write_text("{bad json line}\n", encoding="utf-8")

    loaded = await store.load()
    assert loaded is not None
    # Backup represents the previous successful snapshot.
    assert loaded.active_strategies["Old"] == "running"
    assert loaded.position_manager["ownership"]["AAPL"]["strategy"] == "old"


@pytest.mark.asyncio
async def test_runtime_state_store_returns_none_when_all_sources_invalid(tmp_path):
    path = tmp_path / "runtime_state.json"
    store = RuntimeStateStore(str(path))
    path.write_text("{broken", encoding="utf-8")
    path.with_suffix(".json.bak").write_text("{broken", encoding="utf-8")
    path.with_suffix(".replay.jsonl").write_text("{broken", encoding="utf-8")

    loaded = await store.load()
    assert loaded is None


@pytest.mark.asyncio
async def test_runtime_state_store_exists_with_replay_only(tmp_path):
    path = tmp_path / "runtime_state.json"
    store = RuntimeStateStore(str(path))
    await store.save(
        position_manager=_DummyPositionManager({"ownership": {"AAPL": {"strategy": "S3"}}}),
        active_strategies={"S3": "running"},
    )

    path.unlink()
    backup_path = path.with_suffix(".json.bak")
    if backup_path.exists():
        backup_path.unlink()

    assert store.exists() is True
    loaded = await store.load()
    assert loaded is not None
    assert loaded.active_strategies["S3"] == "running"
