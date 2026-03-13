# Runtime Architecture

This repository supports multiple execution paths. They overlap in purpose, but they are not interchangeable.

## Current Entry Points

### `main.py`

Primary multi-mode CLI:

- `live`
- `backtest`
- `optimize`
- `replay`
- `research`

Use this when you want the broadest built-in operational surface: strategy manager, replay artifacts, governance gate, run registry, and research tooling.

### `live_trader.py`

Single-strategy runtime focused on direct paper/live session management with explicit risk-profile controls.

Use this when you want:

- one strategy at a time
- direct symbol selection
- low-resource launches
- the risk-profile presets used by `scripts/run_low_resource_profile.py`

### `run_adaptive.py`

Standalone adaptive-strategy runner with its own scanner, backtest mode, and live mode.

This is a separate path from `main.py`. It is not wired into the `main.py` research/replay/governance surface.

### `web/app.py`

FastAPI monitoring app that serves:

- HTML dashboard
- JSON API endpoints for health, account, positions, trades, performance, and market status

This app observes broker/database state. It is not the execution control plane.

### `start.py`

Deployment wrapper that launches:

- `uvicorn web.app:app`
- `run_adaptive.py`

Use this for deployment-style environments where both processes should run together. It is not the primary local CLI.

## What the Code Currently Does Well

- Backtesting and validation paths are heavily covered by tests
- The live runtimes include substantial risk, reconciliation, and incident/SLO machinery
- The monitoring surface can operate even when broker/database initialization is partially unavailable

## Architectural Overlaps To Be Aware Of

### Two live runtimes

`main.py live` and `live_trader.py` both support ongoing trading sessions, but they differ in:

- strategy naming conventions
- broker/session orchestration
- runtime features and defaults
- intended operating mode

That means docs and operators need to be explicit about which live path they are using.

### Adaptive runtime is separate

`run_adaptive.py` remains its own product path. It does not share the same command surface as `main.py`, and deployment wrappers that use it should not assume `main.py` flags or behavior.

### Dashboard is monitoring-only

`web/app.py` should be treated as a read-focused monitoring/API layer. If trading control endpoints are desired later, they should be introduced explicitly rather than implied by the current dashboard.

## Recently Corrected Mismatches

- `start.py` no longer passes an unsupported `--force` flag to `run_adaptive.py`
- several `scripts/*.py` entrypoints now bootstrap the repo root correctly when executed as documented
- the stale compose PID-file healthcheck was removed from the paper-trading service
