# Trading Bot

Algorithmic trading repository for Alpaca with backtesting, paper/live execution paths, monitoring dashboards, and research/governance tooling.

This repository has multiple runtime entrypoints. The commands below reflect the current supported paths in the codebase, not older historical workflows.

## Supported Execution Paths

- `main.py`: primary multi-mode CLI for `live`, `backtest`, `optimize`, `replay`, and `research`
- `live_trader.py`: single-strategy paper-only runner with risk-profile controls; used by the low-resource launcher
- `run_adaptive.py`: standalone adaptive strategy runner and opportunity scanner
- `web/app.py`: FastAPI monitoring dashboard and JSON API
- `start.py`: deployment wrapper that launches `web.app` and `run_adaptive.py` together

Architecture notes live in `docs/RUNTIME_ARCHITECTURE.md`.

## Setup

```bash
uv sync --group dev --group test
source .venv/bin/activate
cp .env.example .env
```

Set Alpaca credentials in `.env` or your shell environment:

```bash
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
PAPER=true
```

## Common Commands

Run the full test suite:

```bash
uv run pytest
```

Run only unit tests without coverage:

```bash
uv run pytest tests/unit/ --no-cov
```

Run a backtest:

```bash
uv run python main.py backtest \
  --strategy MomentumStrategy \
  --symbols AAPL,MSFT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

Run a validated backtest report:

```bash
uv run python scripts/validated_backtest_report.py \
  --strategy MomentumStrategy \
  --symbols AAPL,MSFT \
  --start-date 2014-01-01 \
  --end-date 2024-12-31 \
  --output results/validated_backtest_report.md \
  --json results/validated_backtest_report.json
```

Start paper trading with the main runtime:

```bash
uv run python main.py live --strategy MomentumStrategy --force
```

`main.py live` now enforces current validation evidence by default. Generate a fresh artifact bundle with `scripts/generate_validation_artifacts.py` before startup, and for `--real` also ensure `results/validation/precheck/go_live_precheck_summary.json` is present and passing. Use `--skip-validation` only for controlled dry runs.

Start the single-strategy paper trader directly:

```bash
uv run python live_trader.py --strategy momentum --symbols AAPL MSFT
```

Start the low-resource profile:

```bash
uv run python scripts/run_low_resource_profile.py --dry-run
uv run python scripts/run_low_resource_profile.py
```

Launch the interactive quickstart:

```bash
uv run python scripts/quickstart.py
```

Run the FastAPI dashboard locally:

```bash
uv run python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Run the terminal dashboard:

```bash
uv run python scripts/dashboard.py
```

Emergency kill switch:

```bash
uv run python scripts/kill_switch.py --confirm "HALT TRADING" --cancel-orders --liquidate
```

## Documentation Map

- `QUICKSTART.md`: current local setup and first-run commands
- `DOCKER.md`: Docker and `docker compose` workflows
- `docs/RUNTIME_ARCHITECTURE.md`: runtime entrypoints and architectural overlaps
- `docs/LOW_HARDWARE_PROFILE.md`: Raspberry Pi / low-resource deployment guidance
- `docs/OPERATIONS_RUNBOOK.md`: operational procedures and incident handling
- `docs/README.md`: documentation index with authoritative vs historical guides

## Repository Layout

- `engine/`: backtesting, evaluation, validation, and replay logic
- `brokers/`: Alpaca/backtest/options broker adapters
- `strategies/`: strategy implementations and base classes
- `utils/`: risk, reconciliation, audit, runtime-state, and ops helpers
- `web/`: FastAPI dashboard and templates
- `scripts/`: runnable utilities and reporting scripts
- `tests/`: unit and higher-level tests
- `docs/`: active guides plus historical implementation notes
- `results/`, `audit_logs/`: generated outputs and runtime artifacts

## Current Reality

- `main.py live` is the primary live/paper operator path; `live_trader.py` is a separate paper-only single-strategy runtime.
- `run_adaptive.py` remains a separate execution path and is not integrated into `main.py`.
- `web/app.py` is a monitoring surface, not the trading control plane.
- `start.py` is intended for deployment environments, not as the primary local CLI.
