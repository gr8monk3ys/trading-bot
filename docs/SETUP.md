# Setup and Run Modes

This document describes the current, supported setup path and the operator surfaces that exist in the repo today.

## Environment Setup

Preferred:

```bash
uv sync --group dev --group test
```

Fallback:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
```

Copy the environment template:

```bash
cp .env.example .env
```

Required values for paper trading:

```bash
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
PAPER=True
```

## Primary Entry Points

### `main.py`

Canonical operator CLI:

```bash
python main.py {live,backtest,optimize,replay,research} [flags...]
```

Supported examples:

```bash
python main.py live --strategy MomentumStrategy --force
python main.py backtest --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2024-01-01 --end-date 2024-12-31
python main.py backtest --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2024-01-01 --end-date 2024-12-31 --validated
python main.py replay --run-id <run_id> --limit 50
python main.py research --research-action check --experiment-id <experiment_id>
```

### `live_trader.py`

Alternate paper-only launcher for direct single-strategy runs with explicit runtime risk settings:

```bash
python live_trader.py --strategy momentum --risk-profile balanced --symbols AAPL MSFT
```

Use this when you specifically want the narrower `live_trader.py` flags for paper trading. Prefer `main.py` for general operations, especially any real-money path.

### `web/app.py`

Local dashboard and API:

```bash
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### `start.py`

Deployment wrapper that launches the dashboard and `run_adaptive.py` together:

```bash
python start.py
```

This is primarily for platform-style deployments rather than local development.

## Safe Workflow

1. Validate credentials and imports.
2. Run a backtest.
3. Run a validated backtest.
4. Start paper trading.
5. Use the dashboard and replay artifacts to inspect behavior.
6. Do not use `--real` until governance artifacts are in place.

## Artifacts and State

- `results/runs/<run_id>/`: runtime artifacts for replay and ops reporting
- `results/validation/`: validation outputs and gate evidence
- `audit_logs/`: trade-lifecycle audit records
- `data/`: runtime database and fetcher state
- `logs/`: application logs
