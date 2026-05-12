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

Single canonical operator CLI:

```bash
python main.py {live,backtest,optimize} [flags...]
```

Supported examples:

```bash
python main.py live --strategy MomentumStrategy --force
python main.py live --strategy adaptive                                 # auto-scan symbols
python main.py live --strategy adaptive --scan-only                     # preview scan
python main.py live --strategy adaptive --regime-only                   # inspect regime
python main.py live --strategy momentum --risk-profile balanced --symbols AAPL,MSFT
python main.py backtest --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2024-01-01 --end-date 2024-12-31
python main.py optimize --strategy MomentumStrategy --start-date 2024-01-01 --end-date 2024-06-30
```

The previous `live_trader.py` (risk-profile presets + audit log) and
`run_adaptive.py` (regime/scan inspection + auto-symbol scanner) entry points
were merged into `main.py` by Phase 2 of the form-cleanup refactor.

### `web/app.py`

Local dashboard and API:

```bash
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### `start.py`

Deployment wrapper that launches the dashboard and `main.py live --strategy adaptive` together:

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
