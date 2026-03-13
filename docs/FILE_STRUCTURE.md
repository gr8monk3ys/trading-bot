# Repository File Structure

## Core Layout

```text
trading-bot/
├── brokers/          Broker integrations and order construction
├── data/             Local databases, fetchers, cached datasets, and runtime state
├── docs/             Canonical documentation and runbooks
├── engine/           Backtesting, performance, validation, and orchestration logic
├── examples/         Smoke tests and illustrative entry scripts
├── execution/        Execution-layer helpers
├── factors/          Factor and portfolio research modules
├── infra/            Deployment and infrastructure-related assets
├── ml/               Machine-learning components
├── research/         Research materials and analysis assets
├── results/          Generated validation, report, and run outputs
├── scripts/          Operational utilities, reports, and local helpers
├── strategies/       Trading strategies
├── tests/            Unit and integration tests
├── utils/            Shared runtime, risk, analytics, and ops utilities
├── web/              FastAPI dashboard, routes, and templates
├── main.py           Primary CLI
├── live_trader.py    Narrower alternate live launcher
├── run_adaptive.py   Adaptive-strategy launcher
├── start.py          Deployment wrapper for bot + dashboard
├── Dockerfile        Multi-stage container build
└── docker-compose.yml
```

## Top-Level Files Worth Knowing

- `README.md`: current project overview
- `QUICKSTART.md`: shortest path to a working setup
- `DOCKER.md`: container usage
- `CICD.md`: CI/CD guidance
- `SECURITY.md`: security notes
- `pyproject.toml`: canonical Python package metadata and dependencies
- `pytest.ini`: test and coverage configuration

## Working Conventions

- Prefer `main.py` for operator workflows.
- Treat `results/`, `audit_logs/`, `logs/`, coverage output, and local caches as generated artifacts.
- Prefer documentation under `docs/` for durable guidance; many root markdown files are historical reports or specialized guides.

## Organization Guidance

Low-risk cleanup that aligns with the current repo:

- Keep runtime outputs out of the repo root.
- Keep new durable guides under `docs/`.
- Treat point-in-time reports and research snapshots as archive material, not landing documentation.
