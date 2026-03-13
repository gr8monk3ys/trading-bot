# Docker Guide

This repository ships a multi-stage Docker image and a `docker compose` file with a few distinct runtime profiles.

## What the Image Does

- Builder stage installs TA-Lib and resolves Python dependencies with `uv`
- Runtime stage copies only the built virtualenv and application code
- The shipped runtime image runs as non-root
- The runtime image does not include `uv`; it is builder-only
- The image default command is `python start.py`

`start.py` is the deployment wrapper that launches:

- `web.app` via `uvicorn`
- `run_adaptive.py`

## Compose Services

`docker-compose.yml` currently defines these primary services:

- `trading-bot-paper`: paper trading via `python main.py live --strategy MomentumStrategy --force`
- `trading-bot-crypto`: low-resource crypto profile via `scripts/run_low_resource_profile.py`
- `trading-bot-stock`: low-resource stock profile via `scripts/run_low_resource_profile.py`
- `backtest`: on-demand backtest container

The low-resource services are the most current containerized runtime paths.

## Prerequisites

```bash
docker --version
docker compose version
cp .env.example .env
```

Set at least:

```bash
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
PAPER=true
```

## Common Workflows

Build the paper-trading image:

```bash
docker compose build trading-bot-paper
```

Run the paper-trading service:

```bash
docker compose up -d trading-bot-paper
docker compose logs -f trading-bot-paper
```

Run the crypto low-resource profile:

```bash
docker compose --profile crypto up -d trading-bot-crypto
docker compose logs -f trading-bot-crypto
```

Run the stock low-resource profile:

```bash
docker compose --profile stock up -d trading-bot-stock
docker compose logs -f trading-bot-stock
```

Run a backtest container:

```bash
docker compose --profile tools run --rm backtest
```

Stop everything:

```bash
docker compose down
```

## Volumes

Compose mounts host paths into the container for persistent state:

- `./logs` or `./logs_*`
- `./data` or `./data_*`
- `./results` or `./results_*`
- `./audit_logs` or `./audit_logs_*`
- `./config.py:/app/config.py:ro`

## Health Checks

- The image-level Dockerfile healthcheck targets `http://localhost:8000/api/health` when `start.py` is used
- The compose live-trading services do not expose an HTTP endpoint, so their healthchecks are disabled
- Container restart policy is the primary process-liveness mechanism for those services

## Recommended Container Paths

Use one of these depending on intent:

- `trading-bot-paper` for the legacy `main.py live` paper path
- `trading-bot-crypto` for the current low-resource 24/7 crypto path
- `trading-bot-stock` for the current low-resource weekday stock path

If you want the dashboard and adaptive runner together behind the image default command, run the image directly without a compose command override.
