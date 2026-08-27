# Setup

## Prerequisites

- Python 3.10+ and [`uv`](https://docs.astral.sh/uv/)
- The TA-Lib C library
- An Alpaca paper-trading account (only needed for `main.py live` and the connection test)

TA-Lib on macOS:

```bash
brew install ta-lib
```

TA-Lib on Ubuntu/Debian (same recipe CI uses):

```bash
wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
./configure --prefix=/usr && make && sudo make install
```

## Install

```bash
uv sync --group dev --group test
cp .env.example .env
```

`.env` for paper trading:

```env
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
PAPER=True
DASHBOARD_TOKEN=          # only if you run scripts/dashboard.py; it fails closed without one
```

`PAPER` defaults to true. Live trading requires `--real` and is not recommended: no strategy here beats buy-and-hold (see the README).

## Run

```bash
uv run python tests/test_connection.py        # verify Alpaca credentials
uv run python main.py backtest --strategy MomentumStrategyBacktest \
    --symbols SPY,QQQ --start-date 2024-01-01 --end-date 2024-12-31
uv run python main.py live                    # paper trading, MomentumStrategy
uv run python main.py live --strategy adaptive --regime-only   # inspect regime, no orders
uv run python main.py optimize --strategy MomentumStrategy --start-date 2024-01-01 --end-date 2024-06-30
uv run python scripts/run_etf_baseline.py     # regenerate the canonical 2020-2024 sweep
```

`main.py --help` lists every flag. `scripts/paper_smoke_test.py` proves the live order path against the paper API (1-share unfillable limit, audit-log check, cancel).

## Test and lint

```bash
uv run pytest tests/unit/          # what CI runs (with -W error on top)
uv run pytest tests/               # also the slower integration tests
uv run ruff check . && uv run black --check .
uv run mypy strategies/ brokers/ engine/ utils/   # advisory in CI
```

`asyncio_mode = auto` is set in `pytest.ini`: do not add `@pytest.mark.asyncio`.

## CI

One workflow, `.github/workflows/ci.yml`: ruff, black, mypy (advisory), then `pytest tests/unit/` on Python 3.10 and 3.11 with a strict-warnings pass first. Coverage goes to Codecov. CodeQL runs through GitHub's default setup, not a workflow file. Tests that touch the broker use the `ALPACA_*_TEST` repository secrets.

## Docker

`Dockerfile` and `docker-compose.yml` build a paper-trading container; see [`DOCKER.md`](DOCKER.md). No image is published.
