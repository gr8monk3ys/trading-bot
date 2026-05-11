# Quick Start

This guide is the stable, repo-level quick start. It replaces the older runtime snapshot that previously lived in this file.

The canonical interface is `main.py`. The older `live_trader.py` and `run_adaptive.py` entry points were consolidated into `main.py` subcommands by Phase 2 of the form-cleanup refactor.

## 1. Install Prerequisites

- Python 3.10+
- `uv`
- TA-Lib system library
- Alpaca paper-trading account

macOS:

```bash
brew install ta-lib
```

Ubuntu or Debian:

```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

## 2. Sync the Environment

From the repo root:

```bash
uv sync --group dev --group test
cp .env.example .env
```

Set the following in `.env`:

```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
PAPER=True
```

## 3. Verify Connectivity

```bash
uv run python tests/test_connection.py
```

Optional interactive helper:

```bash
uv run python scripts/quickstart.py
```

## 4. Backtest First

Use a backtest before starting the live paper runtime:

```bash
uv run python main.py backtest \
  --strategy MomentumStrategy \
  --symbols AAPL,MSFT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

Optional validated report workflow:

```bash
uv run python scripts/validated_backtest_report.py \
  --strategy MomentumStrategy \
  --symbols AAPL,MSFT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output results/validated_backtest_report.md \
  --json results/validated_backtest_report.json
```

## 5. Start Paper Trading

```bash
uv run python main.py live \
  --strategy MomentumStrategy \
  --symbols AAPL,MSFT \
  --force
```

Notes:

- `--force` skips the market-open check for testing and supervised dry runs.
- Omit `--real` unless you are intentionally starting the real-money path.
- `main.py live --real` is subject to the governance gate by default.

## 6. Start Monitoring

FastAPI dashboard:

```bash
uv run python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Terminal dashboard:

```bash
uv run python scripts/dashboard.py
```

Emergency kill switch:

```bash
uv run python scripts/kill_switch.py --confirm "HALT TRADING" --cancel-orders --liquidate
```

## 7. Where to Go Next

- `README.md`: top-level overview and command map
- `docs/SETUP.md`: detailed environment setup
- `DOCKER.md`: container usage and deployment notes

## Current Notes

- `main.py` is the single canonical CLI; previous `live_trader.py` and `run_adaptive.py` runners were merged in.
- Use `python main.py live --risk-profile balanced` for the old `live_trader.py` presets.
- Use `python main.py live --strategy adaptive --scan-only` or `--regime-only` for the old `run_adaptive.py` inspection modes.
