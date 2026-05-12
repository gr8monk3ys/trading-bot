# Installation Guide

## Prerequisites

- Python `3.10`
- TA-Lib system library
- Alpaca paper-trading credentials if you want to exercise broker-backed flows

## Install TA-Lib

macOS:

```bash
brew install ta-lib
```

Ubuntu/Debian:

```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

Windows:

- Install a TA-Lib binary compatible with Python `3.10`, then install the Python package inside the project environment.

## Create the Python environment

Preferred with `uv`:

```bash
uv sync --group dev --group test
```

Fallback with `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"
```

## Configure credentials

```bash
cp .env.example .env
```

Populate:

```bash
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
PAPER=True
```

## Verify the installation

```bash
python tests/test_connection.py
python examples/smoke_test.py
uv run pytest tests/test_imports.py --no-cov
```

## First commands

Backtest:

```bash
python main.py backtest --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2024-01-01 --end-date 2024-12-31
```

Paper trading:

```bash
python main.py live --strategy MomentumStrategy --force
```

Dashboard:

```bash
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```
