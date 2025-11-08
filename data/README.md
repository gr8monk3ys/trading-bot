# Data Directory

This directory is used for storing data files used by the trading bot:

## Structure

- `/historical/`: Historical market data for backtesting
- `/results/`: Backtest and analysis results
- `/logs/`: Trading bot log files
- `/models/`: Trained model files for ML-based strategies

## Data Management

By default, this directory is excluded from git tracking (via .gitignore) to avoid committing large data files or sensitive information.

Use the utilities in the `utils` package to download and manage market data.
