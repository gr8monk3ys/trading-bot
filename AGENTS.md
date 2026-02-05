# Repository Guidelines

## Project Structure & Module Organization
- `engine/`: Core backtesting, evaluation, and validation logic (e.g., `backtest_engine.py`, `validated_backtest.py`).
- `brokers/`: Broker integrations and order building (Alpaca, backtest broker, options).
- `strategies/`: Trading strategies and shared base classes.
- `utils/`: Risk, execution, audit logging, and analytics utilities.
- `tests/`: Pytest-based unit tests (`tests/unit/`).
- `scripts/`: CLI utilities and reports (e.g., validated backtest report).
- `docs/`: Additional documentation and setup guides.
- `results/` and `audit_logs/`: Generated outputs and audit trail artifacts.

## Build, Test, and Development Commands
- `uv run pytest` — Run the test suite with coverage gates (preferred).
- `pytest` — Works if pytest is installed in the active environment.
- `python main.py backtest --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2024-01-01 --end-date 2024-12-31` — Run a backtest.
- `python main.py live --strategy MomentumStrategy --force` — Start live/paper trading (websocket auto-starts for audit logging).
- `python scripts/validated_backtest_report.py --strategy MomentumStrategy --symbols AAPL,MSFT --start-date 2014-01-01 --end-date 2024-12-31 --output results/validated_backtest_report.md --json results/validated_backtest_report.json` — Generate validated backtest report.

## Coding Style & Naming Conventions
- Python 3.10+.
- Indentation: 4 spaces; keep functions small and focused.
- Prefer explicit, readable names (e.g., `max_portfolio_risk`).
- Use `snake_case` for functions/variables and `PascalCase` for classes.
- Logging uses `logging` module; avoid printing in core modules.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio`.
- Coverage: enforced via `pytest.ini` (fail-under currently 30%).
- Test naming: `tests/unit/test_*.py`, functions `test_*`.

## Commit & Pull Request Guidelines
- Commit messages follow a Conventional Commits style (`feat:`, `fix:`, `docs:`).
- PRs should include a short summary, test results, and any relevant artifacts (e.g., report outputs in `results/`).

## Security & Configuration Tips
- Configure Alpaca credentials in `.env` or environment variables.
- Do not commit API keys or generated logs with sensitive data.
- Audit logs are written to `audit_logs/` for trade lifecycle traceability.
