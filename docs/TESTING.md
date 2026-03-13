# Testing Guide

## Environment

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

If broker-backed tests are needed, configure `.env` with Alpaca paper credentials first.

## Default Test Command

```bash
uv run pytest
```

This uses `pytest.ini`, including coverage reporting and warning gates.

## Focused Commands

Run a small security-focused subset:

```bash
uv run pytest tests/unit/test_web_app_security.py tests/unit/test_secrets_audit.py tests/unit/test_html_sanitization.py --no-cov
```

Run import and setup checks:

```bash
uv run pytest tests/test_imports.py tests/test_connection.py --no-cov
```

Run only unit tests:

```bash
uv run pytest tests/unit/
```

## Other Validation

Lint:

```bash
uv run ruff check .
```

Type check:

```bash
uv run mypy strategies/ brokers/ engine/ utils/
```

`ruff` is a dependable gate. `mypy` is not yet clean across the full repo, so treat it as diagnostic rather than release-blocking unless you are specifically working on typing debt.

## Smoke Checks

```bash
python examples/smoke_test.py
python tests/test_connection.py
.venv/bin/python main.py --help
.venv/bin/python live_trader.py --help
```

## Test Categories

- `tests/unit/`: primary regression coverage
- `tests/integration/`: slower flows and runtime integration
- `tests/test_connection.py`: broker connectivity, usually skipped without credentials
- `examples/smoke_test.py`: import and object-construction sanity check

## Notes

- Some tests are intentionally skipped when credentials or external services are unavailable.
- Generated coverage output lands in `coverage.xml` and `htmlcov/`.
- Runtime artifacts created by scripts should stay in `results/` and remain uncommitted.
