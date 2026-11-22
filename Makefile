.PHONY: check

# Same gates CI runs: lint, format, strict-warning unit tests.
check:
	uv run ruff check .
	uv run black --check .
	uv run pytest tests/unit/ --no-cov -W error -W ignore::DeprecationWarning -W ignore::PendingDeprecationWarning -W ignore::pytest.PytestUnraisableExceptionWarning
