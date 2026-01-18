# Project Rules

Project-specific rules for the trading-bot repository.

## Code Style

```yaml
python:
  version: "3.10"
  style: pep8
  typing: preferred
  docstrings: google_style
  max_line_length: 100
```

## Architecture Rules

```yaml
patterns:
  async_await: required_for_broker_operations
  strategy_inheritance: BaseStrategy
  error_handling: fail_safe_for_trading
  imports: avoid_circular_imports
```

## Testing

```yaml
testing:
  framework: pytest
  async_support: pytest-asyncio
  coverage_target: 80%
  fixtures: use_conftest_for_shared
  mocking: AsyncMock_for_broker
```

Test file naming: `test_*.py` or `*_test.py`

## Trading Safety

```yaml
safety:
  default_mode: paper_trading
  circuit_breaker: required
  position_limits: enforce_in_risk_manager
  correlation_checks: before_new_positions
```

## Git

```yaml
commits:
  style: conventional_commits
  format: "type: description"
  types: [feat, fix, refactor, test, docs, chore]
```

## Rule Priority

1. Trading safety (highest)
2. Correctness
3. Performance
4. Style (lowest)
