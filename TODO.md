# TODO

## Purpose
This file tracks remaining production-hardening and quality tasks referenced by docs.

## Safety Hardening
- [ ] Remove all remaining direct broker order submissions from strategy implementations.
- [ ] Add gateway-enforcement tests for every live strategy class.
- [ ] Add regression test for circuit-breaker emergency liquidation with gateway enforcement enabled.

## Operations
- [ ] Validate incident ticket webhook integration in a non-test environment.
- [ ] Add runbook ownership and escalation roster links.

## Testing & Coverage
- [x] Raise measured engine coverage to `>=98%`.
- [x] Add branch-focused tests for `engine/validated_backtest.py` edge paths.
- [x] Add branch-focused tests for `engine/performance_metrics.py` significance/correction paths.
