# TODO

## Purpose
This file tracks remaining production-hardening and quality tasks referenced by docs.

## Safety Hardening
- [x] Remove all remaining direct broker order submissions from strategy implementations.
- [x] Add gateway-enforcement tests for every live strategy class.
- [x] Add regression test for circuit-breaker emergency liquidation with gateway enforcement enabled.

## Operations
- [x] Validate incident ticket webhook integration in a non-test environment.
- [x] Add runbook ownership and escalation roster links.
- [x] Add automated validation that incident ownership/escalation docs are fully populated (no placeholders).
- [x] Add deterministic multi-broker failover/failback chaos drill coverage.
- [x] Add unified runtime industrial readiness gate for incident docs, chaos drills, ticket-drill evidence, and live failover probing.

## Testing & Coverage
- [x] Raise measured engine coverage to `>=98%`.
- [x] Add branch-focused tests for `engine/validated_backtest.py` edge paths.
- [x] Add branch-focused tests for `engine/performance_metrics.py` significance/correction paths.
