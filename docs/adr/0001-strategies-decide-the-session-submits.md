---
status: accepted
---
# Strategies decide; the session submits

Today each strategy orchestrates sizing, risk, breaker, order building and submission itself, across 16 hops and 9 files, and every order-path bug since July (#74, #81, #84, #89) landed on that chain. We decided a strategy is a decider: it receives a bar, its bar history and the portfolio, and returns order intents. The session owns submission. Strategies therefore hold no broker reference, which makes the gateway-compliance lint trivially true instead of a glob that missed the two strategies that trade.

## Consequences

- `MomentumStrategy` and `MeanReversionStrategy` lose `_execute_*` methods (~400 duplicated lines) and their `broker` constructor argument.
- The AdaptiveStrategy arm-subscription bug class (#89) becomes unrepresentable: arms never subscribe to anything.
- Tests assert on returned intents, not on mocked broker calls.
