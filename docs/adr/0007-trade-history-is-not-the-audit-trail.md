---
status: accepted
---
# Trade history is one store, separate from the audit trail

Three modules model a trade record with three schemas; only the hash-chained audit log is written in the live path, so the web dashboard and the CLI dashboard read databases nothing writes to. We decided trade history is one module with one writer (the recorder in order submission) and a SQLite adapter plus an in-memory adapter for tests; the audit trail stays a tamper-evidence event chain and is never queried for performance. `utils/database/` and `utils/performance_tracker.py` are deleted and the dashboards read trade history.
