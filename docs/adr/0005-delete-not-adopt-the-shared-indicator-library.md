---
status: proposed
---
# Delete the shared indicator library rather than adopt it

`utils/indicators.py` and `utils/indicator_analysis.py` (871 lines, 1072 test lines) have no importer outside their own tests; each strategy calls TA-Lib directly. Adopting the library would route the baseline's signal computations through code that has never produced a cited number, risking silent verdict drift for no caller benefit. We decided to delete it. The same reasoning removes `BacktestEngine.run()` with its private metrics, the inert `persist_artifacts` parameter, the four top-level re-export facades, and walk-forward validation, which has no entrypoint and which the honest baseline explicitly does not use; the `pyproject.toml` description is corrected to match. `archive/research` keeps the history.
