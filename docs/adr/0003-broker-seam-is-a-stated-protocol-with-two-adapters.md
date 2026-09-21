---
status: proposed
---
# The broker seam is a stated protocol with exactly two adapters

`brokers/broker_interface.py` (410 lines) has zero implementations, while the seam the code actually crosses is implicit and inconsistent: positions are async objects on Alpaca and sync dicts in the backtest, which is where #74's precedence bug lived. We decided to delete the abstract class and write down the narrow protocol both brokers already satisfy: async positions with one signed-quantity shape, account equity and cash, a price at a time, and submit. The backtest broker becomes async so no caller needs an `isawaitable` probe. Two adapters, Alpaca and backtest, justify the seam; a third would need a reason.
