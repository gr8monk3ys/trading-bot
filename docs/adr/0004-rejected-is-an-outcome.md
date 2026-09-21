---
status: accepted
---
# Rejected and halted are order outcomes, not fills of zero

The backtest gateway treats any truthy broker reply as success, so a simulated rejection under the stressed profile, or a fractional exit truncated to zero by `int(quantity)`, returns `success=True` with nothing filled. We decided order submission returns one outcome type with an explicit status (filled, accepted, partially filled, rejected, halted) and a reason; there is no boolean success flag and quantities stay fractional. Entries, exits and end-of-backtest liquidation all go through the same submission so they share validation, protective-level registration and audit.
