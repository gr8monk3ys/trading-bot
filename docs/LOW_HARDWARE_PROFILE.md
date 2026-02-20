# Low-Hardware Paper-Trading Profile (Raspberry Pi Class)

This profile is for **paper trading only** on constrained hardware (for example, Raspberry Pi 4/5).

## Goals
- Keep CPU and RAM usage predictable.
- Reduce strategy churn and order frequency.
- Keep safety gates enabled.

## Recommended Usage

```bash
python scripts/run_low_resource_profile.py --dry-run
python scripts/run_low_resource_profile.py
```

Defaults are intentionally conservative:
- Strategy: `momentum`
- Symbols: `AAPL MSFT`
- Position size: `2%`
- Stop loss: `1%`
- Take profit: `2%`
- Paper mode forced on by default (`PAPER=true`)

## Practical Notes for Raspberry Pi
1. Disable optional ML and other heavyweight workflows on the same device.
2. Keep symbol count low (2-4 symbols).
3. Prefer ethernet over Wi-Fi for lower packet loss.
4. Use a process supervisor (systemd) for restart-on-failure.
5. Keep logs on local SSD if possible (microSD wear is real under heavy writes).

## What this profile does *not* do
- It does not change strategy internals.
- It does not guarantee profitability.
- It does not make live-capital deployment safe by itself.

Use this profile as a stable paper-trading baseline before scaling up.
