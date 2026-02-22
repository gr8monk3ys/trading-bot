# Infrastructure As Code Baseline

This directory stores deployment manifests that can be versioned, reviewed, and promoted.

## Included
- `systemd/trading-bot.service`
- `systemd/trading-bot-watchdog.service`
- `systemd/trading-bot-watchdog.timer`
- `systemd/trading-bot-runtime-gate.service`
- `systemd/trading-bot-runtime-gate.timer`
- `systemd/trading-bot-ops-metrics.service`
- `systemd/trading-bot-ops-metrics.timer`
- `systemd/trading-bot-incident-automation.service`
- `systemd/trading-bot-incident-automation.timer`

## Usage
Copy units to `/etc/systemd/system/` on the target host:

```bash
sudo cp infra/systemd/trading-bot*.service infra/systemd/trading-bot*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-bot.service
sudo systemctl enable --now trading-bot-watchdog.timer
sudo systemctl enable --now trading-bot-runtime-gate.timer
sudo systemctl enable --now trading-bot-ops-metrics.timer
sudo systemctl enable --now trading-bot-incident-automation.timer
```

Adjust paths/user values in unit files before enabling in production.
