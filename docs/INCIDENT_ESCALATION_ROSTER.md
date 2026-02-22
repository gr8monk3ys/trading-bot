# Incident Escalation Roster

## Purpose
This roster defines who gets paged and in what order during trading incidents.

## Escalation Links
- Pager policy URL: `https://app.pagerduty.com/service-directory/PTRADINGBOT`
- Chat war-room URL: `https://slack.com/app_redirect?channel=trading-incidents`
- Ticket queue URL: `https://linear.app/trading-bot/team/INC`

## Escalation Chain
| Priority | Role | Contact | Escalate After |
| --- | --- | --- | --- |
| 1 | Primary on-call | `PagerDuty: Trading Bot Primary On-Call` | Immediate |
| 2 | Secondary on-call | `PagerDuty: Trading Bot Secondary On-Call` | 10 minutes |
| 3 | Trading infra owner | `Slack: @trading-infra-owner` | 20 minutes |
| 4 | Risk owner | `Slack: @risk-oncall` | 30 minutes |

## External Provider Escalation
| Provider | Link |
| --- | --- |
| Alpaca status | `https://status.alpaca.markets/` |
| Alpaca support | `https://alpaca.markets/support` |

## Notes
- Keep this file current with on-call rotations.
- Review quarterly and after any incident postmortem.
