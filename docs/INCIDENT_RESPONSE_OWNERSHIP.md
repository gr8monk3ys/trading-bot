# Incident Response Ownership

## Purpose
This document defines ownership for trading-operations incidents and who is accountable for response quality.

## Service Ownership
- Service: `trading-bot`
- Service owner (role/team): `Trading Systems Team`
- Technical owner (role/team): `Execution Platform Engineering`
- Operations owner (role/team): `Site Reliability Engineering`
- Last reviewed: `2026-02-21`

## On-Call Accountability
- Primary on-call role: `Trading Bot Primary On-Call`
- Secondary on-call role: `Trading Bot Secondary On-Call`
- Incident commander fallback role: `SRE Incident Commander`

## Linked Procedures
- Primary operations runbook: [OPERATIONS_RUNBOOK.md](./OPERATIONS_RUNBOOK.md)
- Escalation roster: [INCIDENT_ESCALATION_ROSTER.md](./INCIDENT_ESCALATION_ROSTER.md)
- Kill switch procedure: [scripts/kill_switch.py](../scripts/kill_switch.py)

## Ownership SLO
- Ack critical incidents in `<= 15 minutes`.
- Assign an incident commander in `<= 10 minutes` for unresolved critical events.
- Post incident summary with corrective actions within `1 business day`.
