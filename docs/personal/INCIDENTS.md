# Incident Log

> A dated log of every weird thing that happens with the bot. Crashes, unexpected trades, broker issues, configuration mistakes, security scares — all of it. Even the small stuff.
>
> Why log everything? Because patterns emerge over months. The fourth time you see the same kind of issue, you'll wish you'd written down the first three.

## How to Use This File

When something happens, add a new entry at the **top** of the incidents section below. Use the template format. Be specific. Future-you will thank present-you.

If the incident is severe (real money lost, security breach, regulatory issue), also:
1. Open a GitHub issue with label `incident`
2. Update `RUNBOOK.md` if the response procedure needs improvement
3. Add to `DECISIONS.md` if the incident drove a major architectural change

## Severity Levels

- **P0 (Critical):** Real money lost, account compromised, illegal activity, bot doing something dangerous. Stop everything. Investigate immediately.
- **P1 (High):** Bot crashed during market hours, missed expected trades, unauthorized access attempt detected. Fix same day.
- **P2 (Medium):** Strategy underperforming expectations, deployment glitches, non-critical errors. Fix this week.
- **P3 (Low):** Cosmetic issues, minor log noise, documentation gaps. Fix when convenient.

## Entry Template

YYYY-MM-DD — [Short title]
Severity: P0 / P1 / P2 / P3
What happened: Plain description of the symptom. What did you observe?
When: Date, time (with timezone), how long it lasted.
Impact: Money lost, trades missed, downtime, security exposure. Be specific. "Bot was down" is not impact — "bot was down for 2 hours during market open, missed 3 expected trades, no positions affected" is impact.
Root cause: Why did this happen? If unknown, say "unknown" — don't fabricate.
Resolution: What did you do to fix it? Be specific enough that you could repeat the fix.
Prevention: What change (code, runbook, monitoring, process) would prevent recurrence?
Follow-up actions: GitHub issues created, runbook updates, dependency upgrades, etc. Link them.
Status: Resolved / In progress / Won't fix / Recurring (link prior incidents)

---

## Incidents

<!-- Newest at top. Add new entries above the placeholder below. -->

### No incidents yet

Project is in Phase 0 (setup). First entries will likely come during Phase 1 paper trading.

Common categories to expect:
- Configuration mistakes (`.env` typos, wrong symbols, etc.)
- Broker API quirks (rate limits, weird error codes)
- Network or hosting issues (Railway hiccups, internet outages)
- Strategy bugs (unexpected signals, edge cases)
- Documentation drift (runbook said one thing, reality is another)

---

## Patterns and Trends

(This section will get filled in as incidents accumulate. Look for repeated root causes — they indicate systemic issues worth fixing properly.)

## Lessons Learned Index

(As incidents are resolved, distill key lessons here for quick reference. One line each, link to full incident.)