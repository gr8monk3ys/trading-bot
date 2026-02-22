# Compliance & Governance Controls

## Scope
This policy defines minimum governance controls required before moving from paper trading to live capital.

## Required Controls For Live Mode
- Dual approval required (`>= 2` unique approvers).
- Explicit capital limit (`max_notional_usd`).
- Strategy allowlist defined.
- KYC and AML attestations set to `true`.
- Best-execution policy acknowledgment set to `true`.
- Approval must include:
  - `approved_at` (ISO-8601)
  - `expires_at` (ISO-8601, future timestamp)

## Evidence Artifact
Create and maintain:
- `results/governance/live_approval.json`

Expected schema:
```json
{
  "approvers": ["ops_lead", "risk_lead"],
  "approved_at": "2026-02-21T12:00:00+00:00",
  "expires_at": "2026-05-22T12:00:00+00:00",
  "max_notional_usd": 100000,
  "kyc_attestation": true,
  "aml_attestation": true,
  "best_execution_policy_ack": true,
  "strategy_allowlist": ["momentum", "mean_reversion"]
}
```

## Automated Gate
Run:
```bash
python scripts/governance_gate.py --mode live --output results/validation/governance_gate.json
```

Gate behavior:
- `paper` mode: policy must exist; live approval artifact is optional.
- `live` mode: approval artifact is mandatory and must satisfy all controls.
