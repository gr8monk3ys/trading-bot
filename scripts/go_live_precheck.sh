#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ROOT="${DEFAULT_REPO_ROOT}"
OUTPUT_DIR="results/validation/precheck"

LOCAL_ONLY=false
RUN_CHAOS=false
RUN_TICKET_DRILL=false
RUN_FAILOVER_PROBE=false
RUN_SECRETS_AUDIT=false
RUN_GOVERNANCE_GATE=false

SKIP_DEPLOYMENT_PREFLIGHT=false
SKIP_RUNTIME_WATCHDOG=false
SKIP_RUNTIME_GATE=false
ENFORCE_IB_API_GATE=true

usage() {
  cat <<'EOF'
Usage: scripts/go_live_precheck.sh [options]

Options:
  --repo-root PATH                  Repository root (default: script parent)
  --output-dir PATH                 Artifact output directory (default: results/validation/precheck)
  --local-only                      Skip external runtime probes (no network, no live failover)
  --run-chaos                       Run chaos drills in runtime industrial gate
  --run-ticket-drill                Run ticket drill in runtime industrial gate (requires INCIDENT_TICKETING_WEBHOOK_URL)
  --run-failover-probe              Run live multi-broker failover probe in runtime industrial gate
  --run-secrets-audit               Run secrets rotation/leak audit during deployment preflight
  --run-governance-gate             Run compliance/governance gate (use --governance-mode live for live capital)
  --governance-mode MODE            Governance mode: paper|live (default: paper)
  --skip-deployment-preflight       Skip deployment preflight stage
  --skip-runtime-watchdog           Skip runtime watchdog stage
  --skip-runtime-gate               Skip runtime industrial gate stage
  --no-enforce-ib-api-gate          Allow go-live precheck without IB API readiness gate
  -h, --help                        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --local-only)
      LOCAL_ONLY=true
      shift
      ;;
    --run-chaos)
      RUN_CHAOS=true
      shift
      ;;
    --run-ticket-drill)
      RUN_TICKET_DRILL=true
      shift
      ;;
    --run-failover-probe)
      RUN_FAILOVER_PROBE=true
      shift
      ;;
    --run-secrets-audit)
      RUN_SECRETS_AUDIT=true
      shift
      ;;
    --run-governance-gate)
      RUN_GOVERNANCE_GATE=true
      shift
      ;;
    --governance-mode)
      GOVERNANCE_MODE="$2"
      shift 2
      ;;
    --skip-deployment-preflight)
      SKIP_DEPLOYMENT_PREFLIGHT=true
      shift
      ;;
    --skip-runtime-watchdog)
      SKIP_RUNTIME_WATCHDOG=true
      shift
      ;;
    --skip-runtime-gate)
      SKIP_RUNTIME_GATE=true
      shift
      ;;
    --no-enforce-ib-api-gate)
      ENFORCE_IB_API_GATE=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${LOCAL_ONLY}" == "true" ]]; then
  RUN_CHAOS=false
  RUN_TICKET_DRILL=false
  RUN_FAILOVER_PROBE=false
  RUN_SECRETS_AUDIT=false
  RUN_GOVERNANCE_GATE=false
  ENFORCE_IB_API_GATE=false
fi

if [[ "${ENFORCE_IB_API_GATE}" == "true" && "${SKIP_RUNTIME_WATCHDOG}" == "true" ]]; then
  echo "runtime_watchdog cannot be skipped while IB API gate enforcement is enabled." >&2
  echo "Use --no-enforce-ib-api-gate only for non-go-live dry runs." >&2
  exit 2
fi

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
GOVERNANCE_MODE="${GOVERNANCE_MODE:-paper}"
if [[ "${GOVERNANCE_MODE}" != "paper" && "${GOVERNANCE_MODE}" != "live" ]]; then
  echo "Invalid --governance-mode '${GOVERNANCE_MODE}'. Expected paper|live." >&2
  exit 2
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No python interpreter found." >&2
    exit 127
  fi
fi

run_step() {
  local label="$1"
  shift
  echo "========================================================================"
  echo "RUNNING: ${label}"
  echo "========================================================================"
  "$@"
  return $?
}

INCIDENT_JSON="${OUTPUT_DIR}/incident_contacts.json"
PREFLIGHT_JSON="${OUTPUT_DIR}/deployment_preflight.json"
WATCHDOG_JSON="${OUTPUT_DIR}/runtime_watchdog.json"
GATE_JSON="${OUTPUT_DIR}/runtime_industrial_gate.json"
GOVERNANCE_JSON="${OUTPUT_DIR}/governance_gate.json"
SUMMARY_JSON="${OUTPUT_DIR}/go_live_precheck_summary.json"

INCIDENT_RC=0
PREFLIGHT_RC=0
WATCHDOG_RC=0
GATE_RC=0
GOVERNANCE_RC=0

if run_step \
  "incident_contacts" \
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/validate_incident_contacts.py" \
    --ownership-doc "${REPO_ROOT}/docs/INCIDENT_RESPONSE_OWNERSHIP.md" \
    --escalation-doc "${REPO_ROOT}/docs/INCIDENT_ESCALATION_ROSTER.md" \
    --json-output "${INCIDENT_JSON}"; then
  INCIDENT_RC=0
else
  INCIDENT_RC=$?
fi

if [[ "${SKIP_DEPLOYMENT_PREFLIGHT}" == "false" ]]; then
  REQUIRED_ENV="ALPACA_API_KEY,ALPACA_SECRET_KEY"
  if [[ "${LOCAL_ONLY}" == "true" ]]; then
    REQUIRED_ENV=""
  fi
  PREFLIGHT_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/deployment_preflight.py"
    --repo-root "${REPO_ROOT}"
    --required-env "${REQUIRED_ENV}"
    --output "${PREFLIGHT_JSON}"
  )
  if [[ "${RUN_SECRETS_AUDIT}" == "true" ]]; then
    PREFLIGHT_CMD+=(--run-secrets-audit)
  fi
  if run_step "deployment_preflight" "${PREFLIGHT_CMD[@]}"; then
    PREFLIGHT_RC=0
  else
    PREFLIGHT_RC=$?
  fi
fi

if [[ "${SKIP_RUNTIME_WATCHDOG}" == "false" ]]; then
  WATCHDOG_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/runtime_watchdog.py"
    --output "${WATCHDOG_JSON}"
  )
  if [[ "${LOCAL_ONLY}" == "true" ]]; then
    WATCHDOG_CMD+=(--no-check-alpaca --no-check-ticket-webhook --no-check-ib-port --no-check-ib-api)
  fi
  if [[ "${ENFORCE_IB_API_GATE}" == "false" ]]; then
    WATCHDOG_CMD+=(--no-check-ib-api)
  fi
  if run_step "runtime_watchdog" "${WATCHDOG_CMD[@]}"; then
    WATCHDOG_RC=0
  else
    WATCHDOG_RC=$?
  fi
fi

if [[ "${SKIP_RUNTIME_GATE}" == "false" ]]; then
  GATE_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/runtime_industrial_gate.py"
    --repo-root "${REPO_ROOT}"
    --output "${GATE_JSON}"
  )

  if [[ "${RUN_CHAOS}" == "false" ]]; then
    GATE_CMD+=(--no-run-chaos-drill)
  fi

  if [[ "${RUN_TICKET_DRILL}" == "true" ]]; then
    GATE_CMD+=(--run-ticket-drill)
    if [[ -n "${INCIDENT_TICKETING_WEBHOOK_URL:-}" ]]; then
      GATE_CMD+=(--ticket-webhook-url "${INCIDENT_TICKETING_WEBHOOK_URL}")
    fi
    if [[ -n "${INCIDENT_RESPONSE_RUNBOOK_URL:-}" ]]; then
      GATE_CMD+=(--ticket-runbook-url "${INCIDENT_RESPONSE_RUNBOOK_URL}")
    fi
    if [[ -n "${INCIDENT_ESCALATION_ROSTER_URL:-}" ]]; then
      GATE_CMD+=(--ticket-escalation-roster-url "${INCIDENT_ESCALATION_ROSTER_URL}")
    fi
  fi

  if [[ "${RUN_FAILOVER_PROBE}" == "true" ]]; then
    GATE_CMD+=(--run-failover-probe)
  fi

  if run_step "runtime_industrial_gate" "${GATE_CMD[@]}"; then
    GATE_RC=0
  else
    GATE_RC=$?
  fi
fi

if [[ "${RUN_GOVERNANCE_GATE}" == "true" ]]; then
  if run_step \
    "governance_gate" \
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/governance_gate.py" \
      --repo-root "${REPO_ROOT}" \
      --mode "${GOVERNANCE_MODE}" \
      --output "${GOVERNANCE_JSON}"; then
    GOVERNANCE_RC=0
  else
    GOVERNANCE_RC=$?
  fi
fi

export INCIDENT_RC PREFLIGHT_RC WATCHDOG_RC GATE_RC
export GOVERNANCE_RC
export INCIDENT_JSON PREFLIGHT_JSON WATCHDOG_JSON GATE_JSON GOVERNANCE_JSON SUMMARY_JSON
export SKIP_DEPLOYMENT_PREFLIGHT SKIP_RUNTIME_WATCHDOG SKIP_RUNTIME_GATE
export RUN_GOVERNANCE_GATE

"${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone


def step(name: str, rc_key: str, path_key: str, skipped: bool) -> dict:
    if skipped:
        return {
            "name": name,
            "passed": True,
            "skipped": True,
            "status_code": 0,
            "artifact": os.environ.get(path_key, ""),
        }
    status = int(os.environ.get(rc_key, "1") or 1)
    return {
        "name": name,
        "passed": status == 0,
        "skipped": False,
        "status_code": status,
        "artifact": os.environ.get(path_key, ""),
    }


steps = [
    step("incident_contacts", "INCIDENT_RC", "INCIDENT_JSON", skipped=False),
    step(
        "deployment_preflight",
        "PREFLIGHT_RC",
        "PREFLIGHT_JSON",
        skipped=os.environ.get("SKIP_DEPLOYMENT_PREFLIGHT", "false").lower() == "true",
    ),
    step(
        "runtime_watchdog",
        "WATCHDOG_RC",
        "WATCHDOG_JSON",
        skipped=os.environ.get("SKIP_RUNTIME_WATCHDOG", "false").lower() == "true",
    ),
    step(
        "runtime_industrial_gate",
        "GATE_RC",
        "GATE_JSON",
        skipped=os.environ.get("SKIP_RUNTIME_GATE", "false").lower() == "true",
    ),
    step(
        "governance_gate",
        "GOVERNANCE_RC",
        "GOVERNANCE_JSON",
        skipped=os.environ.get("RUN_GOVERNANCE_GATE", "false").lower() != "true",
    ),
]

ready = all(step_data["passed"] for step_data in steps)
summary = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "ready": ready,
    "steps": steps,
}

summary_path = os.environ["SUMMARY_JSON"]
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("========================================================================")
print("GO-LIVE PRECHECK SUMMARY")
print("========================================================================")
print(f"Ready: {'YES' if ready else 'NO'}")
print(f"Summary artifact: {summary_path}")
for step_data in steps:
    status = "PASS" if step_data["passed"] else "FAIL"
    skipped = " (skipped)" if step_data["skipped"] else ""
    print(
        f"[{status}] {step_data['name']}{skipped}: "
        f"code={step_data['status_code']} artifact={step_data['artifact']}"
    )
PY

OVERALL_RC=0
if [[ "${INCIDENT_RC}" -ne 0 ]]; then OVERALL_RC=1; fi
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then OVERALL_RC=1; fi
if [[ "${WATCHDOG_RC}" -ne 0 ]]; then OVERALL_RC=1; fi
if [[ "${GATE_RC}" -ne 0 ]]; then OVERALL_RC=1; fi
if [[ "${GOVERNANCE_RC}" -ne 0 ]]; then OVERALL_RC=1; fi

exit "${OVERALL_RC}"
