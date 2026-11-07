#!/usr/bin/env bash
# Reproduce SMWM planning and run the fixed action-bank diagnostic.
# Large checkpoints, datasets, and raw episode arrays stay under ANYGOAL_HOME
# and results/anygoal_contact_audit_v1. They are not committed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/anygoal_contact_env.sh"

PYTHON="${ANYGOAL_ROOT}/.venv/bin/python"
RESULTS="${ANYGOAL_AUDIT_RESULTS}"
DOCS="${ROOT}/docs/results/anygoal_contact_audit_v1"
mkdir -p "${RESULTS}/logs" "${DOCS}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "AnyGoal environment is missing at ${PYTHON}" >&2
  exit 1
fi

cd "${ANYGOAL_ROOT}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

"${PYTHON}" "${ROOT}/scripts/anygoal_contact_diagnostic.py" record-env \
  --output "${RESULTS}/environment.json"

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  "${PYTHON}" "${ROOT}/scripts/anygoal_contact_download.py" --verify-existing \
    || "${PYTHON}" "${ROOT}/scripts/anygoal_contact_download.py"
fi

if ! anygoal models list | awk '$1=="sensorimotor" && $2=="ready" {found=1} END {exit !found}'; then
  anygoal models install sensorimotor
fi
anygoal models list | tee "${RESULTS}/logs/models.txt"

smoke_metrics="${RESULTS}/smoke/smwm/extractpeg/seed_0/metrics.json"
if [[ ! -f "${smoke_metrics}" ]]; then
  anygoal eval planning --model sensorimotor --dataset extractpeg --seed 0 -- \
    "output.directory=${RESULTS}/smoke/smwm/extractpeg" \
    eval.num_eval=2 \
    | tee "${RESULTS}/logs/smoke_extractpeg.txt"
fi

for dataset in insertpeg avoid_peg extractpeg; do
  metrics="${RESULTS}/reproduction/smwm/${dataset}/seed_0/metrics.json"
  if [[ -f "${metrics}" ]]; then
    echo "Reusing ${metrics}"
    continue
  fi
  anygoal eval planning --model sensorimotor --dataset "${dataset}" --seed 0 -- \
    "output.directory=${RESULTS}/reproduction/smwm/${dataset}" \
    | tee "${RESULTS}/logs/smwm_${dataset}.txt"
done

"${PYTHON}" "${ROOT}/scripts/anygoal_contact_diagnostic.py" stage-a-audit \
  --results "${RESULTS}" \
  --data-root "${ANYGOAL_DATA_ROOT}"

if [[ ! -f "${RESULTS}/stage_b/summary.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/anygoal_contact_diagnostic.py" stage-b \
    --results "${RESULTS}" \
    --data-root "${ANYGOAL_DATA_ROOT}" \
    --checkpoint "${ANYGOAL_CHECKPOINT_ROOT}/sensorimotor_insertpeg.ckpt" \
    --anygoal-root "${ANYGOAL_ROOT}" \
    | tee "${RESULTS}/logs/stage_b.txt"
fi

"${PYTHON}" "${ROOT}/scripts/anygoal_contact_diagnostic.py" report \
  --results "${RESULTS}" \
  --docs "${DOCS}"

echo "Wrote ${DOCS}/summary.md"
