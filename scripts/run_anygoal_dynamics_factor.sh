#!/usr/bin/env bash
# Dynamics factorial for the corrected Extract-Peg bank.
# Uses two free GPUs. Does not overwrite results/anygoal_contact_audit_v1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/anygoal_contact_env.sh"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
PYTHON="${ANYGOAL_ROOT}/.venv/bin/python"
OUTPUT="${ROOT}/results/anygoal_dynamics_factor_v1"
DOCS="${ROOT}/docs/results/anygoal_dynamics_factor_v1"
CHECKPOINT="${ANYGOAL_CHECKPOINT_ROOT}/sensorimotor_insertpeg.ckpt"
mkdir -p "${OUTPUT}" "${DOCS}"

if [[ ! -f "${OUTPUT}/bank.npz" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    "${PYTHON}" "${ROOT}/scripts/anygoal_dynamics_factor.py" prepare \
    --output "${OUTPUT}" --data-root "${ANYGOAL_DATA_ROOT}" --source extract
fi

CONDITIONS=(evaluator_10 noweld_25 weld_10 weld_25 extract_noweld_10 extract_noweld_25)
GPU_CYCLE=(1 4 1 4 1 4)
for index in "${!CONDITIONS[@]}"; do
  name="${CONDITIONS[$index]}"
  if [[ -f "${OUTPUT}/bank/${name}.npz" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU_CYCLE[$index]}" \
    "${PYTHON}" "${ROOT}/scripts/anygoal_dynamics_factor.py" rollout \
    --output "${OUTPUT}" --condition "${name}" \
    > "${OUTPUT}/${name}.log" 2>&1 &
done
wait

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
  "${PYTHON}" "${ROOT}/scripts/anygoal_dynamics_factor.py" score \
  --output "${OUTPUT}" --checkpoint "${CHECKPOINT}" --anygoal-root "${ANYGOAL_ROOT}" \
  --condition evaluator_10 --condition noweld_25 --condition weld_10 --condition weld_25 \
  --condition extract_noweld_10 --condition extract_noweld_25

"${PYTHON}" "${ROOT}/scripts/anygoal_dynamics_factor.py" summarize \
  --output "${OUTPUT}" --docs "${DOCS}"
