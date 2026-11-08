#!/usr/bin/env bash
# Why the Insert-Peg SMWM ranking fails. Reuses the anchor bank and checkpoint.
# Does not train a model and does not overwrite earlier AnyGoal result directories.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/anygoal_contact_env.sh"
export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYTHON="${ANYGOAL_ROOT}/.venv/bin/python"
OUTPUT="${ROOT}/results/anygoal_ranking_mechanism_v1"
DOCS="${ROOT}/docs/results/anygoal_ranking_mechanism_v1"
ANCHOR="${ROOT}/results/anygoal_insert_anchor_v1"
CHECKPOINT="${ANYGOAL_CHECKPOINT_ROOT}/sensorimotor_insertpeg.ckpt"
mkdir -p "${OUTPUT}" "${DOCS}"

if [[ ! -f "${OUTPUT}/support.json" ]]; then
  "${PYTHON}" "${ROOT}/scripts/anygoal_ranking_mechanism.py" support \
    --anchor "${ANCHOR}" \
    --data-root "${ANYGOAL_DATA_ROOT}" \
    --output "${OUTPUT}"
fi

if [[ ! -f "${OUTPUT}/horizon.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${PYTHON}" "${ROOT}/scripts/anygoal_ranking_mechanism.py" horizon \
    --anchor "${ANCHOR}" \
    --checkpoint "${CHECKPOINT}" \
    --anygoal-root "${ANYGOAL_ROOT}" \
    --output "${OUTPUT}"
fi

"${PYTHON}" "${ROOT}/scripts/anygoal_ranking_mechanism.py" report \
  --support "${OUTPUT}/support.json" \
  --horizon "${OUTPUT}/horizon.json" \
  --docs "${DOCS}"
cp "${OUTPUT}/support.json" "${OUTPUT}/horizon.json" "${DOCS}/"
