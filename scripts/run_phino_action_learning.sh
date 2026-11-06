#!/usr/bin/env bash
# Bounded contrastive fine-tuning sweep for the action-conditioned PHINO pilot.
# One GPU; every run excludes its test geometry (leave-one-geometry-out folds).
# Usage (inside a Slurm GPU allocation, from the SIMPACT root):
#   CUDA_VISIBLE_DEVICES=<one gpu> bash scripts/run_phino_action_learning_v1.sh
set -euo pipefail
ROOT=results/phino_action_learning_v2
PY=.venv/bin/python
GEOMS="white_coconut_milk_carton blue_milk_carton mesh_cow mesh_cube mesh_nut"
mkdir -p "$ROOT/runs" "$ROOT/logs" "$ROOT/features"

run() {  # mode seed geometry
  local name="$1_s$2_$3" dir="$ROOT/runs/$1_s$2_$3"
  if [[ ! -f "$dir/done.json" ]]; then
    timeout 45m $PY scripts/phino_action_contrastive_train.py --mode "$1" --seed "$2" \
      --test-geometry "$3" --output-dir "$dir" > "$ROOT/logs/train_$name.log" 2>&1
  fi
  if [[ ! -f "$ROOT/features/ft_$name.npz" ]]; then
    timeout 20m $PY scripts/phino_action_probe.py extract --name "ft_$name" \
      --checkpoint "$dir/checkpoint.pt" > "$ROOT/logs/extract_$name.log" 2>&1
  fi
  echo "$(date -Is) finished $name"
}

for seed in 0 1; do
  for g in $GEOMS; do run clip "$seed" "$g"; done
done
for g in $GEOMS; do run static 0 "$g"; done
echo "$(date -Is) ALL_DONE"
