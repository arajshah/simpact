#!/usr/bin/env bash
# Spatial-readout (4x4 pooled patch tokens) features for every fine-tuned checkpoint
# of the action-conditioned PHINO pilot. Run after scripts/run_phino_action_learning.sh.
#   CUDA_VISIBLE_DEVICES=<one gpu> bash scripts/run_phino_action_grid_features.sh
set -euo pipefail
ROOT=results/phino_action_learning_v2
PY=.venv/bin/python
for dir in "$ROOT"/runs/*/; do
  name=$(basename "$dir")
  out="$ROOT/features/grid_ft_$name.npz"
  [[ -f "$dir/done.json" && ! -f "$out" ]] || continue
  timeout 20m $PY scripts/phino_action_probe.py extract --grid --name "grid_ft_$name" \
    --checkpoint "$dir/checkpoint.pt" > "$ROOT/logs/extract_grid_$name.log" 2>&1
  echo "$(date -Is) grid features $name"
done
echo "$(date -Is) GRID_DONE"
