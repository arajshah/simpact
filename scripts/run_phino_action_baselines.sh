#!/usr/bin/env bash
# Matched baselines for the action-conditioned PHINO pilot.
# Does not retrain trajectory-similarity checkpoints and does not write into
# results/phino_action_learning_v2. Five geometries run at once, one per GPU.
# Usage (from the SIMPACT root, with at least five free GPUs):
#   bash scripts/run_phino_action_baselines.sh
set -euo pipefail
ROOT=results/phino_action_baselines_v1
PY=.venv/bin/python
GEOMS="white_coconut_milk_carton blue_milk_carton mesh_cow mesh_cube mesh_nut"
GPUS=(0 1 2 3 4)
GRID_SUBSETS=(
  static
  push_x_v2
  push_lo_1.6N
  push_hi_3.2N
  toss_4.0N
  push_lo_1.6N+push_hi_3.2N
  push_hi_3.2N+toss_4.0N
  push_lo_1.6N+push_hi_3.2N+toss_4.0N
  push_x_v2+push_lo_1.6N+push_hi_3.2N+toss_4.0N
)
mkdir -p "$ROOT/runs" "$ROOT/logs" "$ROOT/features"

run_job() {
  local obj="$1" seed="$2" geom="$3" gpu="$4"
  local name="${obj}_s${seed}_${geom}"
  local dir="$ROOT/runs/$name"
  export CUDA_VISIBLE_DEVICES="$gpu"
  if [[ ! -f "$dir/done.json" ]]; then
    if [[ -e "$dir" ]]; then
      echo "incomplete run directory: $dir" >&2
      return 1
    fi
    timeout 45m $PY scripts/phino_action_contrastive_train.py \
      --objective "$obj" --mode clip --seed "$seed" \
      --test-geometry "$geom" --output-dir "$dir" \
      > "$ROOT/logs/train_${name}.log" 2>&1
  fi
  if [[ ! -f "$ROOT/features/ft_${name}.npz" ]]; then
    timeout 30m $PY scripts/phino_action_probe.py extract \
      --name "ft_${name}" --features-dir "$ROOT/features" \
      --checkpoint "$dir/checkpoint.pt" \
      > "$ROOT/logs/extract_${name}.log" 2>&1
  fi
  if [[ ! -f "$ROOT/features/grid_ft_${name}.npz" ]]; then
    timeout 30m $PY scripts/phino_action_probe.py extract --grid \
      --name "grid_ft_${name}" --features-dir "$ROOT/features" \
      --checkpoint "$dir/checkpoint.pt" \
      > "$ROOT/logs/extract_grid_${name}.log" 2>&1
  fi
  echo "$(date -Is) finished $name"
}

wait_wave() {
  local fail=0
  for pid in "$@"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "a baseline job failed" >&2
    return 1
  fi
}

for obj in instance regression; do
  for seed in 0 1; do
    pids=()
    i=0
    for geom in $GEOMS; do
      run_job "$obj" "$seed" "$geom" "${GPUS[$i]}" &
      pids+=("$!")
      i=$((i + 1))
    done
    wait_wave "${pids[@]}"
    echo "$(date -Is) finished ${obj} seed ${seed}"
  done
done

if [[ ! -f "$ROOT/probes_pooled/results.json" ]]; then
  $PY scripts/phino_action_probe.py evaluate \
    --output-dir "$ROOT/probes_pooled" \
    --baseline image_position \
    --splits logo within \
    --encoder ft_instance_s0="$ROOT/features/ft_instance_s0_{fold}.npz" \
    --encoder ft_instance_s1="$ROOT/features/ft_instance_s1_{fold}.npz" \
    --encoder ft_regression_s0="$ROOT/features/ft_regression_s0_{fold}.npz" \
    --encoder ft_regression_s1="$ROOT/features/ft_regression_s1_{fold}.npz" \
    > "$ROOT/logs/probes_pooled.log" 2>&1
fi
if [[ ! -f "$ROOT/probes_grid/results.json" ]]; then
  $PY scripts/phino_action_probe.py evaluate \
    --output-dir "$ROOT/probes_grid" \
    --baseline image_position \
    --splits logo within \
    --subsets "${GRID_SUBSETS[@]}" \
    --encoder grid_ft_instance_s0="$ROOT/features/grid_ft_instance_s0_{fold}.npz" \
    --encoder grid_ft_instance_s1="$ROOT/features/grid_ft_instance_s1_{fold}.npz" \
    --encoder grid_ft_regression_s0="$ROOT/features/grid_ft_regression_s0_{fold}.npz" \
    --encoder grid_ft_regression_s1="$ROOT/features/grid_ft_regression_s1_{fold}.npz" \
    > "$ROOT/logs/probes_grid.log" 2>&1
fi
echo "$(date -Is) ALL_DONE"
