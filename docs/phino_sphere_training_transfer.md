# PHINO: matched sphere-in-training object transfer experiment

This is an **analysis-only** comparison on the existing `results/phino_mesh_grid_cutoff_1m_v1` simulated seven-geometry dataset. It does not regenerate trajectories, alter accepted/rejected flags, change interventions, or train PHINO/DINOv2.

## Question

Does including a sphere in regression training help or hurt prediction of synthetic mass/friction on **the same six held-out non-sphere geometries**?

For each one of the six non-sphere geometries:

- Test on the identical valid mass-friction configurations of the held-out geometry in both conditions.
- **Without sphere:** Train ridge on the remaining five non-sphere geometries.
- **With sphere:** Train the same ridge on those five plus any valid sphere trajectories.
- Keep the three predeclared feature sets, five action sets, regularization `alpha=10`, and training-only feature scaling fixed. Report mass and friction mean absolute error (MAE) and the training-mean baseline.
- Do not impute rejected actions. Each action set has its own available-row population; per-object coverage is saved.
- Compare MAEs on the identical test cases. `delta_mae_with_minus_without > 0` means that adding sphere training **hurt** the matched test errors.

Only the training set changes, not the evaluation set; however, **training sample size and distribution both change**. This does not isolate whether any effect is due to additional data or altered motion physics. Do not treat selection among feature sets on these seven geometries as independent model generalization evidence.

## Run on Brown

From the repo root in the existing `(simpact)` environment:

~~~bash
git pull --ff-only origin phino-physics-experiments
python scripts/analyze_phino_pilot_grid.py --self-test
python scripts/analyze_phino_pilot_grid.py \
  --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
  --sphere-transfer \
  | tee results/phino_sphere_training_comparison.log
~~~

Do **not** combine `--sphere-transfer` with `--feature-ablation` or `--also-object-holdout`, since this mode already performs its own matched geometry holdouts and tests all fixed feature sets.

## Output

Written only to `results/phino_mesh_grid_cutoff_1m_v1/recovery_analysis/sphere_transfer/`, leaving the existing recovery reports unchanged:

- `matched_predictions.csv`: matched predictions with training condition and per-fold training counts
- `matched_scores.csv`: mass and friction MAE without vs with sphere; signed MAE difference
- `per_object_scores.csv`: each held-out non-sphere geometry and training condition
- `coverage.csv`: valid/missing configurations for each geometry/action set
- `sphere_motion.csv`: sphere Push X/Y travel, translational speed, angular speed, and approximate `v/(R * |omega|)` using **assumed radius 0.05 m**
- `report.json`: metadata and explicit limitations

The rolling-speed ratio is only a coarse magnitude comparison, **not a physical no-slip test**. Angular speed includes all rotation axes, and contact-point velocity, slip, support normal, mesh shape, and simulation rendering are not evaluated here. A value near 1 alone does not demonstrate rolling without slip.

Examine the earlier GIFs and the new speed summary together. An outcome may show that the sphere rolls or spins rapidly, but quantitative physical accuracy remains unverified.

**Important:** These are synthetic physical properties and a small exploratory dataset. No real-world geometry/physics inference or PHINO model training has been demonstrated.
