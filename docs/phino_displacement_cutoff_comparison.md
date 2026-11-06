# PHINO displacement-cutoff comparison: 0.25 m versus 1.0 m

The original seven-geometry grid used a **0.25 m raw displacement cutoff**. A subsequent sphere-only diagnostic completed all 18 Push X/Y trials for all 50 frames without reported numerical errors; sphere raw travel ranged from **0.484 m to 0.791 m**. These observations establish numerical completion, **not** physical realism. Inspect the nominal GIFs before interpreting the rolling motion as trustworthy.

The generator already offers `--max-displacement`. This update retains the default 0.25 m cutoff and adds the explicit comparison fields below for accepted trajectories:

- Per action: `action_metrics[action].max_raw_m` and `action_metrics[action].exceeds_legacy_displacement_cutoff`.
- Top-level manifest: `legacy_displacement_cutoff_m`, `max_raw_displacement_m`, `cutoff_differs_from_legacy`.
- Stdout: `VALID_ACTIONS_OVER_ORIGINAL_25CM_CUTOFF`.

The limit is a **validation-policy choice**, not a numerical-stability guarantee. Existing tilt, PHINO velocity-clipping, and MuJoCo warning checks still apply. No previously rejected trajectory is relabeled; this is a new independently generated grid with explicit provenance.

## Rerun a separate grid on Brown

From the SIMPACT repository root:

~~~bash
git pull --ff-only origin phino-physics-experiments
python scripts/phino_simpact_pilot_v2.py --self-test

python scripts/phino_simpact_pilot_v2.py \
  --scene results/rigid_0103/scene.xml \
  --scene-glob 'results/phino_mesh_scenes_v1/*/scene.xml' \
  --max-displacement 1.0 \
  --output-dir results/phino_mesh_grid_cutoff_1m_v1 \
  | tee results/phino_mesh_grid_cutoff_1m_v1.log
~~~

Do **not** overwrite `results/phino_mesh_grid_v1`, which contains the baseline 0.25 m experiment.

Then analyze the new dataset without changing its features, regularization, or holdout splits:

~~~bash
python scripts/analyze_phino_pilot_grid.py \
  --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
  --feature-ablation --also-object-holdout \
  | tee results/phino_mesh_recovery_cutoff_1m_v1.log
~~~

Compare action validity, per-geometry coverage, rejected reasons, and mean absolute error to the **existing** baseline. Note that any score change may reflect a change in which trajectories are evaluated, not an actual model improvement. A controlled matched-subset comparison can distinguish those effects.

Do not infer that friction is identifiable on rolling spheres from the MuJoCo contact-friction parameter alone. Do not train PHINO or use these simulated mass/friction settings as measured real-world labels.

The larger cutoff does not override the hard limits in the separate sphere diagnostic, and the existing four-action settings remain fixed.
