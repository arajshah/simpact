# Spatial-representation diagnostic

The fine-tuned 4×4 evaluation in `results/phino_action_learning_v2/probes_grid_ft/` is complete: 140 result rows, 40460 predictions, all finite, same nine intervention subsets as the frozen 4×4 run. This file only reads those outputs and recomputes per-geometry position R² from the saved features. Pooled position R² matches the stored probe files.

Subsets behind the n=1..4 means (not every combination; n=2 and n=3 are representative):

- n=0: `static`
- n=1: `push_hi_3.2N`
- n=1: `push_lo_1.6N`
- n=1: `push_x_v2`
- n=1: `toss_4.0N`
- n=2: `push_hi_3.2N+toss_4.0N`
- n=2: `push_lo_1.6N+push_hi_3.2N`
- n=3: `push_lo_1.6N+push_hi_3.2N+toss_4.0N`
- n=4: `push_x_v2+push_lo_1.6N+push_hi_3.2N+toss_4.0N`

Leave-one-geometry-out mean nMAE on those subsets (mass / friction). n=0 is the shared before image and is omitted here.

| representation | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| frozen pooled | 0.88 / 0.95 | 0.83 / 0.79 | 0.75 / 0.79 | 0.76 / 0.86 |
| fine-tuned pooled, seed 0 | 0.87 / 0.85 | 0.69 / 0.68 | 0.61 / 0.54 | 0.61 / 0.60 |
| fine-tuned pooled, seed 1 | 0.89 / 0.84 | 0.71 / 0.66 | 0.61 / 0.54 | 0.60 / 0.61 |
| frozen 4x4 | 0.81 / 0.84 | 0.72 / 0.58 | 0.66 / 0.54 | 0.67 / 0.59 |
| fine-tuned 4x4, seed 0 | 0.84 / 0.83 | 0.64 / 0.59 | 0.55 / 0.45 | 0.55 / 0.50 |
| fine-tuned 4x4, seed 1 | 0.80 / 0.84 | 0.68 / 0.58 | 0.55 / 0.42 | 0.51 / 0.48 |
| image-position tracker | 0.47 / 0.58 | 0.09 / 0.12 | 0.07 / 0.11 | 0.04 / 0.12 |

Per held-out geometry, leave-one-geometry-out nMAE. Each cell is the mean over the subsets of that size.

### mass, 1 intervention

| geometry | n | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | image-position tracker |
|---|---|---|---|---|---|---|---|---|
| blue carton | 46 | 0.83 | 0.87 | 0.95 | 0.74 | 0.80 | 0.81 | 0.39 |
| cow | 85 | 0.87 | 1.04 | 1.03 | 0.82 | 1.04 | 0.87 | 0.42 |
| cube | 85 | 0.97 | 0.85 | 0.86 | 0.91 | 0.81 | 0.82 | 0.60 |
| nut | 70 | 0.83 | 0.71 | 0.73 | 0.72 | 0.69 | 0.69 | 0.44 |
| white carton | 6 | 0.89 | 0.70 | 0.69 | 0.69 | 0.68 | 0.66 | 0.48 |

### mass, 2 interventions

| geometry | n | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | image-position tracker |
|---|---|---|---|---|---|---|---|---|
| blue carton | 46 | 0.67 | 0.73 | 0.78 | 0.60 | 0.64 | 0.68 | 0.10 |
| cow | 85 | 0.81 | 0.73 | 0.74 | 0.72 | 0.71 | 0.79 | 0.08 |
| cube | 85 | 1.01 | 0.73 | 0.70 | 0.86 | 0.68 | 0.71 | 0.11 |
| nut | 70 | 0.74 | 0.57 | 0.65 | 0.65 | 0.52 | 0.52 | 0.09 |
| white carton | 6 | 0.87 | 0.54 | 0.54 | 0.60 | 0.55 | 0.49 | 0.06 |

### mass, 3 interventions

| geometry | n | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | image-position tracker |
|---|---|---|---|---|---|---|---|---|
| blue carton | 46 | 0.55 | 0.50 | 0.51 | 0.54 | 0.46 | 0.47 | 0.04 |
| cow | 85 | 0.69 | 0.66 | 0.65 | 0.57 | 0.64 | 0.62 | 0.08 |
| cube | 85 | 1.03 | 0.69 | 0.70 | 0.88 | 0.63 | 0.57 | 0.08 |
| nut | 70 | 0.60 | 0.55 | 0.52 | 0.57 | 0.41 | 0.50 | 0.06 |
| white carton | 6 | 1.04 | 0.58 | 0.60 | 0.69 | 0.55 | 0.53 | 0.06 |

### mass, 4 interventions

| geometry | n | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | image-position tracker |
|---|---|---|---|---|---|---|---|---|
| blue carton | 46 | 0.57 | 0.50 | 0.50 | 0.56 | 0.46 | 0.47 | 0.03 |
| cow | 85 | 0.75 | 0.68 | 0.66 | 0.62 | 0.66 | 0.62 | 0.06 |
| cube | 85 | 1.00 | 0.63 | 0.65 | 0.89 | 0.59 | 0.52 | 0.04 |
| nut | 70 | 0.60 | 0.56 | 0.53 | 0.54 | 0.43 | 0.42 | 0.05 |
| white carton | 6 | 0.96 | 0.61 | 0.63 | 0.70 | 0.58 | 0.55 | 0.02 |

### friction, 1 intervention

| geometry | n | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | image-position tracker |
|---|---|---|---|---|---|---|---|---|
| blue carton | 46 | 0.87 | 0.98 | 0.90 | 0.80 | 0.99 | 1.01 | 0.58 |
| cow | 85 | 0.92 | 0.86 | 0.88 | 0.84 | 0.82 | 0.85 | 0.59 |
| cube | 85 | 0.93 | 0.80 | 0.79 | 0.87 | 0.82 | 0.79 | 0.59 |
| nut | 70 | 1.08 | 0.86 | 0.82 | 0.82 | 0.80 | 0.84 | 0.57 |
| white carton | 6 | 0.94 | 0.77 | 0.79 | 0.64 | 0.66 | 0.68 | 0.55 |

### friction, 2 interventions

| geometry | n | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | image-position tracker |
|---|---|---|---|---|---|---|---|---|
| blue carton | 46 | 0.65 | 0.78 | 0.79 | 0.48 | 1.00 | 0.91 | 0.11 |
| cow | 85 | 0.79 | 0.73 | 0.74 | 0.74 | 0.57 | 0.62 | 0.12 |
| cube | 85 | 0.62 | 0.67 | 0.62 | 0.57 | 0.55 | 0.52 | 0.13 |
| nut | 70 | 1.07 | 0.57 | 0.54 | 0.44 | 0.45 | 0.46 | 0.13 |
| white carton | 6 | 0.90 | 0.57 | 0.68 | 0.46 | 0.42 | 0.49 | 0.06 |

### friction, 3 interventions

| geometry | n | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | image-position tracker |
|---|---|---|---|---|---|---|---|---|
| blue carton | 46 | 0.63 | 0.49 | 0.49 | 0.51 | 0.47 | 0.43 | 0.08 |
| cow | 85 | 0.81 | 0.64 | 0.64 | 0.77 | 0.55 | 0.54 | 0.13 |
| cube | 85 | 0.58 | 0.59 | 0.53 | 0.52 | 0.51 | 0.42 | 0.09 |
| nut | 70 | 1.08 | 0.39 | 0.45 | 0.31 | 0.25 | 0.27 | 0.12 |
| white carton | 6 | 1.26 | 0.45 | 0.58 | 0.41 | 0.34 | 0.30 | 0.04 |

### friction, 4 interventions

| geometry | n | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | image-position tracker |
|---|---|---|---|---|---|---|---|---|
| blue carton | 46 | 0.77 | 0.56 | 0.51 | 0.51 | 0.51 | 0.43 | 0.07 |
| cow | 85 | 0.96 | 0.78 | 0.76 | 0.93 | 0.71 | 0.67 | 0.17 |
| cube | 85 | 0.78 | 0.46 | 0.44 | 0.43 | 0.36 | 0.33 | 0.10 |
| nut | 70 | 0.88 | 0.56 | 0.63 | 0.43 | 0.43 | 0.46 | 0.13 |
| white carton | 6 | 0.90 | 0.80 | 0.89 | 0.52 | 0.53 | 0.52 | 0.03 |

## Position decoding

Leave-one-geometry-out R² for the 15-d image-position target (centroid shift and log-area change), one action at a time. The probe is the same unclipped grouped ridge used in the original evaluation. Pooled R² matches the stored probe files and keeps every channel. Per-geometry R² uses that geometry's own mean and drops channels with no within-geometry variation, which happens for some toss components.

| representation | push_x_v2 | push_lo_1.6N | push_hi_3.2N | toss_4.0N |
|---|---|---|---|---|
| frozen pooled | -0.54 | 0.08 | 0.02 | 0.04 |
| fine-tuned pooled, seed 0 | -0.32 | 0.19 | 0.04 | -0.01 |
| fine-tuned pooled, seed 1 | -0.35 | 0.16 | 0.04 | -0.01 |
| static fine-tune, pooled | -0.39 | 0.11 | 0.03 | -0.00 |
| frozen 4x4 | -0.36 | 0.18 | 0.04 | 0.02 |
| fine-tuned 4x4, seed 0 | -0.24 | 0.20 | 0.08 | 0.02 |
| fine-tuned 4x4, seed 1 | -0.27 | 0.20 | 0.07 | -0.03 |
| static fine-tune, 4x4 | -0.39 | 0.23 | 0.11 | 0.06 |

Mean per-geometry R² across the four actions. R² uses each held-out geometry's own mean.

| geometry | frozen pooled | fine-tuned pooled, seed 0 | fine-tuned pooled, seed 1 | static fine-tune, pooled | frozen 4x4 | fine-tuned 4x4, seed 0 | fine-tuned 4x4, seed 1 | static fine-tune, 4x4 |
|---|---|---|---|---|---|---|---|---|
| blue carton (46) | -0.72 | -0.73 | -0.66 | -0.83 | -0.89 | -0.90 | -0.60 | -0.52 |
| cow (85) | -0.90 | -0.80 | -1.02 | -0.87 | -0.89 | -0.86 | -1.01 | -0.91 |
| cube (85) | -4.83 | -3.93 | -4.17 | -4.81 | -4.71 | -3.90 | -3.81 | -4.63 |
| nut (70) | -1.43 | -1.88 | -1.66 | -1.46 | -1.69 | -1.72 | -1.76 | -1.52 |
| white carton (6) | -12.76 | -5.18 | -4.81 | -9.99 | -10.15 | -4.16 | -3.38 | -10.51 |

Channels dropped as constant within the held-out geometry: blue carton `toss_4.0N` keeps 12/15; cube `toss_4.0N` keeps 12/15; white carton `toss_4.0N` keeps 12/15.

Static-image fine-tune control, four interventions, leave-one-geometry-out nMAE (mass / friction):

- static fine-tune, pooled: 0.86 / 0.83
- static fine-tune, 4x4: 0.66 / 0.55
