# Trajectory-similarity baselines

Intervention subsets share the same configurations and overlap in actions. Averages over subsets are descriptive. The paired summary's unit is the held-out geometry. Subset counts are not a sample size, and this report does not treat them as independent draws.

Leave-one-geometry-out normalized MAE at all four interventions. One intervention set, so no subset average enters the primary paired differences.

The 4x4 curves use the published nine-subset list. Count 1 is every single action and count 4 is the full set. Count 2 averages two representative subsets (push_lo+push_hi and push_hi+toss), and count 3 is the single subset push_lo+push_hi+toss. Those are not every combination.

Fairness limitations:

- Generic InfoNCE uses the same clips, shared-frame colour jitter and translation, batch of 24, 400 steps, backbone learning rate 1e-5, head learning rate 1e-3, weight decay 0.05, 40-step warmup, and last four trainable blocks as trajectory-similarity SoftInfoNCE. Its temperature is the same learnable parameter, initialized at 0.07. It does not read a trajectory similarity.
- Supervised regression uses those same training choices and the same two augmented views per step. Its target is log mass and log friction, standardized with the mean and population standard deviation of the training geometries only. It sees the probe's label family during training. Trajectory similarity and generic InfoNCE do not. The regression head has the same hidden width as the contrastive head, is not L2-normalized, and is discarded before the probe.
- Every learned run trains on one intervention per batch. The probe concatenates one to four interventions afterwards. Regression therefore asks every single intervention, including a friction-blind toss, to predict both properties.
- No learning rate, step count, augmentation, or ridge penalty was chosen using a test geometry. The ridge penalty is chosen by grouped cross-validation inside the training geometries only.
- Sample-weighted nMAE gives cow, cube, and nut most of the weight. White carton has six configurations. The paired differences give each held-out geometry one vote.
- The trajectory-similarity checkpoints from the earlier pilot were not retrained. The image-position tracker was recomputed with the new probe files and checked against that pilot.

Configurations: 292.

## Pooled CLS‖mean-patch

Sample-weighted leave-one-geometry-out nMAE, seed-averaged. For pooled features, each count averages every subset of that size. Those subsets are dependent. On the 4×4 readout, count 2 averages the two published representative subsets and count 3 is one representative subset.

| method | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| frozen DINOv2 | 0.88 / 0.95 | 0.85 / 0.85 | 0.80 / 0.88 | 0.76 / 0.86 |
| trajectory similarity | 0.88 / 0.85 | 0.74 / 0.72 | 0.67 / 0.65 | 0.60 / 0.60 |
| generic InfoNCE | 0.83 / 0.85 | 0.74 / 0.76 | 0.69 / 0.70 | 0.62 / 0.66 |
| supervised regression | 0.84 / 0.88 | 0.73 / 0.79 | 0.64 / 0.69 | 0.54 / 0.60 |
| image-position tracker | 0.47 / 0.58 | 0.25 / 0.29 | 0.07 / 0.13 | 0.04 / 0.12 |

Primary endpoint: all four interventions, per held-out geometry. Each learned method is seed 0 / seed 1. Frozen and the tracker have one fit.

### mass

| geometry | n | frozen | trajectory | InfoNCE | regression | tracker |
|---|---|---|---|---|---|---|
| blue carton | 46 | 0.57 | 0.50 / 0.50 | 0.39 / 0.48 | 0.36 / 0.40 | 0.03 |
| cow | 85 | 0.75 | 0.68 / 0.66 | 0.74 / 0.64 | 0.34 / 0.26 | 0.06 |
| cube | 85 | 1.00 | 0.63 / 0.65 | 0.75 / 0.77 | 0.95 / 0.87 | 0.04 |
| nut | 70 | 0.60 | 0.56 / 0.53 | 0.58 / 0.40 | 0.52 / 0.41 | 0.05 |
| white carton | 6 | 0.96 | 0.61 / 0.63 | 0.63 / 0.59 | 1.01 / 0.93 | 0.02 |
| sample-weighted | 292 | 0.76 | 0.61 / 0.60 | 0.65 / 0.59 | 0.58 / 0.51 | 0.04 |

Paired geometry differences at this endpoint (first method minus second; negative means the first has lower error). Each geometry counts once.

- trajectory_minus_instance: -0.001 (geometry range -0.119 to +0.070; both seeds lower 1/5; sign disagreement 3/5; mean |seed difference| 0.079)
- trajectory_minus_regression: -0.009 (geometry range -0.348 to +0.370; both seeds lower 2/5; sign disagreement 0/5; mean |seed difference| 0.078)
- instance_minus_regression: -0.009 (geometry range -0.360 to +0.388; both seeds lower 2/5; sign disagreement 1/5; mean |seed difference| 0.054)
- trajectory_minus_frozen: -0.179 (geometry range -0.360 to -0.057; both seeds lower 5/5; sign disagreement 0/5; mean |seed difference| 0.019)

### friction

| geometry | n | frozen | trajectory | InfoNCE | regression | tracker |
|---|---|---|---|---|---|---|
| blue carton | 46 | 0.77 | 0.56 / 0.51 | 0.53 / 0.57 | 0.47 / 0.54 | 0.07 |
| cow | 85 | 0.96 | 0.78 / 0.76 | 0.84 / 1.01 | 0.77 / 0.72 | 0.17 |
| cube | 85 | 0.78 | 0.46 / 0.44 | 0.42 / 0.45 | 0.52 / 0.46 | 0.10 |
| nut | 70 | 0.88 | 0.56 / 0.63 | 0.66 / 0.68 | 0.62 / 0.63 | 0.13 |
| white carton | 6 | 0.90 | 0.80 / 0.89 | 0.76 / 0.52 | 0.57 / 0.65 | 0.03 |
| sample-weighted | 292 | 0.86 | 0.60 / 0.61 | 0.63 / 0.69 | 0.61 / 0.59 | 0.12 |

Paired geometry differences at this endpoint (first method minus second; negative means the first has lower error). Each geometry counts once.

- trajectory_minus_instance: -0.005 (geometry range -0.156 to +0.207; both seeds lower 2/5; sign disagreement 2/5; mean |seed difference| 0.138)
- trajectory_minus_regression: +0.047 (geometry range -0.040 to +0.242; both seeds lower 1/5; sign disagreement 2/5; mean |seed difference| 0.053)
- instance_minus_regression: +0.052 (geometry range -0.056 to +0.181; both seeds lower 1/5; sign disagreement 1/5; mean |seed difference| 0.131)
- trajectory_minus_frozen: -0.218 (geometry range -0.334 to -0.049; both seeds lower 5/5; sign disagreement 0/5; mean |seed difference| 0.048)

Within-geometry held-out physics, all four interventions, sample-weighted nMAE (seed 0 / seed 1 where two seeds exist):

| method | mass | friction |
|---|---|---|
| frozen DINOv2 | 0.35 | 0.69 |
| trajectory similarity | 0.34 / 0.33 | 0.62 / 0.61 |
| generic InfoNCE | 0.33 / 0.31 | 0.58 / 0.59 |
| supervised regression | 0.31 / 0.31 | 0.54 / 0.54 |
| image-position tracker | 0.08 | 0.19 |

## 4×4 spatial readout

Sample-weighted leave-one-geometry-out nMAE, seed-averaged. For pooled features, each count averages every subset of that size. Those subsets are dependent. On the 4×4 readout, count 2 averages the two published representative subsets and count 3 is one representative subset.

| method | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| frozen DINOv2 | 0.81 / 0.84 | 0.72 / 0.58 | 0.66 / 0.54 | 0.67 / 0.59 |
| trajectory similarity | 0.82 / 0.84 | 0.66 / 0.58 | 0.55 / 0.44 | 0.53 / 0.49 |
| generic InfoNCE | 0.80 / 0.84 | 0.69 / 0.63 | 0.56 / 0.48 | 0.55 / 0.52 |
| supervised regression | 0.79 / 0.85 | 0.67 / 0.68 | 0.49 / 0.50 | 0.48 / 0.52 |
| image-position tracker | 0.47 / 0.58 | 0.09 / 0.12 | 0.07 / 0.11 | 0.04 / 0.12 |

Primary endpoint: all four interventions, per held-out geometry. Each learned method is seed 0 / seed 1. Frozen and the tracker have one fit.

### mass

| geometry | n | frozen | trajectory | InfoNCE | regression | tracker |
|---|---|---|---|---|---|---|
| blue carton | 46 | 0.56 | 0.46 / 0.47 | 0.34 / 0.41 | 0.33 / 0.37 | 0.03 |
| cow | 85 | 0.62 | 0.66 / 0.62 | 0.77 / 0.64 | 0.27 / 0.28 | 0.06 |
| cube | 85 | 0.89 | 0.59 / 0.52 | 0.59 / 0.63 | 0.83 / 0.74 | 0.04 |
| nut | 70 | 0.54 | 0.43 / 0.42 | 0.47 / 0.35 | 0.53 / 0.37 | 0.05 |
| white carton | 6 | 0.70 | 0.58 / 0.55 | 0.62 / 0.54 | 0.70 / 0.62 | 0.02 |
| sample-weighted | 292 | 0.67 | 0.55 / 0.51 | 0.57 / 0.53 | 0.51 / 0.45 | 0.04 |

Paired geometry differences at this endpoint (first method minus second; negative means the first has lower error). Each geometry counts once.

- trajectory_minus_instance: -0.007 (geometry range -0.065 to +0.086; both seeds lower 1/5; sign disagreement 3/5; mean |seed difference| 0.085)
- trajectory_minus_regression: +0.025 (geometry range -0.229 to +0.365; both seeds lower 2/5; sign disagreement 1/5; mean |seed difference| 0.051)
- instance_minus_regression: +0.032 (geometry range -0.173 to +0.430; both seeds lower 3/5; sign disagreement 0/5; mean |seed difference| 0.064)
- trajectory_minus_frozen: -0.133 (geometry range -0.332 to +0.020; both seeds lower 4/5; sign disagreement 0/5; mean |seed difference| 0.033)

### friction

| geometry | n | frozen | trajectory | InfoNCE | regression | tracker |
|---|---|---|---|---|---|---|
| blue carton | 46 | 0.51 | 0.51 / 0.43 | 0.53 / 0.58 | 0.46 / 0.56 | 0.07 |
| cow | 85 | 0.93 | 0.71 / 0.67 | 0.68 / 0.74 | 0.78 / 0.66 | 0.17 |
| cube | 85 | 0.43 | 0.36 / 0.33 | 0.36 / 0.37 | 0.44 / 0.39 | 0.10 |
| nut | 70 | 0.43 | 0.43 / 0.46 | 0.45 / 0.45 | 0.40 / 0.38 | 0.13 |
| white carton | 6 | 0.52 | 0.53 / 0.52 | 0.50 / 0.40 | 0.49 / 0.58 | 0.03 |
| sample-weighted | 292 | 0.59 | 0.50 / 0.48 | 0.50 / 0.53 | 0.54 / 0.49 | 0.12 |

Paired geometry differences at this endpoint (first method minus second; negative means the first has lower error). Each geometry counts once.

- trajectory_minus_instance: -0.013 (geometry range -0.087 to +0.075; both seeds lower 2/5; sign disagreement 2/5; mean |seed difference| 0.075)
- trajectory_minus_regression: -0.020 (geometry range -0.072 to +0.054; both seeds lower 1/5; sign disagreement 3/5; mean |seed difference| 0.092)
- instance_minus_regression: -0.007 (geometry range -0.086 to +0.059; both seeds lower 1/5; sign disagreement 2/5; mean |seed difference| 0.107)
- trajectory_minus_frozen: -0.069 (geometry range -0.240 to +0.013; both seeds lower 2/5; sign disagreement 3/5; mean |seed difference| 0.039)

Within-geometry held-out physics, all four interventions, sample-weighted nMAE (seed 0 / seed 1 where two seeds exist):

| method | mass | friction |
|---|---|---|
| frozen DINOv2 | 0.33 | 0.58 |
| trajectory similarity | 0.32 / 0.32 | 0.54 / 0.54 |
| generic InfoNCE | 0.31 / 0.30 | 0.51 / 0.52 |
| supervised regression | 0.30 / 0.30 | 0.49 / 0.49 |
| image-position tracker | 0.08 | 0.19 |

## Position decoding

Leave-one-geometry-out mean R² of the 15-d image-position target, stored by the probe. This is a diagnostic, not the primary endpoint.

### Pooled

| method | push_x_v2 | push_lo_1.6N | push_hi_3.2N | toss_4.0N |
|---|---|---|---|---|
| trajectory seed 0 | -0.32 | 0.19 | 0.04 | -0.01 |
| trajectory seed 1 | -0.35 | 0.16 | 0.04 | -0.01 |
| InfoNCE seed 0 | -0.53 | 0.10 | 0.08 | -0.09 |
| InfoNCE seed 1 | -0.76 | 0.13 | 0.02 | 0.01 |
| regression seed 0 | -0.69 | 0.23 | -0.04 | 0.04 |
| regression seed 1 | -0.44 | 0.21 | 0.06 | 0.02 |

### 4×4

| method | push_x_v2 | push_lo_1.6N | push_hi_3.2N | toss_4.0N |
|---|---|---|---|---|
| trajectory seed 0 | -0.24 | 0.20 | 0.08 | 0.02 |
| trajectory seed 1 | -0.27 | 0.20 | 0.07 | -0.03 |
| InfoNCE seed 0 | -0.33 | 0.22 | -0.08 | -0.06 |
| InfoNCE seed 1 | -0.42 | 0.20 | 0.03 | 0.01 |
| regression seed 0 | -0.45 | 0.22 | 0.02 | 0.03 |
| regression seed 1 | -0.36 | 0.22 | 0.08 | -0.01 |
