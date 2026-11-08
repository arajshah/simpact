# AnyGoal contact-action diagnostic

## Stage A

| Dataset | Dataset-goal success | Paper SMWM | Native insert success | Final object success | Recorded-action replay | Train-hash overlap | Label |
|---|---:|---:|---:|---:|---:|---:|---|
| insertpeg | 76.0 | 70.0 | 6.0 | 80.0 | 88.0 | 50 / 50 | base-task reference on training source |
| avoid_peg | 78.0 | 74.0 | 2.0 | 98.0 | 26.0 | 0 / 50 | held-out goal file |
| extractpeg | 28.0 | 24.0 | 28.0 | 30.0 | 0.0 | 0 / 50 | held-out goal file |

Released 33-dimensional state predictors, velocities excluded, seed 0, 50 evaluations:

| Model | Dataset | Dataset-goal success | Native insert success |
|---|---|---:|---:|
| toy-adaln | insertpeg | 72.0 | 14.0 |
| toy-adaln | avoid_peg | 94.0 | 10.0 |
| toy-adaln | extractpeg | 22.0 | 26.0 |
| toy-action-duplicate | insertpeg | 76.0 | 4.0 |
| toy-action-duplicate | avoid_peg | 86.0 | 4.0 |
| toy-action-duplicate | extractpeg | 28.0 | 28.0 |

## Stage B

Decision: **NO-GO**

Recorded-action replay reached the recorded goal on fewer than 90% of states. This is a restoration, dynamics, or goal-alignment failure. Predicted-latent scores are not interpreted as evidence about the world model.

| Replay set | Full goal reached | Object pose reached | Median hand position | Median object position |
|---|---:|---:|---:|---:|
| Extract-Peg, first 50 at index 10 | 0.060 | 0.600 | 0.0759 | 0.0113 |
| insertpeg_train | 0.780 | 0.780 | 0.0008 | 0.0010 |
| avoid_peg | 0.000 | 0.820 | 0.1641 | 0.0000 |

Recorded continuations were replayed with InsertPegEnv.step, the evaluator used by the released planner. The 90% goal-reach check failed, so predicted-latent and observed-latent costs were not computed. The training-file reference uses the same player.

## Stage B, corrected extract dynamics

Starts at the first stored frame rather than index 10, and steps ExtractPegEnv for 25 physics substeps with the grasp weld engaged. The published InsertPegEnv.step results are unchanged.

Decision: **PENDING-STAGE-C**

Hypothesis H is supported on this bank: predicted-latent selection has high opportunity regret, and replacing the rollout with the actual future under the same terminal latent cost reduces that regret. Section 8.8 still requires the inference-only controls before a GO to standard-baseline diagnosis. This does not authorize a new representation method.

Recorded continuation at step 25 reaches the goal on 94.0% of episodes. Median one-step hand/object error 0.26 / 0.23 mm. Median 25-step hand/object error 1.39 / 0.68 mm.

| Quantity | Mean | 95% interval |
|---|---:|---:|
| P opportunity regret | 0.700 | [0.580, 0.820] |
| O opportunity regret | 0.060 | [0.000, 0.140] |
| P minus O regret | 0.640 | [0.500, 0.760] |
| H rate | 0.640 | [0.500, 0.780] |
| P and O fail while the bank succeeds | 0.060 | [0.000, 0.140] |

Bank solvability: 0.980.
Argmin agreement: 0.080.
Mean Spearman P vs O: 0.236.

### Replanning control

Open-loop 25-step CEM success 0.480 [0.340, 0.620]. Replan every 5 raw actions 0.460 [0.320, 0.600]. Model-query counts 1500 and 7500.

Extract start states beyond the training 95th-percentile neighbor distance: 0 / 50.

Final decision on the corrected protocol: **GO only to the memorandum's standard-baseline comparison.** Replanning does not remove the prediction failure. This does not authorize a new representation objective. The grasp weld is a real dynamical difference from Insert-Peg training and has to be available to those baselines.

