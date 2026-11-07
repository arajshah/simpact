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

The index-10 training reference above is a different 50-episode draw from the Stage A starts. On those Stage A starts, recorded-action replay reaches the dataset goal for 88% of Insert-Peg, 26% of Avoid-Peg, and 0% of Extract-Peg. Seeds 1 and 2, latent action-bank scores, and the Section 8.6 inference controls were not run. The replay gate is a stop, not a repaired benchmark.

