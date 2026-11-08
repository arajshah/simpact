# What the Extract-Peg failure is

The corrected bank failure is real, and it is not explained by the weld. Reaching that particular Extract-Peg pose does require the weld and the 25-substep action together. Ranking actions is a separate failure, and the same checkpoint already shows it on Insert-Peg under the training integrator.

## Dynamics, same goal and same bank

Fifty Extract-Peg episodes, the recorded 25-step goal, and the seed-0 16-action bank are fixed. Predicted costs are identical across integrators. Slot-contact edits in `ExtractPegEnv` change no state: the no-weld extract rollouts match the Insert-Peg rollouts exactly.

| Integrator | Bank solvable | Recorded action reaches the goal | Predicted success | Conditional predicted regret |
|---|---:|---:|---:|---:|
| No weld, 10 substeps | 0.10 | 0.02 | 0.02 | not comparable (5 episodes) |
| No weld, 25 substeps | 0.02 | 0.00 | 0.00 | not comparable (1 episode) |
| Weld, 10 substeps | 0.30 | 0.16 | 0.08 | not comparable (15 episodes) |
| Weld, 25 substeps | 0.98 | 0.94 | 0.28 | 0.71 [0.59, 0.84] |

Unconditional predicted regret on the weld/25 bank is 0.70 [0.58, 0.82]. Observed-latent regret is 0.06 [0.00, 0.14]. Those match the corrected audit, including the selected action on every episode. The best candidate under the training integrator is still about 32 mm from the peg goal. The episodes start inside the native insertion criterion (50/50) and the recorded 25-step goal is inside it on 14/50, so insertion and the dataset goal are different outcomes.

Paired terminal latent error, weld/25 minus the training integrator, is 0.52 [-0.40, 1.42]. The model is not a better predictor when the simulator is put back on the training timestep. Zeroing the initial velocity leaves weld/25 solvability at 0.98.

## Goals at fixed dynamics

Insert, Avoid, and Extract goals were generated under different integrators. Swapping those dataset goals is not a goal-only test. Under the weld and 25 substeps, both the recorded outcome and the farthest other realized outcome are physically achieved. Predicted regret is 0.80 [0.68, 0.90] and 0.94 [0.86, 1.00]. Observed regret is 0 for both. The selected action changes with the goal on 0.58 [0.44, 0.72] of episodes and still fails.

On the first 50 Insert-Peg episodes, at step 0, under the training integrator, the episode's own 25-step goal is bank-solvable on 0.84. Predicted regret is 0.64 [0.50, 0.76]. Observed regret is 0.06 [0.00, 0.14]. One model block, scored against the image five actions later, has predicted regret 0.78 [0.66, 0.88] and observed regret 0. After five actions, 8.4 candidates still match the recorded pose on average, so chance success is 0.53; the one-block selector succeeds on 0.22.

## Baselines on that insert bank

| Ranker | Information | Success | Opportunity regret |
|---|---|---:|---:|
| True state, Insert-Peg physics | privileged state and simulator | 0.84 | 0.00 |
| Observed latent cost | privileged future images | 0.78 | 0.06 [0.00, 0.14] |
| Uniform candidate | none | 0.37 | 0.48 |
| SMWM predicted latent cost | images; these episodes were in its training file | 0.20 | 0.64 [0.50, 0.76] |
| Goal-conditioned regression, 2000 steps | images and actions from episodes 50–129 | 0.28 | 0.56 [0.42, 0.70] |
| Image retrieval, same 1273 pairs | nearest neighbor is 36 cm away in hand position | 0.18 | 0.66 [0.52, 0.78] |
| Pose network plus simulator | held-out object error 19.5 cm; competence bar was 2 cm | 0.30 | 0.54 [0.40, 0.68] |

The simulator ceiling solves the bank. The image methods in this budget do not beat uniform selection, and the pose network misses its pre-set error bar, so their regrets are not evidence that a competent standard method fails.

## Decision

**NO-GO for a new learning objective.** The Extract-Peg pose is unreachable in this bank without both the weld and the longer action. The ranking failure remains when the goal and the integrator are the ones SMWM was trained on, including at a single model block. A true-state mechanics model removes that ranking gap. A competent image-to-state or goal-conditioned baseline was not obtained here, so the residual is not yet an equal-information failure of existing methods.

**Next experiment.** Train an image-to-state estimator on insert episodes 50–1460 until held-out median hand and object error are at most 2 cm. Then rank this same anchor bank from the estimated state with Insert-Peg physics. Stop if the regret interval falls below 0.10. Do not open a new objective from the present gap.
