# Visual state estimation plus known physics

**Decision: INCONCLUSIVE.** The estimator did not reach the 2 cm held-out checkpoint, so estimated-state physics was not accepted as a test of the simulator. The 0.48 regret below is what this inaccurate arm estimate does. It is not evidence that a competent state estimate fails, and it is not a reason to propose a new learning objective.

Episodes 0–49 were the existing Insert-Peg anchor and were excluded from training, normalization, and model selection. Episodes 50–1299 trained. Episodes 1300–1460 validated. The encoder is the cached DINOv2-B checkpoint with an MLP head. State labels are simulator observations. SMWM does not receive those labels.

## Held-out accuracy

| Quantity | Mixed frames, median / 90th | Episode starts, median / 90th |
|---|---:|---:|
| Hand position | 4.9 cm / 16.1 cm | 11.0 cm / 22.9 cm |
| Object position | 4.5 cm / 10.6 cm | 6.7 cm / 13.3 cm |
| Target position | 2.8 cm / 6.0 cm | 3.4 cm / 7.7 cm |
| Object angle | 0.10 rad / 0.29 rad | 0.13 rad / 0.33 rad |
| Worst joint angle | 0.39 rad / 1.35 rad | 0.91 rad / 2.07 rad |
| Frames with both positions under 5 cm | 31% | 7% |

The anchor starts, which were not used to train or select the model, match the episode-start validation: hand 9.7 cm, object 6.4 cm, object angle 0.12 rad. Peg orientation is recovered. The arm configuration at the initial approach pose is not.

An equal-dimension loss on raw quaternions was worse on orientation (object angle 1.17 rad) because about 10% of nearby frames flip the quaternion sign. The reported model regresses sin and cos of the planar angle and averages the loss by state group. That change fixed the angle and did not get either hand or object position to 2 cm. Full-backbone finetuning did not either. One L40 was used for about 0.4 GPU-hour.

## What that estimate does to the bank

These rankings use the episode-start model. Success is the true outcome from the original start, not the outcome inside the estimated simulator.

| Ranker | Success | Opportunity regret |
|---|---:|---:|
| True state, Insert-Peg physics | 0.84 | 0.00 [0.00, 0.00] |
| True pose, object velocity set to zero | 0.76 | 0.08 [0.02, 0.16] |
| True arm, remaining state estimated | 0.72 | 0.12 [0.04, 0.22] |
| Estimated start and estimated goal | 0.36 | 0.48 [0.34, 0.62] |
| SMWM predicted latent cost | 0.20 | 0.64 [0.50, 0.76] |

Replacing only the estimated object pose, object velocity, or slot target leaves regret at 0.46–0.54. Replacing the arm drops it to 0.12. The arm estimate is the variable that matters.

Object-position noise added to the true start, with everything else true, does not recreate SMWM's gap: regret is 0.00, 0.08, 0.10, 0.04, and 0.06 at 0, 0.5, 1, 2, and 5 cm. A 2 cm object error would have been accurate enough for this bank. The estimator's problem is the arm, not that 2 cm of object error is too coarse for contact.

Anchor arm velocity is already zero. Setting the true object's initial velocity to zero, with the true pose kept, raises regret only to 0.08. Velocity is not what produces the 0.64 SMWM regret at these starts. It is also not observable from one image; the reported velocity errors are predictions, not measurements.

## Decision

**INCONCLUSIVE.** The pre-registered 2 cm checkpoint was not met on episode-start validation (hand 11.0 cm, object 6.7 cm). Estimated-state physics therefore remains untested at the accuracy the gate requires. The descriptive regret of 0.48 comes from that arm error. A true arm plus the estimated remainder already reaches 0.12, and the true-state simulator remains at 0.

**Next measurement, still not a new objective.** Get episode-start hand and object median error to 2 cm on episodes 1300–1460 without using episodes 0–49, then repeat this ranking. Stop if the regret interval falls below 0.10. Do not read the present 0.48 as a failure of known physics.
