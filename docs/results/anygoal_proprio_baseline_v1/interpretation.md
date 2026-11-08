# Measured arm joints plus known physics

**Decision: INCONCLUSIVE for a new learning objective.** Current joint readings remove the arm-estimation bottleneck. The ranking gap that remains is an inaccurate estimate of the goal pose from the goal image, not a failure of physics once that pose is known.

## What each system can see

SMWM's planner transform receives the current image and the goal image. It does not receive proprioception. The Insert-Peg environment reports a separate 33-d proprio field: eight joint positions as sin/cos, eight joint velocities, five touch readings, and the hand pose. The object and the slot target are in the privileged part of the 44-d state, and the dataset stores them under privileged names.

The hand pose is forward kinematics of those joints. Restoring a true state and reading the hand back agrees to 1e-8 m. A stored frame and a render from that state differ by 2.7 gray levels on average, so the image and the label are aligned.

Episode starts are a different pose from the rest of the trajectory. On the training episodes, 83% of start configurations put at least one joint outside the central 90% of the mixed-frame sample. The hand is 46 cm above the floor at the start and 20 cm later, and it is 41 cm from the peg at the start and 13 cm later. That is the distribution shift behind the earlier 11 cm start-frame hand error. It is not an alignment bug.

## Ranking on the same 50-episode bank

The current arm, including joint velocity, is the measured proprio state. The object and slot at the start come from the episode-start visual checkpoint. The goal pose comes from the mixed-frame visual checkpoint, because the goal is 25 steps later. Neither checkpoint was chosen using episodes 0–49. Success is the true outcome from the original start.

| Ranker | Information | Success | Opportunity regret |
|---|---|---:|---:|
| True state | privileged simulator state | 0.84 | 0.00 [0.00, 0.00] |
| Measured arm, visual object, goal pose given | proprio plus a specified goal | 0.74 | 0.10 [0.02, 0.20] |
| Measured arm, true object, visual goal | privileged object, visual goal | 0.46 | 0.38 [0.24, 0.52] |
| Measured arm, visual object and visual goal | proprio plus images | 0.42 | 0.42 [0.28, 0.56] |
| SMWM predicted cost | images only | 0.20 | 0.64 [0.50, 0.76] |

Zeroing the estimated object velocity does not change the visual-goal result (regret 0.40 [0.26, 0.54]). On the anchor goal images the mixed-frame estimator has median hand error 10.8 cm and object error 9.4 cm, and 16 of 50 object estimates are more than 20 cm off, 11 of them more than 40 cm. Replacing that object estimate with the truth, while the goal is still visual, only moves regret from 0.42 to 0.38. The goal-image estimate is the remaining error.

When the goal pose is given and the arm is measured, visual object error of 6.4 cm leaves regret at 0.10 [0.02, 0.20]. The upper end of that interval is 0.20, so this is the gate's neighborhood rather than a tight pass. It is also a different interface from a goal image.

## Decision

**INCONCLUSIVE.** A robot that reads its encoders and runs Insert-Peg physics does not need a learned arm estimate. SMWM does not use those readings. With the arm measured and the goal pose specified, regret is 0.10 [0.02, 0.20]. With the goal read from the image by the current estimator, regret stays at 0.42 because that estimate is often tens of centimeters off. That is an inaccurate goal-pose estimate, not evidence that accurate state plus physics fails, and not a reason for a new learning objective.

**Next measurement.** Estimate the goal pose from the goal image accurately enough that the held-out object and hand errors are inside 2 cm, without using episodes 0–49, then repeat the measured-arm ranking. The current arm is no longer the obstacle.
