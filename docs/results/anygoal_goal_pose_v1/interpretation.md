# Goal pose from the goal image

**Decision: INCONCLUSIVE.** A goal-offset readout lowers the measured-arm ranking regret from 0.42 to 0.14 [0.06, 0.24]. Held-out hand error is 2.9 cm and object error is 5.2 cm, so the 2 cm checkpoint is not met and this is not a completed test of known physics. It is also not a reason for a new learning objective.

## What the goal image is

The goal is the simulator state 25 steps after the episode start, rendered by the same fixed side camera as every other frame. That camera sits 16 m away on the side of the planar arm, with a 4-degree field of view, about 5 mm per pixel. Projecting the true wrist point and inverting the camera, with the known plane of the arm, recovers that point to numerical error. The stored goal frame and a re-render differ by about 2.7 gray levels. The image and the label are aligned, and the pose is not fundamentally ambiguous once the right pixel is known.

The wrist origin used by the goal metric is not a visible geom. The earlier estimator rebuilt it by forward kinematics from predicted joints, which is why goal-hand error stayed near 11 cm. This model predicts the metric quantities directly: hand position, hand angle, object position, and object angle. Training images are offsets 10, 20, 25, 30, and 40 from episodes 50–1299. Validation is offset 25 from episodes 1300–1460. Episodes 0–49 were not used to fit or select it. The labels are simulator poses. SMWM does not receive them.

## Held-out accuracy at the goal offset

| Quantity | Median | 90th percentile |
|---|---:|---:|
| Hand position | 2.9 cm | 6.6 cm |
| Hand angle | 0.064 rad | 0.17 rad |
| Object position | 5.2 cm | 51 cm |
| Object angle | 0.079 rad | 1.73 rad |

On the untouched anchor goal images the hand median is 3.4 cm and no hand error exceeds 20 cm. The object median is 4.2 cm, but 14 of 50 object estimates still exceed 20 cm.

## Ranking

The current arm is the measured joint state. The current object is the existing visual estimate, with object velocity set to zero. Success is the true outcome from the original start.

| Ranker | Success | Opportunity regret |
|---|---:|---:|
| True state | 0.84 | 0.00 [0.00, 0.00] |
| Measured arm, visual object, true goal | 0.78 | 0.06 [0.00, 0.14] |
| Measured arm, visual object, estimated goal | 0.70 | 0.14 [0.06, 0.24] |
| Same, true goal hand and estimated object | 0.76 | 0.08 [0.02, 0.16] |
| Same, estimated hand and true goal object | 0.70 | 0.14 [0.06, 0.24] |
| SMWM predicted cost | 0.20 | 0.64 [0.50, 0.76] |

Replacing the estimated goal object with the truth does not change regret. Replacing the estimated goal hand lowers it from 0.14 to 0.08. The object tail is real and does not drive this ranking. The remaining gap above the true-goal row is the last few centimeters of goal-hand error.

## Decision

**INCONCLUSIVE.** The pipeline with measured joints and a specified goal pose is at 0.06 [0.00, 0.14]. Reading the goal from the image moves that to 0.14 [0.06, 0.24]. The interval still reaches 0.24, and held-out object error is 5.2 cm, so the 2 cm rule is not satisfied. The result does show that the previous 0.42 regret was the old joint-based goal hand, and that a direct goal-hand readout is what the ranking uses. It does not show that an accurate goal pose leaves the physics pipeline failing, and it does not explain SMWM's latent predictions.
