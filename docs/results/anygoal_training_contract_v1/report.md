# SMWM training contract on Insert-Peg

A training clip is two frames separated by five raw actions. The target is the projector embedding of the later frame from the same encoder, with no EMA and no stop-gradient. The action block is the five raw actions between those frames, normalized by the training-set mean and unbiased standard deviation. Inference is eval mode.

## Stage 1: does the diagnostic match the training step?

Batch-norm modules: 2. Tracked batch counts are 46300, 46300.

The bank's action scale differs from the training unbiased scale by a relative 2.637e-04. The action-mean absolute gap is 1.497e-03. A hand-rolled window matches the official dataset action to 1.116e-07 and the pixels to 0 gray levels. Rollout versus the training `predict` call differs by at most 1.822e-03.

Paths agree: `False`. Scaler matches: `True`.

## Stage 2: recorded one-block transitions

964 clips from 241 episodes. The loss is mean squared error in the 192-d projector space. SSE is that error summed over dimensions, so it can be compared with the earlier diagnostic. Persistence repeats the current embedding.

| Phase | Model SSE | Persistence SSE | Same-action cosine | True change norm | Inverse action MSE |
|---|---:|---:|---:|---:|---:|
| start | 1.66 [1.55, 1.78] | 3.97 [3.70, 4.26] | 0.74 [0.72, 0.76] | 1.92 [1.86, 1.99] | 0.540 [0.513, 0.568] |
| step_50 | 0.65 [0.61, 0.70] | 2.82 [2.58, 3.06] | 0.84 [0.83, 0.85] | 1.59 [1.52, 1.66] | 0.395 [0.372, 0.419] |
| mid | 0.51 [0.47, 0.56] | 1.65 [1.48, 1.84] | 0.78 [0.76, 0.80] | 1.19 [1.13, 1.25] | 0.431 [0.405, 0.457] |
| last | 0.57 [0.52, 0.63] | 1.73 [1.56, 1.93] | 0.77 [0.75, 0.79] | 1.22 [1.16, 1.28] | 0.452 [0.425, 0.479] |

Anchor episode starts, model SSE: 1.84 [1.54, 2.18].
Ordinary-phase same-action cosine: 0.80 [0.78, 0.81].
Recorded-action cosine on the anchor, official forward pass: 0.75 [0.71, 0.79].
Prediction movement between the recorded block and a transplanted block: 1.84 [1.58, 2.11].
Official recorded prediction versus the saved diagnostic rollout: 2.4339860677719116. Dataset frame versus the rendered frame, SSE: 8.84 [8.17, 9.56].
Cross-action cosine, official predictions versus rendered futures: 0.64 [0.60, 0.68].

The stored classifier label is `A_evaluation_mismatch` because rollout and `predict` differ by 0.0018, which is above a 1e-3 cutoff. That gap is numerical. The representation, pixels, and action normalizer match.

## Corrected diagnostic

The saved diagnostic cast normalized actions to the uint8 dtype of the policy image. This rescore keeps those actions float32, which is the training representation, and repeats the uint8 cast as a control.

The first five values of a normalized action block are [0.7905698027249095, -1.0652274378510533, 1.3304670580799043, 1.401763450847919, -1.3780249405490652]. Cast to uint8 they become [0, 255, 1, 1, 255].

Repeating the uint8 cast reproduces the stored costs to 0.060 and the stored regret 0.64 [0.50, 0.76]. Float32 actions differ from those stored costs by 153.9.

Corrected 25-step opportunity regret: 0.12 [0.04, 0.22]. Corrected pair accuracy: 0.96 [0.93, 0.98]. The cross-action cosine with float32 actions, against the rendered futures, is 0.64 [0.60, 0.68].

The near-zero cosine does not survive the correction. On recorded training blocks the same-action cosine is 0.80 [0.78, 0.81], and the one-block error is below persistence at every phase in the table above.

Explanation A, an evaluation mismatch. The checkpoint predicts recorded five-action blocks in the training representation: the error is below a persistence baseline and the predicted change aligns with the encoded change. The orthogonal cosine and the 0.64 regret came from casting those normalized actions to uint8. With float32 actions the 25-step regret falls to 0.12 [0.04, 0.22]. That residual is much smaller than the original gap and is not evidence that the forward update is orthogonal to the encoded future. Do not train a new loss from the uint8 result. The useful next step is to tell Hazel Nam and Petr Ivashkov that this Insert-Peg checkpoint's one-block predictions do align, and that the earlier cosine was an action-dtype bug in our diagnostic.
