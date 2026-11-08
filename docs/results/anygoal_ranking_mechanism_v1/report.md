# Why SMWM misranks the Insert-Peg action bank

Fifty Insert-Peg episodes, the seed-0 bank of 16 sequences, and `sensorimotor_insertpeg.ckpt`. The goal is the recorded frame 25 actions after the start. Predicted opportunity regret on this bank is 0.64 [0.50, 0.78]. The same goal cost on the encoded simulated future is 0.06 [0.00, 0.14].

## Stage 1: state-action coverage

Candidate 0 is the episode's own first 25 actions. Candidates 1-7 add Gaussian noise at 0.1 training standard deviations. Candidates 8-15 are 25-step windows from other training episodes. The pose ball is 5 cm in hand and object position and 0.15 rad in every arm joint, measured against frames outside episodes 0-49. Nearest-neighbor distance is not used as evidence of support.

Labels: 50 observed continuations, 350 perturbed actions, and 400 recombined expert actions. Transplanted windows come from 1411 other training episodes. Their source hand is 0.41 m from the anchor start (minimum 0.03 m).

Outside episodes 0–49 there are 1158883 frames. A 5 cm / 0.15 rad ball around the anchor start contains another training frame in 0 of 50 episodes (median count 0). The recorded continuation itself leaves that ball within the 25 steps (0.11 [0.10, 0.12] of its steps remain inside). An empty or tiny ball means these starts are not a cloud of alternative recorded interventions. It does not, by itself, explain a prediction that is equally wrong on the stored continuation.

| Group | Physical success | Prediction SSE | Predicted goal cost | Observed goal cost | Chosen by predicted cost |
|---|---:|---:|---:|---:|---:|
| recorded | 0.82 [0.70, 0.92] | 51.41 [46.09, 57.43] | 60.19 [54.16, 66.82] | 7.35 [6.68, 8.04] | 0.00 [0.00, 0.00] |
| perturbation | 0.72 [0.60, 0.82] | 51.92 [46.54, 58.00] | 60.73 [54.75, 67.25] | 7.62 [6.94, 8.33] | 0.22 [0.12, 0.34] |
| transplant | 0.00 [0.00, 0.00] | 52.52 [48.77, 56.38] | 60.40 [55.69, 64.97] | 43.87 [39.73, 48.08] | 0.78 [0.66, 0.88] |

Paired prediction SSE, transplant minus recorded: 1.11 [-3.74, 5.79].
Selection counts across the 16 candidates are [0, 2, 2, 1, 3, 1, 1, 1, 6, 5, 4, 5, 5, 4, 6, 4]. When the recorded continuation succeeds (0.82 of episodes), predicted regret is 0.76 [0.61, 0.88] and the model selects that continuation on 0.00 [0.00, 0.00] of those episodes. Observed-latent regret on the same episodes is 0.05 [0.00, 0.12].

Pair accuracy of the predicted goal cost is 0.50 [0.41, 0.60]. Pair accuracy of the observed goal cost is 0.98 [0.96, 0.99]. Uniform selection has regret 0.47 [0.41, 0.54]. Predicted regret minus that expectation is 0.17 [0.06, 0.27].

On pairs whose actual futures differ by more than 1 in latent norm, the predicted change has cosine -0.03 [-0.08, 0.02] with the actual change, parallel component -0.10 [-0.26, 0.07], predicted norm 2.84 [2.72, 2.96], and observed norm 5.74 [5.40, 6.08].

Coverage decision: `present_on_recorded_transitions`.

## Stage 2: first block versus later blocks

Horizons are raw Insert-Peg actions. SMWM consumes them in blocks of five, so one raw action is not a model prediction. Autoregressive predictions start from the true first frame. Teacher predictions encode the true frame at the previous block boundary and predict one block. Goal costs compare a prediction with the encoded 25-step goal image. The 25-step physical success is the saved evaluator outcome.

One raw action is not a model output. Across the bank it moves the hand by 0.0072 [0.0066, 0.0078] m, so candidates have not separated yet.

| Horizon | Recorded AR SSE | Recorded teacher SSE | Transplant AR SSE | Predicted regret vs 25-step success | Pair accuracy |
|---:|---:|---:|---:|---:|---:|
| 5 | 46.71 [40.62, 53.05] | 46.71 [40.60, 53.12] | 47.80 [43.79, 51.82] | 0.74 [0.62, 0.86] | 0.46 [0.36, 0.56] |
| 10 | 52.13 [46.29, 58.11] | 49.50 [43.79, 55.45] | 51.46 [47.34, 55.62] | 0.76 [0.64, 0.88] | 0.52 [0.43, 0.61] |
| 25 | 51.39 [45.93, 57.37] | 45.77 [40.19, 51.62] | 52.51 [48.67, 56.32] | 0.64 [0.50, 0.76] | 0.50 [0.41, 0.60] |

Recorded first-block SSE is 46.71 [40.63, 53.14]. Recorded fifth-block autoregressive SSE is 51.39 [46.02, 57.53]. The last-block prediction that starts from the true image at step 20 still has recorded error 45.77 [40.19, 51.62]. The recorded hand has left the isolated start pose by then, so the local error is not confined to that pose.

Horizon decision: `present_at_first_block`.

At the first block, where the outcome has already moved, effect cosine is -0.03 [-0.07, 0.02], predicted norm 3.33 [3.00, 3.71], observed norm 2.09 [1.99, 2.20].

## Stage 3

Not run. The observed-latent goal cost already reads the encoded future. Its pair accuracy is far above chance and its regret is near zero, so the outcome encoding retains the distinction the decision needs. A supervised probe would add labels the world model does not have, and a failed probe would not show that the information is absent.

## Stage 4: action conditioning

The start image is held fixed and the 16 action sequences vary. Small perturbations move the encoded future by 0.51 [0.47, 0.55] and the prediction by 0.45 [0.34, 0.57]. Transplanted sequences move the encoded future by 5.90 [5.57, 6.23] and the prediction by 2.92 [2.80, 3.05]. The prediction does respond to the action. On futures that actually separate, its component along the true change is -0.10 [-0.26, 0.07]. The same cosine at the first block is -0.03 [-0.07, 0.02].

## What this supports

The 0.64 regret is a bad ordering of goal costs. Successful recorded and perturbed actions end near the goal in the encoder, and transplanted actions do not. The forward model assigns those groups nearly the same goal cost, because its action-conditioned update points somewhere other than the actual next embedding. That error is already present on the stored expert continuation, whose start state and 25 actions are a training trajectory. The horizon comparison places the same failure at the first action block.

AnyGoal reports that expert data can make a world model under-use the commanded action, and that a low prediction error need not yield a usable plan. The Insert-Peg result agrees that the planner's latent cost is not a goal ranking. It does not agree that the failure appears only after the action leaves the expert support: the expert continuation is predicted no better. SMWM's inverse loss is designed to keep action information in the encoder of real transitions. That part is working here, because the encoded real futures rank the goal. The forward rollout used for planning does not inherit a usable direction. Action Influence Maximization shows that expert demonstrations can make the next state predictable without the action, which would show up as a near-zero predicted change. The measured change is not near zero when the action actually changes the future. The direction of that change is unrelated to the future the encoder records.

Unresolved: this is one checkpoint and the 50 training episodes used as the anchor. Teacher-forced blocks later in those same episodes are not a held-out episode. The transplanted sequences are expert strings from other poses, not every intervention the training support could contain. No new loss was trained.

Next action: stop. The ranking error is on the stored expert continuation, and it is already present at the first action block. Another representation objective is not the measurement this result asks for.
