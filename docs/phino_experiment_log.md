# SIMPACT → PHINO experiment log

Reproducible experiments only. Current project state lives in [`phino_cursor_handoff.md`](./phino_cursor_handoff.md).

## 2026-11-05 — v0 PHINO dataset export and dataloader smoke test

**Question.** Can the existing 1.0-m SIMPACT grid be loaded by PHINO's real contrastive dataloader with correct image ↔ geometry ↔ physics ↔ fingerprint associations and no geometry leakage?

**Hypothesis.** Yes, with a 4-action normalizer and a geometry-disjoint split. This tests **interface compatibility only**. It says nothing about whether PHINO can learn hidden physics from static images.

**Data and code.**
- Source: `results/phino_mesh_grid_cutoff_1m_v1/manifest.json` (sha256 `69d208de…66d2`).
- SIMPACT `622693b` (`scripts/export_phino_dataset.py`).
- PHINO `3f1134d` (`eval/grouped-physics-probes-oct28`; contrastive loader identical to `main`).
- Command: `MUJOCO_GL=egl .venv/bin/python scripts/export_phino_dataset.py --grid-dir results/phino_mesh_grid_cutoff_1m_v1 --phino-root /nbu/people/ashah264/phino --output-dir data/phino_simpact_v0` (seed 0, two held-out geometries, one 224×224 view per geometry).

**Split** (seeded before any result; whole geometries):
- Train: bunny, cow, nut, blue carton, white carton (29 configurations).
- Held out: cube, sphere (11 configurations).

**Results.**
- **Validation.** 40/63 complete fingerprints. Every accepted fingerprint equals its four saved per-action traces in manifest order. 23 configurations were excluded (lift failures: 22; sphere `push_y` tilt: 1).
- **Mass coverage.** No 0.25 kg configuration is complete, because the 2.973 N lift exceeds the object's weight and overshoots the 1 m cap. Exported label counts are 0.30 kg: 14 and 0.35 kg: 15 in train.
- **Renders.** Object pixel coverage by segmentation is 0.106 (cow) to 0.670 (cube). PHINO's unmodified `render_multiview` framed the carton at about 0.5% coverage because it includes the table box.
- **Normalizer.** `TrajectoryNormalizer(4, 50)` was fit on 29 train fingerprints; σ = 2.425.
- **Train similarities.** 406 pairs: median 0.485, range 0.151–0.962, none below 1e-3. 70 pairs share one image.
- **Held-out similarities** (train normalizer): 55 pairs, median 6.5e-5; 56% are below 1e-3. Sphere `push_x`/`push_y` displacement and yaw-rate channels dominate, with sphere z-scores up to 40.2 (cube up to 8.3).
- **Dataloader smoke test.** `create_contrastive_dataloader(batch_size=4, num_workers=0, hard_neg_ratio=0, hard_pos_ratio=0)` returned `img_i`/`img_j` of shape `(4, 3, 224, 224)` for both splits. Batch similarities and mass/friction labels matched the exported files, and every loader pair was a precomputed pair.
- **Source integrity.** `sha256sum -c` over all 573 files of both source grids passed after the export.

**Limitations.**
- With seven geometries and two mass levels, held-out conclusions will have very low statistical power.
- Labels are synthetic simulator assignments, and restitution is a placeholder.
- Same-image pairs carry targets that a static-image encoder cannot satisfy.
- SoftInfoNCE uses only diagonal soft labels.

**Conclusion.** The interface works: one PHINO batch loads cleanly from SIMPACT-generated data with verified associations. No representation learning was performed or claimed. Held-out similarity saturation means trajectory-similarity labels for the sphere carry little gradation under a train-fit normalizer. Any held-out evaluation using these labels should account for that.

**Next decision (research owner).** Accept or revise the cube+sphere holdout, decide whether the missing 0.25 kg level matters, and choose the next experiment's observation modality before any training.

## 2026-11-05: Action-conditioned visual observability pilot

**Question.** Do images taken before and after a known push distinguish the grid's mass/friction configurations, which a static image cannot? What remains ambiguous?

**Hypothesis.** The final image after one push differs across configurations, but a single horizontal push confounds mass with friction.

**Data and code.**
- Source: the existing 1.0-m grid (manifest `69d208de…66d2`) and its saved valid `push_x` traces. `push_x` is valid for all 63 configurations, including 0.25 kg. No physics was regenerated.
- Code: SIMPACT `a8f813b`, `scripts/phino_visual_observability_pilot.py`. The run used that exact code before it was committed, so `summary.json` records base commit `5df97af` with a dirty working tree.
- Command: `MUJOCO_GL=egl .venv/bin/python scripts/phino_visual_observability_pilot.py` writes to `results/phino_visual_observability_v1/` (Brown only, about 6 MB).
- Committed summary: [`docs/results/phino_visual_observability_v1/`](./results/phino_visual_observability_v1/) holds `summary.json`, `state_metrics.csv`, and figures.

**Protocol.**
- **Rendering:** each frame is a kinematic replay: the object is placed at the saved pose in its isolated scene and rendered at 224×224, PHINO's input size. Camera, lighting, and action are fixed per geometry, and the camera is framed once on the union of all nine trajectories.
- **Replay check:** re-simulating one configuration per geometry reproduced the saved trace with a maximum difference of 0.0 for all seven geometries.
- **Image-space position:** the object's centroid in each frame, taken from MuJoCo segmentation masks.
- **Pairwise comparisons:** all 36 within-geometry pairs are compared on state final position, state path, `push_x`+`lateral_flick` final positions, image final centroid, image centroid path, and mean absolute pixel difference.
- **Start-aligned mode:** re-anchors each saved relative motion to one reference start pose, so every configuration has an identical initial image.

**Unexpected data-validity finding: the grid's starting states depend on friction.** The generator's settle step lets the object slide or rock differently depending on the friction coefficient; mass has no effect.

| Geometry | Start offset across friction | Residual start velocity | Matched start? |
|---|---|---|---|
| white carton | 0.8 mm, 0.2° | ≈0 | yes |
| blue carton | 2.1 mm, 0.9° | 0 | no (borderline) |
| bunny | 8.0 mm, 15.0° | 0.03 m/s, 1.12 rad/s | no |
| cow | 25.2 mm, 44.6° | 0 | no |
| cube, nut | ≤0.01 mm, 0.0° | ≈0 | yes |
| sphere | 3.1 mm, 3.6° | 0.03 m/s, 0.55 rad/s | no |

Consequences:
- For bunny and cow, the simulated pre-push images themselves differ by friction: 18 and 27 of 36 pairs have a mean pixel difference of at least 1/255. In the grid's own states, friction is therefore visible before any action.
- Fingerprint quaternion channels are absolute, so they carry this settle orientation too. PHINO similarities and earlier ridge probes on cow and bunny may partly reflect settling rather than the action response. This is not yet quantified.
- The PHINO v0 export images are unaffected, because they were rendered once per geometry from PHINO's own settle.

**Results** (start-aligned images, `push_x`, 224 px; `state_metrics.csv` and `summary.json` have per-configuration values):
- **Single push, final image.**
  - Across all seven geometries, 250 of 252 within-geometry pairs differ by at least 1 px in final centroid, and 240 by at least 3 px.
  - One grid step moves the final centroid by 8–21 px on average for mass (0.05 kg) and 11–25 px for friction (0.1).
  - Start-aligned initial images are identical (maximum difference ≤ 1/255).
  - **So yes: on this 3×3 grid, the response image distinguishes almost every configuration, whereas a single static image (export v0) cannot.**
- **What stays ambiguous: mass trades off against friction.**
  - Every geometry's closest pair differs in both properties along the heavy/low-friction versus light/high-friction diagonal. Examples: cube 0.30 kg/0.4 vs 0.35 kg/0.3 differ by 0.9 mm (0.7 px); blue carton by 2.0 mm (0.9 px).
  - Log-log fits of final displacement give mass and friction exponents with R² 0.83–0.98, except bunny `push_x` (0.57) and `lateral_flick` (0.42). For the cube, displacement ∝ m^-4.75 μ^-3.86 under `push_x`.
  - A push therefore reveals mainly one mass–friction combination. The grid points are distinct only because the 3×3 levels don't lie exactly on one iso-response curve. Over a continuous range, one push would not identify mass and friction separately.
- **The time course adds little.** The minimum image-path separation is within 0.5 px of the minimum final-position separation (cube 0.69 vs 0.71 px). Under these short impulses the paths have the same shape and differ in scale.
- **A second horizontal action does not break the confound.**
  - `lateral_flick`'s (mass, friction) exponent direction is within 0.7–5.6° of `push_x`'s for cube, cow, and both cartons. Combining both final positions raises the cube's minimum separation only from 0.9 to 1.9 mm.
  - Bunny (50°) and nut (16°) differ more, but bunny's fits are poor (R² 0.57 and 0.42).
  - The obvious confound-breaker, the mass-dependent lift, is invalid at 0.25 kg.
- **Sphere.**
  - Push displacement ∝ m^-0.96 μ^-0.04: a push reveals mass, but essentially nothing about friction.
  - The sphere is still rolling at 2.5 s, so its final frame is truncated in time.
  - Its whole-image mean pixel difference is below 1/255 for all 36 pairs despite at least 2.6 px of centroid separation: whole-image similarity largely misses a small moving object.
- **Unverified regime.** The cube still moves 1–7 mm when μmg (0.98–1.37 N) exceeds the 0.8 N push. This may be friction coupling from the simultaneous 0.02 N·m yaw torque, or contact compliance; not verified.

**Limitations.**
- Noiseless synthetic renders with a known camera, action, and segmentation: an upper bound on visual distinguishability.
- Start-aligned mode removes only the static settle offset, not its effect on the dynamics.
- Three levels per property; mass and friction are simulator assignments.
- Camera framing per geometry uses that geometry's trajectory extent.

**Conclusion.** Action-conditioned images carry physics information that static images lack. With one horizontal push, that information is essentially one mass–friction combination per geometry (mass alone for the sphere), not both properties. The grid's friction-dependent settling is a confound that must be resolved before any image or fingerprint study on cow, bunny, sphere, or blue carton.

**Is a frozen DINOv2 probe the right next experiment?** Not as a test of separate mass/friction recovery on the current data:
- The targets are confounded by design.
- The cow and bunny initial states leak friction.
- The useful signal is a few-pixel object displacement, which pooled whole-image features may discard.

A frozen DINOv2 probe is appropriate only as a narrower signal-preservation diagnostic:
- **Inputs:** start-aligned (initial, final) image pairs, using patch tokens rather than only the CLS token.
- **Target:** the identifiable response, i.e. log displacement or the fitted mass–friction combination.
- **Evaluation:** within geometry and geometry-held-out.
- **Reference baselines:** the segmentation-centroid baseline and the privileged-state baseline.
- **Restriction:** matched-start geometries only (white carton, cube, nut), or after the settle confound is fixed.

Separating mass from friction visually first needs an action design that breaks the confound, for example two push magnitudes or a lift valid across the mass range. That requires new simulation and therefore an explicit experiment specification.

## 2026-11-05: Stage 1 audit of friction-dependent settling

**Question.** How much does the grid's friction-dependent settle affect (a) PHINO trajectory similarities in the v0 export and (b) the earlier ridge mass/friction recovery results? Which earlier conclusions survive?

**Data and code.**
- Grid `results/phino_mesh_grid_cutoff_1m_v1/` (manifest `69d208de…66d2`) and export `data/phino_simpact_v0/`, both read-only.
- SIMPACT `5d1c91d`, `scripts/audit_phino_settle_confound.py`. The run used that code before it was committed.
- Command: `MUJOCO_GL=egl .venv/bin/python scripts/audit_phino_settle_confound.py` writes to `results/phino_settle_audit_v1/`; a copy of the outputs is in [`docs/results/phino_settle_audit_v1/`](./results/phino_settle_audit_v1/).
- MuJoCo was used only to recompute each configuration's settled start (the generator's own deterministic `build_model`) and to run a zero-action control: the object left untouched for the same 2.5 s window. No grid data was regenerated.

**Integrity checks.**
- The refit normalizer reproduces the exported `traj_norm.npz`.
- All 40 published `recovery_scores.csv` cells were reproduced exactly before any re-partitioning.
- The settle-swap transform is an exact identity when the orientation is unchanged (unit-tested).

**Settle diagnostics.** A geometry is "clean" if its start differs by less than 1 mm and 1° across friction levels and it is essentially at rest; the classification is computed, not assigned by name. Mass never changes the start (≤0.001 mm).

| Geometry | Start offset across friction | Zero-action drift over 2.5 s | Drift / `push_x` response (max) | Class |
|---|---|---|---|---|
| white carton | 0.8 mm, 0.25° | 0.04 mm | 0.4% | clean |
| cube, nut | ≤0.01 mm, 0.00° | 0.00 mm | ≤0.2% | clean |
| blue carton | 2.1 mm, 0.9° | 0.18 mm | 0.5% | confounded (static pose only) |
| cow | 25.2 mm, 44.6° | 0.00 mm | 0% | confounded (static pose only) |
| bunny | 8.0 mm, 15.0° | 1.93 mm, 2.5° | **14%** (median 5.5%) | confounded (pose + residual motion) |
| sphere | 3.1 mm, 3.6° | 8.66 mm, 8.9° | 1.7% | confounded (pose + residual rolling) |

**PHINO similarity audit** (v0 export; train-fit normalizer; seed 0).
- **Static settle effect: material only for cow and bunny.** Swapping only the settled orientation between two friction levels, with an identical response, reduces PHINO similarity to:
  - cow: mean 0.91, minimum 0.83, a median 16% share of the actual friction-pair distance;
  - bunny: mean 0.95, minimum 0.88, a 4% share;
  - sphere: at least 0.986;
  - clean geometries and blue carton: 1.000.
- **Bunny residual motion** contaminates its responses directly: settling motion is up to 14% of the push response.
- **Within-geometry rankings are stable** under start-relative fingerprints: rank correlation 0.985 (clean) and 0.975 (confounded). Mean similarity changes are 0.058 and 0.144.
- **New finding: cross-geometry similarities depend on mesh body-frame conventions.**
  - Replacing absolute quaternions with rotation-since-start (and body-frame with world-frame angular velocity) changes train cross-geometry similarities by 0.152 on average (maximum 0.46), with rank correlation 0.742.
  - Absolute orientation encodes each mesh's arbitrary authoring frame, not dynamics.
  - PHINO's reference engine records the same absolute channels, so this also applies upstream. It is recorded here, not modified.
- **Held-out saturation is unchanged** (held-out mean change 0.009). The earlier finding that 56% of held-out pairs fall below 1e-3 stands.

**Ridge audit.** MAE is shown as ridge / training-mean baseline; mass in kg, friction as a coefficient.

| Earlier result | Original | Clean held-out geometries only | Verdict |
|---|---|---|---|
| Physics holdout (leave one (m, μ) out; geometries seen), `push_x`+`push_y`, original features | mass 0.0319/0.0375, friction 0.0590/0.0749 | mass 0.0338/0.0378, friction 0.0511/0.0755; clean-only training: 0.0345/0.0375, 0.0554/0.0750 | **Holds.** Friction gain about 30%; mass gain modest, about 10%. |
| Six-geometry object holdout, motion magnitudes, `push_x`+`push_y`, mass | 0.0266/0.0333 | 0.0283/0.0333 | **Weaker.** About half of the reported gain came from confounded test objects (0.0249). |
| Six-geometry object holdout, no vertical position, `push_x`+`push_y`, friction | 0.0471/0.0667 | 0.0452/0.0667 | **Holds.** |
| Sphere held out: extrapolation failure | MAE up to 0.20 kg / 0.59 | n/a | **Holds.** Sphere settle drift is 1.7% of its 480–720 mm push response, far too small to explain the failure. |
| Matched with/without sphere: mixed | mixed | still mixed (e.g. no-vertical friction 0.0452 → 0.0494; original `push_x`+`push_y` friction 0.0546 → 0.0449) | **Holds as "mixed".** |
| Seven-geometry object holdout (sphere included as a test geometry) | fails (friction 0.11–0.13 vs 0.067) | clean held-out geometries beat baseline on friction (0.045–0.055) | Failure is driven by the confounded test set, mainly the sphere. |

The ridge features are start-relative positions plus speeds and orientation change from the first saved frame. The static settle offset therefore enters them only through its effect on dynamics. Training on the three clean geometries alone (two per fold) is unstable: for example, original features with all three non-lift actions give mass 0.106 and friction 0.32. So this audit cannot tell whether the cross-geometry gains *require* confounded training data.

**Limitations.**
- Three clean geometries.
- The settle effect on dynamics (a different contact pose) cannot be separated from friction in the existing data.
- The counterfactual swaps only orientation, not position (position channels are already start-relative).

**Conclusion.**
- **Still valid:** within-geometry recovery on held-out physics, with friction more recoverable than mass; friction transfer to clean held-out geometries; the sphere's out-of-distribution failure; the "mixed" sphere-inclusion result.
- **Confounded:** any result evaluated on cow or bunny (pose, plus bunny residual motion), and part of the mass-transfer gain.
- **Invalid as physical similarity:** v0 cross-geometry PHINO similarities.

v3 must use matched, at-rest starts and start-relative orientation channels.

## 2026-11-05: Stage 3 state-only identifiability pilot (v3 interventions)

**Question.** Which v3 interventions make mass and friction separately identifiable from simulator states, at ideal precision and at realistic 224×224 visual precision? Does the toss give a measurable, mass-specific airborne response? Does the evidence pass the pre-registered gate for building the full v3 dataset?

**Design.** Approved spec [`phino_v3_identifiability_spec.md`](./phino_v3_identifiability_spec.md) §6, plus two amendments:
- **Precision models.**
  - *visual224:* 1 px localization error, with px/m set by a fixed camera framing each geometry's motion for that action (overhead-oblique for pushes, side view for the toss). Speed and deceleration errors are propagated by finite differences or a least-squares fit at the observation rate; yaw is treated as invisible for the sphere.
  - *ideal:* the same model with 1000× smaller errors.
  - The spec's original σ values are also reported.
- **Toss validation:** airborne frames from contact detection, apex size in pixels, friction-to-mass sensitivity ratio, and toss-only off-grid mass prediction.

**Data and code.**
- SIMPACT `c3bf96a`: `scripts/phino_simpact_grid_v3.py` and `scripts/phino_v3_identifiability_pilot.py`. The run used that code before it was committed.
- Command: `.venv/bin/python scripts/phino_v3_identifiability_pilot.py` (about 75 s with 7 workers) writes to `results/phino_v3_identifiability_pilot_v1/` (Brown only; includes a 40 MB `traces.npz`).
- Committed summary: [`docs/results/phino_v3_identifiability_pilot_v1/`](./results/phino_v3_identifiability_pilot_v1/).
- **Simulations:**
  - 7 geometries × 85 configurations (3×3 grid, 4 off-grid points, 72 finite-difference neighbours);
  - each with a common start, a zero-action control, and 7 candidate actions;
  - plus a half-timestep rerun of the centre configuration.
  - No full grid, renders, or training. Earlier grids are byte-identical.

**Start protocol (S1–S3).**

| Geometry | Approved S1 | Drift reading | Configurations with verified start |
|---|---|---|---|
| blue carton, cow | pass | pass | 85/85 |
| cube, nut | fail | pass | 85/85 |
| white carton | fail | fail (0.0104 mm vs 0.01 mm) | 38/85; S2 fails from resting-contact chatter (speed 5e-4–1e-3 m/s) |
| bunny | fail | fail | 2/85; residual rocking about 0.1 rad/s |
| sphere | fail | fail | 0/85; keeps rolling (S3 fails 82/85) |

The approved S1 speed thresholds (1e-4 m/s, 1e-3 rad/s) are below MuJoCo's resting-contact jitter for mesh contacts. That is why cube and nut fail S1 even though they are motionless to within 0.001 mm.

**Validity and pre-registered selection.**
- **Pure centre-of-mass pushes topple tall objects:** white carton 3/9 valid at every push candidate (tilts about 90°), blue carton 4–6/9; nut flips at 2.0 N and 4.0 N. The sphere exceeds the 1 m cap under every new push, and fails the 60° tilt rule under both tosses (7/9).
- **Result:** the pre-registered rule selects **no push candidate**, whether or not the sphere is included. The toss (4.0 N) is selected only when the sphere is excluded.
- **Post-hoc analysis:** a validity-only fallback (most-valid candidate per family: push_lo 1.6 N, push_hi 3.2 N, toss 4.0 N) and all 8 candidate combinations were analysed as clearly labelled post-hoc analyses.

**Gate.** **Fails.** At most 2 of 6 non-sphere geometries pass, and only in some settings, in every one of these combinations:
- all candidate combinations;
- ideal, visual224, and spec precision;
- both S1 readings;
- M1 thresholds of 7, 8, or 9 of 9 points, and M2 ratio thresholds of 0.4, 0.5, or 0.6.

For the fallback choice:
- ideal precision: cow and cube pass (drift reading), cow only (approved S1);
- visual224: cow only, with friction ratio 0.50, exactly at the threshold.

**Which interventions identify what** (fallback candidates; "verified" = blue carton, cow, cube, nut; M1 ≥ 8/9 and M2 ≤ 0.5):

| Intervention / observation | Ideal state precision | 224-px visual precision |
|---|---|---|
| Final image only (O1) | cube only: friction (push_lo); both (all three actions). This relies on final-yaw and toss landing-position features. Cow passes none. | nothing |
| v2 reference push (O2) | friction only (cow, cube, nut); 21 of 36 points have non-smooth derivatives | nothing |
| One new push, 20 Hz (O2) | friction only for push_lo on cube; push_hi nothing | nothing |
| One new push, 100 Hz (O4) | **both** (cow, cube), via post-push deceleration | nothing: per-frame differencing noise grows with frame rate |
| push_lo + push_hi, 20 Hz | nothing | nothing |
| **toss (O2/O4)** | **mass**: 4/4 verified geometries | **mass**: 4/4 |
| push_hi + toss, 20 Hz | both (cube); mass (cow) | mass (cow, cube) |
| push_lo + push_hi + toss, 20 Hz | **both** (cow, cube) | both (cow, at threshold 0.50); mass (cube, friction 0.52) |

**Hypotheses.**
- **H1** (two pushes, final state only, ill-conditioned) **is supported.**
  - In real contacts the final-displacement sensitivity directions of push_lo and push_hi are only **2.1–8.4°** apart (Coulomb toy model: about 5°).
  - Fitted exponents match the toy model closely (push_hi: m^−2.3 μ^−1.35 against a predicted m^−2.38 μ^−1.38).
  - At ideal precision only the final-yaw feature gives local full rank, and off-grid prediction still fails (cow 1.02/0.98).
- **H2** (push_hi identifies μ through deceleration where sliding is long enough) **is supported physically but not at 20 Hz or 224 px.**
  - Deceleration is resolvable at 20 Hz in 5–8 of 9 configurations, and in all 9 at 100 Hz.
  - At ideal precision push_hi alone identifies both properties at 100 Hz (cow 0.25/0.32, cube 0.27/0.16), but not at 20 Hz.
- **H3** (the toss makes mass identifiable) **is supported.**
  - The toss is airborne for at least 5 frames with an apex of 38–202 mm (at least 12 px).
  - Its apex is **identical across all five geometries** with a verified start (it depends on force over mass only), and its friction sensitivity is 0.000.
  - Toss-only off-grid mass error is 0.07–0.12 of baseline (ideal) and 0.26–0.29 (visual), against about 1.1 for friction.
  - The toss sensitivity direction is 30–38° from the pushes.
  - **Amendment 2 is satisfied** for blue carton, cow, cube, and nut. It cannot be assessed for white carton, bunny, or sphere, which have no verified starts.
- **H4** (sphere friction unidentifiable) is **not testable**: the sphere fails the start protocol and the validity rules.

**Other measurements.**
- **Cross-geometry** (leave one geometry out among the 4 verified, all three actions, 20 Hz): mass/friction error ratios are 0.08–0.14 / 0.12–0.20 (ideal) and 0.15–0.20 / 0.25–0.27 (visual). These are better than the within-geometry results, which suggests the 9-point within-geometry test (M2) is limited by sample size. Only 4 similar geometries were used, so treat this as indicative.
- **Solver robustness:** cow and cube features are stable at half the timestep. Blue carton and bunny push features change by more than the visual σ, consistent with contacts near tipping.

**Limitations.**
- Idealized pixel-noise model: no occlusion, motion blur, or segmentation error.
- 9 training points per geometry.
- Post-hoc candidate choice.
- The sphere and three other geometries are unevaluable as designed.

**Conclusion.**
- The toss is a validated, shape-independent, friction-blind mass intervention.
- Two push strengths observed only at the end remain degenerate, as predicted.
- Friction requires time-resolved post-push deceleration. It is identifiable from ideal states (100 Hz single push, or all three actions at 20 Hz) but only marginally at 224 px and 20 Hz.
- The evidence does **not** support generating the full v3 dataset with this protocol: the gate fails because of protocol problems (start criteria, push toppling, sphere), not only because of identifiability.
- Proposed v3.1 revisions are in the spec, §10.

## 2026-11-06: Action-conditioned representation pilot

**Question.** On the existing Stage 3 traces, does contrastive fine-tuning of DINOv2 with physical-intervention supervision improve mass and friction prediction relative to frozen DINOv2, and does that change with the number of observed interventions?

**Hypothesis.** Clip fine-tuning should beat a frozen encoder and a static-image fine-tune that sees identical images for every physics assignment. More interventions should help if each one adds information. This does not reopen the failed Stage 3 gate and does not justify a new physics grid.

**Data and code.**
- Observations: `data/phino_simpact_v3obs_v2/` (Brown only; provenance in [`docs/results/phino_action_learning_v2/dataset_provenance.json`](./results/phino_action_learning_v2/dataset_provenance.json)). Kinematic replay of saved Stage 3 traces. No new physics. Replay of one toss per included geometry matched the saved trace with maximum difference 0.0.
- Inclusion: configurations whose recomputed start passed S2 and S3, and geometries with at least 20 such configurations. Bunny (2) and sphere (0) are excluded. Frames are anchored at the canonical rest pose, and every clip of a geometry and camera shares one before image. A discarded v1 anchored each configuration at its own start; sub-0.1 mm friction-dependent offsets leaked into static probes (within-geometry nMAE 0.39). Do not use `data/phino_simpact_v3obs_v1/` or `results/phino_action_learning_v1/`.
- The probe table keeps the 292 configurations that have all four saved actions: blue carton 46, cow 85, cube 85, nut 70, white carton 6. White carton has 38 start-ok configurations, but most pushes were not saved.
- Actions: `push_x_v2`, `push_lo_1.6N`, `push_hi_3.2N`, `toss_4.0N`. Frames at 0, 0.15, 0.25, 0.35, 0.50, and 2.50 s, 224×224.
- SIMPACT generator and trainer: commit `f48d1e1` (dataset provenance records base `f0bcc05` with those scripts still uncommitted at generation time). PHINO `3f1134d`. The 4×4 spatial readout was added after `f48d1e1` and was used only to extract features, not to retrain.
- Commands, from the SIMPACT root inside the existing GPU allocation (no new Slurm job): `bash scripts/run_phino_action_learning.sh`, then `bash scripts/run_phino_action_grid_features.sh`. Pooled probes: `results/phino_action_learning_v2/probes_all/`. Frozen spatial probes: `results/phino_action_learning_v2/probes_grid_frozen/`.
- All 15 runs wrote `done.json` at 400 steps. No training process is still running. Logs contain no tracebacks.

**Protocol.**
- Representation, identical for every encoder: for each intervention, concatenate `f(frame) − f(before)` over the five post-onset frames; concatenate interventions in fixed action order. The static control is the push-camera before image.
- Probe: standardized dual ridge on log mass and log friction separately. Alpha is chosen by inner grouped cross-validation on the training split only. Predictions are clipped to the training log-label range ± 0.5. nMAE is MAE divided by the MAE of the training-mean predictor on the same test samples (1.0 = no better than the mean).
- Splits: leave-one-geometry-out, and within-geometry leave-one-physics-cluster-out. The test geometry is excluded from that fold's training, normalizer fit, and fine-tuning.
- Fine-tuning: PHINO `DynaCLIPModel` (DINOv2-B/14, last 4 blocks trainable, CLS‖mean-patch) and dense SoftInfoNCE. Each batch is one intervention. The target is the full pairwise similarity of that intervention's start-relative trajectory, with `TrajectoryNormalizer(n_actions=1, timesteps=50)` fit on training geometries only. Colour jitter and a small translation are shared across a clip; flips are not used. 400 steps, batch 24 clips, backbone learning rate 1e-5, head 1e-3. Seeds 0 and 1 for clips; seed 0 for the static control. About 49 minutes of training in total.
- `metrics.csv` records every 10th step, which never lands on `push_hi`. That is a logging gap. Training still cycles all four actions.

**Results.** Scaling curves average nMAE over every intervention subset of a given size, on the same 292 configurations. The figure is [`docs/results/phino_action_learning_v2/intervention_scaling.png`](./results/phino_action_learning_v2/intervention_scaling.png); the table is [`scaling_summary.md`](./results/phino_action_learning_v2/scaling_summary.md).

Leave-one-geometry-out, mean nMAE (mass / friction):

| Representation | 0 (static) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Frozen DINOv2 | 1.00 / 0.99 | 0.88 / 0.95 | 0.85 / 0.85 | 0.80 / 0.88 | 0.76 / 0.86 |
| Clip fine-tune, seed 0 | 1.00 / 0.97 | 0.87 / 0.85 | 0.74 / 0.73 | 0.67 / 0.66 | 0.61 / 0.60 |
| Clip fine-tune, seed 1 | 1.00 / 0.97 | 0.89 / 0.84 | 0.73 / 0.72 | 0.67 / 0.65 | 0.60 / 0.61 |
| Static-image fine-tune | 1.00 / 0.99 | 0.89 / 1.00 | 0.88 / 0.83 | 0.87 / 0.84 | 0.86 / 0.83 |
| Image-position tracker | — | 0.47 / 0.58 | 0.25 / 0.29 | 0.07 / 0.13 | 0.04 / 0.12 |
| Privileged state summary | — | 0.56 / 0.63 | 0.34 / 0.39 | 0.21 / 0.23 | 0.20 / 0.18 |

- **Clip fine-tuning beats frozen features across geometries.** Mean delta nMAE versus frozen, over the 15 non-static subsets: seed 0, −0.094 mass (14/15 subsets) and −0.154 friction (14/15); seed 1, −0.093 (14/15) and −0.163 (15/15). With all four interventions, every held-out geometry improves. Cube mass goes from 1.00 (frozen) to 0.63 / 0.65. White carton friction is the weak cell (0.80 and 0.89 on only 6 configurations).
- **The static-image control does not.** Its leave-one-geometry-out mass delta versus frozen is +0.035 (4/15 subsets improved). Static probes on the shared before image are 1.00, so the v1 friction leak is gone. Logged clip loss falls from about 3.0 to 1.3–2.0; static loss stays near 2.5–2.8.
- **Within a geometry, fine-tuning adds little for mass** (mean delta −0.003 and −0.014). Frozen features already track the toss: within-geometry toss mass nMAE is 0.02. Friction within a geometry improves only slightly (about −0.06).
- **The image-position tracker is the ceiling on these renders.** Fifteen numbers from the segmentation centroid and area, with the same ridge probe, reach 0.04 / 0.12 at four interventions. The privileged physics summary (displacement, peak speed, apex, yaw, stop time) is 0.20 / 0.18. Raw state frames and the full trajectory, fed to the same linear probe, stay near 0.7–1.0 across geometries: a high-dimensional linear fit is a poor privileged baseline here.
- **Pooled DINOv2 features do not keep that image motion.** Leave-one-geometry-out R² for decoding the centroid shift and log-area change is at most 0.19 (clip seed 0, `push_lo`) and is negative for `push_x` (−0.54 frozen, −0.32 and −0.35 after clip fine-tuning).
- **A frozen 4×4 average of patch tokens helps only partly,** on the same nine representative subsets (`probes_grid_frozen`). At four interventions, leave-one-geometry-out nMAE is 0.67 / 0.59, versus 0.76 / 0.86 for pooled features and 0.04 / 0.12 for the tracker. Position-decoding R² stays ≤ 0.18. Static remains 1.00 / 0.98.

**Limitations.**
- Five geometries, and the white-carton fold has six configurations. Labels are synthetic simulator assignments. Renders are noiseless, with a known camera and no occlusion.
- The Stage 3 gate failed. These actions are the post-hoc fallback pushes plus the reference push and the toss, not a re-registered protocol. Configurations without a saved trace for an action are absent.
- 400 steps and two seeds. The probe is linear. Within-geometry gains can be hidden by the small clusters.
- The static control was run at seed 0 only.
- Cross-geometry similarity targets are trajectory similarities, not a claim that those trajectories match real materials.

**Conclusion.** Action-conditioned images on this grid contain mass and friction information that a static image does not: a tracker that reads object position already predicts both, and more interventions help it. Contrastive fine-tuning of pooled DINOv2 features preserves some of that information across geometries and beats a static-image fine-tune, but it remains several times worse than the tracker. The frozen spatial readout does not close the gap. Do not treat this as evidence that PHINO has learned transferable physical properties, and do not generate the v3.1 grid until R1–R6 are approved.

**Spatial-readout diagnostic, completed the same day.** `results/phino_action_learning_v2/probes_grid_ft/` was already finished: 140 result rows, 40460 predictions, all finite, both clip seeds plus the static control, both splits, and position decoding. It uses the same nine subsets as `probes_grid_frozen`. No model was retrained and no features were regenerated. `scripts/phino_spatial_diagnostic.py` checks that completeness and recomputes per-geometry position R² from the saved banks. Its pooled R² matches the probe files. Tables: [`docs/results/phino_action_learning_v2/spatial_diagnostic/summary.md`](./results/phino_action_learning_v2/spatial_diagnostic/summary.md).

The nine subsets are every single action, the full four-action set, `push_lo+push_hi`, `push_hi+toss`, and `push_lo+push_hi+toss`. Means at one and at four interventions therefore match the curves above. Means at two and three interventions do not: they are not averages over every combination.

Leave-one-geometry-out mean nMAE on these subsets (mass / friction):

| Representation | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Frozen pooled | 0.88 / 0.95 | 0.83 / 0.79 | 0.75 / 0.79 | 0.76 / 0.86 |
| Fine-tuned pooled, seed 0 | 0.87 / 0.85 | 0.69 / 0.68 | 0.61 / 0.54 | 0.61 / 0.60 |
| Fine-tuned pooled, seed 1 | 0.89 / 0.84 | 0.71 / 0.66 | 0.61 / 0.54 | 0.60 / 0.61 |
| Frozen 4×4 | 0.81 / 0.84 | 0.72 / 0.58 | 0.66 / 0.54 | 0.67 / 0.59 |
| Fine-tuned 4×4, seed 0 | 0.84 / 0.83 | 0.64 / 0.59 | 0.55 / 0.45 | 0.55 / 0.50 |
| Fine-tuned 4×4, seed 1 | 0.80 / 0.84 | 0.68 / 0.58 | 0.55 / 0.42 | 0.51 / 0.48 |
| Image-position tracker | 0.47 / 0.58 | 0.09 / 0.12 | 0.07 / 0.11 | 0.04 / 0.12 |

Four interventions, per held-out geometry. Each pair is mass / friction. In the fine-tuned columns the first pair is seed 0 and the second is seed 1.

| Geometry | n | Frozen pooled | Fine-tuned pooled | Frozen 4×4 | Fine-tuned 4×4 | Tracker |
|---|---|---|---|---|---|---|
| Blue carton | 46 | 0.57 / 0.77 | 0.50 / 0.56, 0.50 / 0.51 | 0.56 / 0.51 | 0.46 / 0.51, 0.47 / 0.43 | 0.03 / 0.07 |
| Cow | 85 | 0.75 / 0.96 | 0.68 / 0.78, 0.66 / 0.76 | 0.62 / 0.93 | 0.66 / 0.71, 0.62 / 0.67 | 0.06 / 0.17 |
| Cube | 85 | 1.00 / 0.78 | 0.63 / 0.46, 0.65 / 0.44 | 0.89 / 0.43 | 0.59 / 0.36, 0.52 / 0.33 | 0.04 / 0.10 |
| Nut | 70 | 0.60 / 0.88 | 0.56 / 0.56, 0.53 / 0.63 | 0.54 / 0.43 | 0.43 / 0.43, 0.42 / 0.46 | 0.05 / 0.13 |
| White carton | 6 | 0.96 / 0.90 | 0.61 / 0.80, 0.63 / 0.89 | 0.70 / 0.52 | 0.58 / 0.53, 0.55 / 0.52 | 0.02 / 0.03 |

Per-geometry tables for one, two, and three interventions are in the summary. Within a geometry, four interventions, the tracker is 0.08 / 0.19 and the best visual readout (fine-tuned 4×4) is about 0.32 / 0.54. The large remaining gap is the cross-geometry one.

Position decoding, pooled leave-one-geometry-out R² (the stored metric):

| Representation | push_x | push_lo | push_hi | toss |
|---|---|---|---|---|
| Frozen pooled | −0.54 | 0.08 | 0.02 | 0.04 |
| Fine-tuned pooled, seed 0 / seed 1 | −0.32 / −0.35 | 0.19 / 0.16 | 0.04 / 0.04 | −0.01 / −0.01 |
| Frozen 4×4 | −0.36 | 0.18 | 0.04 | 0.02 |
| Fine-tuned 4×4, seed 0 / seed 1 | −0.24 / −0.27 | 0.20 / 0.20 | 0.08 / 0.07 | 0.02 / −0.03 |

Among frozen and clip-fine-tuned readouts the best cell is 0.20; the static-trained 4×4 control reaches 0.23 on the low push. Averaged over actions, horizontal centroid shift is the only positive channel (frozen 4×4 about 0.32; fine-tuned 4×4 about 0.35–0.37). Vertical shift and log-area stay negative. Inside a held-out geometry, R² on the channels that actually vary is negative for every geometry. Both seeds improve that within-geometry R² together in 7 of 16 action–geometry cells once the six-configuration white carton is set aside, and in 10 of 20 cells if it is kept. Cube `push_x` is largely negative (about −14 to −18): the transferred linear map extrapolates badly there. Three toss channels are exactly constant for the blue carton, the cube, and the white carton, and those channels are omitted from the within-geometry R² only.

The static-image 4×4 control, at four interventions, is 0.66 / 0.55. That sits next to frozen 4×4 (0.67 / 0.59) and behind clip-fine-tuned 4×4 (0.51–0.55 / 0.48–0.50). The static pooled control remains 0.86 / 0.83.

**What this distinguishes.** The fine-tuned 4×4 numbers are a spatial readout of backbones trained with the pooled contrastive loss. The loss never saw the 4×4 features.

- The renders contain the mass and friction signal: the tracker, which reads only the segmentation centroid and area, reaches 0.04 / 0.12 across all held-out predictions at four interventions; its per-geometry scores differ.
- Replacing the pooled descriptor with a 4×4 average, backbone held fixed, moves four-intervention nMAE from 0.76 / 0.86 to 0.67 / 0.59 and mean position R² from about −0.10 to −0.03. That is a real spatial effect. The tracker, at 0.04 / 0.12, remains far ahead, and position R² stays near zero.
- Clip fine-tuning improves the averages for both seeds, on pooled features and on the 4×4 readout, and it beats the static control. Cow mass is the clear exception: frozen 4×4 is 0.62, and the two fine-tuned seeds are 0.66 and 0.62. Friction gains that both seeds share are cow and cube; blue carton, nut, and white carton move by about 0.03 or disagree in sign.
- The same fine-tuning lifts pooled position R² by only a few hundredths. Both seeds agree on that small lift for `push_x`, `push_lo`, and `push_hi`. The level stays near zero, and the within-geometry gains are not shared across geometries.

Changing the frozen readout improves the aggregate predictions, so pooling contributes to the gap. The 4×4 readout does not close it. This does not rule out further spatial loss. Neither the readout comparison nor the fine-tuning comparison isolates the effect of a different training objective. The evidence supports a representation gap: image position predicts the labels, and a linear probe of these DINOv2 features keeps only part of that information. Two hypotheses are consistent with the gap and were not tested: a mask-aligned or finer readout could still find the centroid in the patch tokens, and an objective aimed at image motion could teach that centroid more directly than trajectory-similarity SoftInfoNCE.

**Added limitations.** White carton has six configurations. Two- and three-intervention means use the representative subsets listed above. Within-geometry position R² is undefined on constant toss channels and is unstable at n = 6. The 4×4 grid is a linear average of patch tokens, not a learned spatial head.

**Paired fold audit (Nov 6).** From the unrounded LOGO fold nMAEs in [diagnostic.json](results/phino_action_learning_v2/spatial_diagnostic/diagnostic.json), [both-seed counts](results/phino_action_learning_v2/spatial_diagnostic/paired_fold_counts.csv) compare each clip-fine-tuned 4×4 seed against frozen 4×4 for the same target, subset design and held-out geometry. At 1 / 2 / 3 / 4 interventions both seeds lower mass error on 3 / 3 / 4 / 4 of five geometries; friction on 1 / 2 / 5 / 2. Counts are descriptive, not independent replications or statistical tests. In particular the three-action result is for one specific subset. No new probe fit or change to the Brown outputs was required.

## 2026-11-06: Phase 1 objective baselines

**Question.** On the same action-conditioned clips, does trajectory-similarity supervision have a consistent advantage over generic instance-discrimination InfoNCE, and how does either compare with fine-tuning that directly predicts log mass and log friction?

**Hypothesis.** If the earlier gain over frozen DINOv2 came from the trajectory target, trajectory similarity should beat a matched InfoNCE run that sees the same two augmented views and no trajectory target. Direct supervision is a different comparison: it is allowed to see the probe's label family on the training geometries, so it can beat both contrastive methods without that being evidence for a self-supervised objective. This does not reopen the Stage 3 gate and does not justify a new physics grid.

**Data and code.**
- Observations: the existing `data/phino_simpact_v3obs_v2/` clips. No new physics. The probe set is the same 292 configurations with all four actions.
- Trajectory-similarity checkpoints and their pooled and 4×4 probes were reused from `results/phino_action_learning_v2/`. They were not retrained.
- New runs: `scripts/phino_action_contrastive_train.py --objective instance` and `--objective regression`, launched by `scripts/run_phino_action_baselines.sh`. Outputs: `results/phino_action_baselines_v1/` (Brown only). Committed tables: [`docs/results/phino_action_baselines_v1/`](./results/phino_action_baselines_v1/).
- Twenty new runs, all 400 steps, training time 122 minutes in total (4.1–9.9 minutes each). Losses decreased on every run and stayed finite. InfoNCE final losses were 0.26–0.81. Regression final mean squared error on the standardized labels was 0.24–0.49.

**Protocol.** Matched to the trajectory run: DINOv2-B/14, last 4 blocks trainable, batch of 24 clips, one intervention per step, backbone learning rate 1e-5, head learning rate 1e-3, weight decay 0.05, 40-step warmup, two augmented views with shared colour jitter and translation and no flips. Seeds 0 and 1. Each fold excludes its test geometry from training rows, from the trajectory normalizer (InfoNCE does not fit one), and from the regression label standardizer. No hyperparameter was chosen on a test geometry. The ridge probe is unchanged, including its training-only grouped penalty selection.
- **InfoNCE.** Symmetric cross-entropy. The positive pair is the two views of one clip. Other clips in the batch are negatives. Temperature is the same learnable parameter as SoftInfoNCE, initialized at 0.07. The loss function does not accept a similarity matrix.
- **Regression.** Mean squared error on log mass and log friction, standardized by the mean and population standard deviation of the unique training configurations. The head has the contrastive head's hidden width, is not L2-normalized, and is discarded before the probe.
- **Evaluation.** Frozen pooled features and a 4×4 average of patch tokens, then the same standardized dual ridge. The primary endpoint is leave-one-geometry-out nMAE at all four interventions. That is one intervention set, so the paired differences do not average dependent subsets. Scaling curves still average subsets of a given size, and those averages are descriptive only. For the 4×4 readout, counts 2 and 3 use the published representative subsets, not every combination. The image-position tracker was recomputed in the new probe files and matched the previous files.
- **Paired summary.** For each held-out geometry, average the two seeds, then report the mean, the range across the five geometries, how many geometries have both seeds in the same direction, and the mean absolute seed difference. Five geometries are the unit. Subset counts are not.

**Results.** Sample-weighted leave-one-geometry-out nMAE at four interventions (mass / friction; learned methods are seed 0 / seed 1):

| Representation | Pooled | 4×4 |
|---|---|---|
| Frozen DINOv2 | 0.76 / 0.86 | 0.67 / 0.59 |
| Trajectory similarity | 0.61 / 0.60, 0.60 / 0.61 | 0.55 / 0.50, 0.51 / 0.48 |
| Generic InfoNCE | 0.65 / 0.63, 0.59 / 0.69 | 0.57 / 0.50, 0.53 / 0.53 |
| Supervised regression | 0.58 / 0.61, 0.51 / 0.59 | 0.51 / 0.54, 0.45 / 0.49 |
| Image-position tracker | 0.04 / 0.12 | 0.04 / 0.12 |

Giving each geometry one vote, instead of weighting by its number of configurations, the pooled four-intervention nMAE is frozen 0.78 / 0.86, trajectory 0.60 / 0.64, InfoNCE 0.60 / 0.64, and regression 0.61 / 0.59. The regression mass advantage in the sample-weighted row is not present in this unweighted average.

Per held-out geometry, pooled, four interventions. Learned entries are seed 0 / seed 1.

| Geometry | n | Frozen mass | Trajectory mass | InfoNCE mass | Regression mass | Frozen friction | Trajectory friction | InfoNCE friction | Regression friction |
|---|---|---|---|---|---|---|---|---|---|
| Blue carton | 46 | 0.57 | 0.50 / 0.50 | 0.39 / 0.48 | 0.36 / 0.40 | 0.77 | 0.56 / 0.51 | 0.53 / 0.57 | 0.47 / 0.54 |
| Cow | 85 | 0.75 | 0.68 / 0.66 | 0.74 / 0.64 | 0.34 / 0.26 | 0.96 | 0.78 / 0.76 | 0.84 / 1.01 | 0.77 / 0.72 |
| Cube | 85 | 1.00 | 0.63 / 0.65 | 0.75 / 0.77 | 0.95 / 0.87 | 0.78 | 0.46 / 0.44 | 0.42 / 0.45 | 0.52 / 0.46 |
| Nut | 70 | 0.60 | 0.56 / 0.53 | 0.58 / 0.40 | 0.52 / 0.41 | 0.88 | 0.56 / 0.63 | 0.66 / 0.68 | 0.62 / 0.63 |
| White carton | 6 | 0.96 | 0.61 / 0.63 | 0.63 / 0.59 | 1.01 / 0.93 | 0.90 | 0.80 / 0.89 | 0.76 / 0.52 | 0.57 / 0.65 |

Trajectory minus InfoNCE, geometry-level, four interventions (negative means trajectory has lower error):

| Readout | Target | Mean delta | Geometry range | Both seeds favor trajectory | Seeds disagree in sign | Mean \|seed difference\| |
|---|---|---|---|---|---|---|
| Pooled | mass | −0.001 | −0.119 to +0.070 | 1/5 | 3/5 | 0.079 |
| Pooled | friction | −0.005 | −0.156 to +0.207 | 2/5 | 2/5 | 0.138 |
| 4×4 | mass | −0.007 | −0.065 to +0.086 | 1/5 | 3/5 | 0.085 |
| 4×4 | friction | −0.013 | −0.087 to +0.075 | 2/5 | 2/5 | 0.075 |

The mean differences are smaller than the seed scatter. Trajectory minus frozen remains consistently negative on the pooled probe (both seeds, 5/5 geometries, both targets). That repeats the earlier pilot; it is not new evidence for the trajectory target, because InfoNCE produces the same geometry-equal error.

Regression versus the contrastive runs is a geometry split, not a uniform ranking. Both regression seeds lower cow pooled mass from about 0.67 to about 0.30, and both leave cube mass near the frozen value (0.95 / 0.87 against frozen 1.00) while trajectory reaches 0.63 / 0.65. White-carton mass goes the same way: regression 1.01 / 0.93 against frozen 0.96, trajectory 0.61 / 0.63. The paired trajectory-minus-regression mass delta therefore ranges from −0.35 to +0.37 across geometries. Its mean is about zero. Friction differences are smaller than, or comparable to, the seed gap.

Within a geometry, four interventions, pooled sample-weighted nMAE is frozen 0.35 / 0.69, trajectory about 0.33 / 0.62, InfoNCE about 0.32 / 0.59, and regression 0.31 / 0.54. The tracker is 0.08 / 0.19. The ordering is the same and the gaps are small.

Scaling curves from one to four interventions do not create a trajectory lead that is absent at four. On pooled features the three learned methods stay within about 0.08 nMAE of each other at every count. Position-decoding R² stays near zero for all three. Pooled `push_x` is negative (trajectory about −0.33, InfoNCE about −0.53 to −0.76, regression about −0.44 to −0.69). The best cell is still about 0.23.

**Limitations.**
- Five geometries, and white carton has six configurations. Two seeds. A mean delta of 0.01 cannot be separated from seed noise.
- Intervention subsets share configurations. The scaling means are descriptions. The paired counts above use geometries, not subsets.
- Regression sees log mass and log friction during training. The contrastive runs do not. Each method is still trained on one intervention at a time, so regression is asked to predict both properties from actions that do not identify both, including friction from the toss.
- The evaluated predictor is a linear probe of backbone features. The training heads are discarded. Same budget and augmentation do not make the regression loss scale identical to InfoNCE, and the learning rates were not retuned.
- The 4×4 curves at two and three interventions use the published representative subsets. The four-intervention comparison does not.
- Sample-weighted nMAE is dominated by cow, cube, and nut. The cow regression result moves that average without moving every geometry.
- Labels are synthetic simulator assignments. Renders are noiseless. The Stage 3 gate is still failed. These actions are the post-hoc fallback set.

**Conclusion.** Trajectory-similarity supervision does not have a consistent advantage over generic contrastive learning on this probe. The two contrastive objectives match once each held-out geometry is counted once, and the seeds do not agree on a winner. Direct physical-property supervision is better on some geometries, especially cow mass, and worse on others, especially cube and white-carton mass. It does not dominate the contrastive runs, and none of the three approaches approaches the image-position tracker. The earlier improvement over frozen DINOv2 should not be credited to the trajectory target. Do not start a Phase 2 objective or a new physics grid on the strength of this comparison.

## 2026-11-06: Elastic reciprocity feasibility

**Question.** On a clamped flexible part, do small-force probes at two scoring ports give Maxwell–Betti labels that are accurate enough, and useful enough, to choose a first contact action on an unfamiliar geometry? Is that advantage still there once a spring model can see the geometry?

**Hypothesis.** In the small-load regime the measured secant is symmetric to within the solver's uncertainty, and loading the scoring ports predicts the alignment and protection response of candidate actions that were not executed. The hypothesis under test is the intervention strategy, not a learned encoder. A deployable spring model is the baseline that can reject the strategy.

**Data and code.**
- New fixture. No use of `results/phino_mesh_grid_cutoff_1m_v1/` or the Phase 1 checkpoints.
- `scripts/phino_reciprocity_feasibility.py`, tests in `tests/test_phino_reciprocity.py` (10 passed).
- Command: `PYTHONPATH=scripts .venv/bin/python scripts/phino_reciprocity_feasibility.py`.
- Local run: `results/phino_reciprocity_feasibility_v1/` (includes model XML and compliance matrices). Committed tables and figures: [`docs/results/phino_reciprocity_feasibility_v1/`](results/phino_reciprocity_feasibility_v1/). Full write-up: [`docs/phino_reciprocity_feasibility.md`](phino_reciprocity_feasibility.md).

**Protocol.** Four planar spring assemblies (straight, 60° branch, L, crank), one shared goal point, and a utility fixed before the MuJoCo ranking. Candidate set: 8 ports × 4 directions × 2 forces. Acceptance of a reciprocal pair: displacement mismatch at most max(25 µm, 2% of the larger block). A clear action gap was pre-declared at 0.05 utility. The deployable baseline sees the drawing and the declared 4.0 N·m/rad rate. The oracle reads MuJoCo joint stiffness. Neither number was fit on the measured ranking.

**Results.**
- Settled repeats differ by 0. Solver settings that reach equilibrium move the tip reading by ≤ 0.35 µm. A 0.4 s settle does not equilibrate and shifts it by 57 µm.
- Worst reciprocal mismatch at 0.05 N: 55, 19, 40, and 14 µm (1.1% or less). The secant matches the independent linearization to 0.42–0.97%.
- Reversing 0.05 N flips the transverse displacement with error 0. A 3.0 N load is 113 mm away from the linear extrapolation. A stop 2.5 mm above the protected port leaves the 0.05 N reciprocal gap at 0.15 µm and raises the 0.15 N gap to 1.62 mm.
- Measured optima: straight `a7 +y 0.05 N` (utility 0.861, gap 0.167); branch `a4 +y 0.15 N` (0.550, gap 0.049); L `a5 +x 0.15 N` (0.777, gap 0.093); crank `a7 +x 0.15 N` (0.802, gap 0.043). The straight-beam action scores 0.096, 0, and 0 on the other three. `a0 +y 0.05 N` scores 0 everywhere.
- Deployable mechanics and the stiffness oracle both have regret 0. Doubling the assumed rate, which is not the baseline, has regret 0.05–0.37.
- Four scoring-port probes recover the executed optimum (regret 0). Same-force error is at most 65 µm. Scaled to 0.15 N the error reaches 376 µm. Action–action blocks stay unidentified (21 or 28 pairs). The contact and 3 N responses are not identified.

**Limitations.**
- Lumped planar springs, gravity off, one declared rate, four geometries, one utility.
- Branched and crank runner-up gaps are just under the 0.05 line.
- The probes' success is a direct calculator on four labels, not a representation result.

**Conclusion.** Reciprocity holds for this fixture at 0.05 N and fails at a contact stop and at 3 N, which is the expected content of the theorem rather than a new finding. Visible geometry does change the best first action, and a fixed action does not transfer. The deployable spring model already selects that action, so there is no equal-cost reason to collect reciprocal probes instead. **NO-GO** for a Phase 2 learning experiment on this intervention. Do not hide a spring rate in order to manufacture a residual.

## 2026-11-07: AnyGoal contact-action diagnostic

**Question.** On the released Extract-Peg task, does the frozen Sensorimotor World Model select inferior contact sequences because its predicted consequences are wrong, once the candidate actions are held fixed and prediction is separated from goal scoring?

**Hypothesis H, pre-registered in the Nov 7 memorandum.** For a substantial fraction of test states whose goal is attainable in a common candidate bank, SMWM selects a failing sequence while scoring the actual resulting images with the same latent goal cost selects a successful sequence. If H fails, this test gives no evidence for a predictive-learning bottleneck.

**Data and code.**
- Protocol: `docs/research/PHINO-Problem-First-Memorandum-2026-11-07.md`, Section 8. No new training.
- AnyGoal checkout `f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e`, isolated at `/cs/local/nbu/people/ashah264/anygoal-audit`. SMWM source `9d22bcc15c6ad8af8cc93337a950df2b49a866b9`.
- Hugging Face model revision `b27f9dac9c750410d1331d258a95eff0d56570c2` and dataset revision `1719e7dab2fbeb07a0d5c3cf9cd881bc49510c88`.
- Checkpoint `sensorimotor_insertpeg.ckpt`, sha256 `14ef756d0a0ea701079d6629208950f003d96b2a9dba9dc0b9cde63f1d301f3b`, matching the release sidecar. State checkpoints `toy-adaln_insertpeg.pt` (`929d8bc9…`) and `toy-action-duplicate_insertpeg.pt` (`b39d1202…`) likewise match their sidecars. Both state configs are 33-dimensional with `include_velocities: false`.
- Files: training `manipulator_insert_peg_weak_expert_train.h5` sha256 `7e676e32…d98d` (40,032,526,769 bytes); `avoid_peg_test.h5` sha256 `83c83532…2644`; `extract_peg_test.h5` sha256 `afc53cff…32a1`.
- Runtime: Python 3.10.22, torch 2.9.1+cu128, MuJoCo 3.8.1, stable-worldmodel 0.1.1, RTX 5060 Ti. The editable install used the CUDA 12.8 index with `unsafe-best-match` because a plain extra index made the pinned `tqdm` unsatisfiable. Direct pins match the repository. Some transitive versions differ from `requirements.txt`; that difference is recorded, not hidden. A one-sample `get_cost` call on the loaded SMWM returned a finite `(1, samples)` cost, so no criterion patch was applied.
- Audit code: commit `4c66439`, `scripts/anygoal_contact_diagnostic.py`, `scripts/anygoal_contact_download.py`, `scripts/run_anygoal_contact_audit.sh`. Twelve unit tests in `tests/test_anygoal_contact_diagnostic.py`.
- Primary metric: the dataset-goal criterion (hand and peg pose within 0.04 m / 0.10 rad). Native Insert-Peg reward is reported separately. Insert-Peg planning reads the training file; the memo's label is "base-task reference on training source."

**Stage A.** Native `anygoal eval planning`, SMWM, seed 0, 50 evaluations. Smoke runs of 2 episodes were not used as results. Wall time was 76 s, 89 s, and 102 s.

| Dataset | Dataset-goal | Paper SMWM | Native insert reward | Final object pose | Recorded 25-step replay on these starts | Train-hash overlap |
|---|---:|---:|---:|---:|---:|---:|
| Insert-Peg | 76% | 70% | 6% | 80% | 88% | 50/50 |
| Avoid-Peg | 78% | 74% | 2% | 98% | 26% | 0/50 |
| Extract-Peg | 28% | 24% | 28% | 30% | 0% | 0/50 |

The gaps from the paper are +6, +4, and +4 percentage points. None crosses the 10-point audit trigger. The sampler is the released `sample_start_indices`, including its `choice(len(valid_indices) - 1)` exclusion of the last valid row. SMWM action and observation scalers are fit on `manipulator_insert_peg_weak_expert_train`, including when the evaluation file is Avoid or Extract.

Because Extract-Peg stayed near the reported failure, the released 33-d state predictors were run on the same seed and evaluation count. A 44-d vector was not passed to them.

| Model | Insert-Peg | Avoid-Peg | Extract-Peg |
|---|---:|---:|---:|
| SMWM | 76% | 78% | 28% |
| AdaLN, 33-d, no velocities | 72% | 94% | 22% |
| Action-duplicate, 33-d, no velocities | 76% | 86% | 28% |

**Stage B.** The pre-registered bank is the first 50 Extract-Peg episodes in episode-id order, restored at index 10, with a 25-step continuation. All 400 episodes have length 199, so none were excluded for length. Before any latent score, the recorded continuation was replayed with `InsertPegEnv.step`, which is the planner's integrator.

| Replay set | Full dataset goal | Object pose only | Median hand position error | Median object position error |
|---|---:|---:|---:|---:|
| Extract-Peg, index 10, first 50 | 6% | 60% | 76 mm | 11 mm |
| Avoid-Peg, same rule | 0% | 82% | 164 mm | ~0 mm |
| Insert-Peg training file, same rule | 78% | 78% | 0.8 mm | 1.0 mm |

Pose restoration before the step matches the stored arm, hand, object, and target. Touch sensors are outputs and are not restored. One training-file transition from the episode start matches the next stored pose to about 0.3 mm. The same player does not match Extract-Peg at any physics substep count from 1 to 40. Clearing the solver warm start inside this environment changes a continued rollout by less than 1e-7, so warm-start loss does not explain the Extract-Peg gap. The Extract-Peg collector in the pinned source advances 25 MuJoCo substeps per action, while `env.step` advances one 10 ms control interval; replaying with 25 substeps also misses the stored goals.

The 90% replay gate fails. Those rates are the state after all 25 actions. Counting a hit at any intermediate step, which is what the evaluator's ever-success flag does, raises Extract-Peg only to 8/50 at index 10 and 7/50 on the Stage A starts. Predicted-latent cost, observed-latent cost, opportunity regret, and ranking agreement were not computed. Seeds 1 and 2 and the Section 8.6 controls were not run: the memorandum expands that budget only if the action-bank diagnostic still shows a prediction failure.

**Limitations.**
- One SMWM checkpoint and one seed. Fifty episodes.
- Insert-Peg is not a held-out split.
- The Stage A starts and the index-10 bank are different populations. Both show the Extract-Peg replay failure.
- Final object success is the terminal pose, not an ever-achieved object-only flag, except where the replay audit recorded ever-style object checks at the final frame only.
- State predictors use history length 3 and a resolved `max_episode_steps` of 100; SMWM uses history 1 and 50. Both evaluations stop at the shared budget of 50 steps.

**Conclusion.** The published SMWM numbers reproduce within 6 percentage points, and the Extract-Peg drop relative to Insert-Peg remains for SMWM and for both released state predictors. That does not identify a predictive-model failure. On the episodes being scored, the recorded Extract-Peg actions are at the dataset goal after 25 steps 0 times out of 50, and at some intermediate step 7 times out of 50. The same evaluator replays Insert-Peg actions to the 25-step goal 88% of the time. Hypothesis H was not tested. **NO-GO** for treating this gap as evidence that a new predictive objective is required. The next question is which integrator and action interface generated the released Avoid-Peg and Extract-Peg files, and whether the planner is executing that same transition. A fixed action bank is informative only after those two agree.

## 2026-11-07: Extract-Peg replay correction and the fixed action bank

**Question.** Which integrator and hidden state generated the released Extract-Peg and Avoid-Peg files, and once trajectories are reproduced, does SMWM's predicted latent cost select worse actions than the same cost on the real future?

**What the files are.** `extract_peg_test.h5` attributes record `task=manipulator_extract_peg`, seed 20260926, a 40% prefix trim, and `derived_from` a v2 expert file. The trim script `scripts/data/trim_extract_peg_v3.py` (present before AnyGoal commit `3dfc844`, absent from the pinned `f0cf8c2` tree) copies states and actions; it does not re-simulate. Every episode has 199 transitions, which is `331 - floor(0.4 * 331)`. The current collector uses the same 331-step script, `ExtractPegEnv`, a hand-peg weld, and `physics.step(nstep=25)`. `InsertPegEnv.step` advances one 10 ms control interval, 10 substeps, and has no weld. Avoid-Peg uses the same 25-substep integrator on `InsertPegEnv`. Its slot-collision edits are not required to replay the tested episodes.

**Trajectory reproduction, separate from goal success.** Re-locking the weld at the first stored frame and integrating 25 substeps per action:

| Check | Result |
|---|---|
| One-step median hand / object error | 0.26 mm / 0.23 mm |
| Error after 25 actions, median hand / object | 1.39 mm / 0.68 mm |
| Recorded 25-step state inside the goal tolerances | 47/50 (94%) |
| Same replay from index 10, where the original weld anchor is 12 steps stale | 25-step goal rate 50%; one-step median hand 1.5 mm |

The published `InsertPegEnv.step` replay remains 0/50 at the final step on the Stage A starts. Avoid-Peg under 25 substeps, on 20 episodes, has a maximum goal-component error of 0.0036; `env.step` leaves a median hand error of 74 mm. Insert-Peg remains the positive control for `env.step` (88% of Stage A starts). The index-10 preregistered bank is not used for model scores: the weld anchor is not in the 44-d state, and the first stored frame is where re-locking it matches the generator. That revision is separate from the published evaluation, which was not changed.

**Corrected Stage B.** Fifty episodes, start at the first stored frame, bank seed 0, 16 candidates, SMWM `sensorimotor_insertpeg.ckpt`. Native `get_cost`. True utility is the dataset-goal tolerance on the corrected rollout.

| Quantity | Mean | 95% episode bootstrap interval |
|---|---:|---:|
| Recorded continuation reaches the 25-step goal | 0.94 | |
| Bank solvability | 0.98 | |
| P opportunity regret | 0.70 | [0.58, 0.82] |
| O opportunity regret | 0.06 | [0.00, 0.14] |
| P minus O | 0.64 | [0.50, 0.76] |
| H: P fails and O succeeds | 0.64 | [0.50, 0.78] |
| P and O both fail while the bank succeeds | 0.06 | [0.00, 0.14] |

Argmin agreement is 0.08. Mean Spearman correlation of the two costs is 0.24. The Section 8.8 numerical gates that this bank can answer are met.

**Stage C.** In the same corrected dynamics, published CEM (300 samples, 30 iterations, horizon 5) executed open-loop for 25 actions succeeds on 0.48 [0.34, 0.62] of the 50 episodes. Replanning every 5 raw actions, with the unused plan tail as the warm start, succeeds on 0.46 [0.32, 0.60]. Query counts are 1,500 and 7,500. Replanning does not remove the failure. The 50 start states, in the 33-d essential coordinates and z-scored against an 8,000-frame training sample, are all inside that sample's 95th-percentile neighbor distance (median distance 2.45 versus threshold 4.07).

**Limitations.** The weld anchor is reconstructed, not read from the file. Full-episode open-loop error still has a heavy tail, so the claim is for the 25-step horizon, not for all 199 steps. The fixed bank and the CEM control are different action sets. State predictors were not rerun inside `ExtractPegEnv`. Coverage is about the start pose, not the welded future.

**Conclusion.** Hypothesis H holds for this corrected bank: the latent goal cost ranks the real futures, and the model's predicted futures do not. Shorter replanning does not fix that. **GO only to the memorandum's standard-baseline comparison** (a contact or mechanics model that is allowed to know the weld and the 25-substep action, plus a direct goal-conditioned baseline on the same images). **Not a GO for a new representation objective.** The published 28% Extract-Peg number remains a score under `InsertPegEnv.step`, which is not the data-generating transition.

## 2026-11-08: Dynamics isolation and standard baselines

**Question.** How much of the corrected Extract-Peg ranking failure is the weld, the 25-substep action, or the goal, and does a true-state mechanics model or a small goal-conditioned method already rank the same candidates when the transition is one the training data support?

**Hypothesis.** If the failure is the unsupported transition, predicted regret stays high only under the weld and 25 substeps, and it drops when the same goal and bank are executed with the training integrator and the bank still contains a success. If predicted regret stays high on Insert-Peg goals under that integrator, the weld does not explain the ranking failure. A true-state simulator is the mechanics ceiling. An image method counts as a competent equal-information baseline only if it beats uniform selection, or, for a pose estimator, if held-out median hand and object error are at most 2 cm.

**Data and code.**
- Same pinned AnyGoal checkout, checkpoint, and 50 Extract-Peg episodes as the corrected audit. Parent revision `42c8f64`.
- `scripts/anygoal_dynamics_factor.py`, `scripts/anygoal_standard_baselines.py`, `scripts/run_anygoal_dynamics_factor.sh`. Tests in `tests/test_anygoal_dynamics_factor.py`.
- Commands, from the SIMPACT root, with the AnyGoal 3.10 environment: prepare and rollout the six integrators; score the four primary ones on GPU; summarize to `docs/results/anygoal_dynamics_factor_v1/`. The insert anchor is the first 50 training episodes at step 0, with train-chunk candidates drawn from the other episodes. Retrieval, the 2000-step models, and the one-block probe write to `results/anygoal_insert_baselines_v1/` locally.
- Physics timestep 0.001 s, so 10 substeps are 10 ms and 25 substeps are 25 ms. Two L40S GPUs. Weld/25 recorded replay reproduced the corrected audit before the full bank: goal rate 0.94, median terminal hand/object error 1.39 mm / 0.68 mm. Recomputed predicted costs agree with the corrected bank on every selected action (maximum absolute cost difference 0.037).

**Protocol.** The Extract-Peg comparison holds the checkpoint, the seed-0 bank, the initial state and image, and the recorded 25-step state and image fixed. Integrators are Insert-Peg `env.step` (no weld, 10 substeps), Insert-Peg physics for 25 substeps, Extract-Peg with the grasp weld for 10 substeps, and Extract-Peg with the weld for 25 substeps. The extract model's slot-contact edits were also run with the weld off. Opportunity regret is compared across integrators only when solvability is at least 0.80. The insert anchor uses the training integrator and each episode's own 25-step goal. The one-block probe uses five actions and takes its goal to be the recorded action's own image at that time. The pose network and the goal-conditioned regression see episodes 50–129 only, 16 windows per episode, stride 10, 2000 Adam steps, batch 64, learning rate 1e-3. Episodes 130–139 are the pose check and are not training rows.

**Results.** Slot edits do not move the state. The no-weld extract rollouts match the Insert-Peg rollouts exactly, including pixels. Predicted costs are identical across all four integrators because they depend only on the start image, the goal image, and the actions.

| Integrator | Solvability | Recorded goal | Predicted success | Observed success | Latent SSE | Median recorded object error |
|---|---:|---:|---:|---:|---:|---:|
| No weld, 10 substeps | 0.10 | 0.02 | 0.02 | 0.08 | 67.5 | 34 mm |
| No weld, 25 substeps | 0.02 | 0.00 | 0.00 | 0.02 | 64.9 | 34 mm |
| Weld, 10 substeps | 0.30 | 0.16 | 0.08 | 0.16 | 68.1 | 22 mm |
| Weld, 25 substeps | 0.98 | 0.94 | 0.28 | 0.92 | 68.0 | 0.7 mm |

On the weld/25 bank, unconditional predicted regret remains 0.70 [0.58, 0.82] and observed regret 0.06 [0.00, 0.14]. Conditional on the 49 solvable episodes, predicted regret is 0.71 [0.59, 0.84]. The other three integrators are below the 0.80 solvability floor, so their small unconditional regrets are not ranking results. The paired latent-error difference, weld/25 minus the training integrator, is 0.52 [-0.40, 1.42]. Zeroing arm and object velocity at the start of the weld/25 rollout leaves solvability at 0.98 and the recorded-action goal rate at 0.92. Every episode begins inside the native insertion criterion. The recorded 25-step goal is inside it on 0.28 of episodes.

Under the weld and 25 substeps, a second goal was the farthest realized terminal outside the dataset tolerance of the recorded terminal. Both goals are achieved by a bank action. Predicted regret is 0.80 [0.68, 0.90] for the recorded outcome and 0.94 [0.86, 1.00] for the alternate. Observed regret is 0.00 for both. The argmin changes on 0.58 [0.44, 0.72] of episodes. Median latent distance between the two goal images is 44. Insert, Avoid, and Extract dataset goals were not swapped: they were not generated under one integrator.

Insert anchor, training integrator, own 25-step goals, solvability 0.84, recorded-action success 0.82:

| Ranker | Success | Opportunity regret |
|---|---:|---:|
| True state and Insert-Peg physics | 0.84 | 0.00 |
| Observed latent cost | 0.78 | 0.06 [0.00, 0.14] |
| Uniform candidate | 0.37 | 0.48 |
| SMWM predicted cost | 0.20 | 0.64 [0.50, 0.76] |
| Goal-conditioned regression | 0.28 | 0.56 [0.42, 0.70] |
| Image retrieval | 0.18 | 0.66 [0.52, 0.78] |
| Pose network, estimated goal, then physics | 0.30 | 0.54 [0.40, 0.68] |

The true-state simulator's regret is 0 because it selects a successful candidate whenever one exists. Its success equals solvability. The retrieval neighbor is 36 cm away in hand position and 39 cm in object position, and its action is far from the recorded action (median L2 11.3). Held-out pose error is 6.9 cm at the hand and 19.5 cm at the object, against the 2 cm bar declared before the anchor simulation. On the anchor starts the same network is at 17 cm and 40 cm. The regression success of 0.28 is below uniform selection at 0.37. Those three image methods are therefore not competent baselines. Figure: [`insert_anchor_baselines.png`](results/anygoal_dynamics_factor_v1/insert_anchor_baselines.png).

One model block on the same anchor, with the goal defined as the recorded action's image after five actions: solvability 1.00, mean 8.4 candidates inside the tolerance, chance success 0.53. Predicted regret is 0.78 [0.66, 0.88] and success is 0.22. Observed regret is 0.00. One-block latent SSE is 47, against 52 for the full 25-step prediction on this anchor.

**Limitations.**
- One SMWM checkpoint. The insert anchor episodes are inside its training file; the new image methods were not trained on them.
- The recorded continuation is a privileged member of the bank. That makes the bank easier for a method that can propose the expert action.
- The 2000-step CNN and the 80-episode pool were a declared small budget. Their failure is a budget and coverage result.
- The one-block goal is the recorded outcome itself, so observed regret of 0 is the cost recognizing that image. The predicted selector still lands below chance.
- The published Insert-Peg planner result, 76% at seed 0, uses CEM and the evaluator's sampled starts. It is not this bank.
- Short-horizon and 25-step regrets answer different goals. Both are reported separately.

**Conclusion.** Both the weld and the 25 ms action are required before this bank can reach the recorded Extract-Peg pose. The training integrator does not make that pose solvable, and the model's latent error does not improve reliably when that integrator is restored. The ranking failure itself is already present for Insert-Peg goals under the training integrator, at 25 steps and at one model block, while a true-state simulator ranks the 25-step bank with regret 0. The image baselines trained here do not meet the competence bar, so they do not show that a fair standard method fails. **NO-GO for a new learning objective.** The single next measurement is an image-to-state estimator that first reaches 2 cm held-out error on insert episodes outside the anchor, then ranks this same bank. A regret interval below 0.10 stops the diagnosis.

## 2026-11-08: Supervised visual state estimation

**Question.** Can a conventional image-to-state estimator, trained with simulator-state labels, plus Insert-Peg physics, remove SMWM's ranking failure on the 50-episode insert bank?

**Hypothesis.** If the failure is only that SMWM does not recover a state the simulator can roll forward, an estimator with at most 2 cm median held-out hand and object error will rank the same bank at regret 0.10 or below. If the estimator never reaches that accuracy, the physics baseline is untested. State labels are extra supervision that SMWM does not get.

**Data and code.**
- Parent revision `a97b5c5`. Scripts: `scripts/anygoal_state_estimator.py`. Tests: `tests/test_anygoal_state_estimator.py`.
- Train episodes 50–1299. Validation episodes 1300–1460. Evaluation episodes 0–49, used only for the final ranking. Normalization is fit on the training episodes.
- Encoder: local DINOv2-B (`dinov2_vitb14_pretrain.pth`), then an MLP on the class token concatenated with the mean patch token. Head-only training, then the last four blocks, then all blocks. About 0.4 GPU-hour on one L40.
- The first loss regressed raw quaternion components and averaged every dimension together. About 10% of nearby frames flip the quaternion sign, and that run left object angle at 1.17 rad. The reported model regresses sin and cos of the planar angle and averages the loss within each state group.
- Commands write caches and checkpoints under `results/anygoal_state_estimator_v1/` locally. The committed summary is [`docs/results/anygoal_state_estimator_v1/`](results/anygoal_state_estimator_v1/).

**Protocol.** Validation reports median and 90th percentile position, angle, joint, and velocity error after `set_state`, which recomputes the hand from the arm. A frame is counted as physically usable when both hand and object errors are under 5 cm. The bank ranking initializes Insert-Peg from the estimated start, compares the simulated terminal with the goal, and scores the chosen action by the saved true-start outcome. The estimated goal's hand pose is filled by forward kinematics; `set_state` does not read the hand coordinates. Model selection uses validation only.

**Results.** Mixed-frame validation, 1,282 frames:

| Quantity | Median | 90th percentile |
|---|---:|---:|
| Hand position | 4.9 cm | 16.1 cm |
| Object position | 4.5 cm | 10.6 cm |
| Target position | 2.8 cm | 6.0 cm |
| Object angle | 0.096 rad | 0.29 rad |
| Worst joint | 0.39 rad | 1.35 rad |
| Both positions under 5 cm | 31% | |

Episode-start validation, 161 frames, which is the anchor's phase:

| Quantity | Median | 90th percentile |
|---|---:|---:|
| Hand position | 11.0 cm | 22.9 cm |
| Object position | 6.7 cm | 13.3 cm |
| Target position | 3.4 cm | 7.7 cm |
| Object angle | 0.13 rad | 0.33 rad |
| Worst joint | 0.91 rad | 2.07 rad |
| Both positions under 5 cm | 7% | |

The untouched anchor starts agree with that second row: hand 9.7 cm, object 6.4 cm, object angle 0.12 rad. Peg orientation is recovered. The initial arm pose is not.

Anchor ranking. Success is the true-start outcome.

| Ranker | Success | Opportunity regret |
|---|---:|---:|
| True state | 0.84 | 0.00 [0.00, 0.00] |
| True pose, object velocity zero | 0.76 | 0.08 [0.02, 0.16] |
| True arm, rest estimated | 0.72 | 0.12 [0.04, 0.22] |
| Estimated start and goal | 0.36 | 0.48 [0.34, 0.62] |
| Estimated start, true goal | 0.30 | 0.54 [0.40, 0.68] |
| SMWM predicted cost | 0.20 | 0.64 [0.50, 0.76] |

Restoring only the estimated object pose, object velocity, or slot target leaves regret at 0.46, 0.48, and 0.54. Restoring the arm is the substitution that moves the result. Anchor arm velocity is already zero. Gaussian noise on the true object's position, with the rest of the state true, gives regret 0.08, 0.10, 0.04, and 0.06 at 0.5, 1, 2, and 5 cm. A 2 cm object error would not have created SMWM's gap. The missing piece is the arm at the initial pose.

**Limitations.**
- One encoder family and one small training schedule. The episode-start finetune used 1,250 images and was still improving hand error at epoch 12, from 15.7 cm to 11.0 cm, while object error stayed near 6.7 cm.
- Velocity at a single initial image is not a measurement. At these starts it is a small ranking effect.
- The 0.48 regret uses an estimator that missed the 2 cm bar. It is reported so the arm diagnosis is concrete. It is not the pre-registered test of known physics.
- State labels are additional supervision. A success would not have meant SMWM had the same training signal.

**Conclusion.** **INCONCLUSIVE.** The estimator does not reach 2 cm median hand and object error on the episode-start distribution the anchor uses, so estimated-state physics remains untested at the required accuracy. The error that dominates the descriptive ranking is the arm, not object position, object velocity, or the slot target. The true-state simulator still has regret 0. Do not treat this as evidence that a competent state estimate fails, and do not open a new learning objective. The next measurement is the same ranking after episode-start validation is actually inside 2 cm.

## 2026-11-08: Proprioceptive arm plus physics

**Question.** Are joint readings available to a conventional controller, and does using them, with vision only for the remaining state, remove SMWM's ranking failure?

**Hypothesis.** If the robot exposes its encoders, the current arm does not have to be inferred from pixels. Regret should fall to about 0.10 when those readings initialize Insert-Peg physics and the object and goal are no worse than the errors already shown to be tolerable. If the goal image still cannot supply the goal pose, that remaining error is an inaccurate estimate, not a failure of physics given the true pose.

**Data and code.**
- Same 50-episode insert anchor, bank, and true-start outcomes as `a97b5c5` / `c0db08d`.
- `scripts/anygoal_proprio_baseline.py` and `tests/test_anygoal_proprio_baseline.py`.
- Current object estimate: the episode-start DINOv2 checkpoint. Goal estimate: the mixed-frame DINOv2 checkpoint. Both were fixed before this ranking and were not fit on episodes 0–49.
- SMWM planning config `anygoal/resources/plan/sensorimotor/insert_peg.yaml` transforms `pixels` and `goal` only. `InsertPegEnv._info` returns `proprio` separately from `privilege`.

**Protocol.** The measured start copies joint positions, joint velocities, and hand pose from the proprio field. Object pose, object velocity, and the slot target stay visual. The goal pose is the visual estimate with the hand recomputed by forward kinematics. A second row replaces that goal with the true goal pose and is labeled as a specified goal, not as an image-only result. Success is the saved true-start outcome.

**Results.**
- Forward kinematics reproduces the stored hand to a maximum of 1e-8 m on 20 anchor states. The first stored frame and a render from that state differ by 2.7 gray levels.
- Of training episode starts, 83% have a joint outside the 5th–95th percentile of the mixed-frame sample. Median hand height is 46 cm at the start and 20 cm later. Median hand-to-peg distance is 41 cm at the start and 13 cm later.
- Anchor visual errors at the start: hand 9.7 cm, object 6.4 cm, object angle 0.12 rad. At the goal image: hand 10.8 cm, object 9.4 cm, and 16 of 50 object estimates are more than 20 cm off (11 more than 40 cm).

| Ranker | Success | Opportunity regret |
|---|---:|---:|
| True state | 0.84 | 0.00 [0.00, 0.00] |
| Measured arm, visual object, true goal pose | 0.74 | 0.10 [0.02, 0.20] |
| Measured arm, true object, visual goal | 0.46 | 0.38 [0.24, 0.52] |
| Measured arm, visual object and visual goal | 0.42 | 0.42 [0.28, 0.56] |
| Same, object velocity set to zero | 0.44 | 0.40 [0.26, 0.54] |
| SMWM predicted cost | 0.20 | 0.64 [0.50, 0.76] |

**Limitations.** The 0.10 result gives the goal pose directly. The AnyGoal interface is a goal image, and this estimator does not read that image well enough. The interval [0.02, 0.20] is wider than a tight pass of the 0.10 rule. One checkpoint pair. Proprioception is a different information tier from SMWM's images.

**Conclusion.** **INCONCLUSIVE for a new objective.** Encoders and forward kinematics already supply the current arm, and SMWM does not use them. With that arm and a specified goal pose, regret is 0.10 [0.02, 0.20]. Reading the goal from the image leaves regret at 0.42 because 16 of 50 goal estimates are off by more than 20 cm. The next measurement is a goal-pose estimate that is actually inside 2 cm, followed by the same measured-arm ranking. Do not train another current-arm estimator, and do not open a new learning objective from this inaccurate goal estimate.

## 2026-11-08: Goal pose from the goal image

**Question.** Can the goal hand and peg poses be read from the goal image well enough that measured joints plus Insert-Peg physics remove SMWM's ranking gap?

**Hypothesis.** The metric hand is the wrist origin. Rebuilding it from estimated joints was the previous error. Predicting the metric pose directly, on frames at the same 25-step offset as the goals, should reach 2 cm median hand and object error on episodes 1300–1460. If it does, the measured-arm ranking should fall to about 0.10. If it does not, the baseline stays untested.

**Data and code.**
- `scripts/anygoal_goal_pose.py`, tests in `tests/test_anygoal_goal_pose.py`.
- Train: offsets 10, 20, 25, 30, and 40 on episodes 50–1299 (6,250 frames). Validation: offset 25 on episodes 1300–1460 (161 frames). Episodes 0–49 unused until the final ranking.
- DINOv2-B initialized from the mixed-frame state estimator, last four blocks and a new 8-d head, 14 epochs, about 5 minutes on one L40. Labels are simulator hand and object poses. SMWM does not receive them.
- The fixed camera is 16 m to the side with a 4-degree field of view, about 5 mm per pixel. Inverting it on the true wrist point is exact. The wrist-origin geom itself has no pixels. Stored frames match a re-render to about 2.7 gray levels.

**Results.** Validation at offset 25: hand 2.9 cm median and 6.6 cm at the 90th percentile, hand angle 0.064 rad, object 5.2 cm median and 51 cm at the 90th percentile, object angle 0.079 rad. Anchor goal images, not used for selection: hand 3.4 cm median, no hand error above 20 cm, object 4.2 cm median, 14 of 50 object estimates above 20 cm.

| Ranker | Success | Opportunity regret |
|---|---:|---:|
| True state | 0.84 | 0.00 [0.00, 0.00] |
| Measured arm, visual object, true goal | 0.78 | 0.06 [0.00, 0.14] |
| Measured arm, visual object, estimated goal | 0.70 | 0.14 [0.06, 0.24] |
| True goal hand, estimated goal object | 0.76 | 0.08 [0.02, 0.16] |
| Estimated goal hand, true goal object | 0.70 | 0.14 [0.06, 0.24] |
| SMWM predicted cost | 0.20 | 0.64 [0.50, 0.76] |

The current arm is measured proprioception, and object velocity at the start is set to zero. Replacing the estimated goal object does not change regret. Replacing the estimated goal hand accounts for the difference from the true-goal row.

**Limitations.** Both medians are above 2 cm, so the pre-registered accuracy rule is not met. The object tail is large and, on this bank, unused by the ranking. One head and one offset neighborhood. This does not explain why SMWM's latent predictions rank poorly.

**Conclusion.** **INCONCLUSIVE.** Direct goal-hand readout cuts the previous visual-goal regret from 0.42 to 0.14 [0.06, 0.24]. The interval still reaches 0.24, and the object estimate is not inside 2 cm. The true-goal pipeline with the same measured arm is at 0.06 [0.00, 0.14]. The residual above that line is the last few centimeters of goal-hand error. Do not treat the old 0.42 as a physics failure, and do not open a new learning objective. The same ranking should be repeated only after offset-25 validation is inside 2 cm for both hand and object.

## 2026-11-08: Why the Insert-Peg predictions rank the wrong action

**Question.** On the 50-episode Insert-Peg bank, SMWM's predicted goal cost has opportunity regret 0.64 and the same cost on the encoded future has regret 0.06. Are the wrong decisions concentrated on actions the training set does not support, and do they appear only after autoregressive rollout?

**Hypothesis.** If the failure is missing coverage, prediction error and ranking error should sit on perturbed and transplanted sequences, and the recorded continuation — an exact training trajectory — should be ranked correctly when it succeeds. If the failure is accumulated rollout error, a one-block prediction from the true image should rank the 25-step outcome much better than the five-block rollout.

**Data and code.**
- Same anchor, bank, checkpoint, and saved terminal scores as `a97b5c5`. Episodes 0–49, start at the first stored frame, training integrator, 25-step recorded goal.
- `scripts/anygoal_ranking_mechanism.py`, `scripts/run_anygoal_ranking_mechanism.sh`, tests in `tests/test_anygoal_ranking_mechanism.py`.
- The seed-0 bank was rebuilt from the training file and matched the saved actions. Candidate 0 is that episode's own 25 actions. Candidates 1–7 are 0.1-standard-deviation perturbations. Candidates 8–15 are 25-step windows from the other 1,411 training episodes.
- No new world model and no CEM. The horizon pass re-simulated the saved bank and matched its terminal states.

**Protocol.** Prediction error is the sum of squared differences between the predicted embedding and the embedding of the simulated image. The decision label is the saved 25-step physical success. Opportunity regret, pair accuracy, and action-effect cosine use episode-level bootstrap intervals. A 5 cm / 0.15 rad pose ball is a screen for other training frames near the start. It is not treated as proof of support. One raw action is not a model prediction. The first valid horizon is one block of five actions. Teacher-forced predictions encode the true image at the previous block boundary.

**Results.** The bank contains 50 observed continuations, 350 perturbed actions, and 400 recombined expert actions. Transplant source hands are 0.41 m from the anchor start at the median and 0.03 m at the minimum. No frame outside episodes 0–49 falls in the pose ball around an anchor start. The recorded trajectory leaves that ball; about 0.11 of its steps remain inside.

| Group | Physical success | Prediction SSE | Predicted goal cost | Observed goal cost | Chosen by the prediction |
|---|---:|---:|---:|---:|---:|
| Recorded | 0.82 [0.70, 0.92] | 51.41 [46.09, 57.43] | 60.19 [54.16, 66.82] | 7.35 [6.68, 8.04] | 0.00 [0.00, 0.00] |
| Perturbation | 0.72 [0.60, 0.82] | 51.92 [46.54, 58.00] | 60.73 [54.75, 67.25] | 7.62 [6.94, 8.33] | 0.22 [0.12, 0.34] |
| Transplant | 0.00 [0.00, 0.00] | 52.52 [48.77, 56.38] | 60.40 [55.69, 64.97] | 43.87 [39.73, 48.08] | 0.78 [0.66, 0.88] |

Transplant SSE minus recorded SSE is 1.11 [-3.74, 5.79]. On the 41 episodes where the recorded continuation succeeds, predicted regret is 0.76 [0.61, 0.88] and the recorded action is selected 0 times. Observed regret there is 0.05 [0.00, 0.12]. Predicted pair accuracy is 0.50 [0.41, 0.60]. Observed pair accuracy is 0.98 [0.96, 0.99]. Predicted regret exceeds uniform selection by 0.17 [0.06, 0.27].

Perturbations move the encoded future by 0.51 [0.47, 0.55] and the prediction by 0.45 [0.34, 0.57]. Transplants move the encoded future by 5.90 [5.57, 6.23] and the prediction by 2.92 [2.80, 3.05]. The component of the predicted change along the actual change is -0.10 [-0.26, 0.07], and the cosine is -0.03 [-0.08, 0.02].

One raw action moves the hand by 7.2 mm. At five raw actions, recorded and autoregressive error are the same number, 46.71, and transplant error is 47.80. Predicted regret against the 25-step outcome is already 0.74 [0.62, 0.86], with pair accuracy 0.46 [0.36, 0.56]. At 25 actions the autoregressive recorded error is 51.39 [46.02, 57.53] and the regret is 0.64 [0.50, 0.76]. Predicting the last block from the true image at step 20 still has recorded error 45.77 [40.19, 51.62]. Figure: [`ranking_mechanism.png`](results/anygoal_ranking_mechanism_v1/ranking_mechanism.png).

**Limitations.** One checkpoint. The anchor episodes are inside the training file. The pose ball can miss a supported intervention that differs in one joint by more than 0.15 rad. Transplanted windows are expert strings from other poses, not an exhaustive set of actions that occurred at similar states. Teacher-forced steps later in these episodes are not a held-out episode.

**Conclusion.** **STOP for a new representation objective.** The ranking failure is present on the stored expert continuation, not only on poorly supported actions, and it is present at the first action block. The encoder of the real future still ranks the goal. The forward update moves when the action moves, in a direction unrelated to that future. AnyGoal, SMWM, and Action Influence Maximization already describe weak or misdirected action effects under expert data. They do not by themselves say the expert continuation would be predicted as badly as a recombined one. That is what this bank shows. Do not train a new loss from this gap. Full write-up: [`docs/results/anygoal_ranking_mechanism_v1/report.md`](results/anygoal_ranking_mechanism_v1/report.md).

## 2026-11-08: The orthogonal cosine was an action-dtype bug

**Question.** Are SMWM's predicted latent changes really orthogonal to the encoded changes on recorded Insert-Peg transitions, or did the diagnostic compare a different action or a different target than the training step?

**Hypothesis.** If the checkpoint fails its own forward loss, one-block error on recorded clips will sit at or above a persistence baseline and the same-action cosine will be near zero. If the diagnostic cast or rescaled the action, the official training step on the same frames will not reproduce the SSE of about 47.

**Data and code.**
- Same checkpoint, `sensorimotor_insertpeg.ckpt`, and the Insert-Peg training file. No new training.
- `scripts/anygoal_training_contract.py` reconstructs `HDF5Dataset` clips with frameskip 5 and the trainer's unbiased action normalizer. Tests in `tests/test_anygoal_training_contract.py`.
- The earlier 0.64 result is unchanged in `docs/results/anygoal_ranking_mechanism_v1/`. The correction is `docs/results/anygoal_training_contract_v1/`.

**Protocol.** The training target is the projector embedding of the frame five raw actions later, from the same encoder. There is no EMA and no stop-gradient. Batch-norm running statistics are used in eval mode. 964 clips from 241 episodes cover the start, step 50, the middle, and the last valid clip. The saved bank is then rescored with float32 actions and, as a control, with the uint8 cast the diagnostic had been applying.

**Results.** Projector and prediction-head batch-norm modules have each seen 46,300 batches. A hand-built window matches the official dataset pixels exactly and the normalized action to 1e-7. Rollout and `predict` differ by at most 0.0018.

| Phase | Model SSE | Persistence SSE | Same-action cosine |
|---|---:|---:|---:|
| Start | 1.66 [1.55, 1.78] | 3.97 [3.70, 4.26] | 0.74 [0.72, 0.76] |
| Step 50 | 0.65 [0.61, 0.70] | 2.82 [2.58, 3.06] | 0.84 [0.83, 0.85] |
| Middle | 0.51 [0.47, 0.56] | 1.65 [1.48, 1.84] | 0.78 [0.76, 0.80] |
| Last clip | 0.57 [0.52, 0.63] | 1.73 [1.56, 1.93] | 0.77 [0.75, 0.79] |

Anchor starts are 1.84 [1.54, 2.18]. The cross-action cosine with the real normalized actions, against the rendered futures, is 0.64 [0.60, 0.68].

The policy image is uint8. The diagnostic built the action tensor with that dtype. A normalized block beginning `[0.79, -1.07, 1.33, 1.40, -1.38]` became `[0, 255, 1, 1, 255]`. Repeating that cast matches the stored costs to 0.06 and reproduces regret 0.64 [0.50, 0.76]. Float32 actions give regret 0.12 [0.04, 0.22] and pair accuracy 0.96 [0.93, 0.98].

**Limitations.** One checkpoint. The corrected regret is not zero; dataset frames and re-rendered frames still differ by SSE 8.84 [8.17, 9.56]. Inverse-action MSE on the normalized 25-d block is about 0.4 to 0.5. The 0.12 interval is wider than the observed-future regret of 0.06.

**Conclusion.** **The orthogonal cosine was an evaluation bug.** The training forward pass on recorded transitions beats persistence and aligns with the encoded change, including at the initial pose. The 0.64 regret and the cosine of about −0.03 used actions that had been wrapped into uint8. Correcting the dtype removes that result. This is not a new failure of SMWM's forward objective, and it is not the action-ignoring mechanism in AnyGoal or Action Influence Maximization. Do not train a new loss from the uint8 numbers. Tell Hazel Nam and Petr Ivashkov that the Insert-Peg checkpoint's one-block predictions do align once the action is left in float32.
