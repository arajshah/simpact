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
