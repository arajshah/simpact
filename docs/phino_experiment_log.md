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
