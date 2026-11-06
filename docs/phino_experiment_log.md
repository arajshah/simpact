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
