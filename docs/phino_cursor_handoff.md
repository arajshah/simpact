# SIMPACT → PHINO: Cursor agent handoff

Updated: November 6, 2026. This is the **working implementation handoff**, not a replacement for code/manifests. Keep it short and update it when a milestone changes.

## Goal and research boundary

Build a reproducible bridge between simulated controlled rigid-object interactions in SIMPACT and PHINO's image-based representation-training data interface. The immediate result is **one validated PHINO dataloader batch**, not a paper claim or a full training run. The overall research question is whether action-response observations can support representations of object dynamics that transfer to new geometries.

**Critical:** We have shown partial recovery of synthetic mass/friction from privileged simulator trajectories, **not** recovery of unknown physics from pixels. When the same static object image is paired with arbitrarily different hidden mass/friction, a deterministic image encoder cannot infer the instance-specific assignment. Random camera viewpoints do not resolve that. Therefore a static-image PHINO smoke test validates interface compatibility only. Later visual-physics research should consider action-conditioned images/video and known actions, or physical priors statistically tied to visible material cues. Do not quietly treat the interface smoke test as successful physical learning.

## Repositories and where code/data live

- SIMPACT upstream: `ShaoxiongYao/simpact`; user's fork: **`arajshah/simpact`**, branch **`phino-physics-experiments`**. Work starts here.
- PHINO upstream: `Hongyang-Du/phino`; user's fork: **`arajshah/phino`**, branch **`main`**. Existing PHINO `train/`, `scripts/`, and `configs/` implement the intended data reader, normalizer, image renderer, and DINOv2-B/14 trainer.
- Brown server's SIMPACT working directory: `/nbu/people/ashah264/simpact` (same tree as `/cs/local/nbu/people/ashah264/simpact`). PHINO is at `/nbu/people/ashah264/phino`, checked out on `eval/grouped-physics-probes-oct28` (`3f1134d`); its contrastive loader is identical to `main`. Python environment: the SIMPACT uv `.venv` has MuJoCo 3.9, torch, and torchvision; rendering uses `MUJOCO_GL=egl` on a GPU node.
- Existing **generated results are on Brown, NOT in GitHub**. A remote/cloud Cursor Agent cloning GitHub will not automatically see `results/phino_mesh_grid_cutoff_1m_v1/` or local MuJoCo assets. Use local/SSH-enabled execution for real data or a small deterministic synthetic fixture for cloud work. Do not upload entire datasets/checkpoints to Git just to supply the agent.
- Two source repos are separate. Use relative links and explicit `--phino-root`/repo paths. Do not assume PHINO appears as a submodule of SIMPACT.

## Existing simulation results (do not regenerate them now)

- Source SIMPACT generator: `scripts/phino_simpact_pilot_v2.py`; multi-scene assets: `scripts/build_phino_mesh_scenes.py`.
- Seven geometries: two real2sim cartons plus five test meshes (cube, bunny, cow, sphere, nut). They are synthetic interactions with **manually assigned** three masses `[0.25, 0.30, 0.35]` kg and three contact friction coefficients `[0.20, 0.30, 0.40]`. There are 63 object-physics configurations.
- Four fixed wrench actions, ordered `push_x`, `push_y`, `grasp_lift_release`, `lateral_flick`. Each valid trace is `(50, 13)`: position relative to start (3), quaternion wxyz (4), linear velocity (3), angular velocity (3). A **complete** accepted fingerprint is `(200, 13)`, four traces concatenated in manifest action order. Rejected actions are missing. Neither physics labels nor validity flags enter the trajectory feature vectors.
- Original grid: `results/phino_mesh_grid_v1/`, 0.25-m displacement cap; expanded grid: **`results/phino_mesh_grid_cutoff_1m_v1/`**, 1.0-m cap. Both contain manifest, `action_trajectories/`, `fingerprints/`, and `isolated_scenes/`. Do not modify them.
- The 0.25-m cap systematically excluded sphere pushes. With 1.0 m, sphere `push_x` 9/9 and `push_y` 8/9 were valid in the analysis. Sphere rotation included substantial vertical-axis spinning (around 78–81% of squared angular speed), so do not assume sphere motion is pure rolling.
- The 1.0-m grid has **40/63** complete fingerprints. **All 21 configurations at 0.25 kg are incomplete**: `grasp_lift_release` lifts with 2.973 N, which exceeds a 0.25 kg object's weight (≈2.45 N), so it overshoots 1.0 m. Exported complete fingerprints therefore cover only masses 0.30 and 0.35 kg. The other two exclusions are bunny 0.30/0.40 (lift tilt) and sphere 0.35/0.20 (`push_y` tilt).
- `scripts/analyze_phino_pilot_grid.py` probes synthetic property recovery with ridge regression on **trajectory-derived** features. Results showed some transfer to nonsphere geometries and major extrapolation failure when sphere was held out; matched train-with/without-sphere experiments showed a mixed benefit. These are diagnostics, not PHINO training.
- No more sphere sweeps or torque ablations are needed for immediate integration.

## Exact PHINO training interface to reuse

Read [`docs/phino_pretraining_integration_contract.md`](./phino_pretraining_integration_contract.md) for the full mapping, source paths, and caveats. Core PHINO files:

1. `arajshah/phino/scripts/pretrain.py` calls `train.data.dataset.create_contrastive_dataloader` and `train.trainers.pretrain.DynaCLIPTrainer`.
2. `arajshah/phino/train/data/dataset.py` reads `metadata.json`, `similarity_matrix.npz`, `fingerprints/fp_*.npz`, and image paths.
3. `arajshah/phino/train/data/generation.py` defines the reference on-disk metadata and fingerprint schema.
4. `arajshah/phino/train/data/mujoco_engine.py` defines `TrajectoryNormalizer`. Important: its **default is 5 actions**, but SIMPACT has **4**. Instantiate with `n_actions=4, timesteps=50`; fit only on training data.
5. `arajshah/phino/train/data/assets.py` has `render_multiview` for canonical 224×224 images. Validate on each isolated SIMPACT object scene. A separate renderer adapter is acceptable if this fails.
6. `arajshah/phino/train/losses/contrastive.py` uses image pairs and pairwise dynamics similarity in SoftInfoNCE. This code uses a diagonal-weighted soft-label matrix, not a complete `B×B` dynamics target. Treat it as a modeling limitation, not a proven full similarity objective.

Originally planned output (separate directory; **not** a rewrite of SIMPACT results). As implemented, this layout appears once under `train/` and once under `heldout/` (see milestone status below):

```text
data/phino_simpact_v0/
  metadata.json
  images/<object>_v0.png             # or other validated paths
  fingerprints/fp_000000.npz         # flat_trajectory: (200,13)
  ...
  traj_norm.npz                       # 4 actions × 50 frames
  similarity_matrix.npz              # pairs: (N,2), similarities: (N,)
  provenance.json
```

PHINO metadata must associate each accepted configuration with exactly one existing synthetic physics record and one image group for its geometry; reference `mass`, `static_friction`, `restitution`, `image_path`, `view_paths`, `fingerprint_path`, `global_idx`, `asset_id`, `image_group`. SIMPACT's single contact-friction coefficient is **not** a measured dynamic/static friction decomposition; restitution is **not** provided as a measured property. Explicitly document placeholder mapping decisions, do not invent scientific measurements. The existing PHINO static-image hard-negative mining protocol (same image/different randomly assigned physics) has an identifiability contradiction.

## Milestone status: dataset export and dataloader smoke test (done, Nov 5, 2026)

`scripts/export_phino_dataset.py` (commit `622693b`) plus `tests/test_export_phino_dataset.py` (18 synthetic-fixture tests). On Brown, from the SIMPACT root:

```bash
.venv/bin/python scripts/export_phino_dataset.py --validate-only        # grid checks only
MUJOCO_GL=egl .venv/bin/python scripts/export_phino_dataset.py \
  --grid-dir results/phino_mesh_grid_cutoff_1m_v1 --phino-root /nbu/people/ashah264/phino \
  --output-dir data/phino_simpact_v0                                     # refuses existing output
MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_export_phino_dataset.py
```

Output `data/phino_simpact_v0/` (gitignored, Brown only) has `train/` and `heldout/`, each a self-contained PHINO `data_dir`, plus `provenance.json`, `geometry_splits.json`, and `contact_sheet.png`. PHINO's contrastive loader samples every `metadata.json` entry, so **only `train/` may be passed to pretraining**. The normalizer is `TrajectoryNormalizer(4, 50)`, fit on `train/` only; `heldout/traj_norm.npz` is a byte copy.

Verified on the real grid: all 40 accepted fingerprints were exported unchanged (train 29, heldout 11; 23 exclusions with reasons). The real PHINO `create_contrastive_dataloader` returned a batch from both splits with hard mining off. Both source grids are byte-identical afterwards. Details and numbers: `docs/phino_experiment_log.md`.

Compatibility decisions, recorded in `provenance.json`:
- `static_friction` and `dynamic_friction` both take SIMPACT's `contact_friction`, because MuJoCo has a single sliding coefficient.
- `restitution` is a constant 0.0 placeholder, matching MuJoCo's default damping ratio of 1.0 under PHINO's mapping.
- Renders use PHINO's scene composition, settling, and camera, but frame only the object body. PHINO's default framing includes the 1.2 m carton table and shrinks the carton to about 0.5% of the pixels.

### Needs research-owner review before any training run
1. **Geometry split.** The default split (seed 0, two held out) holds out **cube and sphere**. It was chosen by seed before any results existed. The sphere is the known out-of-distribution geometry, so this is a demanding extrapolation test. Re-export with `--holdout-geometry` if a different design is wanted.
2. **Mass coverage.** Only 2 of 3 mass levels survive (see above). Fixing this requires a grid or validation-rule change, which needs an explicit experiment specification.
3. **Held-out similarity saturation.** Under the train-fit normalizer, 56% of held-out pairs have similarity below 1e-3. Sphere push displacement and yaw-rate channels reach up to 40 training standard deviations.
4. **Static-image identifiability** (above). In `train/`, 70 of 406 pairs share one image but have different physics targets.

## Milestone status: action-conditioned visual observability pilot (done, Nov 5, 2026)

`scripts/phino_visual_observability_pilot.py` (commit `a8f813b`) replays saved valid `push_x` traces kinematically. It does not regenerate physics. Each geometry's camera, lighting, and action are fixed, and the replay reproduces saved traces exactly. Run with `MUJOCO_GL=egl .venv/bin/python scripts/phino_visual_observability_pilot.py`; full output goes to `results/phino_visual_observability_v1/` (Brown only), with summaries and figures in `docs/results/phino_visual_observability_v1/`. Full write-up: `docs/phino_experiment_log.md`.

- **Response images distinguish configurations; static images do not.** At 224 px, 250 of 252 within-geometry pairs differ by at least 1 px in final object position after one push. One grid step moves the object 8–25 px.
- **One horizontal push reveals one mass–friction combination, not both.**
  - The closest pairs always trade mass against friction (cube 0.30 kg/0.4 vs 0.35 kg/0.3 differ by 0.7 px).
  - `lateral_flick` responds to nearly the same combination (0.7–5.6° apart for cube, cow, and cartons), so it doesn't break the confound.
  - The time course adds little beyond the final frame.
  - Sphere push displacement depends on mass only (μ exponent −0.04).
- **New data-validity problem in the existing grid.** The generator's settle step depends on friction, so starting poses differ across friction levels:

  | Geometry | Start offset across friction | Residual motion at push start |
  |---|---|---|
  | cow | 25 mm, 45° | none |
  | bunny | 8 mm, 15° | still moving (1.1 rad/s) |
  | sphere | 3 mm, 3.6° | still moving (0.55 rad/s) |
  | blue carton | 2 mm, 0.9° | none |
  | white carton, cube, nut | matched | none |

  Consequences:
  - The simulated pre-push images for cow and bunny already differ by friction.
  - Absolute quaternion fingerprint channels carry the settle orientation, so earlier cow/bunny trajectory analyses may partly reflect settling.
  - The PHINO v0 export images are unaffected (one PHINO-settled render per geometry).

## Research plan (approved Nov 5, 2026): identifiability before representation learning

Central question: **which physical interventions make hidden object properties identifiable, and can learned visual representations preserve that information?** The plan has four stages, each gated on research-owner approval:
1. audit the existing results;
2. write the v3 specification;
3. run a state-only identifiability pilot;
4. representation learning: render action-conditioned observations; frozen DINOv2 linear probes vs PHINO contrastive fine-tuning on identical inputs, splits, and protocol; plus a static-image negative control.

### Stage 1: audit (done)
`scripts/audit_phino_settle_confound.py` (commit `5d1c91d`); outputs in `docs/results/phino_settle_audit_v1/`; full write-up in the experiment log. The audit reproduces all 40 published ridge cells exactly.

| Earlier result | Verdict |
|---|---|
| Within-geometry recovery on held-out physics | Holds on clean geometries (friction about 30% better than baseline, mass about 10%) |
| Friction transfer to held-out geometries | Holds (0.045 vs 0.067) |
| Mass transfer to held-out geometries | Weaker on clean test objects (0.0283 vs published 0.0266; baseline 0.0333) |
| Sphere out-of-distribution failure | Holds (settle drift is 1.7% of the sphere's response) |
| "Mixed" sphere-inclusion result | Holds as mixed |
| v0 PHINO within-geometry similarities | Contaminated for cow (settle-only swap drops similarity to as low as 0.83) and bunny (settling motion up to 14% of its response) |
| v0 PHINO cross-geometry similarities | Mostly reflect mesh body-frame conventions (rank correlation 0.74 vs start-relative). **Do not train on v0 similarities.** |

### Stage 2: v3 specification (approved with two amendments, Nov 5)
[`docs/phino_v3_identifiability_spec.md`](./phino_v3_identifiability_spec.md) proposes:
- a canonical at-rest start with acceptance checks plus a zero-action control;
- pure centre-of-mass pushes at two strengths (`push_lo`, `push_hi`);
- a friction-independent `toss` replacing the v2 lift;
- start-relative recording;
- pre-registered, validity-only candidate selection;
- Stage 3 metrics: local Jacobian/Cramér–Rao identifiability, off-grid held-out prediction, confusability, and solver robustness. Hypotheses and a gate are fixed in advance.

A Coulomb toy model already predicts that two push strengths observed only at the final frame are nearly degenerate (about 5° apart). Time-resolved deceleration or the toss should break this, and Stage 3 must verify it on real contacts.

### Stage 3: state-only identifiability pilot (done; gate failed)
`scripts/phino_simpact_grid_v3.py` and `scripts/phino_v3_identifiability_pilot.py` (commit `c3bf96a`); outputs in `docs/results/phino_v3_identifiability_pilot_v1/`; full write-up in the experiment log.

- **Mass:** the vertical toss is validated. Its apex is shape-independent and friction-blind (sensitivity 0.000), it is airborne for at least 5 frames, and it predicts mass at 0.07–0.12 of baseline error (ideal) and 0.26–0.29 (224 px).
- **Friction:** needs time-resolved post-push deceleration. Identifiable from ideal states (one push at 100 Hz, or push_lo + push_hi + toss at 20 Hz; cow and cube); at 224 px and 20 Hz only marginal (cow 0.50, cube 0.52).
- **Two pushes, final positions only:** degenerate. Sensitivity directions are only 2.1–8.4° apart, matching the Coulomb prediction.
- **Gate fails** (at most 2 of 6) under every variant, mainly because of protocol problems:
  - S1/S2 speed thresholds are below contact jitter;
  - the bunny rocks and the sphere rolls;
  - centre-of-mass pushes topple the cartons and flip the nut;
  - the sphere exceeds the displacement cap.
- The pre-registered selection picks no push candidate.

## Milestone status: action-conditioned representation pilot (done, Nov 6, 2026)

Ran because the research owner asked for a bounded learning comparison on existing traces, without waiting for v3.1 and without new physics. Code: `f48d1e1` plus a later 4×4 spatial readout in `scripts/phino_action_probe.py`. Full write-up: `docs/phino_experiment_log.md`. Summaries: `docs/results/phino_action_learning_v2/`.

- Dataset `data/phino_simpact_v3obs_v2/` (Brown only): kinematic replay of saved Stage 3 traces, canonical rest pose, one shared before image per geometry and camera. Bunny and sphere excluded. An earlier v1 leaked friction through sub-0.1 mm start offsets and must not be used.
- 15 leave-one-geometry-out runs finished (clip seeds 0 and 1, plus a static-image control). No training job is still running.
- On 292 configurations with all four actions, contrastive clip fine-tuning beats frozen DINOv2 on leave-one-geometry-out (four interventions: mass/friction nMAE about 0.60/0.61 versus 0.76/0.86) and beats the static control (about 0.86/0.83). Both seeds agree. The static control does not improve on frozen features.
- An image-position tracker on the same clips is far better (about 0.04/0.12). Pooled DINOv2 features do not preserve that image-space motion across geometries (position-decoding R² ≤ 0.2, often negative).
- A frozen 4×4 patch readout narrows the gap only partly (four interventions: 0.67/0.59). Fine-tuned spatial features are probed in `results/phino_action_learning_v2/probes_grid_ft/` and recorded in the experiment log.

## Next task

Do not generate a new physics grid, and do not start another training sweep, until R1–R6 in the spec's §10 are approved and a re-registered Stage 3 gate passes. On the current clips the ceiling is the image-position tracker, not another pooled-feature fine-tune. The open scientific question is whether a spatial readout can keep that motion signal; the frozen 4×4 result says only partly.

## Handoff protocol

- Cursor is the coding agent. It should inspect source code, propose a minimal plan, implement the adapter, run lightweight tests, and publish a concise PR.
- User is the research owner who approves design changes and runs data/GPU checks on Brown.
- ChatGPT is optional for **scientific review, experimental validity, and interpreting results**, not a required messenger for every implementation step.
- Keep this handoff current when research findings or the next task changes. Preserve project history in the GitHub docs, not the large general Brown academic document.
