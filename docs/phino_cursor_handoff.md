# SIMPACT → PHINO: Cursor agent handoff

Updated: November 5, 2026. This is the **working implementation handoff**, not a replacement for code/manifests. Keep it short and update it when a milestone changes.

## Goal and research boundary

Build a reproducible bridge between simulated controlled rigid-object interactions in SIMPACT and PHINO's image-based representation-training data interface. The immediate result is **one validated PHINO dataloader batch**, not a paper claim or a full training run. The overall research question is whether action-response observations can support representations of object dynamics that transfer to new geometries.

**Critical:** We have shown partial recovery of synthetic mass/friction from privileged simulator trajectories, **not** recovery of unknown physics from pixels. When the same static object image is paired with arbitrarily different hidden mass/friction, a deterministic image encoder cannot infer the instance-specific assignment. Random camera viewpoints do not resolve that. Therefore a static-image PHINO smoke test validates interface compatibility only. Later visual-physics research should consider action-conditioned images/video and known actions, or physical priors statistically tied to visible material cues. Do not quietly treat the interface smoke test as successful physical learning.

## Repositories and where code/data live

- SIMPACT upstream: `ShaoxiongYao/simpact`; user's fork: **`arajshah/simpact`**, branch **`phino-physics-experiments`**. Work starts here.
- PHINO upstream: `Hongyang-Du/phino`; user's fork: **`arajshah/phino`**, branch **`main`**. Existing PHINO `train/`, `scripts/`, and `configs/` implement the intended data reader, normalizer, image renderer, and DINOv2-B/14 trainer.
- Brown server's SIMPACT working directory: `/nbu/people/ashah264/simpact`. PHINO repository was previously at `/nbu/people/ashah264/phino` (verify). Python environment: `(simpact)`.
- Existing **generated results are on Brown, NOT in GitHub**. A remote/cloud Cursor Agent cloning GitHub will not automatically see `results/phino_mesh_grid_cutoff_1m_v1/` or local MuJoCo assets. Use local/SSH-enabled execution for real data or a small deterministic synthetic fixture for cloud work. Do not upload entire datasets/checkpoints to Git just to supply the agent.
- Two source repos are separate. Use relative links and explicit `--phino-root`/repo paths. Do not assume PHINO appears as a submodule of SIMPACT.

## Existing simulation results (do not regenerate them now)

- Source SIMPACT generator: `scripts/phino_simpact_pilot_v2.py`; multi-scene assets: `scripts/build_phino_mesh_scenes.py`.
- Seven geometries: two real2sim cartons plus five test meshes (cube, bunny, cow, sphere, nut). They are synthetic interactions with **manually assigned** three masses `[0.25, 0.30, 0.35]` kg and three contact friction coefficients `[0.20, 0.30, 0.40]`. There are 63 object-physics configurations.
- Four fixed wrench actions, ordered `push_x`, `push_y`, `grasp_lift_release`, `lateral_flick`. Each valid trace is `(50, 13)`: position relative to start (3), quaternion wxyz (4), linear velocity (3), angular velocity (3). A **complete** accepted fingerprint is `(200, 13)`, four traces concatenated in manifest action order. Rejected actions are missing. Neither physics labels nor validity flags enter the trajectory feature vectors.
- Original grid: `results/phino_mesh_grid_v1/`, 0.25-m displacement cap; expanded grid: **`results/phino_mesh_grid_cutoff_1m_v1/`**, 1.0-m cap. Both contain manifest, `action_trajectories/`, `fingerprints/`, and `isolated_scenes/`. Do not modify them.
- The 0.25-m cap systematically excluded sphere pushes. With 1.0 m, sphere `push_x` 9/9 and `push_y` 8/9 were valid in the analysis. Sphere rotation included substantial vertical-axis spinning (around 78–81% of squared angular speed), so do not assume sphere motion is pure rolling.
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

Target new output (separate directory; **not** a rewrite of SIMPACT results):

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

## Immediate Cursor implementation assignment

**One narrowly scoped task:** Add `scripts/export_phino_dataset.py` in SIMPACT (and tests), taking `--grid-dir results/phino_mesh_grid_cutoff_1m_v1`, `--phino-root <path>`, `--output-dir data/phino_simpact_v0`.

Acceptance criteria:

- Validate the source manifest, hashes/identifiers where present, action order, `(200,13)` fingerprint shapes, finiteness, no missing/duplicated pair mapping, and image/fingerprint associations.
- Include only complete accepted fingerprints; report the exact number and reasons for omitted configurations. Do not impute or relabel rejections.
- Render and validate at least a canonical image **once per geometry** from the isolated scene (rather than once per mass/friction setting). Do not treat multiple camera views as separate physics observations.
- Emit PHINO-compatible metadata, sequential indices, stable image groups and relative paths, new fingerprint files, a 4-action fitted normalizer, pairwise similarities, provenance, deterministic seed, and explicit geometry-disjoint holdout grouping.
- A lightweight smoke test must load the generated metadata/fingerprints and fetch one PHINO dataloader batch with finite similarities and usable images. Start with hard mining ratios both zero and a small precomputed similarity pair file. No PHINO training or checkpoint required.
- Failure should be explicit if rendering is unavailable, source files are absent, format is wrong, label mapping is ambiguous, or the output exists. Include self-test with synthetic fixtures so cloud agents can test without the Brown dataset.
- Avoid silent data leakage: train-only normalizer/label calibration and geometry-based split for later evaluations. If this cannot be implemented correctly without a data format change, explain the issue rather than fitting to all examples.
- Commit descriptively to a branch, open a PR or present diff, list tests run, failures, and exact Brown commands. **Do not overwrite the two previous grids**.

## Handoff protocol

- Cursor is the coding agent. It should inspect source code, propose a minimal plan, implement the adapter, run lightweight tests, and publish a concise PR.
- User is the research owner who approves design changes and runs data/GPU checks on Brown.
- ChatGPT is optional for **scientific review, experimental validity, and interpreting results**, not a required messenger for every implementation step.
- Keep this handoff current when research findings or the next task changes. Preserve project history in the GitHub docs, not the large general Brown academic document.
