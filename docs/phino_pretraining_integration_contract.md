# SIMPACT → PHINO pretraining: verified data contract and next step

**Status:** Code inspection only, November 5, 2026. No PHINO/DINOv2 training has been performed on the SIMPACT dataset. This is an exploratory synthetic dynamics dataset, not measured real-world physics.

## Actual code paths inspected

PHINO fork `arajshah/phino`, branch `main`:

- [`scripts/pretrain.py`](https://github.com/arajshah/phino/blob/main/scripts/pretrain.py) constructs the model, loss, contrastive dataloader, and trainer. Configured by [`configs/pretrain.yaml`](https://github.com/arajshah/phino/blob/main/configs/pretrain.yaml), whose `data.data_dir` defaults to `data/phino_data`.
- [`train/data/dataset.py`](https://github.com/arajshah/phino/blob/main/train/data/dataset.py), `DynaCLIPContrastiveDataset`, loads `metadata.json` and optional `similarity_matrix.npz`; reads `fingerprints/*.npz` for mining; loads `image_path` or `view_paths` for images; returns two images and a trajectory-derived similarity.
- [`train/data/generation.py`](https://github.com/arajshah/phino/blob/main/train/data/generation.py) is the reference for the on-disk schema and `TrajectoryNormalizer` usage.
- [`train/data/mujoco_engine.py`](https://github.com/arajshah/phino/blob/main/train/data/mujoco_engine.py) defines `TrajectoryNormalizer`, the per-action/per-channel normalization and distance-to-similarity mapping.
- [`train/data/assets.py`](https://github.com/arajshah/phino/blob/main/train/data/assets.py) exposes `render_multiview(mjcf_path, out_dir, asset_id, ...)`. This needs a smoke test on *isolated* SIMPACT object scenes, not the original two-carton scene.
- [`train/losses/contrastive.py`](https://github.com/arajshah/phino/blob/main/train/losses/contrastive.py) implements SoftInfoNCE from the image pair and dynamics similarity. No trajectory goes directly into the vision model.

Existing SIMPACT branch `phino-physics-experiments`:

- `scripts/phino_simpact_pilot_v2.py` produces `results/phino_mesh_grid_cutoff_1m_v1/manifest.json`.
- `results/phino_mesh_grid_cutoff_1m_v1/fingerprints/` contains only **complete** four-action fingerprints: `flat_trajectory` shaped **(200,13)**, consisting of four 50-frame segments.
- The grid also has partial single-action trajectories and synthetic labels `mass_kg` and `contact_friction`. **Do not** pretend partial traces are complete fingerprints.

## Required PHINO dataset layout

The *new* adapter should produce a separate directory (do not overwrite any existing results):

```text
data/phino_simpact_v0/
  metadata.json
  images/<object_id>_v0.png             # optionally additional views
  fingerprints/fp_000000.npz           # flat_trajectory=(200,13)
  fingerprints/fp_000001.npz
  ...
  traj_norm.npz
  similarity_matrix.npz
  provenance.json                       # new adapter metadata, not required by PHINO
```

For each **accepted full fingerprint**, one `metadata.json` entry must include at least:

```json
{
  "global_idx": 0,
  "image_group": 0,
  "asset_id": "object_identifier",
  "category": "synthetic_mesh",
  "material": "unknown",
  "mjcf_path": "/absolute/path/to/isolated_scene.xml",
  "image_path": "images/object_identifier_v0.png",
  "view_paths": ["images/object_identifier_v0.png"],
  "fingerprint_path": "fingerprints/fp_000000.npz",
  "mass": 0.30,
  "static_friction": 0.30,
  "dynamic_friction": 0.30,
  "restitution": 0.0
}
```

This is a **format example, not a record copied from results**. The adapter must assign contiguous `global_idx` values, consistent `image_group` for repeated physics of the same geometry, and stable relative paths. Physics values must come from each corresponding simulation record. The sample `dynamic_friction` and `restitution` values above are **placeholders**; SIMPACT's single contact-friction parameter is not a calibrated decomposition into the PHINO generator's friction/restitution semantics. Any chosen mapping needs explicit provenance and must not be interpreted as recovered material properties.

Each `fingerprints/fp_*.npz` should contain `flat_trajectory` with shape `(200,13)`; copying `mass`/`static_friction` into the .npz metadata is useful but the loader chiefly reads `flat_trajectory`. Verify path and label integrity before any training.

### Normalization and pair labels

- SIMPACT currently uses **4 actions**; PHINO's reference generator uses **5**. Instantiate `TrajectoryNormalizer(n_actions=4, timesteps=50)`, **not** the default 5. Keep the action order exactly as declared in SIMPACT's manifest.
- Fit on the complete training fingerprints and save `traj_norm.npz`. Compute `similarity_matrix.npz` with `pairs` (two global indices per row) and `similarities` (one real number in [0,1] per pair) using the **same normalizer**. Keep reproducible random seed and do not fit normalization on held-out evaluation objects.
- The training loader optionally mines additional pairs on demand. Its `hard_neg_ratio` and `hard_pos_ratio` defaults are 0.3 each, which trigger fingerprint loading. For an initial **interface smoke test**, set both to zero and use a small precomputed pair file.
- **Do not use the existing image-training results as evidence that hidden physics is identifiable.** Use separate geometry-level splits for any preliminary evaluation.

### Rendering

- The grid contains `isolated_scenes/` MJCF files; use the **per-object** file for a canonical render. The two-carton source `scene.xml` is not a single-object input.
- Confirm PHINO's `render_multiview` handles the generated mesh assets and ground plane; the existing PHINO renderer may have assumptions about source scenes. If it fails, adapt **only** the image-rendering interface, not the trajectory simulation again.
- Store one geometry's views once; repeating them across mass/friction variants does **not** introduce physical evidence.

## Critical identifiability limitation

A **deterministic image encoder cannot distinguish two hidden mass/friction configurations with identical image input**. PHINO's dataset explicitly assigns different physics to the same image group; its loader samples different camera views and claims this makes such hard negatives learnable. Different viewpoints do **not** reveal arbitrarily varied mass or friction. The optimization target can therefore be contradictory for indistinguishable visual inputs.

This does not mean image representations are useless: appearance can predict **statistical priors** for material, geometry, or likely mass when labels correlate with visible cues. But random reassignment of mass/friction to the same geometry breaks that statistical inference of instance-specific hidden properties.

A meaningful dynamics-conditioned visual experiment will require **observations of response under known actions**, such as pre/post interaction images or short videos + action metadata, or a task whose labels are statistically related to visible material cues. That would require a model input change and should be a separate research decision.

## Smallest next implementation, not another physics sweep

Add **one** converter: `scripts/export_phino_dataset.py` in the SIMPACT fork. It should:

1. Read and validate the existing grid manifest; include **only complete** 200×13 fingerprints. Log the exact omitted configurations.
2. Generate or reuse one image per isolated object; verify images are readable.
3. Emit PHINO-compatible `metadata.json`, `fingerprints/`, `traj_norm.npz`, `similarity_matrix.npz`, and `provenance.json`. Use explicit 4-action normalizer, reproducible pairs, and fixed geometry split.
4. Run a dataset/dataloader smoke test with **no DINOv2 training**: retrieve one batch and check shapes, finite labels, and image-path resolution.

**Success condition for this integration:** One PHINO dataloader batch loads cleanly from SIMPACT-generated data with correct image ↔ object ↔ physics ↔ fingerprint associations. That proves interface compatibility **only**.

**Not the next step:** more sphere torque or friction sweeps, hyperparameter tuning, or a 30,000-step training run on seven geometries.
