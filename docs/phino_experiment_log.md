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
