"""Synthetic-fixture tests for scripts/export_phino_dataset.py.

Validation tests need only numpy. Export tests need a PHINO checkout (PHINO_ROOT
env var, the Brown default path, or a sibling ``phino`` directory); the
dataloader smoke test additionally needs torch/torchvision, and the renderer test
needs MuJoCo with a working offscreen GL backend.
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest

import export_phino_dataset as ex
import phino_simpact_pilot_v2 as pilot

ACTIONS = ex.EXPECTED_ACTION_ORDER
OBJECTS = ("box_a", "box_b", "box_c", "box_d")
MASSES = (0.30, 0.35)
FRICTIONS = (0.20, 0.40)


def _find_phino_root() -> Path | None:
    candidates = [os.environ.get("PHINO_ROOT"), ex.DEFAULT_PHINO_ROOT,
                  Path(__file__).resolve().parents[2] / "phino"]
    for c in candidates:
        if c and (Path(c) / "train" / "data" / "mujoco_engine.py").is_file():
            return Path(c)
    return None


PHINO_ROOT = _find_phino_root()
needs_phino = pytest.mark.skipif(PHINO_ROOT is None, reason="PHINO checkout not found")


def _trajectory(rng: np.random.Generator) -> np.ndarray:
    traj = np.zeros((ex.FRAMES_PER_ACTION, ex.STATE_DIM), dtype=np.float32)
    traj[:, :3] = np.cumsum(rng.normal(0, 0.01, (ex.FRAMES_PER_ACTION, 3)), axis=0)
    quat = rng.normal(0, 1, (ex.FRAMES_PER_ACTION, 4))
    traj[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    traj[:, 7:] = rng.normal(0, 0.3, (ex.FRAMES_PER_ACTION, 6))
    return traj


def make_grid(root: Path, reject: set[tuple[str, float, float, str]] = frozenset(),
              objects=OBJECTS) -> Path:
    grid = root / "grid"
    for sub in ("isolated_scenes", "fingerprints", "action_trajectories"):
        (grid / sub).mkdir(parents=True)
    index, records = [], []
    for k, body in enumerate(objects):
        object_id = f"scene_{k}__{body}"
        (grid / "isolated_scenes" / f"{pilot._slug(object_id)}.xml").write_text(
            '<mujoco><worldbody><geom name="table" type="plane" size="1 1 0.1"/>'
            f'<body name="{body}" pos="0 0 0.05"><freejoint/>'
            f'<geom type="box" size="0.03 0.04 0.05" mass="0.3" rgba="0.{k + 2} 0.3 0.8 1"/>'
            '</body></worldbody></mujoco>')
        index.append({"object_id": object_id, "body_name": body, "scene_id": f"scene_{k}",
                      "source_scene": f"/src/{body}.xml", "source_scene_sha256": "0" * 64})
        for m in MASSES:
            for f in FRICTIONS:
                slug = f"{pilot._slug(object_id)}_m{m:.3f}_mu{f:.3f}"
                rng = np.random.default_rng(zlib.crc32(slug.encode()))
                row = {"object": object_id, "source_body_name": body, "mass_kg": m,
                       "contact_friction": f, "action_status": {}, "status": "rejected"}
                traces = {}
                for action in ACTIONS:
                    if (body, m, f, action) in reject:
                        row["action_status"][action] = {"status": "rejected", "valid": False,
                                                        "reason": "synthetic rejection"}
                        continue
                    traces[action] = _trajectory(rng)
                    rel = f"action_trajectories/{slug}__{action}.npz"
                    np.savez_compressed(grid / rel, trajectory=traces[action],
                                        object_name=object_id, action_name=action,
                                        mass_kg=m, contact_friction=f, frames_per_action=50)
                    row["action_status"][action] = {"status": "valid", "valid": True,
                                                    "trajectory_path": rel}
                if len(traces) == len(ACTIONS):
                    rel = f"fingerprints/{slug}.npz"
                    np.savez_compressed(grid / rel,
                                        flat_trajectory=np.concatenate([traces[a] for a in ACTIONS]),
                                        action_names=np.asarray(ACTIONS), mass_kg=m,
                                        contact_friction=f, frames_per_action=50)
                    row.update(status="accepted", fingerprint_path=rel)
                else:
                    row["failed_actions"] = [a for a in ACTIONS if a not in traces]
                records.append(row)
    accepted = sum(r["status"] == "accepted" for r in records)
    manifest = {
        "format_version": ex.SUPPORTED_GRID_FORMATS[0], "action_order": list(ACTIONS),
        "frames_per_action": 50, "feature_order": ex.EXPECTED_FEATURE_ORDER,
        "mass_grid_kg": list(MASSES), "friction_grid": list(FRICTIONS),
        "source_object_index": index, "records": records, "accepted_count": accepted,
        "rejected_count": len(records) - accepted, "total_count": len(records)}
    (grid / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return grid


def _edit_manifest(grid: Path, fn) -> None:
    path = grid / "manifest.json"
    manifest = json.loads(path.read_text())
    fn(manifest)
    path.write_text(json.dumps(manifest))


def fake_renderer(geom, images_dir, n_views, img_size):
    from PIL import Image
    paths = []
    for k in range(n_views):
        arr = np.full((img_size, img_size, 3), 200, dtype=np.uint8)
        q = img_size // 4
        arr[q:3 * q, q:3 * q] = (40 * geom.image_group % 255, 80, 160 + k)
        path = images_dir / f"{geom.object_id}_v{k}.png"
        Image.fromarray(arr).save(path)
        paths.append(path)
    return ex.RenderResult(paths, [0.25] * n_views, {"fake": True})


# ---------------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------------
def test_load_grid_keeps_only_complete_and_reports_exclusions(tmp_path):
    grid = make_grid(tmp_path, reject={("box_a", 0.30, 0.20, "grasp_lift_release")})
    inv = ex.load_grid(grid)
    assert len(inv.accepted) == 15 and len(inv.excluded) == 1
    excl = inv.excluded[0]
    assert excl["object_id"] == "scene_0__box_a" and excl["mass_kg"] == 0.30
    assert excl["failed_actions"] == [{"action": "grasp_lift_release", "status": "rejected",
                                       "reason": "synthetic rejection"}]
    assert all(c.trajectory.shape == (200, 13) for c in inv.accepted)


def test_coverage_flags_label_level_with_no_complete_fingerprints(tmp_path):
    reject = {(b, 0.30, f, "grasp_lift_release") for b in OBJECTS for f in FRICTIONS}
    cov = ex.coverage_report(ex.load_grid(make_grid(tmp_path, reject=reject)))
    assert cov["accepted_complete_fingerprints"] == 8
    assert any("mass_kg=0.3 " in w for w in cov["warnings"])


def test_missing_accepted_fingerprint_is_integrity_error(tmp_path):
    grid = make_grid(tmp_path)
    next((grid / "fingerprints").glob("*.npz")).unlink()
    with pytest.raises(ex.ExportError, match="missing"):
        ex.load_grid(grid)


def test_missing_action_trace_for_accepted_record_fails(tmp_path):
    grid = make_grid(tmp_path)
    next((grid / "action_trajectories").glob("*__push_y.npz")).unlink()
    with pytest.raises(ex.ExportError, match="missing valid push_y"):
        ex.load_grid(grid)


@pytest.mark.parametrize("bad, match", [
    (np.zeros((150, 13), dtype=np.float32), "shape"),
    (np.full((200, 13), np.nan, dtype=np.float32), "non-finite"),
])
def test_bad_fingerprint_array_fails(tmp_path, bad, match):
    grid = make_grid(tmp_path)
    path = sorted((grid / "fingerprints").glob("*.npz"))[0]
    np.savez_compressed(path, flat_trajectory=bad)
    with pytest.raises(ex.ExportError, match=match):
        ex.load_grid(grid)


def test_manifest_action_order_mismatch_fails(tmp_path):
    grid = make_grid(tmp_path)
    _edit_manifest(grid, lambda m: m.update(action_order=list(reversed(ACTIONS))))
    with pytest.raises(ex.ExportError, match="action_order"):
        ex.load_grid(grid)


def test_fingerprint_segments_in_wrong_order_fail(tmp_path):
    grid = make_grid(tmp_path)
    path = sorted((grid / "fingerprints").glob("*.npz"))[0]
    flat = np.load(path)["flat_trajectory"]
    swapped = np.concatenate([flat[50:100], flat[:50], flat[100:]])
    np.savez_compressed(path, flat_trajectory=swapped, action_names=np.asarray(ACTIONS))
    with pytest.raises(ex.ExportError, match="segment 0"):
        ex.load_grid(grid)


def test_fingerprint_label_disagreeing_with_record_fails(tmp_path):
    grid = make_grid(tmp_path)
    path = sorted((grid / "fingerprints").glob("*.npz"))[0]
    flat = np.load(path)["flat_trajectory"]
    np.savez_compressed(path, flat_trajectory=flat, mass_kg=9.0)
    with pytest.raises(ex.ExportError, match="mass_kg"):
        ex.load_grid(grid)


def test_duplicate_configuration_fails(tmp_path):
    grid = make_grid(tmp_path)
    _edit_manifest(grid, lambda m: m["records"].append(m["records"][0]))
    with pytest.raises(ex.ExportError, match="duplicate configuration"):
        ex.load_grid(grid)


def test_accepted_status_with_invalid_action_fails(tmp_path):
    grid = make_grid(tmp_path)

    def corrupt(m):
        m["records"][0]["action_status"]["push_x"]["valid"] = False
    _edit_manifest(grid, corrupt)
    with pytest.raises(ex.ExportError, match="inconsistent"):
        ex.load_grid(grid)


def test_unreferenced_fingerprint_file_fails(tmp_path):
    grid = make_grid(tmp_path)
    np.savez_compressed(grid / "fingerprints" / "orphan.npz", flat_trajectory=np.zeros((200, 13)))
    with pytest.raises(ex.ExportError, match="unreferenced"):
        ex.load_grid(grid)


def test_isolated_scene_with_wrong_body_fails(tmp_path):
    grid = make_grid(tmp_path)
    scene = next((grid / "isolated_scenes").glob("*box_a.xml"))
    scene.write_text(scene.read_text().replace('name="box_a"', 'name="other"'))
    with pytest.raises(ex.ExportError, match="free bodies"):
        ex.load_grid(grid)


# ---------------------------------------------------------------------------
# Geometry split
# ---------------------------------------------------------------------------
def test_split_holds_out_whole_geometries_deterministically():
    ids = [f"g{i}" for i in range(7)]
    a = ex.assign_geometry_split(ids, None, 2, seed=0)
    b = ex.assign_geometry_split(list(reversed(ids)), None, 2, seed=0)
    assert a == b and len(a["heldout"]) == 2 and len(a["train"]) == 5
    assert set(a["train"]).isdisjoint(a["heldout"])
    explicit = ex.assign_geometry_split(ids, ["g3"], 2, seed=0)
    assert explicit["heldout"] == ["g3"]
    with pytest.raises(ex.ExportError, match="Unknown"):
        ex.assign_geometry_split(ids, ["nope"], 2, seed=0)
    with pytest.raises(ex.ExportError, match="n-holdout"):
        ex.assign_geometry_split(ids, None, 7, seed=0)


# ---------------------------------------------------------------------------
# End-to-end export (needs PHINO)
# ---------------------------------------------------------------------------
def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


@needs_phino
def test_export_end_to_end_is_leak_free_and_consistent(tmp_path):
    grid = make_grid(tmp_path, reject={("box_b", 0.35, 0.40, "push_y")})
    out = tmp_path / "out"
    prov = ex.export_dataset(grid, out, PHINO_ROOT, holdout=["scene_3__box_d"], seed=0,
                             render_fn=fake_renderer, run_smoke=_torch_available(), argv=[])
    inv = ex.load_grid(grid)
    by_record = {c.record_index: c for c in inv.accepted}
    assert prov["export_counts"] == {"train": 11, "heldout": 4}
    assert len(prov["excluded_configurations"]) == 1

    train = json.loads((out / "train" / "metadata.json").read_text())
    heldout = json.loads((out / "heldout" / "metadata.json").read_text())
    assert [e["global_idx"] for e in train] == list(range(11))
    assert {e["asset_id"] for e in heldout} == {"scene_3__box_d"}
    assert "scene_3__box_d" not in {e["asset_id"] for e in train}
    for split, meta in (("train", train), ("heldout", heldout)):
        for e in meta:
            src = by_record[e["source_record_index"]]
            assert (e["asset_id"], e["mass"], e["static_friction"]) == (
                src.object_id, src.mass_kg, src.contact_friction)
            assert e["restitution"] == ex.RESTITUTION_PLACEHOLDER
            fp = np.load(out / split / e["fingerprint_path"])["flat_trajectory"]
            assert np.array_equal(fp, src.trajectory)
            assert e["image_path"] == f"images/{e['asset_id']}_v0.png"
    groups = {}
    for e in train + heldout:
        assert groups.setdefault(e["asset_id"], e["image_group"]) == e["image_group"]
    assert len(set(groups.values())) == len(groups)

    from train.data.mujoco_engine import TrajectoryNormalizer
    norm = TrajectoryNormalizer.load(str(out / "train" / "traj_norm.npz"))
    assert (norm.n_actions, norm.timesteps) == (4, 50)
    train_fps = [by_record[e["source_record_index"]].trajectory for e in train]
    refit = TrajectoryNormalizer(4, 50).fit(train_fps, rng=np.random.default_rng(0))
    assert np.allclose(norm.mean, refit.mean) and np.isclose(norm.sigma, refit.sigma)
    all_fps = train_fps + [by_record[e["source_record_index"]].trajectory for e in heldout]
    leaky = TrajectoryNormalizer(4, 50).fit(all_fps, rng=np.random.default_rng(0))
    assert not np.allclose(norm.mean, leaky.mean)
    assert ((out / "train" / "traj_norm.npz").read_bytes()
            == (out / "heldout" / "traj_norm.npz").read_bytes())

    sim = np.load(out / "train" / "similarity_matrix.npz")
    assert sim["pairs"].shape == (55, 2) and sim["similarities"].shape == (55,)
    assert sim["pairs"].max() < 11 and (sim["pairs"][:, 0] < sim["pairs"][:, 1]).all()
    assert np.isfinite(sim["similarities"]).all()
    i, j = sim["pairs"][7]
    assert np.isclose(sim["similarities"][7], norm.similarity(train_fps[i], train_fps[j]))
    assert sim["same_image_group"].sum() == 6 + 3 + 6  # box_a, box_b (one rejected), box_c

    split = json.loads((out / "geometry_splits.json").read_text())
    assert split["heldout"] == ["scene_3__box_d"]
    assert (out / "contact_sheet.png").is_file() and (out / "provenance.json").is_file()
    assert not list(tmp_path.glob(".out.stage-*"))
    if _torch_available():
        assert prov["dataloader_smoke_test"]["train"]["ok"]
        assert prov["dataloader_smoke_test"]["heldout"]["ok"]
    with pytest.raises(ex.ExportError, match="already exists"):
        ex.export_dataset(grid, out, PHINO_ROOT, render_fn=fake_renderer, run_smoke=False)


@needs_phino
def test_export_is_deterministic(tmp_path):
    grid = make_grid(tmp_path)
    for name in ("a", "b"):
        ex.export_dataset(grid, tmp_path / name, PHINO_ROOT, n_holdout=1, seed=3,
                          render_fn=fake_renderer, run_smoke=False, argv=[])
    for split in ex.SPLITS:
        a, b = tmp_path / "a" / split, tmp_path / "b" / split
        assert (a / "metadata.json").read_text() == (b / "metadata.json").read_text()
        for name in ("similarity_matrix.npz", "traj_norm.npz"):
            da, db = np.load(a / name), np.load(b / name)
            assert all(np.array_equal(da[k], db[k]) for k in da.files)


@needs_phino
def test_failed_render_leaves_no_partial_output(tmp_path):
    grid = make_grid(tmp_path)

    def blank_renderer(geom, images_dir, n_views, img_size):
        result = fake_renderer(geom, images_dir, n_views, img_size)
        return ex.RenderResult(result.paths, [0.001] * n_views)

    with pytest.raises(ex.ExportError, match="framing failed"):
        ex.export_dataset(grid, tmp_path / "out", PHINO_ROOT, render_fn=blank_renderer,
                          run_smoke=False)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.stage-*"))


@needs_phino
def test_phino_renderer_frames_isolated_object(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close()
    except Exception as exc:  # no offscreen GL on this host
        pytest.skip(f"MuJoCo offscreen rendering unavailable: {exc}")
    grid = make_grid(tmp_path, objects=("box_a",))
    geom = ex.load_grid(grid).geometries[0]
    engine = ex.load_phino_module(PHINO_ROOT, "train.data.mujoco_engine")
    images = tmp_path / "images"
    images.mkdir()
    result = ex.make_phino_renderer(engine)(geom, images, 2, 224)
    assert [p.name for p in result.paths] == ["scene_0__box_a_v0.png", "scene_0__box_a_v1.png"]
    for path, frac in zip(result.paths, result.object_pixel_fractions):
        ex.validate_image(path, 224)
        assert frac > ex.MIN_OBJECT_PIXEL_FRACTION
