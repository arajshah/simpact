#!/usr/bin/env python3
"""Export a validated SIMPACT four-action grid as a PHINO contrastive dataset.

From the SIMPACT repository root, with the project environment:
    python scripts/export_phino_dataset.py --validate-only
    python scripts/export_phino_dataset.py \
        --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
        --phino-root /nbu/people/ashah264/phino \
        --output-dir data/phino_simpact_v0

Output layout (each split directory is a self-contained PHINO ``data_dir``):
    <output-dir>/
        provenance.json            source hashes, counts, exclusions, mappings, smoke result
        geometry_splits.json       geometry -> split assignment (whole geometries only)
        contact_sheet.png          canonical view of every geometry, for human inspection
        train/                     the ONLY directory to pass to PHINO pretraining
            metadata.json, images/, fingerprints/fp_XXXXXX.npz,
            traj_norm.npz (fit on train only), similarity_matrix.npz
        heldout/                   held-out geometries for later evaluation, same format;
            traj_norm.npz is a byte copy of the train fit (never refit here)

PHINO's contrastive loader samples from every metadata entry it is given, so
held-out geometries are kept out of ``train/metadata.json`` entirely.

Only complete accepted (200,13) fingerprints are exported. Rejected actions are
reported as exclusions; an accepted configuration with a missing or inconsistent
file is a data-integrity error and aborts the export. Physics labels are the
simulator's assigned values, not measurements, and one static image per geometry
cannot reveal which mass/friction assignment produced a trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

import phino_simpact_pilot_v2 as pilot

EXPORT_FORMAT = "simpact_phino_contrastive_export_v0"
SUPPORTED_GRID_FORMATS = ("phino_simpact_four_action_physics_grid_v2_multi_scene",)
EXPECTED_ACTION_ORDER = ("push_x", "push_y", "grasp_lift_release", "lateral_flick")
EXPECTED_FEATURE_ORDER = (
    "[relative_position_xyz, quaternion_wxyz, linear_velocity_xyz, angular_velocity_xyz]")
FRAMES_PER_ACTION = 50
STATE_DIM = 13
FINGERPRINT_SHAPE = (len(EXPECTED_ACTION_ORDER) * FRAMES_PER_ACTION, STATE_DIM)
SPLITS = ("train", "heldout")
RESTITUTION_PLACEHOLDER = 0.0
MIN_OBJECT_PIXEL_FRACTION = 0.02
MIN_IMAGE_STD = 2.0
DEFAULT_PHINO_ROOT = Path("/nbu/people/ashah264/phino")

LABEL_MAPPING = {
    "mass": "SIMPACT record mass_kg: simulator-assigned body mass, not measured.",
    "static_friction": (
        "SIMPACT record contact_friction: MuJoCo sliding coefficient geom_friction[:,0] "
        "set on every collision geom (object and support). PHINO's own engine writes "
        "static_friction into the same slot. Not a measured material coefficient."),
    "dynamic_friction": (
        "Copied from contact_friction. MuJoCo has a single sliding coefficient, so there "
        "is no separate kinetic value; PHINO's 0.8*static convention is a label rule its "
        "engine never simulates and is NOT applied here. PHINO's loader does not read it."),
    "restitution": (
        f"Constant placeholder {RESTITUTION_PLACEHOLDER}. SIMPACT left MuJoCo's default "
        "contact solref (damping ratio 1.0), which PHINO's engine maps from restitution 0 "
        "(dampratio = clip(1 - 0.7 r, 0.3, 1)); time constants differ (0.02 s vs 0.01 s). "
        "Not measured and carries no information; do not train or probe on it."),
}

LIMITATIONS = [
    "Interface-compatibility export only: no PHINO training has been run on these data.",
    "Every geometry has ONE static image shared by all of its mass/friction variants; a "
    "deterministic image encoder cannot identify which assigned physics produced a "
    "trajectory. Same-image pairs (flagged same_image_group) carry contradictory targets.",
    "Physics labels are synthetic simulator assignments, not measured properties.",
    "Exclusions are label-dependent (see provenance.warnings); the exported label "
    "distribution is not the designed grid.",
    "PHINO SoftInfoNCE uses diagonal-only soft labels from the matched pair similarity, "
    "not a dense batchwise dynamics target.",
    "Seven geometries: geometry-held-out results will have very low statistical power.",
]


class ExportError(RuntimeError):
    """Raised when source data, PHINO, rendering, or output validation is invalid."""


@dataclass(frozen=True)
class Geometry:
    object_id: str
    body_name: str
    scene_id: str
    source_scene: str
    source_scene_sha256: str
    isolated_scene: Path
    isolated_scene_sha256: str
    image_group: int


@dataclass(frozen=True)
class Configuration:
    record_index: int
    object_id: str
    mass_kg: float
    contact_friction: float
    source_fingerprint: str
    source_fingerprint_sha256: str
    trajectory: np.ndarray = field(repr=False, compare=False)


@dataclass
class GridInventory:
    grid_dir: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    geometries: list[Geometry]
    accepted: list[Configuration]
    excluded: list[dict[str, Any]]

    def geometry(self, object_id: str) -> Geometry:
        for geom in self.geometries:
            if geom.object_id == object_id:
                return geom
        raise KeyError(object_id)


@dataclass
class RenderResult:
    paths: list[Path]
    object_pixel_fractions: list[float]
    details: dict[str, Any] = field(default_factory=dict)


RenderFn = Callable[[Geometry, Path, int, int], RenderResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _same_value(a: Any, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-9))


def _git_state(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"path": str(path), "commit": run("rev-parse", "HEAD"),
                "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "tracked_changes": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"path": str(path), "commit": None, "error": str(exc)}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except Exception as exc:  # corrupted zip, pickled object arrays, ...
        raise ExportError(f"Unreadable NPZ {path}: {type(exc).__name__}: {exc}") from exc


def _scalar_str(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


# ---------------------------------------------------------------------------
# Source grid loading and validation (no PHINO, no rendering)
# ---------------------------------------------------------------------------
def _check_trajectory(array: np.ndarray, shape: tuple[int, int], where: str) -> None:
    if array.shape != shape:
        raise ExportError(f"{where}: shape {array.shape}, expected {shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ExportError(f"{where}: non-float dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ExportError(f"{where}: non-finite values")
    quat_norm = np.linalg.norm(array[:, 3:7], axis=1)
    if not np.allclose(quat_norm, 1.0, atol=1e-3):
        raise ExportError(f"{where}: columns 3:7 are not unit quaternions "
                          "(feature layout mismatch)")


def _load_geometries(grid_dir: Path, manifest: dict[str, Any]) -> list[Geometry]:
    index = manifest.get("source_object_index")
    if not isinstance(index, list) or not index:
        raise ExportError("Manifest has no source_object_index")
    ids = [entry.get("object_id") for entry in index]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ExportError(f"Missing or duplicate object_id in source_object_index: {ids}")
    geometries = []
    for group, entry in enumerate(index):
        object_id, body = entry["object_id"], entry.get("body_name")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", object_id):
            raise ExportError(f"object_id is not filename-safe: {object_id!r}")
        scene = grid_dir / "isolated_scenes" / f"{pilot._slug(object_id)}.xml"
        if not scene.is_file():
            raise ExportError(f"Missing isolated scene for {object_id}: {scene}")
        try:
            worldbody = ET.parse(scene).getroot().find("worldbody")
        except ET.ParseError as exc:
            raise ExportError(f"Invalid isolated scene XML {scene}: {exc}") from exc
        free = ([b.get("name") for b in worldbody.findall("body")
                 if b.find("freejoint") is not None] if worldbody is not None else [])
        if free != [body]:
            raise ExportError(f"Isolated scene {scene} has free bodies {free}, expected [{body!r}]")
        geometries.append(Geometry(
            object_id=object_id, body_name=body, scene_id=entry.get("scene_id", ""),
            source_scene=entry.get("source_scene", ""),
            source_scene_sha256=entry.get("source_scene_sha256", ""),
            isolated_scene=scene.resolve(), isolated_scene_sha256=sha256_file(scene),
            image_group=group))
    return geometries


def _load_accepted(grid_dir: Path, index: int, record: dict[str, Any],
                   action_order: tuple[str, ...]) -> Configuration:
    where = f"record {index} ({record['object']} m={record['mass_kg']} mu={record['contact_friction']})"
    rel = record.get("fingerprint_path")
    if not rel:
        raise ExportError(f"{where}: accepted without fingerprint_path")
    path = grid_dir / rel
    if not path.is_file():
        raise ExportError(f"{where}: accepted fingerprint file missing: {path}")
    data = _load_npz(path)
    if "flat_trajectory" not in data:
        raise ExportError(f"{where}: {path} has no flat_trajectory")
    flat = data["flat_trajectory"]
    _check_trajectory(flat, FINGERPRINT_SHAPE, f"{where} fingerprint")
    if "action_names" in data and tuple(str(x) for x in data["action_names"]) != action_order:
        raise ExportError(f"{where}: fingerprint action_names {data['action_names']} "
                          f"!= manifest order {action_order}")
    for key, expected in (("mass_kg", record["mass_kg"]),
                          ("contact_friction", record["contact_friction"])):
        if key in data and not _same_value(data[key], expected):
            raise ExportError(f"{where}: fingerprint {key}={data[key]} != record {expected}")
    if "frames_per_action" in data and int(data["frames_per_action"]) != FRAMES_PER_ACTION:
        raise ExportError(f"{where}: fingerprint frames_per_action={data['frames_per_action']}")
    for k, action in enumerate(action_order):
        status = record["action_status"][action]
        trace_path = grid_dir / status.get("trajectory_path", "")
        if not status.get("trajectory_path") or not trace_path.is_file():
            raise ExportError(f"{where}: missing valid {action} trajectory {trace_path}")
        trace = _load_npz(trace_path)
        if "trajectory" not in trace:
            raise ExportError(f"{where}: {trace_path} has no trajectory")
        _check_trajectory(trace["trajectory"], (FRAMES_PER_ACTION, STATE_DIM),
                          f"{where} {action}")
        if ("object_name" in trace and _scalar_str(trace["object_name"]) != record["object"]) or \
           ("action_name" in trace and _scalar_str(trace["action_name"]) != action):
            raise ExportError(f"{where}: {trace_path} object/action identity mismatch")
        for key, expected in (("mass_kg", record["mass_kg"]),
                              ("contact_friction", record["contact_friction"])):
            if key in trace and not _same_value(trace[key], expected):
                raise ExportError(f"{where}: {trace_path} {key}={trace[key]} != {expected}")
        segment = flat[k * FRAMES_PER_ACTION:(k + 1) * FRAMES_PER_ACTION]
        if not np.array_equal(segment, trace["trajectory"]):
            raise ExportError(f"{where}: fingerprint segment {k} does not equal the "
                              f"{action} trajectory (action order or content mismatch)")
    return Configuration(
        record_index=index, object_id=record["object"], mass_kg=float(record["mass_kg"]),
        contact_friction=float(record["contact_friction"]), source_fingerprint=rel,
        source_fingerprint_sha256=sha256_file(path), trajectory=np.array(flat, copy=True))


def load_grid(grid_dir: Path) -> GridInventory:
    """Validate a SIMPACT grid and return only complete accepted fingerprints."""
    grid_dir = grid_dir.expanduser().resolve()
    manifest_path = grid_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ExportError(f"Grid manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"Invalid manifest JSON {manifest_path}: {exc}") from exc

    if manifest.get("format_version") not in SUPPORTED_GRID_FORMATS:
        raise ExportError(f"Unsupported grid format {manifest.get('format_version')!r}")
    action_order = tuple(manifest.get("action_order", ()))
    if action_order != EXPECTED_ACTION_ORDER:
        raise ExportError(f"Manifest action_order {list(action_order)} != expected "
                          f"{list(EXPECTED_ACTION_ORDER)}; normalizer segments would be misaligned")
    if manifest.get("frames_per_action") != FRAMES_PER_ACTION:
        raise ExportError(f"frames_per_action={manifest.get('frames_per_action')}, "
                          f"expected {FRAMES_PER_ACTION}")
    if manifest.get("feature_order") != EXPECTED_FEATURE_ORDER:
        raise ExportError(f"Unexpected feature_order {manifest.get('feature_order')!r}")
    masses = [float(x) for x in manifest.get("mass_grid_kg", [])]
    frictions = [float(x) for x in manifest.get("friction_grid", [])]
    if not masses or not frictions or not np.isfinite(masses + frictions).all():
        raise ExportError("Manifest mass_grid_kg/friction_grid missing or non-finite")

    geometries = _load_geometries(grid_dir, manifest)
    object_ids = {g.object_id for g in geometries}
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ExportError("Manifest has no records list")

    seen: set[tuple[str, float, float]] = set()
    accepted: list[Configuration] = []
    excluded: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        obj = record.get("object")
        if obj not in object_ids:
            raise ExportError(f"record {index}: unknown object {obj!r}")
        mass, mu = float(record["mass_kg"]), float(record["contact_friction"])
        if not any(_same_value(mass, m) for m in masses) or \
           not any(_same_value(mu, f) for f in frictions):
            raise ExportError(f"record {index}: mass/friction ({mass}, {mu}) not in manifest grid")
        key = (obj, round(mass, 9), round(mu, 9))
        if key in seen:
            raise ExportError(f"record {index}: duplicate configuration {key}")
        seen.add(key)
        statuses = record.get("action_status", {})
        if set(statuses) != set(action_order):
            raise ExportError(f"record {index}: action_status keys {sorted(statuses)} "
                              f"!= {sorted(action_order)}")
        all_valid = all(statuses[a].get("valid") is True for a in action_order)
        status = record.get("status")
        if status == "accepted" and all_valid:
            accepted.append(_load_accepted(grid_dir, index, record, action_order))
        elif status == "rejected" and not all_valid:
            if record.get("fingerprint_path"):
                raise ExportError(f"record {index}: rejected but has fingerprint_path")
            failed = [a for a in action_order if statuses[a].get("valid") is not True]
            if "failed_actions" in record and list(record["failed_actions"]) != failed:
                raise ExportError(f"record {index}: failed_actions {record['failed_actions']} "
                                  f"disagree with action_status {failed}")
            excluded.append({
                "record_index": index, "object_id": obj, "mass_kg": mass,
                "contact_friction": mu,
                "failed_actions": [{"action": a, "status": statuses[a].get("status"),
                                    "reason": statuses[a].get("reason", "")} for a in failed],
            })
        else:
            raise ExportError(f"record {index}: status={status!r} inconsistent with "
                              f"per-action validity (all_valid={all_valid})")

    expected = {(g, round(m, 9), round(f, 9)) for g in object_ids for m in masses for f in frictions}
    if seen != expected:
        raise ExportError(f"Records do not cover the declared grid exactly: "
                          f"missing={sorted(expected - seen)} extra={sorted(seen - expected)}")
    on_disk = {p.name for p in (grid_dir / "fingerprints").glob("*.npz")}
    referenced = {Path(c.source_fingerprint).name for c in accepted}
    if on_disk != referenced:
        raise ExportError(f"Fingerprint files and accepted records disagree: "
                          f"unreferenced={sorted(on_disk - referenced)} "
                          f"missing={sorted(referenced - on_disk)}")
    digests: dict[str, int] = {}
    for config in accepted:
        digest = hashlib.sha256(config.trajectory.tobytes()).hexdigest()
        if digest in digests:
            raise ExportError(f"records {digests[digest]} and {config.record_index} have "
                              "byte-identical fingerprints")
        digests[digest] = config.record_index
    for key, value in (("accepted_count", len(accepted)), ("rejected_count", len(excluded)),
                       ("total_count", len(records))):
        if key in manifest and manifest[key] != value:
            raise ExportError(f"Manifest {key}={manifest[key]} but validation found {value}")

    return GridInventory(grid_dir=grid_dir, manifest=manifest,
                         manifest_sha256=sha256_file(manifest_path),
                         geometries=geometries, accepted=accepted, excluded=excluded)


def coverage_report(inv: GridInventory) -> dict[str, Any]:
    """Counts by geometry, physics level, and action; flags label-dependent missingness."""
    masses = [float(x) for x in inv.manifest["mass_grid_kg"]]
    frictions = [float(x) for x in inv.manifest["friction_grid"]]
    per_geom = {g.object_id: {"accepted": 0, "total": len(masses) * len(frictions)}
                for g in inv.geometries}
    per_physics = {f"m{m:.3f}_mu{f:.3f}": {"accepted": 0, "total": len(inv.geometries)}
                   for m in masses for f in frictions}
    for c in inv.accepted:
        per_geom[c.object_id]["accepted"] += 1
        per_physics[f"m{c.mass_kg:.3f}_mu{c.contact_friction:.3f}"]["accepted"] += 1
    failures: dict[str, int] = {a: 0 for a in EXPECTED_ACTION_ORDER}
    for e in inv.excluded:
        for f in e["failed_actions"]:
            failures[f["action"]] += 1
    warnings = []
    for label, values, attr in (("mass_kg", masses, "mass_kg"),
                                ("contact_friction", frictions, "contact_friction")):
        per_level = {v: sum(_same_value(getattr(c, attr), v) for c in inv.accepted) for v in values}
        for value, count in per_level.items():
            if count == 0:
                warnings.append(f"{label}={value} has 0 complete fingerprints; the exported "
                                f"label distribution omits this grid level entirely")
    for object_id, counts in per_geom.items():
        if counts["accepted"] == 0:
            warnings.append(f"{object_id} has 0 complete fingerprints and is absent from metadata")
    return {"total_configurations": len(inv.accepted) + len(inv.excluded),
            "accepted_complete_fingerprints": len(inv.accepted),
            "excluded_configurations": len(inv.excluded),
            "geometries_with_accepted": sum(v["accepted"] > 0 for v in per_geom.values()),
            "per_geometry": per_geom, "per_physics": per_physics,
            "excluded_by_failed_action": failures, "warnings": warnings}


# ---------------------------------------------------------------------------
# Geometry split
# ---------------------------------------------------------------------------
def assign_geometry_split(object_ids: list[str], holdout: list[str] | None,
                          n_holdout: int, seed: int) -> dict[str, Any]:
    """Assign whole geometries to train/heldout before any statistics are computed."""
    ids = sorted(object_ids)
    if holdout:
        unknown = sorted(set(holdout) - set(ids))
        if unknown:
            raise ExportError(f"Unknown --holdout-geometry values: {unknown}")
        if len(set(holdout)) != len(holdout):
            raise ExportError("Duplicate --holdout-geometry values")
        chosen, method = sorted(holdout), "explicit --holdout-geometry list"
    else:
        if not 1 <= n_holdout < len(ids):
            raise ExportError(f"--n-holdout must be in [1, {len(ids) - 1}], got {n_holdout}")
        order = np.random.default_rng(seed).permutation(len(ids))
        chosen = sorted(ids[i] for i in order[:n_holdout])
        method = (f"numpy default_rng({seed}).permutation over sorted geometry IDs; "
                  f"first {n_holdout} held out; chosen before any training or evaluation")
    if len(chosen) >= len(ids):
        raise ExportError("Holdout would leave no training geometries")
    return {"seed": seed, "method": method,
            "assignments": {g: ("heldout" if g in chosen else "train") for g in ids},
            "train": [g for g in ids if g not in chosen], "heldout": chosen}


# ---------------------------------------------------------------------------
# PHINO integration
# ---------------------------------------------------------------------------
def load_phino_module(phino_root: Path, name: str):
    root = phino_root.expanduser().resolve()
    if not (root / "train" / "data" / "mujoco_engine.py").is_file():
        raise ExportError(f"PHINO repository not found or incomplete at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ExportError(f"{name} imported from {module.__file__}, not from {root}")
    return module


def make_phino_renderer(engine_module) -> RenderFn:
    """PHINO render_multiview scene/settle/camera, framed on the object body only.

    PHINO frames on every non-plane geom; the real2sim carton scenes contain a
    1.2 m table box, which shrinks the carton to a few pixels.
    """
    def render(geom: Geometry, images_dir: Path, n_views: int, img_size: int) -> RenderResult:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        from PIL import Image

        eng = engine_module.MuJoCoPhysicsEngine()
        model, data, info = eng.build_scene(str(geom.isolated_scene))
        bid = info["body_id"]
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) != geom.body_name:
            raise ExportError(f"{geom.object_id}: PHINO selected body "
                              f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)!r}")
        mujoco.mj_resetData(model, data)
        data.qpos[info["qadr"] + 2] = eng.spawn_height
        for _ in range(eng.settle_steps):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all():
            raise ExportError(f"{geom.object_id}: non-finite state after settling")

        def in_subtree(body: int) -> bool:
            while body > 0:
                if body == bid:
                    return True
                body = int(model.body_parentid[body])
            return False

        object_geoms = np.array([g for g in range(model.ngeom)
                                 if in_subtree(int(model.geom_bodyid[g]))])
        if object_geoms.size == 0:
            raise ExportError(f"{geom.object_id}: object body has no geoms")
        center = data.xpos[bid].copy()
        radius = max(float(np.linalg.norm(data.geom_xpos[g] - center)) + float(model.geom_rbound[g])
                     for g in object_geoms) or 0.1
        distance = max(0.15, 2.2 * radius)

        paths, fractions = [], []
        renderer = mujoco.Renderer(model, height=img_size, width=img_size)
        try:
            for k in range(n_views):
                cam = mujoco.MjvCamera()
                cam.lookat[:] = center
                cam.distance = distance
                cam.azimuth = 120.0 + k * (360.0 / n_views)
                cam.elevation = -20.0 if k % 2 == 0 else -45.0
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=cam)
                seg = renderer.render()
                renderer.disable_segmentation_rendering()
                is_obj = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], object_geoms)
                path = images_dir / f"{geom.object_id}_v{k}.png"
                Image.fromarray(rgb).save(path)
                paths.append(path)
                fractions.append(float(is_obj.mean()))
        finally:
            renderer.close()
        return RenderResult(paths, fractions, {
            "settled_position": [float(x) for x in center],
            "framing_radius_m": radius, "camera_distance_m": distance})
    return render


def validate_image(path: Path, img_size: int) -> None:
    from PIL import Image
    try:
        with Image.open(path) as img:
            img.load()
            mode, size, arr = img.mode, img.size, np.asarray(img)
    except Exception as exc:
        raise ExportError(f"Unreadable image {path}: {exc}") from exc
    if mode != "RGB" or size != (img_size, img_size):
        raise ExportError(f"Image {path}: mode={mode} size={size}, expected RGB {img_size}x{img_size}")
    if float(arr.std()) < MIN_IMAGE_STD:
        raise ExportError(f"Image {path} is nearly uniform (std={arr.std():.2f})")


# ---------------------------------------------------------------------------
# Writing one split directory
# ---------------------------------------------------------------------------
def _metadata_entry(gidx: int, split: str, config: Configuration, geom: Geometry,
                    view_paths: list[str]) -> dict[str, Any]:
    return {
        "global_idx": gidx,
        "image_group": geom.image_group,
        "asset_id": geom.object_id,
        "category": geom.body_name,
        "material": "unknown",
        "mjcf_path": str(geom.isolated_scene),
        "image_path": view_paths[0],
        "view_paths": view_paths,
        "fingerprint_path": f"fingerprints/fp_{gidx:06d}.npz",
        "mass": config.mass_kg,
        "static_friction": config.contact_friction,
        "dynamic_friction": config.contact_friction,
        "restitution": RESTITUTION_PLACEHOLDER,
        "split": split,
        "contact_friction": config.contact_friction,
        "physics_labels": "synthetic_simulator_assignment",
        "source_record_index": config.record_index,
        "source_fingerprint_path": config.source_fingerprint,
        "source_fingerprint_sha256": config.source_fingerprint_sha256,
    }


def _write_split(split_dir: Path, split: str, configs: list[Configuration],
                 inv: GridInventory, views: dict[str, list[str]]) -> list[dict[str, Any]]:
    (split_dir / "fingerprints").mkdir(parents=True, exist_ok=True)
    metadata = []
    for gidx, config in enumerate(configs):
        geom = inv.geometry(config.object_id)
        entry = _metadata_entry(gidx, split, config, geom, views[geom.object_id])
        np.savez_compressed(
            split_dir / entry["fingerprint_path"], flat_trajectory=config.trajectory,
            mass=config.mass_kg, static_friction=config.contact_friction,
            dynamic_friction=config.contact_friction, restitution=RESTITUTION_PLACEHOLDER,
            mass_kg=config.mass_kg, contact_friction=config.contact_friction,
            action_names=np.asarray(EXPECTED_ACTION_ORDER), asset_id=geom.object_id,
            source_fingerprint_path=config.source_fingerprint)
        metadata.append(entry)
    _write_json(split_dir / "metadata.json", metadata)
    return metadata


def _similarity_pairs(normalizer, trajectories: list[np.ndarray],
                      groups: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(trajectories)
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)],
                     dtype=np.int64).reshape(-1, 2)
    sims = np.array([normalizer.similarity(trajectories[i], trajectories[j]) for i, j in pairs],
                    dtype=np.float32)
    if not np.isfinite(sims).all() or (sims < 0).any() or (sims > 1).any():
        raise ExportError("Similarities must be finite and within [0, 1]")
    same = np.array([groups[i] == groups[j] for i, j in pairs], dtype=bool)
    return pairs, sims, same


def _similarity_stats(sims: np.ndarray, same: np.ndarray) -> dict[str, Any]:
    if sims.size == 0:
        return {"pairs": 0}
    return {"pairs": int(sims.size), "same_image_group_pairs": int(same.sum()),
            "min": float(sims.min()), "median": float(np.median(sims)),
            "mean": float(sims.mean()), "max": float(sims.max()),
            "fraction_below_1e-3": float((sims < 1e-3).mean())}


# ---------------------------------------------------------------------------
# Post-write verification and PHINO dataloader smoke test
# ---------------------------------------------------------------------------
def verify_export(root: Path, inv: GridInventory, split_info: dict[str, Any],
                  normalizer_cls) -> None:
    """Reload every written file and cross-check it against the validated source."""
    by_record = {c.record_index: c for c in inv.accepted}
    seen_records: set[int] = set()
    group_of: dict[str, int] = {}
    split_assets: dict[str, set[str]] = {}
    norm_bytes = None
    for split in SPLITS:
        d = root / split
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        split_assets[split] = {e["asset_id"] for e in meta}
        if not meta:
            raise ExportError(f"{split}: empty metadata")
        for pos, e in enumerate(meta):
            if e["global_idx"] != pos:
                raise ExportError(f"{split}: non-contiguous global_idx at {pos}")
            if split_info["assignments"].get(e["asset_id"]) != split:
                raise ExportError(f"{split}: {e['asset_id']} belongs to another split")
            if group_of.setdefault(e["asset_id"], e["image_group"]) != e["image_group"]:
                raise ExportError(f"{e['asset_id']}: inconsistent image_group")
            src = by_record.get(e["source_record_index"])
            if src is None or src.object_id != e["asset_id"] or e["source_record_index"] in seen_records:
                raise ExportError(f"{split}[{pos}]: bad or duplicate source record mapping")
            seen_records.add(e["source_record_index"])
            if not (_same_value(e["mass"], src.mass_kg)
                    and _same_value(e["static_friction"], src.contact_friction)):
                raise ExportError(f"{split}[{pos}]: physics labels differ from source record")
            fp = _load_npz(d / e["fingerprint_path"])
            if not np.array_equal(fp["flat_trajectory"], src.trajectory):
                raise ExportError(f"{split}[{pos}]: exported fingerprint differs from source")
            for rel in e["view_paths"]:
                if Path(rel).is_absolute() or not (d / rel).is_file():
                    raise ExportError(f"{split}[{pos}]: image path {rel} not inside {d}")
                if not Path(rel).name.startswith(f"{e['asset_id']}_v"):
                    raise ExportError(f"{split}[{pos}]: image {rel} not from {e['asset_id']}")
        norm_path = d / "traj_norm.npz"
        raw = norm_path.read_bytes()
        if norm_bytes is not None and raw != norm_bytes:
            raise ExportError("heldout/traj_norm.npz is not a copy of the train fit")
        norm_bytes = raw
        norm = normalizer_cls.load(str(norm_path))
        if norm.n_actions != len(EXPECTED_ACTION_ORDER) or norm.timesteps != FRAMES_PER_ACTION:
            raise ExportError(f"Normalizer is {norm.n_actions}x{norm.timesteps}, expected 4x50")
        sim = _load_npz(d / "similarity_matrix.npz")
        pairs, sims = sim["pairs"], sim["similarities"]
        if pairs.ndim != 2 or pairs.shape[1] != 2 or sims.shape != (len(pairs),):
            raise ExportError(f"{split}: malformed similarity_matrix.npz")
        if (pairs < 0).any() or (pairs >= len(meta)).any() or (pairs[:, 0] >= pairs[:, 1]).any():
            raise ExportError(f"{split}: similarity pair indices out of range or unordered")
        if len({tuple(p) for p in pairs.tolist()}) != len(pairs):
            raise ExportError(f"{split}: duplicate similarity pairs")
        trajs = [by_record[e["source_record_index"]].trajectory for e in meta]
        redo = np.array([norm.similarity(trajs[i], trajs[j]) for i, j in pairs], dtype=np.float32)
        if not np.allclose(redo, sims, rtol=0, atol=1e-6):
            raise ExportError(f"{split}: stored similarities not reproducible from traj_norm.npz")
    if split_assets["train"] & split_assets["heldout"]:
        raise ExportError("Geometry appears in both train and heldout")
    if len(set(group_of.values())) != len(group_of):
        raise ExportError("Two geometries share an image_group")
    if seen_records != set(by_record):
        raise ExportError(f"Accepted records not exported: {sorted(set(by_record) - seen_records)}")


def smoke_test_dataloader(data_dir: Path, phino_root: Path, batch_size: int = 4,
                          seed: int = 0) -> dict[str, Any]:
    """Fetch one batch through PHINO's real contrastive dataloader (no training)."""
    import random
    import torch

    dataset_mod = load_phino_module(phino_root, "train.data.dataset")
    data_dir = data_dir.resolve()
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    sim = _load_npz(data_dir / "similarity_matrix.npz")
    lookup = {tuple(map(int, p)): float(s) for p, s in zip(sim["pairs"], sim["similarities"])}
    if len(lookup) < 1:
        raise ExportError(f"{data_dir}: no precomputed pairs for the smoke test")
    before = set(p.name for p in data_dir.iterdir())
    random.seed(seed)
    torch.manual_seed(seed)
    loader = dataset_mod.create_contrastive_dataloader(
        data_dir=str(data_dir), batch_size=batch_size, num_workers=0,
        num_pairs=2 * batch_size, hard_neg_ratio=0.0, hard_pos_ratio=0.0,
        random_views=True, seed=seed)
    for i, j, s in loader.dataset.pairs:
        key = (min(i, j), max(i, j))
        if key not in lookup or not np.isclose(s, lookup[key], atol=1e-6):
            raise ExportError(f"Loader pair {key} is not a precomputed similarity pair")
    batch = next(iter(loader))
    required = {"img_i", "img_j", "dynamics_similarity", "physics_i", "physics_j",
                "mass_i", "mass_j", "friction_i", "friction_j", "idx_i", "idx_j"}
    if not required <= set(batch):
        raise ExportError(f"Batch missing keys {sorted(required - set(batch))}")
    for key in ("img_i", "img_j"):
        img = batch[key]
        if tuple(img.shape) != (batch_size, 3, 224, 224) or not torch.isfinite(img).all():
            raise ExportError(f"{key}: shape {tuple(img.shape)} or non-finite values")
        if (img.flatten(1).std(dim=1) <= 0).any():
            raise ExportError(f"{key}: constant image tensor in batch")
    sims = batch["dynamics_similarity"]
    if not torch.isfinite(sims).all() or (sims < 0).any() or (sims > 1).any():
        raise ExportError("Batch dynamics_similarity not finite in [0, 1]")
    for b in range(batch_size):
        i, j = int(batch["idx_i"][b]), int(batch["idx_j"][b])
        if not np.isclose(float(sims[b]), lookup[(min(i, j), max(i, j))], atol=1e-6):
            raise ExportError(f"Batch similarity for ({i},{j}) differs from similarity_matrix")
        for side, k in (("i", i), ("j", j)):
            if not (np.isclose(float(batch[f"mass_{side}"][b]), meta[k]["mass"], atol=1e-6) and
                    np.isclose(float(batch[f"friction_{side}"][b]), meta[k]["static_friction"], atol=1e-6)):
                raise ExportError(f"Batch labels for idx {k} differ from metadata")
            for rel in meta[k]["view_paths"]:
                resolved = dataset_mod.resolve_image_path(data_dir, rel)
                if resolved is None or resolved.resolve() != (data_dir / rel).resolve():
                    raise ExportError(f"PHINO resolves {rel} to {resolved}, not inside {data_dir}")
    created = set(p.name for p in data_dir.iterdir()) - before
    if created:
        raise ExportError(f"Dataloader created unexpected files in {data_dir}: {sorted(created)}")
    return {"data_dir": data_dir.name, "batch_size": batch_size, "num_workers": 0,
            "hard_neg_ratio": 0.0, "hard_pos_ratio": 0.0, "random_views": True,
            "loader_pairs": len(loader.dataset.pairs),
            "image_shape": list(batch["img_i"].shape),
            "idx_pairs": [[int(a), int(b)] for a, b in zip(batch["idx_i"], batch["idx_j"])],
            "dynamics_similarity": [float(x) for x in sims], "ok": True}


# ---------------------------------------------------------------------------
# Export orchestration
# ---------------------------------------------------------------------------
def export_dataset(grid_dir: Path, output_dir: Path, phino_root: Path, *,
                   holdout: list[str] | None = None, n_holdout: int = 2, seed: int = 0,
                   n_views: int = 1, img_size: int = 224, render_fn: RenderFn | None = None,
                   run_smoke: bool = True, smoke_batch_size: int = 4,
                   argv: list[str] | None = None) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ExportError(f"Output directory already exists: {output_dir}; choose a new --output-dir")
    if n_views < 1 or img_size < 32:
        raise ExportError("--n-views must be >= 1 and --img-size >= 32")
    inv = load_grid(grid_dir)
    coverage = coverage_report(inv)
    split_info = assign_geometry_split([g.object_id for g in inv.geometries], holdout,
                                       n_holdout, seed)
    configs = {s: sorted((c for c in inv.accepted if split_info["assignments"][c.object_id] == s),
                         key=lambda c: (inv.geometry(c.object_id).image_group,
                                        c.mass_kg, c.contact_friction))
               for s in SPLITS}
    if len({c.object_id for c in configs["train"]}) < 2:
        raise ExportError("Training split needs accepted data from at least two geometries")
    if not configs["heldout"]:
        raise ExportError("Held-out geometries have no accepted fingerprints; choose another holdout")

    engine_mod = load_phino_module(phino_root, "train.data.mujoco_engine")
    normalizer_cls = engine_mod.TrajectoryNormalizer
    render_fn = render_fn or make_phino_renderer(engine_mod)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        images: dict[str, dict[str, Any]] = {}
        views: dict[str, list[str]] = {}
        for geom in inv.geometries:
            split = split_info["assignments"][geom.object_id]
            img_dir = stage / split / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            result = render_fn(geom, img_dir, n_views, img_size)
            if len(result.paths) != n_views:
                raise ExportError(f"{geom.object_id}: renderer returned {len(result.paths)} views")
            for path, frac in zip(result.paths, result.object_pixel_fractions):
                validate_image(path, img_size)
                if frac < MIN_OBJECT_PIXEL_FRACTION:
                    raise ExportError(f"{path}: object covers {frac:.4f} of pixels "
                                      f"(< {MIN_OBJECT_PIXEL_FRACTION}); framing failed")
            views[geom.object_id] = [f"images/{p.name}" for p in result.paths]
            images[geom.object_id] = {
                "split": split, "image_group": geom.image_group, "view_paths": views[geom.object_id],
                "sha256": [sha256_file(p) for p in result.paths],
                "object_pixel_fraction": result.object_pixel_fractions,
                "accepted_configurations": coverage["per_geometry"][geom.object_id]["accepted"],
                **result.details}

        metadata = {s: _write_split(stage / s, s, configs[s], inv, views) for s in SPLITS}

        rng = np.random.default_rng(seed)
        normalizer = normalizer_cls(n_actions=len(EXPECTED_ACTION_ORDER),
                                    timesteps=FRAMES_PER_ACTION)
        normalizer.fit([c.trajectory for c in configs["train"]], rng=rng)
        if not (np.isfinite(normalizer.mean).all() and np.isfinite(normalizer.std).all()
                and (normalizer.std > 0).all() and np.isfinite(normalizer.sigma)
                and normalizer.sigma > 0):
            raise ExportError("Fitted normalizer has non-finite or non-positive statistics")
        if normalizer.mean.shape != (len(EXPECTED_ACTION_ORDER), STATE_DIM):
            raise ExportError(f"Normalizer mean shape {normalizer.mean.shape}")
        normalizer.save(str(stage / "train" / "traj_norm.npz"))
        shutil.copyfile(stage / "train" / "traj_norm.npz", stage / "heldout" / "traj_norm.npz")
        normalizer = normalizer_cls.load(str(stage / "train" / "traj_norm.npz"))

        similarity: dict[str, Any] = {}
        for s in SPLITS:
            pairs, sims, same = _similarity_pairs(
                normalizer, [c.trajectory for c in configs[s]],
                [e["image_group"] for e in metadata[s]])
            np.savez_compressed(stage / s / "similarity_matrix.npz", pairs=pairs,
                                similarities=sims, same_image_group=same)
            similarity[s] = _similarity_stats(sims, same)

        from PIL import Image
        canon = [np.asarray(Image.open(stage / images[g.object_id]["split"] / views[g.object_id][0]))
                 for g in inv.geometries]
        Image.fromarray(np.concatenate(canon, axis=1)).save(stage / "contact_sheet.png")

        verify_export(stage, inv, split_info, normalizer_cls)
        smoke = ({s: smoke_test_dataloader(stage / s, phino_root, smoke_batch_size, seed)
                  for s in SPLITS} if run_smoke else {"skipped": True})

        _write_json(stage / "geometry_splits.json", {
            **split_info,
            "image_groups": {g.object_id: g.image_group for g in inv.geometries},
            "note": ("Pass ONLY train/ to PHINO pretraining. Normalizer and similarity "
                     "calibration use train/ only; heldout/ reuses the train fit.")})
        provenance = {
            "export_format": EXPORT_FORMAT,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": argv if argv is not None else sys.argv,
            "exporter_sha256": sha256_file(Path(__file__)),
            "simpact_git": _git_state(Path(__file__).resolve().parent.parent),
            "phino_git": _git_state(phino_root.expanduser().resolve()),
            "source_grid": {
                "path": str(inv.grid_dir), "manifest_sha256": inv.manifest_sha256,
                **{k: inv.manifest.get(k) for k in (
                    "format_version", "action_order", "frames_per_action", "sampling_Hz",
                    "feature_order", "action_settings", "max_raw_displacement_m",
                    "legacy_displacement_cutoff_m", "max_tilt_deg", "mass_grid_kg",
                    "friction_grid", "validity_semantics", "contact_friction_semantics",
                    "source_scenes")},
                "isolated_scenes": {g.object_id: {"path": str(g.isolated_scene),
                                                  "sha256": g.isolated_scene_sha256}
                                    for g in inv.geometries}},
            "coverage": coverage,
            "warnings": coverage["warnings"],
            "excluded_configurations": inv.excluded,
            "export_counts": {s: len(configs[s]) for s in SPLITS},
            "seed": seed,
            "geometry_split": split_info,
            "label_mapping": LABEL_MAPPING,
            "rendering": {
                "renderer": ("PHINO MuJoCoPhysicsEngine.build_scene + render_multiview settle "
                             "and camera convention; framing radius restricted to the object "
                             "body's geoms (PHINO default framing includes the carton table)"),
                "img_size": img_size, "n_views_per_geometry": n_views,
                "camera": {"base_azimuth_deg": 120.0, "elevation_deg": "-20 even views, -45 odd",
                           "distance": "max(0.15, 2.2 * object bounding radius)"},
                "min_object_pixel_fraction": MIN_OBJECT_PIXEL_FRACTION,
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "images_per_geometry_not_per_physics": True, "geometries": images},
            "normalization": {
                "class": "train.data.mujoco_engine.TrajectoryNormalizer",
                "n_actions": len(EXPECTED_ACTION_ORDER), "timesteps": FRAMES_PER_ACTION,
                "fit_on": "train split only", "n_fit": len(configs["train"]),
                "rng": f"numpy default_rng({seed})", "sigma": float(normalizer.sigma),
                "heldout_copy": "byte-identical copy of train/traj_norm.npz"},
            "similarity": {
                "method": "all unique unordered pairs (i<j) within each split, train-fit normalizer",
                "extra_array": "same_image_group (bool): pairs sharing one static image",
                **similarity},
            "dataloader_smoke_test": smoke,
            "limitations": LIMITATIONS,
        }
        _write_json(stage / "provenance.json", provenance)
        if output_dir.exists():
            raise ExportError(f"Output directory appeared during export: {output_dir}")
        stage.rename(output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return provenance


def _print_inventory(inv: GridInventory) -> None:
    cov = coverage_report(inv)
    print(f"GRID={inv.grid_dir}")
    print(f"MANIFEST_SHA256={inv.manifest_sha256}")
    print(f"COMPLETE_FINGERPRINTS={cov['accepted_complete_fingerprints']}/"
          f"{cov['total_configurations']}")
    for object_id, c in cov["per_geometry"].items():
        print(f"  {object_id:<55} {c['accepted']}/{c['total']}")
    print("Excluded by failed action: " + ", ".join(
        f"{a}={n}" for a, n in cov["excluded_by_failed_action"].items()))
    for e in inv.excluded:
        reasons = "; ".join(f"{f['action']}: {f['reason']}" for f in e["failed_actions"])
        print(f"  EXCLUDED {e['object_id']} m={e['mass_kg']:.3f} mu={e['contact_friction']:.3f} "
              f"-> {reasons}")
    for warning in cov["warnings"]:
        print(f"WARNING: {warning}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", type=Path, default=Path("results/phino_mesh_grid_cutoff_1m_v1"))
    ap.add_argument("--phino-root", type=Path, default=DEFAULT_PHINO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=Path("data/phino_simpact_v0"))
    ap.add_argument("--holdout-geometry", action="append", default=None,
                    help="Object ID to hold out (repeatable); overrides --n-holdout")
    ap.add_argument("--n-holdout", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seeds the geometry split, normalizer calibration, and smoke test")
    ap.add_argument("--n-views", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--smoke-batch-size", type=int, default=4)
    ap.add_argument("--skip-dataloader-smoke", action="store_true")
    ap.add_argument("--validate-only", action="store_true",
                    help="Validate the grid and print coverage; needs neither PHINO nor a GPU")
    args = ap.parse_args()
    try:
        if args.validate_only:
            _print_inventory(load_grid(args.grid_dir))
            print("GRID_VALIDATION_OK")
            return
        prov = export_dataset(
            args.grid_dir, args.output_dir, args.phino_root, holdout=args.holdout_geometry,
            n_holdout=args.n_holdout, seed=args.seed, n_views=args.n_views,
            img_size=args.img_size, run_smoke=not args.skip_dataloader_smoke,
            smoke_batch_size=args.smoke_batch_size, argv=sys.argv)
    except ExportError as exc:
        sys.exit(f"EXPORT_FAILED: {exc}")
    print(f"OUTPUT={args.output_dir.resolve()}")
    print(f"TRAIN_GEOMETRIES={','.join(prov['geometry_split']['train'])}")
    print(f"HELDOUT_GEOMETRIES={','.join(prov['geometry_split']['heldout'])}")
    print(f"EXPORTED train={prov['export_counts']['train']} heldout={prov['export_counts']['heldout']} "
          f"excluded={len(prov['excluded_configurations'])}")
    for warning in prov["warnings"]:
        print(f"WARNING: {warning}")
    smoke = prov["dataloader_smoke_test"]
    print("DATALOADER_SMOKE=" + ("skipped" if smoke.get("skipped") else "ok"))
    print("INTERFACE_EXPORT_ONLY: no PHINO training; static images cannot identify assigned physics")
    print("PHINO_EXPORT_OK")


if __name__ == "__main__":
    main()
