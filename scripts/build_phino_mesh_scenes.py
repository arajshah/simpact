#!/usr/bin/env python3
"""Build controlled, isolated MuJoCo scenes from locally available sample meshes.

Run from the SIMPACT repository root inside its (simpact) environment:

    python scripts/build_phino_mesh_scenes.py --self-test
    python scripts/build_phino_mesh_scenes.py

Then run the unchanged multi-scene PHINO generator:

    python scripts/phino_simpact_pilot_v2.py \
      --scene results/rigid_0103/scene.xml \
      --scene-glob 'results/phino_mesh_scenes_v1/*/scene.xml' \
      --output-dir results/phino_mesh_grid_v1

Only a simple, geometry-controlled synthetic test. It does NOT reconstruct new
objects, estimate real mass/friction, or validate the real-world material physics.

Requires numpy, trimesh, scipy and MuJoCo in the SIMPACT environment. The
self-test exercises mesh handling and XML generation without requiring MuJoCo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

try:
    import trimesh
except ImportError as exc:
    raise SystemExit('Missing dependency: pip install trimesh scipy') from exc

DEFAULT_MESHES = {
    'cube': 'pymeshlab/tests/sample_meshes/cube.obj',
    'bunny': 'pymeshlab/tests/sample_meshes/bunny.obj',
    'cow': 'pymeshlab/tests/sample_meshes/cow.obj',
    'sphere': 'pyvista/examples/sphere.ply',
    'nut': 'pyvista/examples/nut.ply',
}
COLORS = {
    'cube': (0.35, 0.63, 0.87, 1),
    'bunny': (0.85, 0.52, 0.49, 1),
    'cow': (0.61, 0.76, 0.50, 1),
    'sphere': (0.94, 0.78, 0.35, 1),
    'nut': (0.68, 0.57, 0.85, 1),
}


def fmt(values: Any) -> str:
    return ' '.join(f'{float(v):.9g}' for v in values)


def slug(raw: str) -> str:
    if not raw or not raw[0].isalpha() or not all(
        c.isascii() and (c.isalnum() or c == '_') for c in raw
    ):
        raise ValueError(f'Object name must match [A-Za-z][A-Za-z0-9_]*: {raw!r}')
    return raw


def find_default_sources(site_packages_root: Path) -> dict[str, Path]:
    """Allow Python-version-specific .venv/lib/python*/site-packages paths."""
    root = site_packages_root
    if not root.is_dir():
        raise FileNotFoundError(f'Not a directory: {root}')
    # Accept the SIMPACT .venv folder, or the site-packages folder directly.
    if root.name == 'site-packages':
        candidates = [root]
    else:
        candidates = sorted(root.glob('lib/python*/site-packages'))
    if not candidates:
        raise FileNotFoundError(f'No site-packages found beneath {root}')
    result: dict[str, Path] = {}
    for name, relative in DEFAULT_MESHES.items():
        matches = [p / relative for p in candidates if (p / relative).is_file()]
        if not matches:
            raise FileNotFoundError(
                f'Missing mesh for {name}: {relative}. '
                f'Pass --mesh {name}=PATH to supply a different file.')
        result[name] = matches[0].resolve()
    return result


def load_source_mesh(path: Path, *, max_faces: int) -> trimesh.Trimesh:
    if path.suffix.lower() not in {'.obj', '.stl', '.ply', '.glb'}:
        raise ValueError(f'Unsupported mesh source extension: {path.suffix}')
    # trimesh.load(force='mesh') handles OBJ / PLY / STL and some scene files.
    data = trimesh.load(str(path), force='mesh', process=True)
    if not isinstance(data, trimesh.Trimesh):
        raise ValueError(f'Could not convert {path} to a triangle mesh')
    mesh = data.copy()
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3 or len(mesh.faces) < 4:
        raise ValueError(f'Mesh has fewer than four triangles: {path}')
    if len(mesh.faces) > max_faces:
        raise ValueError(f'{path}: {len(mesh.faces)} faces exceed max {max_faces}; '
                         'decimate the mesh or pass --max-visual-faces explicitly')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f'Mesh contains non-finite vertices: {path}')
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) < 4:
        raise ValueError(f'Mesh has fewer than four distinct vertices: {path}')
    return mesh


def normalize_and_hull(mesh: trimesh.Trimesh, max_dim: float):
    """Uniform scale, then move the convex-hull center of mass to body origin.

    Using the hull COM aligns the body's reference frame with the approximate
    physical COM used for the collision surrogate, not the visual bounding box.
    """
    visual = mesh.copy()
    source_bounds = np.asarray(visual.bounds, dtype=float)
    source_size = source_bounds[1] - source_bounds[0]
    if np.min(source_size) <= 1e-10:
        raise ValueError(f'Degenerate source mesh extents: {source_size}')
    bbox_center = (source_bounds[0] + source_bounds[1]) / 2
    scale = float(max_dim / np.max(source_size))
    visual.apply_translation(-bbox_center)
    visual.apply_scale(scale)

    try:
        collision = visual.convex_hull
    except Exception as exc:
        raise ValueError(f'Convex hull construction failed: {exc}') from exc
    if not np.isfinite(collision.volume) or collision.volume <= 1e-12:
        raise ValueError(f'Invalid collision hull volume: {collision.volume}')
    hull_center = np.asarray(collision.center_mass, dtype=float)
    if not np.isfinite(hull_center).all():
        raise ValueError('Nonfinite hull center of mass')
    visual.apply_translation(-hull_center)
    collision.apply_translation(-hull_center)
    collision.fix_normals()
    if not collision.is_watertight:
        raise ValueError('Collision convex hull is not watertight')
    extents = np.asarray(collision.extents, dtype=float)
    if not np.allclose(np.max(extents), max_dim, atol=1e-6):
        raise ValueError(f'Unexpected normalized dimension {extents}')
    return visual, collision, {
        'source_extents': source_size.tolist(),
        'source_bbox_center': bbox_center.tolist(),
        'uniform_scale_factor': scale,
        'center_of_mass_offset_scaled': hull_center.tolist(),
        'normalized_extents_m': extents.tolist(),
        'visual_faces': int(len(visual.faces)),
        'visual_watertight': bool(visual.is_watertight),
        'collision_faces': int(len(collision.faces)),
        'collision_vertices': int(len(collision.vertices)),
        'collision_watertight': bool(collision.is_watertight),
        'collision_volume_m3': float(collision.volume),
        'collision_bounds_m': np.asarray(collision.bounds, dtype=float).tolist(),
    }


def make_xml(name: str, collision: trimesh.Trimesh, *, initial_clearance: float,
             nominal_mass: float, nominal_friction: float) -> ET.Element:
    """One free-jointed rigid body and one plane, as expected by PHINO_generator."""
    name = slug(name)
    root = ET.Element('mujoco', {'model': f'phino_mesh_{name}'})
    ET.SubElement(root, 'compiler', {
        'angle': 'radian', 'inertiafromgeom': 'auto',
        'inertiagrouprange': '1 1', 'meshdir': '.'
    })
    ET.SubElement(root, 'option', {
        'timestep': '0.002', 'gravity': '0 0 -9.81',
        'integrator': 'implicitfast', 'iterations': '100'
    })
    asset = ET.SubElement(root, 'asset')
    # Collision surrogate is convex and watertight. Only it contributes inertia.
    ET.SubElement(asset, 'mesh', {'name': f'{name}_collision',
                                  'file': 'collision.obj', 'inertia': 'convex'})
    ET.SubElement(asset, 'mesh', {'name': f'{name}_visual', 'file': 'visual.obj'})
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', {'pos': '0 0 1.5', 'dir': '0 0 -1.0'})
    ET.SubElement(world, 'geom', {
        'name': 'table', 'type': 'plane', 'size': '1 1 0.1',
        'friction': f'{nominal_friction:.6g} 0.005 0.0001',
        'contype': '1', 'conaffinity': '1', 'rgba': '0.5 0.54 0.57 1'
    })
    spawn_height = -float(collision.bounds[0, 2]) + initial_clearance
    body = ET.SubElement(world, 'body', {
        'name': f'mesh_{name}', 'pos': fmt((0., 0., spawn_height))
    })
    ET.SubElement(body, 'freejoint', {'name': f'mesh_{name}_freejoint'})
    ET.SubElement(body, 'geom', {
        'name': f'mesh_{name}_collision_geom', 'type': 'mesh',
        'mesh': f'{name}_collision', 'group': '1', 'mass': f'{nominal_mass:.7g}',
        'friction': f'{nominal_friction:.6g} 0.005 0.0001',
        'contype': '1', 'conaffinity': '1', 'rgba': '0.1 0.1 0.1 0'
    })
    ET.SubElement(body, 'geom', {
        'name': f'mesh_{name}_visual_geom', 'type': 'mesh',
        'mesh': f'{name}_visual', 'group': '2', 'mass': '0',
        'contype': '0', 'conaffinity': '0',
        'rgba': fmt(COLORS.get(name, (0.64, 0.74, 0.80, 1)))
    })
    return root


def inspect_xml(path: Path, name: str) -> None:
    root = ET.parse(path).getroot()
    body = root.find("./worldbody/body")
    assert body is not None and body.get('name') == f'mesh_{name}'
    assert body.find('freejoint') is not None
    assert len(root.findall("./worldbody/body")) == 1
    assert len(body.findall('geom')) == 2
    assert len(root.findall('./asset/mesh')) == 2
    assert root.find("./worldbody/geom[@name='table']") is not None
    for mesh_node in root.findall('./asset/mesh'):
        assert (path.parent / str(mesh_node.get('file'))).is_file()


def validate_mujoco_scene(xml_file: Path, *, nominal_mass: float,
                          settle_steps: int = 100) -> dict[str, float]:
    """Compile each actual XML, check inertia/geometry/contact and finite steps."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError('MuJoCo not installed. Run in (simpact) or, for '
                           'offline XML generation only, --skip-mujoco-check') from exc
    model = mujoco.MjModel.from_xml_path(str(xml_file))
    if model.nbody != 2 or model.nq != 7 or model.nv != 6:
        raise RuntimeError(f'Unexpected model topology for {xml_file}: '
                           f'nbody={model.nbody}, nq={model.nq}, nv={model.nv}')
    if not np.isclose(model.body_mass[1], nominal_mass, atol=1e-5):
        raise RuntimeError(f'Unexpected mass {model.body_mass[1]} kg in {xml_file}')
    if not (np.all(np.isfinite(model.body_inertia)) and
            np.min(model.body_inertia[1]) > 0):
        raise RuntimeError(f'Nonpositive body inertia in {xml_file}')
    data = mujoco.MjData(model)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f'Nonfinite simulation while settling {xml_file}')
    return {'mujoco_body_mass_kg': float(model.body_mass[1]),
            'settle_duration_s': float(data.time),
            'settle_translation_m': float(np.linalg.norm(data.qpos[:3] - model.qpos0[:3])),
            'settle_z_m': float(data.qpos[2])}


def build_one(name: str, source: Path, out_root: Path, *, max_dim: float,
              initial_clearance: float, nominal_mass: float,
              nominal_friction: float, max_faces: int,
              skip_mujoco_check: bool) -> dict[str, Any]:
    visual_raw = load_source_mesh(source, max_faces=max_faces)
    visual, collision, info = normalize_and_hull(visual_raw, max_dim)
    out = out_root / f'object_{name}'
    out.mkdir(parents=True)
    visual.export(str(out / 'visual.obj'), file_type='obj')
    collision.export(str(out / 'collision.obj'), file_type='obj')
    root = make_xml(name, collision, initial_clearance=initial_clearance,
                    nominal_mass=nominal_mass, nominal_friction=nominal_friction)
    xml_path = out / 'scene.xml'
    ET.ElementTree(root).write(xml_path, encoding='utf-8', xml_declaration=True)
    inspect_xml(xml_path, name)
    result = {
        'name': name, 'body_name': f'mesh_{name}',
        'source_mesh': str(source.resolve()),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scene_relative_path': str(xml_path.relative_to(out_root)),
        'visual_relative_path': str((out / 'visual.obj').relative_to(out_root)),
        'collision_relative_path': str((out / 'collision.obj').relative_to(out_root)),
        **info,
    }
    if not skip_mujoco_check:
        result.update(validate_mujoco_scene(xml_path, nominal_mass=nominal_mass))
    return result


def publish(stage: Path, final: Path, force: bool) -> None:
    if final.exists() and not force:
        raise FileExistsError(f'{final} exists. Use --force to replace generated scenes.')
    backup = final.with_name(final.name + '.previous')
    if backup.exists():
        raise FileExistsError(f'{backup} exists. Inspect/remove it manually first.')
    if final.exists():
        final.rename(backup)
    try:
        stage.rename(final)
    except Exception:
        if backup.exists():
            backup.rename(final)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build_all(sources: dict[str, Path], destination: Path, *, max_dim: float,
              initial_clearance: float, nominal_mass: float,
              nominal_friction: float, max_faces: int,
              skip_mujoco_check: bool, force: bool) -> list[dict[str, Any]]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage at the same filesystem level for atomic rename on success.
    with tempfile.TemporaryDirectory(prefix='.phino_mesh_stage_',
                                     dir=destination.parent) as tmp:
        stage = Path(tmp) / 'scenes'
        stage.mkdir()
        rows = []
        for name, source in sources.items():
            print(f'BUILD {name}: {source}', flush=True)
            info = build_one(name, source, stage, max_dim=max_dim,
                             initial_clearance=initial_clearance,
                             nominal_mass=nominal_mass,
                             nominal_friction=nominal_friction,
                             max_faces=max_faces,
                             skip_mujoco_check=skip_mujoco_check)
            rows.append(info)
            print(f'  OK {name}: faces={info["visual_faces"]}, '
                  f'hull_faces={info["collision_faces"]}, '
                  f'extents={np.round(info["normalized_extents_m"], 4).tolist()} m',
                  flush=True)
        manifest = {
            'dataset_type': 'synthetic_mesh_geometry_validation',
            'generated_from_reconstructed_scenes': False,
            'convex_collision_surrogate': True,
            'simulation_is_not_real_physical_measurement': True,
            'nominal_mass_kg': nominal_mass,
            'nominal_sliding_friction': nominal_friction,
            'max_bounding_box_dimension_m': max_dim,
            'initial_clearance_m': initial_clearance,
            'mujoCo_compile_and_smoke_checked': not skip_mujoco_check,
            'objects': rows,
            'cautions': [
                'All collision meshes are convex hulls; non-convex details do not collide.',
                'Source mesh orientation is preserved. Stability and contact may vary by shape.',
                'Each body has equal nominal mass, but geometry changes its inertia.',
                'The existing PHINO fixed external wrenches are not physical robot grasps.',
                'Object labels are synthetic and not recoverable from static pixels alone.',
            ],
        }
        (stage / 'mesh_scene_manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        publish(stage, destination, force=force)
    return rows


def self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mesh_a = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
        mesh_b = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        p_a, p_b = tmp / 'box.obj', tmp / 'sphere.obj'
        mesh_a.export(p_a)
        mesh_b.export(p_b)
        out = tmp / 'output'
        rows = build_all({'cube': p_a, 'sphere': p_b}, out,
                         max_dim=0.1, initial_clearance=0.002,
                         nominal_mass=0.3, nominal_friction=0.3,
                         max_faces=100000, skip_mujoco_check=True, force=False)
        assert len(rows) == 2
        assert len(list(out.glob('*/scene.xml'))) == 2
        assert all(abs(max(r['normalized_extents_m']) - 0.1) < 1e-6 for r in rows)
        assert all(r['collision_volume_m3'] > 0 for r in rows)
        assert len(json.loads((out / 'mesh_scene_manifest.json').read_text())['objects']) == 2
        for name in ('cube', 'sphere'):
            inspect_xml(out / f'object_{name}' / 'scene.xml', name)
        try:
            build_all({'cube': p_a}, out, max_dim=.1, initial_clearance=.002,
                      nominal_mass=.3, nominal_friction=.3, max_faces=100000,
                      skip_mujoco_check=True, force=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Failed overwrite safety check')
    print('PHINO_MESH_SCENE_SELF_TEST_OK')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true', help='Synthetic mesh/XML tests')
    ap.add_argument('--venv-root', type=Path, default=Path('.venv'),
                    help='SIMPACT venv path (default .venv) or site-packages folder')
    ap.add_argument('--mesh', action='append', default=[], metavar='NAME=PATH',
                    help='Use specified meshes instead of five sample meshes; repeat')
    ap.add_argument('--output-dir', type=Path,
                    default=Path('results/phino_mesh_scenes_v1'))
    ap.add_argument('--max-dimension', type=float, default=0.10,
                    help='Largest object bounding-box dimension in metres')
    ap.add_argument('--initial-clearance', type=float, default=0.002,
                    help='Height of lowest hull vertex above floor at t=0')
    ap.add_argument('--nominal-mass', type=float, default=0.30)
    ap.add_argument('--nominal-friction', type=float, default=0.30)
    ap.add_argument('--max-visual-faces', type=int, default=120000)
    ap.add_argument('--skip-mujoco-check', action='store_true',
                    help='Offline only; cannot ensure compiled scene is valid')
    ap.add_argument('--force', action='store_true',
                    help='Replace previous scene directory only after build succeeds')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    for key, value in [('max-dimension', args.max_dimension),
                       ('initial-clearance', args.initial_clearance),
                       ('nominal-mass', args.nominal_mass),
                       ('nominal-friction', args.nominal_friction)]:
        if not math.isfinite(value) or value <= 0:
            ap.error(f'--{key} must be finite and > 0')
    if args.initial_clearance >= args.max_dimension / 2:
        ap.error('Initial clearance is too large relative to object dimension')
    if args.max_visual_faces < 4:
        ap.error('--max-visual-faces must be at least 4')
    if args.mesh:
        sources = {}
        for spec in args.mesh:
            if '=' not in spec:
                ap.error('--mesh expects NAME=PATH')
            name, raw_path = spec.split('=', 1)
            try:
                name = slug(name)
            except ValueError as exc:
                ap.error(str(exc))
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                ap.error(f'Mesh path does not exist: {path}')
            if name in sources:
                ap.error(f'Duplicate mesh name: {name}')
            sources[name] = path
    else:
        sources = find_default_sources(args.venv_root)
    rows = build_all(sources, args.output_dir, max_dim=args.max_dimension,
                     initial_clearance=args.initial_clearance,
                     nominal_mass=args.nominal_mass,
                     nominal_friction=args.nominal_friction,
                     max_faces=args.max_visual_faces,
                     skip_mujoco_check=args.skip_mujoco_check,
                     force=args.force)
    out = args.output_dir.resolve()
    print(f'SCENES={len(rows)}')
    print(f'SCENE_GLOB={out}/*/scene.xml')
    print(f'MANIFEST={out}/mesh_scene_manifest.json')
    if args.skip_mujoco_check:
        print('WARNING: MuJoCo compilation was not checked')
    print('PHINO_MESH_SCENES_OK')


if __name__ == '__main__':
    main()

