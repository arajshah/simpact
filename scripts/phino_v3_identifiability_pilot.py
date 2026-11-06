#!/usr/bin/env python3
"""Stage 3: state-only identifiability pilot for the v3 interventions.

Implements docs/phino_v3_identifiability_spec.md section 6 with the approved
amendments (ideal vs 224-px visual precision; toss airborne/mass validation).
No images are rendered, no models are trained, no full grid is written.

    python scripts/phino_v3_identifiability_pilot.py \
        --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
        --phino-root /nbu/people/ashah264/phino \
        --output-dir results/phino_v3_identifiability_pilot_v1

Per geometry it simulates the 3x3 grid, 4 off-grid points, and central
finite-difference neighbours (+-0.05 and +-0.025 in log m and log mu) for every
candidate action, a zero-action control, and a half-timestep rerun of the
centre configuration. Every start uses the v3 common-start protocol (S1-S3).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import export_phino_dataset as export
import phino_simpact_grid_v3 as v3

MASSES = (0.25, 0.30, 0.35)
FRICTIONS = (0.20, 0.30, 0.40)
OFF_GRID = [(m, f) for m in (0.275, 0.325) for f in (0.25, 0.35)]
FD_STEPS = (0.05, 0.025)
CENTER = (0.30, 0.30)
FAMILIES = ("push_x_v2", "push_lo", "push_hi", "toss")
SUBSETS = (("push_x_v2",), ("push_lo",), ("push_hi",), ("toss",), ("push_lo", "push_hi"),
           ("push_hi", "toss"), ("push_lo", "push_hi", "toss"))
GATE_SUBSET = ("push_lo", "push_hi", "toss")
PUSH_FEATURES = ("final_displacement", "peak_speed", "deceleration", "final_yaw")
TOSS_FEATURES = ("apex_height", "apex_time", "final_displacement")
O1_FEATURES = {"push": ("final_displacement", "final_yaw"), "toss": ("final_displacement",)}
OBSERVATIONS = ("O1", "O2", "O4")
PRECISIONS = ("ideal", "visual224", "spec")
REST_SPEED = 0.01
MIN_DECEL_SAMPLES = 3
KAPPA_MAX = 100.0
SD_LOG_MASS_MAX = 0.046
SD_LOG_FRICTION_MAX = 0.072
FD_CONSISTENCY = 0.2
RIDGE_ALPHA = 1.0
M2_RATIO = 0.5
NOISE_REPEATS = 20
IMAGE_PX = 224
CAMERA_MARGIN = 2.8            # visible width = 2 * 1.4 * half-extent (visual pilot framing)
VISUAL_PX_ERROR = 1.0
IDEAL_FACTOR = 1e-3
MIN_TOSS_APEX_PX = 3.0
MIN_AIRBORNE_FRAMES = 2
TOSS_FRICTION_RATIO_MAX = 0.1
DRIFT_S1_MAX_M = 1e-5
SPEC_SIGMA = {"final_displacement": 1e-3, "apex_height": 1e-3, "peak_speed": 0.02,
              "deceleration": 0.1, "apex_time": 0.05, "final_yaw": 1.0}


def config_key(m: float, f: float) -> str:
    return f"m{m:.6f}_mu{f:.6f}"


def build_configs() -> list[dict[str, Any]]:
    configs = [{"key": config_key(m, f), "mass": m, "friction": f, "kind": "grid"}
               for m in MASSES for f in FRICTIONS]
    configs += [{"key": config_key(m, f), "mass": m, "friction": f, "kind": "offgrid"}
                for m, f in OFF_GRID]
    for m, f in [(c["mass"], c["friction"]) for c in configs if c["kind"] == "grid"]:
        for delta in FD_STEPS:
            for param in ("mass", "friction"):
                for sign in (1, -1):
                    mm = m * np.exp(sign * delta) if param == "mass" else m
                    ff = f * np.exp(sign * delta) if param == "friction" else f
                    configs.append({"key": config_key(mm, ff), "mass": float(mm), "friction": float(ff),
                                    "kind": "fd", "center": config_key(m, f), "param": param,
                                    "sign": sign, "delta": delta})
    return configs


# ---------------------------------------------------------------------------
# Simulation worker (one geometry per process)
# ---------------------------------------------------------------------------
def simulate_geometry(job: dict[str, Any]) -> dict[str, Any]:
    import mujoco
    engine_mod = export.load_phino_module(Path(job["phino_root"]), "train.data.mujoco_engine")
    engine = engine_mod.MuJoCoPhysicsEngine()
    originals = {a.name: a for a in engine_mod.DIAGNOSTIC_ACTIONS}
    scene, body = Path(job["scene"]), job["body"]
    ref = v3.make_sim(scene, body, *CENTER, engine, mujoco)
    canonical = v3.canonical_rest_pose(ref, mujoco)
    q_star = canonical["q_star"]
    radius = v3.object_radius(ref, q_star, mujoco)
    starts, zero, rollouts = {}, {}, {}

    def run_config(key: str, mass: float, friction: float, scale: float = 1.0):
        try:
            sim = v3.make_sim(scene, body, mass, friction, engine, mujoco, timestep_scale=scale)
            st = v3.start_state(sim, q_star, engine, mujoco)
        except v3.pilot.RejectedTrial as exc:
            return {"ok": False, "error": str(exc)}, None, {}
        st_out = {k: v for k, v in st.items() if k not in ("q0", "v0")}
        z = v3.rollout(sim, st["q0"], st["v0"], v3.ZERO_ACTION, engine, mujoco, originals,
                       job["max_displacement"], job["max_tilt_deg"])
        if z["valid"]:
            disp = float(np.linalg.norm(z["trace20"][:, :3], axis=1).max())
            q = z["trace20"][-1, 3:7]
            rot = float(np.degrees(2 * np.arccos(np.clip(abs(q[0]), 0, 1))))
            zsum = {"valid": True, "max_displacement_m": disp, "final_rotation_deg": rot,
                    "ok": disp < v3.S3_MAX_DISPLACEMENT_M and rot < v3.S3_MAX_ROTATION_DEG}
        else:
            zsum = {"valid": False, "reason": z["reason"], "ok": False}
        acts = {a.name: v3.rollout(sim, st["q0"], st["v0"], a, engine, mujoco, originals,
                                   job["max_displacement"], job["max_tilt_deg"]) for a in v3.ACTIONS}
        return st_out, zsum, acts

    for c in job["configs"]:
        starts[c["key"]], zero[c["key"]], rollouts[c["key"]] = run_config(c["key"], c["mass"], c["friction"])
    half = run_config("half_dt", *CENTER, scale=0.5)
    return {"object_id": job["object_id"], "canonical": {k: v for k, v in canonical.items() if k != "q_star"},
            "q_star": q_star.tolist(), "radius_m": radius, "starts": starts, "zero": zero,
            "rollouts": rollouts, "half_dt": {"start": half[0], "zero": half[1], "rollouts": half[2]}}


# ---------------------------------------------------------------------------
# Features and precision
# ---------------------------------------------------------------------------
def _parabolic_peak(y: np.ndarray, k: int) -> tuple[float, float]:
    if 0 < k < len(y) - 1:
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        if den < 0:
            off = 0.5 * (a - c) / den
            return float(b - 0.25 * (a - c) * off), float(off)
    return float(y[k]), 0.0


def action_features(trace: np.ndarray, family: str, force_frames: int, per_frame: int, dt: float
                    ) -> dict[str, float]:
    """Pre-declared features from a start-relative trace sampled every ``dt`` seconds."""
    disp = float(np.linalg.norm(trace[-1, :2]))
    if family == "toss":
        k = int(np.argmax(trace[:, 2]))
        apex, off = _parabolic_peak(trace[:, 2], k)
        return {"apex_height": apex, "apex_time": (k + 1 + off) * dt, "final_displacement": disp,
                "n_decel": 0}
    speed = np.linalg.norm(trace[:, 7:9], axis=1)
    first_free = force_frames * per_frame
    stop = next((k for k in range(first_free, len(speed)) if speed[k] < REST_SPEED), len(speed))
    window = np.arange(first_free, stop)
    decel = (float(-np.polyfit((window + 1) * dt, speed[window], 1)[0])
             if len(window) >= MIN_DECEL_SAMPLES else float("nan"))
    q = trace[:, 3:7]
    yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                               1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    return {"final_displacement": disp, "peak_speed": float(speed.max()), "deceleration": decel,
            "final_yaw": float(np.degrees(yaw[-1])), "n_decel": int(len(window))}


def kind_of(family: str) -> str:
    return "toss" if family == "toss" else "push"


def observed_features(family: str, obs: str) -> tuple[str, ...]:
    if obs == "O1":
        return O1_FEATURES[kind_of(family)]
    return TOSS_FEATURES if family == "toss" else PUSH_FEATURES


def sigma(feature: str, precision: str, dt: float, px_per_m: float, radius: float,
          symmetric: bool, n_decel: int) -> float:
    """Measurement standard deviation of one feature under a precision model."""
    if precision == "spec":
        return SPEC_SIGMA[feature]
    pos = VISUAL_PX_ERROR / px_per_m
    if feature in ("final_displacement", "apex_height"):
        s = np.sqrt(2) * pos
    elif feature == "peak_speed":
        s = np.sqrt(2) * pos / dt
    elif feature == "deceleration":
        n = max(n_decel, MIN_DECEL_SAMPLES)
        s = (np.sqrt(2) * pos / dt) / (dt * np.sqrt(n * (n * n - 1) / 12))
    elif feature == "apex_time":
        s = dt
    elif feature == "final_yaw":
        s = float("inf") if symmetric else float(np.degrees(np.sqrt(2) * pos / radius))
    else:
        raise KeyError(feature)
    return s * (IDEAL_FACTOR if precision == "ideal" else 1.0)


class Geometry:
    """Features, validity, and visual calibration for one simulated geometry."""

    def __init__(self, result: dict[str, Any], configs: list[dict[str, Any]], dt: float):
        self.id = result["object_id"]
        self.symmetric = self.id.endswith("__mesh_sphere")
        self.result, self.dt = result, dt
        self.configs = {c["key"]: c for c in configs}
        self.radius = float(result["radius_m"])
        starts, zero = result["starts"], result["zero"]
        self.start_ok = {k: bool(starts[k].get("ok")) and bool(zero[k] and zero[k].get("ok"))
                         for k in self.configs}
        self.feats: dict[tuple[str, str, int], dict[str, float] | None] = {}
        for key, acts in result["rollouts"].items():
            for name, r in acts.items():
                a = v3.ACTION_BY_NAME[name]
                for rate, per in ((20, 1), (100, v3.HIGH_RATE_PER_FRAME)):
                    self.feats[(key, name, rate)] = (
                        action_features(r[f"trace{rate}"], a.family, a.frames, per, dt / per)
                        if r["valid"] else None)
        self.px_per_m: dict[str, float] = {}
        for a in v3.ACTIONS:
            traces = [result["rollouts"][k][a.name]["trace20"] for k, c in self.configs.items()
                      if c["kind"] == "grid" and result["rollouts"].get(k)
                      and result["rollouts"][k][a.name]["valid"]]
            if not traces:
                continue
            pts = np.concatenate([t[:, :3] for t in traces] + [np.zeros((1, 3))])
            spans = [np.ptp(pts[:, 0]), np.ptp(pts[:, 1])]
            if a.family == "toss":
                spans.append(np.ptp(pts[:, 2]))
            self.px_per_m[a.name] = IMAGE_PX / (CAMERA_MARGIN * (max(spans) / 2 + self.radius))

    def valid(self, key: str, action: str) -> bool:
        r = self.result["rollouts"].get(key)
        return bool(r) and r[action]["valid"]

    def vector(self, key: str, actions: tuple[str, ...], obs: str
               ) -> tuple[list[str], np.ndarray, list[tuple[str, str, int]]] | None:
        rate = 100 if obs == "O4" else 20
        names, vals, meta = [], [], []
        for name in actions:
            f = self.feats.get((key, name, rate))
            if f is None:
                return None
            fam = v3.ACTION_BY_NAME[name].family
            for feat in observed_features(fam, obs):
                names.append(f"{name}:{feat}")
                vals.append(f[feat])
                meta.append((name, feat, f["n_decel"]))
        return names, np.array(vals, dtype=float), meta

    def sigmas(self, meta, obs: str, precision: str) -> np.ndarray:
        dt = self.dt / (v3.HIGH_RATE_PER_FRAME if obs == "O4" else 1)
        return np.array([sigma(feat, precision, dt, self.px_per_m.get(name, np.nan), self.radius,
                               self.symmetric, n) for name, feat, n in meta])

    def keys(self, kind: str) -> list[str]:
        return [k for k, c in self.configs.items() if c["kind"] == kind]

    def fd_key(self, center: str, param: str, sign: int, delta: float) -> str:
        for k, c in self.configs.items():
            if (c["kind"] == "fd" and c["center"] == center and c["param"] == param
                    and c["sign"] == sign and np.isclose(c["delta"], delta)):
                return k
        raise KeyError((center, param, sign, delta))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def fisher(J: np.ndarray, s: np.ndarray) -> dict[str, float]:
    keep = np.isfinite(s) & (s > 0)
    A = J[keep] / s[keep, None]
    if A.shape[0] == 0:
        return {"kappa": float("inf"), "sd_log_mass": float("inf"), "sd_log_friction": float("inf")}
    sv = np.linalg.svd(A, compute_uv=False)
    kappa = float(sv[0] / sv[1]) if len(sv) > 1 and sv[1] > 0 else float("inf")
    F = A.T @ A
    if np.linalg.cond(F) > 1e12:
        sd = (float("inf"), float("inf"))
    else:
        cov = np.linalg.inv(F)
        sd = tuple(float(np.sqrt(max(cov[i, i], 0))) for i in range(2))
    return {"kappa": kappa, "sd_log_mass": sd[0], "sd_log_friction": sd[1]}


def local_identifiability(g: Geometry, actions: tuple[str, ...], obs: str, precision: str
                          ) -> list[dict[str, Any]]:
    """M1 (with FD consistency) and M4 at every grid point."""
    pool = g.keys("grid") + g.keys("offgrid")
    rows = []
    for center in g.keys("grid"):
        row = {"point": center, "status": "ok"}
        evals = {}
        for delta in FD_STEPS:
            keys = [center] + [g.fd_key(center, p, s, delta) for p in ("mass", "friction") for s in (1, -1)]
            vecs = [g.vector(k, actions, obs) for k in keys]
            if any(v is None for v in vecs) or not all(g.start_ok[k] for k in keys):
                row["status"] = "invalid_or_unverified_start"
                break
            evals[delta] = vecs
        if row["status"] != "ok":
            rows.append(row)
            continue
        names, _, meta = evals[FD_STEPS[0]][0]
        vals = {d: np.array([v[1] for v in vecs]) for d, vecs in evals.items()}
        keep = np.all([np.isfinite(vals[d]).all(axis=0) for d in FD_STEPS], axis=0)
        if not keep.any():
            row["status"] = "no_finite_features"
            rows.append(row)
            continue
        s = g.sigmas([m for m, k in zip(meta, keep) if k], obs, precision)
        J = {d: np.column_stack([(vals[d][1] - vals[d][2])[keep] / (2 * d),
                                 (vals[d][3] - vals[d][4])[keep] / (2 * d)]) for d in FD_STEPS}
        res = fisher(J[FD_STEPS[0]], s)
        fin = np.isfinite(s)
        a1, a2 = J[FD_STEPS[0]][fin] / s[fin, None], J[FD_STEPS[1]][fin] / s[fin, None]
        norm = np.linalg.norm(a1)
        flagged = bool(norm == 0 or np.linalg.norm(a1 - a2) / norm > FD_CONSISTENCY)
        f_center = vals[FD_STEPS[0]][0][keep]
        confusable = 0
        for other in pool:
            if other == center or not g.start_ok[other]:
                continue
            ov = g.vector(other, actions, obs)
            if ov is None:
                continue
            diff = (ov[1][keep] - f_center) / s
            diff = diff[np.isfinite(diff)]
            if diff.size and np.linalg.norm(diff) < 1.0:
                confusable += 1
        mass_ok = res["sd_log_mass"] <= SD_LOG_MASS_MAX
        fric_ok = res["sd_log_friction"] <= SD_LOG_FRICTION_MAX
        both = res["kappa"] < KAPPA_MAX and mass_ok and fric_ok
        rescue = flagged and confusable == 0
        row.update(res, n_features=int(keep.sum()), dropped_features=[n for n, k in zip(names, keep) if not k],
                   fd_flagged=flagged, confusable=confusable,
                   mass_identified=bool(mass_ok or rescue), friction_identified=bool(fric_ok or rescue),
                   both_identified=bool(both or rescue))
        rows.append(row)
    return rows


def _ridge(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd < 1e-12] = 1.0
    ztr, zte = (xtr - mu) / sd, (xte - mu) / sd
    ym = ytr.mean(0)
    w = np.linalg.solve(ztr.T @ ztr + alpha * np.eye(ztr.shape[1]), ztr.T @ (ytr - ym))
    return ym + zte @ w


def _design(g: Geometry, keys: list[str], actions, obs) -> tuple[np.ndarray, np.ndarray, list, list[str]]:
    rows, ys, used, meta = [], [], [], None
    for k in keys:
        v = g.vector(k, actions, obs)
        if v is None or not g.start_ok[k]:
            continue
        rows.append(v[1])
        meta = v[2]
        c = g.configs[k]
        ys.append([np.log(c["mass"]), np.log(c["friction"])])
        used.append(k)
    return (np.array(rows), np.array(ys), meta, used) if rows else (np.empty((0, 0)), np.empty((0, 2)), [], [])


def held_out_prediction(g: Geometry, actions: tuple[str, ...], obs: str, precision: str,
                        seed: int) -> dict[str, Any]:
    """M2: ridge on standardized features; train 3x3 grid, test 4 off-grid points."""
    xtr, ytr, meta, _ = _design(g, g.keys("grid"), actions, obs)
    xte, yte, _, _ = _design(g, g.keys("offgrid"), actions, obs)
    if len(xtr) < 6 or len(xte) < 2:
        return {"status": "insufficient_valid_rows", "n_train": len(xtr), "n_test": len(xte), "pass": False}
    keep = np.isfinite(xtr).all(0) & np.isfinite(xte).all(0)
    if not keep.any():
        return {"status": "no_finite_features", "pass": False}
    xtr, xte = xtr[:, keep], xte[:, keep]
    s = g.sigmas([m for m, k in zip(meta, keep) if k], obs, precision)
    fin = np.isfinite(s)
    xtr, xte, s = xtr[:, fin], xte[:, fin], s[fin]
    if xtr.shape[1] == 0:
        return {"status": "no_observable_features", "pass": False}
    rng = np.random.default_rng(seed)
    reps = 1 if precision == "ideal" else NOISE_REPEATS
    errs = []
    for _ in range(reps):
        ntr = xtr + (rng.normal(size=xtr.shape) * s if precision != "ideal" else 0)
        nte = xte + (rng.normal(size=xte.shape) * s if precision != "ideal" else 0)
        errs.append(np.abs(_ridge(ntr, ytr, nte) - yte).mean(0))
    mae = np.mean(errs, axis=0)
    base = np.abs(yte - ytr.mean(0)).mean(0)
    ratio = mae / base
    return {"status": "ok", "n_train": len(xtr), "n_test": len(xte), "n_features": int(xtr.shape[1]),
            "mae_log_mass": float(mae[0]), "mae_log_friction": float(mae[1]),
            "baseline_log_mass": float(base[0]), "baseline_log_friction": float(base[1]),
            "ratio_mass": float(ratio[0]), "ratio_friction": float(ratio[1]),
            "pass_mass": bool(ratio[0] <= M2_RATIO), "pass_friction": bool(ratio[1] <= M2_RATIO),
            "pass": bool((ratio <= M2_RATIO).all())}


def full_trajectory_prediction(g: Geometry, actions: tuple[str, ...]) -> dict[str, Any]:
    """M2 with O3 (entire 20 Hz start-relative traces), ideal precision only."""
    def xy(keys):
        X, Y = [], []
        for k in keys:
            if not g.start_ok[k] or not all(g.valid(k, a) for a in actions):
                continue
            X.append(np.concatenate([g.result["rollouts"][k][a]["trace20"].ravel() for a in actions]))
            c = g.configs[k]
            Y.append([np.log(c["mass"]), np.log(c["friction"])])
        return np.array(X), np.array(Y)
    xtr, ytr = xy(g.keys("grid"))
    xte, yte = xy(g.keys("offgrid"))
    if len(xtr) < 6 or len(xte) < 2:
        return {"status": "insufficient_valid_rows", "pass": False}
    mae = np.abs(_ridge(xtr, ytr, xte) - yte).mean(0)
    ratio = mae / np.abs(yte - ytr.mean(0)).mean(0)
    return {"status": "ok", "ratio_mass": float(ratio[0]), "ratio_friction": float(ratio[1]),
            "pass": bool((ratio <= M2_RATIO).all())}


def cross_geometry(geoms: list[Geometry], actions: tuple[str, ...], obs: str, precision: str,
                   seed: int) -> list[dict[str, Any]]:
    """M3: leave one non-sphere geometry out; sphere tested against all non-sphere training."""
    data = {}
    for g in geoms:
        x, y, meta, _ = _design(g, g.keys("grid") + g.keys("offgrid"), actions, obs)
        if len(x):
            data[g.id] = (x, y, g.sigmas(meta, obs, precision))
    out = []
    non_sphere = [g.id for g in geoms if not g.symmetric and g.id in data]
    rng = np.random.default_rng(seed)
    for test in data:
        train = [k for k in non_sphere if k != test]
        if len(train) < 2:
            continue
        xtr = np.concatenate([data[k][0] for k in train])
        ytr = np.concatenate([data[k][1] for k in train])
        xte, yte = data[test][0], data[test][1]
        keep = np.isfinite(xtr).all(0) & np.isfinite(xte).all(0)
        if precision != "ideal":
            keep &= np.isfinite(data[test][2])
        if not keep.any():
            continue
        reps = 1 if precision == "ideal" else NOISE_REPEATS
        errs = []
        for _ in range(reps):
            noise = (lambda x, k: rng.normal(size=x.shape) * data[k][2][keep]) if precision != "ideal" \
                else (lambda x, k: 0)
            ntr = np.concatenate([data[k][0][:, keep] + noise(data[k][0][:, keep], k) for k in train])
            nte = xte[:, keep] + noise(xte[:, keep], test)
            errs.append(np.abs(_ridge(ntr, ytr, nte) - yte).mean(0))
        mae = np.mean(errs, axis=0)
        base = np.abs(yte - ytr.mean(0)).mean(0)
        out.append({"test_geometry": test, "n_train_geometries": len(train),
                    "ratio_mass": float(mae[0] / base[0]), "ratio_friction": float(mae[1] / base[1])})
    return out


def toss_validation(g: Geometry, toss: str, seed: int) -> dict[str, Any]:
    """Amendment 2: measurable airborne response, independently informative about mass."""
    grid = g.keys("grid")
    valid = [k for k in grid if g.valid(k, toss) and g.start_ok[k]]
    if not valid:
        return {"status": "no_valid_toss", "independently_informative_about_mass": False}
    air = [g.result["rollouts"][k][toss]["airborne_frames"] for k in valid]
    apex = [g.feats[(k, toss, 20)]["apex_height"] for k in valid]
    apex_px = [a * g.px_per_m[toss] for a in apex]
    ratios = []
    for c in grid:
        try:
            ks = [g.fd_key(c, p, s, FD_STEPS[0]) for p in ("mass", "friction") for s in (1, -1)]
        except KeyError:
            continue
        if not all(g.valid(k, toss) for k in ks):
            continue
        a = [np.log(max(g.feats[(k, toss, 20)]["apex_height"], 1e-9)) for k in ks]
        dm, dmu = (a[0] - a[1]) / (2 * FD_STEPS[0]), (a[2] - a[3]) / (2 * FD_STEPS[0])
        ratios.append(abs(dmu) / max(abs(dm), 1e-12))
    m2 = {p: held_out_prediction(g, (toss,), "O2", p, seed) for p in ("ideal", "visual224")}
    checks = {
        "airborne_all_valid_grid": bool(min(air) >= MIN_AIRBORNE_FRAMES),
        "apex_visible_all_valid_grid": bool(min(apex_px) >= MIN_TOSS_APEX_PX),
        "friction_insensitive": bool(ratios and np.median(ratios) <= TOSS_FRICTION_RATIO_MAX),
        "mass_predictive_ideal": bool(m2["ideal"].get("pass_mass", False)),
        "mass_predictive_visual224": bool(m2["visual224"].get("pass_mass", False)),
    }
    return {"status": "ok", "n_valid_grid": len(valid), "airborne_frames_min": int(min(air)),
            "apex_mm_range": [1000 * min(apex), 1000 * max(apex)],
            "apex_px_min": float(min(apex_px)), "px_per_m": g.px_per_m[toss],
            "median_friction_to_mass_sensitivity": float(np.median(ratios)) if ratios else None,
            "m2_ideal": m2["ideal"], "m2_visual224": m2["visual224"], "checks": checks,
            "independently_informative_about_mass": bool(all(checks.values()))}


def response_exponents(g: Geometry, action: str) -> dict[str, Any] | None:
    """Grid log-log fit of the primary response (push: final displacement; toss: apex)."""
    fam = v3.ACTION_BY_NAME[action].family
    feat = "apex_height" if fam == "toss" else "final_displacement"
    rows = []
    for k in g.keys("grid"):
        f = g.feats.get((k, action, 20))
        if f is not None and g.start_ok[k] and f[feat] > 1e-6:
            c = g.configs[k]
            rows.append((np.log(c["mass"]), np.log(c["friction"]), np.log(f[feat])))
    if len(rows) < 4:
        return None
    X = np.array(rows)
    coef = np.linalg.lstsq(np.c_[np.ones(len(X)), X[:, :2]], X[:, 2], rcond=None)[0]
    return {"feature": feat, "mass_exponent": float(coef[1]), "friction_exponent": float(coef[2]),
            "n": len(rows)}


def direction_angle(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    u = np.array([a["mass_exponent"], a["friction_exponent"]])
    w = np.array([b["mass_exponent"], b["friction_exponent"]])
    return float(np.degrees(np.arccos(np.clip(abs(u @ w) / (np.linalg.norm(u) * np.linalg.norm(w)), 0, 1))))


def write_figures(out: Path, geoms: list[Geometry], verdict_rows: list[dict[str, Any]],
                  exponents: dict[str, dict[str, Any]], choice: dict[str, str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = {g.id: g.id.split("__")[-1].replace("_milk_carton", "").replace("mesh_", "")
             .replace("white_coconut", "white") for g in geoms}
    # 1. identifiability matrix for the post-hoc choice
    rows_lbl = [(s, o) for s in ["+".join(x) for x in SUBSETS] for o in OBSERVATIONS]
    code = {(False, False): 0, (True, False): 1, (False, True): 2, (True, True): 3}
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
    for ax, prec in zip(axes, ("ideal", "visual224")):
        M = np.full((len(rows_lbl), len(geoms)), np.nan)
        for i, (s, o) in enumerate(rows_lbl):
            for j, g in enumerate(geoms):
                r = [v for v in verdict_rows if v["choice"] == "post_hoc_max_validity" and v["subset"] == s
                     and v["obs"] == o and v["precision"] == prec and v["geometry"] == g.id]
                if r:
                    M[i, j] = 3 if r[0]["both"] else code[(r[0]["mass"], r[0]["friction"])]
        cmap = matplotlib.colors.ListedColormap(["#dddddd", "#4c9be8", "#f2a541", "#2e9e4f"])
        ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
        ax.set_xticks(range(len(geoms)), [short[g.id] for g in geoms], rotation=45, fontsize=8)
        ax.set_yticks(range(len(rows_lbl)), [f"{s} [{o}]" for s, o in rows_lbl], fontsize=7)
        ax.set_title(f"{prec}: grey none, blue mass, orange friction, green both", fontsize=9)
    fig.suptitle(f"Stage 3 identifiability (M1 >= 8/9 and M2 <= 0.5), post-hoc choice {choice}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "identifiability_matrix.png", dpi=110)
    plt.close(fig)
    # 2. sensitivity directions
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"push_x_v2": "grey", "push_lo": "tab:blue", "push_hi": "tab:red", "toss": "tab:green"}
    for g in geoms:
        for fam in FAMILIES:
            e = exponents.get(g.id, {}).get(choice[fam])
            if e:
                x, y = e["mass_exponent"], e["friction_exponent"]
                ax.plot([0, x], [0, y], color=colors[fam], alpha=0.6)
                ax.plot([x], [y], "o", color=colors[fam], ms=3)
                ax.text(x, y, short[g.id], fontsize=6)
    for fam, c in colors.items():
        ax.plot([], [], color=c, label=f"{fam} ({choice[fam]})")
    ax.axhline(0, lw=0.5, color="k")
    ax.axvline(0, lw=0.5, color="k")
    ax.set_xlabel("d log response / d log mass")
    ax.set_ylabel("d log response / d log friction")
    ax.set_title("Response sensitivity directions (grid fit; push: final displacement, toss: apex)",
                 fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_directions.png", dpi=110)
    plt.close(fig)
    # 3. toss apex vs mass
    toss = choice["toss"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in geoms:
        pts = [(g.configs[k]["mass"], g.configs[k]["friction"], g.feats[(k, toss, 20)]["apex_height"])
               for k in g.keys("grid") + g.keys("offgrid") if g.feats.get((k, toss, 20)) and g.start_ok[k]]
        if pts:
            p = np.array(pts)
            ax.scatter(p[:, 0], 1000 * p[:, 2], s=10 + 60 * (p[:, 1] - 0.15), label=short[g.id], alpha=0.6)
    ax.set_xlabel("mass (kg)")
    ax.set_ylabel(f"{toss} apex height (mm)")
    ax.set_title("Toss apex vs mass (marker size = friction; geometries overlap); verified starts only",
                 fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "toss_apex_vs_mass.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Candidate selection (validity only)
# ---------------------------------------------------------------------------
def validity_table(geoms: list[Geometry]) -> dict[str, dict[str, Any]]:
    table = {}
    for a in v3.ACTIONS:
        per_geom = {}
        for g in geoms:
            grid = g.keys("grid")
            ok = [g.valid(k, a.name) for k in grid]
            air = [g.result["rollouts"][k][a.name].get("airborne_frames", 0) >= MIN_AIRBORNE_FRAMES
                   for k in grid if g.valid(k, a.name)]
            per_geom[g.id] = {"valid": int(sum(ok)), "total": len(grid), "all_valid_airborne": all(air) if air else False,
                              "reasons": sorted({(g.result["rollouts"].get(k) or {}).get(
                                  a.name, {"reason": "start failed"})["reason"].split(" at ")[0][:40]
                                  for k in grid if not g.valid(k, a.name)})}
        total = sum(v["valid"] for v in per_geom.values())
        n = sum(v["total"] for v in per_geom.values())
        table[a.name] = {"family": a.family, "magnitude": a.magnitude, "valid": total, "total": n,
                         "per_geometry": per_geom}
    return table


def select(table: dict[str, dict[str, Any]], geometries: list[str] | None = None) -> dict[str, str | None]:
    """Pre-registered rule over the given geometries (default: all)."""
    choice: dict[str, str | None] = {"push_x_v2": "push_x_v2"}
    for fam in ("push_lo", "push_hi", "toss"):
        cands = [a for a in v3.ACTIONS if a.family == fam]
        ok = []
        for a in cands:
            pg = {k: v for k, v in table[a.name]["per_geometry"].items()
                  if geometries is None or k in geometries}
            frac = sum(v["valid"] for v in pg.values()) / sum(v["total"] for v in pg.values())
            rule = frac >= 0.95 and all(v["valid"] >= 8 for v in pg.values())
            if fam == "toss":
                rule = rule and all(v["all_valid_airborne"] for v in pg.values())
            if rule:
                ok.append(a)
        if not ok:
            choice[fam] = None
        else:
            pick = min(ok, key=lambda a: a.magnitude) if fam == "toss" else max(ok, key=lambda a: a.magnitude)
            choice[fam] = pick.name
    return choice


def fallback_select(table: dict[str, dict[str, Any]], geometries: list[str]) -> dict[str, str]:
    """POST HOC, validity-only: per family the candidate valid in the most non-sphere grid runs."""
    choice = {"push_x_v2": "push_x_v2"}
    for fam in ("push_lo", "push_hi", "toss"):
        cands = [a for a in v3.ACTIONS if a.family == fam]

        def score(a):
            pg = table[a.name]["per_geometry"]
            return (sum(pg[g]["valid"] for g in geometries),
                    a.magnitude if fam != "toss" else -a.magnitude)
        choice[fam] = max(cands, key=score).name
    return choice


# ---------------------------------------------------------------------------
def evaluate_choice(geoms: list[Geometry], choice: dict[str, str | None], seed: int
                    ) -> dict[str, Any]:
    results = {}
    for subset in SUBSETS:
        if any(choice.get(f) is None for f in subset):
            continue
        actions = tuple(choice[f] for f in subset)
        label = "+".join(subset)
        for obs in OBSERVATIONS:
            for precision in PRECISIONS:
                for g in geoms:
                    pts = local_identifiability(g, actions, obs, precision)
                    m2 = held_out_prediction(g, actions, obs, precision, seed)
                    results[(label, obs, precision, g.id)] = {"points": pts, "m2": m2}
        for g in geoms:
            results[(label, "O3", "ideal", g.id)] = {"points": [], "m2": full_trajectory_prediction(g, actions)}
    return results


def geometry_verdict(entry: dict[str, Any], m1_needed: int = 8, m2_ratio: float = M2_RATIO
                     ) -> dict[str, Any]:
    pts = entry["points"]
    m2 = entry["m2"]
    n = lambda key: sum(1 for p in pts if p.get(key))  # noqa: E731
    m2_mass = m2.get("status") == "ok" and m2["ratio_mass"] <= m2_ratio
    m2_fric = m2.get("status") == "ok" and m2["ratio_friction"] <= m2_ratio
    has_points = bool(pts)
    return {"m1_both": n("both_identified"), "m1_mass": n("mass_identified"),
            "m1_friction": n("friction_identified"),
            "m2_mass": bool(m2_mass), "m2_friction": bool(m2_fric),
            "mass": bool((not has_points or n("mass_identified") >= m1_needed) and m2_mass),
            "friction": bool((not has_points or n("friction_identified") >= m1_needed) and m2_fric),
            "both": bool((not has_points or n("both_identified") >= m1_needed) and m2_mass and m2_fric)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", type=Path, default=Path("results/phino_mesh_grid_cutoff_1m_v1"))
    ap.add_argument("--phino-root", type=Path, default=export.DEFAULT_PHINO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=Path("results/phino_v3_identifiability_pilot_v1"))
    ap.add_argument("--geometry", action="append", default=None, help="Object ID substring (repeatable)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.output_dir.expanduser().resolve()
    if out.exists():
        sys.exit(f"Output directory already exists: {out}")
    inv = export.load_grid(args.grid_dir)
    geoms_src = [g for g in inv.geometries
                 if not args.geometry or any(s in g.object_id for s in args.geometry)]
    engine_mod = export.load_phino_module(args.phino_root, "train.data.mujoco_engine")
    dt = float(engine_mod.MuJoCoPhysicsEngine().frame_dt)
    configs = build_configs()
    jobs = [{"object_id": g.object_id, "scene": str(g.isolated_scene), "body": g.body_name,
             "phino_root": str(args.phino_root), "configs": configs,
             "max_displacement": float(inv.manifest["max_raw_displacement_m"]),
             "max_tilt_deg": float(inv.manifest["max_tilt_deg"])} for g in geoms_src]
    with mp.get_context("fork").Pool(min(args.workers, len(jobs))) as pool:
        raw = pool.map(simulate_geometry, jobs)

    geoms = [Geometry(r, configs, dt) for r in raw]
    non_sphere = [g.id for g in geoms if not g.symmetric]
    table = validity_table(geoms)
    choices = {"preregistered_all_geometries": select(table),
               "preregistered_non_sphere": select(table, non_sphere),
               "post_hoc_max_validity": fallback_select(table, non_sphere)}

    s1 = {g.id: {**g.result["canonical"],
                 "converged_drift_reading": bool(g.result["canonical"]["final_window_drift_m"] <= DRIFT_S1_MAX_M),
                 "configs_start_ok": int(sum(g.start_ok.values())), "configs_total": len(g.start_ok),
                 "s2_failures": int(sum(not g.result["starts"][k].get("ok") for k in g.configs)),
                 "s3_failures": int(sum(not (g.result["zero"][k] or {}).get("ok") for k in g.configs))}
          for g in geoms}
    for g in geoms:
        s1[g.id]["s1_s3_pass_strict"] = bool(s1[g.id]["converged"] and s1[g.id]["configs_start_ok"] == len(g.configs))
        s1[g.id]["s1_s3_pass_drift"] = bool(s1[g.id]["converged_drift_reading"]
                                            and s1[g.id]["configs_start_ok"] == len(g.configs))

    evaluations, verdict_rows, point_rows, m2_rows = {}, [], [], []
    combos = {"post_hoc_max_validity": choices["post_hoc_max_validity"]}
    for lo, hi, toss in itertools.product(*[[a.name for a in v3.ACTIONS if a.family == f]
                                             for f in ("push_lo", "push_hi", "toss")]):
        combos.setdefault(f"combo:{lo}+{hi}+{toss}", {"push_x_v2": "push_x_v2", "push_lo": lo,
                                                      "push_hi": hi, "toss": toss})
    for name in ("preregistered_all_geometries", "preregistered_non_sphere"):
        if all(choices[name].values()):
            combos[name] = choices[name]
    for cname, choice in combos.items():
        full = cname == "post_hoc_max_validity"
        if full:
            res = evaluate_choice(geoms, choice, args.seed)
        else:   # gate subset, O2 only, for gate sensitivity across candidate combinations
            actions = tuple(choice[f] for f in GATE_SUBSET)
            res = {}
            for precision in PRECISIONS:
                for g in geoms:
                    res[("+".join(GATE_SUBSET), "O2", precision, g.id)] = {
                        "points": local_identifiability(g, actions, "O2", precision),
                        "m2": held_out_prediction(g, actions, "O2", precision, args.seed)}
        evaluations[cname] = res
        for (label, obs, precision, gid), entry in res.items():
            v = geometry_verdict(entry)
            verdict_rows.append({"choice": cname, "subset": label, "obs": obs, "precision": precision,
                                 "geometry": gid, **v,
                                 **{f"m2_{k}": entry["m2"].get(k) for k in ("ratio_mass", "ratio_friction", "status")}})
            if full:
                for p in entry["points"]:
                    point_rows.append({"subset": label, "obs": obs, "precision": precision, "geometry": gid,
                                       **{k: (json.dumps(v2) if isinstance(v2, list) else v2) for k, v2 in p.items()}})
                m2_rows.append({"subset": label, "obs": obs, "precision": precision, "geometry": gid,
                                **{k: v2 for k, v2 in entry["m2"].items()}})

    def gate(cname: str, precision: str, s1_key: str, m1_needed=8, m2_ratio=M2_RATIO) -> dict[str, Any]:
        res = evaluations[cname]
        passing = []
        for gid in non_sphere:
            entry = res.get(("+".join(GATE_SUBSET), "O2", precision, gid))
            if entry is None:
                continue
            v = geometry_verdict(entry, m1_needed, m2_ratio)
            if v["both"] and s1[gid][s1_key]:
                passing.append(gid)
        return {"n_pass": len(passing), "passing": passing}

    gates = {}
    for cname in evaluations:
        for precision in PRECISIONS:
            for s1_key in ("s1_s3_pass_strict", "s1_s3_pass_drift"):
                grid_sens = {f"m1>={k},m2<={r}": gate(cname, precision, s1_key, k, r)["n_pass"]
                             for k in (7, 8, 9) for r in (0.4, 0.5, 0.6)}
                g0 = gate(cname, precision, s1_key)
                gates[f"{cname}|{precision}|{s1_key}"] = {
                    **g0, "pass_5_of_6": g0["n_pass"] >= 5, "pass_4_of_6": g0["n_pass"] >= 4,
                    "pass_6_of_6": g0["n_pass"] >= 6, "threshold_sensitivity_n_pass": grid_sens}

    fb = choices["post_hoc_max_validity"]
    tosses = {g.id: {t.name: toss_validation(g, t.name, args.seed) for t in v3.ACTIONS if t.family == "toss"}
              for g in geoms}
    cross = {f"{p}": cross_geometry(geoms, tuple(fb[f] for f in GATE_SUBSET), "O2", p, args.seed)
             for p in ("ideal", "visual224")}
    robustness = {}
    for g in geoms:
        half = g.result["half_dt"]["rollouts"]
        base_key = config_key(*CENTER)
        rows = {}
        for a in v3.ACTIONS:
            if not half or not half.get(a.name, {}).get("valid") or not g.valid(base_key, a.name):
                rows[a.name] = {"status": "invalid"}
                continue
            f1 = g.feats[(base_key, a.name, 20)]
            f2 = action_features(half[a.name]["trace20"], a.family, a.frames, 1, dt)
            row = {}
            for feat in observed_features(a.family, "O2"):
                sv = sigma(feat, "visual224", dt, g.px_per_m.get(a.name, np.nan), g.radius, g.symmetric, f1["n_decel"])
                delta = abs(f2[feat] - f1[feat])
                row[feat] = {"abs_change": float(delta), "visual_sigma": float(sv),
                             "solver_dependent": bool(np.isfinite(delta) and delta > sv)}
            rows[a.name] = row
        robustness[g.id] = rows

    exponents = {}
    for g in geoms:
        fits = {a.name: response_exponents(g, a.name) for a in v3.ACTIONS}
        fits["angle_push_lo_vs_push_hi_deg"] = direction_angle(fits[fb["push_lo"]], fits[fb["push_hi"]])
        fits["angle_push_lo_vs_toss_deg"] = direction_angle(fits[fb["push_lo"]], fits[fb["toss"]])
        fits["angle_push_hi_vs_toss_deg"] = direction_angle(fits[fb["push_hi"]], fits[fb["toss"]])
        exponents[g.id] = fits
    deceleration_resolution = {
        g.id: {a.name: {str(rate): [int(sum(np.isfinite(g.feats[(k, a.name, rate)]["deceleration"])
                                            for k in g.keys("grid") if g.feats.get((k, a.name, rate))
                                            and g.start_ok[k])),
                                    int(sum(1 for k in g.keys("grid") if g.feats.get((k, a.name, rate))
                                            and g.start_ok[k]))]
                        for rate in (20, 100)}
               for a in v3.ACTIONS if a.family != "toss"} for g in geoms}

    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.stage-", dir=str(out.parent)))
    try:
        write_figures(stage, geoms, verdict_rows, exponents, fb)
        np.savez_compressed(stage / "traces.npz", **{
            f"{g.id}|{k}|{a}|{rate}": g.result["rollouts"][k][a][f"trace{rate}"]
            for g in geoms for k in g.configs if g.result["rollouts"].get(k)
            for a in g.result["rollouts"][k] if g.result["rollouts"][k][a]["valid"] for rate in (20, 100)})
        for name, rows in (("verdicts.csv", verdict_rows), ("local_identifiability.csv", point_rows),
                           ("held_out_prediction.csv", m2_rows)):
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with open(stage / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        summary = {
            "format": "simpact_phino_v3_identifiability_pilot_v1", "command": sys.argv,
            "spec": "docs/phino_v3_identifiability_spec.md (+ amendments: ideal vs visual224 precision; toss validation)",
            "simpact_git": export._git_state(Path(__file__).resolve().parent.parent),
            "phino_git": export._git_state(args.phino_root.expanduser().resolve()),
            "source_grid": str(inv.grid_dir), "manifest_sha256": inv.manifest_sha256,
            "thresholds": {"kappa": KAPPA_MAX, "sd_log_mass": SD_LOG_MASS_MAX,
                           "sd_log_friction": SD_LOG_FRICTION_MAX, "fd_consistency": FD_CONSISTENCY,
                           "m2_ratio": M2_RATIO, "ridge_alpha": RIDGE_ALPHA, "noise_repeats": NOISE_REPEATS,
                           "visual_px_error": VISUAL_PX_ERROR, "ideal_factor": IDEAL_FACTOR,
                           "min_toss_apex_px": MIN_TOSS_APEX_PX, "min_airborne_frames": MIN_AIRBORNE_FRAMES,
                           "s1_drift_reading_m": DRIFT_S1_MAX_M},
            "visual_calibration": {g.id: {"px_per_m": g.px_per_m, "radius_m": g.radius} for g in geoms},
            "start_protocol": s1, "validity": table, "selection": choices, "gates": gates,
            "toss_validation": tosses, "cross_geometry": cross, "solver_robustness": robustness,
            "response_exponents": exponents,
            "deceleration_resolved_counts_[resolved,total]": deceleration_resolution,
        }
        export._write_json(stage / "summary.json", summary)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"OUTPUT={out}")
    print("V3_IDENTIFIABILITY_PILOT_OK")


if __name__ == "__main__":
    main()
