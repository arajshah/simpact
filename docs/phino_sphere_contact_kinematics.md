# Distinguish sphere rolling, sliding, and yaw spin

**Why:** The 1.0 m grid included the sphere, but the previous `v/(R*|omega|)` diagnostic combined *all* angular-velocity components and does not establish rolling. SIMPACT's PHINO `push_x` and `push_y` actions also apply a z-axis torque, so the sphere may slide while spinning about the vertical axis. The previous output's 0.42–0.47 speed ratios must not be presented as evidence of rolling.

MuJoCo **free-joint linear velocity is world-frame, whereas free-joint angular velocity is body-frame**. The new analysis converts the angular velocity to the world frame using the saved orientation quaternion before computing kinematic contact diagnostics. See the official MuJoCo Overview, 'Floating objects'.

## Run on Brown

From the SIMPACT repository root and `(simpact)` environment:

~~~bash
git pull --ff-only origin phino-physics-experiments
python scripts/analyze_phino_sphere_contact.py --self-test

python scripts/analyze_phino_sphere_contact.py \
  --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
  | tee results/phino_sphere_contact.log
~~~

Only existing saved `(50, 13)` trajectories are read. **No simulation rerun is necessary.** The approximate sphere radius defaults to `0.05 m`; override with `--radius-m` only if geometry validation supplies a different value.

Output: `results/phino_mesh_grid_cutoff_1m_v1/recovery_analysis/sphere_contact_diagnostic/`

- `summary.csv` reports coverage, peak planar speed, peak horizontal and vertical angular speeds, the fraction of squared angular speed about world z, nominal contact slip, and the maximum tilt reported by the generator.
- `frames.csv` contains 50 per-frame states per accepted push, including converted world angular velocity and nominal slip velocity.
- `report.json` records formulas, assumptions, and trial-level results.

## How to interpret

For a centered sphere of radius `R` on a stationary horizontal plane, the *nominal* bottom-point horizontal velocity is

`v_bottom,xy = v_center,xy + (omega_world × [0,0,-R])_xy`

or

`v_bottom,x = v_center,x - R * omega_world,y`

`v_bottom,y = v_center,y + R * omega_world,x`

Near-zero horizontal bottom-point velocity is **consistent with** rolling without slipping, *assuming* that the bottom of the sphere is actually the contact point. A high yaw-spin fraction means much of the angular speed is around the vertical axis and cannot explain translation by rolling on a horizontal plane. A high nominal slip speed is evidence against ideal no-slip rolling under these assumptions.

**Limits:** We do not inspect true MuJoCo contact points, body center-of-mass offsets, contact impulses, instantaneous geometry, or the actual surface velocity. Radius is estimated from the normalized bounding box rather than calibrated from the collision hull. Small nominal slip is not proof of real contact or physically accurate rolling.

This is a kinematic diagnostic, **not** PHINO model training and not real-world physical-property inference. Keep all current comparison datasets and raw results unchanged.
