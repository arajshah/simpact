# v3 identifiability grid: experimental specification

**Status (Nov 5, 2026).** §2–§6 were approved with two amendments: an ideal vs 224-px visual precision comparison, and toss airborne/mass validation. The Stage 3 pilot ran and **failed the gate**; see §10 and `docs/phino_experiment_log.md`. The full v3 grid has **not** been generated. §10 proposes v3.1 revisions that need approval before any rerun. Existing grids, exports, and results are preserved and read-only.

Central question: **Which physical interventions make hidden object properties identifiable, and can learned visual representations preserve that information?** v3 addresses the first half with simulator states only; Stage 4 (representations) starts only if Stage 3 passes.

## 1. Evidence motivating v3

From the visual pilot and the Stage 1 audit (`docs/phino_experiment_log.md`):

1. **Settling depends on friction.** The v2 settle depends on friction for cow, bunny, sphere, and blue carton (up to 25 mm and 45°). Bunny and sphere also start the action still moving. White carton, cube, and nut are clean.
2. **One horizontal push reveals one mass–friction combination.** For sliders, `lateral_flick` adds an almost parallel direction (0.7–5.6°). For the sphere, a push is friction-blind.
3. **Two push strengths are not enough by themselves.** A Coulomb point-mass model shows that final displacements under two push strengths respond along nearly parallel log-sensitivity directions:

   | Candidate | d log d / d log m | d log d / d log μ | Direction |
   |---|---|---|---|
   | 1.6 N, 0.2 s | −3.23 | −2.23 | −145° |
   | 4.0 N, 0.1 s | −2.28 | −1.28 | −151° |
   | toss, 4.0 N, 0.2 s | −4.78 | 0.00 | 180° |

   What breaks the degeneracy in that model is (a) the post-push deceleration, which equals μg independent of mass but is only measurable if sliding lasts several frames, and (b) a friction-independent vertical toss. This is a toy model; v3 must test it on the real contacts.
4. **Absolute quaternions encode mesh conventions.** Absolute quaternion and body-frame angular-velocity channels encode each mesh's arbitrary body frame, which distorts cross-geometry PHINO similarities (rank correlation 0.74 vs start-relative).

## 2. Unchanged from v2 (for continuity)

- **Geometries:** the seven isolated scenes in `results/phino_mesh_grid_cutoff_1m_v1/isolated_scenes/`, used read-only and with sha256 recorded.
- **Physics grid:** mass {0.25, 0.30, 0.35} kg × contact friction {0.20, 0.30, 0.40}. These are synthetic assignments. Friction is set on every collision geom (object and support), as in v2; mass and inertia are scaled on the object body.
- **Timing:** PHINO engine settings: 20 Hz frames, 50 frames per action, `engine.sim_timestep`, implicitfast integrator, at least 100 solver iterations.
- **Validity caps:** 1.0 m raw displacement, 60° relative tilt, no PHINO velocity clipping. Each action is judged independently; rejected actions are reported, never imputed.
- **The v2 generator `scripts/phino_simpact_pilot_v2.py` is not modified.** v3 is a new script that imports its helpers.

## 3. Initial-state protocol (identical start, zero residual motion)

- **S1. Canonical rest pose per geometry, computed once and independent of the physics configuration.**
  - Settle from the scene's pose at the reference configuration (0.30 kg, μ 0.30) using the v2 settle steps.
  - Then relax: step with velocity damping (qvel × 0.9 per step) until linear speed < 1e-4 m/s and angular speed < 1e-3 rad/s for 0.5 s. Give up after 10 s and report failure.
  - Record the canonical pose `q*`. Damping is used only here, never during experiments.
- **S2. Per-configuration start.**
  - Build the model with the configuration's mass and friction, then set qpos = `q*` and qvel = 0.
  - Run a 0.5 s zero-action hold without damping.
  - Accept the configuration only if the post-hold pose is within 0.1 mm and 0.1° of `q*`, linear speed < 1e-3 m/s, and angular speed < 1e-2 rad/s.
  - The post-hold state `(q0, v0)` is the start of every action. Rejections are reported per configuration.
- **S3. Zero-action control.** Record one 50-frame trace per configuration with no applied wrench. Accept only if the maximum displacement is < 0.5 mm and the rotation < 0.5°. Analyses report this trace as the non-action motion floor.

## 4. Interventions

All actions are fixed world-frame forces applied at the object's centre of mass (MuJoCo `xfrc_applied`), with no torque unless stated. They never depend on hidden mass or friction. Force is applied for the first *n* frames of each 50-frame window.

| Action | Purpose | Candidates (Stage 3 selects one per row) |
|---|---|---|
| `push_x_v2` | Reference: the unchanged v2 push (0.8 N for 4 frames, plus 0.02 N·m yaw torque) | — |
| `push_lo` | Low strength, above the largest Coulomb threshold μmg (1.37 N) | 1.6 N or 2.0 N, for 4 frames |
| `push_hi` | High strength, longer slide so deceleration spans several frames | 3.2 N or 4.0 N, for 2 frames |
| `toss` | Recalibrated lift, valid for all three masses: vertical force, then release (no hold), so flight is friction-independent | 4.0 N or 4.5 N (> 3.43 N weight of 0.35 kg), for 4 frames |

- **Predicted toss apex (Coulomb model):** 202–38 mm (4.0 N) or 301–80 mm (4.5 N), from 0.25 to 0.35 kg, within the 1 m cap.
- **v2 `grasp_lift_release` is retired:** its hold force equalled the weight of 0.30 kg, which made every 0.25 kg trial overshoot.
- **Pre-registered candidate selection, based on validity only and never on identifiability:**
  - **Pushes:** choose the strongest candidate valid for at least 95% of the 63 configurations and at least 8/9 per geometry. A stronger push gives a longer slide and so better-resolved deceleration.
  - **Toss:** choose the weakest candidate that leaves the support for all three masses and meets the same validity rule.
  - Identifiability metrics for non-selected candidates may be reported, but they must not change the selection.

## 5. Recording

Per configuration and action:
- **PHINO-layout trace** `(50, 13)`: start-relative position, absolute quaternion, linear velocity, body-frame angular velocity, for compatibility.
- **Start-relative trace** `(50, 13)`: start-relative position, rotation since `q0` (sign-continuous from identity), linear velocity, world-frame angular velocity. **This is the default for analysis and future PHINO exports.**
- **Diagnostic 100 Hz trace:** used only for the observation-rate analysis in §6, never for PHINO or image experiments.
- **Initial state and checks:** `q*`, `q0`, `v0`, the S2/S3 outcomes, and the zero-action trace.

The manifest declares both channel layouts, the action parameters, the selected candidates, the reason for every rejection, and hashes of scenes and code.

## 6. Stage 3: state-only identifiability pilot (runs only after approval)

**Configurations per geometry (all 7 geometries):**
- the 3×3 grid;
- 4 off-grid test points: m ∈ {0.275, 0.325} × μ ∈ {0.25, 0.35};
- finite-difference neighbours at every grid point: ±0.05 in log m and in log μ, and ±0.025 for consistency checks.

All candidates and the zero-action control are simulated. Compute is negligible (a few thousand 2.5 s rollouts on CPU).

**Pre-declared features**, from 20 Hz start-relative states:
- **Pushes:**
  - final planar displacement;
  - peak planar speed;
  - post-push deceleration: the least-squares slope of planar speed from the end of force to the first frame below 0.01 m/s. It needs at least 3 frames; otherwise it is flagged "unresolved at 20 Hz".
  - final yaw change.
- **Toss:** apex height above the start; time of apex; final planar displacement.

**Observation models:**
- **O1, final state only:** the final frame of each action, i.e. what a single post-interaction image can show.
- **O2, time-resolved:** the 20 Hz features above, i.e. what a video at PHINO's rate can show.
- **O3, full trajectory:** the entire start-relative 50×13 trace per action (ridge regression).
- **O4, high rate:** the O2 features recomputed at 100 Hz. This separates "physically identifiable" from "identifiable at 20 Hz".

**Action subsets:** {`push_x_v2`}, {`push_lo`}, {`push_hi`}, {`push_lo`, `push_hi`}, {`push_hi`, `toss`}, {`push_lo`, `push_hi`, `toss`}.

**Metrics:**
- **M1, local identifiability.**
  - At each grid point, J = ∂features/∂(log m, log μ) by central differences.
  - Each feature is scaled by a resolution σ: 1 mm for displacement and height, 0.02 m/s for speed (1 mm per frame), 0.1 m/s² for deceleration, 0.05 s for time, 1° for yaw.
  - Report singular values, the condition number κ of Σ^(-1/2) J, and Cramér–Rao standard deviations from (Jᵀ Σ⁻¹ J)⁻¹.
  - **A grid point is identifiable if** κ < 100, sd(log m) ≤ 0.046, and sd(log μ) ≤ 0.072. These thresholds are one quarter of the smallest grid step in log units.
  - Where the ±0.05 and ±0.025 derivatives disagree by more than 20% (non-smooth contact, stick/slip), the point is flagged and judged by M2/M4 only.
- **M2, held-out prediction within each geometry.**
  - Model: ridge regression (α = 1.0, fixed; standardized features) predicting (log m, log μ).
  - Train on the 9 grid points and test on the 4 off-grid points; also leave one grid point out.
  - Report MAE in log units against the training-mean baseline.
  - **Pass if** off-grid MAE ≤ 0.5 × baseline MAE for *both* properties.
- **M3, cross-geometry prediction.** Leave one geometry out with the same model; the sphere is reported separately. Reported, not gating: it informs Stage 4.
- **M4, global confusability.** Count pairs of distinct (m, μ) among the 13 configurations whose feature vectors fall within unit resolution-scaled distance. This is a non-local check that M1 cannot provide.
- **Solver robustness.** Rerun the centre configuration with half the simulation timestep. Flag features whose change exceeds their resolution σ as solver-dependent.

**Pre-registered hypotheses:**
- **H1:** {`push_lo`, `push_hi`} under O1 is ill-conditioned (fails M1 at most grid points) for sliders.
- **H2:** under O2, `push_hi` identifies μ through deceleration where sliding lasts at least 3 frames, and fails where it does not (heavy, high-friction configurations).
- **H3:** adding `toss` makes mass identifiable for all non-sphere geometries (toss friction sensitivity ≈ 0).
- **H4:** sphere friction stays unidentifiable under all actions, except possibly `push_hi` at low μ: a solid sphere pushed through its centre slips only if F > 3.5 μmg.

**Gate to the full v3 grid and Stage 4** (selected candidates, O2, subset {`push_lo`, `push_hi`, `toss`}):
- M1 identifiable at ≥ 8/9 grid points, **and**
- M2 pass,
- for at least 5 of the 6 non-sphere geometries, each of which must pass S1–S3.

The sphere is a pre-registered negative case and does not block the gate. If the gate fails, report which property, geometry, and observation model fails, and propose revised interventions; do not proceed.

## 7. Are more geometries needed?

- **For Stage 3: no.** Identifiability is assessed per geometry. The seven shapes span a box, two real2sim cartons, three irregular meshes, and a roller.
- **For Stage 4: yes.** With seven geometries, a geometry-held-out split trains on five, mostly one shape per class. The audit showed cross-geometry results are fragile (sphere failure; unstable clean-only training).
- **Recommendation: after Stage 3 passes, add a v3b set of about 20 procedurally generated MuJoCo primitives:**
  - shapes: boxes of varied aspect ratio, lying and upright cylinders, capsules, ellipsoids;
  - size 5–15 cm, using the same mass/friction grid;
  - pre-registered class-stratified geometry splits, fixed before any Stage 4 result.
  - Primitives need no perception pipeline and allow controlled shape variation. This is a separate approval.

## 8. Outputs and code (no overwrites)

| Item | Path | When |
|---|---|---|
| v3 generator | `scripts/phino_simpact_grid_v3.py` (imports v2 helpers) | with Stage 3 |
| Stage 3 pilot | `scripts/phino_v3_identifiability_pilot.py` → `results/phino_v3_identifiability_pilot_v1/` | after this spec is approved |
| Full v3 grid | `results/phino_mesh_grid_v3/`: 3×3 × 7 geometries, selected actions plus the zero-action control | after the Stage 3 gate passes and you approve |
| PHINO export v1 | `data/phino_simpact_v1/`, start-relative channels, `TrajectoryNormalizer(n_actions=k)` | Stage 4 |

Tests cover S1–S3 acceptance, the action schedules (no dependence on mass or friction), recording layouts, feature extraction, and the M1/M4 computations on analytic toy models with known Jacobians.

## 9. Decisions requested

1. Approve §3 (settle protocol) and §4 (actions, candidates, validity-only selection rule).
2. Approve the §6 metrics, thresholds, hypotheses, and gate, or adjust the resolution σ values and the 5-of-6 rule.
3. Confirm that `push_x_v2` is retained only as a reference, and that `grasp_lift_release` and `lateral_flick` are retired in v3.
4. Decide on v3b geometry expansion (recommended, after the Stage 3 gate).
5. Confirm start-relative channels as the default for future PHINO exports. PHINO's reference engine uses absolute channels, so this is a deliberate deviation.

**Decisions recorded (Nov 5, 2026).**
- Approved: the common-start protocol, candidate actions, validity-only selection, start-relative orientation, and §6 metrics.
- `push_x_v2` is kept as a reference; `grasp_lift_release` and `lateral_flick` are retired.
- The 5-of-6 gate is kept provisionally, with threshold sensitivity reported.
- Amendments: evaluate at ideal state precision and at 224-px visual precision; validate the toss's airborne, mass-specific response.

## 10. Stage 3 outcome and proposed v3.1 revisions (awaiting approval)

**Outcome** (code `c3bf96a`; full write-up in the experiment log):
- **The gate fails:** at most 2 of 6 non-sphere geometries pass, under every candidate combination, precision model, S1 reading, and threshold variant.
- **The pre-registered selection fails:** it selects no push candidate.
- **The toss is validated:** shape-independent, friction-blind, airborne for at least 5 frames, and mass-predictive (0.26–0.29 of baseline error at 224 px).
- **Two pushes observed only at the end are degenerate,** as predicted (2.1–8.4° apart).
- **Friction needs time-resolved post-push deceleration:** identifiable from ideal states, only marginal at 224 px and 20 Hz.

The failures come mainly from the protocol, not from identifiability.

**Proposed revisions** (none applied; each needs approval):

- **R1. Start criteria.**
  - **Problem:** the S1/S2 speed thresholds sit below MuJoCo's resting-contact jitter (1e-4–1e-3 m/s), even though positions are stable to 0.001 mm.
  - **Proposal:** judge rest by an undamped 0.5 s hold with displacement ≤ 0.01 mm and rotation ≤ 0.01°, plus the unchanged S3 zero-action control (≤ 0.5 mm over 2.5 s).
  - **Bunny** (rocks about 0.1 rad/s): search a few candidate rest orientations at reference physics and use the most stable one.
  - **Sphere** (polyhedral mesh rolls indefinitely): an analytic sphere collision primitive would be a geometry change; or exclude the sphere from push studies. Your decision.
- **R2. Push application point.**
  - **Problem:** centre-of-mass pushes topple the cartons and flip the nut.
  - **Proposal:** apply the same world-frame force at a fixed low point, 1 cm above the support contact, implemented as an equivalent wrench (force plus r × F torque about the centre of mass). This resembles a low robot pusher.
  - Re-validate candidates with the same validity-only rule. Alternatively, count toppling as a valid categorical response; I don't recommend this, because it changes the response regime.
- **R3. Friction features at 224 px.**
  - **Problem:** finite-difference speeds amplify pixel noise, and more so at higher frame rates.
  - **Proposal:** pre-register deceleration as the curvature of a quadratic fit to post-push positions (not speeds), and add sliding duration (stop time) as a feature. This is a feature-extraction change only, with no new physics.
- **R4. Within-geometry prediction design.**
  - **Problem:** a 9-point linear ridge is sample-limited; cross-geometry prediction with 39 training points did better (friction 0.25–0.27 at 224 px).
  - **Proposal:** train on a 5×5 within-geometry design (25 points, same ranges) and test on 4 off-grid points. Keep α = 1 and the 0.5 ratio.
- **R5. Geometry set for the Stage 3 rerun.** Only four geometries currently have verified starts. Add the recommended v3b primitives (wide, low, and tall shapes) to the Stage 3 rerun, so the gate is judged on more than 4–6 shapes, with the split stratified by class.
- **R6. Sphere and caps.** With fixed, geometry-independent actions the sphere exceeds the 1 m displacement cap. Testing H4 requires either a larger workspace cap for pushes or accepting that the sphere stays outside the push study. The tilt rule is also meaningless for a sphere; exempt rotationally symmetric bodies from it.

**Recommendation.** Do not generate the full v3 dataset. Approve or amend R1–R6, then rerun Stage 3 (about 2 minutes of CPU per 7 geometries) under a re-registered gate.
