# MASTER HANDOFF: Brown Research Project, SIMPACT → PHINO
## Research Context, Engineering State, Scientific Objectives, and Cursor Agent Operating Instructions

**Date:** November 5, 2026  
**Environment:** Brown University computing cluster  
**Agent role:** Primary research-engineering implementation agent  
**Human role:** Research owner, scientific decision-maker, and final approver  
**Current priority:** Complete SIMPACT-to-PHINO dataset integration and validate it end-to-end.

---

# 0. YOUR ROLE AND MISSION

You are joining an ongoing research project involving physical intelligence, representation learning, world models, and simulation.

You are not starting a new project. Significant research, engineering, debugging, and experimentation have already occurred.

Your responsibility is to:

1. Understand the scientific problem and motivation.
2. Recover the project's actual current state from the repositories, generated data, documentation, and existing results.
3. Distinguish validated findings from hypotheses, preliminary experiments, and outdated plans.
4. Continue implementation without unnecessarily reproducing completed work.
5. Make technically sound engineering decisions independently.
6. Identify scientific assumptions that could invalidate an experiment.
7. Preserve experimental reproducibility.
8. Produce high-quality code, tests, documentation, and results.
9. Communicate clearly with the research owner.
10. Help develop the project toward a meaningful research contribution rather than merely a functioning software pipeline.

The research owner is a Brown University master's student in computer science pursuing serious research in reinforcement learning, representation learning, and world models, with longer-term doctoral research ambitions.

This is research software. Scientific validity is at least as important as code functionality.

**Your guiding principle: Implement independently, but do not make consequential scientific assumptions silently.**

You should take ownership of ordinary engineering decisions. Escalate decisions that change the research question, experimental validity, or interpretation.

Do not ask permission for every implementation detail.

Do not invent scientific conclusions to make the project appear more successful.

---

# 1. SOURCES OF TRUTH AND CONTEXT

You have direct access to the Brown server. Use that access.

## 1.1 Canonical Brown research memory

**Document:** Brown Academics - Project Memory

**URL:**

https://docs.google.com/document/d/1ddRUpFKP-F9LsfMpf68XJVDXXvi02s0R0gHk2HZ5rfE/edit

This document preserves the wider Brown academic and research history.

It includes context about:

- Research interests in world models, reinforcement learning, self-supervised learning, and representation learning.
- Relevant Brown faculty and researchers.
- Research collaborations and prospective collaborators.
- Connections between representation learning, physical intelligence, and control.
- Previous papers, ideas, decisions, and research direction changes.
- Academic research goals and longer-term ambitions.

**Important:** This is broad historical context, not the most current operational source for PHINO.

Its last verified modification was October 28, 2026. Some plans may have changed since then.

Read relevant sections when you need to understand the motivation, history, or connections to other research directions.

Do not consume the entire document repeatedly.

Do not copy private academic or personal information into public GitHub artifacts.

Do not assume everything in the memory document remains current.

## 1.2 Primary SIMPACT repository

**GitHub:**

https://github.com/arajshah/simpact

**Working branch:**

`phino-physics-experiments`

**Brown directory:**

`/nbu/people/ashah264/simpact`

**Original upstream project:**

https://github.com/ShaoxiongYao/simpact

This is where most current implementation work should occur.

Important files:

- `AGENTS.md`
- `docs/phino_cursor_handoff.md`
- `docs/phino_pretraining_integration_contract.md`
- `docs/phino_displacement_cutoff_comparison.md`
- `docs/phino_sphere_diagnostics.md`
- `docs/phino_sphere_training_transfer.md`
- `docs/phino_sphere_contact_kinematics.md`

The first three files are essential onboarding material.

Read them completely before implementing new functionality.

Also inspect the actual source code and current Git state.

## 1.3 PHINO repository

**User fork:**

https://github.com/arajshah/phino

**Upstream:**

https://github.com/Hongyang-Du/phino

**Expected Brown directory:**

`/nbu/people/ashah264/phino`

Verify that this directory exists and inspect the repository state.

The previously inspected PHINO branch was `main`. Another experimental branch, `eval/grouped-physics-probes-oct28`, has also existed.

Do not assume the local checkout is on a particular branch.

Do not reset or overwrite local work.

## 1.4 Source authority hierarchy

Use the following priority when sources disagree:

1. Actual runtime behavior, validated tests, and generated artifacts.
2. Current source code and the manifests associated with generated datasets.
3. Current Git commits and configuration files.
4. Current project-specific GitHub documentation.
5. Experiment logs and dated research notes.
6. Brown academic memory document.
7. This onboarding prompt.

This prompt is an orientation document, not a substitute for verification.

If a claim here disagrees with actual data, investigate and report the discrepancy.

---

# 2. BROADER RESEARCH VISION

The overarching interest is in learning representations of physical environments that are useful for intelligent action.

One formulation is:

**What information about the physical world should a learned representation preserve so that an agent can make reliable predictions and control decisions?**

Related questions include:

- Which aspects of physical state matter for a given task?
- How can representations preserve task-relevant dynamics while ignoring irrelevant variation?
- Can an agent infer hidden physical properties through interactions?
- Can representations learned from physical responses generalize to new objects and environments?
- What information is needed for planning and control?
- Can we obtain useful physical representations without reconstructing every irrelevant detail of an environment?
- How do object geometry, appearance, material, mass, friction, and interaction dynamics affect what is learnable?
- How should we evaluate whether a representation actually contains useful physical information?

These questions connect self-supervised representation learning, world models, reinforcement learning, system identification, and embodied intelligence.

The research owner is interested in **a meaningful scientific principle and a credible research contribution**, not an arbitrary collection of engineering improvements.

## 2.1 Why PHINO is relevant

PHINO, *Physical Intuition via Newtonian Observations*, investigates using simulated physical interactions to produce supervision for visual representations.

The broad idea is:

1. Obtain an object representation or image.
2. Associate the object with physical-property hypotheses.
3. Simulate responses under controlled forces.
4. Compare the resulting trajectories.
5. Use those trajectory similarities as a learning signal for a visual representation.

In the inspected implementation, the model uses a DINOv2 vision backbone and a contrastive training objective related to simulated dynamics.

The intended goal is a representation that captures physical relationships rather than only conventional visual semantics.

However, it is essential to distinguish the authors' intended objective from what the implementation and data can actually support.

## 2.2 Why SIMPACT is relevant

SIMPACT provides a path from visual observations and reconstructed objects to physical simulation.

The original SIMPACT pipeline includes components for:

- Object detection and grounding.
- Image segmentation.
- Object reconstruction.
- Pose estimation.
- Scene preparation.
- MuJoCo simulation.
- Physical interaction modeling.

Related tools in the broader pipeline include GroundingDINO, SAM2, SAM-3D, and FoundationPose.

The current experiments use reconstructed objects and mesh assets in controlled MuJoCo simulations.

The long-term attraction is that SIMPACT can help construct or approximate physical environments, while PHINO may help learn representations of their dynamics.

**SIMPACT and PHINO are complementary, but they are not yet a validated end-to-end physical-intelligence system.**

## 2.3 Longer-term research directions

Possible future directions include:

- Learning physical properties from observed interventions.
- Encoding object response to known actions.
- Designing interactions that distinguish otherwise ambiguous physical properties.
- Learning representations that transfer across geometries.
- Using physical representations to improve downstream policies.
- Testing which dynamics information matters for particular control tasks.
- Building representations invariant to task-irrelevant variation.
- Investigating when trajectory similarity provides useful representation-learning supervision.

Other Brown research discussions have also considered task-relevant physical representations, semantic descriptions of physical states, and control-relevant abstraction.

These are useful connections, not approved implementation requirements.

**Do not attempt to combine every research direction into this one project.**

The current work should establish a narrow, reproducible, technically credible foundation.

---

# 3. THE SCIENTIFIC QUESTION WE HAVE ACTUALLY TESTED

Our recent work investigated:

**Can controlled simulated trajectories reveal synthetic mass and friction, including when the object geometry changes?**

We generated object trajectories under fixed external interventions and used simple supervised models to recover the simulator parameters.

This is a diagnostic of information contained in trajectories.

It is not yet an evaluation of learned PHINO image representations.

The distinction is fundamental.

A model that receives a full simulated trajectory has access to physical responses.

A model that receives only a single static object image does not have that same information.

We must never confuse those two settings.

---

# 4. WHAT HAS ALREADY BEEN IMPLEMENTED

The SIMPACT repository contains several experimental scripts.

Inspect them rather than recreating their functionality.

## 4.1 Mesh scene construction

File:

`scripts/build_phino_mesh_scenes.py`

This constructs test MuJoCo scenes for five geometry assets:

- Cube
- Bunny
- Cow
- Sphere
- Nut

These were combined with two existing reconstructed carton objects.

The generated scenes use the existing mesh assets, normalize object scale, and construct suitable collision representations.

The resulting scenes are located under:

`results/phino_mesh_scenes_v1/`

Typical paths resemble:

`results/phino_mesh_scenes_v1/object_sphere/scene.xml`

These are generated data, not files necessarily available in GitHub.

The scene builder has previously passed self-tests and actual MuJoCo compilation checks on Brown.

## 4.2 Four-action physics generator

File:

`scripts/phino_simpact_pilot_v2.py`

This is the primary controlled-interaction dataset generator.

It uses four actions:

1. `push_x`
2. `push_y`
3. `grasp_lift_release`
4. `lateral_flick`

The actions use fixed external world-frame wrenches.

They are not robot-executed grasps or contact-rich manipulation policies.

The external actions are held constant across physical configurations.

This makes differences in response interpretable as consequences of the simulated scene and physical parameters, subject to the simulation's assumptions.

Each action records 50 frames.

Each frame contains 13 numerical values.

The layout is:

`[relative_position_xyz, quaternion_wxyz, linear_velocity_xyz, angular_velocity_xyz]`

The resulting individual action trajectory has shape:

`(50, 13)`

A complete four-action fingerprint has shape:

`(200, 13)`

The action order matters.

Do not reorder the segments without explicitly adapting every consumer.

## 4.3 Physical parameter grid

Seven geometries were evaluated.

For each geometry, the synthetic parameter grid contained:

**Mass:**

- 0.25 kg
- 0.30 kg
- 0.35 kg

**Contact friction coefficient:**

- 0.20
- 0.30
- 0.40

There were:

`7 geometries × 3 masses × 3 friction values = 63 configurations`

There were four attempted actions per configuration:

`63 × 4 = 252 action trials`

These values are **assigned simulation parameters**, not measured real-world properties.

Do not label them as physically measured material coefficients.

Do not infer that the simulator's friction parameter is necessarily equivalent to the static and dynamic friction labels used elsewhere in PHINO.

## 4.4 Validity and rejection

The generator validates physical trajectories and rejects some actions.

Checks include:

- Nonfinite state values.
- Numerical warnings from MuJoCo.
- Maximum raw displacement.
- Maximum relative orientation tilt.
- Whether linear or angular velocity would exceed PHINO's clipping limits.

An action can be rejected independently of the other actions in the same configuration.

A configuration with some rejected actions may still retain the valid individual trajectories.

However:

**Only configurations with all four valid action traces have a complete `(200, 13)` fingerprint.**

Do not concatenate incomplete actions.

Do not impute missing action data.

Do not silently relabel rejected actions as accepted.

The generator also records whether trajectories satisfy a separate desired motion range.

Motion-target satisfaction is not equivalent to numerical validity.

---

# 5. ORIGINAL EXPERIMENT: 0.25-METER CUTOFF

Original multi-geometry output:

`results/phino_mesh_grid_v1/`

The original raw-displacement cutoff was:

`0.25 m`

Recorded overall results:

- 63 physics configurations.
- 252 attempted action trials.
- 210 valid individual actions.
- 35 complete four-action fingerprints.

Action coverage:

| Action | Valid / attempted |
|---|---|
| Push X | 54 / 63 |
| Push Y | 54 / 63 |
| Grasp/lift/release | 41 / 63 |
| Lateral flick | 61 / 63 |

A significant issue appeared:

**All nine sphere Push X trials and all nine sphere Push Y trials exceeded the original displacement threshold.**

Therefore the sphere was excluded from those push-action analyses.

This meant some apparently favorable regression results reflected performance over a restricted geometry population.

This was not necessarily deliberate selection bias, but it was systematic missingness.

We investigated it instead of simply hiding the sphere.

---

# 6. SPHERE DIAGNOSTIC EXPERIMENTS

Files:

- `scripts/diagnose_phino_sphere.py`
- `scripts/analyze_phino_sphere_contact.py`

Documentation:

- `docs/phino_sphere_diagnostics.md`
- `docs/phino_sphere_contact_kinematics.md`

## 6.1 Diagnostic result

The sphere diagnostic was executed over:

- Three masses.
- Three friction coefficients.
- Two push directions.

All 18 diagnostic trajectories completed their intended 50 frames without reported numerical errors.

Their maximum displacements ranged approximately:

`0.484 m to 0.791 m`

Therefore:

**Exceeding the 0.25 m cutoff does not necessarily indicate numerical failure.**

But numerical completion is not proof of physical realism.

We consequently tested a larger displacement cutoff in a separate experiment.

## 6.2 Expanded cutoff

Expanded experiment:

`results/phino_mesh_grid_cutoff_1m_v1/`

Maximum displacement:

`1.0 m`

The expanded grid preserved the original experiment rather than overwriting it.

In the downstream analysis:

- Sphere Push X had 9/9 accepted trajectories.
- Sphere Push Y had 8/9 accepted trajectories.

Do not confuse the separate 18/18 diagnostic completion result with the actual grid's acceptance flags.

The grid retains other validation checks, and those checks can still reject an action.

## 6.3 Sphere motion

We also investigated whether the sphere was predominantly rolling, sliding, or spinning.

The push actions apply both horizontal force and a vertical-axis torque.

MuJoCo's free-joint angular velocity is expressed in body coordinates, so the diagnostic correctly converts it to world coordinates before assessing nominal bottom-contact motion.

Measured aggregate results:

| Metric | Push X | Push Y |
|---|---:|---:|
| Accepted trajectories | 9/9 | 8/9 |
| Median vertical-axis share of squared angular speed | 0.811 | 0.779 |
| Median estimated nominal contact slip | 0.023 m/s | 0.021 m/s |
| Median maximum relative tilt | 51.8° | 57.5° |

These results suggest substantial vertical-axis spinning with approximately rolling-like motion.

They do not prove ideal rolling without slipping.

The nominal contact calculation assumes a spherical contact geometry and approximate radius, not the true resolved contact mechanics.

**Decision:** Stop extending the sphere investigation unless it becomes a blocker.

Do not launch a new torque ablation or another large parameter sweep as part of the current integration task.

The existing diagnostics are sufficient to document the limitation.

---

# 7. PROPERTY-RECOVERY EXPERIMENTS

File:

`scripts/analyze_phino_pilot_grid.py`

The analysis uses simple ridge regression.

Inputs are compact features extracted from the simulated trajectories, including displacement, path length, speed, angular speed, and orientation change.

The targets are:

- Assigned mass.
- Assigned contact friction.

The model is trained on available trajectories and evaluated using held-out configurations or geometries.

A mean-prediction baseline is used for comparison.

The primary ridge penalty was fixed at:

`alpha = 10`

This is a relatively small, exploratory synthetic dataset.

## 7.1 Initial results

Some object-held-out experiments performed better than the training-mean baseline.

For example, on the earlier restricted set of eligible geometries:

- Push X + Push Y using motion-magnitude features achieved mass mean absolute error of approximately 0.0266 kg versus 0.0333 kg for the baseline.
- Push X + Push Y without vertical-position features achieved friction mean absolute error of approximately 0.0471 versus 0.0667 for the baseline.

These were encouraging but limited findings.

They established partial recoverability of synthetic properties from privileged trajectories.

They did not establish general physical understanding.

## 7.2 Sphere generalization failure

When sphere push trajectories became eligible under the larger displacement cutoff, object-held-out performance exposed a major failure.

For example, the original feature set produced sphere errors approximately:

- Push X mass MAE: 0.1576 kg versus a 0.0333 kg mean baseline.
- Push X friction MAE: 0.5399 versus a 0.0667 mean baseline.
- Push Y mass MAE: 0.1968 kg versus a 0.0313 kg mean baseline.
- Push Y friction MAE: 0.5873 versus a 0.0625 mean baseline.

Some sphere displacement features were approximately 19–21 training standard deviations away from the training mean.

This indicated strong distribution shift.

A ridge regression model trained primarily on other geometries did not extrapolate successfully to the sphere's different motion regime.

**Interpretation:** The failure may reflect geometry-dependent dynamics, representation limitations, and out-of-distribution extrapolation.

Do not reduce this to one unverified explanation.

## 7.3 Matched sphere inclusion experiment

We subsequently implemented an experiment that evaluated the same six held-out non-sphere geometries with two training conditions:

- Train on five other non-sphere geometries.
- Train on those five plus the sphere.

The test cases were matched across conditions.

Selected results:

| Features / actions | Target | Without sphere | With sphere |
|---|---|---:|---:|
| Original, all three non-lift actions | Mass | 0.0397 kg | 0.0336 kg |
| Original, all three non-lift actions | Friction | 0.0643 | 0.0551 |
| Motion magnitudes, Push X + Push Y | Mass | 0.0266 kg | 0.0269 kg |
| No vertical position, Push X + Push Y | Friction | 0.0471 | 0.0503 |

Adding sphere data improved some original-feature models but slightly worsened some reduced-feature models.

Thus, the result was mixed.

The sample size and training distribution both change when sphere examples are added.

This experiment does not establish a general causal benefit of adding sphere data.

## 7.4 What the experiments established

Supported conclusions:

1. The simulation pipeline can generate controlled, parameterized trajectories over multiple geometries.
2. Many trajectories are numerically usable.
3. Some synthetic physical properties are recoverable from trajectories using simple models.
4. Generalization across geometries is challenging.
5. The choice of validity cutoff affects dataset coverage.
6. The sphere is an informative out-of-distribution case.
7. The data and evaluation protocol require care.

Unsupported conclusions:

1. That mass and friction are uniquely identifiable in all simulated regimes.
2. That the representations generalize to arbitrary objects.
3. That the simulated friction coefficients correspond to measured materials.
4. That PHINO has learned physical properties from our data.
5. That a static image determines an arbitrarily assigned mass.
6. That these results establish real-world robotic control improvements.

Do not claim any of those unsupported conclusions.

---

# 8. WHY WE STOPPED RUNNING MORE SIMULATION DIAGNOSTICS

We explicitly decided that continuing to debug the sphere would offer diminishing returns.

At that stage, we already had enough evidence to establish both feasibility and limitations.

The central bottleneck had shifted.

We were continuing to refine simulated trajectories, but had not yet connected them to PHINO's actual representation-learning pipeline.

The next useful question was:

**Can the existing SIMPACT-generated data be transformed into a valid PHINO training dataset?**

That is the current engineering problem.

Please preserve this prioritization.

Do not spend substantial time improving the sphere, tweaking the validation cutoff, or tuning ridge regressions unless new evidence shows that such work is necessary for the immediate integration.

---

# 9. PHINO TRAINING CODE: VERIFIED IMPLEMENTATION

The following code was inspected from the user's PHINO fork.

You must re-inspect it locally before modifying anything.

## 9.1 Training entry point

`/nbu/people/ashah264/phino/scripts/pretrain.py`

This creates the model, contrastive loss, dataloader, and trainer.

It uses configuration values from:

`configs/pretrain.yaml`

The training architecture includes a DINOv2-B/14 backbone and the PHINO/DynaCLIP model components.

The current default configuration includes a full training schedule substantially larger than what our dataset warrants.

**Do not launch that training schedule yet.**

## 9.2 Dataset loader

`train/data/dataset.py`

Important class:

`DynaCLIPContrastiveDataset`

The loader reads:

- `metadata.json`
- `similarity_matrix.npz`
- Image files
- Trajectory fingerprint files

It returns image pairs, trajectory similarity labels, and some physical-property metadata.

The loader has hard-negative, hard-positive, and random-pair sampling.

The default hard-pair proportions are approximately:

- 30% same-image-group / different-physics negatives.
- 30% different-image-group / similar-physics positives.
- 40% random pairs.

Do not assume these categories are scientifically sound for our dataset.

For initial interface validation, set the hard-mining proportions to zero and use a small precomputed pair file.

## 9.3 Data-generation format

`train/data/generation.py`

This is the reference implementation for the expected on-disk data schema.

It generates metadata, fingerprints, and pairwise trajectory similarities.

Its physics configuration includes mass, static friction, dynamic friction, and restitution.

SIMPACT does not directly provide matching measurements for all those fields.

## 9.4 Normalization

`train/data/mujoco_engine.py`

Important class:

`TrajectoryNormalizer`

PHINO's original implementation expects five diagnostic actions.

The default normalizer is constructed for five actions with 50 frames each.

Our SIMPACT generator provides **four** actions.

This is a concrete data-format mismatch.

For SIMPACT fingerprints, use:

`TrajectoryNormalizer(n_actions=4, timesteps=50)`

The normalizer standardizes trajectory values per action and channel.

The similarity function uses normalized trajectory mean squared error and a calibrated scale to produce similarity values.

Use one consistently fitted normalizer for a given training dataset.

Do not silently reshape or pad our fingerprints to five actions.

Do not compare different action sets as if their normalized distances had identical meanings.

## 9.5 Images

`train/data/assets.py`

Important function:

`render_multiview(...)`

PHINO can render images from MuJoCo scenes.

The intended image resolution is around 224 × 224 pixels for the existing visual encoder.

Before writing a new renderer, check whether PHINO's existing renderer works on our isolated SIMPACT scenes.

Our original two-carton scene contains more than one free-jointed object.

Use the isolated, per-object scene files for canonical object renders.

## 9.6 Contrastive loss

`train/losses/contrastive.py`

Important class:

`SoftInfoNCELoss`

The loss receives image embeddings and trajectory-derived pair similarity values.

The inspected implementation creates a batch similarity target whose diagonal contains the matched-pair similarity, while off-diagonal values are zero before softmax.

This is not the same thing as having measured all pairwise trajectory similarities among every element in a batch.

Be precise when describing the training objective.

Do not rewrite the loss as part of the dataset adapter task unless the interface itself requires it.

A separate research review of the loss design may be useful later.

---

# 10. CRITICAL SCIENTIFIC ISSUE: STATIC-IMAGE IDENTIFIABILITY

This is arguably the most important conceptual issue in the project.

Suppose we render one object image:

`I = image(object_geometry)`

Then simulate the same object's geometry under different assigned physics:

`theta_1 = (mass_1, friction_1)`

`theta_2 = (mass_2, friction_2)`

The rendered static image may be identical under both configurations.

A deterministic image encoder computes:

`z = f(I)`

If the input image is identical, the encoder produces the same representation.

Yet the simulated physical responses differ.

Therefore, the network cannot identify the arbitrarily assigned hidden physical configuration from that single unchanged image.

Random camera viewpoints do not solve this fundamental problem.

They can create different visual inputs, but those visual differences do not reliably encode the independently assigned hidden physics.

The model may learn view-dependent shortcuts rather than physical properties.

## 10.1 What static images can still support

Static images may provide probabilistic information about:

- Geometry.
- Visible materials.
- Object dimensions when scale is known.
- Likely physical properties associated with appearance.
- Shape-dependent response tendencies.

They do not uniquely determine arbitrary unobserved assignments of mass and friction.

A sensible model may learn physical priors from appearance.

It cannot recover information that is not present in its observations.

## 10.2 Why action observations matter

An intervention can supply missing physical information.

For example:

`image + known force + observed displacement`

contains more information about mass and friction than:

`image alone`

Two objects that look identical can exhibit different trajectories when acted upon.

Their responses may help distinguish hidden physical parameters.

This suggests a possible future direction:

**Learn visual physical representations conditioned on controlled interactions and observed responses.**

Possible inputs include:

- Initial object images.
- Known applied actions.
- Post-action images.
- Short image sequences.
- Video observations of motion.

A useful downstream representation could encode aspects of dynamics that change control outcomes.

However, this is a future research decision.

Do not silently redesign PHINO into an action-conditioned video model while implementing the current dataset adapter.

## 10.3 The current task's purpose

The current adapter is an **engineering and interface validation milestone**.

It should show that SIMPACT data can be loaded by PHINO's existing pipeline.

It should not be presented as proof that PHINO can infer hidden physics from static images.

The research owner will decide whether the next experiment should change the observation modality or scientific objective.

---

# 11. THE CURRENT IMPLEMENTATION TASK

**Implement a reliable SIMPACT → PHINO dataset export pipeline.**

Suggested script:

`scripts/export_phino_dataset.py`

Implement in the SIMPACT repository unless actual repository design gives a compelling reason for a different location.

The converter should use the already generated grid.

Default intended input:

`results/phino_mesh_grid_cutoff_1m_v1/`

Default intended output:

`data/phino_simpact_v0/`

Make input and output locations configurable.

Do not overwrite existing data.

Do not rerun physics simulations.

## 11.1 Input inventory

Inspect the following files and directories:

`manifest.json`

`fingerprints/`

`action_trajectories/`

`isolated_scenes/`

Also inspect source scene metadata and the per-object scene paths stored in the manifest.

Record:

- Total configurations.
- Accepted complete fingerprints.
- Rejected or incomplete configurations.
- Missing actions.
- Valid configurations by geometry.
- Valid configurations by mass and friction.
- Any missing or invalid source files.
- Number of geometries actually represented in complete fingerprints.

Do not assume all 63 configurations have usable four-action fingerprints.

Derive the exact count from the generated manifest.

## 11.2 Validate each input

Check:

- Expected source format.
- Declared action order.
- Correct number of actions.
- 50 frames per action.
- Complete fingerprint shape `(200, 13)`.
- Numeric finiteness.
- Correct object ID.
- Correct mass/friction association.
- Correct source file path.
- No duplicate object/physics configurations.
- No duplicate output IDs.
- No missing required files.
- Consistency between accepted statuses and saved fingerprints.
- Consistency of embedded fingerprint metadata where available.
- Valid frame ordering and feature layout.

Fail clearly when something is wrong.

Never silently discard a corrupted accepted fingerprint.

Report exclusions separately from corruption.

A rejected action is a legitimate exclusion.

A missing file for an accepted action is a data-integrity problem.

## 11.3 Render object images

The dataset needs images before it can be consumed by the PHINO visual dataloader.

We have seven geometries, but image rendering has not yet been validated for the new PHINO-compatible dataset.

Start by testing the existing PHINO renderer on one isolated SIMPACT object scene.

Then render canonical views for the relevant geometries.

Requirements:

- Use the actual isolated object scene.
- Keep the object visible and properly framed.
- Produce readable RGB images.
- Use a consistent image resolution.
- Prefer 224 × 224 initially to match PHINO's image transform.
- Use fixed camera conventions for reproducibility.
- Validate image files after rendering.
- Record the image-to-geometry association.
- Render each geometry only once unless multiple views are intentionally requested.
- Do not regenerate identical images separately for every mass/friction configuration.
- Do not pretend a camera viewpoint reveals hidden assigned physical properties.

A small contact sheet showing all seven canonical renders would be useful for human inspection, but should not become a major separate implementation project.

If some geometries have no accepted complete fingerprints, still document their image availability and dataset exclusion status.

## 11.4 Emit PHINO-compatible metadata

The target metadata should follow PHINO's existing loader.

Each accepted configuration needs a record with stable identifiers.

Important fields include:

- `global_idx`
- `image_group`
- `asset_id`
- `category`
- `material`
- `mjcf_path`
- `image_path`
- `view_paths`
- `fingerprint_path`
- `mass`
- `static_friction`
- `dynamic_friction`
- `restitution`

Assign contiguous global indices.

All physics variants of the same geometry should share an image group.

Use relative file paths where the PHINO loader supports them.

Do not create metadata entries for nonexistent fingerprints.

The synthetic mass label should come from the exact simulation record.

For the friction fields:

SIMPACT's contact friction setting is not necessarily a direct measurement of PHINO's static or dynamic friction semantics.

If the loader requires fields that SIMPACT does not supply, implement a documented compatibility mapping or placeholder.

Record its provenance explicitly.

Do not present placeholder values as experimentally measured physical properties.

If the correct mapping requires a scientific decision, explain the options and escalate.

## 11.5 Fingerprint export

Create the expected PHINO fingerprint naming and file layout.

For example:

`fingerprints/fp_000000.npz`

Each exported file should contain:

`flat_trajectory`

with shape:

`(200, 13)`

Keep the precise action order from the source manifest.

Additional metadata may be copied if useful for verification.

Do not alter trajectory values merely to satisfy formatting.

Verify that exported fingerprints match their source data.

## 11.6 Normalization

Use the PHINO `TrajectoryNormalizer` implementation where possible.

Instantiate it with four actions.

Fit it only on the designated training subset.

Ensure the data split is established before fitting dataset-level statistics.

Save the normalizer in PHINO's expected format.

Verify that a subsequent process can load the file and reproduce the same transformations.

Check finite means, standard deviations, and similarity values.

Do not fit normalization statistics using held-out evaluation geometries.

## 11.7 Trajectory similarities

Create:

`similarity_matrix.npz`

The expected arrays are:

- `pairs`, shape `(N, 2)`.
- `similarities`, shape `(N,)`.

The pair indices must refer to valid metadata indices.

Generate pairs deterministically using a fixed random seed.

Avoid malformed pairs, duplicate identifiers, and invalid similarity values.

Check that all similarity values are finite and within the expected range.

Use the same trajectory normalizer for every pair in the same training dataset.

For a small dataset, generating all unique eligible unordered pairs may be reasonable.

Choose the simplest reproducible approach consistent with the existing loader.

Do not mix train and held-out evaluation geometries in training pair generation.

## 11.8 Train/evaluation separation

Group splits by geometry.

All physics configurations and image views of the same geometry must remain together.

We have only seven geometries.

Therefore, statistical conclusions from held-out geometries will be limited.

The immediate milestone does not require a final benchmark.

Nevertheless, create or record a reproducible geometry-disjoint split so that later experiments cannot accidentally leak an object's physics configurations between training and evaluation.

Do not choose which objects are held out based on observed downstream performance.

Do not fit normalizers on held-out data.

Do not mine training pairs using hidden evaluation labels.

If the current PHINO loader does not natively support this split, implement a small adapter-side split manifest or separate training/evaluation metadata rather than rewriting the whole dataloader prematurely.

## 11.9 Provenance

Create a provenance file, such as:

`provenance.json`

It should record:

- SIMPACT repository revision.
- PHINO repository revision.
- Source grid location.
- Source manifest identity or hash.
- Source action order.
- Source cutoff.
- Source physical parameter grid.
- Included and excluded configuration counts.
- Export settings.
- Random seed.
- Image-rendering settings.
- Normalization settings.
- Similarity-generation settings.
- Geometry split assignments.
- Any compatibility mappings.
- Warnings and unresolved limitations.

This metadata should make it possible to reconstruct how the exported dataset was produced.

---

# 12. EXPECTED DATASET LAYOUT

A reasonable output layout is:

    data/phino_simpact_v0/
        metadata.json
        provenance.json
        images/
            object_0_v0.png
            object_1_v0.png
            ...
        fingerprints/
            fp_000000.npz
            fp_000001.npz
            ...
        traj_norm.npz
        similarity_matrix.npz
        geometry_splits.json

You may adapt this layout to the exact PHINO loader contract.

Do not change the contract unnecessarily.

If you produce separate train/evaluation directories to prevent leakage, document them precisely.

Generated data and images should remain on Brown unless there is a clear reason to publish a small representative artifact.

Do not commit large binary datasets to GitHub.

---

# 13. IMPLEMENTATION STRATEGY

Work in small, verifiable steps.

## Phase A: Environment and repository inspection

First inspect:

- Current working directory.
- Current Git branch.
- Uncommitted changes.
- Git remotes.
- Relevant Python environment.
- PHINO checkout.
- Existing generated datasets.
- Available dependencies.
- Whether execution is occurring on a login or compute node.

Do not overwrite local changes.

Use the existing `(simpact)` environment where appropriate.

Do not install a large new environment unless necessary.

Inspect source code before modifying it.

## Phase B: Dataset contract validation

Verify that the existing grid matches the expected schema.

Inventory all configurations.

Identify any incompatibilities between the two repositories.

Write or update a short implementation plan based on the actual data.

Avoid extensive speculative planning.

The aim is to begin useful implementation quickly.

## Phase C: Adapter and unit tests

Implement manifest loading, validation, output mapping, and provenance.

Use small synthetic fixtures to test:

- Accepted complete fingerprints.
- Incomplete configurations.
- Missing files.
- Duplicate identifiers.
- Incorrect shapes.
- Nonfinite values.
- Incorrect action order.
- Incorrect metadata.
- Deterministic output ordering.

Do not require a GPU for the basic tests.

## Phase D: Image rendering

Render one object successfully.

Validate the image.

Then render the remaining relevant objects.

If the existing PHINO rendering helper fails, diagnose the actual incompatibility.

Add the smallest necessary adapter.

Do not rewrite the scene generator or reimplement the entire renderer without evidence that this is necessary.

## Phase E: Pairwise similarity data

Fit the appropriate four-action normalizer.

Generate pairwise similarity labels.

Verify their numerical properties and pair-index integrity.

## Phase F: PHINO dataloader smoke test

Load the exported dataset using the real PHINO dataloader.

Begin with:

- No hard mining.
- A small number of precomputed pairs.
- A small batch.
- One worker if that reduces debugging complexity.

Validate:

- Images load successfully.
- Image tensor shapes are correct.
- Fingerprint references resolve.
- Metadata aligns with actual physics records.
- Pairwise similarities are finite.
- The batch has the expected keys and tensor shapes.
- The dataset does not rely on hardcoded paths from another machine.

**This is the first major success condition.**

Do not begin full training merely because the dataloader works.

## Phase G: Documentation and handoff

After successful validation:

- Update the project handoff.
- Record the exact dataset size and coverage.
- Document compatibility decisions.
- Record the tests performed.
- Save commands for future reproduction.
- Commit the source changes.

---

# 14. BROWN CLUSTER OPERATING RULES

You are running on Brown's computing infrastructure.

Respect the difference between login nodes and compute nodes.

Use login nodes for:

- Git operations.
- Source inspection.
- Lightweight code edits.
- Small static tests.
- Documentation.

Use Slurm-allocated compute resources for:

- GPU rendering when needed.
- Heavy MuJoCo work.
- Large data-processing jobs.
- PHINO model training.
- Long-running numerical experiments.

Check for an existing allocation before requesting another.

Avoid interfering with active jobs.

Do not terminate unrelated sessions or tasks.

Do not assume arbitrary access to all GPUs displayed by system utilities.

Use the allocated device visibility and scheduler environment.

If a command cannot run because the agent is on the wrong node, provide a safe alternative and describe the blocker.

Do not perform long training while the research owner is expecting a simple data-interface validation.

---

# 15. ENGINEERING QUALITY STANDARDS

Prefer correct, simple implementations.

Use existing library functions.

Avoid unnecessary dependencies.

Avoid duplicate implementations of the same conversion or normalization logic.

Make failures explicit.

Do not silently skip unexpected errors.

Separate:

- Data loading.
- Validation.
- Rendering.
- Fingerprint conversion.
- Similarity construction.
- Export.
- Smoke testing.

Make behavior deterministic where possible.

Use stable object ordering and random seeds.

Avoid path-dependent assumptions.

Support reproducible reruns.

Do not overwrite existing output directories without an explicit user request.

Prefer staging output and publishing only after all required files pass validation.

Avoid leaving a partially completed dataset that appears usable.

Add tests for behavior that could corrupt scientific results.

Examples include:

- Wrong physical labels.
- Wrong trajectory-to-image association.
- Wrong geometry grouping.
- Missing actions treated as present.
- Normalizer leakage.
- Inconsistent indices.
- Reusing an old incompatible cache.
- Accidentally mixing action orders.

Be skeptical of tests that only check whether a script exits successfully.

---

# 16. EXPERIMENTAL AND SCIENTIFIC STANDARDS

A functioning pipeline is not equivalent to a scientifically valid experiment.

Before making a scientific claim, identify:

1. What was observed.
2. What variable was changed.
3. What information was available to the model.
4. What target was predicted.
5. What data were used for training.
6. What data were held out.
7. What baseline was used.
8. What metric was measured.
9. What confounding explanations remain.
10. What the result does and does not establish.

Use clear terminology.

Distinguish:

- Simulation parameters versus measured physical properties.
- Trajectory recoverability versus image-based prediction.
- Geometry transfer versus unseen-parameter transfer.
- Interpolation versus extrapolation.
- Numerical validity versus physical realism.
- Model improvement versus dataset-selection effects.
- A learned representation versus privileged simulator access.
- Data-interface compatibility versus successful representation learning.

Do not inflate results.

A negative experimental finding is valuable when the protocol is correct and the failure is informative.

Do not tune the experiment until it produces a favorable number.

Preserve the original baseline and compare changes explicitly.

---

# 17. IMPORTANT RESEARCH DESIGN RISKS

Track these issues rather than hiding them.

## Risk A: Static-image ambiguity

The same image can correspond to multiple arbitrarily assigned physical configurations.

The model cannot infer which hidden assignment generated an identical image without additional information.

This is the central concern for future visual training.

## Risk B: Geometry-dependent response

The same external force can produce very different motions for different shapes.

The sphere exposed this directly.

A model may learn shape-specific relationships rather than transferable physical representations.

## Risk C: Simulated parameter semantics

Mass and friction are assigned by the simulator.

The friction parameter may not be equivalent to experimentally measured material friction.

Do not interpret it as such.

## Risk D: Very small dataset

Seven geometries and 63 parameter combinations are insufficient to support broad claims about world-model pretraining.

The actual number of complete four-action fingerprints is even smaller.

The dataset is currently a feasibility and integration resource.

## Risk E: Missingness

Trajectories are systematically rejected in some regimes.

The missingness depends on the object and action.

This can distort evaluation if ignored.

## Risk F: Training-label construction

Trajectory similarities depend on action choices, normalization, and the resulting motion distribution.

They are not universal measures of physical equivalence.

## Risk G: PHINO contrastive-loss semantics

The inspected SoftInfoNCE implementation does not use a fully measured dense batchwise dynamics-similarity target.

It uses diagonal pair similarity values within its soft-label construction.

This may matter for a future objective-level evaluation.

Do not silently modify it during data integration.

## Risk H: Data leakage

Repeated physics configurations and multiple views of the same geometry must not cross geometry-level evaluation boundaries.

Fitting normalization on all objects would invalidate a held-out-geometry evaluation.

## Risk I: Engineering success mistaken for research success

The ability to render images and load a batch is useful.

It does not prove that the representation contains meaningful hidden physics.

Keep these milestones separate.

---

# 18. WHAT NOT TO DO RIGHT NOW

Do not:

- Start another sphere torque ablation.
- Rerun the seven-geometry physics grid unnecessarily.
- Change the existing action schedules.
- Increase the mass/friction grid without a research reason.
- Rewrite PHINO's contrastive objective.
- Rewrite the DINOv2 backbone.
- Launch the default full PHINO training schedule.
- Treat the Brown memory document as an implementation specification.
- Fit a normalizer using held-out evaluation objects.
- Mix rejected and accepted trajectories without documenting the policy.
- Fabricate missing physical values.
- Introduce a new simulation engine.
- Over-engineer the export system into a large generic framework.
- Produce unsupported claims about real-world physical intelligence.
- Spend excessive time polishing peripheral infrastructure.

We are deliberately moving away from endless preparatory work toward a concrete research milestone.

---

# 19. PROJECT MANAGEMENT AND GITHUB WORKFLOW

Cursor is the primary implementation agent.

The human research owner determines scientific priorities.

ChatGPT may periodically review scientific decisions, implementation results, and experimental findings.

Do not require ChatGPT approval for ordinary coding work.

## 19.1 Repository hygiene

Before editing:

- Inspect the current Git status.
- Confirm the current branch.
- Confirm the repository remotes.
- Preserve uncommitted work.
- Avoid destructive reset operations.

Commit focused work with descriptive messages.

Examples:

`feat(phino): export complete SIMPACT trajectories into training dataset`

`test(phino): verify four-action fingerprint conversion and geometry splits`

`fix(phino): resolve isolated mesh assets during image rendering`

`docs(phino): record dataset coverage and PHINO loader validation`

Push only to the appropriate user-owned repository and branch.

Do not push directly to upstream third-party repositories.

If modifying PHINO itself becomes necessary, use a dedicated branch and separate focused commit or pull request in the PHINO fork.

## 19.2 Progress documentation

Maintain:

`docs/phino_cursor_handoff.md`

This is the current compact project-state document.

After a major milestone, update:

- What was completed.
- What was verified.
- Which commits contain the work.
- What remains broken.
- Important scientific caveats.
- The next concrete task.

Do not turn this file into an ever-growing transcript.

Keep detailed experiment history separately.

If a real experiment is conducted, maintain:

`docs/phino_experiment_log.md`

Record:

- Date.
- Research question.
- Hypothesis.
- Dataset.
- Code revision.
- Protocol.
- Baseline.
- Results.
- Limitations.
- Conclusion.
- Next decision.

Only create this additional file when there is useful experiment history to record.

## 19.3 What belongs in GitHub

Commit:

- Source code.
- Tests.
- Configuration files.
- Reproduction scripts.
- Small result summaries.
- Documentation.
- Provenance schemas.
- Research decision records.

Do not commit:

- Large generated datasets.
- Model checkpoints.
- Private credentials.
- API keys.
- SSH keys.
- Raw cached assets that do not belong in version control.
- Full copies of the personal Brown research memory document.
- Large temporary logs.

Use paths and hashes to identify larger artifacts stored on Brown.

## 19.4 Reporting progress

Do not report only that a task is finished.

For every major milestone, state:

- What changed.
- Why it was necessary.
- How it was tested.
- What worked.
- What did not work.
- The exact data and environment used.
- The new commit or PR.
- The remaining blockers.
- The recommended next action.

If something could not be validated, say so explicitly.

Do not claim that tests passed unless they actually ran.

---

# 20. HOW TO COLLABORATE WITH THE RESEARCH OWNER

The research owner values deep understanding and independent reasoning.

Communicate concisely during routine implementation, but preserve technical depth when explaining important decisions.

Use direct language.

Define unfamiliar terms where necessary.

Challenge unsupported assumptions respectfully.

Do not agree with a research idea merely because the owner suggested it.

When an approach is weak, identify precisely why.

Offer a better formulation or discriminating experiment.

For active problem-solving discussions, avoid unnecessarily revealing every implementation detail before the owner has an opportunity to reason through it.

However, when explicitly asked to implement something, perform the task.

Do not introduce needless permission checkpoints for ordinary engineering work.

Escalate only decisions that materially affect correctness, scientific validity, reproducibility, or resource use.

Do not optimize for the appearance of productivity.

Optimize for real research progress.

---

# 21. HOW TO HANDLE THE BROADER BROWN RESEARCH CONTEXT

The Brown memory document contains additional research directions.

Some may be relevant later:

- Self-supervised representation learning.
- World models.
- Predictive representations.
- Planning and control.
- Object-centric physical representations.
- Latent actions.
- Task-relevant physical state.
- Informative interventions.
- Representation generalization.
- Evaluation of what information embeddings preserve.

These connections can inform scientific discussion.

They should not automatically expand the implementation scope.

If you identify a promising connection between PHINO and another research thread, record it as a possible idea and explain what evidence would make it worth pursuing.

Do not independently start implementing unrelated research ideas.

The current priority is integration.

The broader aspiration is a strong, focused scientific contribution.

---

# 22. AFTER THE DATASET ADAPTER SUCCEEDS

Once the adapter and dataloader smoke test work, stop and report the result.

Do not automatically begin large-scale training.

At that point, the research owner should choose the next scientific experiment.

The key decision will likely be among several approaches.

### Direction A: Static-image pipeline baseline

Run a tightly controlled baseline using the existing PHINO static-image architecture.

Purpose:

Establish empirical behavior and limitations of the original training setup.

Important:

Because our assigned physical labels are hidden from the static input, this is not expected to recover arbitrary instance-specific physics.

The baseline may still help identify what the architecture can learn about geometry or visible physical priors.

### Direction B: Action-conditioned visual representation

Provide visual observations before and after a known intervention.

Potential inputs:

`(initial image, action, response image sequence)`

Purpose:

Test whether the representation captures physical differences actually visible through interactions.

This direction is more directly aligned with physical system identification.

### Direction C: Task-relevant response representations

Instead of predicting all physical parameters, learn representations preserving differences that matter for a downstream task or controller.

For example, two objects may have different precise masses but behave equivalently for a given action policy.

Another pair may look similar but require different control decisions.

Purpose:

Investigate what physical information a representation must preserve to support intelligent action.

This is a larger research direction and needs a carefully defined task and evaluation.

### Direction D: Better generalization tests

Develop stronger evaluations of transfer across:

- Object geometries.
- Physical parameter ranges.
- Applied actions.
- Motion regimes.
- Contact configurations.

Purpose:

Determine whether learned representations capture transferable dynamics rather than fitting particular objects or interventions.

This should follow a clear model and experimental hypothesis, not merely more data generation.

**Do not choose among these directions without discussion with the research owner.**

---

# 23. IMMEDIATE EXECUTION CHECKLIST

Begin by completing the following sequence.

### Step 1: Read the current project instructions

From the SIMPACT root, inspect:

- `AGENTS.md`
- `docs/phino_cursor_handoff.md`
- `docs/phino_pretraining_integration_contract.md`

Read the broader Brown memory only where relevant to motivation or research direction.

### Step 2: Inspect both repositories

Confirm:

- Current branches.
- Local changes.
- Source versions.
- Python environment.
- Relevant modules.
- Existing test commands.

### Step 3: Inspect the actual generated data

Verify:

`results/phino_mesh_grid_cutoff_1m_v1/manifest.json`

Count eligible complete fingerprints.

Inspect image and isolated-scene availability.

Do not guess dataset counts.

### Step 4: Verify PHINO's exact input schema

Read the real loader, generator, normalizer, and rendering functions.

Reconcile any differences from the documentation.

### Step 5: Implement the minimal exporter

Write:

`scripts/export_phino_dataset.py`

or an equivalent well-justified implementation.

Add unit tests.

### Step 6: Render and validate images

Use isolated scenes.

Check output quality and file integrity.

### Step 7: Generate metadata, normalizer, and pair similarities

Enforce deterministic output and geometry-grouped splitting.

### Step 8: Run the real PHINO dataloader

Fetch at least one valid batch.

Verify all expected keys and shapes.

### Step 9: Report and commit

Provide a concise technical report and descriptive commits.

Update the compact project handoff.

Stop before full training.

---

# 24. ACCEPTANCE CRITERIA

The immediate task is complete only when all of these conditions hold:

1. The original SIMPACT grids remain unchanged.
2. Every exported configuration corresponds to a valid accepted complete fingerprint.
3. The number and reasons for excluded configurations are reported accurately.
4. All exported fingerprint arrays have the expected shape and contain finite values.
5. The image files exist, are readable, and correspond to the correct object geometries.
6. Metadata indices, physical labels, image groups, and fingerprint paths are consistent.
7. The four-action normalization scheme is correct.
8. Similarity files contain valid, reproducible pairs and finite similarity values.
9. Dataset-level training statistics exclude held-out evaluation geometries.
10. The real PHINO dataloader can load a batch successfully.
11. Unit tests cover the main format and integrity failure cases.
12. The export process is reproducible and records adequate provenance.
13. Generated artifacts do not overwrite prior experiments.
14. Source changes are committed with descriptive messages.
15. Important scientific limitations are documented.
16. No unsupported PHINO-training or physics-inference claim is made.

At completion, report the exact output directory, files, counts, tests, and commit hashes.

---

# 25. FIRST RESPONSE TO THE RESEARCH OWNER

Before making extensive changes, provide a concise initial assessment that includes:

1. Confirmation that you found both repositories and the generated grid.
2. The current Git branch and whether there are local changes.
3. The number of usable full fingerprints, if you can determine it quickly.
4. Any material incompatibilities between SIMPACT and PHINO.
5. A short implementation plan.
6. Any genuine scientific design decisions that require approval.

Then begin implementation.

Do not turn this into a long speculative planning session.

Do not re-explain the entire research history unless asked.

Do not restart already completed simulation experiments.

---

# 26. FINAL RESEARCH PRINCIPLE

The objective is not merely to generate a large dataset, train a neural network, and report a favorable metric.

The objective is to understand **what physical information can be learned, from which observations, under which interventions, and whether that information transfers to new situations or helps control decisions.**

Our work so far indicates that controlled trajectories contain information about assigned physical parameters, but naive transfer across geometries can fail.

The next stage is to connect those dynamics to visual representation learning while respecting what the visual observations actually reveal.

Work toward that goal.

Remain skeptical of unsupported claims.

Preserve the evidence.

Prefer one decisive, well-designed experiment to many loosely motivated experiments.

**Begin with repository inspection and SIMPACT-to-PHINO dataset integration.**

**Your immediate success condition is a reproducible, validated PHINO dataloader batch built from real SIMPACT-generated data on Brown.**

After reaching that milestone, stop, document the result, and present the next scientific decision to the research owner.
