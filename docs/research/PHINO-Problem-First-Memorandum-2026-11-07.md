# PHINO / SIMPACT: Problem-first research memorandum

**Evidence cutoff:** November 7, 2026.  
**Decision:** Conditional GO for one small falsification study. NO-GO for a new representation-learning method at present.

## 1. Recommendation

Investigate **whether learned world models can choose reliable contact actions for a new goal when the training data came from a different behavior policy**.

Use the released **AnyGoal Insert-Peg → Extract-Peg** setting as the first diagnostic. Start with its strongest tested visual baseline on Extract-Peg, **Sensorimotor World Model (SMWM)**. Do not start with PHINO, a new loss, or an intentionally weak DINO baseline.

This is the best *investigation*, not an established novelty opportunity. The important remaining question is whether the failure survives controls for:

1. Missing velocity or observation history.
2. Inadequate coverage of the relevant contact regime.
3. A goal-distance function that ranks physically wrong states as good.
4. Long open-loop action execution or inadequate optimization.
5. Standard structured dynamics, system identification, and direct goal-conditioned control.

**None of the three candidates examined below currently establishes that a fundamentally new learning principle is necessary.** AnyGoal provides the best public starting point for finding out. If ordinary state estimation plus mechanics, a standard probabilistic predictor, or existing structured dynamics solves the matched problem, reject the learning-method direction.

The potential scientific question is narrower than “world models do not understand physics”:

> Which information and interaction coverage let an agent reuse learned contact dynamics under a change of goal, and which observed decision failures remain after those requirements are met?

This connects physical prediction to decisions. It also permits an informative negative result. It does not require preserving single-image inference, contrastive learning, or PHINO.

## 2. Sources and scope

I read the supplied Brown Academics memory and the required SIMPACT files. The verified branch head was **53a0ca1716d23f58dada0e4a603d220ac1f8b281**, dated November 7, 2026. I read the matched-baseline summary, the current handoff and log, and the reciprocity report. All required preliminary sources were accessible.

For the three shortlisted capabilities, I examined the primary papers, experimental protocols, relevant appendices, and public resource manifests. I traced competing work on factored dynamics, contact models, goal-conditioned control, belief-space planning, and dynamic robot execution. I checked current paper versions and release availability. This is a targeted research investigation, not a claim to have exhaustively read every publication through the cutoff.

**Evidence labels in this memo:**

- **Reported:** An author's experiment or the supplied project record. Not independently reproduced here.
- **Verified implementation fact:** Found in source code, configuration, or an artifact manifest.
- **Deduction:** Follows from stated assumptions or an explicit mathematical example.
- **Hypothesis:** A plausible explanation requiring an experiment.

No training, simulation campaign, or benchmark evaluation was run for this memorandum. Public artifact listings were checked; large checkpoints were not downloaded or executed.

### What the project evidence establishes

| Investigation | Established within its tested setting | Not established |
|---|---|---|
| PHINO matched objectives | With five held-out geometries, two training seeds, and the recorded budget, trajectory-similarity supervision did not consistently beat matched instance discrimination. Its advantage over frozen DINOv2 does not isolate the value of physical similarity labels. | A universal equivalence of objectives; failure of all interaction supervision; inability to learn useful physical representations. |
| Spatial/position diagnostic | Spatial features helped some property probes, but simple post-interaction position tracking remained much stronger. | A pure pooling bottleneck, a general physical-understanding failure, or a pre-interaction image-only solution. |
| Elastic reciprocity | Sparse probes predicted useful unexecuted responses in the small-load regime. Geometry changed the optimal action. A mechanics baseline using the drawing and declared spring rate nevertheless achieved zero regret. | A benefit from learning; continuum-bracket generalization; usefulness of reciprocity outside its assumptions. |
| OneWorld audit | The recorded finite/Gaussian examples expose approximation or guidance failures that standard probabilistic modeling can address. | A measured failure of a released OneWorld checkpoint or a new probabilistic principle. |

The Phase 1 comparison used initial **and post-action** images. A tracker exploiting visible displacement is a legitimate baseline there. It is not evidence that mass or friction can be recovered from an initial image whose appearance is independent of those variables.

The matched, geometry-equal four-intervention differences for pooled features between trajectory and generic contrastive objectives were approximately −0.001 mass and −0.005 friction nMAE. Those differences were small relative to seed variation. The approximately 0.04/0.12 tracker errors are a different reference point: they show that the interaction observations contain useful information that the tested learning pipeline does not extract well. They do not identify which component causes the loss. [P1–P4]

For reciprocity, use the repository's detailed tolerances. The latest report contains a roughly 1.3% relative mismatch for one L-shaped case; “all at most 1.1%” is not the safest summary of the full record. This does not change the zero-regret baseline or the NO-GO decision. [P4]

**Abandon:** “physics labels must help,” “complexity implies a learning opportunity,” and “a finite failure case implies a new principle.” Retain the interest in physical decisions, but require a failure against the strongest appropriate alternative.

## 3. Literature synthesis

The literature suggests three distinct capability gaps. Mixing them would make diagnosis difficult.

| Rank | Capability to investigate | Concrete evidence | Main unresolved diagnostic |
|---|---|---|---|
| 1 | Reuse contact dynamics for a changed goal, beyond the demonstrator's behavior | AnyGoal reports substantial contact-task failures across visual and selected state predictors. | Is the failure due to learnable dynamics, or observation, coverage, cost, and planner choices? |
| 2 | Gather physical information while avoiding irreversible side effects | RoboQuest agents often commit before gathering evidence or disturb the scene while inspecting it. | Does a standard belief planner with the same learned perceptual interface already solve the task? |
| 3 | Anticipate and act in a scene that moves during inference | RoboDyna and ReflexBench expose failures in interception, timing, and dynamic contact. | Do tracking, adequate history, latency handling, and standard predictive control remove the gap? |

Several broader results constrain the interpretation:

- **V-JEPA 2** demonstrates that interaction-trained latent models can support image-goal robot planning. Thus, “interactions teach reusable visual representations” is already established as a broad idea. It does not settle the narrower goal-shifted contact question. [L16]
- **Dreamer 4** demonstrates substantial offline imagination-based control in Minecraft. A failure of a small manipulation model cannot support a universal claim about world models. Conversely, Minecraft performance is not a direct test of robot contact transfer. [L17]
- **OGBench** provides direct goal-conditioned alternatives, including CRL, HIQL, GCIQL, and behavior cloning. A predictor-based solution must be compared against competent methods that need not reconstruct the world. [L15]
- **MoCoDA, ContactNets, and graph simulators** already provide principles for local composition and physical structure. A new “mechanism-aware” representation must offer more than reintroducing their assumptions. [L3–L6]
- The recent **MMBench2** study finds gains from coverage-aware sampling and additional interaction. Such results make data coverage a serious alternative explanation. Additional rollouts are additional information, not evidence that a representation objective is better at equal cost. [L7]

I did not select RoboTwin-Phys as a first experiment. Its reported physical/randomization degradation is interesting, but the inspected release still marked data as TBD and code as forthcoming. Its perturbations also mix physical and observation changes. DLO-Lab is useful infrastructure, but strong analytical optimization in its reported tasks weakens a generic claim that those tasks require a new representation. [L20–L21]

## 4. Candidate 1: Contact decisions under a change of goal

### Concrete capability and evidence

A robot trained on insertion should be able to use what it learned about the same arm, peg, and slot to extract or reposition the peg. This is relevant to recovery, assembly, and reuse of experience for new goals.

The initial study concerns changed goals in the same physical environment. It does not establish transfer to new objects, materials, embodiments, or arbitrary new actions. Single-image inference is not imposed: use the history needed to make the action consequences observable.

**Reported evidence.** AnyGoal v2, revised October 28, 2026, compares learned and frozen-feature world models on goal variants. Selected absolute success rates from Table 12 are:

| Model | Insert-Peg | Extract-Peg | Push-T | Stack-T |
|---|---:|---:|---:|---:|
| Sensorimotor WM | 70.0% | 24.0% | 84.7% | 32.0% |
| LeWorldModel | 26.7% | 6.0% | 84.7% | 46.0% |
| DINO-WM integration | 35.3% | 11.3% | 74.0% | 40.7% |
| State predictor, AdaLN | 74.0% | 16.7% | 92.7% | 70.7% |

These are author-reported evaluations, not matched retraining results from this investigation. Importantly, SMWM reaches **85.33% on Avoid-T**. The easier claim that all these models fail to move around an object is therefore not supported by the current version. [L1, Table 12]

### Information and reproducibility

Training supplies images and actions from the base-task data. Deployment supplies current observations and a goal image. State-based diagnostics receive additional privileged quantities and must be reported separately.

I verified public code, model files, and evaluation datasets. AnyGoal's release integrates pinned upstream model sources. The tested DINO-WM integration should not be silently equated with every configuration of the original DINO-WM paper. [R1–R5]

Three implementation facts materially affect diagnosis:

1. **Incomplete state:** the Insert-Peg state adapter uses 33 features when velocities are excluded. The underlying 44-dimensional record includes eight arm velocities and three object velocities that this reduced representation omits. Three-frame history may help infer them, but “oracle state” is not automatically a complete Markov state. [R3]
2. **Observation frequency:** the SMWM planning configuration uses five model blocks, five raw actions per block, and a receding horizon of five blocks. That means 25 raw actions before replanning. [R4]
3. **Base-data reuse:** the SMWM configuration sets training-data validation to true, and the base planning configuration names the training file. The published base score must not be called a clean held-out, within-goal reference without an episode-level split audit. [R5]

These facts do **not** prove that the benchmark is invalid or that the failure disappears. They prevent a stronger causal conclusion than the evidence supports.

### Scientific diagnosis

There are at least five competing explanations:

| Explanation | Observation that would support it | Consequence |
|---|---|---|
| Observation aliasing | The same supplied observation/history can lead to different outcomes because velocity or contact state differs. | Fix memory or belief estimation. Do not ask an image encoder to infer absent information. |
| Missing interaction coverage | Extraction requires release/slip configurations absent from training. | A data-support problem remains. More informative data may help, but that is not an objective advantage. |
| Goal-cost error | Actual future images are available, yet the learned latent distance chooses the wrong physical outcome. | Test a standard physical goal metric or goal-conditioned value function first. |
| Prediction error | Real-image scoring chooses well, but predicted-image/latent scoring does not, on the same action bank. | A dynamics or predictive-representation failure is implicated. |
| Planning/control error | Shorter action execution or a stronger optimizer succeeds without changing the model. | The initial failure does not justify a new learning principle. |

**Deduction: familiar marginal values do not make a counterfactual identifiable.**

Let training inputs satisfy \(x_1=x_2\), with both values 0 and 1 observed, and both actions \(a\in\{0,1\}\) observed. Consider two dynamics:

\[
f_A(x_1,x_2,a)=a x_1,\qquad
f_B(x_1,x_2,a)=a x_2.
\]

Both fit every training observation exactly. At \((x_1,x_2,a)=(1,0,1)\), their predictions differ. All individual coordinates and actions are familiar. The mechanism is still unidentified.

This is an elementary counterexample, **not a novelty claim**. It shows why a task's factor-wise support checks cannot establish that data alone determine the correct action effect. Known structure or additional separating observations are necessary in this example.

For contact, the relevant conditioning variables may include relative geometry, velocity, actuation, and contact mode. Coverage of each separately is weaker than coverage of the combinations on which the local transition actually depends.

### Closest competing methods and novelty threats

| Prior | Objective or mechanism | What it already addresses | What it does not automatically establish here |
|---|---|---|---|
| MoCoDA, NeurIPS 2022 [L3] | Fit locally factored transition distributions, then generate counterfactual training transitions. | Generalization to unseen global combinations under known local structure and sufficient parent-variable coverage. | Recovering unknown structure from these images or identifying unobserved contact regimes. |
| ContactNets, CoRL 2020 [L4] | Learn signed-distance/contact Jacobian functions with a complementarity-based surrogate loss. | Discontinuous impact and stiction without pretending the state-transition function is globally smooth. | Full visual state inference or arbitrary compliant/contact dynamics. It assumes known contact-free dynamics and restrictive contact physics. |
| Graph Network Simulators, ICML 2020 [L5] | Learn local particle interactions by message passing. | Reusable interaction structure and physical rollout prediction. | That a particular image-to-particle front end or policy-biased dataset suffices. |
| PIEGraph, available arXiv version May 4, 2026 [L6] | Equivariant action-conditioned graph guidance coupled to a spring–mass model. | Few-interaction prediction and planning across rigid and deformable object tasks; includes T-block manipulation. | The identical AnyGoal data, goals, observation interface, and budget. |
| WM3C, ICLR 2025 [L8] | Language-guided causal components with conditional dynamics, identifiability assumptions, and adaptation. | Compositional latent mechanisms and task transfer are already explicit learning targets. | Identification from one expert behavior without its auxiliary variation, smoothness, and conditional-independence assumptions. |
| SMWM and LeWM [L2, R1] | Forward prediction plus inverse-action supervision, or a non-collapse regularizer. | Strong action-conditioned learned representations without physical property labels. | Reliable behavior outside the interaction distribution. |
| OGBench direct methods [L15] | Goal-conditioned value learning, hierarchy, or behavior cloning. | Decisions without a general-purpose forward model. | Success in this specific transfer setup; this must be tested, not dismissed. |
| Classical mechanics plus state estimation | Estimate state/parameters and optimize actions through a physical model. | Potentially the entire task if geometry, dynamics, and state are recoverable. | A fair RGB solution if it is secretly supplied simulator state or parameters. |

MoCoDA is especially important. Its theorem needs **known local factorization** and enough samples for the relevant local inputs. AnyGoal's own theoretical discussion also conditions generalization on identified mechanisms and local support. Merely proposing to “learn reusable factors” would duplicate existing motivation.

Interaction Asymmetry provides further theory for compositional representations under restricted cross-component interactions and invertible smooth mappings. Its theorem is not a blanket guarantee for hybrid contact systems. An extension would need a new result, not just a contact example. [L9]

**Opportunity status:** The observed decision failure is credible enough to investigate. A residual failure against standard structured methods is **not demonstrated**. The potentially significant contribution would have to explain and overcome that residual under explicit, sufficient information conditions.

## 5. Candidate 2: Information gathering with irreversible physical consequences

### Concrete capability and evidence

Consider a robot that must identify a parcel by weight, inspect a mug while retaining its contents, or test a support without destroying the configuration it must later use. The needed action changes both the robot's information and the physical world.

RoboQuest's October 7 preprint reports **23.2% overall success** for its best evaluated frontier agent. Its task analysis reports premature parcel selection and failures to prevent or repair disturbances. Fine-tuned \(\pi_{0.5}\) performs poorly. The later project leaderboard lists a different best model at 23.8%; these are separate versions, not interchangeable results. [L10, R6]

A useful counterexample to an overbroad gap claim: the paper's strongest agent performs well on the covered Puzzle Box test. “Agents cannot infer any hidden mechanism” is not supported.

### Information at training and deployment

The benchmark provides demonstrations and a common visuomotor interface. Hidden properties require inspection or testing. Deployment includes visual observations, robot measurements, history, and bounded actions. Hidden object identity or weight is not simply supplied.

The benchmark, demonstrations, and assets are public. I verified the repository and dataset listing. Reproduction of closed frontier-model results still depends on API availability, model version, and inference cost. Public environment availability does not guarantee an identical proprietary model snapshot. [R6–R7]

### Diagnosis and strongest objections

The failure could come from:

- A deficient belief over what remains unknown.
- Failure to value information before committing.
- A missing model of the physical side effects of inspection.
- Weak perception or motor execution.
- A long-horizon memory/interface limitation.

The benchmark's failure attribution is useful descriptive evidence, not a randomized causal intervention separating these explanations. Some isolated manipulation skills remain unreliable. Their errors can compound over long episodes.

The standard formulation is already a partially observable Markov decision process:

\[
b_{t+1}(s') \propto p(o_{t+1}\mid s',a_t)
\int p(s'\mid s,a_t)b_t(s)\,ds.
\]

A belief-space planner can trade information against task progress and irreversible loss. This is not a new objective.

### Closest methods that threaten novelty

- **Interactive perception** has long treated action as a source of information. [L11]
- **POMCPOW/PFT-DPW** address planning with continuous states, actions, and observations using particle beliefs. These need suitable transition and observation models; their existence is not a claim that all perception problems are solved. [L12]
- **PO-PDDL**, listed by its authors as accepted to NeurIPS 2026, learns symbolic transition/observation models from demonstrations and uses online belief planning. Its reported tasks include hidden cup contents, active inspection, constraints, and irreversible spill penalties. It reports success on its five real-world tasks, with ten trials each. Its stated limitations include closed-domain assumptions and dependence on the visual-language backend. [L13]
- **Imagine-TAMP**, September 2026, combines belief shaping, visibility estimation, and non-unit task costs. In a paired shelf study, visibility-aware planning improves success from 46% to 84%. It tackles sensing-versus-manipulation choices directly. [L14]

These methods do not establish that RoboQuest is solved. They establish that the proposed general remedy is already known.

**First rejection test:** Give a standard belief planner the same observation stream, the same motor skills, and transition/observation estimates derived from the same demonstrations. Do not give it true hidden properties. If that combination solves the chosen task, reject a new representation claim. An oracle-symbolic version is only a ceiling that tests whether planning could be the bottleneck.

**Opportunity status:** High practical importance. Real generalist-agent failure. Weak evidence of a new learning gap after strong probabilistic baselines. Rank second.

## 6. Candidate 3: Anticipation under real time and dynamic contact

### Concrete capability and evidence

A robot must intercept an object, release at the right phase, or insert into a moving target while the scene continues to evolve during computation.

RoboDyna reports overall success around **17%, 11%, and 5%** for its evaluated \(\pi_{0.5}\), X-VLA, and FastWAM checkpoints. It includes held-out combinations of variations. However, those checkpoints use different training-step budgets, and the suite does not establish failure of a tuned classical predictive controller. Its aggregate sync–async difference is small, despite changes in individual outcomes. [L18]

ReflexBench adds stronger dynamic-specific comparisons. All seven reported methods remain at or below **7.3% ball-catching success**; rotating-peg insertion also remains difficult. Several other tasks are much easier. The evidence therefore supports particular dynamic-contact failures, not universal dynamic incompetence. [L19]

### Information and reproducibility

A fair deployment interface must specify observation timestamps, frame history, action duration, robot state, and measured inference latency. A single un-timestamped image does not reveal arbitrary velocity or phase. That is a missing-information problem.

RoboDyna supplies code, demonstrations, and publicly listed X-VLA and \(\pi_{0.5}\) checkpoints. I verified the model listings. The Reflex project supplies a benchmark and dataset link, but its model code was marked “after acceptance” when inspected. Do not plan a reproduction around unverified ReflexVLA weights. [R8–R10]

### Diagnosis and novelty threats

The obvious hypotheses are already heavily studied:

- Short history fails to estimate velocity.
- An action chunk becomes stale before it executes.
- A model predicts a future at the wrong time.
- Training lacks enough interception/contact examples.
- A controller cannot reach a predicted interception point.
- Future-feature prediction helps, but the remaining error is motor control rather than representation.

**DynamicVLA** already combines efficient inference with continuous execution and latency-aware action streaming. **Real-Time Chunking** and **REMAC** address asynchronous chunk consistency. **ReflexVLA** already combines latent future prediction, temporal fusion, and latency optimization. Thus, “add dynamics prediction/history so the robot anticipates” is not a new principle. [L19, L22–L24]

The simplest serious competitor is visual tracking plus a calibrated motion model and receding-horizon interception controller. Its assumptions must be explicit. If it is given velocity or object models unavailable to the learned agent, it is a diagnostic ceiling, not a matched baseline.

**First rejection test:** On a public interception task, compare the best released policy with a tracker-plus-predictive controller under identical camera streams and wall-clock constraints. Sweep latency and history separately. If ordinary tracking/control closes the failure, reject a new representation project. Only then consider whether unmodeled impacts or ambiguous future branches create a residual learning problem.

**Opportunity status:** Strong application value and usable public infrastructure. Higher systems confounding, substantial existing competition, and weaker fit to a small first learning-principle study. Rank third.

## 7. Ranking and decision

These are qualitative judgments, not publication probabilities.

| Criterion | Contact goal transfer | Information seeking with side effects | Dynamic anticipation |
|---|---|---|---|
| Scientific importance/generalization | High | High | High |
| Evidence that tested modern systems fail | Strong, conditional on protocol | Strong for evaluated agents | Strong for particular tasks |
| Evidence that appropriate standard methods still fail | **Not established** | **Not established** | **Not established** |
| Potential for a new general principle | Conditional: unknown mechanisms plus adequate local evidence | Conditional: learning transferable observation/action models | Conditional: prediction–control coupling beyond timing fixes |
| Reproducible first baseline | Strong: released AnyGoal checkpoints/configs | Medium: public environment, costly/versioned frontier APIs | Strong for RoboDyna; incomplete for some newer methods |
| Fit to Brown expertise/resources | Strong | Moderate | Moderate |
| Main novelty threat | MoCoDA, ContactNets, PIEGraph, standard goal control | POMDPs, PO-PDDL, Imagine-TAMP | DynamicVLA, ReflexVLA, RTC, classical MPC |
| Current decision | **GO only for falsification** | Hold | Hold |

AnyGoal is directly related to work by Heejeong Nam, Randall Balestriero, and collaborators. This is a scientific overlap, not a reason to assume novelty or collaboration access. The project memory also records a planned conversation with Sergio Orozco. That makes PIEGraph an accessible intellectual comparison, but no collaboration or code access is presumed.

The recommended direction could support an ICLR/ICML-level result **only if** it produces a general result about when physical action effects are identifiable and reusable, together with a method that improves matched strong baselines across more than one task. A benchmark patch, a longer history window, or one successful extraction controller would not by itself meet that ambition. A carefully isolated robotics result could be useful for CoRL without establishing a universal learning principle.

## 8. Exact first falsification study for Cursor

### 8.1 Scientific question

> On the released Extract-Peg task, does SMWM select inferior contact-action sequences because its predicted action consequences are wrong, after holding the candidate actions fixed and separating prediction from goal scoring? Does the failure remain when standard inference changes and available-information checks are applied?

This is deliberately a **checkpoint capability audit**. It is not yet a causal comparison of training objectives.

**Pre-registered working hypothesis H:** For a substantial fraction of test states with an attainable goal in a common candidate bank, SMWM selects a failing sequence while scoring the *actual* resulting images with the same latent goal metric selects a successful sequence.

If H fails, there is no evidence from this test for the proposed predictive-learning bottleneck.

### 8.2 Pinned resources

| Resource | Verified identifier |
|---|---|
| AnyGoal repository | [https://github.com/galilai-group/AnyGoal](https://github.com/galilai-group/AnyGoal) |
| AnyGoal code revision | f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e |
| SMWM upstream source revision in release manifest | 9d22bcc15c6ad8af8cc93337a950df2b49a866b9 |
| AnyGoal model repository | [https://huggingface.co/HazelNam/anygoal](https://huggingface.co/HazelNam/anygoal) |
| Model revision | b27f9dac9c750410d1331d258a95eff0d56570c2 |
| AnyGoal dataset repository | [https://huggingface.co/datasets/HazelNam/anygoal](https://huggingface.co/datasets/HazelNam/anygoal) |
| Dataset revision | 1719e7dab2fbeb07a0d5c3cf9cd881bc49510c88 |
| Main checkpoint | sensorimotor_insertpeg.ckpt |
| Diagnostic state checkpoints | toy-adaln_insertpeg.pt; toy-action-duplicate_insertpeg.pt |
| Frozen-feature comparator, if needed | prejepa_insertpeg.pt |
| Training/reference data | manipulator_insert_peg_weak_expert_train.h5 |
| Primary test data | extract_peg_test.h5 |
| Easier control data | avoid_peg_test.h5 |

I verified that these files are listed publicly. Runtime compatibility and file checksums still need verification after download. The commands below follow the released CLI, but were not executed here.

### 8.3 Installation and immutable inputs

Create a new audit directory. Do not change SIMPACT's existing output directories.

~~~bash
git clone https://github.com/galilai-group/AnyGoal.git anygoal-audit
cd anygoal-audit
git checkout f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e

uv venv --python 3.10
source .venv/bin/activate
uv pip install -e '.[eval]'

anygoal --version
anygoal list datasets
anygoal list models
anygoal models install sensorimotor
anygoal config path data
anygoal config path checkpoints
~~~

Use the pinned Hugging Face revisions rather than downloading an unrecorded future main branch. This small script downloads only the needed files into AnyGoal's resolved directories:

~~~python
from pathlib import Path
import shutil
import subprocess
from huggingface_hub import hf_hub_download

model_rev = "b27f9dac9c750410d1331d258a95eff0d56570c2"
data_rev = "1719e7dab2fbeb07a0d5c3cf9cd881bc49510c88"

def target_dir(kind):
    result = subprocess.check_output(
        ["anygoal", "config", "path", kind], text=True
    ).strip()
    directory = Path(result)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

specs = [
    ("model", model_rev, target_dir("checkpoints"), [
        "sensorimotor_insertpeg.ckpt",
        "sensorimotor_insertpeg.json",
        "toy-adaln_insertpeg.pt",
        "toy-adaln_insertpeg.json",
        "toy-action-duplicate_insertpeg.pt",
        "toy-action-duplicate_insertpeg.json",
    ]),
    ("dataset", data_rev, target_dir("data"), [
        "manipulator_insert_peg_weak_expert_train.h5",
        "extract_peg_test.h5",
        "avoid_peg_test.h5",
    ]),
]

for repo_type, revision, directory, filenames in specs:
    for filename in filenames:
        destination = directory / filename
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite {destination}. "
                "Use an isolated AnyGoal home or verify the existing hash."
            )
        downloaded = hf_hub_download(
            repo_id="HazelNam/anygoal",
            repo_type=repo_type,
            revision=revision,
            filename=filename,
        )
        shutil.copy2(downloaded, destination)
~~~

Record dependency versions, CUDA/GPU details, file SHA-256 hashes, resolved configs, and upstream source revisions. Use the repository's requirements as the compatibility reference. Do not conceal a changed dependency as a replication.

### 8.4 Stage A: Validate and reproduce

1. Run a two-episode smoke test in a fresh output directory. Confirm checkpoint loading, image/action normalization, and state restoration. Do not treat these episodes as scientific results.
2. First run the native published planning setup for Insert-Peg, Avoid-Peg, and Extract-Peg with 50 evaluations at seed 0. Save episode identities, initial states, goal states, start indices, successes, and four physical goal errors. Expand to the published seeds 0, 1, and 2 only if the initial failure survives the action-bank diagnostic. This avoids spending the full replication budget on a direction that an early control rejects.
3. Compare against the paper's SMWM references: 70%, 74%, and 24%, respectively. These are reference targets, not guaranteed exact reproducibility.
4. Audit whether the base evaluations overlap training episodes. If they do, label them “base-task reference on training source,” not held-out generalization.
5. If the visual failure survives, evaluate the released state predictors on the same test instances as diagnostics. Retain their original feature interface. **Do not feed a 44-dimensional vector into a checkpoint trained on 33 features.**

Example invocation:

~~~bash
anygoal eval planning --model sensorimotor --dataset extractpeg --seed 0 -- output.directory=/absolute/new-audit/reproduction/smwm/extractpeg
~~~

Repeat for the other datasets/seeds with distinct output directories. The extra controls require an audit script; they are not claimed to exist in the upstream CLI.

**Critical evaluation detail:** use the dataset-goal criterion for Extract-Peg, not the native Insert-Peg reward. The released adapter checks hand and object pose against the target state. Report both task-relevant object success and the full benchmark criterion, without silently changing which is primary.

If the aggregate differs from the reference by more than 10 percentage points, first investigate version, normalization, episode sampling, and metric differences. That threshold is an audit trigger, not a statistical proof that a reproduction failed.

### 8.5 Stage B: Fixed action-bank diagnostic

Start with the first 50 distinct Extract-Peg episodes in a recorded deterministic ordering. Use a start index that permits the required context and a 25-step continuation; use index 10 where valid. Record exclusions before evaluating models. Report if this population differs from Stage A.

For each restored state, construct a fixed bank of 16 valid 25-step action sequences:

- One recorded continuation from that evaluation episode.
- Seven perturbations of that continuation, with Gaussian noise scaled by 0.1 times each action dimension's training standard deviation, clipped to valid limits.
- Eight contiguous 25-step action chunks drawn deterministically from the base training file.

Use bank seed 0. The recorded continuation is deliberately privileged **candidate construction for a diagnostic**, not a deployable controller or training example. Every scoring method receives the same action candidates. The goal is to isolate ranking when a feasible action is available.

Execute each candidate once from the same restored state. This is **20,000 raw evaluation steps** for 50 × 16 × 25, plus a small deterministic-restoration check. Do not use these results to train any model.

Verify that replaying the recorded continuation reaches its recorded goal under the restoration procedure. If fewer than 90% do so, diagnose replay/goal/context validity before interpreting model scores. Do not replace failed examples until they look favorable.

Score the common bank in three ways:

- **P: Predicted-latent cost.** The frozen model predicts each sequence and applies its native goal cost.
- **O: Observed-latent cost.** Encode the actual simulated candidate futures and apply exactly the same goal cost and time aggregation.
- **T: True task utility.** Use the evaluator's physical success criterion and recorded pose errors.

O and T are privileged diagnostics. They are not deployable alternatives and must not appear in a “matched RGB baseline” column.

Let \(u_i(a)\in\{0,1\}\) be true success and \(\hat a_i\) the selected candidate. Define:

\[
R_i=\max_{a\in\mathcal A_i}u_i(a)-u_i(\hat a_i).
\]

This is opportunity regret: it is one only when the bank contains a successful sequence but the scorer selects a failing one. Report bank solvability separately. Also report the four physical errors, candidate ties, and selection agreement.

Interpretation:

- P fails, O succeeds: incorrect predicted consequences are implicated.
- P and O fail, T succeeds: latent goal scoring is implicated.
- T finds no success: this bank cannot diagnose decision ranking for that state.
- State predictors fail too: visual encoding alone cannot explain all the failures, but missing state/history still must be checked.

This diagnostic avoids comparing incomparable latent and pixel MSE values.

### 8.6 Stage C: Small inference-only controls

Apply these only if Stage B finds a clear failure.

**Replanning frequency:** Compare execution of 25 raw actions with execution of 5 raw actions before replanning. Keep the 25-step prediction horizon.

Under a 50-step episode, the native setup allows up to:

\[
2\times300\times30\times5=90{,}000
\]

predicted model-block transitions. A matched-query alternative uses ten replans, 300 candidates, six optimization iterations, and five predicted blocks:

\[
10\times300\times6\times5=90{,}000.
\]

Keep the elite fraction consistent. Count actual calls, including early termination. Equal model-query counts do not imply equal wall time, so record both. This is one declared alternative, not an unlimited hyperparameter search.

**State and context audit:** Compare the actual model input with recorded velocities, contact/touch signals, and recent actions. Quantify whether similar input histories hide materially different velocities/outcomes. The presence of velocity in the dataset does not mean the visual model received it. If the interface is insufficient, stop the representation claim.

**Contact-coverage audit:** Predefine diagnostic features: hand–peg and peg–slot relative poses, relative velocities, touch/contact regime, and action. Compare test queries with the training distribution using standardized nearest-neighbor distances and regime counts. Use a leave-one-episode-out training reference to set a 95th-percentile distance threshold. This is an operational coverage check, not proof of causal sufficiency or global support.

**Planner ceiling:** A small true-simulator search with the same action limits and horizon can test whether the optimizer itself is adequate. It is a privileged ceiling, not a deployable baseline. The fixed-bank test should precede this more expensive check.

### 8.7 Equal information, supervision, and compute

| Comparison | Information allowed | Training/data budget |
|---|---|---|
| Frozen visual checkpoint variants | Their native visual input and identical goal/candidate actions | No new training or training rollouts |
| Observed-latent/true-utility diagnostics | Counterfactual evaluation futures or true states | Evaluation only; explicitly privileged |
| Released state predictors | Their documented state/history input | Existing checkpoints; separate information tier |
| Later direct goal predictor/value baseline | Same images, actions, goals, and rollout-derived labels as a proposed method | Same source episodes, simulator calls, and declared tuning budget |
| Later structured/physical baseline | Same recoverable state and declared parameters, or clearly labeled extra privilege | Same training trajectories; parameter fitting counts as training |

Direct utility, success, outcome, contact-event, or property labels derived from a recorded simulation are available to **all** methods in the corresponding information tier. They do not cost another physical interaction simply because PHINO would encode them differently.

A method needing new probes must be compared against alternatives given the same probe outcomes. A gain from new data is a data-collection result until a matched-data comparison shows otherwise.

For this first study, use one available Brown GPU and CPU simulation, with a provisional **eight GPU-hour evaluation cap after setup**. This is a stop budget, not a runtime prediction. Report incomplete coverage if exceeded; do not shrink the test silently or label an incomplete run a negative result.

### 8.8 Quantitative decision gates

These thresholds are **proposed research-management criteria**, not numbers taken from a paper.

Use episode-level bootstrap confidence intervals. Multiple optimizer seeds on one episode are not independent environment samples. A single checkpoint does not establish consistency across training seeds.

**GO to further standard-baseline diagnosis only if:**

1. Replay and metric checks pass; the common candidate bank contains success on at least 80% of the 50 test states.
2. P has mean opportunity regret at least 0.20, with its 95% interval lower bound above 0.10.
3. Replacing prediction with actual futures under the same latent cost reduces regret by at least 0.15; the paired interval excludes zero.
4. The inference-only controls do not remove most of the failure.
5. There is a meaningful subset with adequate observation history and operational local support. Report its size and uncertainty; do not extrapolate from a few selected cases.

**NO-GO for the proposed predictive-learning direction if:**

- The strongest checkpoint already succeeds adequately and has low opportunity regret.
- The failure is a state-restoration, normalization, goal-metric, or evaluation bug.
- Goal-cost replacement or ordinary replanning closes the gap.
- Required information is absent from the observation/history.
- Successful behavior depends on an unobserved contact mechanism and the available data do not distinguish plausible alternatives.
- The apparent failure disappears under a competent standard structured model, mechanics-plus-estimation pipeline, or direct goal-conditioned method.

**Passing this audit still does not authorize a new method.** Before developing one, a bounded follow-up must test standard complete-state/recurrent predictors, a structured contact model, a mechanics baseline with estimated parameters where applicable, and a direct goal-conditioned baseline. They must share the relevant data and information tier. A standard baseline within five percentage points of the applicable task ceiling, or removing at least 80% of the observed gap, is a practical stop signal.

A residual failure should recur on at least one other contact family before claiming a general learning problem. Stack-T is a possible second diagnostic, not an automatic next campaign.

## 9. What could justify a genuinely important contribution?

The promising unresolved possibility is **learning transferable action effects when the interaction structure is unknown but the data contain enough evidence to identify it**. This is a hypothesis about a residual problem, not a demonstrated gap.

A significant result would need to establish all of the following:

1. Which structural or intervention assumptions make the decisions identifiable.
2. Why appropriate existing predictors and structured methods still fail under those assumptions.
3. A learning mechanism that removes that failure without additional hidden supervision or simulator access.
4. Transfer across goals, configurations, and at least one task family not used to tune the method.
5. A measurable decision gain, not only nicer embeddings or lower reconstruction error.

If the first audit instead points to missing information, sparse intervention coverage, or ordinary control deficiencies, accept that answer. Do not rename it a representation problem.

**Final recommendation:** Spend the next unit of effort falsifying the contact-transfer diagnosis. Keep the other two directions as alternatives. Do not launch Phase 2 learning merely because several modern models score poorly.

## 10. Primary sources and implementation links

### Project evidence

- **[P1] Brown Academics memory**, read November 7, 2026: [https://docs.google.com/document/d/1xuTJs3twcxWM6SPshdHZxLV-aqu5ymrclGaH5JWcXDY/edit](https://docs.google.com/document/d/1xuTJs3twcxWM6SPshdHZxLV-aqu5ymrclGaH5JWcXDY/edit)
- **[P2] Experiment log**, verified branch head: [https://github.com/arajshah/simpact/blob/53a0ca1716d23f58dada0e4a603d220ac1f8b281/docs/phino_experiment_log.md](https://github.com/arajshah/simpact/blob/53a0ca1716d23f58dada0e4a603d220ac1f8b281/docs/phino_experiment_log.md)
- **[P3] Matched Phase 1 baselines:** [https://github.com/arajshah/simpact/blob/53a0ca1716d23f58dada0e4a603d220ac1f8b281/docs/results/phino_action_baselines_v1/summary.md](https://github.com/arajshah/simpact/blob/53a0ca1716d23f58dada0e4a603d220ac1f8b281/docs/results/phino_action_baselines_v1/summary.md)
- **[P4] Reciprocity feasibility:** [https://github.com/arajshah/simpact/blob/53a0ca1716d23f58dada0e4a603d220ac1f8b281/docs/phino_reciprocity_feasibility.md](https://github.com/arajshah/simpact/blob/53a0ca1716d23f58dada0e4a603d220ac1f8b281/docs/phino_reciprocity_feasibility.md)
- **[P5] Current handoff:** [https://github.com/arajshah/simpact/blob/53a0ca1716d23f58dada0e4a603d220ac1f8b281/docs/phino_cursor_handoff.md](https://github.com/arajshah/simpact/blob/53a0ca1716d23f58dada0e4a603d220ac1f8b281/docs/phino_cursor_handoff.md)

### Scientific literature

- **[L1] AnyGoal: On World Model Generalization Beyond Expert Rollouts.** Nam et al. Preprint, October 19, 2026; v2 October 28. Main experiments, Appendix I/Table 12 and Appendix J examined. [https://arxiv.org/abs/2610.26844v2](https://arxiv.org/abs/2610.26844v2)
- **[L2] Sensorimotor World Models: Perception for Action via Inverse Dynamics.** Ivashkov, Balestriero, Schölkopf. June 18, 2026 preprint. Forward/inverse objective and physical-state probes examined. [https://arxiv.org/abs/2606.20104](https://arxiv.org/abs/2606.20104) ; [https://petr-ivashkov.github.io/sensorimotor-world-model.github.io/](https://petr-ivashkov.github.io/sensorimotor-world-model.github.io/)
- **[L3] MoCoDA: Model-based Counterfactual Data Augmentation.** Pitis et al. NeurIPS 2022; arXiv October 20, 2022. Theory, robotics experiment and appendix assumptions examined. [https://arxiv.org/abs/2210.11287](https://arxiv.org/abs/2210.11287) ; [https://github.com/spitis/mocoda](https://github.com/spitis/mocoda)
- **[L4] ContactNets: Learning Discontinuous Contact Dynamics with Smooth, Implicit Representations.** Pfrommer, Halm, Posa. CoRL 2020; proceedings 2021. Formulation, assumptions, cube-toss experiment and limitations examined. [https://proceedings.mlr.press/v155/pfrommer21a.html](https://proceedings.mlr.press/v155/pfrommer21a.html) ; [https://github.com/DAIRLab/contact-nets](https://github.com/DAIRLab/contact-nets)
- **[L5] Learning to Simulate Complex Physics with Graph Networks.** Sanchez-Gonzalez et al. ICML 2020, July 13–18. [https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html) ; [https://sites.google.com/view/learning-to-simulate](https://sites.google.com/view/learning-to-simulate)
- **[L6] Learning Equivariant Neural-Augmented Object Dynamics From Few Interactions (PIEGraph).** Orozco et al. Available arXiv version May 4, 2026. Mechanism, simulated planning, real-robot comparisons and analytical ablations examined. [https://arxiv.org/abs/2605.02699](https://arxiv.org/abs/2605.02699) . The RAI publication page labels CoRL 2025; this memo cites the dated arXiv version rather than resolving that metadata discrepancy.
- **[L7] Hallucination in World Models is Predictable and Preventable / MMBench2.** Hansen and Wang. June 25, 2026 preprint. Coverage-aware training and online-data distinctions examined. [https://arxiv.org/abs/2606.27326](https://arxiv.org/abs/2606.27326) ; [https://www.nicklashansen.com/mmbench2/](https://www.nicklashansen.com/mmbench2/)
- **[L8] Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning (WM3C).** Wang and Huang. ICLR 2025. Theorem assumptions, adaptation setting and limitations examined. [https://proceedings.iclr.cc/paper_files/paper/2025/hash/79d86433c2acd12b6fa98553435d226e-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/79d86433c2acd12b6fa98553435d226e-Abstract-Conference.html)
- **[L9] Interaction Asymmetry: A General Principle for Learning Composable Abstractions.** ICLR 2025; original preprint November 2024. Theorems 4.3–4.4 and scope examined. [https://arxiv.org/abs/2411.07784](https://arxiv.org/abs/2411.07784)
- **[L10] RoboQuest: Generalist Physical Agents that Search, Inspect and Test.** Liu et al. October 7, 2026 preprint. Main results, execution tests, evaluation/memory protocol and failure-attribution appendices examined. [https://arxiv.org/abs/2610.10388v1](https://arxiv.org/abs/2610.10388v1) ; [https://declare-lab.github.io/RoboQuest/](https://declare-lab.github.io/RoboQuest/)
- **[L11] Interactive Perception: Leveraging Action in Perception and Perception in Action.** Bohg et al. IEEE Transactions on Robotics 33(6), December 2017. Foundational scope reference. [https://arxiv.org/abs/1604.03670](https://arxiv.org/abs/1604.03670)
- **[L12] Online Algorithms for POMDPs with Continuous State, Action, and Observation Spaces.** Sunberg and Kochenderfer. ICAPS 2018. [https://ojs.aaai.org/index.php/ICAPS/article/view/13882](https://ojs.aaai.org/index.php/ICAPS/article/view/13882)
- **[L13] PO-PDDL: Learning Symbolic POMDPs from Visual Demonstrations for Robot Planning Under Uncertainty.** Tang et al. June 2026 preprint; authors' current page lists NeurIPS 2026 acceptance. Formulation, DESPOT planning, matched demonstration baselines, ablations and limitations examined. [https://arxiv.org/abs/2606.15654](https://arxiv.org/abs/2606.15654) ; [https://roboticsjtu.github.io/PO-PDDL/](https://roboticsjtu.github.io/PO-PDDL/)
- **[L14] Imagine-TAMP: Imagination-Guided Task and Motion Planning in Partial Observability.** Singha et al. September 2026 preprint. Belief/cost formulation, paired shelf experiment and task limits examined. [https://arxiv.org/abs/2609.20396](https://arxiv.org/abs/2609.20396)
- **[L15] OGBench: Benchmarking Offline Goal-Conditioned RL.** Park et al. ICLR 2025. [https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd92623ac899357312aaa8915853699-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd92623ac899357312aaa8915853699-Abstract-Conference.html) ; [https://github.com/seohongpark/ogbench](https://github.com/seohongpark/ogbench)
- **[L16] V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.** Assran et al. June 11, 2025. [https://arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985) ; [https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)
- **[L17] Training Agents Inside of Scalable World Models (Dreamer 4).** Hafner, Yan, Lillicrap. September 2025 preprint. [https://arxiv.org/abs/2509.24527](https://arxiv.org/abs/2509.24527) ; [https://danijar.com/project/dreamer4/](https://danijar.com/project/dreamer4/)
- **[L18] Benchmarking Robotic Manipulation under Physically Realistic Dynamics (RoboDyna).** 2026; authors list NeurIPS 2026 Workshop on Robot Learning with World Models, not the main conference. Paper, training-budget and sync–async details examined. [https://robodyna.github.io/robodyna/](https://robodyna.github.io/robodyna/)
- **[L19] Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation.** Chen et al. August 14, 2026 preprint. Task-specific results, inference/history controls and ablations examined. [https://arxiv.org/abs/2608.14379](https://arxiv.org/abs/2608.14379) ; [https://reflexvla.github.io/](https://reflexvla.github.io/)
- **[L20] RoboTwin-Phys.** September 22, 2026 preprint. Release limitations checked. [https://arxiv.org/abs/2609.26292](https://arxiv.org/abs/2609.26292)
- **[L21] DLO-Lab.** ICML 2026; June 2026 arXiv version. Analytical/learning comparisons examined. [https://arxiv.org/abs/2606.04206](https://arxiv.org/abs/2606.04206)
- **[L22] DynamicVLA.** January 29, 2026 initial preprint; current methods/results/limitations examined. [https://arxiv.org/abs/2601.22153](https://arxiv.org/abs/2601.22153) ; [https://github.com/hzxie/DynamicVLA](https://github.com/hzxie/DynamicVLA)
- **[L23] Real-Time Chunking.** Current implementation documentation, checked November 7, 2026. [https://huggingface.co/docs/lerobot/rtc](https://huggingface.co/docs/lerobot/rtc)
- **[L24] Real-Time Robot Execution with Masked Action Chunking (REMAC).** ICLR 2026. [https://proceedings.iclr.cc/paper_files/paper/2026/hash/c35834443b7881e782e52b3519fe27c7-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c35834443b7881e782e52b3519fe27c7-Abstract-Conference.html)

### Verified resource pointers

- **[R1] AnyGoal code and public CLI:** [https://github.com/galilai-group/AnyGoal/tree/f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e](https://github.com/galilai-group/AnyGoal/tree/f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e)
- **[R2] Model/data release listings:** [https://huggingface.co/HazelNam/anygoal](https://huggingface.co/HazelNam/anygoal) ; [https://huggingface.co/datasets/HazelNam/anygoal](https://huggingface.co/datasets/HazelNam/anygoal)
- **[R3] State-interface finding:** [https://github.com/galilai-group/AnyGoal/blob/f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e/anygoal/datasets/insert_peg/true_state.py](https://github.com/galilai-group/AnyGoal/blob/f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e/anygoal/datasets/insert_peg/true_state.py) and [https://github.com/galilai-group/AnyGoal/blob/f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e/anygoal/resources/eval/training/toy/adaln/insertpeg.yaml](https://github.com/galilai-group/AnyGoal/blob/f0cf8c28acfe0ae35b5946dbb637cd6e391d8e0e/anygoal/resources/eval/training/toy/adaln/insertpeg.yaml) . The complete simulator observation and restoration are defined in anygoal/envs/insert_peg/env.py. Keep the reduced-state model's feature selection distinct from that environment observation.
- **[R4] SMWM planning settings:** anygoal/resources/plan/sensorimotor/insert_peg.yaml at R1.
- **[R5] SMWM training/interface settings:** anygoal/resources/train/sensorimotor/insert_peg.yaml and anygoal/runners/plan/planning_sensorimotor.py at R1.
- **[R6] RoboQuest code:** [https://github.com/declare-lab/RoboQuest](https://github.com/declare-lab/RoboQuest)
- **[R7] RoboQuest demonstrations:** [https://huggingface.co/datasets/declare-lab/RoboQuest](https://huggingface.co/datasets/declare-lab/RoboQuest)
- **[R8] RoboDyna resources:** [https://huggingface.co/RoboDyna](https://huggingface.co/RoboDyna) ; [https://robodyna.github.io/robodyna/](https://robodyna.github.io/robodyna/)
- **[R9] RoboDyna weights:** [https://huggingface.co/RoboDyna/pi05-baseline-bs256](https://huggingface.co/RoboDyna/pi05-baseline-bs256) at a0d436c8ac9a775783014107390236cf50e43fb9 ; [https://huggingface.co/RoboDyna/X-VLA-RoboDyna](https://huggingface.co/RoboDyna/X-VLA-RoboDyna) at c628bbfd70dc1f945773d098ea7ab89356f1f20e
- **[R10] ReflexBench/data and model release status:** [https://reflexvla.github.io/](https://reflexvla.github.io/)

**Access limitations:** Some arXiv and Hugging Face pages failed in the search browser, but the relevant PDFs or API manifests were retrieved directly. This is not an inaccessible-paper claim. PIEGraph implementation/checkpoint availability was not established. Closed-model RoboQuest reproducibility and unreleased ReflexVLA/RoboTwin-Phys artifacts remain limitations. No local GPU allocation, checkpoint execution, or robot availability was verified.

Prepared November 7, 2026, America/New_York.
