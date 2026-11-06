# SIMPACT research agent instructions

This repository contains SIMPACT plus experimental SIMPACT → PHINO synthetic-physics work. For PHINO tasks, first read `docs/phino_cursor_handoff.md` (the current project state) and `docs/phino_pretraining_integration_contract.md` (the implementation contract), then inspect the relevant code. On initial onboarding, also read `docs/phino_initial_handoff.md` once for the complete research history; on subsequent tasks consult it only when historical context is needed. If that file is absent, continue with the current handoff and contract rather than blocking work. The handoff summarizes context but **source code and actual manifests take precedence**. PHINO's training code lives in the separate repository `arajshah/phino`.

## Working rules

- Make one focused change per task. Inspect existing implementations before adding new ones. Prefer reusable adapters and tests over another simulation or model rewrite.
- Use descriptive commit messages such as `feat(phino): export validated SIMPACT fingerprints for training`; include a concise verification summary in the final response/PR.
- Do not overwrite any existing `results/`, grid manifests, checkpoints, or research outputs. Use new output directories and keep generated data, credentials, weights, and large artifacts out of Git.
- Respect manifest-declared action order, physics values, rejected statuses, and provenance. Never silently admit rejected action traces or fabricate missing data.
- Do not change validated action forces, cutoff defaults, mass/friction grids, validation rules, or evaluation splits without an explicit experiment specification.
- Keep dataset splits grouped by geometry; do not use test data to fit normalizers, calibrate hyperparameters, select features, or choose among methods.
- Keep static-image training claims separate from recoverability using privileged simulated trajectories. Identical images with different arbitrarily assigned mass/friction do **not** reveal those hidden properties.
- Code must fail clearly on invalid shapes, missing assets, inconsistent identifiers, non-finite values, duplicate entries, or wrong normalization/action counts.
- Add lightweight tests and self-tests for adapters. Run tests available in the current environment; accurately report what you could not execute. GPU/MuJoCo checks on Brown may require a human-run command.
- Do not launch long training or expensive simulation sweeps unless requested. The current next task is **dataset export and one dataloader smoke test**, not physics debugging or full PHINO fine-tuning.
- When a research design choice is ambiguous or could make results scientifically invalid, document the issue and request human review rather than hiding it behind code.

## Research records

- Keep `docs/phino_cursor_handoff.md` concise and current. Update it after major milestones with verified progress, blockers, commit references, and the next task; do not turn it into a transcript.
- Use `docs/phino_experiment_log.md` for reproducible experiments, including hypothesis, dataset and split, code revision, metrics, limitations, and conclusions. Create it when there is an actual experiment to record; do not fabricate results or force an empty log.
- Preserve `docs/phino_initial_handoff.md` as a historical onboarding snapshot. Do not overwrite or constantly append to it; prefer the current handoff for changing status.

## First code task

Create a small, reproducible adapter for the existing seven-geometry SIMPACT grid to PHINO's expected training dataset contract. See `docs/phino_pretraining_integration_contract.md` for inputs, target files, and caveats. Preserve the original grid and do not claim model training has occurred.