# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

Chow-Liu (CL) tree vs functional connectivity (FC) analysis of the motor system across
**MSC**, **cast-induced plasticity**, and **perinatal stroke (PS1 = `sub-CIMT001`)**.
Authoritative science plan: `plans_and_reports/motor_system_plan.md`. Engineering plan:
`plans_and_reports/repo_organization_plan.md`. This repo is the clean successor to
`../fmri_connectivity_trees` (which also carried ABIDE/LISTEN/CNN/TVB — all out of scope here).

## Architecture (read before editing)

- **`src/mict/` is an installable package** (`pip install -e .`). Import as
  `from mict.chow_liu import chow_liu_tree`. **Never** add `sys.path` / `os.path.dirname`
  hacks — that fragility is exactly what this repo was built to remove.
- **Pipeline boundary = the ROI-timeseries matrix `(T × n_ROI)`.** Upstream (ROI definition
  + extraction) is space-specific; downstream (`mutual_info`, `connectivity`, `chow_liu`,
  analyses) is space-agnostic and shared. One MI implementation (`mict.mutual_info`), one CL
  implementation (`mict.chow_liu`) — do not fork them.
- **Code vs data vs outputs:** code in `src/`; data via `data/` symlinks (git-ignored);
  outputs in `results/<dataset>/<space>/...` (git-ignored). Config drives paths
  (`config/*.yaml`), not hard-coded constants.

## Conventions

- ROI labels are identical across datasets and spaces (`config/roi_scheme.yaml`).
- Primary space is **surface (fs_LR_32k)**; cerebellum/thalamus live in the CIFTI subcortical
  volume. Volumetric MNI is a deferred optional cross-check.
- The **audited confound model** (full cosine set, frame-censoring + non-steady-state
  regressors, motion + derivatives, TR = 2.2, odd-run task z-maps) is the default in
  `mict.preprocessing` — carried forward from the April-2026 audit, not re-litigated.

## Environment

`conda activate dynamric` then `pip install -e .`. `git`, `wb_command`, FreeSurfer are not on
the default login PATH (`module load` / container them). The numpy MI path needs no GPU; torch
is optional acceleration.

## Tests

`pytest` runs the core-engine tests (MI estimator + CL tree) on synthetic data — no GPU, no
real fMRI. Add tests alongside new core functions.
