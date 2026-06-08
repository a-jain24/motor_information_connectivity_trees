# Repository Organization Plan — `motor_information_connectivity_trees`

**Date:** 2026-06-04
**Purpose:** Define a clean, motor-focused repository that implements the scientific plan in
`plans_and_reports/motor_system_plan.md`, reusing the mature code from `fmri_connectivity_trees`
while leaving its clutter behind.

This document is the *engineering* plan (how the repo is laid out and what is ported).
`motor_system_plan.md` is the *scientific* plan (what we measure and why) and is authoritative
wherever the two disagree.

> **Settled decisions (2026-06-04):**
> **D1 — Surface/CIFTI (fs_LR_32k grayordinates) is the PRIMARY space.** It is the literature space
> (Gordon 2017, Newbold 2020, inter-effector study) *and* the most efficient cross-dataset path,
> because it is the one common space all three datasets reach cheaply: **MSC and cast are both
> already fully WU/Gordon-processed in fs_LR_32k on `/mfs`** (rest dtseries, surface motor contrasts,
> individual parcellations, surfaces, precomputed geodesics — verified §5), and only PS1 is processed
> fresh. Surface-first also avoids building the ROI front end twice. **A volumetric (MNI) cross-check
> is optional/deferred** —
> the space-agnostic core (§2.4) lets us add it later for a robustness panel if we want it.
> **D2 — Simulations omitted** for now.
> **D3 — ROI definition leads with the literature method per region:** task-peak + geodesic
> 400-vertex / 40-voxel expansion (Newbold) for M1 effectors and cerebellum; Gordon gradient-
> watershed + Infomap network parcels for SMA/PMd/PMv; FC-seeded thresholding for motor thalamus.

---

## 1. What this repo is (and is not)

**Is:** a focused pipeline to build and compare **Chow-Liu (CL) trees** vs **functional
connectivity (FC)** on a fixed motor-ROI scheme across three precision-imaging datasets —
**Midnight Scan Club (MSC)**, **cast-induced plasticity (cast1/2/3)**, and **perinatal stroke
(PS1 = `sub-CIMT001`)** — to answer Questions 1–5 of `motor_system_plan.md`. Analysis runs on
**fs_LR_32k grayordinates** (cortex on the fs_LR_32k surface; cerebellum and thalamus in the CIFTI
subcortical volume) — the HCP space the source literature uses and the common space all three
datasets reach most cheaply.

**Is not:** the general CL-vs-FC research platform. The old repo also carried ABIDE (autism
classification), LISTEN (narrative listening), a BrainNetCNN model zoo, and a TVB/neural-mass
simulation framework. **None of that is in scope here**, and per D2 the simulation track is omitted.

### Source repo at a glance (what we are migrating from)

| Track | Location in old repo | Disposition |
|---|---|---|
| Motor cortex localizer + **GCSS** (mature, motor, volumetric) | `code/.../midnight_scan_club/canonical_circuits/motor_cortex/` | **Port — core (convert front end to surface)** |
| Core CL / MI / clustering / paths | `code/.../midnight_scan_club/msc_*.py` | **Port — core (space-agnostic)** |
| ABIDE empirical + autism CNN | `code/functional_connectivity/abide/`, `code/classification/`, `classification/` | **Drop** |
| LISTEN | `code/functional_connectivity/listen/`, `code/jupyter_notebooks/listen/` | **Drop** |
| Exploratory notebooks (~20) | `code/jupyter_notebooks/`, root `*.ipynb` | **Drop** (logic already in scripts) |
| TVB / neural-mass simulation | `code/simulations/` | **Drop (D2)** |

---

## 2. Guiding principles for the clean repo

Lessons from the clutter and bugs in `fmri_connectivity_trees`:

1. **Code is an installable package; data and outputs live outside git.** The old repo mixed
   ~20 GB of ROI timeseries, NIfTI atlases, and figure PDFs into the same tree as code, behind a
   minimal `.gitignore`. Here, code is importable and `data/`, `results/`, `atlases/` (large files)
   are git-ignored and referenced by config.

2. **No deep nesting, no `sys.path`/`os.path.dirname` chains.** The old motor code lived 6
   directories deep and relied on hand-rolled `os.path.dirname` chains to import siblings — which
   silently broke on every move (`analysis_plan.md`, "Path setup"). An installed package
   (`pip install -e .`) makes `from mict.chow_liu import ...` work from anywhere.

3. **One implementation of each algorithm.** The old repo had MI in ≥4 places with subtle clamp
   differences. Here there is exactly one `mict.mutual_info` and one `mict.chow_liu`.

4. **The pipeline boundary is the ROI-timeseries matrix `(T × n_ROI)`.** Everything *upstream*
   (ROI definition + extraction) is **space-specific**; everything *downstream* (MI / FC / CL / all
   Q1–Q5 analyses) is **space-agnostic** and shared. We build the **surface** front end now and the
   shared core once; an optional volumetric front end can be slotted behind the same boundary later
   without touching the analyses. This is also what makes the eventual surface-vs-volume robustness
   check cheap.

5. **Dataset-agnostic core; dataset-specific orchestration.** The three datasets differ in scanner,
   TR (2.2 s vs 1.1 s), session count, subject count (10 vs 3 vs 1), and native processing state
   (§5). Core functions take parameters; per-dataset CLIs supply specifics from config.

6. **Carry the bug fixes forward as defaults, not patches.** The April-2026 preprocessing audit
   (odd-run z-maps, full cosine set, frame-censoring regressors, motion derivatives, TR = 2.2) is
   the *default* confound model in `mict.preprocessing`. The broken
   `msc_extract_timeseries_optimized.py` is not ported.

---

## 3. Proposed directory structure

```
motor_information_connectivity_trees/
├── README.md
├── CLAUDE.md                      # guidance for Claude Code (port + update from old repo)
├── pyproject.toml                 # installable package `mict`; pinned deps
├── environment.yml                # cross-platform conda env (Linux cluster — see §6)
├── .gitignore                     # Python/neuroimaging (replace the VisualStudio default!)
│
├── config/
│   ├── datasets.yaml              # per-dataset roots, TR, sessions, task labels, surface paths
│   ├── atlases.yaml               # fsLR-32k Glasser dlabel + surfaces; SUIT, Morel (subcort volume)
│   └── roi_scheme.yaml            # the 18-ROI motor scheme + expected MNI/grayordinate anchors
│
├── src/mict/                      # THE package — flat, importable, grayordinate-aware
│   ├── paths.py                   # ← msc_paths.py, generalized to 3 datasets + config-driven
│   ├── cifti.py                   # CIFTI/GIFTI I/O (nibabel.cifti2) + wb_command wrappers — CORE
│   ├── surface.py                 # fs_LR_32k geodesic distance / contiguous-vertex neighborhoods — CORE
│   ├── io.py                      # loaders/savers for dtseries / mi / fc / cl, keyed by {space}
│   ├── preprocessing.py           # confound model (audited default); cleans CIFTI dtseries
│   ├── mutual_info.py             # ← msc_mutual_info.py (the single MI implementation)
│   ├── connectivity.py            # FC (←msc_covariance) + MI + CL adjacency — SPACE-AGNOSTIC
│   ├── chow_liu.py                # ← msc_chow_liu.py tree-building — SPACE-AGNOSTIC
│   ├── clustering.py              # ← msc_clustering.py (dendrogram/tanglegram/ARI)
│   ├── rois.py                    # ← canonical_utils.py: 18-ROI scheme over grayordinates
│   ├── localizer.py               # task-peak + geodesic 400-vertex (cortex) / 40-voxel (cereb) expansion
│   ├── parcellation.py            # ← gcss.py: Gordon surface gradient-watershed + Infomap network parcels
│   ├── thalamus.py                # FC-seeded motor thalamus (Morel VLa/VLpd/VLpv, subcort volume) — new
│   ├── cerebellum.py              # SUIT task/FC-based cerebellar ROIs (subcort volume) — new
│   ├── lesion.py                  # lesion masking + absent/displaced-ROI handling — for PS1
│   ├── stats.py                   # jaccard, consensus vote, conservation tests, bootstrap
│   └── viz/
│       ├── trees.py               # ← chow_liu draw_hierarchical_tree, consensus
│       ├── matrices.py            # FC/MI/CL heatmaps with ROI-group dividers
│       └── brain.py               # fs_LR surface plots (plot_surf_*) + subcortical-volume overlays
│
├── pipelines/                     # thin CLIs that wire `mict` functions per dataset/step
│   ├── make_grayordinates.py      # PS1 only — produce fs_LR_32k CIFTI (MSC + cast already native)
│   ├── define_rois.py             # step 1: localizer / watershed / thalamus / cerebellum → ROIs
│   ├── extract_timeseries.py      # step 2: ROIs → cleaned parcellated dtseries timeseries
│   ├── compute_connectivity.py    # step 3: timeseries → FC, MI, CL adjacency  (space-agnostic)
│   └── run_dataset.sh             # orchestrate steps for {cast, msc, perinatal_stroke}
│
├── analyses/                      # one folder per scientific question (motor_system_plan §Q1-Q5)
│   ├── q1_healthy_conservation/   # CL vs FC Jaccard across 10 MSC subjects
│   ├── q2_cast_disuse/            # pre/cast/post topology change
│   ├── q3_stroke_reorganization/  # PS1 vs MSC consensus; displaced-node edges
│   ├── q4_longitudinal/           # PS1 per-session CL stability over ~500 days
│   ├── q5_inter_effector/         # inter-effector hub recovery
│   ├── confounds/                 # roi_size, conditional/premotor-only CL, test-retest
│   └── space_comparison/          # [optional] surface-vs-volume CL/FC agreement (deferred cross-check)
│
├── jobs/                          # SLURM scripts (only PS1 preprocessing is compute-heavy — §10)
├── tests/                         # pytest suite (← test_suite_plan.md)
│
├── atlases/                       # small atlas files committed; large ones git-ignored + fetched
├── data/                          # git-ignored: symlinks to datalad datasets + generated grayordinates
├── results/                       # git-ignored: ROI masks, connectivity matrices, figures (per dataset/space)
│
└── plans_and_reports/
    ├── motor_system_plan.md       # authoritative science plan (already present)
    ├── repo_organization_plan.md  # this file
    └── methods/                   # ported sub-plans (§8)
```

**Why a flat `src/mict/`**: the old `code/functional_connectivity/<dataset>/` layout forced one copy
of each script per dataset and buried shared logic. A flat package with dataset-specific *config*
(not code) removes the duplication and ends the path-resolution fragility. Outputs are written under
`results/<dataset>/surface/...` (and `…/volume/...` if the optional cross-check is built).

---

## 4. File-by-file migration map

Paths are relative to `fmri_connectivity_trees/`. The ROI front end is **converted to the surface**
(the old code was volumetric); the space-agnostic core ports with light edits.

### Core engine — port largely as-is (space-agnostic)

| Old file | → New location | Action |
|---|---|---|
| `code/.../msc_chow_liu.py` (482) | `src/mict/chow_liu.py` + `viz/trees.py` | Refactor: split tree-building from plotting; add `color_fn=` param (`circuit_plan.md` §Key Design Decision 6) |
| `code/.../msc_mutual_info.py` (290) | `src/mict/mutual_info.py` | Port; the *only* MI function (delete the `analyze_sim`/TVB copies) |
| `code/.../msc_covariance.py` (302) | fold into `src/mict/connectivity.py` | Refactor: FC + MI + CL in one module |
| `code/.../msc_clustering.py` (623) | `src/mict/clustering.py` | Port (used by confound analyses) |
| `code/.../msc_paths.py` (147) | `src/mict/paths.py` | Refactor: read roots from `config/datasets.yaml`; keep device/host probe; add cast + perinatal_stroke roots; key outputs by `{space}` |
| `canonical_circuits/canonical_utils.py` (282) | `src/mict/rois.py` | Refactor: keep atlas index constants; redefine the ROI scheme over **grayordinates** (cortex = fs_LR_32k vertex sets by Glasser dlabel; subcortex = CIFTI volume voxels). Extend the settled **11-ROI** scheme to the **18-ROI** target (add bilateral Thal, Cereb hand/foot/face; reconcile `tongue`↔`M1_face`) |

### ROI definition + extraction — port + convert to surface (the biggest refactor)

| Old file | → New location | Action |
|---|---|---|
| `motor_cortex/msc_localizer_roi_timeseries.py` (512) | `src/mict/localizer.py` + `src/mict/preprocessing.py` | Refactor: promote the audited confound model to `preprocessing.py`; replace volumetric top-k voxel selection with **surface task-peak + geodesic 400-vertex expansion** (Newbold); keep odd-run z-map averaging. Needs surface task z-maps (§9.3) |
| `motor_cortex/gcss/gcss_timeseries.py` (840) | `src/mict/parcellation.py` + `pipelines/define_rois.py` | Refactor: the GCSS group-prob→watershed→intersection becomes the **surface** Gordon gradient-watershed for SMA/PMd/PMv (§C) — where the method natively lives. **Reuse, don't recompute, the existing individual parcellations**: both MSC and cast ship `surface_parcellation/*.dtseries.nii` + Infomap `*.dscalar.nii` on disk; only PS1 is computed de novo. Add lesion masking (`perinatal_stroke_plan.md`) |
| `motor_cortex/gcss/gcss_connectivity.py` (448) | `pipelines/compute_connectivity.py` | Refactor: dataset-agnostic; writes `results/<dataset>/surface/{fc,mi,cl}` |
| `motor_cortex/mc_localizer_fc_cl.py` (407) | merge into `pipelines/compute_connectivity.py` + `viz/matrices.py` | Port figures |

### Visualization — port + convert to surface

| Old file | → New location | Action |
|---|---|---|
| `motor_cortex/gcss/gcss_group_figures.py` (1380) | `src/mict/viz/trees.py` + `analyses/*/figures.py` | Refactor: publication-figure monolith; split generic plotters from question-specific assembly |
| `motor_cortex/mc_localizer_brain_viz.py` (574) | `src/mict/viz/brain.py` | Refactor: its fsaverage surface-projection path (Viz 4) is the seed; generalize to fs_LR_32k `plot_surf_*` |
| `motor_cortex/gcss_viz_subcortical.py` (827) | `src/mict/viz/brain.py` | Port: subcortical/cerebellar **volume** overlays (Thal/Cereb live in the CIFTI subcortical volume) |
| `motor_cortex/gcss_viz_group_maps.py` (867) | `src/mict/viz/brain.py` | Refactor group maps to fs_LR surface |

### New modules — required by `motor_system_plan.md` + the surface decision

| New file | Implements | Reuses |
|---|---|---|
| `src/mict/cifti.py` | `.dtseries.nii`/`.dlabel.nii` I/O (`nibabel.cifti2`) + `wb_command` wrappers (`-cifti-parcellate`, `-cifti-smoothing`, `-metric-*`) | nibabel; Connectome Workbench (§6) |
| `src/mict/surface.py` | fs_LR_32k geodesic distance + contiguous-vertex neighborhoods (400-vertex expansion) | `gdist` or `wb_command -surface-geodesic-distance` |
| `src/mict/thalamus.py` | §D motor thalamus: seed FC from cortical ROIs → per-subject 95th-pct threshold → Morel VLa/VLpd/VLpv labeling (subcortical volume) | Morel atlas; `connectivity.py` |
| `src/mict/cerebellum.py` | §E cerebellar ROIs: task peak + 40-voxel expansion in SUIT lobules V/VI (subcortical volume), FC fallback | SUIT atlas (§5); `localizer.py` |
| `src/mict/lesion.py` | PS1 absent/**displaced** ROI handling: NaN in connectivity, CL on remaining nodes, displacement-vector reporting | `perinatal_stroke_plan.md` masking + edge-disruption |
| `pipelines/make_grayordinates.py` | fs_LR_32k CIFTI for **PS1 only** (MSC + cast already native): WU pipeline, else fMRIPrep `--cifti-output` | ciftify / fMRIPrep |

### Drop entirely

`code/functional_connectivity/abide/*`, `…/listen/*`, `code/classification/*`, `classification/*`,
all `code/jupyter_notebooks/*`, root `*.ipynb`, `msc_extract_timeseries_optimized.py` (broken),
and all of `code/simulations/` (D2).

---

## 5. Data management

The three datasets are in **three different native processing states** (verified 2026-06-04). The
key efficiency point: **the surface is the common space each one reaches most cheaply.**

| Dataset | On disk | Surface readiness (the path we use) |
|---|---|---|
| **MSC** (10 subj) | `…/cerebellum_reliability/ds000224/derivatives/surface_pipeline/sub-MSC01..10/` (+ a second copy under `…/archived_projects/midnight_scan_club/derivatives/surface_pipeline/`) | **READY — full WU/Gordon fs_LR_32k grayordinate processing on disk for all 10 subjects.** `processed_restingstate_timecourses/ses-func0N/cifti/` (rest dtseries → connectivity), `task_timecourses/` + **`task_contrasts_cifti/motor/*-motor_contrasts_32k_fsLR.dscalar.nii`** (surface motor effector z-maps → the localizer, **already computed**), `surface_parcellation/*_parcels.dtseries.nii` + `*_parcel_networks.dscalar.nii` (gradient-watershed + Infomap), `fs_LR_Talairach/fsaverage_LR32k/` surfaces, `cifti_distances/*.mat` (precomputed geodesic + euclidean distance matrix). No preprocessing, no GLM, no projection — wire straight in |
| **cast** (cast1/2/3) | `…/datasets/cast_induced_plasticity/derivatives/{surface,volume}/` | **READY — same WU/Newbold fs_LR_32k layout** (same lab/pipeline as MSC): `processed_restingstate_timecourses` + `task_timecourses`, `surface_parcellation/*.dtseries.nii` + Infomap `*.dscalar.nii`, `fs_LR_Talairach` surfaces. No preprocessing; wire straight in |
| **PS1** (sub-CIMT001) | `…/datasets/perinatal_stroke/sub-CIMT001/` (raw BIDS, ds004498) **+** `…/datasets/perinatal_stroke-mriqc/` (**MRIQC v24 QC derivatives** — IQMs only, verified 2026-06-04) | **STILL NEEDS PREPROCESSING.** The new `-mriqc` dir is **quality control, not fMRIPrep**: per-run `*_bold.json` IQMs + `group_bold.tsv` (FD, tSNR, etc.), no preprocessed BOLD/CIFTI. Use it to **QC-select runs** (exclude high-motion runs from the 56 motor + 64 rest) feeding preprocessing/analysis. Connectivity still requires fMRIPrep → fsLR CIFTI: run the **WU/Gordon surface pipeline** (identical layout to MSC/cast) or fMRIPrep `--output-spaces fsLR --cifti-output 91k`. 60 fieldmaps, T1w+T2w, **DWI** (Q3 corticospinal cross-check) present. A TACC/OpenNeuro processing setup exists under the mriqc dataset's `code/tacc-openneuro/`. |
| MSC volumetric ROI timeseries (≈20 GB) | `…/datasets/midnight_scan_club/` (`connectivity/`, `roi_time_series/`) | Superseded by the surface `processed_restingstate_timecourses` above — ignore; keep as read-only reference |

MSC + cast are **both already fs_LR_32k (WU/Gordon, 711-2B Talairach) with identical derivative
layouts**, so one loader + config handles both. All datasets are symlinked into `data/` and
referenced from `config/datasets.yaml`; nothing is added to the new git history.

**Correction to `motor_system_plan.md`:** it expects the MSC surface parcellations at
`datasets/midnight_scan_club/derivatives/surface_pipeline/…`. They exist, but at
`cerebellum_reliability/ds000224/derivatives/surface_pipeline/` (all 10 subjects, plus an archived
copy) — point `config/datasets.yaml` there. (cast1=MSC02 / cast2=MSC06 also have their own copies
under the cast derivatives.)

**Grayordinate generation (`pipelines/make_grayordinates.py`) — PS1 only.** MSC and cast are already
fs_LR_32k on disk (no recon-all, no ciftify, no fetch, no geodesic computation — distances are
precomputed). Only PS1 is processed: preferably through the lab's WU/Gordon surface pipeline for an
identical layout/space to MSC+cast, else fMRIPrep `--output-spaces fsLR --cifti-output 91k`
(recon-all included) with subcortex resampled to match.

**Atlases.**
- *Surface (cortex):* Glasser/HCP-MMP1 **fs_LR_32k `.dlabel.nii`** + fs_LR_32k midthickness/pial/
  inflated `.surf.gii` + medial-wall ROI (BALSA / HCP S1200 / templateflow / neuromaps) — **not on
  disk; fetch.** Only the *volumetric* HCP-MMP1 is present (`atlases/glasser/…nii.gz`).
- *Subcortex (volume):* SUIT — **not in old `atlases/`**, at `…/egcerebellum/.../atl-Anatom_space-
  MNI…res-2.nii` (`cerebellar_subcortical_gcss_plan.md`); Morel — `atlases/MorelAtlasMNI152` (65 M,
  git-ignore). Carried as the CIFTI subcortical-volume structures.

**`.gitignore`** — replace the VisualStudio/.NET default with a Python/neuroimaging ignore:
`__pycache__/`, `*.py[cod]`, `nilearn_cache/`, `.ipynb_checkpoints/`, `data/`, `results/`, `*.nii`,
`*.nii.gz`, `*.dtseries.nii`, `*.dlabel.nii`, `*.gii`, `*.npy` (except small committed refs),
figure `*.pdf`, conda env dir.

---

## 6. Environment

The old `environments/dynamric1.yml` is an **exact-build osx-arm64 (Apple Silicon) export** — it
will not solve on this Linux cluster. The surface decision makes the CIFTI tooling core deps.

- `environment.yml`: name `dynamric`, Python 3.11, channel `conda-forge`; deps without build strings
  — `nilearn`, `nibabel` (CIFTI-2), `numpy`, `scipy`, `scikit-learn`, `scikit-image` (watershed),
  `networkx`, `pandas`, `matplotlib`, `pytorch` (CUDA build for the GPU MI), `pyyaml`, `pytest`, plus
  surface tooling: **`connectome-workbench` (`wb_command`)** for reading/parcellating/smoothing the
  existing dtseries, and **`nibabel`** CIFTI-2. `gdist` is **not needed** (MSC/cast ship precomputed
  geodesic distance matrices). `ciftify` + fMRIPrep + FreeSurfer are needed **only for PS1**
  (module/container deps, noted in README), since MSC + cast surface data already exist.
- `pyproject.toml`: declare `mict` editable so `pip install -e .` provides imports — removes all the
  old `sys.path.insert` fragility.

(`git`/`wb_command` are not on this login shell's PATH; `module load`/container them, or commit from
a host that has them. Not a blocker for writing code.)

---

## 7. Simulations — omitted (D2)

No `simulations/` module is migrated. The project's ground-truth-recovery validation (old
`6_2_2026_report.md` §3) can be cited by reference to `fmri_connectivity_trees`. The trimmed module
can be added later if a reviewer asks; it has no dependency on the rest of this repo.

---

## 8. Plans & reports to carry over

Port into `plans_and_reports/methods/`, with noted reconciliations:

| Old plan (`6_3_2026/plans/`) | Status in new repo |
|---|---|
| `cl_vs_fc_comparison_plan.md` | **Port** → `analyses/q1_healthy_conservation/` (primary hypothesis test) |
| `roi_size_confound_plan.md` | **Port** → `analyses/confounds/` (referenced by `motor_system_plan.md` §C) |
| `conditional_MI_CL_plan.md` | **Port** → `analyses/confounds/` (premotor-only / residual CL) |
| `test_retest_reliability_plan.md` | **Port** → supports Q4 + fROI reliability (odd/even split) |
| `cerebellar_subcortical_gcss_plan.md` | **Port** → spec for `cerebellum.py` + `thalamus.py` (§D/§E) |
| `test_suite_plan.md` | **Port** → `tests/` (guards the audited confound model) |
| `perinatal_stroke_plan.md` | **Port with correction banner.** Lesion-masking + edge-disruption *code* is reusable, but its premise (unilateral MCA; cohort N≈10–20 to acquire) is **superseded** by `motor_system_plan.md`: PS1 is a **single** subject, **bilateral** cystic lesions, **task data on disk**, **displaced** (not merely absent) representations. `motor_system_plan.md` Q3/Q4 is authoritative. |
| `bug_fixes_plan.md` | **Mostly N/A** — broken script not ported; `TR=2.2`/confound fixes become defaults in `mict.preprocessing`. Keep a one-paragraph "audited confound model" note. |
| `circuit_plan.md`, `motor_cortex/analysis_plan.md`, `gcss/gcss_plan.md` | **Port as design reference** (ROI scheme, 11→18 rationale, preprocessing audit). The `corticocerebellar_motor/` module they describe was **never implemented**; build it fresh as `cerebellum.py`. |

---

## 9. Methodological reconciliations to resolve during the port

1. **Image space — RESOLVED: surface/CIFTI primary (D1).** Cortex on fs_LR_32k (Glasser dlabel,
   Gordon surface watershed, geodesic 400-vertex expansion); cerebellum + thalamus in the CIFTI
   subcortical volume (SUIT/Morel; 40-voxel expansion). CIFTI smoothing per `motor_system_plan.md`
   §pipeline steps 7–8 (geodesic σ ≈ 2.55 mm surface; 3D σ ≈ 2.55 mm subcortex). Volume is an
   optional deferred cross-check (`analyses/space_comparison/`).

2. **ROI count: 11 → 18.** The settled GCSS scheme is 11 cortical ROIs. `motor_system_plan.md`
   targets ~18 (adds bilateral motor thalamus + bilateral cerebellar hand/foot/face; bilateral
   PMd/PMv). `rois.py` defines the 18-ROI target as grayordinate sets; `thalamus.py`/`cerebellum.py`
   supply the new subcortical nodes. Note the `tongue`↔`M1_face` naming reconciliation.

3. **ROI method — RESOLVED: literature method per region (D3).**
   - **M1 foot/hand/face + cerebellar effectors:** task-GLM peak + **geodesic 400-vertex (cortex) /
     40-voxel (cerebellum) expansion** (Newbold 2020) → `localizer.py`/`cerebellum.py`.
   - **SMA/PMd/PMv:** Gordon 2017 **surface gradient-watershed parcellation** + Infomap network
     assignment, refined by task → `parcellation.py`. **Reuse the precomputed individual
     parcellations** (both MSC and cast on disk); compute de novo only for PS1.
   - **Motor thalamus:** inter-effector **FC-seed + per-subject 95th-pct threshold** + Morel labeling
     → `thalamus.py`.
   - Keep ROI **labels identical across datasets** per `motor_system_plan.md` §Coherence.

4. **PS1 = displaced, not absent.** PS1's right-hand representation is displaced to the left
   postcentral gyrus. `localizer.py` must search whole surviving cortex for PS1 (no anatomical
   constraint) and `lesion.py` must report a *displacement vector* per effector, not just NaN.

> **9.3 — Surface task GLM z-maps — RESOLVED for MSC + cast.** MSC ships precomputed surface motor
> contrasts (`task_contrasts_cifti/motor/*-motor_contrasts_32k_fsLR.dscalar.nii`, raw + smoothed) and
> cast ships surface `task_timecourses` — the localizer reads these directly (no GLM re-fit, no
> volumetric→surface projection). Only PS1's GLM is run fresh on its surface dtseries. *(Confirm the
> MSC motor contrast effector labels — bilateral foot/hand/tongue vs lateralized — when wiring in;
> the odd-run averaging convention from the old audit still applies if z-maps are recomputed.)*

---

## 10. Phased implementation roadmap

Ordered by **data readiness**: build the surface front end once, then bring up datasets
cheapest-first. **Phases 0–2 are implemented and validated** (status below); Phases 3–6 remain.

**Phase 0 — Scaffolding. ✅ DONE.** `.gitignore`, `pyproject.toml`, portable `environment.yml`,
`config/*.yaml`, package skeleton, dataset symlinks.

**Phase 1 — Core engine. ✅ DONE.** Space-agnostic core (`chow_liu`, `mutual_info`, `connectivity`,
`stats`, `paths`, `rois`, `io`) + surface foundation (`cifti`, `surface`, `wu`, `preprocessing`).
**15/15 unit tests green.** Key discovery: the WU files are **CIFTI-1** — a raw read scrambles the
data (→ near-zero FC); `mict.cifti` converts CIFTI-1→CIFTI-2 via `wb_command` then reads via nibabel
(validated: temporal autocorr 0.82, bilateral M1_face FC 0.92). See [[mict-wu-cifti-data-access]].

**Phase 2 — MSC + cast end-to-end on the surface. ✅ DONE.**
- **MSC (Q1):** 10 subjects, 12 task-defined ROIs (6 M1 + 6 cerebellar effectors, Newbold peak +
  expansion); consensus tree; **bilateral M1 pairs 10/10**, L_M1_face↔R_M1_face MI=0.871 (matches the
  old volumetric report). CL-vs-FC conservation null for this 12-ROI subset (honest — needs the full
  ROI set).
- **cast (Q2):** cast1/cast2 (=MSC02/06) reuse the MSC masks (verified byte-identical grayordinates);
  fetched a balanced subset from OpenNeuro ds002766 S3 (`git-annex get`); split pre/cast/post via
  `sessions.tsv`. **Casting hypothesis confirmed:** the disused right-hand M1 edge
  (L_M1_hand↔R_M1_hand) dips during cast and recovers post in both subjects.

**Phase 3 — Remaining ROI methods → full 19-ROI scheme (next).** Add premotor SMA/PMd/PMv (§C, Gordon
surface gradient-watershed + Infomap — the parcellations are already on disk in `surface_parcellation`)
and motor thalamus (§D, FC-seed + Morel labeling, on the CIFTI subcortical voxels). Re-run MSC + cast
at 19 ROIs; re-test Q1 conservation with the richer premotor branch where CL's advantage may appear.

**Phase 4 — Figure suite (NEW objective B).** Build the participant-specific + consensus motor-map
figures (§12): the headline **subject CL-tree grid** + **consensus CL tree**, plus brain-space ROI/CL
maps. Wire into each dataset's analysis so MSC, cast (per condition), and PS1 each emit subject +
consensus figures.

**Phase 5 — PS1 (NEW recalibration, objective A). *The only dataset needing preprocessing.***
- **5a — Run QC:** read the new MRIQC `group_bold.tsv` IQMs → select runs (drop high-motion ones from
  the 56 motor + 64 rest). New helper `mict.qc` (read MRIQC IQMs).
- **5b — Preprocess:** WU/Gordon surface pipeline (preferred — identical layout to MSC/cast) or
  fMRIPrep `--output-spaces fsLR --cifti-output 91k` (SLURM). **MRIQC is QC only — it does not replace
  this step.**
- **5c — Analyze:** `lesion.py` (NaN + displacement vectors), PS1 unconstrained peak search; build
  PS1's CL trees + subject/consensus figures. Drives **Q3 (reorganization)** (vs the MSC consensus) and
  **Q4 (longitudinal)**; DWI supports the Q3 corticospinal cross-check.

**Phase 6 — Cross-cutting + optional volume cross-check.** `analyses/q5_inter_effector/` +
`analyses/confounds/` (ROI-size, premotor-only CL, test-retest). *Optional:* a volumetric front end +
`analyses/space_comparison/` robustness panel.

---

## 11. Status

**Implemented & validated (Phases 0–2):** MSC Q1 (consensus + conservation) and cast Q2 (disuse
recovery) run end-to-end on real surface data; 15/15 tests green. The CIFTI-1 read bug is fixed.

**Settled decisions:** D1 surface/CIFTI primary (volume = optional cross-check); D2 simulations
omitted; D3 literature ROI method per region.

**Data/surface availability (verified on disk 2026-06-04):**
- **MSC + cast:** full WU/Gordon fs_LR_32k processing on disk (MSC at
  `…/archived_projects/midnight_scan_club/derivatives/surface_pipeline/` — the **materialized** copy;
  the `ds000224` datalad copy is annex-pointers only). cast content fetched on demand from ds002766 S3.
- **PS1:** raw BIDS **+ new MRIQC QC derivatives** (`perinatal_stroke-mriqc`, ds004498). **MRIQC is
  quality control, not preprocessing** — PS1 connectivity still needs fMRIPrep/WU → fsLR CIFTI
  (Phase 5b). The MRIQC IQMs gate run selection (Phase 5a).

**Near-term priorities:** (Phase 3) finish the 19-ROI scheme; (Phase 4) the figure suite (§12); then
(Phase 5) PS1 once preprocessed.

---

## 12. Figure suite — participant-specific motor maps + consensus (objective B)

Goal: reproduce (and generalize to the 19-ROI scheme) the polished publication figures from the old
`gcss/analysis/pub_figures/`, especially the two the user singled out — **per-subject CL-tree grid**
(`fig6_subject_cl_trees`) and **consensus CL tree** (`fig5_consensus_cl_tree`) — plus brain-space
"maps of the motor system." All figures are **ROI-scheme-agnostic**: node colours come from
`config/roi_scheme.yaml` / `mict.rois.REGION_COLORS`, so they work for the current 12-ROI set and the
eventual 19-ROI set without change.

Source code to port: `gcss_group_figures.py` (CL-tree + matrix figures) and `gcss_viz_group_maps.py`
/ `mc_localizer_brain_viz.py` (brain maps) → `mict.viz`.

### Headline figures (the user's favourites)

**F1 — Subject CL-tree grid** (`mict.viz.trees.subject_cl_tree_grid`) — a `2×N` panel, one CL tree
per subject (or per cast condition), drawn with `draw_hierarchical_tree`. Styling, per the original:
- node colour = ROI region (`REGION_COLORS`); node outline (black) if **hub** (degree ≥ 3);
- edge style by **consensus frequency** across subjects: **canonical** (≥70%) solid bold,
  **intermediate** (40–69%) medium, **idiosyncratic** (<40%) thin dashed;
- per-panel title with the subject's **mean MI**; shared ROI-colour + edge-style legend.

**F2 — Consensus CL tree** (`mict.viz.trees.consensus_cl_tree_figure`) — single panel, the group (or
condition) consensus. Per the original: fixed **anatomical** node layout (foot top → tongue
bottom-left → SMA/PMd/PMv right); **edge width ∝ frequency**, opacity ∝ frequency, each edge labelled
`k/N`; node size ∝ mean degree; node colour = region. (The current `analyses/q1` already emits a basic
forest-aware consensus tree via `draw_hierarchical_tree`; F2 upgrades it to the frequency-weighted
anatomical-layout style.)

### Brain-space maps of the motor system (`mict.viz.brain`)

- **F3 — Subject fROI mosaic** (`subject_roi_mosaic`): subjects × ROIs grid of each participant's
  ROI on the fs_LR surface / MNI — shows individual variability (old fig3 / fig11).
- **F4 — Group ROI atlas** (`group_roi_atlas`): all ROIs on one brain, region-coloured (old fig2).
- **F5 — CL tree on brain** (`cl_tree_on_brain`): CL edges drawn between ROI centroids on the surface
  / glass brain, per-subject and consensus (old fig9). Edge width ∝ MI/frequency.
- **F6 — ROI flat-map** (`roi_flatmap`): ROIs on the fs_LR flat map (old fig12).

These are the "participant-specific maps of the motor system" + the "consensus map" the user asked
for: F1/F3/F5(subject) are participant-specific; F2/F4/F5(group) are the consensus maps.

### Supporting figures (port opportunistically)

Group FC/MI matrices (old fig4), FC-vs-MI scatter with CL edges highlighted (fig8), hub-degree bars
(fig7), aesthetic single/group tree (fig10). All driven by the same `results/<dataset>/<space>/…`
matrices already produced by `compute_connectivity`.

### Wiring

Each dataset/condition analysis (`analyses/q1…q5/`) calls the figure suite on its loaded matrices, so
every run emits the participant-specific grid **and** the consensus map for that group — MSC (N=10),
cast (per pre/cast/post), and PS1 (single subject vs the MSC consensus). A thin
`mict.viz.figures.motor_report(dataset, space, subjects)` convenience wraps F1+F2 (+ optional brain
maps) so a new dataset gets both headline figures in one call.

---

## 13. Apples-to-apples: CL vs FC-based sparse networks (objective C)

Full method spec + literature + results: **`plans_and_reports/methods/fc_sparse_networks.md`**.
Implemented `mict.fc_sparse` + `analyses/q1_healthy_conservation/cl_vs_fc_sparse.py`.

**Goal.** Compare the CL tree (MST of mutual information) against sparse graphs from *linear*
FC at **matched sparsity and identical graph machinery**, so any CL advantage is attributable,
not a sparsity artifact. Each method yields subject-specific **and** consensus networks rendered
with the same F1/F2 figures (§12) — so CL and FC are shown side-by-side in one style.

**Methods** (all sparse; the MST variants give exactly N−1 edges = CL's budget):
- **FC-MST** — MST of |Pearson r| (Tewarie 2015). The exact *linear twin* of CL: under
  Gaussianity `MI = −½log(1−r²)` is monotonic in |r|, so CL = FC-MST; **the gap isolates
  nonlinearity** (reported as the per-subject CL↔FC-MST edge agreement).
- **Partial-correlation MST** — MST of |partial r| from the (shrunk) precision matrix; the
  *direct-connectivity* linear analog (Smith 2011; removes indirect paths).
- **Graphical LASSO** — L1-sparse precision (Friedman 2008; Varoquaux & Craddock 2013), the
  established sparse-connectome reference (density data-driven, not fixed to N−1).
- **FC top-(N−1)** — density-matched thresholded |r| baseline.
- *(Effective connectivity — Granger/DCM — noted as a directed future extension; deferred.)*

**Early MSC result (12 ROIs, N=10):** CL recovers the bilateral hand-M1 edge **10/10** vs **8/10**
for every matched-sparsity FC method; CL↔FC-MST agreement **0.608** (⇒ ~40% of CL edges reflect
non-Gaussian structure); CL conservation trends above FC-MST/partial-MST (p≈0.08–0.13, n.s. at 12
ROIs — revisit at 19 ROIs where the premotor branch is less dominated by the universal bilateral
pairs). GLASSO stays near-dense at this scale, confirming the matched-sparsity MST comparison is
the fair control.

**Roadmap placement.** Folded into **Phase 4** (figure suite) and the Q1 analysis: every dataset
emits CL **and** FC-sparse subject/consensus maps + the `cl_vs_fc_sparse` comparison table.
