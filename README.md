# motor_information_connectivity_trees

Information-theory (Chow-Liu tree) vs functional-connectivity analysis of the motor
system across three precision-imaging datasets — **Midnight Scan Club (MSC)**,
**cast-induced plasticity (cast1/2/3)**, and **perinatal stroke (PS1 = `sub-CIMT001`)**.

Scientific plan: [`plans_and_reports/motor_system_plan.md`](plans_and_reports/motor_system_plan.md).
Engineering plan: [`plans_and_reports/repo_organization_plan.md`](plans_and_reports/repo_organization_plan.md).

## Layout

```
src/mict/        installable package (the engine) — `pip install -e .`
pipelines/       thin CLIs: define_rois → extract_timeseries → compute_connectivity
analyses/        one folder per scientific question (Q1–Q5) + confounds
config/          datasets.yaml · atlases.yaml · roi_scheme.yaml
tests/           pytest suite
data/            git-ignored symlinks to datasets (see below)
results/         git-ignored outputs (per dataset / space)
atlases/         small atlas refs committed; large ones git-ignored
```

## Setup

```bash
conda env create -f environment.yml
conda activate dynamric
pip install -e .
pytest                 # core engine tests (numpy MI + CL tree); no GPU/data needed
```

## Image space

Cortex is analyzed on the **fs_LR_32k surface** (the literature space; MSC + cast are
already WU/Gordon-processed on disk); cerebellum and thalamus in the CIFTI subcortical
volume. An optional volumetric (MNI) cross-check is deferred. See the engineering plan §9.

## Data

`data/` holds symlinks only (nothing committed):

| link | target |
|---|---|
| `msc_surface_pipeline` | `cerebellum_reliability/ds000224/derivatives/surface_pipeline` |
| `cast_induced_plasticity` | `fmri_connectivity_trees/datasets/cast_induced_plasticity` |
| `perinatal_stroke` | `fmri_connectivity_trees/datasets/perinatal_stroke` (raw — needs preprocessing) |
| `msc_fmriprep_volume`, `msc_firstLevel_glm` | MSC volumetric derivatives (optional cross-check) |

Note: `git`, `wb_command`, and FreeSurfer are not on the default login PATH — `module load`
or container them.
