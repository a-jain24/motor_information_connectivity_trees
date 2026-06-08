# PS1 (perinatal stroke, sub-CIMT001 / ds004498) preprocessing recipe

**Date:** 2026-06-04
**Goal:** get PS1's resting-state + motor-task fMRI into **fs_LR_32k grayordinate CIFTI** so the
`mict` surface pipeline runs on it unchanged and it is comparable to MSC + cast. Run this async
while we keep building; outputs feed Phase 5 (Q3 reorganization, Q4 longitudinal).

> **The honest hard part:** PS1 has **bilateral cystic cortical lesions (~20% volume loss)**.
> FreeSurfer `recon-all` (which fMRIPrep runs to make surfaces) is the genuine risk — surfaces
> near the cysts will likely be wrong without a lesion mask and manual QC. Plan for a test run +
> surface QC before committing all ~18 sessions. A volumetric fallback is in §7.

On-disk facts (verified 2026-06-04):
- Raw BIDS: `…/fmri_connectivity_trees/datasets/perinatal_stroke/sub-CIMT001/` — **annex pointers,
  content NOT fetched** (fetch first, §1). ~18 sessions, 56 motor + 64 rest runs, fieldmaps, T1w/T2w, DWI.
- MRIQC (QC only): `…/datasets/perinatal_stroke-mriqc/` — use for run selection (§2).
- fMRIPrep container: `/mfs/io/groups/dmello/software/fmriprep/fmriprep-25.0.0.sif` (or `-23.1.4.sif`
  to match the MSC-era run). FS license: `/mfs/io/groups/dmello/software/fmriprep/license.txt`.
- Runtime: `singularity` (v4.2.2). `git`/`git-annex`/`datalad` in `~/miniconda3/bin`.

---

## 0. How the original authors process this — and what's available (web-checked 2026-06-04)

**They do NOT use fMRIPrep.** Laumann, Gordon, Dosenbach et al. process PS1 with the in-house
**Washington University precision functional mapping (PFM) pipeline** — the *same* pipeline that
produced the MSC and cast `surface_pipeline` derivatives we already use. Per Gordon et al. 2017
(*Neuron*) / `motor_system_plan.md`: 4dfp volumetric preprocessing → atlas registration to
**711-2B Talairach** (not MNI) → field-map distortion correction → FD>0.2 censoring →
whole-brain+WM+ventricle+Volterra nuisance → **FreeSurfer recon-all on native T1 → register the
native surface to fs_LR_32k** (Caret/Connectome Workbench) → ribbon-constrained sampling to
mid-thickness → geodesic smoothing → individual **gradient-watershed parcellation + Infomap
networks** → seed-based FC. The Lancet correspondence is brief; the method is in its appendix
(pp 3–12) and Gordon 2017. Contact: laumannt@wustl.edu.

**So the fMRIPrep recipe below is a runnable *substitute*, not a replication.** Differences vs the
MSC/cast processing: volumetric space (fMRIPrep **MNI** vs WU **Talairach**), surface-registration
method, and nuisance model. The cortical **fs_LR_32k mesh is identical**, so cortex is comparable
at the vertex level (where the displaced hand representation lives) — but PS1 processed differently
from MSC/cast introduces a *pipeline confound* into the Q3 PS1-vs-MSC comparison.

**Availability — checked, and the answer is no shortcut:**
- **ds004498 on OpenNeuro is raw BIDS only** — no derivatives, no surface data, no parcellation,
  **no lesion mask** (verified: OpenNeuro file tree + the Dosenbach Lab data page list only raw
  BIDS for "Perinatal Stroke"). Contrast cast/ds002766, whose surface derivatives *are* on
  OpenNeuro — PS1's were not released. So a lesion mask must be **created** (manual segmentation),
  and surfaces must be **generated**.

**Recommended path, best→most-accessible:**
1. **Email the authors (laumannt@wustl.edu / Dosenbach lab).** They *have* the processed PS1
   surfaces, individual parcellation, and almost certainly a lesion mask (they ran PFM on this
   brain). Getting those directly = perfect comparability with MSC/cast and skips all processing +
   the lesioned-brain recon-all problem. **Highest-value action — do this first.**
2. **DCAN `abcd-hcp-pipeline`** (containerized HCP/Gordon-style BIDS-App → fs_LR_32k CIFTI + Gordon
   parcellation; designed to be robust across age, i.e. developmental data). Closer to the
   WU/Gordon tradition than fMRIPrep, and runnable. Still MNI-volumetric. A good middle option.
   <https://github.com/DCAN-Labs/abcd-hcp-pipeline>
3. **fMRIPrep** (§4 below) — most accessible/supported, least matched to MSC/cast. Use if 1–2 fall
   through; flag the pipeline difference when reporting Q3.

---

## 1. Fetch the raw data (ds004498 is on OpenNeuro S3, like cast)

```bash
export PATH=~/miniconda3/bin:$PATH
cd /mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/perinatal_stroke
# fetch anatomy + fieldmaps + func (rest + motor). Whole subject is large; you can scope to
# anat+fmap+func and skip dwi if not needed yet:
git-annex get sub-CIMT001/ses-*/anat sub-CIMT001/ses-*/fmap sub-CIMT001/ses-*/func
# sanity: T1w content now present (not a dangling symlink)
find sub-CIMT001 -name '*T1w.nii.gz' | head -1 | xargs readlink -f | xargs ls -lh
```

## 2. Select runs with the MRIQC you already have (Phase 5a)

MRIQC gives per-run IQMs; drop high-motion runs before connectivity (you can also exclude them
from fMRIPrep with a `--bids-filter-file`, but simplest is to run all and filter at analysis time).

```bash
# fetch + read the group IQM table
cd /mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/perinatal_stroke-mriqc
~/miniconda3/bin/git-annex get group_bold.tsv
python - <<'PY'
import pandas as pd
df = pd.read_csv("group_bold.tsv", sep="\t")
# fd_mean = mean framewise displacement per run; flag > 0.2 mm (Gordon/MSC threshold)
keep = df[df["fd_mean"] <= 0.2]
print(f"{len(keep)}/{len(df)} runs pass fd_mean<=0.2")
print(df[["bids_name","fd_mean","tsnr"]].sort_values("fd_mean").to_string(index=False))
PY
```
Record the keep-list; it becomes the run filter for `mict` connectivity (drop the rest).

## 3. (Critical) lesion mask

**Anchor session: `ses-38659`** (chosen 2026-06-07). All 7 sessions have one T1w + one T2w; T1w
quality is near-flat across them (MRIQC CJV 1.06–1.21), but T2w varies a lot and the *cyst border is
clearest on the bright T2w*. ses-38659 pairs a near-best T1w (CJV 1.125, 3rd of 7) with the **best
T2w (CJV 1.502)** from the *same scan session* → trivial co-registration, head in one position.
Content fetched + verified (2026-06-07): both ~0.8 mm isotropic, int16 (T1w 224×256×212, T2w
256×256×211) at `…/perinatal_stroke/sub-CIMT001/ses-38659/anat/`.

Create a binary lesion mask on the **native T1w** (manual is most reliable for cystic lesions;
ITK-SNAP). Segment *on the T1w grid* — that's the space fMRIPrep/recon-all and the BIDS `roi`
convention consume. Use the T2w as a visual aid for the cyst boundary, but it's on a **different
grid** (256×256×211 ≠ T1w 224×256×212), so co-register it to the T1w first, then load it as an
overlay layer in ITK-SNAP:

```bash
PS=/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/perinatal_stroke
A=$PS/sub-CIMT001/ses-38659/anat
# rigid (6-dof) T2w -> T1w so it shares the T1w grid (FSL; or use ITK-SNAP Tools->Registration)
flirt -in $A/sub-CIMT001_ses-38659_run-01_T2w.nii.gz \
      -ref $A/sub-CIMT001_ses-38659_run-01_T1w.nii.gz \
      -dof 6 -cost normmi \
      -out $A/sub-CIMT001_ses-38659_run-01_desc-coreg_T2w.nii.gz
# then in ITK-SNAP: open the T1w as main image, "Add Another Image" -> the coreg T2w as overlay,
# paint label 1 = cystic cavity / necrotic tissue, save on the T1w grid as:
```
```
sub-CIMT001/ses-38659/anat/sub-CIMT001_ses-38659_label-lesion_roi.nii.gz   # 1 = lesion, 0 = else
```

Place it in BIDS so fMRIPrep uses it for **registration cost-function masking** (ANTs ignores
lesioned voxels when warping to MNI — essential at 20% volume loss). This markedly improves MNI
registration but does **not** by itself fix recon-all surfaces over the cysts — see §5.

## 4. fMRIPrep command (surfaces + CIFTI + MNI)

Unlike the MSC fMRIPrep run (`--fs-no-reconall`, MNI-only — its surfaces came from the WU pipeline),
PS1 needs surfaces **from** fMRIPrep, so we **keep recon-all** and request fsLR + CIFTI:

```bash
IMG=/mfs/io/groups/dmello/software/fmriprep/fmriprep-25.0.0.sif
FS_LICENSE=/mfs/io/groups/dmello/software/fmriprep/license.txt
BIDS=/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/perinatal_stroke
OUT=/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/perinatal_stroke-fmriprep
WORK=/mfs/io/groups/dmello/projects/dynamric/work/fmriprep_ps1   # on /mfs: dmello nodes are diskless (/tmp=RAM)
FILTER=/mfs/io/groups/dmello/projects/dynamric/motor_information_connectivity_trees/plans_and_reports/methods/ps1_bids_filter.json

mkdir -p "$OUT" "$WORK"
unset PYTHONPATH
singularity run --cleanenv -B /mfs "$IMG" \
    "$BIDS" "$OUT" participant \
    --participant-label sub-CIMT001 \
    --bids-filter-file "$FILTER" \
    --fs-license-file "$FS_LICENSE" \
    --output-spaces MNI152NLin2009cAsym fsLR fsaverage \
    --cifti-output 91k \
    --use-syn-sdc --force syn-sdc \
    --bold2anat-dof 6 \
    --skip-bids-validation --md-only-boilerplate \
    --mem-mb 460000 --nthreads 64 --omp-nthreads 8 \
    -w "$WORK"
```
Notes: `--cifti-output 91k` → `*_space-fsLR_den-91k_bold.dtseries.nii` (the file we need; same
grayordinate space the lab already produces, e.g. under `sad_music/ds003085-fmriprep`). Keep MNI
output too (for the CIFTI subcortical structures, lesion overlay, and the optional volume
cross-check). Use the phasediff fieldmaps that are present; `--use-syn-sdc` is a safety net.

**`--bids-filter-file ps1_bids_filter.json`** pins the anatomical reference to `ses-38659` only —
its T1w, its T2w, and the lesion `roi` — so the single anatomical reference matches the grid the
lesion mask was drawn on (without this, 7 T1ws make fMRIPrep build its own template and the mask's
space no longer matches). The filter restricts **only** the anat entities; `bold`/`fmap`/`sbref` are
left unspecified, so functional + fieldmap runs from **all** sessions are still processed. Contents:

```json
{
  "t1w": {"datatype": "anat", "session": "38659", "suffix": "T1w"},
  "t2w": {"datatype": "anat", "session": "38659", "suffix": "T2w"},
  "roi": {"datatype": "anat", "session": "38659", "suffix": "roi"}
}
```
(Session labels in pybids queries drop the `ses-` prefix.) This makes the single-reference path
deterministic; it is **incompatible with `--longitudinal`** — if you later want an unbiased
within-subject template across the ~500-day span, drop the `t1w`/`t2w` lines (keep `roi`) and add
`--longitudinal`, but then the lesion mask must be resampled into the within-subject template space.

## 4b. Run strategy: smoke-test first, then batch (and why a GPU doesn't help)

**A GPU does not accelerate fMRIPrep.** The long poles are `recon-all` (FreeSurfer, largely
*single-threaded*) and ANTs registration / BOLD resampling (CPU OpenMP). There is no GPU path in the
standard anatomical+BOLD flow, so on a GPU node the speedup comes from its **CPU cores + RAM**, not
the GPU — scale `--nthreads/--omp-nthreads/--mem-mb` to the node. Rough single-subject budget:
**recon-all + anatomical ≈ 10–16 h, paid ONCE**, then **≈ 20–40 min per BOLD run**. The cohort is
18 sessions / **120 bold runs** (56 motor + 64 rest, all TR = 2.2 s); "half" ≈ 9 sessions ≈ ~60 runs
≈ **~1.5–2 days** wall. The anatomical cost is fixed, so the *other* half adds only its BOLD time
(~1 day). Note only 7 sessions carry anat; the 11 func-only sessions rely entirely on the single
ses-38659 anatomical reference via the §4 pin — which is exactly why pinning is required.

### Smoke test (one session) — do this before committing the batch
Process only ses-38659 (its anat + 4 bold runs) and QC the surfaces over the cysts before launching
~60 runs. Use the smoke filter `plans_and_reports/methods/ps1_bids_filter_smoke.json`, which
restricts *everything* (incl. `bold`/`fmap`) to ses-38659:

```json
{
  "t1w": {"datatype": "anat", "session": "38659", "suffix": "T1w"},
  "t2w": {"datatype": "anat", "session": "38659", "suffix": "T2w"},
  "roi": {"datatype": "anat", "session": "38659", "suffix": "roi"},
  "bold": {"datatype": "func", "session": "38659", "suffix": "bold"},
  "fmap": {"datatype": "fmap", "session": "38659"}
}
```
The block below is the *conceptual* command; the **ready-to-submit** version (adds the TemplateFlow
bind, the system-singularity path, and SLURM headers) is `ps1_fmriprep_smoke.sbatch` — see §5 for the
full procedure.
```bash
# env vars as §4, plus a STABLE FreeSurfer dir we reuse across all later batches:
FSDIR=$OUT/sourcedata/freesurfer
SMOKE=/mfs/io/groups/dmello/projects/dynamric/motor_information_connectivity_trees/plans_and_reports/methods/ps1_bids_filter_smoke.json
mkdir -p "$OUT" "$WORK" "$FSDIR"
singularity run --cleanenv -B /mfs "$IMG" \
    "$BIDS" "$OUT" participant \
    --participant-label sub-CIMT001 \
    --bids-filter-file "$SMOKE" \
    --fs-subjects-dir "$FSDIR" \
    --fs-license-file "$FS_LICENSE" \
    --output-spaces MNI152NLin2009cAsym fsLR fsaverage \
    --cifti-output 91k --use-syn-sdc --force syn-sdc --bold2anat-dof 6 \
    --skip-bids-validation --md-only-boilerplate \
    --mem-mb 460000 --nthreads 64 --omp-nthreads 8 -w "$WORK"
```
**QC gate:** in freeview, load the T1w and overlay `$FSDIR/sub-CIMT001/surf/{lh,rh}.white` +
`{lh,rh}.pial`; confirm the surfaces neither bleed into nor skip over the cysts. If they fail → §7
(edit mask / enantiomorphic fill / volumetric fallback) **before** scaling up. This run is the
~10–16 h anatomical cost + 4 runs; everything after reuses it.

### Batch the rest, reusing recon-all (skip the long pole)
Because `--fs-subjects-dir "$FSDIR"` now holds a **completed** recon-all for sub-CIMT001, every later
run **detects it and skips recon-all** — batches 2+ are BOLD-only (~20–40 min/run, no ~10–16 h
anatomical). Select sessions per batch with a `bold`/`fmap` session **list** (anat stays pinned to
ses-38659). Save as `ps1_bids_filter_batch.json` after picking your half from the §2 low-motion
keep-list (example uses 9 of 18):

```json
{
  "t1w": {"datatype": "anat", "session": "38659", "suffix": "T1w"},
  "t2w": {"datatype": "anat", "session": "38659", "suffix": "T2w"},
  "roi": {"datatype": "anat", "session": "38659", "suffix": "roi"},
  "bold": {"datatype": "func", "suffix": "bold",
           "session": ["38659","38660","38719","38780","38781","38834","38835","00000","0001"]},
  "fmap": {"datatype": "fmap",
           "session": ["38659","38660","38719","38780","38781","38834","38835","00000","0001"]}
}
```
Run the §4 command but with `--bids-filter-file ps1_bids_filter_batch.json` **and the same**
`--fs-subjects-dir "$FSDIR"`. fMRIPrep is idempotent — already-finished outputs are skipped, so the
smoke session reappearing in the list (and any overlap between batches) is safe. For the *second*
half, swap in the other 9 sessions; recon-all is still skipped. (Optional extra speedup: add
`--derivatives "$OUT"` to also reuse the prior **anatomical normalization**, saving the ~30–60 min
ANTs step per batch — `--fs-subjects-dir` alone already removes the dominant cost.)

## 5. SLURM job — exact run procedure

Ready-to-submit scripts live next to this doc (verified 2026-06-08): `ps1_fmriprep_smoke.sbatch`,
`ps1_fmriprep_batch.sbatch`, and the three filter JSONs. They target the lab's **`dmello`** partition
(64-core / 512 GB nodes, 14-day limit, non-preemptible — the right home for a 10–16 h recon-all; the
`*-preempt` partitions risk mid-run preemption and a GPU buys nothing, §4b). TemplateFlow is
pre-staged at `/mfs/io/groups/dmello/projects/dynamric/templateflow` and bind-mounted to
`/templateflow` so the offline compute nodes never try to download templates.

Each job requests a **full exclusive node** (`--nodes=1 --ntasks=1 --exclusive --cpus-per-task=64
--mem=490G`, with fMRIPrep `--nthreads 64 --omp-nthreads 8 --mem-mb 460000`). The BOLD-heavy **batch**
saturates this (≈64/omp = 8 concurrent ANTs-heavy tasks plus many light ones); the **smoke** job's
recon-all is single-threaded, so it under-uses the node during the ~10–16 h anatomical stage — fine
to leave as-is, or dial its header back to ~16 cores since `dmello` has only 4 nodes.

### Step 0 (login node, ONE TIME): fetch the fMRI content — annex pointers won't run
`git-annex get` needs internet, so it **must run on the login node, not inside the job**. fMRIPrep
fails on dangling symlinks. Fetch anat (done) + func + fmap for the sessions you'll process:

```bash
export PATH=~/miniconda3/bin:$PATH
cd /mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/perinatal_stroke
git-annex get sub-CIMT001/ses-38659/anat                       # pinned anatomical reference
for s in 00000 0001 38659 38660 38719 38780 38781 38834 38835; do   # the 9 batch sessions
  git-annex get sub-CIMT001/ses-$s/func sub-CIMT001/ses-$s/fmap || true
done
# sanity: no dangling content (should print nothing)
find sub-CIMT001/ses-38659/func -name '*_bold.nii.gz' | while read f; do [ -e "$(readlink -f "$f")" ] || echo "MISSING $f"; done
```
(For the smoke test alone you only need `ses-38659/{anat,func,fmap}`.)

### Step 1: smoke test (one session), then QC
```bash
cd /mfs/io/groups/dmello/projects/dynamric/motor_information_connectivity_trees/plans_and_reports/methods
sbatch ps1_fmriprep_smoke.sbatch          # ~10–16 h recon-all + 4 runs; logs in ../../logs/
```
When it finishes, QC the surfaces over the cysts (the script prints the exact `freeview` line). If
the white/pial surfaces bleed into or skip the cysts → §7 (edit mask / enantiomorphic fill /
volumetric fallback) **before** the batch.

### Step 2: half batch (reuses recon-all → BOLD-only)
```bash
sbatch ps1_fmriprep_batch.sbatch          # skips recon-all via --fs-subjects-dir; ~1–1.5 days
```
Edit which 9 sessions via `ps1_bids_filter_batch.json` (default = the early 38xxx + 0000x sessions;
swap to the §2 low-motion keep-list if you prefer). For the **other** half later, edit that JSON's
`bold`/`fmap` session lists and resubmit — recon-all stays skipped.

To make Step 2 wait for Step 1 automatically instead of QC-gating by hand (not recommended for a
lesioned brain, but possible):
```bash
JID=$(sbatch --parsable ps1_fmriprep_smoke.sbatch)
sbatch --dependency=afterok:$JID ps1_fmriprep_batch.sbatch
```
Monitor: `squeue -u $USER` · tail logs: `tail -f ../../logs/ps1_smoke_*.log`.

## 6. After fMRIPrep — wire into `mict`

The dtseries land at `…/perinatal_stroke-fmriprep/sub-CIMT001/ses-*/func/
sub-CIMT001_ses-*_task-{rest,motor}_*_space-fsLR_den-91k_bold.dtseries.nii`. Then:
1. symlink into `data/` and point `config/datasets.yaml` `perinatal_stroke.surface` at it
   (`ready: true`, `root:`, `rest_glob:` to the fMRIPrep func dirs);
2. `mict.cifti.read_cifti` already handles these (CIFTI-2 from fMRIPrep — no wb_command convert needed);
3. motor-task ROIs: fit the first-level GLM on the motor dtseries (fMRIPrep has no precomputed
   contrasts like the WU MSC data) → effector peak + expansion via `mict.surface`. **For PS1, search
   whole surviving cortex (no anatomical constraint)** — the right-hand representation is displaced
   to the left postcentral gyrus (`mict.lesion` records the displacement vector);
4. exclude the high-motion runs from §2; censor with fMRIPrep's `framewise_displacement` confound.

## 7. If recon-all surfaces fail on the lesioned brain (fallback)

Try, in order: (a) improve/edit the lesion mask and re-run; (b) **enantiomorphic / virtual lesion
filling** of the T1w before recon-all (mirror healthy tissue into the cyst), then feed a manual
`--fs-subjects-dir`; (c) **volumetric fallback** — run fMRIPrep `--fs-no-reconall --output-spaces
MNI152NLin2009cAsym` (exactly the MSC recipe) and analyze PS1 in the volume; this then requires the
optional **volumetric MSC/cast path** for comparison (plan §9 "space_comparison"). The cortex is
where the displaced hand rep lives, so surfaces are strongly preferred — but (c) guarantees a result.

---

### Space-comparability note
fMRIPrep's fsLR-32k cortex uses the **same vertex mesh** as the WU MSC/cast surfaces, so cortical
ROIs/connectivity are comparable **at the vertex level** across PS1 ↔ MSC/cast. The volumetric
registration differs (fMRIPrep MNI vs WU 711-2B Talairach), so the CIFTI **subcortical** voxels
(cerebellum/thalamus) are in MNI for PS1 vs Talairach for MSC/cast — fine within-PS1, and the
cortical motor-reorganization questions (Q3) are unaffected.
