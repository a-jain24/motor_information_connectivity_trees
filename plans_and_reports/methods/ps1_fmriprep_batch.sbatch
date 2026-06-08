#!/bin/bash
# PS1 (sub-CIMT001) fMRIPrep — HALF BATCH: 9 of 18 sessions (the early/38xxx + 0000x sessions;
# edit ps1_bids_filter_batch.json to choose a different half, e.g. the §2 low-motion keep-list).
# REUSES the recon-all produced by ps1_fmriprep_smoke.sbatch via --fs-subjects-dir, so this run is
# BOLD-only (recon-all skipped). Run the smoke test + QC FIRST.
#   sbatch ps1_fmriprep_batch.sbatch
#SBATCH -J ps1_batch
#SBATCH -p dmello
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=490G
# Full exclusive dmello node (64 cores / ~490 GB). recon-all is skipped here (reused via
# --fs-subjects-dir), so this stage is BOLD-only and parallelizes well across the many runs:
# nipype runs up to 64/omp-nthreads ANTs-heavy tasks concurrently, which is where the cores pay off.
#SBATCH -o /mfs/io/groups/dmello/projects/dynamric/motor_information_connectivity_trees/logs/ps1_batch_%j.log
#SBATCH -e /mfs/io/groups/dmello/projects/dynamric/motor_information_connectivity_trees/logs/ps1_batch_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=dhvani.jain@utsouthwestern.edu
set -euo pipefail

# Use the system singularity (singularity-ce 4.2.2); the `module load singularity/3.8.6` build
# lacks a setuid install and errors with "No setuid installation found".
SING=/usr/bin/singularity

REPO=/mfs/io/groups/dmello/projects/dynamric/motor_information_connectivity_trees
IMG=/mfs/io/groups/dmello/software/fmriprep/fmriprep-25.0.0.sif
FS_LICENSE=/mfs/io/groups/dmello/software/fmriprep/license.txt
BIDS=/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/perinatal_stroke
OUT=/mfs/io/groups/dmello/projects/dynamric/fmri_connectivity_trees/datasets/perinatal_stroke-fmriprep
WORK=/mfs/io/groups/dmello/projects/dynamric/work/fmriprep_ps1
FSDIR=$OUT/sourcedata/freesurfer
FILTER=$REPO/plans_and_reports/methods/ps1_bids_filter_batch.json
TEMPLATEFLOW=/mfs/io/groups/dmello/projects/dynamric/templateflow

# dmello compute nodes are DISKLESS — / and /tmp are tmpfs (RAM). Keep ALL heavy I/O on the /mfs
# network filesystem: the fMRIPrep work dir (-w) AND the container's /tmp (FreeSurfer/ANTs scratch).
# Otherwise temp files fill RAM and OOM-kill the job against the 490G cgroup. (There is no local
# scratch disk: /scratch is admin-gated, /tmp is RAM.)
mkdir -p "$OUT" "$WORK" "$WORK/tmp" "$FSDIR" "$REPO/logs"
unset PYTHONPATH
export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow
export SINGULARITYENV_TMPDIR=/tmp

"$SING" run --cleanenv \
    -B /mfs \
    -B "$TEMPLATEFLOW":/templateflow \
    -B "$WORK/tmp":/tmp \
    "$IMG" \
    "$BIDS" "$OUT" participant \
    --participant-label sub-CIMT001 \
    --bids-filter-file "$FILTER" \
    --fs-subjects-dir "$FSDIR" \
    --fs-license-file "$FS_LICENSE" \
    --output-spaces MNI152NLin2009cAsym fsLR fsaverage \
    --cifti-output 91k \
    --use-syn-sdc \
    --bold2anat-dof 6 \
    --skip-bids-validation --md-only-boilerplate \
    --notrack \
    --mem-mb 460000 --nthreads 64 --omp-nthreads 8 \
    -w "$WORK"

echo "BATCH DONE. dtseries at:"
echo "  $OUT/sub-CIMT001/ses-*/func/*_space-fsLR_den-91k_bold.dtseries.nii"
