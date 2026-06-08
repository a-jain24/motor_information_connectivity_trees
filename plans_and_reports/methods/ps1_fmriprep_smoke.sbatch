#!/bin/bash
# PS1 (sub-CIMT001) fMRIPrep — SMOKE TEST: ses-38659 only (anat + its 4 bold runs).
# Builds the lesion-masked anatomical reference + recon-all ONCE, into a stable FS dir we reuse
# for the batch job. QC the surfaces over the cysts BEFORE submitting ps1_fmriprep_batch.sbatch.
#   sbatch ps1_fmriprep_smoke.sbatch
#SBATCH -J ps1_smoke
#SBATCH -p dmello
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=490G
# NB: the smoke test is dominated by recon-all, which is largely SINGLE-THREADED — most of these 64
# cores sit idle for the ~10-16 h anatomical stage (only the 4 BOLD runs parallelize). A full
# exclusive node mainly pays off for the BATCH job; dial this back to ~16 cores if you'd rather not
# hold a whole node (only 4 exist on dmello) just to build the anatomical reference.
#SBATCH -o /mfs/io/groups/dmello/projects/dynamric/motor_information_connectivity_trees/logs/ps1_smoke_%j.log
#SBATCH -e /mfs/io/groups/dmello/projects/dynamric/motor_information_connectivity_trees/logs/ps1_smoke_%j.log
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
FILTER=$REPO/plans_and_reports/methods/ps1_bids_filter_smoke.json
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

echo "SMOKE DONE. QC surfaces in freeview before submitting ps1_fmriprep_batch.sbatch:"
echo "  freeview -v $BIDS/sub-CIMT001/ses-38659/anat/sub-CIMT001_ses-38659_run-01_T1w.nii.gz \\"
echo "    -f $FSDIR/sub-CIMT001/surf/lh.white:edgecolor=blue $FSDIR/sub-CIMT001/surf/lh.pial:edgecolor=red \\"
echo "       $FSDIR/sub-CIMT001/surf/rh.white:edgecolor=blue $FSDIR/sub-CIMT001/surf/rh.pial:edgecolor=red"
