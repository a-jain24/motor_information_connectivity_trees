# atlases/

Atlas file **locations** are recorded in `config/atlases.yaml`; this directory holds only
small committed reference files (`.txt`/`.tsv` label tables). Large NIfTI/GIFTI/CIFTI atlas
files are git-ignored — fetch or symlink them per `config/atlases.yaml`.

| Atlas | Use | Status |
|---|---|---|
| Glasser/HCP-MMP1 fs_LR_32k `.dlabel.nii` | cortical parcellation (surface) | **fetch** (BALSA / templateflow / neuromaps) |
| fs_LR_32k surfaces | geodesic expansion + viz | on disk (in WU `surface_pipeline`; group mesh in lab software) |
| SUIT `atl-Anatom` (MNI, 2 mm) | cerebellar ROIs (subcortical volume) | on disk (egcerebellum) |
| Morel thalamus (MNI152) | motor thalamus labeling | on disk (fmri_connectivity_trees) |

The only outstanding fetch is the Glasser fs_LR_32k `.dlabel.nii` (Phase 2 prerequisite).
