"""premotor.py — SMA / PMd / PMv ROI definition (motor_system_plan.md §C).

The literature method (§C) is Gordon gradient-watershed parcellation + Infomap network
assignment refined by task. As a tractable, on-disk realization we define each premotor
area **functionally + anatomically**: the top task-active vertices (combined-motor
``Allcondition_avg`` contrast) within an anatomical premotor coordinate box on the
fs_LR_32k surface. Boxes follow premotor anatomy (Picard & Strick 2001) and the GCSS
centroid routing (gcss_plan.md): SMA medial superior-frontal; PMd dorsolateral; PMv
ventrolateral. (The full individual gradient-watershed parcellation is the planned
refinement; this gives anatomically-labeled, task-weighted premotor ROIs now.)

Coordinates are the surfaces' native (711-2B Talairach) mm — fine for within-MSC routing.
"""

from __future__ import annotations

import glob

import numpy as np

from . import cifti as _cifti

# Talairach mm coordinate boxes per premotor area: (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi).
PREMOTOR_BOXES = {
    "SMA":   ((-15, 15), (-15, 12), (52, 200)),    # medial superior frontal (SMA/pre-SMA)
    "L_PMd": ((-45, -18), (-20, 10), (50, 200)),   # dorsolateral precentral/SFG
    "R_PMd": ((18, 45),  (-20, 10), (50, 200)),
    "L_PMv": ((-62, -38), (-14, 16), (20, 48)),    # ventrolateral precentral
    "R_PMv": ((38, 62),  (-14, 16), (20, 48)),
}

CONTRAST = "Allcondition_avg"   # combined-motor activation (present in the WU motor dscalar)


def cortex_coords(cif, surface_dir) -> dict[int, tuple]:
    """Map each cortex grayordinate column → its midthickness (x, y, z) mm.

    Uses the per-subject fs_LR_32k midthickness surfaces (``*.{L,R}.midthickness.32k_fs_LR
    .surf.gii``) and the CIFTI surface vertex indices.
    """
    import nibabel as nib

    out: dict[int, tuple] = {}
    for hemi, struct in [("L", _cifti.CORTEX_LEFT), ("R", _cifti.CORTEX_RIGHT)]:
        hits = glob.glob(str(surface_dir) + f"/*.{hemi}.midthickness.32k_fs_LR.surf.gii")
        if not hits:
            raise FileNotFoundError(f"no {hemi} midthickness surf.gii in {surface_dir}")
        coords = nib.load(hits[0]).darrays[0].data        # (32492, 3)
        m = cif.model(struct)
        for col, vtx in zip(m.columns, m.vertices):
            out[int(col)] = tuple(float(v) for v in coords[vtx])
    return out


def define_premotor_rois(contrasts, coords: dict, n: int = 200,
                         contrast: str = CONTRAST) -> dict[str, np.ndarray]:
    """Coordinate-box fallback (no parcellation): top-``n`` task-active vertices per box."""
    stat = contrasts.data[contrasts.map_index(contrast)]
    cols = np.array(sorted(coords), dtype=int)
    xyz = np.array([coords[int(c)] for c in cols])
    out: dict[str, np.ndarray] = {}
    for label, ((xlo, xhi), (ylo, yhi), (zlo, zhi)) in PREMOTOR_BOXES.items():
        m = ((xyz[:, 0] >= xlo) & (xyz[:, 0] <= xhi)
             & (xyz[:, 1] >= ylo) & (xyz[:, 1] <= yhi)
             & (xyz[:, 2] >= zlo) & (xyz[:, 2] <= zhi))
        cand = cols[m]
        if cand.size == 0:
            continue
        order = np.argsort(stat[cand])[::-1]
        out[label] = np.sort(cand[order[: min(n, cand.size)]]).astype(np.int32)
    return out


def define_premotor_rois_watershed(parcellation, contrasts, coords: dict,
                                   contrast: str = CONTRAST, min_size: int = 60) -> dict[str, np.ndarray]:
    """§C, the gradient-watershed realization. For each premotor anatomical box, take the
    **most task-active individual watershed parcel** whose centroid falls in the box → that
    parcel's vertices are the ROI (respecting the subject's own parcel boundaries).

    parcellation : (G_cortex,) parcel id per cortex grayordinate column (WU surface_parcellation).
    Small parcels (< min_size) are augmented with the next-best in-box parcel for stability.
    """
    stat = contrasts.data[contrasts.map_index(contrast)]
    parc = np.asarray(parcellation)
    pids = [p for p in np.unique(parc) if p > 0]
    info = {}
    for p in pids:
        cols = np.where(parc == p)[0]                      # cortex grayordinate columns
        cen = np.mean([coords[int(c)] for c in cols], axis=0)
        info[p] = (cols, cen, float(stat[cols].mean()))

    out: dict[str, np.ndarray] = {}
    for label, ((xlo, xhi), (ylo, yhi), (zlo, zhi)) in PREMOTOR_BOXES.items():
        cands = [p for p, (cols, cen, act) in info.items()
                 if xlo <= cen[0] <= xhi and ylo <= cen[1] <= yhi and zlo <= cen[2] <= zhi]
        if not cands:
            continue
        cands.sort(key=lambda p: info[p][2], reverse=True)  # most task-active first
        cols = info[cands[0]][0]
        for p in cands[1:]:                                 # grow with next-best if tiny
            if cols.size >= min_size:
                break
            cols = np.union1d(cols, info[p][0])
        out[label] = np.sort(cols).astype(np.int32)
    return out
