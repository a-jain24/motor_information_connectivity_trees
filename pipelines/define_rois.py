"""define_rois.py — define motor ROIs as grayordinate-column index sets.

Implements the literature-primary method (motor_system_plan.md §A/§E): per-effector
task-activation peak + expansion to a preset size (400 grayordinates on cortex,
40 in cerebellum; Newbold 2020). Cortical M1 effectors use the lateralized
vs-baseline contrasts (contralateral hemisphere); cerebellar effectors use the
bilateral contrasts within each cerebellar hemisphere.

Expansion is ``topk`` (highest-activation grayordinates within the structure — cheap,
no distance matrix) by default, or ``geodesic`` (contiguous growth via the precomputed
distance matrix — Newbold-exact; loads ~4 GB).

  python pipelines/define_rois.py --dataset MSC --subjects MSC01 ...
  python pipelines/define_rois.py --dataset MSC --expand geodesic

Outputs: results/<dataset>/surface/roi_masks/<subject>/all_sessions/{roi_masks.npz, roi_keys.json}

NOTE: premotor (SMA/PMd/PMv — network parcels, §C) and motor thalamus (FC-seed, §D)
are defined in a separate step; this script covers the 6 M1 + 6 cerebellar effector ROIs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mict import cifti, wu
from mict.paths import results_path, roi_scheme_config
from mict.rois import roi_scheme

# Cerebellar effector → bilateral contrast map name (peak taken within each cb hemi).
CEREB_CONTRAST = {"hand": "LHand+RHand", "foot": "LLeg+RLeg", "face": "Tongue"}

_CORTEX = {"L": cifti.CORTEX_LEFT, "R": cifti.CORTEX_RIGHT}
_CEREB = {"L": cifti.CEREBELLUM_LEFT, "R": cifti.CEREBELLUM_RIGHT}


def _expand_topk(stat: np.ndarray, candidate_cols: np.ndarray, n: int) -> np.ndarray:
    """Top-``n`` grayordinates by ``stat`` within ``candidate_cols`` (peak included)."""
    vals = stat[candidate_cols]
    order = np.argsort(vals)[::-1]
    n = min(n, len(candidate_cols))
    return np.sort(candidate_cols[order[:n]])


def _expand_geodesic(stat, peak_col, candidate_cols, n, dist_path):
    """Newbold-exact contiguous growth from the peak via the precomputed distance matrix."""
    from mict.surface import grow_region, load_distance_matrix
    dist = load_distance_matrix(dist_path)
    sub = grow_region(int(peak_col), stat, dist, n)
    return np.intersect1d(sub, candidate_cols)


def define_effector_rois(
    dataset: str, subject: str, expand: str = "topk", smoothed: bool = True
) -> dict[str, np.ndarray]:
    """Return ``{roi_label: grayordinate_column_indices}`` for the effector ROIs."""
    contrasts = wu.load_motor_contrasts(dataset, subject, smoothed=smoothed)
    scheme = roi_scheme_config()["expansion"]
    n_cortex = scheme["cortex_n_vertices"]
    n_cereb = scheme["cerebellum_n_voxels"]
    dist_path = wu.distance_matrix_path(dataset, subject) if expand == "geodesic" else None

    masks: dict[str, np.ndarray] = {}
    for r in roi_scheme():
        if r["method"] != "task_peak_expand":
            continue
        eff, hemi, region = r["effector"], r["hemi"], r["region"]

        if region == "M1":
            map_name = wu.EFFECTOR_CONTRAST[f"{eff}_{hemi}"]
            cand = contrasts.model(_CORTEX[hemi]).columns
            n = n_cortex
        elif region == "Cereb":
            map_name = CEREB_CONTRAST[eff]
            cand = contrasts.model(_CEREB[hemi]).columns
            n = n_cereb
        else:
            continue

        stat = contrasts.data[contrasts.map_index(map_name)]  # (G,)
        peak = cand[int(np.argmax(stat[cand]))]
        if expand == "geodesic":
            cols = _expand_geodesic(stat, peak, cand, n, dist_path)
        else:
            cols = _expand_topk(stat, cand, n)
        masks[r["label"]] = np.asarray(cols, dtype=np.int32)
    return masks


def define_all_rois(dataset: str, subject: str, expand: str = "topk",
                    smoothed: bool = True, premotor_n: int = 200,
                    thal_sessions: int = 3) -> dict[str, np.ndarray]:
    """The full 19-ROI scheme: M1 + cerebellar effectors (§A/§E) + premotor (§C) +
    motor thalamus (§D). Cortex/cerebellum ROIs come from task contrasts; premotor from
    combined-motor activation within anatomical boxes; thalamus from an FC-seed.
    """
    from mict import wu
    from mict.premotor import (cortex_coords, define_premotor_rois,
                               define_premotor_rois_watershed)
    from mict.subcortex import define_thalamus_rois

    masks = define_effector_rois(dataset, subject, expand=expand, smoothed=smoothed)

    # premotor (§C): gradient-watershed parcels (preferred) — the most task-active individual
    # parcel per premotor anatomical box; falls back to a coordinate box if no parcellation.
    contrasts = wu.load_motor_contrasts(dataset, subject, smoothed=smoothed)
    coords = cortex_coords(contrasts, wu.surface_dir(dataset, subject))
    try:
        parc = wu.load_parcellation(dataset, subject)
        masks.update(define_premotor_rois_watershed(parc, contrasts, coords))
    except FileNotFoundError:
        masks.update(define_premotor_rois(contrasts, coords, n=premotor_n))

    # thalamus (§D): FC-seed from the M1 effector vertices (a few sessions suffice to locate it)
    m1 = [k for k in masks if k.startswith(("L_M1", "R_M1"))]
    seed_cols = np.concatenate([masks[k] for k in m1])
    sessions = wu.rest_sessions(dataset, subject)[:thal_sessions]
    rest = wu.load_rest_concat(dataset, subject, sessions=sessions)
    cif = wu.load_motor_contrasts(dataset, subject)  # any cifti with thalamus structures + same G
    masks.update(define_thalamus_rois(rest, cif, seed_cols))
    return masks


def save_masks(masks: dict, dataset: str, subject: str) -> str:
    d = results_path(dataset, "surface", "roi_masks", subject)
    np.savez(d / "roi_masks.npz", **{k: v for k, v in masks.items()})
    with open(d / "roi_keys.json", "w") as f:
        json.dump({"roi_keys": list(masks.keys()),
                   "sizes": {k: int(v.size) for k, v in masks.items()}}, f, indent=2)
    return str(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MSC")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="default: all subjects in config/datasets.yaml")
    ap.add_argument("--expand", choices=["topk", "geodesic"], default="topk")
    ap.add_argument("--rois", choices=["effectors", "full"], default="full",
                    help="effectors = 12 (M1+cerebellar); full = 19 (+ premotor + thalamus)")
    ap.add_argument("--no-smooth", action="store_true", help="use unsmoothed contrasts")
    args = ap.parse_args()

    from mict.paths import dataset as dcfg
    subjects = args.subjects or dcfg(args.dataset)["subjects"]
    for sub in subjects:
        if args.rois == "full":
            masks = define_all_rois(args.dataset, sub, expand=args.expand,
                                    smoothed=not args.no_smooth)
        else:
            masks = define_effector_rois(args.dataset, sub, expand=args.expand,
                                         smoothed=not args.no_smooth)
        d = save_masks(masks, args.dataset, sub)
        sizes = {k: v.size for k, v in masks.items()}
        print(f"{sub}: {len(masks)} ROIs {sizes} -> {d}")


if __name__ == "__main__":
    main()
