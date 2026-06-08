"""ROI-size confound check on the 19-ROI MI (roi_size_confound_plan.md).

The 19 ROIs span ~17–400 grayordinates (M1 400, cerebellar 40, premotor watershed parcels
~60–250, thalamus ~18). The histogram MI estimator benefits from spatial averaging, so
larger ROIs can show higher MI for non-neural reasons. This quantifies the confound and
checks whether the CL tree survives size-residualization.

  python analyses/confounds/roi_size.py --dataset MSC

Outputs (results/<dataset>/surface/confounds/): roi_size.{json,txt}
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from mict.chow_liu import chow_liu_tree, edge_set
from mict.io import load_matrix, load_roi_keys
from mict.paths import dataset as dcfg
from mict.paths import results_path
from mict.stats import jaccard


def run(dataset: str, space: str = "surface") -> dict:
    subjects = dcfg(dataset)["subjects"]
    mis, sizes, keys = [], [], None
    for sub in subjects:
        try:
            mi = load_matrix(dataset, space, "mi", sub)
        except FileNotFoundError:
            continue
        keys = keys or load_roi_keys(dataset, space, sub)
        mis.append(mi)
        d = results_path(dataset, space, "roi_masks", sub, create=False)
        npz = np.load(d / "roi_masks.npz")
        sizes.append([int(npz[k].size) for k in keys])

    mi_mean = np.mean(mis, axis=0)
    size_mean = np.mean(sizes, axis=0)
    n = len(keys)
    geom = np.sqrt(np.outer(size_mean, size_mean))     # geometric-mean pair size
    iu = np.triu_indices(n, 1)
    r, p = spearmanr(geom[iu], mi_mean[iu])

    # residualize group-mean MI on log(geom size); rebuild CL tree on residual
    X = np.log(geom[iu]).reshape(-1, 1)
    reg = LinearRegression().fit(X, mi_mean[iu])
    resid = mi_mean[iu] - reg.predict(X)
    mi_resid = np.zeros((n, n)); mi_resid[iu] = resid; mi_resid += mi_resid.T
    pos = mi_resid - mi_resid.min() + 1e-3; np.fill_diagonal(pos, 0)

    tree_raw = chow_liu_tree(mi_mean)
    tree_resid = chow_liu_tree(pos)
    es_raw, es_resid = edge_set(tree_raw), edge_set(tree_resid)
    surv = jaccard(es_raw, es_resid)

    canon = [("L_M1_hand", "R_M1_hand"), ("L_M1_foot", "R_M1_foot"), ("L_M1_face", "R_M1_face")]
    ki = {k: i for i, k in enumerate(keys)}
    canon_survive = {f"{a}__{b}": bool(frozenset((ki[a], ki[b])) in es_resid)
                     for a, b in canon if a in ki and b in ki}

    out = {
        "dataset": dataset, "n_rois": n, "n_subjects": len(mis),
        "roi_sizes": {k: float(s) for k, s in zip(keys, size_mean)},
        "size_vs_mi_spearman_r": float(r), "size_vs_mi_p": float(p),
        "confound_flag": bool(r > 0.4 and p < 0.05),
        "cl_edge_jaccard_raw_vs_residualized": float(surv),
        "canonical_edges_survive_residualization": canon_survive,
    }
    d = results_path(dataset, space, "confounds", create=True)
    with open(d / "roi_size.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = [
        f"ROI-size confound — {dataset} ({space}), {n} ROIs, N={len(mis)}", "",
        "ROI sizes (mean grayordinates): " + ", ".join(f"{k}={int(s)}" for k, s in zip(keys, size_mean)),
        "",
        f"Spearman(geom-mean ROI size, group-mean MI) across {len(iu[0])} pairs: "
        f"r={r:.3f}, p={p:.3g}  →  {'CONFOUND present' if out['confound_flag'] else 'no strong confound'}",
        f"CL edge-set Jaccard, raw vs size-residualized MI: {surv:.3f} "
        f"({tree_raw.number_of_edges()} edges)",
        "Canonical bilateral-M1 edges surviving residualization: "
        + ", ".join(f"{k.split('__')[0][2:]}={'yes' if v else 'no'}" for k, v in canon_survive.items()),
    ]
    txt = "\n".join(lines)
    with open(d / "roi_size.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MSC")
    ap.add_argument("--space", default="surface")
    args = ap.parse_args()
    run(args.dataset, args.space)


if __name__ == "__main__":
    main()
