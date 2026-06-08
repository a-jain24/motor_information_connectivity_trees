"""CL vs FC-based sparse networks — apples-to-apples (Q1 extension).

For each method (Chow-Liu MI tree and the FC analogs at matched sparsity), build the
per-subject sparse network and compare on:
  1. inter-subject conservation (mean pairwise edge-set Jaccard) — the core hypothesis;
  2. canonical bilateral-M1 edge recovery (consensus frequency of L↔R hand/foot/face);
  3. CL ≡ FC-MST agreement per subject — under Gaussianity these are identical, so the
     gap quantifies the *nonlinear* dependence MI adds over correlation;
  4. paired t-test of CL vs each method on conservation.

  python analyses/q1_healthy_conservation/cl_vs_fc_sparse.py --dataset MSC

Outputs (results/<dataset>/surface/q1/): cl_vs_fc_sparse.json, cl_vs_fc_sparse.txt
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np

from mict.fc_sparse import METHODS, METHOD_LABELS, edge_frequency_matrix, sparse_network
from mict.paths import results_path
from mict.stats import jaccard, pairwise_jaccard
from mict.viz.figures import load_subject_timeseries
from mict.paths import dataset as dcfg

CANONICAL = [("L_M1_hand", "R_M1_hand"), ("L_M1_foot", "R_M1_foot"), ("L_M1_face", "R_M1_face")]


def run(dataset: str, space: str = "surface", num_bins: int = 100,
        methods=tuple(METHODS)) -> dict:
    subjects = dcfg(dataset)["subjects"]
    ts_by_sub = load_subject_timeseries(dataset, space, subjects)
    keys = next(iter(ts_by_sub.values()))[1]
    n = len(keys)
    kidx = {k: i for i, k in enumerate(keys)}

    # per-method per-subject edge sets
    edges = {m: {} for m in methods}
    for sub, (ts, _k) in ts_by_sub.items():
        for m in methods:
            edges[m][sub] = sparse_network(ts, m, num_bins=num_bins)["edges"]
    subs = list(ts_by_sub)

    summary = {"dataset": dataset, "n_subjects": len(subs), "n_rois": n, "methods": {}}
    for m in methods:
        jac = pairwise_jaccard(edges[m])
        freq = edge_frequency_matrix(edges[m], n)
        canon = {f"{a}__{b}": float(freq[kidx[a], kidx[b]])
                 for a, b in CANONICAL if a in kidx and b in kidx}
        summary["methods"][m] = {
            "label": METHOD_LABELS[m],
            "conservation_jaccard_mean": float(np.mean(jac)),
            "conservation_jaccard_sd": float(np.std(jac)),
            "canonical_edge_freq": canon,
            "mean_edges": float(np.mean([len(edges[m][s]) for s in subs])),
        }

    # CL ≡ FC-MST agreement per subject (nonlinearity probe)
    if "cl" in methods and "fc_mst" in methods:
        agree = [jaccard(edges["cl"][s], edges["fc_mst"][s]) for s in subs]
        summary["cl_vs_fcmst_edge_agreement_mean"] = float(np.mean(agree))
        summary["cl_vs_fcmst_edge_agreement_sd"] = float(np.std(agree))

    # paired CL-vs-method conservation tests
    from scipy.stats import ttest_rel
    pairs = list(combinations(subs, 2))
    jc_cl = [jaccard(edges["cl"][a], edges["cl"][b]) for a, b in pairs] if "cl" in methods else None
    summary["paired_vs_cl"] = {}
    if jc_cl is not None:
        for m in methods:
            if m == "cl":
                continue
            jc_m = [jaccard(edges[m][a], edges[m][b]) for a, b in pairs]
            t, p = ttest_rel(jc_cl, jc_m)
            summary["paired_vs_cl"][m] = {"delta_cl_minus_m": float(np.mean(jc_cl) - np.mean(jc_m)),
                                          "t": float(t), "p": float(p)}

    out = results_path(dataset, space, "q1", create=True)
    with open(out / "cl_vs_fc_sparse.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [f"CL vs FC-sparse networks — {dataset} ({space}), N={len(subs)}, {n} ROIs", ""]
    lines.append(f"{'method':22s} {'cons.Jaccard':>14s} {'edges':>7s}   canonical bilateral-M1 freq")
    for m in methods:
        s = summary["methods"][m]
        cf = " ".join(f"{k.split('__')[0][2:]}:{v:.1f}" for k, v in s["canonical_edge_freq"].items())
        lines.append(f"{s['label']:22s} {s['conservation_jaccard_mean']:.3f}±{s['conservation_jaccard_sd']:.2f}"
                     f"   {s['mean_edges']:5.1f}   {cf}")
    if "cl_vs_fcmst_edge_agreement_mean" in summary:
        a = summary["cl_vs_fcmst_edge_agreement_mean"]
        lines += ["", f"CL ≡ FC-MST per-subject edge agreement: {a:.3f} "
                      f"(1.0 ⇒ Gaussian/linear; <1 ⇒ MI adds nonlinear structure)"]
    if summary["paired_vs_cl"]:
        lines += ["", "Conservation, CL vs each (paired over subject pairs):"]
        for m, r in summary["paired_vs_cl"].items():
            lines.append(f"  CL − {METHOD_LABELS[m]:22s} Δ={r['delta_cl_minus_m']:+.3f}  p={r['p']:.3g}")
    txt = "\n".join(lines)
    with open(out / "cl_vs_fc_sparse.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MSC")
    ap.add_argument("--space", default="surface")
    ap.add_argument("--num-bins", type=int, default=100)
    args = ap.parse_args()
    run(args.dataset, args.space, num_bins=args.num_bins)


if __name__ == "__main__":
    main()
