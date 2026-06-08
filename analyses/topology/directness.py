"""Topology A1 + A2 — does the CL tree capture DIRECT dependencies? (vs matched FC)

A1 (partial correlation = linear directness gold standard): CL-tree edges should have higher
|partial r| than the high-MI pairs the tree *excludes*; excluded high-MI pairs should be
tree-transitive (graph distance 2). Cross-method: compare mean |partial r| of the edge sets
chosen by CL, FC-MST, partial-corr MST, FC-top-N.

A2 (conditional MI = MI-native directness): for an excluded high-MI pair routed A–C–B, the
dependence should be explained away by C (CMI(A,B|C) → 0); CL edges should survive conditioning.

  python analyses/topology/directness.py --dataset MSC

Outputs (results/<dataset>/surface/topology/): directness.{json,txt}
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.stats import ttest_rel

from mict.chow_liu import chow_liu_tree
from mict.conditional import explained_away
from mict.fc_sparse import partial_correlation, sparse_network
from mict.io import load_matrix, load_roi_keys
from mict.paths import dataset as dcfg
from mict.paths import results_path
from mict.viz.figures import load_subject_timeseries


def _edge_idx(es):
    return {tuple(sorted(tuple(e))) for e in es}


def run(dataset, space="surface"):
    subjects = dcfg(dataset)["subjects"]
    ts_by_sub = load_subject_timeseries(dataset, space, subjects)

    a1 = {"cl_edge_pcorr": [], "excluded_pcorr": [],          # A1 paired (per subject)
          "transitive_frac": [],
          "pcorr_by_method": {"cl": [], "fc_mst": [], "pcorr_mst": [], "fc_topn": []}}
    a2 = {"excluded_explained": [], "cl_edge_explained": []}  # A2 paired (per subject)

    for sub, (ts, keys) in ts_by_sub.items():
        n = len(keys)
        mi = load_matrix(dataset, space, "mi", sub)
        pc = np.abs(partial_correlation(ts))
        tree = chow_liu_tree(mi)
        cl_edges = _edge_idx((u, v) for u, v in tree.edges())

        # rank non-tree pairs by MI; the top (= #CL edges) are the "excluded high-MI" pairs
        allpairs = [(i, j) for i, j in combinations(range(n), 2)]
        nontree = [p for p in allpairs if tuple(sorted(p)) not in cl_edges]
        nontree.sort(key=lambda p: mi[p[0], p[1]], reverse=True)
        excluded = nontree[: len(cl_edges)]

        # ── A1: partial-r of CL edges vs excluded high-MI pairs
        a1["cl_edge_pcorr"].append(np.mean([pc[i, j] for i, j in cl_edges]))
        a1["excluded_pcorr"].append(np.mean([pc[i, j] for i, j in excluded]))
        # transitivity of excluded pairs in the CL tree (fraction at graph distance 2)
        d2 = 0
        for i, j in excluded:
            try:
                d2 += int(nx.shortest_path_length(tree, i, j) == 2)
            except nx.NetworkXNoPath:
                pass
        a1["transitive_frac"].append(d2 / max(len(excluded), 1))
        # cross-method mean |partial r| of each method's edge set
        for m in ("cl", "fc_mst", "pcorr_mst", "fc_topn"):
            es = _edge_idx(sparse_network(ts, m)["edges"]) if m != "cl" else cl_edges
            a1["pcorr_by_method"][m].append(np.mean([pc[i, j] for i, j in es]))

        # ── A2: conditional-MI "explained-away" for excluded distance-2 pairs vs CL edges
        ex_vals = []
        for i, j in excluded:
            try:
                path = nx.shortest_path(tree, i, j)
            except nx.NetworkXNoPath:
                continue
            if len(path) == 3:  # A–C–B
                ex_vals.append(explained_away(ts[:, i], ts[:, j], ts[:, path[1]]))
        cl_vals = []
        deg = dict(tree.degree())
        for i, j in cl_edges:
            # condition on the highest-degree neighbour of either endpoint (a competitor node)
            nbrs = [x for x in set(tree.neighbors(i)) | set(tree.neighbors(j)) if x not in (i, j)]
            if not nbrs:
                continue
            c = max(nbrs, key=lambda x: deg.get(x, 0))
            cl_vals.append(explained_away(ts[:, i], ts[:, j], ts[:, c]))
        if ex_vals:
            a2["excluded_explained"].append(float(np.mean(ex_vals)))
        if cl_vals:
            a2["cl_edge_explained"].append(float(np.mean(cl_vals)))

    # ── aggregate + paired tests
    def paired(x, y):
        x, y = np.array(x), np.array(y)
        t, p = ttest_rel(x, y)
        return {"mean_a": float(x.mean()), "mean_b": float(y.mean()),
                "delta": float(x.mean() - y.mean()), "t": float(t), "p": float(p)}

    out = {
        "dataset": dataset, "n_subjects": len(ts_by_sub),
        "A1_partial_r_directness": {
            "cl_vs_excluded": paired(a1["cl_edge_pcorr"], a1["excluded_pcorr"]),
            "excluded_transitive_frac_mean": float(np.mean(a1["transitive_frac"])),
            "mean_pcorr_by_method": {m: float(np.mean(v)) for m, v in a1["pcorr_by_method"].items()},
        },
        "A2_conditional_mi_directness": {
            "excluded_explained_mean": float(np.mean(a2["excluded_explained"])),
            "cl_edge_explained_mean": float(np.mean(a2["cl_edge_explained"])),
            "paired": paired(a2["excluded_explained"], a2["cl_edge_explained"]),
        },
    }
    d = results_path(dataset, space, "topology", create=True)
    with open(d / "directness.json", "w") as f:
        json.dump(out, f, indent=2)

    a1r = out["A1_partial_r_directness"]; a2r = out["A2_conditional_mi_directness"]
    pm = a1r["mean_pcorr_by_method"]
    lines = [
        f"Directness — {dataset} ({space}), N={out['n_subjects']}", "",
        "A1 (partial correlation = linear directness):",
        f"  mean |partial r|:  CL edges = {a1r['cl_vs_excluded']['mean_a']:.3f}  vs  "
        f"excluded high-MI pairs = {a1r['cl_vs_excluded']['mean_b']:.3f}   "
        f"(Δ={a1r['cl_vs_excluded']['delta']:+.3f}, p={a1r['cl_vs_excluded']['p']:.3g})",
        f"  excluded high-MI pairs at tree-distance 2 (transitive): "
        f"{a1r['excluded_transitive_frac_mean']*100:.0f}%",
        f"  mean |partial r| by method:  CL={pm['cl']:.3f}  FC-MST={pm['fc_mst']:.3f}  "
        f"pcorr-MST={pm['pcorr_mst']:.3f}  FC-topN={pm['fc_topn']:.3f}",
        "",
        "A2 (conditional MI = nonlinear directness):",
        f"  fraction of MI explained away by the intermediary: "
        f"excluded pairs = {a2r['excluded_explained_mean']:.2f}  vs  CL edges = {a2r['cl_edge_explained_mean']:.2f}  "
        f"(Δ={a2r['paired']['delta']:+.2f}, p={a2r['paired']['p']:.3g})",
        "",
        "Confirms directness if: CL edges have higher |partial r| than excluded high-MI pairs (A1),",
        "excluded pairs are mostly tree-transitive, and excluded pairs are explained away while CL",
        "edges survive conditioning (A2). If pcorr-MST ≥ CL on |partial r|, the linear-directness",
        "advantage is shared and CL's distinctive contribution is the nonlinear A2 result.",
    ]
    txt = "\n".join(lines)
    with open(d / "directness.txt", "w") as f:
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
