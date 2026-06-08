"""Q1 — Is the motor CL tree conserved across healthy subjects, more than FC?

Loads every MSC subject's connectivity (from compute_connectivity.py), builds the
consensus CL tree, and runs the primary hypothesis test: CL-edge Jaccard vs top-(N-1)
FC-edge Jaccard across all subject pairs (cl_vs_fc_comparison_plan.md).

  python analyses/q1_healthy_conservation/consensus.py --dataset MSC

Outputs (results/<dataset>/surface/q1/): consensus_tree.pdf, conservation.json, summary.txt
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mict.chow_liu import chow_liu_tree, consensus_tree, edge_set
from mict.connectivity import top_n_fc_edges
from mict.io import load_matrix, load_roi_keys
from mict.paths import dataset as dcfg
from mict.paths import results_path
from mict.stats import conservation_test, pairwise_jaccard


def run(dataset: str, space: str = "surface") -> dict:
    subjects = dcfg(dataset)["subjects"]
    keys, fc, cl_sets, fc_sets, trees = None, {}, {}, {}, {}
    loaded = []
    for sub in subjects:
        try:
            mi = load_matrix(dataset, space, "mi", sub)
            fcm = load_matrix(dataset, space, "fc", sub)
        except FileNotFoundError:
            continue
        keys = keys or load_roi_keys(dataset, space, sub)
        n = mi.shape[0]
        t = chow_liu_tree(mi)
        trees[sub] = t
        cl_sets[sub] = edge_set(t)
        fc_sets[sub] = top_n_fc_edges(fcm, n - 1)   # match CL edge count
        loaded.append(sub)

    n = len(keys)
    out_dir = results_path(dataset, space, "q1", create=True)

    # consensus tree (edges in >=50% of subjects)
    ctree, votes = consensus_tree(trees, n, threshold=0.5)
    test = conservation_test(cl_sets, fc_sets)

    # canonical edges (>=50% of subjects) by name
    canon = []
    for i in range(n):
        for j in range(i + 1, n):
            if votes[i, j] >= 0.5:
                canon.append((keys[i], keys[j], float(votes[i, j])))
    canon.sort(key=lambda e: -e[2])

    summary = {
        "subjects": loaded, "n_subjects": len(loaded), "n_rois": n, "roi_keys": keys,
        "conservation": {k: v for k, v in test.items() if k not in ("j_cl", "j_fc")},
        "consensus_edges": canon,
    }
    with open(out_dir / "conservation.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"Q1 healthy conservation — {dataset} ({space}), N={len(loaded)} subjects, {n} ROIs",
        "",
        f"CL inter-subject Jaccard:  {test['cl_jaccard_mean']:.3f} ± {test['cl_jaccard_sd']:.3f}",
        f"FC top-{n-1} Jaccard:       {test['fc_jaccard_mean']:.3f} ± {test['fc_jaccard_sd']:.3f}",
        f"Δ (CL−FC) = {test['delta']:+.3f}   paired t({test['n_pairs']-1})={test['t']:.2f}, p={test['p']:.4g}",
        "",
        "Consensus edges (≥50% of subjects):",
    ]
    lines += [f"  {a:13s} — {b:13s}  {int(round(v*len(loaded)))}/{len(loaded)}" for a, b, v in canon]
    txt = "\n".join(lines)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)

    # consensus figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mict.viz.trees import draw_hierarchical_tree
        if ctree.number_of_nodes():
            fig, ax = plt.subplots(figsize=(12, 8))
            draw_hierarchical_tree(ctree, keys, title=f"{dataset} consensus motor CL tree (N={len(loaded)})", ax=ax)
            fig.savefig(out_dir / "consensus_tree.pdf", bbox_inches="tight")
            plt.close(fig)
    except Exception as e:  # plotting is optional
        print(f"[figure skipped: {e}]")

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MSC")
    ap.add_argument("--space", default="surface")
    args = ap.parse_args()
    run(args.dataset, args.space)


if __name__ == "__main__":
    main()
