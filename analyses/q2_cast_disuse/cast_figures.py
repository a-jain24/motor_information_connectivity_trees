"""Q2 cast figures — disuse trajectory networks (CL vs FC-sparse), apples-to-apples.

For cast1/cast2 (= MSC02/06), the per-condition resting-state connectivity is already
computed (cast_disuse.py). Here we render, for each method (CL tree + FC-sparse analogs):
  - a subjects × conditions tree grid (pre → cast → post), with the casted-limb bilateral
    hand edge (L_M1_hand ↔ R_M1_hand) highlighted in red + its weight labelled;
  - a trajectory line plot of the three bilateral M1 edge weights across pre/cast/post.

Casting was on the RIGHT upper limb → its M1 is the LEFT hemisphere; prediction: the hand
edge dips during `cast` and recovers `post`.

  python analyses/q2_cast_disuse/cast_figures.py --subjects sub-cast1 sub-cast2

Outputs: results/cast/surface/figures/q2/<method>/{trajectory_grid,edge_trajectory}.{pdf,png}
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from mict.fc_sparse import METHOD_LABELS, sparse_network
from mict.paths import dataset as dcfg
from mict.paths import results_path
from mict.viz.trees import condition_grid

CONDITIONS = ["pre", "cast", "post"]
CANON = [("L_M1_hand", "R_M1_hand"), ("L_M1_foot", "R_M1_foot"), ("L_M1_face", "R_M1_face")]


def _load_ts(subject, cond):
    d = results_path("cast", "surface", "timeseries", subject, session=cond, create=False)
    if not (d / "timeseries.npy").exists():
        return None, None
    keys = json.load(open(d / "roi_keys.json"))["roi_keys"]
    return np.load(d / "timeseries.npy"), keys


def _graph(res, n):
    if res["tree"] is not None:
        return res["tree"]
    g = nx.from_numpy_array(res["adjacency"])
    g.remove_edges_from([(u, v) for u, v, d in g.edges(data=True) if d["weight"] == 0])
    return g


def run(subjects, methods=("cl", "fc_mst", "pcorr_mst"), num_bins=100):
    # load all (subject, condition) timeseries once
    ts = {}
    keys = None
    for s in subjects:
        for c in CONDITIONS:
            x, k = _load_ts(s, c)
            if x is not None:
                ts[(s, c)] = x
                keys = keys or k
    if not ts:
        print("no cast per-condition timeseries — run cast_disuse.py first"); return
    n = len(keys)

    for method in methods:
        # build graph + adjacency per (subject, condition)
        graphs, adjs = {}, {}
        for (s, c), x in ts.items():
            res = sparse_network(x, method, num_bins=num_bins)
            graphs[(s, c)] = _graph(res, n)
            adjs[(s, c)] = np.asarray(res["adjacency"])

        out = results_path("cast", "surface", f"figures/q2/{method}", create=True)
        label = METHOD_LABELS[method]

        # (1) trajectory grid: rows=subjects, cols=conditions, hand edge highlighted
        cells = {(ri, ci): graphs.get((s, c))
                 for ri, s in enumerate(subjects) for ci, c in enumerate(CONDITIONS)}
        fig = condition_grid(cells, keys, row_labels=subjects, col_labels=CONDITIONS,
                             highlight=("L_M1_hand", "R_M1_hand"),
                             title=f"cast disuse — {label} (red = casted-limb hand edge)")
        fig.savefig(out / "trajectory_grid.pdf", bbox_inches="tight")
        fig.savefig(out / "trajectory_grid.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # (2) edge-weight trajectory line plot
        ki = {k: i for i, k in enumerate(keys)}
        fig, axes = plt.subplots(1, len(subjects), figsize=(5 * len(subjects), 4), squeeze=False)
        for ax, s in zip(axes[0], subjects):
            for a, b in CANON:
                if a not in ki or b not in ki:
                    continue
                y = [adjs.get((s, c), np.zeros((n, n)))[ki[a], ki[b]] for c in CONDITIONS]
                hand = a.endswith("hand")
                ax.plot(CONDITIONS, y, marker="o", lw=3 if hand else 1.5,
                        label=f"{a[2:]}" + (" (casted)" if hand else ""),
                        color="#d62728" if hand else None, zorder=3 if hand else 1)
            ax.set_title(s); ax.set_ylabel(f"edge weight ({label})"); ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(f"cast disuse — bilateral M1 edge trajectory ({label})", y=1.02)
        fig.tight_layout()
        fig.savefig(out / "edge_trajectory.pdf", bbox_inches="tight")
        fig.savefig(out / "edge_trajectory.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{method}] trajectory_grid + edge_trajectory → {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", default=["sub-cast1", "sub-cast2"])
    ap.add_argument("--methods", nargs="+", default=["cl", "fc_mst", "pcorr_mst"])
    ap.add_argument("--num-bins", type=int, default=100)
    args = ap.parse_args()
    run(args.subjects, methods=tuple(args.methods), num_bins=args.num_bins)


if __name__ == "__main__":
    main()
