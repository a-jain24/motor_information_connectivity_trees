"""Q2 — How does the motor CL tree change with limb disuse (cast)?

For cast1 (=MSC02) and cast2 (=MSC06), reuse the subject's MSC ROI masks (same brain,
identical fs_LR_32k grayordinates), split resting-state sessions into pre / cast / post
(from each subject's sessions.tsv), and compute per-condition FC/MI/CL.

Casting was on the RIGHT upper limb (Newbold 2020) → its M1 is the LEFT hemisphere
(contralateral). Prediction: the bilateral hand edge (L_M1_hand ↔ R_M1_hand) and the
casted-limb M1 coupling weaken during `cast` and recover `post`.

  python analyses/q2_cast_disuse/cast_disuse.py --subjects sub-cast1 sub-cast2

Outputs (results/cast/surface/q2/): q2_summary.txt, q2_edges.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mict import wu
from mict.chow_liu import chow_liu_tree, edge_set
from mict.io import load_matrix, load_roi_keys
from mict.paths import dataset as dcfg
from mict.paths import results_path
from mict.stats import jaccard
from pipelines.compute_connectivity import run as compute_run
from pipelines.extract_timeseries import extract, save_timeseries

CONDITIONS = ["pre", "cast", "post"]
KEY_EDGES = [("L_M1_hand", "R_M1_hand"), ("L_M1_foot", "R_M1_foot"), ("L_M1_face", "R_M1_face")]


def process_subject(subject: str, num_bins: int = 100) -> dict:
    src = wu.roi_source("cast", subject)
    if src is None:
        print(f"  {subject}: no roi_source (cast3 needs its own ROIs) — skipping")
        return {}
    done = {}
    for cond in CONDITIONS:
        sess = wu.sessions_for_condition("cast", subject, cond)
        if not sess:
            print(f"  {subject}/{cond}: no fetched sessions")
            continue
        ts, labels = extract("cast", subject, sessions=sess,
                             mask_dataset=src[0], mask_subject=src[1])
        save_timeseries(ts, labels, "cast", subject, cond)
        compute_run("cast", subject, session=cond, num_bins=num_bins)
        done[cond] = (len(sess), ts.shape[0])
        print(f"  {subject}/{cond}: {len(sess)} sessions, {ts.shape[0]} TRs, {ts.shape[1]} ROIs")
    return done


def analyze_subject(subject: str) -> dict:
    keys, mis, trees = None, {}, {}
    for cond in CONDITIONS:
        try:
            mi = load_matrix("cast", "surface", "mi", subject, session=cond)
        except FileNotFoundError:
            continue
        keys = keys or load_roi_keys("cast", "surface", subject, session=cond)
        mis[cond] = mi
        trees[cond] = chow_liu_tree(mi)
    if not mis:
        return {}

    def gi(name):
        return keys.index(name)

    edges = {}
    for a, b in KEY_EDGES:
        i, j = gi(a), gi(b)
        edges[f"{a}__{b}"] = {c: float(mis[c][i, j]) for c in mis}
    jac = {}
    if "pre" in trees:
        es_pre = edge_set(trees["pre"])
        for c in ("cast", "post"):
            if c in trees:
                jac[f"pre_vs_{c}"] = jaccard(es_pre, edge_set(trees[c]))
    return {"subject": subject, "conditions": list(mis), "key_edge_mi": edges,
            "cl_jaccard_vs_pre": jac, "roi_keys": keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--num-bins", type=int, default=100)
    ap.add_argument("--skip-compute", action="store_true",
                    help="analyze existing per-condition connectivity only")
    args = ap.parse_args()
    subjects = args.subjects or [s for s in dcfg("cast")["subjects"] if wu.roi_source("cast", s)]

    out_dir = results_path("cast", "surface", "q2", create=True)
    results = {}
    for sub in subjects:
        print(f"\n=== {sub} ===")
        if not args.skip_compute:
            process_subject(sub, num_bins=args.num_bins)
        results[sub] = analyze_subject(sub)

    with open(out_dir / "q2_edges.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["Q2 cast-induced disuse — bilateral M1 edge MI across pre/cast/post", ""]
    for sub, r in results.items():
        if not r:
            continue
        lines.append(f"{sub}  (conditions: {', '.join(r['conditions'])})")
        for edge, vals in r["key_edge_mi"].items():
            seq = "  ".join(f"{c}={vals.get(c, float('nan')):.3f}" for c in CONDITIONS if c in vals)
            tag = "  <- casted limb (right hand → left M1)" if edge.startswith("L_M1_hand") else ""
            lines.append(f"   MI {edge:26s} {seq}{tag}")
        if r["cl_jaccard_vs_pre"]:
            lines.append("   CL-tree Jaccard vs pre: " +
                         ", ".join(f"{k.split('_vs_')[1]}={v:.2f}" for k, v in r["cl_jaccard_vs_pre"].items()))
        lines.append("")
    txt = "\n".join(lines)
    with open(out_dir / "q2_summary.txt", "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
