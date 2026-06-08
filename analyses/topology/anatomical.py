"""Topology B1 — anatomical edge scorecard (direct functional ↔ direct anatomy).

Two falsifiable tests of whether the CL tree's topology tracks motor anatomy better than
matched FC:

1. **Homologue recovery** — bilateral homologous pairs (callosal/commissural) should be
   recovered by a direct method. Score each method's consensus frequency for the 9 homologues.

2. **Crossed cortico-cerebellar preference** — the anatomically correct cortico-cerebellar
   coupling is CONTRALATERAL (cortico-ponto-cerebellar + dentato-thalamo-cortical both cross),
   so L cortex ↔ R cerebellum. A *direct* method should prefer the crossed edge over the
   ipsilateral one. We test this at the MI level (data) and at the edge level (per method).

  python analyses/topology/anatomical.py --dataset MSC

Outputs (results/<dataset>/surface/topology/): anatomical.{json,txt}
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import ttest_rel

from mict.fc_sparse import METHOD_LABELS, sparse_network
from mict.io import load_matrix, load_roi_keys
from mict.paths import dataset as dcfg
from mict.paths import results_path
from mict.viz.figures import load_subject_timeseries

HOMOLOGUES = [
    ("L_M1_hand", "R_M1_hand"), ("L_M1_foot", "R_M1_foot"), ("L_M1_face", "R_M1_face"),
    ("L_Cereb_hand", "R_Cereb_hand"), ("L_Cereb_foot", "R_Cereb_foot"), ("L_Cereb_face", "R_Cereb_face"),
    ("L_PMd", "R_PMd"), ("L_PMv", "R_PMv"), ("L_Thal", "R_Thal"),
]
EFFECTORS = ["hand", "foot", "face"]
METHODS = ["cl", "fc_mst", "pcorr_mst", "fc_topn"]


def _eidx(es):
    return {tuple(sorted(tuple(e))) for e in es}


def run(dataset, space="surface"):
    subjects = dcfg(dataset)["subjects"]
    ts_by_sub = load_subject_timeseries(dataset, space, subjects)
    keys = next(iter(ts_by_sub.values()))[1]
    ki = {k: i for i, k in enumerate(keys)}
    nsub = len(ts_by_sub)

    # per-method per-subject edge sets + group MI
    edges = {m: [] for m in METHODS}
    mis = []
    for sub, (ts, _k) in ts_by_sub.items():
        mis.append(load_matrix(dataset, space, "mi", sub))
        for m in METHODS:
            edges[m].append(_eidx(sparse_network(ts, m)["edges"]))
    mi_mean = np.mean(mis, axis=0)

    def freq(method, a, b):
        if a not in ki or b not in ki:
            return None
        e = tuple(sorted((ki[a], ki[b])))
        return np.mean([e in es for es in edges[method]])

    # ── Test 1: homologue recovery scorecard
    homol = {m: {} for m in METHODS}
    for m in METHODS:
        for a, b in HOMOLOGUES:
            f = freq(m, a, b)
            if f is not None:
                homol[m][f"{a}__{b}"] = float(f)
    homol_mean = {m: float(np.mean(list(homol[m].values()))) for m in METHODS}

    # ── Test 2: crossed vs ipsilateral cortico-cerebellar
    cases = []  # (cortical_hemi, effector) → (crossed pair, ipsi pair)
    for h in ("L", "R"):
        opp = "R" if h == "L" else "L"
        for eff in EFFECTORS:
            cort = f"{h}_M1_{eff}"
            crossed = f"{opp}_Cereb_{eff}"
            ipsi = f"{h}_Cereb_{eff}"
            if cort in ki and crossed in ki and ipsi in ki:
                cases.append((cort, crossed, ipsi))

    # MI-level (data): per subject, mean MI(crossed) vs MI(ipsi) over cases
    mi_crossed, mi_ipsi = [], []
    for mi in mis:
        mi_crossed.append(np.mean([mi[ki[c], ki[x]] for c, x, i in cases]))
        mi_ipsi.append(np.mean([mi[ki[c], ki[i]] for c, x, i in cases]))
    t_mi, p_mi = ttest_rel(mi_crossed, mi_ipsi)

    # edge-level (per method): rate that the crossed edge is in the graph vs the ipsi edge
    cc = {}
    for m in METHODS:
        cr = np.mean([freq(m, c, x) for c, x, i in cases])
        ip = np.mean([freq(m, c, i) for c, x, i in cases])
        cc[m] = {"crossed_edge_rate": float(cr), "ipsi_edge_rate": float(ip),
                 "crossed_minus_ipsi": float(cr - ip)}

    out = {
        "dataset": dataset, "n_subjects": nsub,
        "homologue_recovery": {m: {"per_edge": homol[m], "mean": homol_mean[m]} for m in METHODS},
        "crossed_corticocerebellar": {
            "mi_crossed_mean": float(np.mean(mi_crossed)),
            "mi_ipsi_mean": float(np.mean(mi_ipsi)),
            "mi_paired_t": float(t_mi), "mi_paired_p": float(p_mi),
            "edge_rate_by_method": cc,
        },
    }
    d = results_path(dataset, space, "topology", create=True)
    with open(d / "anatomical.json", "w") as f:
        json.dump(out, f, indent=2)

    lines = [f"Topology B1 — anatomical edge scorecard, {dataset} ({space}), N={nsub}", "",
             "1) Homologue recovery (mean consensus frequency over 9 bilateral pairs):"]
    for m in METHODS:
        lines.append(f"     {METHOD_LABELS[m]:22s} {homol_mean[m]:.3f}")
    lines += ["",
              "2) Crossed (contralateral) vs ipsilateral cortico-cerebellar coupling:",
              f"     MI(crossed)={np.mean(mi_crossed):.3f}  vs  MI(ipsi)={np.mean(mi_ipsi):.3f}  "
              f"(paired t={t_mi:.2f}, p={p_mi:.3g})  [crossed = anatomically correct]",
              "     edge-presence rate, crossed vs ipsi:"]
    for m in METHODS:
        v = cc[m]
        lines.append(f"     {METHOD_LABELS[m]:22s} crossed={v['crossed_edge_rate']:.2f} "
                     f"ipsi={v['ipsi_edge_rate']:.2f}  (Δ={v['crossed_minus_ipsi']:+.2f})")
    lines += ["", "Direct/anatomical topology supported if MI(crossed)>MI(ipsi) and methods place the",
              "crossed cortico-cerebellar edge more than the ipsilateral one (positive Δ)."]
    txt = "\n".join(lines)
    with open(d / "anatomical.txt", "w") as f:
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
