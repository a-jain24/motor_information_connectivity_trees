"""Nonlinearity A1 — is the CL↔FC-MST gap real nonlinearity? (surrogate null)

Per subject, compare the observed CL↔FC-MST edge agreement to its distribution under linear
surrogates (Gaussian + multivariate phase-randomized) that preserve correlations but destroy
nonlinearity. Observed agreement *below* the surrogate null ⇒ genuine non-Gaussian structure.

  python analyses/nonlinear/surrogate_null.py --dataset MSC --n-surrogates 100

Outputs (results/<dataset>/surface/nonlinear/): surrogate_null.{json,txt}
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mict.fc_sparse import sparse_network
from mict.paths import dataset as dcfg
from mict.paths import results_path
from mict.stats import jaccard
from mict.surrogates import SURROGATES
from mict.viz.figures import load_subject_timeseries


def _agreement(ts, num_bins):
    cl = sparse_network(ts, "cl", num_bins=num_bins)["edges"]
    fcmst = sparse_network(ts, "fc_mst")["edges"]
    return jaccard(cl, fcmst)


def run(dataset, space="surface", n_surrogates=100, num_bins=100, seed=0):
    ts_by_sub = load_subject_timeseries(dataset, space, dcfg(dataset)["subjects"])
    rng = np.random.default_rng(seed)
    per_subject = {}
    for sub, (ts, _k) in ts_by_sub.items():
        obs = _agreement(ts, num_bins)
        row = {"observed_agreement": obs}
        for name, fn in SURROGATES.items():
            vals = np.array([_agreement(fn(ts, rng), num_bins) for _ in range(n_surrogates)])
            mu, sd = float(vals.mean()), float(vals.std() + 1e-9)
            row[name] = {
                "surrogate_mean": mu, "surrogate_sd": sd,
                "z": (obs - mu) / sd,
                # one-sided p that observed is BELOW the null (more nonlinear)
                "p_below": float((np.sum(vals <= obs) + 1) / (n_surrogates + 1)),
            }
        per_subject[sub] = row

    summary = {"dataset": dataset, "n_subjects": len(per_subject),
               "n_surrogates": n_surrogates, "num_bins": num_bins, "per_subject": per_subject}
    for name in SURROGATES:
        zs = np.array([per_subject[s][name]["z"] for s in per_subject])
        # Stouffer combination of per-subject z (independent subjects)
        summary[name] = {
            "mean_observed": float(np.mean([per_subject[s]["observed_agreement"] for s in per_subject])),
            "mean_surrogate": float(np.mean([per_subject[s][name]["surrogate_mean"] for s in per_subject])),
            "mean_z": float(zs.mean()),
            "stouffer_z": float(zs.sum() / np.sqrt(len(zs))),
            "n_subjects_below_null_p05": int(np.sum([per_subject[s][name]["p_below"] < 0.05 for s in per_subject])),
        }

    d = results_path(dataset, space, "nonlinear", create=True)
    with open(d / "surrogate_null.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [f"Nonlinearity A1 — surrogate null, {dataset} ({space}), "
             f"N={len(per_subject)}, {n_surrogates} surrogates/type", ""]
    for name in SURROGATES:
        s = summary[name]
        lines.append(f"[{name:6s}] observed CL↔FC-MST agreement = {s['mean_observed']:.3f}  vs  "
                     f"surrogate {s['mean_surrogate']:.3f}   mean z={s['mean_z']:+.2f}  "
                     f"Stouffer z={s['stouffer_z']:+.2f}  "
                     f"({s['n_subjects_below_null_p05']}/{len(per_subject)} subjects below null p<.05)")
    lines += ["", "Interpretation: observed << surrogate (negative z, many subjects below null) ⇒",
              "the CL≠FC-MST gap reflects genuine nonlinearity, not estimator noise."]
    txt = "\n".join(lines)
    with open(d / "surrogate_null.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MSC")
    ap.add_argument("--space", default="surface")
    ap.add_argument("--n-surrogates", type=int, default=100)
    ap.add_argument("--num-bins", type=int, default=100)
    args = ap.parse_args()
    run(args.dataset, args.space, n_surrogates=args.n_surrogates, num_bins=args.num_bins)


if __name__ == "__main__":
    main()
