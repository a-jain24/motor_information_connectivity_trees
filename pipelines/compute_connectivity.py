"""compute_connectivity.py — FC, MI, and CL tree from ROI timeseries.

The space-agnostic step: load the (T, n_ROI) matrix from extract_timeseries.py,
compute FC / MI / CL via mict.connectivity, and save under results/.

  python pipelines/compute_connectivity.py --dataset MSC --subjects MSC01 --num-bins 100
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mict.connectivity import compute_connectivity
from mict.io import save_connectivity
from mict.paths import dataset as dcfg
from mict.paths import results_path


def load_timeseries(dataset: str, subject: str, session: str = "all_sessions") -> tuple[np.ndarray, list]:
    d = results_path(dataset, "surface", "timeseries", subject, session=session, create=False)
    ts = np.load(d / "timeseries.npy")
    keys = json.load(open(d / "roi_keys.json"))["roi_keys"]
    return ts, keys


def run(dataset: str, subject: str, session: str = "all_sessions",
        num_bins: int = 100, use_torch: bool = False) -> dict:
    ts, keys = load_timeseries(dataset, subject, session)
    conn = compute_connectivity(ts, num_bins=num_bins, use_torch=use_torch)
    save_connectivity(conn, dataset, "surface", subject, roi_keys=keys, session=session)
    return conn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MSC")
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--session-label", default="all_sessions")
    ap.add_argument("--num-bins", type=int, default=100)
    ap.add_argument("--torch", action="store_true", help="use the torch/GPU MI path")
    args = ap.parse_args()

    subjects = args.subjects or dcfg(args.dataset)["subjects"]
    for sub in subjects:
        conn = run(args.dataset, sub, session=args.session_label,
                   num_bins=args.num_bins, use_torch=args.torch)
        n = conn["fc"].shape[0]
        print(f"{sub}: fc{conn['fc'].shape} mi{conn['mi'].shape} "
              f"CL edges={conn['cl_tree'].number_of_edges()} (expect {n - 1})")


if __name__ == "__main__":
    main()
