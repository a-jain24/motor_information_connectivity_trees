"""extract_timeseries.py — ROI-average the WU rest grayordinate timeseries.

Given grayordinate masks from define_rois.py, average each ROI's grayordinates in the
(already-preprocessed) concatenated resting-state dtseries → an (T, n_ROI) matrix.

Supports:
  --mask-dataset/--mask-subject : reuse another subject's masks (cast1=MSC02,
      cast2=MSC06 share a brain and fs_LR_32k grayordinate space).
  --sessions / --session-label  : restrict to a session subset (e.g. one cast
      condition) and store under that label instead of "all_sessions".

The WU timecourses are analysis-ready (motion-censored + nuisance-regressed); we apply
the per-frame tmask and concatenate, and z-score each ROI across time.

  python pipelines/extract_timeseries.py --dataset MSC --subjects sub-MSC01
  python pipelines/extract_timeseries.py --dataset cast --subjects sub-cast1 \
      --mask-dataset MSC --mask-subject sub-MSC02 --sessions ses-01 ses-02 --session-label pre
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mict import wu
from mict.paths import dataset as dcfg
from mict.paths import results_path


def load_masks(dataset: str, subject: str) -> dict[str, np.ndarray]:
    d = results_path(dataset, "surface", "roi_masks", subject, create=False)
    npz = np.load(d / "roi_masks.npz")
    keys = json.load(open(d / "roi_keys.json"))["roi_keys"]
    return {k: npz[k] for k in keys}


def extract(
    dataset: str,
    subject: str,
    sessions: list[str] | None = None,
    mask_dataset: str | None = None,
    mask_subject: str | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, list]:
    masks = load_masks(mask_dataset or dataset, mask_subject or subject)
    grays = wu.load_rest_concat(dataset, subject, sessions=sessions)  # (T, G)
    G = grays.shape[1]

    labels, keep = [], {}
    for k, cols in masks.items():
        if cols.max() < G:           # guard: cross-subject subcortex layout may differ
            labels.append(k); keep[k] = cols
        else:
            print(f"  [skip {k}: column {cols.max()} >= G={G} (structure mismatch)]")
    ts = np.column_stack([grays[:, keep[k]].mean(axis=1) for k in labels])
    if standardize:
        ts = (ts - ts.mean(0)) / (ts.std(0) + 1e-12)
    return ts.astype(np.float32), labels


def save_timeseries(ts, labels, dataset, subject, session_label="all_sessions") -> str:
    d = results_path(dataset, "surface", "timeseries", subject, session=session_label)
    np.save(d / "timeseries.npy", ts)
    with open(d / "roi_keys.json", "w") as f:
        json.dump({"roi_keys": labels}, f, indent=2)
    return str(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MSC")
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--mask-dataset", default=None)
    ap.add_argument("--mask-subject", default=None)
    ap.add_argument("--sessions", nargs="+", default=None)
    ap.add_argument("--session-label", default="all_sessions")
    args = ap.parse_args()

    subjects = args.subjects or dcfg(args.dataset)["subjects"]
    for sub in subjects:
        ts, labels = extract(args.dataset, sub, sessions=args.sessions,
                             mask_dataset=args.mask_dataset, mask_subject=args.mask_subject)
        d = save_timeseries(ts, labels, args.dataset, sub, args.session_label)
        print(f"{sub} [{args.session_label}]: timeseries {ts.shape} ({len(labels)} ROIs) -> {d}")


if __name__ == "__main__":
    main()
