"""io.py — consistent save/load of connectivity products, keyed by dataset/space.

Layout (all under the git-ignored results/ tree):
    results/<dataset>/<space>/<measure>/<subject>/<session>/<measure>.npy
with the ordered ROI labels stored once alongside as roi_keys.json.
"""

from __future__ import annotations

import json

import numpy as np

from .paths import results_path


def save_matrix(
    matrix: np.ndarray,
    dataset: str,
    space: str,
    measure: str,
    subject: str,
    session: str = "all_sessions",
) -> str:
    """Save a connectivity matrix (fc / mi / cl_adj) and return its path."""
    d = results_path(dataset, space, measure, subject, session)
    path = d / f"{measure}.npy"
    np.save(path, np.asarray(matrix))
    return str(path)


def load_matrix(
    dataset: str,
    space: str,
    measure: str,
    subject: str,
    session: str = "all_sessions",
) -> np.ndarray:
    d = results_path(dataset, space, measure, subject, session, create=False)
    return np.load(d / f"{measure}.npy")


def save_roi_keys(
    roi_keys: list,
    dataset: str,
    space: str,
    subject: str,
    session: str = "all_sessions",
) -> str:
    d = results_path(dataset, space, "roi_keys", subject, session)
    path = d / "roi_keys.json"
    with open(path, "w") as f:
        json.dump({"roi_keys": list(roi_keys)}, f, indent=2)
    return str(path)


def load_roi_keys(
    dataset: str,
    space: str,
    subject: str,
    session: str = "all_sessions",
) -> list:
    d = results_path(dataset, space, "roi_keys", subject, session, create=False)
    with open(d / "roi_keys.json") as f:
        return json.load(f)["roi_keys"]


def save_connectivity(
    conn: dict,
    dataset: str,
    space: str,
    subject: str,
    roi_keys: list | None = None,
    session: str = "all_sessions",
) -> None:
    """Save the dict from ``mict.connectivity.compute_connectivity`` (fc / mi / cl_adj)."""
    save_matrix(conn["fc"], dataset, space, "fc", subject, session)
    save_matrix(conn["mi"], dataset, space, "mi", subject, session)
    save_matrix(conn["cl_adj"], dataset, space, "cl", subject, session)
    if roi_keys is not None:
        save_roi_keys(roi_keys, dataset, space, subject, session)
