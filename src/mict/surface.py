"""surface.py — fs_LR_32k geodesic helpers for the Newbold 400-vertex expansion.

The WU/Gordon ``surface_pipeline`` ships a precomputed per-subject geodesic (surface)
+ euclidean (subcortex) distance matrix
(``cifti_distances/<sub>distmat_surf_geodesic_vol_euc_xhem_large_uint8.mat``), so the
400-vertex expansion needs no on-the-fly geodesic computation.

The Newbold (2020) ROI rule: take the task-activation peak vertex, then grow to a
preset size (400 contiguous vertices on cortex, 40 voxels in cerebellum) along the
highest-activation contiguous neighborhood.
"""

from __future__ import annotations

import numpy as np


def load_distance_matrix(mat_path):
    """Load the precomputed (G, G) geodesic/euclidean distance matrix.

    The WU distmats are MATLAB v7.3 (HDF5, ~4 GB uint8); returns the h5py Dataset so
    rows can be read lazily (``D[i]``). Falls back to scipy.io for v7 .mat files,
    returning a numpy array. Both support row indexing ``D[i]``.
    """
    try:
        import h5py
        f = h5py.File(str(mat_path), "r")
        keys = [k for k in f.keys() if not k.startswith("#")]
        for k in keys:
            d = f[k]
            if d.ndim == 2 and d.shape[0] == d.shape[1]:
                return d  # lazy HDF5 dataset; symmetric so transpose-order is irrelevant
        f.close()
    except (OSError, ImportError):
        pass
    from scipy.io import loadmat
    md = loadmat(str(mat_path))
    for k, v in md.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return arr
    raise ValueError(f"no square distance matrix found in {mat_path}")


def nearest_vertices(seed: int, dist: np.ndarray, n: int) -> np.ndarray:
    """The ``n`` vertices closest to ``seed`` (includes seed), by the distance matrix."""
    order = np.argsort(dist[seed], kind="stable")
    return order[:n]


def grow_region(
    seed: int,
    stat: np.ndarray,
    dist: np.ndarray,
    n: int,
    step_mm: float = 4.0,
) -> np.ndarray:
    """Newbold-style contiguous activation-weighted growth from ``seed`` to ``n`` vertices.

    Greedy region growing: starting at the peak ``seed``, repeatedly add the
    highest-``stat`` vertex among those within ``step_mm`` geodesic distance of the
    current cluster (its frontier), until the cluster has ``n`` vertices. Restricts
    growth to a finite candidate neighborhood (``2*n`` nearest) for speed.

    Parameters
    ----------
    seed : peak vertex index (into the grayordinate/structure array).
    stat : (G,) task-activation statistic (e.g. z-map column).
    dist : (G, G) geodesic distance matrix for the same indexing as ``stat``.
    n : target ROI size (400 cortex / 40 cerebellum).
    step_mm : adjacency radius defining contiguity.
    """
    stat = np.asarray(stat, dtype=float)
    rows: dict[int, np.ndarray] = {}

    def row(v: int) -> np.ndarray:
        if v not in rows:
            rows[v] = np.asarray(dist[v], dtype=np.float32)  # lazy row read (h5py or numpy)
        return rows[v]

    candidates = set(int(v) for v in nearest_vertices(seed, dist, max(2 * n, n + 50)))
    candidates.add(int(seed))

    cluster = {int(seed)}
    while len(cluster) < n:
        # frontier = candidate vertices within step_mm of any cluster member
        cl = np.fromiter(cluster, dtype=int)
        frontier = [v for v in candidates
                    if v not in cluster and row(v)[cl].min() <= step_mm]
        if not frontier:
            break  # no contiguous candidates left
        nxt = max(frontier, key=lambda v: stat[v])
        cluster.add(int(nxt))
    return np.array(sorted(cluster), dtype=int)


def peak_vertex(stat: np.ndarray, restrict: np.ndarray | None = None) -> int:
    """Index of the maximum-``stat`` vertex, optionally within a ``restrict`` index set.

    For PS1 (displaced representations) ``restrict`` should span all surviving cortex
    rather than a canonical anatomical parcel — see motor_system_plan.md §A step 2.
    """
    stat = np.asarray(stat, dtype=float)
    if restrict is None:
        return int(np.nanargmax(stat))
    restrict = np.asarray(restrict, dtype=int)
    return int(restrict[np.nanargmax(stat[restrict])])
