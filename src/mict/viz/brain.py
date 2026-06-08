"""viz/brain.py — brain-space motor maps (motor_system_plan.md figure suite F3–F5).

  plot_roi_surface   : cortical ROIs on the inflated fs_LR_32k surface (4 views) — the
                       participant-specific "map of the motor system".
  plot_cl_connectome : the CL tree (or any sparse adjacency) drawn on the glass brain via
                       ROI centroids (cortex from surface coords; subcortex from CIFTI voxels).

Cortical ROIs render on the surface; cerebellar/thalamic ROIs are subcortical (volume) and
appear in the connectome view via their voxel centroids.
"""

from __future__ import annotations

import glob
import json

import numpy as np

from .. import cifti as _cifti
from .. import wu
from ..paths import results_path
from ..premotor import cortex_coords
from ..rois import REGION_COLORS, region_color, roi_regions

N_VERT_32K = 32492


def _load_masks(dataset, subject, session="all_sessions"):
    d = results_path(dataset, "surface", "roi_masks", subject, session=session, create=False)
    npz = np.load(d / "roi_masks.npz")
    keys = json.load(open(d / "roi_keys.json"))["roi_keys"]
    return {k: npz[k] for k in keys}


def _ref_cifti(dataset, subject):
    """A Cifti carrying the structure (vertices/voxels/affine) for this subject."""
    f = sorted(wu._rest_dir(dataset, subject, wu.rest_sessions(dataset, subject)[0]).glob("*.dtseries.nii"))[0]
    return _cifti.read_cifti(f)


def _surface_file(surf_dir, hemi, kind="inflated"):
    hits = glob.glob(str(surf_dir) + f"/*.{hemi}.{kind}.32k_fs_LR.surf.gii")
    return hits[0] if hits else None


def plot_roi_surface(dataset, subject, kind="inflated", out_dir=None):
    """Render cortical ROIs on the fs_LR_32k surface (L/R × lateral/medial). Returns the fig."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from nilearn import plotting

    masks = _load_masks(dataset, subject)
    cif = _ref_cifti(dataset, subject)
    surf_dir = wu.surface_dir(dataset, subject)

    cort = {"L": cif.model(_cifti.CORTEX_LEFT), "R": cif.model(_cifti.CORTEX_RIGHT)}
    col2v = {h: {int(c): int(v) for c, v in zip(m.columns, m.vertices)} for h, m in cort.items()}
    colset = {h: set(m.columns.tolist()) for h, m in cort.items()}

    # per-vertex integer label maps + colour list (cortical ROIs only)
    labelmap = {"L": np.zeros(N_VERT_32K), "R": np.zeros(N_VERT_32K)}
    names, colors = [], []
    lid = 0
    for roi, cols in masks.items():
        placed = False
        for h in ("L", "R"):
            inside = [int(c) for c in cols if int(c) in colset[h]]
            if inside:
                if not placed:
                    lid += 1; names.append(roi); colors.append(region_color(roi)); placed = True
                for c in inside:
                    labelmap[h][col2v[h][c]] = lid
    if lid == 0:
        return None
    cmap = ListedColormap(["#dddddd"] + colors)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "3d"})
    panels = [("L", "lateral", axes[0, 0]), ("L", "medial", axes[1, 0]),
              ("R", "lateral", axes[0, 1]), ("R", "medial", axes[1, 1])]
    for h, view, ax in panels:
        mesh = _surface_file(surf_dir, h, kind)
        plotting.plot_surf_roi(mesh, labelmap[h], hemi={"L": "left", "R": "right"}[h],
                               view=view, cmap=cmap, vmin=0, vmax=lid, axes=ax,
                               bg_on_data=False, darkness=0.6)
        ax.set_title(f"{h} {view}", fontsize=10)
    handles = [mpatches.Patch(color=c, label=n) for n, c in zip(names, colors)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"{dataset} {subject} — motor ROIs on fs_LR surface", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir = out_dir or results_path(dataset, "surface", "figures/brain", subject, create=True)
    fig.savefig(out_dir / "roi_surface.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "roi_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{subject}] roi_surface → {out_dir}")
    return out_dir


def column_mm(cif, surf_dir) -> dict:
    """{grayordinate column → (x, y, z) mm}: cortex from midthickness, subcortex via affine."""
    mm = dict(cortex_coords(cif, surf_dir))   # cortical columns → mm
    if cif.affine is not None:
        for m in cif.models:
            if m.model_type == "VOXELS" and m.voxels is not None:
                ijk1 = np.c_[m.voxels, np.ones(len(m.voxels))]
                xyz = (cif.affine @ ijk1.T).T[:, :3]
                for col, p in zip(m.columns, xyz):
                    mm[int(col)] = tuple(float(v) for v in p)
    return mm


def roi_centroids(dataset, subject) -> tuple[list, np.ndarray]:
    """Return ``(roi_keys, centroids (n,3) mm)`` for all ROIs (cortex + subcortex)."""
    masks = _load_masks(dataset, subject)
    cif = _ref_cifti(dataset, subject)
    mm = column_mm(cif, wu.surface_dir(dataset, subject))
    keys, cents = [], []
    for roi, cols in masks.items():
        pts = [mm[int(c)] for c in cols if int(c) in mm]
        if pts:
            keys.append(roi); cents.append(np.mean(pts, axis=0))
    return keys, np.array(cents)


def plot_cl_connectome(dataset, subject, measure="cl", out_dir=None):
    """Draw the CL (or fc/mi) adjacency on the glass brain via ROI centroids. Returns the fig."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting

    keys, cents = roi_centroids(dataset, subject)
    adj = _np_load(results_path(dataset, "surface", measure, subject, create=False) / f"{measure}.npy")
    roi_keys = json.load(open(results_path(dataset, "surface", "roi_keys", subject, create=False) / "roi_keys.json"))["roi_keys"]
    # reorder adjacency to the centroid key order
    idx = [roi_keys.index(k) for k in keys]
    A = adj[np.ix_(idx, idx)]
    node_colors = [region_color(k) for k in keys]

    fig = plt.figure(figsize=(12, 5))
    plotting.plot_connectome(A, cents, node_color=node_colors, edge_threshold=None,
                             node_size=60, edge_cmap="Greys", display_mode="lyrz",
                             figure=fig, title=f"{dataset} {subject} — {measure.upper()} on brain")
    out_dir = out_dir or results_path(dataset, "surface", "figures/brain", subject, create=True)
    fig.savefig(out_dir / f"{measure}_connectome.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{measure}_connectome.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{subject}] {measure}_connectome → {out_dir}")
    return out_dir


def _np_load(path):
    return np.load(path)


# ───────────────────────── group / consensus brain maps ─────────────────────────

def group_roi_surface(dataset, subjects, min_frac=0.3, kind="inflated", out_dir=None):
    """Group ROI map: per-vertex overlap across subjects (fs_LR vertices correspond), shown
    winner-take-all by region with vertices below ``min_frac`` consensus left as background.
    Returns the figure. This is the consensus 'map of the motor system'.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from nilearn import plotting

    ref = _ref_cifti(dataset, subjects[0])
    cort = {"L": ref.model(_cifti.CORTEX_LEFT), "R": ref.model(_cifti.CORTEX_RIGHT)}
    col2v = {h: {int(c): int(v) for c, v in zip(m.columns, m.vertices)} for h, m in cort.items()}
    colset = {h: set(m.columns.tolist()) for h, m in cort.items()}

    # accumulate per-ROI per-vertex counts across subjects
    counts = {"L": {}, "R": {}}
    n = 0
    for sub in subjects:
        try:
            masks = _load_masks(dataset, sub)
        except FileNotFoundError:
            continue
        n += 1
        for roi, cols in masks.items():
            for h in ("L", "R"):
                vtx = [col2v[h][int(c)] for c in cols if int(c) in colset[h]]
                if vtx:
                    counts[h].setdefault(roi, np.zeros(N_VERT_32K))
                    counts[h][roi][vtx] += 1
    cort_rois = sorted({r for h in ("L", "R") for r in counts[h]})
    colors = [region_color(r) for r in cort_rois]
    ridx = {r: i + 1 for i, r in enumerate(cort_rois)}
    cmap = ListedColormap(["#dddddd"] + colors)

    # winner-take-all label map per hemi, thresholded at min_frac consensus
    labelmap = {}
    for h in ("L", "R"):
        stack = np.zeros((len(cort_rois), N_VERT_32K))
        for r in counts[h]:
            stack[ridx[r] - 1] = counts[h][r]
        win = stack.argmax(0)
        wval = stack.max(0) / max(n, 1)
        lm = np.where(wval >= min_frac, win + 1, 0).astype(float)
        labelmap[h] = lm

    surf_dir = wu.surface_dir(dataset, subjects[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={"projection": "3d"})
    for h, view, ax in [("L", "lateral", axes[0, 0]), ("L", "medial", axes[1, 0]),
                        ("R", "lateral", axes[0, 1]), ("R", "medial", axes[1, 1])]:
        plotting.plot_surf_roi(_surface_file(surf_dir, h, kind), labelmap[h],
                               hemi={"L": "left", "R": "right"}[h], view=view, cmap=cmap,
                               vmin=0, vmax=len(cort_rois), axes=ax, darkness=0.6)
        ax.set_title(f"{h} {view}", fontsize=10)
    handles = [mpatches.Patch(color=c, label=r) for r, c in zip(cort_rois, colors)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"{dataset} — group motor ROI map (N={n}, consensus ≥{int(min_frac*100)}%)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir = out_dir or results_path(dataset, "surface", "figures/brain", create=True)
    fig.savefig(out_dir / "group_roi_surface.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "group_roi_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  group_roi_surface (N={n}) → {out_dir}")
    return out_dir


def group_cl_connectome(dataset, subjects, out_dir=None):
    """Consensus CL tree on the glass brain: group-mean ROI centroids + edge-frequency
    adjacency (fraction of subjects whose CL tree has each edge). Returns the figure dir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from nilearn import plotting

    from ..chow_liu import chow_liu_tree, edge_set
    from ..fc_sparse import edge_frequency_matrix
    from ..io import load_matrix, load_roi_keys

    # union ROI order (use first subject with data)
    cent_acc, cent_n, edge_sets, keys = {}, {}, {}, None
    for sub in subjects:
        try:
            k, c = roi_centroids(dataset, sub)
            mi = load_matrix(dataset, "surface", "mi", sub)
            rk = load_roi_keys(dataset, "surface", sub)
        except FileNotFoundError:
            continue
        keys = keys or rk
        for roi, xyz in zip(k, c):
            cent_acc[roi] = cent_acc.get(roi, np.zeros(3)) + xyz
            cent_n[roi] = cent_n.get(roi, 0) + 1
        # edge set in the canonical key order
        tree = chow_liu_tree(mi)
        edge_sets[sub] = {frozenset((rk[u], rk[v])) for u, v in tree.edges()}
    keys = [k for k in keys if k in cent_n]
    cents = np.array([cent_acc[k] / cent_n[k] for k in keys])

    # edge-frequency adjacency over the label order
    ki = {k: i for i, k in enumerate(keys)}
    freq = np.zeros((len(keys), len(keys)))
    for es in edge_sets.values():
        for e in es:
            a, b = tuple(e)
            if a in ki and b in ki:
                freq[ki[a], ki[b]] += 1; freq[ki[b], ki[a]] += 1
    freq /= max(len(edge_sets), 1)
    node_colors = [region_color(k) for k in keys]

    fig = plt.figure(figsize=(12, 5))
    plotting.plot_connectome(freq, cents, node_color=node_colors, edge_threshold="50%",
                             node_size=70, edge_cmap="Reds", display_mode="lyrz",
                             figure=fig, title=f"{dataset} — consensus CL tree on brain (N={len(edge_sets)})")
    out_dir = out_dir or results_path(dataset, "surface", "figures/brain", create=True)
    fig.savefig(out_dir / "group_cl_connectome.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "group_cl_connectome.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  group_cl_connectome (N={len(edge_sets)}) → {out_dir}")
    return out_dir
