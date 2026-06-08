"""wu.py — access to WU/Gordon ``surface_pipeline`` derivatives (MSC + cast).

These derivatives are already preprocessed (motion-censored, nuisance-regressed,
fs_LR_32k grayordinate) — so the resting-state timecourses are analysis-ready: we
apply the per-frame ``tmask`` and concatenate sessions, but do NOT re-run the
confound model (that path is for fMRIPrep volume / PS1).

Layout (per ``config/datasets.yaml``):
  {subject}/processed_restingstate_timecourses/{session}/cifti/*_bold_32k_fsLR.dtseries.nii
                                                              /*_tmask.txt
  {subject}/task_contrasts_cifti/motor/{subject}-motor_contrasts_32k_fsLR.dscalar.nii
  {subject}/surface_parcellation/{subject}_parcels.dtseries.nii
                                /{subject}_parcel_networks.dscalar.nii
  {subject}/cifti_distances/{subject}distmat_*_uint8.mat
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import cifti as _cifti
from .paths import dataset as _dataset_cfg
from .paths import resolve

# Effector → the per-effector vs-baseline motor contrast map name (MSC/cast WU GLM).
# Lateralized hand/leg (contralateral M1); tongue is bilateral.
EFFECTOR_CONTRAST = {
    "hand_L": "RHand",   # right-hand movement → left M1
    "hand_R": "LHand",   # left-hand movement  → right M1
    "foot_L": "RLeg",
    "foot_R": "LLeg",
    "face_L": "Tongue",  # bilateral; restrict peak search by hemisphere
    "face_R": "Tongue",
}


def _surface_cfg(dataset_name: str) -> dict:
    return _dataset_cfg(dataset_name)["surface"]


def subject_dir(dataset_name: str, subject: str) -> Path:
    root = resolve(_surface_cfg(dataset_name)["root"])
    return root / subject


def _has_rest_content(dataset_name: str, subject: str, session: str) -> bool:
    """True only if the session's dtseries content is materialized (not an annex pointer)."""
    d = _rest_dir(dataset_name, subject, session)
    if not d.exists():
        return False
    for dts in d.glob("*.dtseries.nii"):
        return os.path.exists(os.path.realpath(dts))  # follows symlink; False if dangling
    return False


def rest_sessions(dataset_name: str, subject: str) -> list[str]:
    """Session names whose resting-state dtseries content is present (fetched), sorted."""
    cfg = _surface_cfg(dataset_name)
    rest_tpl = cfg["rest_glob"]
    base = resolve(cfg["root"])
    # rest_glob may contain {session}; split on it to find the parent and iterate
    if "{session}" in rest_tpl:
        parent_tpl = rest_tpl.split("{session}")[0].format(subject=subject)
        parent = base / parent_tpl
    else:
        parent = base / rest_tpl.format(subject=subject)
    if not parent.exists():
        return []
    return sorted(p.name for p in parent.iterdir()
                  if p.is_dir() and _has_rest_content(dataset_name, subject, p.name))


def _rest_dir(dataset_name: str, subject: str, session: str) -> Path:
    cfg = _surface_cfg(dataset_name)
    base = resolve(cfg["root"])
    return base / cfg["rest_glob"].format(subject=subject, session=session)


def load_rest_dtseries(
    dataset_name: str, subject: str, session: str, apply_tmask: bool = True
) -> np.ndarray:
    """Return the (T_kept, G) resting-state grayordinate timeseries for one session.

    Applies the per-frame ``tmask`` (drops censored frames) when present.
    """
    d = _rest_dir(dataset_name, subject, session)
    dts = sorted(d.glob("*.dtseries.nii"))
    if not dts:
        raise FileNotFoundError(f"no rest dtseries in {d}")
    data = _cifti.read_cifti(dts[0]).data  # (T, G)

    if apply_tmask:
        tm = sorted(d.glob("*_tmask.txt"))
        if tm:
            keep = np.loadtxt(tm[0]).astype(bool)
            if keep.shape[0] == data.shape[0]:
                data = data[keep]
    return data


def load_rest_concat(
    dataset_name: str, subject: str, sessions: list[str] | None = None, apply_tmask: bool = True
) -> np.ndarray:
    """Concatenate all sessions' rest grayordinate timeseries → (T_total, G)."""
    if sessions is None:
        sessions = rest_sessions(dataset_name, subject)
    blocks = [load_rest_dtseries(dataset_name, subject, s, apply_tmask) for s in sessions]
    return np.concatenate(blocks, axis=0)


def load_motor_contrasts(dataset_name: str, subject: str, smoothed: bool = False) -> "_cifti.Cifti":
    """Load the motor contrast dscalar (19 maps × G); ``smoothed`` picks the 2.55 mm file."""
    cfg = _surface_cfg(dataset_name)
    base = resolve(cfg["root"])
    p = base / cfg["motor_contrasts"].format(subject=subject)
    if smoothed:
        p = p.with_name(p.name.replace("_32k_fsLR.dscalar", "_32k_fsLR_smooth2.55.dscalar"))
    return _cifti.read_cifti(p)


def load_parcellation(dataset_name: str, subject: str) -> np.ndarray:
    """Per-(cortex-)grayordinate parcel id, shape (G_cortex,)."""
    cfg = _surface_cfg(dataset_name)
    base = resolve(cfg["root"])
    return _cifti.read_cifti(base / cfg["parcellation"].format(subject=subject)).data[0]


def load_parcel_networks(dataset_name: str, subject: str) -> np.ndarray:
    """Per-(cortex-)grayordinate Infomap network id, shape (G_cortex,)."""
    cfg = _surface_cfg(dataset_name)
    base = resolve(cfg["root"])
    return _cifti.read_cifti(base / cfg["parcel_networks"].format(subject=subject)).data[0]


def load_conditions(dataset_name: str, subject: str) -> dict[str, str]:
    """Return ``{session: condition}`` from the dataset's sessions.tsv (e.g. cast pre/cast/post)."""
    cfg = _dataset_cfg(dataset_name)
    tsv_tpl = cfg.get("sessions_tsv")
    if not tsv_tpl:
        return {}
    path = resolve(tsv_tpl.format(subject=subject))
    out: dict[str, str] = {}
    with open(path) as f:
        header = f.readline().strip().split("\t")
        si = header.index("SESSION") if "SESSION" in header else 0
        ci = header.index("CONDITION") if "CONDITION" in header else 1
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(si, ci):
                out[parts[si]] = parts[ci]
    return out


def sessions_for_condition(dataset_name: str, subject: str, condition: str) -> list[str]:
    """Rest sessions of a given condition that are present on disk (content fetched)."""
    cond = load_conditions(dataset_name, subject)
    present = set(rest_sessions(dataset_name, subject))
    return [s for s in sorted(cond) if cond.get(s) == condition and s in present]


def roi_source(dataset_name: str, subject: str) -> tuple[str, str] | None:
    """For cast1/cast2, the (dataset, subject) whose ROI masks to reuse (same brain)."""
    src = (_dataset_cfg(dataset_name).get("roi_source") or {}).get(subject)
    return (src["dataset"], src["subject"]) if src else None


def surface_dir(dataset_name: str, subject: str) -> Path:
    """fs_LR_32k surface directory (per-subject midthickness/inflated .surf.gii)."""
    cfg = _surface_cfg(dataset_name)
    base = resolve(cfg["root"])
    return base / cfg.get("surfaces", "{subject}/fs_LR_Talairach/fsaverage_LR32k").format(subject=subject)


def distance_matrix_path(dataset_name: str, subject: str) -> Path:
    """Path to the precomputed geodesic/euclidean distance .mat (loaded lazily; ~4 GB)."""
    cfg = _surface_cfg(dataset_name)
    base = resolve(cfg["root"])
    d = base / cfg.get("distances", "{subject}/cifti_distances").format(subject=subject)
    mats = sorted(Path(d).glob("*uint8.mat"))
    if not mats:
        raise FileNotFoundError(f"no distance matrix in {d}")
    return mats[0]
