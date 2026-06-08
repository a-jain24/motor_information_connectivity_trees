"""cifti.py — CIFTI reading (CIFTI-1 and CIFTI-2) + Connectome Workbench wrappers.

The WU/Gordon ``surface_pipeline`` files are **CIFTI-1** (legacy Workbench format).
nibabel only parses CIFTI-2, and reading the CIFTI-1 NIfTI-2 blob raw mis-orders the
data (the CIFTI-1 vs CIFTI-2 on-disk matrix layout differs). The robust, verified
approach is therefore:

    read_cifti(path):
        try nibabel.load  ──(CIFTI-2)──▶ use directly
        on "version 1"   ──▶ wb_command -file-convert -cifti-version-convert .. 2 (temp)
                             ──▶ nibabel.load the temp ──▶ correct (n_series, G) + axes

Validated against ground truth: converted MSC rest data shows temporal lag-1
autocorr ≈ 0.82 (real BOLD) and the expected cortex/cerebellum/thalamus structures.

Data is returned oriented ``(n_series, G)`` — timepoints/maps along axis 0,
grayordinates along axis 1.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np

CORTEX_LEFT = "CIFTI_STRUCTURE_CORTEX_LEFT"
CORTEX_RIGHT = "CIFTI_STRUCTURE_CORTEX_RIGHT"
CEREBELLUM_LEFT = "CIFTI_STRUCTURE_CEREBELLUM_LEFT"
CEREBELLUM_RIGHT = "CIFTI_STRUCTURE_CEREBELLUM_RIGHT"
THALAMUS_LEFT = "CIFTI_STRUCTURE_THALAMUS_LEFT"
THALAMUS_RIGHT = "CIFTI_STRUCTURE_THALAMUS_RIGHT"


@dataclass
class BrainModel:
    name: str
    offset: int
    count: int
    model_type: str  # SURFACE | VOXELS
    vertices: np.ndarray | None = None  # surface: fs_LR vertex id per grayordinate column
    voxels: np.ndarray | None = None    # volume: (n, 3) IJK per grayordinate column

    @property
    def columns(self) -> np.ndarray:
        return np.arange(self.offset, self.offset + self.count)


@dataclass
class Cifti:
    data: np.ndarray   # (n_series, G)
    models: list       # list[BrainModel] in grayordinate order
    map_names: list    # per-series names (dscalar/dlabel) or [] (dtseries)
    affine: np.ndarray | None = None  # voxel(IJK)→mm affine for the subcortical volume

    @property
    def n_grayordinates(self) -> int:
        return self.data.shape[1]

    def model(self, name: str) -> BrainModel:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"structure {name!r} not in CIFTI (have {[m.name for m in self.models]})")

    def columns(self, *names: str) -> np.ndarray:
        return np.concatenate([self.model(n).columns for n in names])

    def cortex_columns(self) -> np.ndarray:
        return self.columns(CORTEX_LEFT, CORTEX_RIGHT)

    def map_index(self, name: str) -> int:
        return self.map_names.index(name)


# ───────────────────────── Connectome Workbench ─────────────────────────

_WB = os.environ.get("WB_COMMAND", "wb_command")


def set_wb_command(path: str) -> None:
    global _WB
    _WB = path


def _resolve_wb() -> str:
    """Resolve wb_command: explicit > PATH > config/datasets.yaml tools.wb_command."""
    if _WB and ("/" in _WB) and os.path.exists(_WB):
        return _WB
    if shutil.which(_WB):
        return _WB
    try:
        from .paths import datasets_config
        cfg = (datasets_config().get("tools") or {}).get("wb_command")
        if cfg and os.path.exists(cfg):
            return cfg
    except Exception:
        pass
    raise RuntimeError(
        "wb_command not found — set WB_COMMAND, set_wb_command(<path>), "
        "or add tools.wb_command to config/datasets.yaml."
    )


def run_wb(*args) -> str:
    res = subprocess.run([_resolve_wb(), *map(str, args)], check=True,
                         capture_output=True, text=True)
    return res.stdout


# ───────────────────────── Reading ─────────────────────────

def _models_from_nibabel(img) -> list[BrainModel]:
    bm = img.header.get_axis(img.ndim - 1)  # BrainModelAxis is the last axis
    out = []
    off = 0
    for name, sl, sub in bm.iter_structures():
        start = int(sl.start) if sl.start is not None else off
        count = len(sub)  # robust: slice bounds can be open (None)
        is_surf = "CORTEX" in str(name)
        out.append(BrainModel(
            name=str(name), offset=start, count=count,
            model_type="SURFACE" if is_surf else "VOXELS",
            vertices=np.asarray(sub.vertex) if is_surf else None,
            voxels=np.asarray(sub.voxel) if not is_surf else None,
        ))
        off = start + count
    return out


def _affine_from_nibabel(img):
    bm = img.header.get_axis(img.ndim - 1)
    return np.asarray(bm.affine) if getattr(bm, "affine", None) is not None else None


def _map_names_from_nibabel(img) -> list[str]:
    ax0 = img.header.get_axis(0)
    return list(ax0.name) if hasattr(ax0, "name") else []


def read_cifti(path) -> Cifti:
    """Read a CIFTI-1/2 file → ``Cifti`` (data ``(n_series, G)``, correct orientation)."""
    import nibabel as nib

    path = str(path)
    try:
        img = nib.load(path)
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        return Cifti(data, _models_from_nibabel(img), _map_names_from_nibabel(img), _affine_from_nibabel(img))
    except ValueError as e:
        if "version 1" not in str(e):
            raise
    # CIFTI-1 → convert to a temp CIFTI-2 and read authoritatively.
    fd, tmp = tempfile.mkstemp(suffix=".nii")
    os.close(fd)
    try:
        run_wb("-file-convert", "-cifti-version-convert", path, "2", tmp)
        img = nib.load(tmp)
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        return Cifti(data, _models_from_nibabel(img), _map_names_from_nibabel(img), _affine_from_nibabel(img))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def cifti_smoothing(in_dtseries, surf_sigma_mm, vol_sigma_mm, left_surf, right_surf, out_path):
    """Geodesic surface + euclidean volume smoothing via ``wb_command -cifti-smoothing``."""
    return run_wb("-cifti-smoothing", in_dtseries, surf_sigma_mm, vol_sigma_mm, "COLUMN",
                  out_path, "-left-surface", left_surf, "-right-surface", right_surf)
