"""paths.py — repo-root, config, and output-path resolution (config-driven).

Replaces the old ``msc_paths.py`` ``_CANDIDATES`` machine-probe + hard-coded MSC
layout. The repo root is derived from this file's location (works for an editable
install), and dataset/atlas locations come from ``config/*.yaml`` — not constants.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

# src/mict/paths.py → parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
ATLAS_DIR = REPO_ROOT / "atlases"


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """Load a YAML config file from config/ (``name`` with or without .yaml)."""
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def datasets_config() -> dict:
    return load_config("datasets")


def atlases_config() -> dict:
    return load_config("atlases")


def roi_scheme_config() -> dict:
    return load_config("roi_scheme")


def dataset(name: str) -> dict:
    """Return the config block for one dataset (MSC / cast / perinatal_stroke)."""
    cfg = datasets_config()["datasets"]
    if name not in cfg:
        raise KeyError(f"unknown dataset {name!r}; known: {list(cfg)}")
    return cfg[name]


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config path: absolute as-is, else relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def results_path(
    dataset_name: str,
    space: str,
    measure: str,
    subject: str | None = None,
    session: str = "all_sessions",
    create: bool = True,
) -> Path:
    """results/<dataset>/<space>/<measure>/[<subject>/<session>/]

    ``measure`` is e.g. ``fc`` / ``mi`` / ``cl`` / ``roi_masks``.
    """
    d = RESULTS_DIR / dataset_name / space / measure
    if subject is not None:
        d = d / subject / session
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def detect_device(requested: str | None = None):
    """Best available torch device (CUDA → MPS → CPU). Returns 'cpu' if torch absent."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
