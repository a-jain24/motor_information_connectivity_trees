"""rois.py — motor ROI scheme + atlas index constants.

Two layers:
  1. The dataset/space-independent ROI *scheme* (labels, regions, methods) read
     from ``config/roi_scheme.yaml`` — the 18-ROI target of motor_system_plan.md.
  2. Glasser360 / SUIT / Morel index constants ported from the old
     ``canonical_utils.py`` (used by the volumetric path and as a cross-reference;
     the surface path parcellates via the fs_LR_32k Glasser dlabel instead).

ROI labels are identical across datasets and spaces so CL trees / FC matrices are
directly comparable.
"""

from __future__ import annotations

from .paths import roi_scheme_config

# ───────────────────────── ROI scheme (config-driven) ─────────────────────────

def roi_scheme() -> list[dict]:
    """The ordered list of ROI dicts (label, region, effector, hemi, space, method)."""
    return roi_scheme_config()["rois"]


def roi_labels() -> list[str]:
    return [r["label"] for r in roi_scheme()]


def roi_regions() -> dict[str, str]:
    return {r["label"]: r["region"] for r in roi_scheme()}


def labels_by_space(space: str) -> list[str]:
    """ROI labels whose definition lives in a given space ('surface' | 'volume')."""
    return [r["label"] for r in roi_scheme() if r["space"] == space]


def legacy_alias_map() -> dict[str, str]:
    """Map any legacy alias (e.g. L_M1_tongue) → current label (L_M1_face)."""
    out = {}
    for r in roi_scheme():
        if "alias" in r:
            out[r["alias"]] = r["label"]
    return out


# Region → color (for trees / matrices / dendrograms). Region names match
# roi_scheme.yaml `region` values.
REGION_COLORS = {
    "M1":    "#e41a1c",
    "SMA":   "#377eb8",
    "PMd":   "#4daf4a",
    "PMv":   "#984ea3",
    "Thal":  "#ff7f00",
    "Cereb": "#a65628",
}


def region_color(label: str) -> str:
    return REGION_COLORS.get(roi_regions().get(label, ""), "#cccccc")


def roi_palette() -> dict[str, tuple]:
    """A fixed, distinct colour per ROI label (stable across subjects) for aesthetic trees."""
    import matplotlib.pyplot as plt
    labels = roi_labels()
    cmap = plt.get_cmap("tab20", max(len(labels), 20))
    return {lab: cmap(i) for i, lab in enumerate(labels)}


# ──────────────── Glasser360 / SUIT / Morel index constants ────────────────
# 0-based indices into the combined glasser360(+SUIT) atlas array.
# Right hemisphere: indices 0–179 | Left hemisphere: 180–359  (NIfTI label = idx + 1)
# Ported verbatim from canonical_circuits/canonical_utils.py.

PRIMARY_MOTOR_IDX = [7, 187]                 # Area 4  (R, L)
PRIMARY_SENS_IDX  = [8, 188, 52, 232,        # 3b, 3a
                     50, 230, 51, 231]       # 1, 2
SMA_IDX           = [54, 234]                # 6mp — SMA proper
PRE_SMA_IDX       = [43, 223]                # 6ma — pre-SMA
SCEF_IDX          = [42, 222]
PARACENTRAL_IDX   = [35, 215, 38, 218]       # 5m, 5L
PMD_IDX           = [53, 233, 77, 257, 95, 275]   # 6d, 6r, 6a
PMV_IDX           = [55, 235, 11, 191]            # 6v, 55b

MOTOR_CORTEX_ALL = sorted(set(
    PRIMARY_MOTOR_IDX + PRIMARY_SENS_IDX + SMA_IDX + PRE_SMA_IDX
    + SCEF_IDX + PARACENTRAL_IDX + PMD_IDX + PMV_IDX
))

# SUIT cerebellar parcels (offset 360 in the combined atlas)
CEREB_OFFSET      = 360
CEREB_MOTOR_LOCAL = [0, 1, 5, 6, 20, 21]     # lobules IV/V, VI, VIII (0-based within 34)
CEREB_MOTOR_IDX   = [CEREB_OFFSET + i for i in CEREB_MOTOR_LOCAL]

# Morel motor-relay thalamic nuclei (FC-seed labeling target; §D)
MOTOR_THALAMIC_NUCLEI = ["VLa", "VLpd", "VLpv", "VAmc", "VApc"]
