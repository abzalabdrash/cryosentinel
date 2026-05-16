"""Pure-Python helpers for v3 ingest sanity checks (Phase A1 / A2).

Kept in this dedicated module instead of inline in an ingestion runner
so the logic is unit-testable without an Earth Engine / GEE dependency.

Functions
---------
- ``kumar_year_for_snapshot(snapshot_year)`` — pick closest Kumar inventory year
- ``compute_mndwi(s2_chip)``                  — MNDWI from a 12-band S2 chip
- ``sanity_check_chip(s2_chip, mask_chip,…)`` — drop-decision under label drift
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────
# Constants — must match TerraMind S2 12-band layout exactly
# ─────────────────────────────────────────────────────────────────
# S2_BANDS_TERRAMIND = [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12]
S2_IDX_GREEN: int = 2   # B3   ~560 nm
S2_IDX_NIR:   int = 8   # B8A  ~865 nm  (narrow NIR)
S2_IDX_SWIR1: int = 10  # B11  ~1610 nm

# v3 sanity-filter defaults — tuned for HMA / Tien Shan / Karakoram
MNDWI_THRESHOLD: float = 0.3
SANITY_IOU_MIN: float = 0.2
SANITY_FP_RATIO_MAX: float = 3.0
SANITY_BG_WATER_FRAC_MAX: float = 0.05

# Kumar inventory years available locally (Glacial_Lake_2016/2022.shp).
KUMAR_AVAILABLE_YEARS: tuple[int, int] = (2016, 2022)


# ─────────────────────────────────────────────────────────────────
# 1. Snapshot-year → Kumar-year mapping
# ─────────────────────────────────────────────────────────────────
def kumar_year_for_snapshot(snapshot_year: int) -> int:
    """Return the closest Kumar inventory year for a Sentinel snapshot year.

    The choice is based on a midpoint split: snapshot years strictly below
    the midpoint (2019) map to Kumar 2016; years at or above map to 2022.
    """
    mid = (KUMAR_AVAILABLE_YEARS[0] + KUMAR_AVAILABLE_YEARS[1]) / 2.0
    return KUMAR_AVAILABLE_YEARS[0] if snapshot_year < mid else KUMAR_AVAILABLE_YEARS[1]


# ─────────────────────────────────────────────────────────────────
# 2. MNDWI (Modified NDWI, Xu 2006) — better than NDWI in glacial regions
#    because SWIR1 distinguishes water (low) from snow (high).
# ─────────────────────────────────────────────────────────────────
def compute_mndwi(
    s2_chip: np.ndarray,
    *,
    green_idx: int = S2_IDX_GREEN,
    swir1_idx: int = S2_IDX_SWIR1,
) -> np.ndarray:
    """MNDWI = (Green - SWIR1) / (Green + SWIR1).

    Parameters
    ----------
    s2_chip : np.ndarray, shape (C, H, W) where C >= max(green_idx, swir1_idx)+1
        Reflectance values (any positive scale; ratio cancels constant scale).
        Pixels where both bands are 0 are treated as nodata and yield 0.

    Returns
    -------
    mndwi : np.ndarray, shape (H, W), dtype float32, range ~[-1, 1]
    """
    if s2_chip.ndim != 3:
        raise ValueError(f"s2_chip must have shape (C, H, W); got {s2_chip.shape}")
    green = s2_chip[green_idx].astype(np.float32)
    swir1 = s2_chip[swir1_idx].astype(np.float32)
    denom = green + swir1
    mndwi = np.zeros_like(green, dtype=np.float32)
    safe = denom > 0
    if np.any(safe):
        mndwi[safe] = (green[safe] - swir1[safe]) / denom[safe]
    return mndwi


# ─────────────────────────────────────────────────────────────────
# 3. Sanity filter — drop chips where Kumar mask & MNDWI strongly disagree
# ─────────────────────────────────────────────────────────────────
def sanity_check_chip(
    s2_chip: np.ndarray,
    mask_chip: np.ndarray,
    *,
    mndwi_threshold: float = MNDWI_THRESHOLD,
    iou_min: float = SANITY_IOU_MIN,
    fp_ratio_max: float = SANITY_FP_RATIO_MAX,
    bg_water_frac_max: float = SANITY_BG_WATER_FRAC_MAX,
) -> tuple[bool, dict]:
    """Two-regime sanity check for chips assembled with stale Kumar labels.

    Regime A — chip has Kumar lakes (mask.sum() > 0):
        keep ⇔ IoU(mask, mndwi_water) ≥ ``iou_min``
                AND mndwi_water_count / mask_count ≤ ``fp_ratio_max``

    Regime B — chip has no Kumar lakes (mask.sum() == 0):
        keep ⇔ MNDWI water fraction ≤ ``bg_water_frac_max``

    Parameters
    ----------
    s2_chip : (C, H, W) array of S2 reflectance (≥ 11 channels in TerraMind layout)
    mask_chip : (H, W) array, nonzero = lake

    Returns
    -------
    keep : bool
    info : dict with keys ``n_mask``, ``n_mndwi_water``, ``iou``, ``fp_ratio``,
           ``bg_water_frac`` for downstream logging / per-chip Parquet metadata.
    """
    if mask_chip.ndim != 2:
        raise ValueError(f"mask_chip must be 2-D; got {mask_chip.shape}")

    mndwi = compute_mndwi(s2_chip)
    water = mndwi > mndwi_threshold
    mask_bool = mask_chip > 0
    n_mask = int(mask_bool.sum())
    n_water = int(water.sum())
    info = {
        "n_mask": n_mask,
        "n_mndwi_water": n_water,
        "iou": 0.0,
        "fp_ratio": 0.0,
        "bg_water_frac": 0.0,
    }

    if n_mask == 0:
        info["bg_water_frac"] = float(water.mean())
        return info["bg_water_frac"] <= bg_water_frac_max, info

    inter = int(np.logical_and(mask_bool, water).sum())
    union = int(np.logical_or(mask_bool, water).sum())
    iou = inter / max(union, 1)
    fp_ratio = n_water / max(n_mask, 1)
    info["iou"] = float(iou)
    info["fp_ratio"] = float(fp_ratio)
    keep = (iou >= iou_min) and (fp_ratio <= fp_ratio_max)
    return keep, info


__all__: Sequence[str] = (
    "S2_IDX_GREEN",
    "S2_IDX_NIR",
    "S2_IDX_SWIR1",
    "MNDWI_THRESHOLD",
    "SANITY_IOU_MIN",
    "SANITY_FP_RATIO_MAX",
    "SANITY_BG_WATER_FRAC_MAX",
    "KUMAR_AVAILABLE_YEARS",
    "kumar_year_for_snapshot",
    "compute_mndwi",
    "sanity_check_chip",
)
