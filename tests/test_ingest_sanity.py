"""Unit tests for v3 ingest sanity helpers (Phase A1 / A2 logic).

These tests are intentionally pure-numpy / pure-python — no Earth Engine,
no network, no rasterio — so they run on any developer machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from cryosentinel.ingest.sanity import (
    KUMAR_AVAILABLE_YEARS,
    MNDWI_THRESHOLD,
    SANITY_BG_WATER_FRAC_MAX,
    SANITY_FP_RATIO_MAX,
    SANITY_IOU_MIN,
    S2_IDX_GREEN,
    S2_IDX_SWIR1,
    compute_mndwi,
    kumar_year_for_snapshot,
    sanity_check_chip,
)


# ─────────────────────────────────────────────────────────────────
# 1. Snapshot-year mapping
# ─────────────────────────────────────────────────────────────────
class TestKumarYearForSnapshot:
    def test_below_midpoint_maps_to_2016(self):
        # midpoint = (2016+2022)/2 = 2019
        for y in (2016, 2017, 2018):
            assert kumar_year_for_snapshot(y) == 2016

    def test_at_or_above_midpoint_maps_to_2022(self):
        for y in (2019, 2020, 2021, 2022, 2023, 2024):
            assert kumar_year_for_snapshot(y) == 2022

    def test_returns_only_kumar_years(self):
        for y in range(2014, 2030):
            assert kumar_year_for_snapshot(y) in KUMAR_AVAILABLE_YEARS


# ─────────────────────────────────────────────────────────────────
# 2. MNDWI formula
# ─────────────────────────────────────────────────────────────────
class TestComputeMNDWI:
    def _make_chip(self, green: int, swir1: int, shape=(4, 4)) -> np.ndarray:
        """Build a 12-channel S2 chip with constant green / swir1 values."""
        chip = np.zeros((12, *shape), dtype=np.uint16)
        chip[S2_IDX_GREEN] = green
        chip[S2_IDX_SWIR1] = swir1
        return chip

    def test_pure_water_mndwi_is_one(self):
        # Pure water: high green, ~zero SWIR1 → MNDWI ≈ +1
        chip = self._make_chip(green=2000, swir1=0)
        mndwi = compute_mndwi(chip)
        np.testing.assert_allclose(mndwi, 1.0)

    def test_pure_snow_mndwi_is_negative(self):
        # Snow: high green AND high SWIR1 (snow is bright in SWIR) → MNDWI < 0 typically
        chip = self._make_chip(green=4000, swir1=6000)
        mndwi = compute_mndwi(chip)
        # (4000-6000)/(4000+6000) = -0.2
        np.testing.assert_allclose(mndwi, -0.2, atol=1e-6)

    def test_balanced_bands_mndwi_is_zero(self):
        chip = self._make_chip(green=1500, swir1=1500)
        mndwi = compute_mndwi(chip)
        np.testing.assert_allclose(mndwi, 0.0)

    def test_nodata_returns_zero(self):
        chip = self._make_chip(green=0, swir1=0)
        mndwi = compute_mndwi(chip)
        np.testing.assert_allclose(mndwi, 0.0)

    def test_threshold_separates_water_from_snow(self):
        """A chip with mixed water + snow pixels gets a clean threshold split."""
        chip = np.zeros((12, 1, 4), dtype=np.uint16)
        # 2 water-like pixels, 2 snow-like
        chip[S2_IDX_GREEN, 0] = [2000, 2000, 4000, 4000]
        chip[S2_IDX_SWIR1, 0] = [   0,  100, 5000, 6000]
        mndwi = compute_mndwi(chip)
        water_mask = mndwi > MNDWI_THRESHOLD
        # First two pixels should be water; last two should not.
        assert water_mask[0, 0] and water_mask[0, 1]
        assert not water_mask[0, 2] and not water_mask[0, 3]

    def test_invalid_shape_raises(self):
        bad = np.zeros((4, 4), dtype=np.uint16)  # missing channel dim
        with pytest.raises(ValueError):
            compute_mndwi(bad)


# ─────────────────────────────────────────────────────────────────
# 3. Sanity-check decisions
# ─────────────────────────────────────────────────────────────────
class TestSanityCheckChip:
    H = W = 32

    def _water_chip(self, water_pixels: int) -> np.ndarray:
        """Construct an S2 chip with `water_pixels` MNDWI-water pixels (rest dry)."""
        chip = np.zeros((12, self.H, self.W), dtype=np.uint16)
        # Default: dry. snow-ish (green=swir → MNDWI=0)
        chip[S2_IDX_GREEN] = 1500
        chip[S2_IDX_SWIR1] = 1500
        # Set first `water_pixels` to high MNDWI (green=2000, swir=0)
        flat_g = chip[S2_IDX_GREEN].reshape(-1)
        flat_s = chip[S2_IDX_SWIR1].reshape(-1)
        flat_g[:water_pixels] = 2000
        flat_s[:water_pixels] = 0
        return chip

    def _mask(self, lake_pixels: int) -> np.ndarray:
        m = np.zeros((self.H, self.W), dtype=np.uint8)
        m.reshape(-1)[:lake_pixels] = 1
        return m

    # ── Regime A: positive chips ─────────────────────────────────
    def test_positive_chip_perfect_agreement_kept(self):
        # Mask: 100 lake pixels at the start. MNDWI: same 100 pixels are water.
        chip = self._water_chip(water_pixels=100)
        mask = self._mask(lake_pixels=100)
        keep, info = sanity_check_chip(chip, mask)
        assert keep, info
        np.testing.assert_allclose(info["iou"], 1.0)
        np.testing.assert_allclose(info["fp_ratio"], 1.0)

    def test_positive_chip_total_disagreement_dropped(self):
        # Mask says 100 pixels lake, but MNDWI shows ZERO water → IoU=0.
        chip = self._water_chip(water_pixels=0)
        mask = self._mask(lake_pixels=100)
        keep, info = sanity_check_chip(chip, mask)
        assert not keep
        assert info["iou"] == 0.0

    def test_positive_chip_below_iou_threshold_dropped(self):
        # Mask: pixels [0..100). MNDWI water: pixels [50..150).
        # Intersection = 50, union = 150 → IoU ≈ 0.33 → above default 0.2 ⇒ kept.
        # We tighten threshold to force a drop.
        chip = np.zeros((12, self.H, self.W), dtype=np.uint16)
        chip[S2_IDX_GREEN] = 1500
        chip[S2_IDX_SWIR1] = 1500
        flat_g = chip[S2_IDX_GREEN].reshape(-1)
        flat_s = chip[S2_IDX_SWIR1].reshape(-1)
        # water in [50, 150)
        flat_g[50:150] = 2000
        flat_s[50:150] = 0

        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        mask.reshape(-1)[:100] = 1   # mask in [0, 100)

        keep, info = sanity_check_chip(chip, mask, iou_min=0.5)
        assert not keep
        # 50 / 150 ≈ 0.333
        assert 0.30 < info["iou"] < 0.36

    def test_positive_chip_high_fp_ratio_dropped(self):
        # Mask says only 10 pixels lake. MNDWI sees 100. fp_ratio = 10 → drop.
        chip = self._water_chip(water_pixels=100)
        mask = self._mask(lake_pixels=10)
        keep, info = sanity_check_chip(chip, mask, fp_ratio_max=3.0)
        assert not keep
        assert info["fp_ratio"] == 10.0

    def test_positive_chip_partial_overlap_passes_default(self):
        # Mask: pixels [0..200). MNDWI water: [100..300). IoU = 100/300 = 0.333.
        chip = np.zeros((12, self.H, self.W), dtype=np.uint16)
        chip[S2_IDX_GREEN] = 1500
        chip[S2_IDX_SWIR1] = 1500
        chip[S2_IDX_GREEN].reshape(-1)[100:300] = 2000
        chip[S2_IDX_SWIR1].reshape(-1)[100:300] = 0
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        mask.reshape(-1)[:200] = 1
        keep, info = sanity_check_chip(chip, mask)  # default iou_min=0.2, fp=3.0
        # IoU 0.333 ≥ 0.2 ✓  and fp_ratio = 200/200 = 1.0 ≤ 3.0 ✓ → keep
        assert keep, info

    # ── Regime B: background chips ───────────────────────────────
    def test_background_chip_dry_kept(self):
        # No mask, no MNDWI water → BG accepted.
        chip = self._water_chip(water_pixels=0)
        mask = self._mask(lake_pixels=0)
        keep, info = sanity_check_chip(chip, mask)
        assert keep
        assert info["bg_water_frac"] == 0.0

    def test_background_chip_with_unlabeled_water_dropped(self):
        # No mask, but MNDWI sees water everywhere → drop (Kumar missed lakes).
        # 200 water pixels out of 1024 (32×32) = 19.5% > 5% threshold → drop.
        chip = self._water_chip(water_pixels=200)
        mask = self._mask(lake_pixels=0)
        keep, info = sanity_check_chip(chip, mask)
        assert not keep
        assert info["bg_water_frac"] > 0.05

    def test_background_chip_minor_water_kept(self):
        # 5 MNDWI-water pixels out of 1024 = 0.49% ≤ 5% → keep.
        chip = self._water_chip(water_pixels=5)
        mask = self._mask(lake_pixels=0)
        keep, info = sanity_check_chip(chip, mask)
        assert keep
        assert info["bg_water_frac"] < 0.05

    # ── Edge cases ───────────────────────────────────────────────
    def test_invalid_mask_shape_raises(self):
        chip = self._water_chip(water_pixels=0)
        bad_mask = np.zeros((self.H,), dtype=np.uint8)  # 1-D
        with pytest.raises(ValueError):
            sanity_check_chip(chip, bad_mask)

    def test_dtype_robustness_uint16_vs_float32(self):
        chip16 = self._water_chip(water_pixels=100)
        chip32 = chip16.astype(np.float32)
        mask = self._mask(lake_pixels=100)
        keep1, info1 = sanity_check_chip(chip16, mask)
        keep2, info2 = sanity_check_chip(chip32, mask)
        assert keep1 == keep2
        assert info1["iou"] == pytest.approx(info2["iou"], abs=1e-6)
