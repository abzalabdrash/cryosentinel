"""Unit tests for Phase D.4 (augmentations) and D.5 (sampler).

CPU-only, no cloud runner / no live data.
"""
from __future__ import annotations

import pytest
import torch

from cryosentinel.data.augmentations import (
    CopyPaste,
    Mosaic,
    MosaicCopyPasteCollator,
    MultiScale,
    SpectralJitter,
)
from cryosentinel.data.sampler import HardNegativeWeightedSampler


# ─────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────
def _make_sample(*, water_block: tuple[int, int, int, int] | None = None,
                  H: int = 32, W: int = 32, seed: int = 0) -> dict:
    """Generate a single fake sample.

    ``water_block`` is ``(y0, y1, x0, x1)`` of the lake region (set to None
    for an empty / background chip).
    """
    g = torch.Generator().manual_seed(seed)
    s2 = torch.randn(12, H, W, generator=g)
    s1 = torch.randn(2,  H, W, generator=g)
    dem = torch.randn(1, H, W, generator=g)
    mask = torch.zeros(H, W, dtype=torch.long)
    if water_block is not None:
        y0, y1, x0, x1 = water_block
        mask[y0:y1, x0:x1] = 1
    valid = torch.ones(H, W, dtype=torch.bool)
    return {
        "S2L2A": s2, "S1GRD": s1, "DEM": dem,
        "mask": mask, "valid_mask": valid,
    }


# ─────────────────────────────────────────────────────────────────
# 1. SpectralJitter
# ─────────────────────────────────────────────────────────────────
class TestSpectralJitter:
    def test_p0_is_no_op(self):
        s = _make_sample()
        out = SpectralJitter(p=0.0)(s)
        assert torch.equal(out["S2L2A"], s["S2L2A"])

    def test_p1_perturbs_bands_independently(self):
        s = _make_sample()
        torch.manual_seed(0)
        out = SpectralJitter(p=1.0, eps_s2=0.05)(s)
        # Mean change per band should be non-zero
        diff = (out["S2L2A"] - s["S2L2A"]).abs()
        assert diff.max().item() > 0
        # Each band scaled by a single gain → identical relative change per pixel within a band
        per_band_ratio = (out["S2L2A"] / s["S2L2A"]).mean(dim=(1, 2))
        # Different gains for different bands (sample 12 → expect spread)
        assert per_band_ratio.std().item() > 1e-3

    def test_zero_eps_is_no_op_for_modality(self):
        s = _make_sample()
        out = SpectralJitter(p=1.0, eps_s2=0.05, eps_s1=0.0, eps_dem=0.0)(s)
        assert torch.equal(out["S1GRD"], s["S1GRD"])
        assert torch.equal(out["DEM"], s["DEM"])
        # S2 should differ
        assert not torch.equal(out["S2L2A"], s["S2L2A"])

    def test_does_not_mutate_input(self):
        s = _make_sample()
        s_orig = {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
        SpectralJitter(p=1.0)(s)
        for k in s_orig:
            assert torch.equal(s[k], s_orig[k]), f"Augmentation mutated input for {k}"


# ─────────────────────────────────────────────────────────────────
# 2. MultiScale
# ─────────────────────────────────────────────────────────────────
class TestMultiScale:
    def test_p0_is_no_op(self):
        s = _make_sample()
        out = MultiScale(p=0.0)(s)
        assert torch.equal(out["S2L2A"], s["S2L2A"])

    def test_preserves_shape(self):
        s = _make_sample(H=32, W=32, water_block=(8, 24, 8, 24))
        torch.manual_seed(42)
        out = MultiScale(p=1.0, scale_min=0.5, scale_max=2.0)(s)
        assert out["S2L2A"].shape == s["S2L2A"].shape
        assert out["S1GRD"].shape == s["S1GRD"].shape
        assert out["DEM"].shape == s["DEM"].shape
        assert out["mask"].shape == s["mask"].shape
        assert out["valid_mask"].shape == s["valid_mask"].shape

    def test_mask_stays_binary(self):
        s = _make_sample(water_block=(8, 24, 8, 24))
        torch.manual_seed(7)
        out = MultiScale(p=1.0, scale_min=1.5, scale_max=1.5)(s)
        # Nearest-neighbor → values must remain in {0, 1}
        unique = torch.unique(out["mask"])
        assert all(int(v.item()) in (0, 1) for v in unique)

    def test_scale_up_then_crop_keeps_some_lake(self):
        s = _make_sample(water_block=(0, 32, 0, 32))  # full lake
        torch.manual_seed(0)
        out = MultiScale(p=1.0, scale_min=1.5, scale_max=1.5)(s)
        # Crop of an enlarged full-lake chip is still full-lake
        assert int(out["mask"].sum()) == s["mask"].numel()


# ─────────────────────────────────────────────────────────────────
# 3. CopyPaste
# ─────────────────────────────────────────────────────────────────
class TestCopyPaste:
    def _batch(self, n: int, water_indices: list[int] | None = None,
                H: int = 16, W: int = 16) -> list[dict]:
        water_indices = water_indices or []
        return [
            _make_sample(
                water_block=(2, 14, 2, 14) if i in water_indices else None,
                H=H, W=W, seed=i,
            )
            for i in range(n)
        ]

    def test_p0_is_no_op(self):
        batch = self._batch(4, water_indices=[0])
        out = CopyPaste(p=0.0)(batch)
        for a, b in zip(out, batch):
            assert torch.equal(a["mask"], b["mask"])

    def test_pastes_lake_into_empty(self):
        batch = self._batch(4, water_indices=[0])  # only sample 0 has water
        # Force CopyPaste with p=1 — every empty target should gain water.
        torch.manual_seed(0)
        out = CopyPaste(p=1.0, min_donor_water_pixels=10, sub_sample_frac=1.0)(batch)
        for i in (1, 2, 3):
            assert int(out[i]["mask"].sum()) >= int(batch[i]["mask"].sum())
            # In particular, every empty chip should now have *some* lake
            assert int(out[i]["mask"].sum()) > 0

    def test_too_few_donors_is_no_op(self):
        # No water anywhere → no donors → should not crash, no change
        batch = self._batch(4, water_indices=[])
        out = CopyPaste(p=1.0, min_donor_water_pixels=10)(batch)
        for a, b in zip(out, batch):
            assert torch.equal(a["mask"], b["mask"])

    def test_modalities_consistent_after_paste(self):
        batch = self._batch(2, water_indices=[0])
        torch.manual_seed(0)
        out = CopyPaste(p=1.0, min_donor_water_pixels=10)(batch)
        # The pasted region in target should match donor's modality values
        target = out[1]
        donor = batch[0]
        copy_region = donor["mask"] > 0
        assert torch.equal(
            target["S2L2A"][:, copy_region],
            donor["S2L2A"][:, copy_region],
        )


# ─────────────────────────────────────────────────────────────────
# 4. Mosaic
# ─────────────────────────────────────────────────────────────────
class TestMosaic:
    def test_passthrough_when_batch_too_small(self):
        batch = [_make_sample(H=8, W=8, seed=i) for i in range(2)]
        out = Mosaic(p=1.0)(batch)
        for a, b in zip(out, batch):
            assert torch.equal(a["S2L2A"], b["S2L2A"])

    def test_preserves_shape(self):
        batch = [_make_sample(H=16, W=16, seed=i) for i in range(8)]
        torch.manual_seed(0)
        out = Mosaic(p=1.0, jitter=0.2)(batch)
        for a, b in zip(out, batch):
            assert a["S2L2A"].shape == b["S2L2A"].shape
            assert a["mask"].shape == b["mask"].shape
            assert a["valid_mask"].shape == b["valid_mask"].shape

    def test_p0_is_no_op(self):
        batch = [_make_sample(H=16, W=16, seed=i) for i in range(8)]
        out = Mosaic(p=0.0)(batch)
        for a, b in zip(out, batch):
            assert torch.equal(a["S2L2A"], b["S2L2A"])


# ─────────────────────────────────────────────────────────────────
# 5. MosaicCopyPasteCollator
# ─────────────────────────────────────────────────────────────────
class TestMosaicCopyPasteCollator:
    def test_collation_works_with_both_off(self):
        batch = [_make_sample(H=8, W=8, seed=i) for i in range(2)]
        coll = MosaicCopyPasteCollator()(batch)
        assert coll["S2L2A"].shape == (2, 12, 8, 8)
        assert coll["mask"].shape == (2, 8, 8)

    def test_collation_works_with_both_on(self):
        batch = [_make_sample(
            water_block=(2, 6, 2, 6) if i % 2 == 0 else None,
            H=8, W=8, seed=i,
        ) for i in range(8)]
        torch.manual_seed(0)
        collator = MosaicCopyPasteCollator(
            mosaic=Mosaic(p=0.5, jitter=0.2),
            copy_paste=CopyPaste(p=0.5, min_donor_water_pixels=4),
        )
        out = collator(batch)
        assert out["S2L2A"].shape == (8, 12, 8, 8)
        assert out["mask"].shape == (8, 8, 8)


# ─────────────────────────────────────────────────────────────────
# 6. HardNegativeWeightedSampler
# ─────────────────────────────────────────────────────────────────
class TestHardNegativeSampler:
    def test_balanced_distribution(self):
        # 90 % positives, 10 % negatives in dataset
        is_pos = [True] * 900 + [False] * 100
        torch.manual_seed(0)
        sampler = HardNegativeWeightedSampler(
            is_pos, positive_to_negative_ratio=3.0,
            num_samples=10_000, generator=torch.Generator().manual_seed(0),
        )
        idx = list(sampler)
        n_pos = sum(1 for i in idx if is_pos[i])
        n_neg = len(idx) - n_pos
        # Should sample ~75 % positives at ratio 3:1
        frac_pos = n_pos / len(idx)
        assert 0.70 <= frac_pos <= 0.80

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            HardNegativeWeightedSampler([True, False], positive_to_negative_ratio=0.0)
        with pytest.raises(ValueError):
            HardNegativeWeightedSampler([True, False], positive_to_negative_ratio=-1.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            HardNegativeWeightedSampler([])

    def test_single_class_falls_back_to_uniform(self):
        # All positive — sampler should still work, give uniform weights.
        sampler = HardNegativeWeightedSampler([True] * 100,
                                              positive_to_negative_ratio=3.0)
        idx = list(sampler)
        assert len(idx) == 100
        assert sampler.n_positive == 100
        assert sampler.n_negative == 0

    def test_length(self):
        sampler = HardNegativeWeightedSampler(
            [True, False, True], num_samples=42,
        )
        assert len(sampler) == 42
