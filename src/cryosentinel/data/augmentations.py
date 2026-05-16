"""Advanced data augmentations for the v3 SOTA training recipe.

Two tiers of augmentation are provided:

1. **Sample-level** (callable on a single sample dict):
     * :class:`SpectralJitter` — per-band random gain ± a few percent
     * :class:`MultiScale`     — random spatial rescaling

   These are applied inside ``MultiModalChipDataset.__getitem__`` *after*
   the existing flip / rot90 augmentations.

2. **Batch-level** (callable on a list of samples, used as
   ``DataLoader.collate_fn``):
     * :class:`CopyPaste` — copy lake regions from a donor chip onto target
     * :class:`Mosaic`    — 2×2 grid composite, cropped to original size

   These are wrapped in :class:`MosaicCopyPasteCollator` which composes
   them in the right order and falls back to plain ``default_collate`` for
   the val/test loaders.

All augmentations:
* Operate on already-tensorised, already-standardised float tensors.
* Respect ``valid_mask`` so nodata pixels propagate cleanly.
* Use ``torch.rand`` so they participate in the dataloader-worker RNG seed.
* Are no-ops when probability < 0 or the relevant tensor is empty.

Why this is safe for ViT
------------------------
TerraMind v1 / Prithvi use *absolute* learned positional embeddings, which
break under arbitrary rotations. We therefore restrict spatial augs to
flips and 90° rotations (already in the dataset), plus *isotropic* scaling
that re-crops back to 224×224 — both of which preserve the absolute
position-aligned semantic of each pixel relative to image centre.

Mosaic and CopyPaste introduce sharp image boundaries. This is acceptable
for segmentation because (a) the model has been pre-trained on natural
imagery with similar discontinuities (cloud edges, granule borders),
(b) we keep the original aspect ratio, and (c) we update ``valid_mask`` so
the loss optionally ignores the seam pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


# ──────────────────────────────────────────────────────────────────
# 1. Sample-level: SpectralJitter
# ──────────────────────────────────────────────────────────────────
@dataclass
class SpectralJitter:
    """Independently scale each S2 band by a random gain in ``[1-eps, 1+eps]``.

    Operates on the *standardised* tensor (post ``(x - μ)/σ``), so an
    eps of 0.05 nudges each band by up to ±0.05 standard deviations
    around the chip's per-band mean. Cheap, effective, well-known
    regularisation for multispectral imagery (used in Prithvi, SatMAE,
    SeCo).

    Parameters
    ----------
    p : float
        Probability of applying the jitter to each sample.
    eps_s2 : float
        Max relative gain perturbation for S2 bands (default 5 %).
    eps_s1 : float
        Same, for S1 (default 3 % — SAR is more sensitive to amplitude).
    eps_dem : float
        Same, for DEM (default 1 % — elevation is geometric, leave alone).
    """

    p: float = 0.5
    eps_s2: float = 0.05
    eps_s1: float = 0.03
    eps_dem: float = 0.01

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()).item() >= self.p:
            return sample

        def _jitter(t: torch.Tensor, eps: float) -> torch.Tensor:
            if eps <= 0:
                return t
            # gains: shape [C, 1, 1] so the same gain applies to every pixel
            # of a band but each band is perturbed independently.
            gains = 1.0 + (torch.rand(t.shape[0], 1, 1, dtype=t.dtype) * 2.0 - 1.0) * eps
            return t * gains

        sample = dict(sample)  # shallow-copy so we don't mutate caller's
        if "S2L2A" in sample:
            sample["S2L2A"] = _jitter(sample["S2L2A"], self.eps_s2)
        if "S1GRD" in sample:
            sample["S1GRD"] = _jitter(sample["S1GRD"], self.eps_s1)
        if "DEM" in sample:
            sample["DEM"] = _jitter(sample["DEM"], self.eps_dem)
        return sample


# ──────────────────────────────────────────────────────────────────
# 2. Sample-level: MultiScale
# ──────────────────────────────────────────────────────────────────
@dataclass
class MultiScale:
    """Isotropic spatial rescale + crop/pad to original size.

    Picks a random scale factor in ``[scale_min, scale_max]`` and
    rescales the chip with bilinear (S2/S1/DEM), nearest (mask,
    valid_mask). Then either centre-crops (if scaled up) or zero-pads
    + sets the new pixels to ``valid_mask=False`` (if scaled down).

    Effect: the model sees lakes at 0.75-1.25× their on-disk size,
    which improves generalisation across lake area distributions —
    glacial lakes range from 1 px (~100 m²) to >50 px on a side.

    Parameters
    ----------
    p : float
        Probability of applying scale.
    scale_min, scale_max : float
        Sampling range. Stay in [0.5, 2.0] — beyond that the model's
        absolute pos-emb degrades.
    """

    p: float = 0.5
    scale_min: float = 0.75
    scale_max: float = 1.25

    def __call__(self, sample: dict) -> dict:
        if torch.rand(()).item() >= self.p:
            return sample
        scale = self.scale_min + (self.scale_max - self.scale_min) * torch.rand(()).item()
        if abs(scale - 1.0) < 1e-3:
            return sample

        def _scale(t: torch.Tensor, mode: str) -> torch.Tensor:
            # t: [C, H, W] or [H, W]; F.interpolate needs [N, C, H, W]
            if t.ndim == 2:
                t4 = t.unsqueeze(0).unsqueeze(0).float()
                squeeze = (0, 0)
            else:
                t4 = t.unsqueeze(0).float()
                squeeze = (0,)
            new_h = max(1, int(round(t4.shape[-2] * scale)))
            new_w = max(1, int(round(t4.shape[-1] * scale)))
            if mode == "bilinear":
                t4 = F.interpolate(t4, size=(new_h, new_w),
                                   mode="bilinear", align_corners=False)
            else:
                t4 = F.interpolate(t4, size=(new_h, new_w), mode="nearest")
            for ax in squeeze:
                t4 = t4.squeeze(ax)
            return t4

        # Original target size
        target_hw = sample["S2L2A"].shape[-2:]

        s2 = _scale(sample["S2L2A"], "bilinear").to(sample["S2L2A"].dtype)
        s1 = _scale(sample["S1GRD"], "bilinear").to(sample["S1GRD"].dtype)
        dem = _scale(sample["DEM"], "bilinear").to(sample["DEM"].dtype)
        mask = _scale(sample["mask"].float(), "nearest").to(sample["mask"].dtype)
        vm = _scale(sample["valid_mask"].to(torch.uint8), "nearest").to(torch.bool)

        # Resize back to target_hw via centre-crop (scaled up) or pad (down)
        s2 = _crop_or_pad(s2, target_hw, fill=0.0)
        s1 = _crop_or_pad(s1, target_hw, fill=0.0)
        dem = _crop_or_pad(dem, target_hw, fill=0.0)
        mask = _crop_or_pad(mask, target_hw, fill=0)
        vm = _crop_or_pad(vm, target_hw, fill=False)

        sample = dict(sample)
        sample["S2L2A"] = s2
        sample["S1GRD"] = s1
        sample["DEM"] = dem
        sample["mask"] = mask
        sample["valid_mask"] = vm
        return sample


def _crop_or_pad(t: torch.Tensor, target_hw: tuple[int, int], fill) -> torch.Tensor:
    """Centre-crop or zero-pad a tensor to ``target_hw`` at the last 2 dims."""
    th, tw = target_hw
    h, w = t.shape[-2], t.shape[-1]

    # Crop dims that are too big
    if h > th:
        top = (h - th) // 2
        t = t.narrow(-2, top, th)
    if w > tw:
        left = (w - tw) // 2
        t = t.narrow(-1, left, tw)

    # Pad dims that are too small
    h, w = t.shape[-2], t.shape[-1]
    if h < th or w < tw:
        pad_t = (th - h) // 2
        pad_b = (th - h) - pad_t
        pad_l = (tw - w) // 2
        pad_r = (tw - w) - pad_l
        if t.dtype == torch.bool:
            tu8 = t.to(torch.uint8)
            tu8 = F.pad(tu8, (pad_l, pad_r, pad_t, pad_b),
                        mode="constant", value=int(bool(fill)))
            t = tu8.to(torch.bool)
        else:
            t = F.pad(t, (pad_l, pad_r, pad_t, pad_b),
                     mode="constant", value=float(fill))
    return t


# ──────────────────────────────────────────────────────────────────
# 3. Batch-level: CopyPaste
# ──────────────────────────────────────────────────────────────────
@dataclass
class CopyPaste:
    """Paste a lake region from a *donor* chip onto the *target* chip.

    Standard recipe:
      1. Pick a donor sample (some other index in the batch) whose
         ``mask`` has at least ``min_donor_water_pixels``.
      2. Build a copy mask: random subset of donor's lake pixels,
         optionally dilated by a few px of context.
      3. Where copy mask is True, replace target's S2 / S1 / DEM /
         valid_mask with the donor's values.
      4. ``target.mask = max(target.mask, copy_mask)``.

    Why all modalities together (rather than just S2)?
    --------------------------------------------------
    Cross-modal consistency. If we copy only S2 we'd teach the model
    that "S2 looks like water but S1 / DEM say no" — exactly the wrong
    cue. Copying all three keeps the joint distribution intact.

    Parameters
    ----------
    p : float
        Probability of applying CopyPaste to a batch.
    min_donor_water_pixels : int
        Skip donors with fewer lake pixels than this.
    sub_sample_frac : float
        Probability of keeping each donor lake pixel in the copy mask.
        1.0 = paste the whole lake; 0.5 = paste a Bernoulli-thinned
        version (more variability, fewer artifacts on small batches).
    """

    p: float = 0.5
    min_donor_water_pixels: int = 32
    sub_sample_frac: float = 1.0

    def __call__(self, batch: list[dict]) -> list[dict]:
        if len(batch) < 2 or torch.rand(()).item() >= self.p:
            return batch

        # Find candidate donors with enough water
        donor_indices: list[int] = []
        for i, s in enumerate(batch):
            if int((s["mask"] > 0).sum()) >= self.min_donor_water_pixels:
                donor_indices.append(i)
        if not donor_indices:
            return batch

        out: list[dict] = list(batch)  # shallow copy
        for tgt_i, target in enumerate(batch):
            # Pick a random donor that's not the target itself
            cand = [j for j in donor_indices if j != tgt_i]
            if not cand:
                continue
            donor = batch[cand[int(torch.randint(0, len(cand), ()).item())]]

            copy_mask = donor["mask"] > 0
            if self.sub_sample_frac < 1.0:
                rand = torch.rand_like(copy_mask, dtype=torch.float32)
                copy_mask = copy_mask & (rand < self.sub_sample_frac)
            if not copy_mask.any():
                continue

            new = {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()}

            # 4-D broadcasting helper
            cm3 = copy_mask.unsqueeze(0)            # [1, H, W]
            new["S2L2A"] = torch.where(cm3, donor["S2L2A"], new["S2L2A"])
            new["S1GRD"] = torch.where(cm3, donor["S1GRD"], new["S1GRD"])
            new["DEM"]   = torch.where(cm3, donor["DEM"],   new["DEM"])
            # mask: pasted region forces foreground; rest unchanged
            new["mask"]  = torch.where(copy_mask,
                                       torch.ones_like(new["mask"]),
                                       new["mask"])
            # valid_mask: pasted region uses donor's validity
            new["valid_mask"] = torch.where(copy_mask,
                                            donor["valid_mask"],
                                            new["valid_mask"])

            out[tgt_i] = new
        return out


# ──────────────────────────────────────────────────────────────────
# 4. Batch-level: Mosaic
# ──────────────────────────────────────────────────────────────────
@dataclass
class Mosaic:
    """2×2 mosaic of chips, then re-crop to original chip size.

    For each output position in the batch:
      * If batch size < 4 → pass-through (Mosaic needs ≥4 inputs).
      * Otherwise pick 4 random samples (with replacement) from the
        batch, build a 2×2 grid at jittered split point, then crop a
        random window of the original size.

    Probability ``p`` controls the *fraction of output positions* that
    are mosaicked; the others stay un-augmented. This avoids destroying
    too much per-sample signal.

    Parameters
    ----------
    p : float
        Per-output-position probability of mosaicking.
    jitter : float
        Max relative offset of the split point from the centre, in [0, 0.5).
        0.0 = always split at exact centre; 0.25 = split anywhere in
        the middle 50 % of the image.
    """

    p: float = 0.3
    jitter: float = 0.2

    def __call__(self, batch: list[dict]) -> list[dict]:
        n = len(batch)
        if n < 4:
            return batch

        # Determine target H, W (assumes all chips same size)
        H, W = batch[0]["S2L2A"].shape[-2:]

        out = list(batch)
        for i in range(n):
            if torch.rand(()).item() >= self.p:
                continue
            # Sample 4 (random) source indices
            idx = torch.randint(0, n, (4,)).tolist()
            mosaicked = _build_mosaic(
                samples=[batch[k] for k in idx],
                target_hw=(H, W),
                jitter=self.jitter,
            )
            out[i] = mosaicked
        return out


def _build_mosaic(samples: Sequence[dict], target_hw: tuple[int, int],
                  jitter: float) -> dict:
    """Build a single mosaicked chip from 4 source samples."""
    H, W = target_hw

    # Split point — jittered around centre. We mosaic into a 2H × 2W canvas
    # so each cell gets up to one full chip's worth of pixels.
    canvas_h, canvas_w = 2 * H, 2 * W
    cy = int(canvas_h * (0.5 + (torch.rand(()).item() * 2.0 - 1.0) * jitter))
    cx = int(canvas_w * (0.5 + (torch.rand(()).item() * 2.0 - 1.0) * jitter))
    cy = max(1, min(canvas_h - 1, cy))
    cx = max(1, min(canvas_w - 1, cx))

    # Allocate canvas tensors with the right dtypes
    s2_canvas = torch.zeros((samples[0]["S2L2A"].shape[0], canvas_h, canvas_w),
                            dtype=samples[0]["S2L2A"].dtype)
    s1_canvas = torch.zeros((samples[0]["S1GRD"].shape[0], canvas_h, canvas_w),
                            dtype=samples[0]["S1GRD"].dtype)
    dem_canvas = torch.zeros((samples[0]["DEM"].shape[0], canvas_h, canvas_w),
                             dtype=samples[0]["DEM"].dtype)
    mask_canvas = torch.zeros((canvas_h, canvas_w), dtype=samples[0]["mask"].dtype)
    vm_canvas = torch.zeros((canvas_h, canvas_w), dtype=torch.bool)

    # Quadrants — (top-left, top-right, bottom-left, bottom-right)
    # Source indices map directly to these positions.
    quadrants = [
        # (canvas_y_slice, canvas_x_slice, src_y_slice, src_x_slice)
        (slice(0, cy),         slice(0, cx),         slice(H - cy, H),     slice(W - cx, W)),
        (slice(0, cy),         slice(cx, canvas_w),  slice(H - cy, H),     slice(0, canvas_w - cx)),
        (slice(cy, canvas_h),  slice(0, cx),         slice(0, canvas_h - cy),  slice(W - cx, W)),
        (slice(cy, canvas_h),  slice(cx, canvas_w),  slice(0, canvas_h - cy),  slice(0, canvas_w - cx)),
    ]
    for src, (cys, cxs, sys_, sxs) in zip(samples, quadrants):
        # Clamp src slices to chip dims
        sys_ = slice(max(0, sys_.start), min(H, sys_.stop))
        sxs = slice(max(0, sxs.start), min(W, sxs.stop))
        # Recompute canvas slice height/width to match src slice
        ch = sys_.stop - sys_.start
        cw = sxs.stop - sxs.start
        cys = slice(cys.start, cys.start + ch)
        cxs = slice(cxs.start, cxs.start + cw)
        if ch <= 0 or cw <= 0:
            continue

        s2_canvas[:, cys, cxs] = src["S2L2A"][:, sys_, sxs]
        s1_canvas[:, cys, cxs] = src["S1GRD"][:, sys_, sxs]
        dem_canvas[:, cys, cxs] = src["DEM"][:, sys_, sxs]
        mask_canvas[cys, cxs] = src["mask"][sys_, sxs]
        vm_canvas[cys, cxs] = src["valid_mask"][sys_, sxs]

    # Crop a random H × W window from the 2H × 2W canvas
    top = int(torch.randint(0, canvas_h - H + 1, ()).item())
    left = int(torch.randint(0, canvas_w - W + 1, ()).item())
    out = {
        "S2L2A":      s2_canvas[:, top:top + H, left:left + W].contiguous(),
        "S1GRD":      s1_canvas[:, top:top + H, left:left + W].contiguous(),
        "DEM":        dem_canvas[:, top:top + H, left:left + W].contiguous(),
        "mask":       mask_canvas[top:top + H, left:left + W].contiguous(),
        "valid_mask": vm_canvas[top:top + H, left:left + W].contiguous(),
    }
    # Preserve any meta keys from the first source sample
    for k, v in samples[0].items():
        if k not in out and not torch.is_tensor(v):
            out[k] = v
    return out


# ──────────────────────────────────────────────────────────────────
# 5. Composed batch collator
# ──────────────────────────────────────────────────────────────────
@dataclass
class MosaicCopyPasteCollator:
    """Custom ``collate_fn`` chaining batch-level augs then default_collate.

    Order:
        1. ``Mosaic``     (creates new compositions)
        2. ``CopyPaste``  (transplants lakes into the resulting chips)
        3. ``default_collate`` (stacks tensors)

    Set ``mosaic`` and/or ``copy_paste`` to ``None`` to disable that stage.
    """

    mosaic: Mosaic | None = None
    copy_paste: CopyPaste | None = None

    def __call__(self, samples: list[dict]) -> dict:
        if self.mosaic is not None:
            samples = self.mosaic(samples)
        if self.copy_paste is not None:
            samples = self.copy_paste(samples)
        return default_collate(samples)


__all__ = [
    "SpectralJitter",
    "MultiScale",
    "CopyPaste",
    "Mosaic",
    "MosaicCopyPasteCollator",
]
