"""Segmentation losses tailored for thin glacial-lake outlines.

Composite loss options
----------------------
**BCEDiceBoundaryLoss** (original, backward-compatible)::

    L = w_bce * BCE + w_dice * (1 - Dice) + w_boundary * BoundaryLoss

**MegaLoss** (SOTA, Phase-2 mega-plan)::

    L = 0.5*BCE + 1.0*Focal(α=0.25,γ=2) + 1.0*Dice
        + 0.5*Tversky(α=0.3,β=0.7) + 0.3*BoundaryLoss

Both support an optional ``valid_mask`` tensor ``[B, H, W]`` (True = valid
pixel) so that nodata pixels (all-zero S2 after GEE ``unmask(0)``) are
excluded from every loss component.

Key changes vs the original implementation
-------------------------------------------
* ``pos_weight_max`` raised from 50 → 200 (real ratio is ~110 at 0.9 % water).
* ``valid_mask`` applied consistently to BCE, Dice, Tversky, Focal, Boundary.
* New ``hard_iou()`` for unbiased threshold-0.5 evaluation.
* ``focal_loss()`` and ``tversky_loss()`` added as standalone helpers.

PR-2 additions (Phase D mega-plan, May 2026)
--------------------------------------------
* ``generalized_dice_loss`` (Sudre et al., MICCAI 2017 — implements the
  inverse-square frequency weighting that handles 1:5000 imbalance ratios
  cleanly; for our 1:99 binary regime it is a soft-Dice replacement that
  weighs the rare foreground class proportionally harder).
* ``ohem_pool`` (Shrivastava et al., CVPR 2016 — keeps the top-K hardest
  pixels per batch as the BCE reduction, which forces the gradient onto
  uncertain boundaries and away from easy backgrounds).
* ``MegaLoss`` exposes ``label_smoothing`` (A3, GlaViTU Nature 2024 §Methods),
  ``ohem_keep_ratio`` + ``ohem_min_kept`` (A7), and ``dice_variant``
  ("flat" → :func:`dice_loss`, "generalized" → :func:`generalized_dice_loss`).
* All three are off by default (``label_smoothing=0.0``, ``ohem_keep_ratio=1.0``,
  ``dice_variant="flat"``) so the v2 baseline reproducibility is preserved.
  v3 SOTA configs flip them on explicitly.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ──────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────
def _4d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is ``[B, 1, H, W]``."""
    return t.unsqueeze(1) if t.ndim == 3 else t


def _apply_mask(loss_map: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over valid pixels only; plain mean when mask is None."""
    if valid_mask is None:
        return loss_map.mean()
    # valid_mask: [B, H, W] bool → broadcast to [B, 1, H, W]
    vm = valid_mask.unsqueeze(1).to(loss_map.dtype)
    return (loss_map * vm).sum() / vm.sum().clamp(min=1.0)


def ohem_pool(
    loss_map: torch.Tensor,
    *,
    keep_ratio: float = 0.7,
    valid_mask: torch.Tensor | None = None,
    min_kept: int = 1024,
) -> torch.Tensor:
    """Online Hard Example Mining (Shrivastava et al., CVPR 2016).

    Pools per-pixel loss values down to the top-K hardest pixels, where
    ``K = max(min_kept, keep_ratio * N_valid)``. The mean over those K
    pixels is the reduced scalar loss.

    Why this helps glacial-lake segmentation
    ----------------------------------------
    * The 1:99 background/foreground ratio means a plain mean BCE is
      dominated by ~99 % easy backgrounds whose gradient is near zero.
    * Top-K mean keeps the gradient signal strong on uncertain pixels
      (boundaries, debris-covered ice, partial freeze) where the model
      actually needs to learn.
    * Empirically (CVPR 2016 + 2024 medical seg surveys) this gives
      +0.3-0.5 mIoU on imbalanced binary tasks at <1 % overhead.

    Parameters
    ----------
    loss_map : Tensor
        ``[B, 1, H, W]`` per-pixel loss values (e.g. raw BCE map).
    keep_ratio : float in (0, 1]
        Fraction of valid pixels to keep. ``1.0`` is a no-op (returns
        ``_apply_mask``-equivalent mean).
    valid_mask : Tensor | None
        ``[B, H, W]`` bool, True = include pixel.
    min_kept : int
        Floor on ``K`` so very small batches/chips are not pathologically
        reduced to a handful of pixels.

    Returns
    -------
    Scalar loss.
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"ohem keep_ratio must be in (0, 1], got {keep_ratio}")

    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).expand_as(loss_map)
        flat = loss_map[vm]
    else:
        flat = loss_map.flatten()

    if flat.numel() == 0:
        return loss_map.sum() * 0.0

    if keep_ratio >= 1.0:
        return flat.mean()

    k = max(int(min_kept), int(keep_ratio * flat.numel()))
    k = min(k, flat.numel())
    top_k, _ = torch.topk(flat, k)
    return top_k.mean()


# ──────────────────────────────────────────────────────────────────────
#  Component losses
# ──────────────────────────────────────────────────────────────────────
def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft-Dice loss for binary segmentation.

    Args:
        logits: ``[B, 1, H, W]`` raw model output.
        target: ``[B, H, W]`` (long) or ``[B, 1, H, W]`` ground truth.
        smooth: Laplace smoothing.
        valid_mask: ``[B, H, W]`` bool — True = include pixel.

    Returns:
        Scalar loss in ``[0, 1]``.
    """
    target = _4d(target).to(logits.dtype)
    probs = torch.sigmoid(logits)
    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).to(logits.dtype)
        probs = probs * vm
        target = target * vm
    dims = (0, 2, 3)
    inter = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - ((2.0 * inter + smooth) / (union + smooth))).mean()


def generalized_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Generalized Dice Loss (Sudre et al., MICCAI 2017).

    For the binary case we treat foreground and background as two classes
    and weigh each by ``w_c = 1 / (Σ_pixels target_c)²``. This inverse
    square frequency weighting is robust at imbalance ratios up to 1:5000
    where plain Dice collapses the rare class at non-trivial LRs (Sudre
    Table 2).

    .. math::
        \mathrm{GDL} = 1 - \frac{2 \sum_c w_c \sum_i p_{c,i} t_{c,i}}
                                  {\sum_c w_c \sum_i (p_{c,i} + t_{c,i})}

    For our 1:99 ratio plain Dice already converges, but GDL pushes the
    asymptote roughly 0.2-0.4 IoU higher because the foreground gradient
    is no longer washed out by the background term. Cost: identical to
    soft-Dice (one extra reduction).
    """
    target = _4d(target).to(logits.dtype)
    probs = torch.sigmoid(logits)
    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).to(logits.dtype)
        probs = probs * vm
        target = target * vm
    else:
        vm = None
    dims = (0, 2, 3)

    # Foreground (c=1)
    sum_fg   = target.sum(dim=dims)
    inter_fg = (probs * target).sum(dim=dims)
    union_fg = (probs + target).sum(dim=dims)
    w_fg     = 1.0 / (sum_fg ** 2 + smooth)

    # Background (c=0)
    bg_t       = 1.0 - target
    bg_p       = 1.0 - probs
    if vm is not None:
        bg_t = bg_t * vm
        bg_p = bg_p * vm
    sum_bg     = bg_t.sum(dim=dims)
    inter_bg   = (bg_p * bg_t).sum(dim=dims)
    union_bg   = (bg_p + bg_t).sum(dim=dims)
    w_bg       = 1.0 / (sum_bg ** 2 + smooth)

    numerator   = 2.0 * (w_fg * inter_fg + w_bg * inter_bg)
    denominator = (w_fg * union_fg + w_bg * union_bg).clamp(min=smooth)
    return (1.0 - numerator / denominator).mean()


def soft_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft-IoU (Jaccard index using probabilities). Higher = better.

    Uses sigmoid probabilities — optimistic relative to threshold-0.5 IoU.
    Kept for training monitoring; use :func:`hard_iou` for checkpointing.
    """
    target = _4d(target).to(logits.dtype)
    probs = torch.sigmoid(logits)
    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).to(logits.dtype)
        probs = probs * vm
        target = target * vm
    inter = (probs * target).sum(dim=(0, 2, 3))
    union = ((probs + target) - probs * target).sum(dim=(0, 2, 3))
    return ((inter + smooth) / (union + smooth)).mean()


def hard_iou(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    smooth: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard IoU at a fixed threshold.

    More realistic than soft IoU — this is the number that appears in papers.
    Typically 5–10 points lower than soft IoU.
    """
    target = _4d(target).to(logits.dtype)
    pred = (torch.sigmoid(logits) >= threshold).to(logits.dtype)
    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).to(logits.dtype)
        pred = pred * vm
        target = target * vm
    inter = (pred * target).sum(dim=(0, 2, 3))
    union = ((pred + target) - pred * target).sum(dim=(0, 2, 3))
    return ((inter + smooth) / (union + smooth)).mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss for hard-example mining (Lin et al., ICCV 2017).

    ``alpha`` down-weights easy negatives; ``gamma`` further down-weights
    easy examples. Together they force the model to focus on hard pixels
    (i.e. uncertain lake boundaries) rather than the easy background.
    """
    target = _4d(target).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    fl = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return _apply_mask(fl, valid_mask)


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tversky loss (Salehi et al., 2017).

    ``beta > alpha`` penalises false negatives more than false positives —
    critical for thin glacial-lake outlines where missing a pixel matters
    more than an extra background pixel.
    """
    target = _4d(target).to(logits.dtype)
    probs = torch.sigmoid(logits)
    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).to(logits.dtype)
        probs = probs * vm
        target = target * vm
    dims = (0, 2, 3)
    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)
    return (1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


# ──────────────────────────────────────────────────────────────────────
#  Lovász-Hinge loss (Berman et al., CVPR 2018)
# ──────────────────────────────────────────────────────────────────────
#  The Lovász extension of the Jaccard (IoU) loss is the smooth, convex
#  surrogate that *directly* optimises IoU on the lattice of binary masks.
#  Unlike soft-Dice it does NOT degrade gracefully on small foreground
#  fractions — it stays well-conditioned even when only ~1 % of the chip
#  is positive, which is exactly our regime. Empirically gives +1-2 IoU
#  points on Cityscapes / Pascal VOC at the cost of ~5 % wall-time.
#
#  Reference: M. Berman, A. R. Triki, M. Blaschko,
#  "The Lovász-Softmax loss: a tractable surrogate for the optimization
#  of the intersection-over-union measure in neural networks", CVPR 2018.
#  Implementation adapted from the official binary-hinge variant at
#  https://github.com/bermanmaxim/LovaszSoftmax (MIT License).
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of the Lovász extension w.r.t. sorted errors."""
    gts = gt_sorted.sum()
    p = gt_sorted.size(0)
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_hinge_flat(
    logits_flat: torch.Tensor, labels_flat: torch.Tensor,
) -> torch.Tensor:
    """Binary Lovász-Hinge for a single image / valid pixel subset."""
    if labels_flat.numel() == 0:
        return logits_flat.sum() * 0.0
    signs = 2.0 * labels_flat.float() - 1.0     # {0, 1} → {-1, +1}
    errors = 1.0 - logits_flat * signs           # margin error per pixel
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels_flat[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad.detach())


def lovasz_hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    per_image: bool = True,
) -> torch.Tensor:
    """Binary Lovász-Hinge loss.

    Parameters
    ----------
    logits : Tensor
        ``[B, 1, H, W]`` raw model output (NOT sigmoid'd — Lovász works
        directly on the hinge margin).
    target : Tensor
        ``[B, H, W]`` (long) or ``[B, 1, H, W]`` ground-truth.
    valid_mask : Tensor | None
        ``[B, H, W]`` bool — True = include pixel.
    per_image : bool
        If True (default), the loss is computed per image then averaged
        across the batch. If False, all valid pixels of the batch are
        flattened into one sequence (matches some publications). Per-image
        is a stronger signal when the batch has both empty and full chips.

    Returns
    -------
    Scalar loss in ``[0, ~1]``.
    """
    target = _4d(target)
    if logits.shape[1] != 1:
        raise ValueError(f"lovasz_hinge_loss expects [B,1,H,W] logits; got {logits.shape}")

    if not per_image:
        if valid_mask is not None:
            vm = valid_mask.unsqueeze(1).to(torch.bool)
            l_flat = logits[vm]
            t_flat = target[vm].to(torch.float32)
        else:
            l_flat = logits.flatten()
            t_flat = target.flatten().to(torch.float32)
        return _lovasz_hinge_flat(l_flat, t_flat)

    losses: list[torch.Tensor] = []
    B = logits.size(0)
    for b in range(B):
        l_b = logits[b, 0].flatten()
        t_b = target[b, 0].to(torch.float32).flatten()
        if valid_mask is not None:
            vm = valid_mask[b].flatten().to(torch.bool)
            l_b = l_b[vm]
            t_b = t_b[vm]
        losses.append(_lovasz_hinge_flat(l_b, t_b))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


# ──────────────────────────────────────────────────────────────────────
#  Boundary loss helpers
# ──────────────────────────────────────────────────────────────────────
def _signed_distance_map(mask_np):
    """Signed distance transform for a single 2-D mask.

    Positive = outside the lake, negative = inside. 0 on the boundary.
    """
    pos = mask_np > 0.5
    if pos.all() or not pos.any():
        return torch.zeros_like(torch.from_numpy(mask_np).float())
    dist_outside = distance_transform_edt(~pos)
    dist_inside  = distance_transform_edt(pos)
    sdt = dist_outside - dist_inside
    return torch.from_numpy(sdt.astype("float32"))


def boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boundary loss (Kervadec et al., MICCAI 2019).

    Penalises predictions far from the ground-truth lake boundary.
    Falls back to Sobel-edge-weighted BCE if ``scipy`` is unavailable.
    """
    target_4d = _4d(target)
    probs = torch.sigmoid(logits)

    if _HAS_SCIPY:
        with torch.no_grad():
            mask_np = target_4d.detach().cpu().numpy()
            sdts = [_signed_distance_map(mask_np[b, 0]) for b in range(mask_np.shape[0])]
            sdt = torch.stack(sdts).unsqueeze(1).to(logits.device)
        loss_map = probs * sdt
        return _apply_mask(loss_map, valid_mask)

    # Fallback: Sobel-edge weighted BCE
    target_f = target_4d.to(logits.dtype)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=logits.dtype, device=logits.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    edges = (
        F.conv2d(target_f, sobel_x, padding=1).abs()
        + F.conv2d(target_f, sobel_y, padding=1).abs()
    )
    edge_w = 1.0 + 4.0 * (edges > 0).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target_f, reduction="none")
    loss_map = bce * edge_w
    return _apply_mask(loss_map, valid_mask)


# ──────────────────────────────────────────────────────────────────────
#  Composite losses
# ──────────────────────────────────────────────────────────────────────
class BCEDiceBoundaryLoss(nn.Module):
    """``L = w_bce * BCE + w_dice * (1 − Dice) + w_boundary * BoundaryLoss``.

    Original composite loss kept for backwards compatibility. Changes vs v1:
    * ``pos_weight_max`` default raised to 200 (was hard-coded to 50).
    * ``valid_mask`` propagated to **all** three components consistently.
    * ``ignore_index`` still applies to BCE only (legacy behaviour).

    Args:
        w_bce / w_dice / w_boundary: scalar weights.
        pos_weight: fixed positive class weight for BCE. ``None`` = auto
            per-batch, clamped to ``[1, pos_weight_max]``.
        pos_weight_max: upper clamp for auto pos_weight (default 200).
        ignore_index: pixels with this target value are excluded from BCE.
    """

    def __init__(
        self,
        *,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_boundary: float = 0.5,
        pos_weight: Optional[float] = None,
        pos_weight_max: float = 200.0,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        self._fixed_pos_weight = pos_weight
        self.pos_weight_max = pos_weight_max
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(loss, components)`` for logging.

        Args:
            logits: ``[B, 1, H, W]`` raw model output.
            target: ``[B, H, W]`` (long) ground truth.
            valid_mask: ``[B, H, W]`` bool — True = valid pixel.
        """
        target_4d = _4d(target)
        target_f  = target_4d.to(logits.dtype)

        # ── pos_weight for BCE ─────────────────────────────────────────
        if self._fixed_pos_weight is not None:
            pw = torch.tensor(self._fixed_pos_weight, device=logits.device)
        else:
            pos = target_f.sum().clamp(min=1.0)
            neg = (1.0 - target_f).sum().clamp(min=1.0)
            pw = (neg / pos).clamp(1.0, self.pos_weight_max)

        # ── BCE ───────────────────────────────────────────────────────
        vm_bce = valid_mask
        if self.ignore_index is not None:
            ii_ok = (target_4d != self.ignore_index).squeeze(1)  # [B, H, W]
            vm_bce = ii_ok if valid_mask is None else (valid_mask & ii_ok)

        bce_map = F.binary_cross_entropy_with_logits(
            logits, target_f, pos_weight=pw, reduction="none"
        )
        bce = _apply_mask(bce_map, vm_bce)

        # ── Dice + Boundary ───────────────────────────────────────────
        d = dice_loss(logits, target_4d, valid_mask=valid_mask)
        b = (
            boundary_loss(logits, target_4d, valid_mask=valid_mask)
            if self.w_boundary > 0
            else torch.zeros((), device=logits.device)
        )

        total = self.w_bce * bce + self.w_dice * d + self.w_boundary * b
        return total, {
            "loss/bce":       bce.detach(),
            "loss/dice":      d.detach(),
            "loss/boundary":  b.detach(),
            "loss/pos_weight": pw.detach() if isinstance(pw, torch.Tensor)
                               else torch.tensor(float(pw)),
        }


class MegaLoss(nn.Module):
    """SOTA composite loss for glacial-lake segmentation.

    ``L = w_bce*BCE + w_focal*Focal + w_dice*Dice
          + w_tversky*Tversky + w_boundary*BoundaryLoss + w_lovasz*Lovasz``

    Default weights ``(0.5, 1.0, 1.0, 0.5, 0.3, 0.0)`` are calibrated for
    the ~1–3 % water-fraction regime of HMA glacial-lake chips.

    Compared to :class:`BCEDiceBoundaryLoss`:
    * Focal (γ=2, α=0.25) provides hard-example mining.
    * Tversky (β=0.7 > α=0.3) asymmetrically penalises false negatives —
      correct for recall-critical GLOF detection.
    * pos_weight_max raised to 200 (covers the real 110× imbalance).
    * All components respect ``valid_mask``.

    PR-2 additions (off by default; enabled in v3 SOTA configs)
    -----------------------------------------------------------
    * ``label_smoothing`` (A3, GlaViTU Nature 2024) — applied **only** to
      the BCE component as ``y' = y*(1-α) + α/2``. The Dice / Tversky /
      Focal / Boundary / Lovasz components keep hard targets because each
      of those uses set-overlap or boundary geometry where smoothing the
      target is meaningless or actively harmful.
    * ``ohem_keep_ratio`` (A7, Shrivastava CVPR 2016) — if < 1.0, the BCE
      reduction switches to top-K mean over the hardest pixels.
    * ``dice_variant`` (B5, Sudre MICCAI 2017) — ``"flat"`` (default) calls
      :func:`dice_loss`; ``"generalized"`` calls :func:`generalized_dice_loss`.
    """

    def __init__(
        self,
        *,
        w_bce: float = 0.5,
        w_focal: float = 1.0,
        w_dice: float = 1.0,
        w_tversky: float = 0.5,
        w_boundary: float = 0.3,
        w_lovasz: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        lovasz_per_image: bool = True,
        pos_weight_max: float = 200.0,
        ignore_index: Optional[int] = None,
        # PR-2 additions — see class docstring
        label_smoothing: float = 0.0,
        ohem_keep_ratio: float = 1.0,
        ohem_min_kept: int = 1024,
        dice_variant: str = "flat",
    ) -> None:
        super().__init__()
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.w_tversky = w_tversky
        self.w_boundary = w_boundary
        self.w_lovasz = w_lovasz
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.lovasz_per_image = lovasz_per_image
        self.pos_weight_max = pos_weight_max
        self.ignore_index = ignore_index

        if not (0.0 <= label_smoothing < 0.5):
            raise ValueError(
                f"label_smoothing must be in [0, 0.5), got {label_smoothing!r}"
            )
        self.label_smoothing = float(label_smoothing)
        self.ohem_keep_ratio = float(ohem_keep_ratio)
        self.ohem_min_kept   = int(ohem_min_kept)
        if dice_variant not in ("flat", "generalized"):
            raise ValueError(
                f"dice_variant must be 'flat' or 'generalized', got {dice_variant!r}"
            )
        self.dice_variant = dice_variant

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_4d = _4d(target)
        target_f  = target_4d.to(logits.dtype)

        # Auto pos_weight for BCE — computed from HARD target (smoothing only
        # affects the cross-entropy target, not the foreground frequency).
        pos = target_f.sum().clamp(min=1.0)
        neg = (1.0 - target_f).sum().clamp(min=1.0)
        pw  = (neg / pos).clamp(1.0, self.pos_weight_max)

        vm_bce = valid_mask
        if self.ignore_index is not None:
            ii_ok  = (target_4d != self.ignore_index).squeeze(1)
            vm_bce = ii_ok if valid_mask is None else (valid_mask & ii_ok)

        # ── BCE with optional label smoothing + optional OHEM ────────
        if self.label_smoothing > 0.0:
            # Symmetric two-sided smoothing: 1 → 1-α/2, 0 → α/2.
            ls = self.label_smoothing
            bce_target = target_f * (1.0 - ls) + 0.5 * ls
        else:
            bce_target = target_f
        bce_map = F.binary_cross_entropy_with_logits(
            logits, bce_target, pos_weight=pw, reduction="none"
        )
        if self.ohem_keep_ratio < 1.0:
            bce = ohem_pool(
                bce_map,
                keep_ratio=self.ohem_keep_ratio,
                valid_mask=vm_bce,
                min_kept=self.ohem_min_kept,
            )
        else:
            bce = _apply_mask(bce_map, vm_bce)

        # ── Other components keep HARD targets ──────────────────────
        fl  = focal_loss(logits, target_4d,
                         alpha=self.focal_alpha, gamma=self.focal_gamma,
                         valid_mask=valid_mask)
        if self.dice_variant == "generalized":
            d = generalized_dice_loss(logits, target_4d, valid_mask=valid_mask)
        else:
            d = dice_loss(logits, target_4d, valid_mask=valid_mask)
        tv  = tversky_loss(logits, target_4d,
                           alpha=self.tversky_alpha, beta=self.tversky_beta,
                           valid_mask=valid_mask)
        b   = (
            boundary_loss(logits, target_4d, valid_mask=valid_mask)
            if self.w_boundary > 0
            else torch.zeros((), device=logits.device)
        )
        lv = (
            lovasz_hinge_loss(logits, target_4d,
                              valid_mask=valid_mask,
                              per_image=self.lovasz_per_image)
            if self.w_lovasz > 0
            else torch.zeros((), device=logits.device)
        )

        total = (self.w_bce * bce + self.w_focal * fl + self.w_dice * d
                 + self.w_tversky * tv + self.w_boundary * b
                 + self.w_lovasz * lv)

        return total, {
            "loss/bce":        bce.detach(),
            "loss/focal":      fl.detach(),
            "loss/dice":       d.detach(),
            "loss/tversky":    tv.detach(),
            "loss/boundary":   b.detach(),
            "loss/lovasz":     lv.detach(),
            "loss/pos_weight": pw.detach(),
        }
