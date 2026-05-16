"""Flip-only Test-Time Augmentation (TTA) for ViT-based segmenters.

Why flip-only?
--------------
TerraMind / Prithvi / SatMAE all use *absolute* sinusoidal or learned
positional embeddings on patch tokens. Naïve 90° / 180° rotations rearrange
the spatial token grid in a way that the absolute pos-emb has never seen
during pre-training, which **degrades** rather than improves the prediction.

Horizontal and vertical flips, on the other hand, act on each row / column
independently and only re-order tokens along one axis — the absolute pos-emb
still maps each spatial location to a valid grid position, so the prediction
remains coherent.

Empirically (CryoSentinel internal tests, see notebooks/eval_tta.ipynb):

* Identity              IoU = 0.78
* + 4-way rot           IoU = 0.71  ← *worse*
* + 4-way flip (this)   IoU = 0.81  ← +0.03 reliable

This module exposes two helpers:

* :func:`flip_only_tta_logits` — averages **logits** of the 4 flip orbit
  augmentations. Use during validation when calibrated probabilities matter.
* :func:`flip_only_tta` — same but operates on a higher-level ``predict``
  callable that takes a multi-modal dict and returns logits.

Both undo the flip on the output before averaging so the prediction stays
aligned with the input.
"""
from __future__ import annotations

from typing import Callable, Mapping

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────
#  Low-level: 4-way flip orbit on a tensor [B, C, H, W]
# ──────────────────────────────────────────────────────────────────────
_FLIP_DIMS: list[tuple[int, ...]] = [
    (),         # identity
    (-1,),      # horizontal flip (mirror left↔right)
    (-2,),      # vertical flip   (mirror top↔bottom)
    (-2, -1),   # both (== 180° rotation, but achieved without rot pos-emb shift)
]


def _flip(t: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    return t.flip(dims=dims) if dims else t


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────
def flip_only_tta_logits(
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor] | torch.Tensor,
    *,
    forward_fn: Callable[[nn.Module, Mapping[str, torch.Tensor] | torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Average logits over the 4-element flip group.

    Args:
        model: any segmenter returning ``[B, 1, H, W]`` logits.
        inputs: either a tensor ``[B, C, H, W]`` or a dict of modality tensors
            (each ``[B, Cm, H, W]``). Flips are applied to the *spatial*
            dimensions of every tensor in the dict.
        forward_fn: how to call the model. Defaults to ``model(inputs)``.

    Returns:
        Logits ``[B, 1, H, W]`` on the original (un-flipped) frame.
    """
    if forward_fn is None:
        def forward_fn(m, x):                              # type: ignore[no-redef]
            return m(x)

    out_logits = None
    for dims in _FLIP_DIMS:
        # Apply flip to every spatial tensor
        if isinstance(inputs, Mapping):
            flipped = {k: _flip(v, dims) if v.ndim == 4 else v
                       for k, v in inputs.items()}
        else:
            flipped = _flip(inputs, dims)

        logits = forward_fn(model, flipped)                # [B, 1, H, W]
        # Undo the flip on the output so it aligns with original frame
        logits = _flip(logits, dims)

        out_logits = logits if out_logits is None else (out_logits + logits)

    assert out_logits is not None
    return out_logits / float(len(_FLIP_DIMS))


def flip_only_tta(
    predict: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
    inputs: Mapping[str, torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Functional variant of :func:`flip_only_tta_logits` for arbitrary callables.

    The callable receives the (possibly flipped) ``inputs`` dict / tensor and
    must return logits on the same spatial frame.
    """
    out_logits = None
    for dims in _FLIP_DIMS:
        if isinstance(inputs, Mapping):
            flipped = {k: _flip(v, dims) if v.ndim == 4 else v
                       for k, v in inputs.items()}
        else:
            flipped = _flip(inputs, dims)
        logits = predict(flipped)
        logits = _flip(logits, dims)
        out_logits = logits if out_logits is None else (out_logits + logits)
    assert out_logits is not None
    return out_logits / float(len(_FLIP_DIMS))
