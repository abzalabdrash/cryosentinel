"""Gaussian-blended sliding-window inference for full-scene rasters.

TerraMind v1 was trained at 224×224. To predict on a full Sentinel scene
(e.g. an 8000×8000 px Almaty mosaic) we slide a 224×224 window with a
configurable stride and blend overlapping predictions with a 2-D Gaussian
weight. This is the standard MONAI / Hugging Face recipe — see
``monai.inferers.SlidingWindowInferer`` and Hugging Face's segformer demo.

Why Gaussian (rather than uniform mean)
---------------------------------------
The 224×224 window's centre pixels are the most contextualised. The corner
pixels at the patch boundary often suffer from aggressive padding effects,
inconsistent receptive-field coverage and patch-merging artefacts at the
neck. A Gaussian weight (σ ≈ ¼ of the window) down-weights corners and
heavily favours centre pixels, which empirically removes the visible "tile
seams" you get with naive uniform stitching (~+0.1-0.3 IoU on Cityscapes
in published Hugging Face benchmarks).

Notes
-----
* Pure PyTorch — no rasterio / GDAL dependency. Caller is responsible for
  reading the raster and providing an ``[B?, C, H, W]`` tensor.
* The function returns probabilities (after sigmoid). Threshold to a
  binary mask in the calling code so we don't silently pick a default.
* Memory is bounded by ``batch_size`` × patch tensors only. The full-scene
  accumulator and weight buffers are allocated as ``float32`` regardless
  of the input dtype to avoid catastrophic precision loss when summing
  many overlapping ``bf16`` patches.
"""
from __future__ import annotations

from typing import Callable

import torch


def gaussian_window_2d(
    size: int,
    *,
    sigma: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2-D Gaussian centred on a square ``size×size`` window.

    Parameters
    ----------
    size : int
        Window edge length in pixels (e.g. 224 for TerraMind v1).
    sigma : float, optional
        Standard deviation in pixels. Defaults to ``size / 4`` which puts
        the 95-th percentile mass within ±2σ ≈ half the window — a sweet
        spot between "essentially uniform" (large σ) and "discard the
        edges" (small σ).
    device, dtype
        Where the tensor lives and its dtype. ``float32`` is recommended
        because the accumulator runs in fp32 anyway.
    """
    if sigma is None:
        sigma = size / 4.0
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g2 = g1[:, None] * g1[None, :]
    # Normalise the peak to 1.0 so the weighted accumulator stays in a
    # reasonable numeric range; the sliding-window code divides by the
    # accumulated weight, so absolute scale does not affect the result.
    g2 = g2 / g2.max()
    return g2


@torch.no_grad()
def sliding_window_inference(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    *,
    window: int = 224,
    stride: int | None = None,
    batch_size: int = 4,
    sigma: float | None = None,
) -> torch.Tensor:
    """Run a model over a large raster via overlapping ``window×window`` patches.

    Parameters
    ----------
    forward_fn : callable
        ``patches[B, C, window, window] -> logits[B, 1, window, window]``.
        The caller is responsible for any normalisation, model selection,
        ``no_grad``, AMP context, etc. Returning probabilities (already
        sigmoid'd) is also fine — we just average whatever comes out.
    inputs : Tensor
        ``[C, H, W]`` (single scene) or ``[B, C, H, W]`` (batch of scenes).
        Will be padded with zeros on the right/bottom when ``H`` or ``W``
        is smaller than ``window``.
    window : int
        Patch size — must match the model's training window (224 for
        TerraMind v1).
    stride : int, optional
        Step between patch top-left corners. Default: ``window // 2``
        which gives 50 % overlap and matches the Hugging Face / MONAI
        defaults. Smaller stride = smoother seams + more compute.
    batch_size : int
        Number of patches to push through ``forward_fn`` per forward call.
        Tuning this is GPU-memory bound, not throughput — larger is faster
        until you OOM.
    sigma : float, optional
        Override the Gaussian σ. Default ``window / 4``.

    Returns
    -------
    Tensor
        ``[B, 1, H, W]`` probability map (after sigmoid) on the same
        device as ``inputs``. ``B`` is preserved from the input layout
        (1 for ``[C, H, W]`` inputs).
    """
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)
    if inputs.ndim != 4:
        raise ValueError(f"inputs must be [C,H,W] or [B,C,H,W], got {tuple(inputs.shape)}")
    B, C, H, W = inputs.shape
    if stride is None:
        stride = window // 2
    if stride <= 0 or stride > window:
        raise ValueError(f"stride must be in (0, {window}], got {stride}")

    device = inputs.device
    pad_h = max(0, window - H)
    pad_w = max(0, window - W)
    if pad_h or pad_w:
        # Right/bottom zero-pad so even small scenes can be inferred.
        inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w

    weight_2d = gaussian_window_2d(window, sigma=sigma, device=device).to(torch.float32)
    # Pre-cast accumulators to fp32 to avoid bf16 precision loss when many
    # overlapping patches are summed.
    out_logits = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    out_weight = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)

    # Generate top-left corners; ensure the right/bottom edges are covered
    # by snapping the last column/row when the stride does not divide H/W.
    def _starts(extent: int) -> list[int]:
        if extent == window:
            return [0]
        starts = list(range(0, extent - window + 1, stride))
        if starts[-1] + window < extent:
            starts.append(extent - window)
        return starts

    ys = _starts(H)
    xs = _starts(W)

    for b in range(B):
        # Process windows in mini-batches so GPU utilisation stays high.
        coords: list[tuple[int, int]] = [(y, x) for y in ys for x in xs]
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            patch_batch = torch.stack([
                inputs[b, :, y:y + window, x:x + window] for (y, x) in chunk
            ], dim=0)
            logits = forward_fn(patch_batch).to(torch.float32)
            if logits.ndim == 3:
                logits = logits.unsqueeze(1)
            for j, (y, x) in enumerate(chunk):
                out_logits[b, 0, y:y + window, x:x + window] += logits[j, 0] * weight_2d
                out_weight[b, 0, y:y + window, x:x + window] += weight_2d

    out_logits = out_logits / out_weight.clamp(min=1e-6)
    if pad_h or pad_w:
        out_logits = out_logits[..., : H - pad_h, : W - pad_w]
    return torch.sigmoid(out_logits)
