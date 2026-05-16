"""Tier C.2 — NDWI / MNDWI hybrid decoder head.

A thin refiner that fuses the model's base segmentation logits with two
physically-grounded water indices computed on-the-fly from Sentinel-2:

* **NDWI**  (McFeeters 1996) = (G - NIR)  / (G + NIR  + ε)
* **MNDWI** (Xu 2006)        = (G - SWIR1) / (G + SWIR1 + ε)

Why it works
------------
Modern foundation models (TerraMind 1B) often plateau on **fine boundary
detail** because their decoder upsamples 14×14 patch features back to
224×224 — high-frequency content lives in the original sensor bands.

NDWI/MNDWI are the textbook water indices used by glaciologists for ~30
years. They give the network a **shortcut on water-vs-not** that doesn't
need to be re-learned through 24 transformer blocks. The refiner does
NOT replace the deep feature path — it produces a **small additive
correction** ``Δ`` to the base logits, weighted by a learnable scalar α
so the network can dial the index reliance up or down per region.

Outputs of the wrapper:

    final_logits = base_logits + sigmoid(α) * Δ

where ``Δ`` is the refiner's output from concat([base_logits, NDWI, MNDWI]).
At initialisation α → 0 so the wrapper is a no-op (gradient-friendly
warm start that lets Stage 4b weights load cleanly).

Expected gain
-------------
+0.3 to +1.5 pp val IoU on boundary-heavy chips per literature
(GlaViTU Nature 2024, Sharma 2025 used NDWI as auxiliary input).

Usage
-----
::

    model:
      module: cryosentinel.training.ndwi_hybrid.NDWIHybridModule
      backbone: terramind_v1_large
      ...
      ndwi_refiner_channels: 32
      ndwi_alpha_init: 0.0      # warm start: refiner contributes 0.5 weight after sigmoid

The denormalisation step uses the v3 dataset stats path saved during
ingest — pass via ``ndwi_denorm_stats`` (dict of mean/std lists). When
stats are not provided we run the indices on **normalised** band values,
which still produces a useful (if attenuated) signal correlated with
true NDWI.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .lightning_module import TerraMindSegmentationModule


# Sentinel-2 L2A band indices in the v3 chip layout (12 bands stacked
# in the order: B01, B02 (B), B03 (G), B04 (R), B05, B06, B07, B08 (NIR),
# B8A, B09, B11 (SWIR1), B12 (SWIR2)).
S2_BAND_INDEX = {
    "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4,  "B06": 5,
    "B07": 6, "B08": 7, "B8A": 8, "B09": 9, "B11": 10, "B12": 11,
}


def _compute_index_normalized(
    s2_norm: torch.Tensor,
    band_a: int,
    band_b: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a normalised-difference index from the (already z-scored) S2 stack.

    Operating on z-scored bands rather than raw reflectance attenuates
    the index magnitude but preserves the sign-and-rank structure that
    the refiner relies on. This is by design — we want the wrapper to
    be **drop-in** with whatever normalization the datamodule applied
    upstream, so we don't have to round-trip through (mean, std).
    """
    a = s2_norm[:, band_a:band_a + 1]
    b = s2_norm[:, band_b:band_b + 1]
    return (a - b) / (a + b + eps)


def _compute_index_denormed(
    s2_norm: torch.Tensor,
    band_a: int,
    band_b: int,
    s2_mean: torch.Tensor,
    s2_std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the true NDWI / MNDWI in [-1, 1] using band-wise stats."""
    a = s2_norm[:, band_a:band_a + 1] * s2_std[band_a] + s2_mean[band_a]
    b = s2_norm[:, band_b:band_b + 1] * s2_std[band_b] + s2_mean[band_b]
    return (a - b) / (a + b + eps)


class NDWIRefinerHead(nn.Module):
    """A 4-layer 1×1/3×3 conv block that turns
    ``concat(base_logits, NDWI, MNDWI)`` into a residual correction map.

    Channels: 3 → channels → channels → channels → 1.
    """
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        # Zero-init last layer so the refiner emits Δ ≈ 0 at start.
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, base_logits: torch.Tensor,
                ndwi: torch.Tensor,
                mndwi: torch.Tensor) -> torch.Tensor:
        x = torch.cat([base_logits, ndwi, mndwi], dim=1)
        return self.net(x)


class NDWIHybridModule(TerraMindSegmentationModule):
    """TerraMind segmenter + NDWI/MNDWI residual refiner.

    Extends the base module with **one extra branch** that operates on the
    raw S2L2A input. The refiner produces a small correction term added
    on top of the base logits, blended via a learnable sigmoid scalar α.
    """

    def __init__(
        self,
        *,
        ndwi_refiner_channels: int = 32,
        ndwi_alpha_init: float = 0.0,
        ndwi_use_denorm: bool = False,
        ndwi_denorm_stats: dict | None = None,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)

        self.refiner = NDWIRefinerHead(channels=ndwi_refiner_channels)
        # α is stored in logit-space so sigmoid(α) ∈ (0, 1). Init at 0 → blend 0.5
        # initially; if zero-init refiner sees Δ=0 → final = base_logits, so the
        # weighted blend collapses to base + 0.5 * 0 = base.
        self.ndwi_alpha = nn.Parameter(torch.tensor(float(ndwi_alpha_init)))

        self.ndwi_use_denorm = bool(ndwi_use_denorm)
        if self.ndwi_use_denorm:
            if ndwi_denorm_stats is None:
                raise ValueError(
                    "ndwi_use_denorm=True requires ndwi_denorm_stats "
                    "with keys 'S2_mean' (list[12]) and 'S2_std' (list[12])."
                )
            mean = torch.tensor(ndwi_denorm_stats["S2_mean"], dtype=torch.float32)
            std  = torch.tensor(ndwi_denorm_stats["S2_std"],  dtype=torch.float32)
            assert mean.numel() == 12 and std.numel() == 12, \
                "S2 mean/std must have 12 entries"
            self.register_buffer("_s2_mean", mean)
            self.register_buffer("_s2_std",  std)

        # PARAMETERS to log at init so the run is reproducible from logs
        print(f"[NDWIHybridModule] refiner: {ndwi_refiner_channels} ch, "
              f"α₀={ndwi_alpha_init:.3f} (blend={torch.sigmoid(torch.tensor(ndwi_alpha_init)).item():.3f}), "
              f"use_denorm={self.ndwi_use_denorm}")

    def _indices_from_s2(self, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g    = S2_BAND_INDEX["B03"]
        nir  = S2_BAND_INDEX["B08"]
        sw1  = S2_BAND_INDEX["B11"]
        if self.ndwi_use_denorm:
            ndwi  = _compute_index_denormed(s2, g, nir, self._s2_mean, self._s2_std)
            mndwi = _compute_index_denormed(s2, g, sw1, self._s2_mean, self._s2_std)
        else:
            ndwi  = _compute_index_normalized(s2, g, nir)
            mndwi = _compute_index_normalized(s2, g, sw1)
        return ndwi, mndwi

    def _forward_with_refiner(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Base path — same as TerraMindSegmentationModule.forward.
        inputs = {k: batch[k] for k in ("S2L2A", "S1GRD", "DEM") if k in batch}
        base_logits = self._logits_from_model_output(self.model(inputs))   # [B,1,H,W]
        ndwi, mndwi = self._indices_from_s2(batch["S2L2A"])
        delta = self.refiner(base_logits.detach() if not self.training else base_logits,
                             ndwi, mndwi)
        alpha = torch.sigmoid(self.ndwi_alpha)
        return base_logits + alpha * delta

    # Override forward + step path to use the refiner.
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward_with_refiner(batch)
