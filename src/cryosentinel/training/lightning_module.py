"""Lightning module for TerraMind multi-modal glacial-lake segmentation.

Architecture (configurable via ``backbone`` arg)
------------------------------------------------
* **Backbone**: ``terramind_v1_large`` by default (~300 M, ViT-L/16, 24 layers,
  dim 1024). Other supported values: ``terramind_v1_base`` (87.9 M, ViT-B/16,
  12 layers), ``terramind_v1_small`` (~22 M, ViT-S/16, 12 layers).
  Per the TerraMind paper (arxiv 2504.11171, ICCV 2025), the **large** model
  outperforms ``base`` by **+5 pp avg mIoU on multimodal datasets** and beats
  every other geospatial foundation model on the PANGAEA water-body benchmark.
* **Necks** auto-adapt to backbone depth: 4 evenly-spaced taps
  (e.g. ``[2, 5, 8, 11]`` for ViT-B; ``[5, 11, 17, 23]`` for ViT-L) → reshape
  → learned-interpolate pyramid.
* **Decoder**: terratorch ``UperNetDecoder`` (preferred for ViT-L) or
  ``UNetDecoder`` (legacy ViT-B baseline).
* **Head**: 1 × 1 conv → 1 logit per pixel.

Loss options
------------
* ``loss_type="bce_dice_boundary"`` (default, original): BCE + Dice + Boundary.
* ``loss_type="mega"`` (SOTA): BCE + Focal(γ=2) + Dice + Tversky(β=0.7) + Boundary.
  Use with ``valid_mask`` in batch to exclude nodata pixels from all components.

Metrics
-------
* ``val/iou``      — **hard IoU** at threshold 0.5. This is what checkpoints are
  monitored on. Lower than soft IoU but honest — matches published numbers.
* ``val/soft_iou`` — soft (probabilistic) IoU; used to track training convergence.

Training-friendly features
--------------------------
* Two-group AdamW (decoder + backbone) with standard ``betas=(0.9, 0.999)``.
  Pre-v3 runs used ``(0.9, 0.95)`` (LLaMA recipe); switched to the ViT-seg
  standard for stability on the boundary-heavy MegaLoss.
* Cosine schedule with linear warm-up.
* Optional layer-wise freezing for low-data fine-tuning.
* Optional ``torch.compile(mode="default")`` (opt-in via ``compile_model``).
  On H100 + PyTorch 2.5 + ViT-L this typically saves ~25% wall-clock; we
  keep the default ``False`` because compile failures on dynamic-shape
  back-bones still show up occasionally — flip on per-config.
* TTA semantics (post-PR-1): val is **fast** (no TTA by default,
  ``val_tta=False``), test runs **flip-only TTA** (``test_tta=True``).
  Pre-PR-1 default was the opposite which doubled val time per epoch
  without helping checkpoint selection.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

try:
    import lightning.pytorch as pl              # type: ignore
except ImportError:
    import pytorch_lightning as pl              # type: ignore

from terratorch.models import EncoderDecoderFactory

from .losses import (
    BCEDiceBoundaryLoss,
    MegaLoss,
    hard_iou,
    soft_iou,
)
from .tta import flip_only_tta_logits


# ──────────────────────────────────────────────────────────────────────
#  Default architecture pieces
# ──────────────────────────────────────────────────────────────────────
# Default neck taps for the most common ViT depths used in TerraMind v1.
# Indices are spread across the encoder so the multi-scale FPN sees features
# from shallow-to-deep blocks. Picked at the 1/4, 1/2, 3/4, last quartile.
DEFAULT_NECKS_BY_DEPTH: dict[int, list[int]] = {
    12: [2, 5, 8, 11],     # ViT-B/16 (terramind_v1_base, terramind_v1_small)
    24: [5, 11, 17, 23],   # ViT-L/16 (terramind_v1_large)
    32: [7, 15, 23, 31],   # placeholder for hypothetical 32-block variant
}


def _depth_for_backbone(backbone: str) -> int:
    """Return the encoder depth (number of transformer blocks) for a known backbone."""
    b = backbone.lower().strip()
    if "large" in b:
        return 24
    if "huge" in b:
        return 32
    return 12  # base / small / default


def _default_necks_for(backbone: str) -> list[dict]:
    indices = DEFAULT_NECKS_BY_DEPTH[_depth_for_backbone(backbone)]
    return [
        {"name": "SelectIndices", "indices": indices},
        {"name": "ReshapeTokensToImage", "remove_cls_token": False},
        {"name": "LearnedInterpolateToPyramidal"},
    ]


# Kept for backward-compat with v2 baselines that imported ``DEFAULT_NECKS``
# expecting the ViT-B layout. New code should use :func:`_default_necks_for`.
DEFAULT_NECKS: list[dict] = _default_necks_for("terramind_v1_base")


def _build_terramind_segmenter(
    *,
    backbone: str,
    modalities: Sequence[str],
    pretrained: bool,
    decoder: str,
    num_classes: int,
    decoder_channels: Sequence[int] | None = None,
    decoder_kwargs: dict | None = None,
    necks: Sequence[dict] | None = None,
) -> nn.Module:
    """Build the TerraMind encoder + decoder via terratorch.

    Two decoder configuration paths are supported:

    1. **Legacy UNet path** (default) — pass ``decoder="UNetDecoder"`` and
       ``decoder_channels=[256, 128, 64, 32]``. The list of per-stage
       channels is forwarded as the standard UNetDecoder argument.

    2. **Extensible path** — pass ``decoder_kwargs={...}`` containing the
       decoder-specific arguments. Required for ``decoder="UperNetDecoder"``
       which takes ``decoder_channels=<int>`` (single value) plus
       ``decoder_scale_modules: bool``.

    Mixing both is not allowed: ``decoder_kwargs`` always overrides the
    legacy ``decoder_channels`` if provided.
    """
    factory = EncoderDecoderFactory()
    factory_kwargs: dict = dict(
        task="segmentation",
        backbone=backbone,
        backbone_pretrained=pretrained,
        backbone_modalities=list(modalities),
        decoder=decoder,
        necks=list(necks if necks is not None else _default_necks_for(backbone)),
        num_classes=num_classes,
    )
    if decoder_kwargs:
        factory_kwargs.update(decoder_kwargs)
    elif decoder_channels is not None:
        factory_kwargs["decoder_channels"] = list(decoder_channels)
    return factory.build_model(**factory_kwargs)


# ──────────────────────────────────────────────────────────────────────
#  LightningModule
# ──────────────────────────────────────────────────────────────────────
class TerraMindSegmentationModule(pl.LightningModule):
    """Binary segmentation of glacial lakes with a TerraMind backbone."""

    def __init__(
        self,
        *,
        backbone: str = "terramind_v1_large",
        modalities: Sequence[str] = ("S2L2A", "S1GRD", "DEM"),
        decoder: str = "UperNetDecoder",
        decoder_channels: Sequence[int] | None = None,
        decoder_kwargs: dict | None = None,
        backbone_pretrained: bool = True,
        freeze_backbone_layers: int = 0,
        # Optimisation
        lr: float = 1e-4,
        backbone_lr_mult: float = 0.1,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        max_steps: int = 20_000,
        # Loss — shared
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_boundary: float = 0.5,
        pos_weight_max: float = 200.0,
        ignore_index: int | None = None,
        loss_type: str = "bce_dice_boundary",   # "bce_dice_boundary" | "mega"
        # MegaLoss extras (only used when loss_type="mega")
        w_focal: float = 1.0,
        w_tversky: float = 0.5,
        w_lovasz: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        lovasz_per_image: bool = True,
        # PR-2 (Phase D mega-plan, May 2026) — MegaLoss extras off by default
        # for v2 baseline reproducibility; v3 SOTA configs flip them on.
        # See losses.MegaLoss docstring for the full rationale.
        #   label_smoothing : A3, GlaViTU Nature 2024 — BCE-only soft targets.
        #   ohem_keep_ratio : A7, Shrivastava CVPR 2016 — top-K hardest pixels.
        #   ohem_min_kept   : floor on K so small batches aren't reduced to noise.
        #   dice_variant    : B5, Sudre MICCAI 2017 — "flat" or "generalized".
        label_smoothing: float = 0.0,
        ohem_keep_ratio: float = 1.0,
        ohem_min_kept: int = 1024,
        dice_variant: str = "flat",
        # Layer-wise LR Decay (Phase D.2) — applied when llrd_decay > 0
        # ``llrd_n_blocks`` defaults to ``None`` and is auto-resolved from the
        # backbone (12 for ViT-B, 24 for ViT-L). Pass an explicit int to
        # override (e.g. for a custom backbone).
        llrd_decay: float = 0.0,
        llrd_n_blocks: int | None = None,
        # TTA — see module docstring. Defaults flipped in PR-1 (v3 SOTA):
        # val is fast (no TTA), test gets the +0.03 IoU TTA boost.
        val_tta: bool = False,
        test_tta: bool = True,
        # AdamW optimizer betas. Standard ViT-seg recipe is (0.9, 0.999).
        # Some legacy configs use the LLaMA recipe (0.9, 0.95) — keep this
        # exposed so we can A/B-test if needed.
        adam_betas: tuple[float, float] = (0.9, 0.999),
        # PyTorch 2.x graph compilation. Off by default (defensive — terratorch
        # backbones occasionally hit dynamic-shape issues with inductor); the
        # v3 configs flip it on explicitly. When enabled, we wrap ``self.model``
        # in ``torch.compile(mode="default")`` and silently fall back to eager
        # if compile fails (the failure does not abort training).
        compile_model: bool = False,
        # Misc
        log_predictions_every_n_steps: int = 0,
    ) -> None:
        super().__init__()
        # Auto-resolve LLRD depth from the backbone if not explicitly set.
        if llrd_n_blocks is None:
            llrd_n_blocks = _depth_for_backbone(backbone)

        # Default decoder_channels depends on which decoder is used.
        # * UNetDecoder expects a list of per-stage channels (4 stages).
        # * UperNetDecoder takes a single int via ``decoder_kwargs``.
        if decoder_channels is None and decoder.lower().strip() == "unetdecoder":
            decoder_channels = (256, 128, 64, 32)

        self.save_hyperparameters(ignore=[])

        self.model = _build_terramind_segmenter(
            backbone=backbone,
            modalities=modalities,
            pretrained=backbone_pretrained,
            decoder=decoder,
            decoder_channels=decoder_channels,
            decoder_kwargs=decoder_kwargs,
            num_classes=1,
        )

        if freeze_backbone_layers > 0:
            self._freeze_encoder_first_n(freeze_backbone_layers)

        # Optional torch.compile — must run BEFORE optimizer construction
        # (which Lightning does after ``__init__`` returns), so that the
        # optimizer sees the compiled forward graph.
        if compile_model:
            self._maybe_compile_model()

        loss_type = loss_type.lower().strip()
        if loss_type == "mega":
            self.loss_fn: nn.Module = MegaLoss(
                w_bce=w_bce, w_focal=w_focal, w_dice=w_dice,
                w_tversky=w_tversky, w_boundary=w_boundary, w_lovasz=w_lovasz,
                focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
                lovasz_per_image=lovasz_per_image,
                pos_weight_max=pos_weight_max,
                ignore_index=ignore_index,
                # PR-2 — cleanly forwarded so legacy configs keep zero-impact
                # defaults while v3 configs activate the new behaviours.
                label_smoothing=label_smoothing,
                ohem_keep_ratio=ohem_keep_ratio,
                ohem_min_kept=ohem_min_kept,
                dice_variant=dice_variant,
            )
        else:
            if loss_type not in ("bce_dice_boundary", "bce"):
                raise ValueError(
                    f"Unknown loss_type={loss_type!r}. "
                    "Choose 'bce_dice_boundary' or 'mega'."
                )
            self.loss_fn = BCEDiceBoundaryLoss(
                w_bce=w_bce, w_dice=w_dice, w_boundary=w_boundary,
                pos_weight_max=pos_weight_max,
                ignore_index=ignore_index,
            )

    # ── Helpers ────────────────────────────────────────────────────────
    def _maybe_compile_model(self) -> None:
        """Wrap ``self.model`` in :func:`torch.compile` when safe.

        Conditions checked:
        * CUDA is available (compile on CPU buys us nothing for ViT-L).
        * PyTorch is at least ``2.5`` (older versions had unstable compile
          for ViT backbones with dynamic image sizes).

        On compile failure we **fall back to eager** rather than raising —
        a missed speed-up is preferable to a crashed training run, and the
        fall-back is logged loudly for the training logs.
        """
        from importlib.metadata import PackageNotFoundError, version as _pkg_ver

        if not torch.cuda.is_available():
            print("[TerraMind] compile_model=True but no CUDA — staying in eager mode")
            return
        try:
            major, minor, *_ = _pkg_ver("torch").split(".")
            ok_version = int(major) > 2 or (int(major) == 2 and int(minor) >= 5)
        except (PackageNotFoundError, ValueError):
            ok_version = False
        if not ok_version:
            print("[TerraMind] compile_model=True but PyTorch < 2.5 — staying in eager mode")
            return
        try:
            self.model = torch.compile(self.model, mode="default")
            print("[TerraMind] torch.compile(mode='default') enabled — expect ~25% speed-up on H100")
        except Exception as e:                              # pragma: no cover
            print(f"[TerraMind] torch.compile failed, falling back to eager: {e}")

    def _freeze_encoder_first_n(self, n: int) -> None:
        """Freeze first ``n`` transformer blocks and positional embeddings.

        We accept three name patterns the various TerraTorch / TerraMind
        releases have used over the years (see :mod:`.optim` for details).
        """
        import re
        block_patterns = [
            re.compile(r"^encoder\.encoder\.(\d+)\."),
            re.compile(r"^encoder\.blocks\.(\d+)\."),
            re.compile(r"^encoder\.(\d+)\."),
        ]
        frozen = 0
        for name, p in self.model.named_parameters():
            for pat in block_patterns:
                m = pat.match(name)
                if m and int(m.group(1)) < n:
                    p.requires_grad = False
                    frozen += 1
                    break
        if n > 0:
            for name, p in self.model.named_parameters():
                if "encoder_embeddings" in name:
                    p.requires_grad = False
        print(f"[TerraMindSegmentationModule] froze {frozen} encoder tensors "
              f"(first {n} transformer blocks)")

    # ── Forward ────────────────────────────────────────────────────────
    def _logits_from_model_output(self, out) -> torch.Tensor:
        """Extract ``[B, 1, H, W]`` logits from a terratorch ``ModelOutput``."""
        x = out.output if hasattr(out, "output") else out
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return x

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: batch[k] for k in ("S2L2A", "S1GRD", "DEM") if k in batch}
        return self._logits_from_model_output(self.model(inputs))

    # ── Step (shared by train / val / test) ────────────────────────────
    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        target     = batch["mask"]                              # [B, H, W] long
        valid_mask = batch.get("valid_mask")                    # [B, H, W] bool, or None

        # PR-1 TTA semantics: opt-in per stage.
        #   * train  → never TTA (forward is in the gradient path)
        #   * val    → TTA only if ``val_tta`` is True (default False, fast val)
        #   * test   → TTA only if ``test_tta`` is True (default True, +~0.03 IoU)
        use_tta = (
            (stage == "val"  and self.hparams.val_tta) or
            (stage == "test" and self.hparams.test_tta)
        )
        if use_tta:
            inputs = {k: batch[k] for k in ("S2L2A", "S1GRD", "DEM") if k in batch}
            logits = flip_only_tta_logits(
                self.model, inputs,
                forward_fn=lambda m, x: self._logits_from_model_output(m(x)),
            )
        else:
            logits = self(batch)

        loss, comps = self.loss_fn(logits, target, valid_mask=valid_mask)

        # Hard IoU (threshold 0.5) — used for checkpointing, honest metric
        iou_hard = hard_iou(logits, target, valid_mask=valid_mask)
        # Soft IoU — tracks training convergence (optimistic)
        iou_soft = soft_iou(logits, target, valid_mask=valid_mask)

        bs = target.size(0)
        on_step = stage == "train"
        self.log(f"{stage}/loss",     loss,     on_step=on_step, on_epoch=True,
                 prog_bar=True,  batch_size=bs)
        self.log(f"{stage}/iou",      iou_hard, on_step=False,   on_epoch=True,
                 prog_bar=True,  batch_size=bs)
        self.log(f"{stage}/soft_iou", iou_soft, on_step=False,   on_epoch=True,
                 prog_bar=False, batch_size=bs)
        for k, v in comps.items():
            metric = k.split("/", 1)[-1]
            self.log(f"{stage}/{metric}", v, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):   return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def test_step(self, batch, batch_idx):       return self._step(batch, "test")

    # ── Optimiser & schedule ───────────────────────────────────────────
    def configure_optimizers(self):
        # Two paths:
        #   (1) llrd_decay > 0  → Layer-wise LR Decay (BERT/ViT-style).
        #       The decoder stays at ``lr``, transformer blocks use
        #       geometrically smaller LRs the deeper they are. This is
        #       the recommended path for v3 SOTA training.
        #   (2) llrd_decay == 0 → legacy 2-group setup (decoder + backbone)
        #       kept for backward compatibility with v2 baselines.
        if float(self.hparams.llrd_decay) > 0.0:
            from .optim import build_param_groups_llrd, summarise_param_groups
            param_groups = build_param_groups_llrd(
                self.model,
                base_lr=self.hparams.lr,
                backbone_lr_mult=self.hparams.backbone_lr_mult,
                decay=self.hparams.llrd_decay,
                n_blocks=self.hparams.llrd_n_blocks,
                weight_decay=self.hparams.weight_decay,
            )
            print(summarise_param_groups(param_groups))
        else:
            backbone_params: list[nn.Parameter] = []
            decoder_params:  list[nn.Parameter] = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("encoder."):
                    backbone_params.append(p)
                else:
                    decoder_params.append(p)

            param_groups = []
            if backbone_params:
                param_groups.append({
                    "params": backbone_params,
                    "lr": self.hparams.lr * self.hparams.backbone_lr_mult,
                    "name": "backbone",
                })
            if decoder_params:
                param_groups.append({
                    "params": decoder_params,
                    "lr": self.hparams.lr,
                    "name": "decoder",
                })

        # AdamW betas: ViT-seg standard is (0.9, 0.999); kept configurable so we
        # can A/B against the legacy (0.9, 0.95) LLaMA recipe if needed.
        betas = tuple(self.hparams.adam_betas)
        if len(betas) != 2:
            raise ValueError(f"adam_betas must be a 2-tuple, got {betas!r}")
        optim = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
            betas=betas,
        )

        warmup = max(1, int(self.hparams.warmup_steps))
        total  = max(warmup + 1, int(self.hparams.max_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / float(warmup)
            progress = (step - warmup) / float(total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


# ──────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    for loss_type in ("bce_dice_boundary", "mega"):
        print(f"\n--- loss_type={loss_type} ---")
        mod = TerraMindSegmentationModule(
            backbone="terramind_v1_base",        # smoke test uses base for speed
            modalities=("S2L2A", "S1GRD", "DEM"),
            decoder="UNetDecoder",
            backbone_pretrained=False,
            max_steps=100,
            warmup_steps=10,
            loss_type=loss_type,
        )
        B = 2
        batch = {
            "S2L2A":      torch.randn(B, 12, 224, 224),
            "S1GRD":      torch.randn(B,  2, 224, 224),
            "DEM":        torch.randn(B,  1, 224, 224),
            "mask":       (torch.rand(B, 224, 224) > 0.95).long(),
            "valid_mask": torch.ones(B, 224, 224, dtype=torch.bool),
        }
        print(f"  Trainable: {sum(p.numel() for p in mod.parameters() if p.requires_grad)/1e6:.1f}M")
        loss = mod.training_step(batch, 0)
        print(f"  train_step loss = {loss.item():.4f}")
    print("\nOK — both loss types run end-to-end.")
