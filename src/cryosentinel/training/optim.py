"""Optimiser-construction helpers for the SOTA training recipe.

Layer-Wise LR Decay (LLRD) — used universally for ViT fine-tuning since
BERT-LARGE 2019. Lower transformer blocks update slowly, higher blocks
update faster, decoder & head update fastest. With a 12-block ViT-B and
decay :math:`\\gamma = 0.75`:

    block 0 ( deepest)    : lr * 0.75^12 = 0.0317  * lr
    block 6 (mid)         : lr * 0.75^6  = 0.178   * lr
    block 11 (closest to  : lr * 0.75^1  = 0.75    * lr
              decoder)
    decoder/head           : lr * 1.0

The function :func:`build_param_groups_llrd` produces the param-group list
that ``torch.optim.AdamW(params, lr=...)`` expects.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional

import re

import torch
from torch import nn


# ──────────────────────────────────────────────────────────────────
#  Layer detection
# ──────────────────────────────────────────────────────────────────
# terratorch TerraMind v1-base wraps blocks under several name patterns
# depending on the version. We accept any of them.
_BLOCK_PATTERNS: tuple[re.Pattern, ...] = (
    # most common — `encoder.encoder.{i}.<rest>` via modal-mim ViT
    re.compile(r"^encoder\.encoder\.(\d+)\."),
    # alternative — `encoder.blocks.{i}.<rest>` (HF ViT style)
    re.compile(r"^encoder\.blocks\.(\d+)\."),
    # multimodal models sometimes stack as `model.encoder.{i}.<rest>`
    re.compile(r"^encoder\.(\d+)\."),
)

# Param name -> "encoder body" if it's part of the backbone but NOT inside
# a numbered transformer block (positional embeddings, patch embedding,
# CLS token, modality embeddings, final layer-norm of the backbone, etc.).
_BACKBONE_BODY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^encoder\.(?:cls_token|pos_embed|patch_embed|modality_embed)"),
    re.compile(r"^encoder\.encoder_embeddings"),
    re.compile(r"^encoder\.embed_tokens"),
    re.compile(r"^encoder\.norm\."),
    re.compile(r"^encoder\.norm$"),
    re.compile(r"^encoder\.[A-Za-z_]+$"),  # bare encoder.<scalar>
)


def _strip_compile_prefix(name: str) -> str:
    """Strip torch.compile's ``_orig_mod.`` prefix when present.

    When a model is wrapped with ``torch.compile(model)``, the resulting
    ``OptimizedModule`` registers the original module as ``_orig_mod``,
    which causes ``named_parameters()`` to emit names like
    ``_orig_mod.encoder.encoder.0.attn.qkv.weight`` instead of the bare
    ``encoder.encoder.0.attn.qkv.weight`` our regex patterns expect.
    Without this strip, every parameter falls through the encoder regex
    matches and lands in the "decoder" group at uniform LR — silently
    disabling LLRD. (Observed 2026-05-10 on acc6/acc7 v3 pretrain runs:
    only 2 param groups appeared in summarise_param_groups output where
    24 block groups were expected.)

    This helper is idempotent — applying it to an already-bare name is
    a no-op — so it is safe to call unconditionally before any pattern
    check, regardless of whether the model was compiled.
    """
    if name.startswith("_orig_mod."):
        return name[len("_orig_mod."):]
    return name


def _block_index_for_param(name: str) -> Optional[int]:
    """Return the 0-indexed block number, or None if param isn't in a block."""
    name = _strip_compile_prefix(name)
    for pat in _BLOCK_PATTERNS:
        m = pat.match(name)
        if m:
            return int(m.group(1))
    return None


def _is_backbone_body(name: str) -> bool:
    """Param is part of the backbone but not a transformer block."""
    name = _strip_compile_prefix(name)
    return any(p.match(name) for p in _BACKBONE_BODY_PATTERNS)


def _is_decoder(name: str) -> bool:
    """Param is part of the decoder / segmentation head (not encoder)."""
    name = _strip_compile_prefix(name)
    return not name.startswith("encoder.")


# ──────────────────────────────────────────────────────────────────
#  Param-group builder
# ──────────────────────────────────────────────────────────────────
def build_param_groups_llrd(
    model: nn.Module,
    *,
    base_lr: float,
    backbone_lr_mult: float = 1.0,
    decay: float = 0.75,
    n_blocks: int = 12,
    weight_decay: float = 1e-4,
    no_decay_keywords: tuple[str, ...] = (
        "bias", "norm.weight", "ln.weight",
        "pos_embed", "cls_token", "modality_embed",
    ),
) -> list[dict]:
    """Build LLRD param-groups for ``torch.optim.AdamW``.

    LR schedule (with the decoder LR being the canonical ``base_lr``):
        block i  (0..n_blocks-1) → ``base_lr * backbone_lr_mult * decay^(n_blocks - i)``
        backbone body            → ``base_lr * backbone_lr_mult * decay^(n_blocks + 1)``
        decoder/head             → ``base_lr``  (no decay)

    Note ``decay^(n_blocks - i)`` makes block 0 (deepest, slowest) the
    smallest LR and block ``n_blocks-1`` (just before the decoder) the
    fastest among the encoder blocks. This is the BERT-style convention
    used by ELECTRA, T5, ViT-pretrain and TerraMind itself.

    Parameters
    ----------
    model : nn.Module
        The full segmentation model (encoder + necks + decoder).
    base_lr : float
        The decoder LR. This is what most schedulers will scale.
    backbone_lr_mult : float
        Multiplier applied to *every* backbone group on top of LLRD —
        kept here for backward-compat with the existing ``backbone_lr_mult``
        knob in the LightningModule. Use 1.0 for pure LLRD.
    decay : float
        The geometric LR decay between successive transformer blocks.
        Common values: 0.65, 0.75, 0.8, 0.9.
    n_blocks : int
        Number of transformer blocks in the backbone (12 for ViT-B).
    weight_decay : float
        Default WD applied to every group except those whose param name
        matches one of ``no_decay_keywords`` (norms, biases, embeddings).
    no_decay_keywords : tuple[str, ...]
        Substrings — if any appears in the parameter name, weight decay
        is set to 0 for that group.

    Returns
    -------
    list[dict]
        Param groups suitable for ``torch.optim.AdamW(groups, lr=base_lr)``.
        Each dict has ``params``, ``lr``, ``weight_decay`` and a debug
        ``name`` for logging.
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"decay must be in (0, 1]; got {decay}")
    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be > 0; got {n_blocks}")

    # Canonical LR per group key. Group keys:
    #   "decoder"
    #   "backbone_body"
    #   "block_{i}"
    lr_for: dict[str, float] = {}
    lr_for["decoder"] = base_lr
    lr_for["backbone_body"] = base_lr * backbone_lr_mult * (decay ** (n_blocks + 1))
    for i in range(n_blocks):
        lr_for[f"block_{i}"] = base_lr * backbone_lr_mult * (decay ** (n_blocks - i))

    # Walk parameters and assign each to a group.
    grouped: "OrderedDict[tuple[str, bool], list[nn.Parameter]]" = OrderedDict()

    def _assign(group_name: str, no_decay: bool, p: nn.Parameter) -> None:
        key = (group_name, no_decay)
        grouped.setdefault(key, []).append(p)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Strip ``_orig_mod.`` prefix once so the raw ``startswith`` check
        # below in the backbone-body branch matches when the model was
        # wrapped by torch.compile. The helper functions also strip
        # internally (idempotent) — this is belt-and-suspenders so the
        # outer raw check is consistent with the helpers.
        clean_name = _strip_compile_prefix(name)
        nd_match = any(kw in clean_name for kw in no_decay_keywords)

        block_idx = _block_index_for_param(clean_name)
        if block_idx is not None:
            if block_idx >= n_blocks:
                raise RuntimeError(
                    f"Param {name!r} has block index {block_idx} >= n_blocks={n_blocks}. "
                    f"Increase n_blocks to match the actual ViT depth."
                )
            _assign(f"block_{block_idx}", nd_match, p)
        elif _is_decoder(clean_name):
            _assign("decoder", nd_match, p)
        elif _is_backbone_body(clean_name) or clean_name.startswith("encoder."):
            _assign("backbone_body", nd_match, p)
        else:  # truly unknown — be conservative, treat as decoder
            _assign("decoder", nd_match, p)

    out: list[dict] = []
    for (group_name, no_decay), params in grouped.items():
        if not params:
            continue
        out.append({
            "params": params,
            "lr": lr_for[group_name],
            "weight_decay": 0.0 if no_decay else weight_decay,
            "name": f"{group_name}{'_nd' if no_decay else ''}",
        })
    return out


# ──────────────────────────────────────────────────────────────────
#  Quick-validate helper — useful for tests / sanity checks
# ──────────────────────────────────────────────────────────────────
def summarise_param_groups(groups: Iterable[dict]) -> str:
    """Pretty-print a param-group list for log output."""
    lines = ["param groups (LLRD):"]
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        lines.append(
            f"  {g['name']:24s}  lr={g['lr']:.3e}  wd={g['weight_decay']:.1e}  "
            f"n_tensors={len(g['params']):3d}  n_params={n_params:>10_}"
        )
    return "\n".join(lines)


__all__ = [
    "build_param_groups_llrd",
    "summarise_param_groups",
]
