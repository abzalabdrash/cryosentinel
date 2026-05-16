"""Smoke test for the LLRD torch.compile prefix fix.

Verifies that build_param_groups_llrd correctly classifies parameters
even when their names carry the ``_orig_mod.`` prefix injected by
``torch.compile(model)``.
"""
from __future__ import annotations

import torch
from torch import nn

from cryosentinel.training.optim import (
    build_param_groups_llrd,
    summarise_param_groups,
    _strip_compile_prefix,
)


def test_strip_compile_prefix_idempotent() -> None:
    assert _strip_compile_prefix("_orig_mod.encoder.encoder.0.attn.weight") == \
        "encoder.encoder.0.attn.weight"
    assert _strip_compile_prefix("encoder.encoder.0.attn.weight") == \
        "encoder.encoder.0.attn.weight"
    assert _strip_compile_prefix("decoder.fpn.0.weight") == "decoder.fpn.0.weight"


class _FakeViTL(nn.Module):
    """Mimics the param-name structure of OptimizedModule(TerraMind ViT-L)."""

    def __init__(self) -> None:
        super().__init__()
        encoder = nn.Module()
        encoder.encoder = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8)) for _ in range(24)
        ])
        encoder.cls_token = nn.Parameter(torch.zeros(1, 1, 8))
        encoder.pos_embed = nn.Parameter(torch.zeros(1, 16, 8))
        encoder.norm = nn.LayerNorm(8)
        decoder = nn.Module()
        decoder.fpn = nn.Linear(8, 1)
        # Register under ``_orig_mod.`` so named_parameters mirrors what
        # torch.compile's OptimizedModule would emit.
        self._orig_mod = nn.ModuleDict({"encoder": encoder, "decoder": decoder})


def test_llrd_groups_compiled_vit_l() -> None:
    m = _FakeViTL()
    groups = build_param_groups_llrd(
        m, base_lr=3e-5, decay=0.75, n_blocks=24, weight_decay=1e-4,
    )
    print(summarise_param_groups(groups))

    group_names = {g["name"].rstrip("_nd").rstrip("_") for g in groups}
    # Cleaner extraction: drop trailing "_nd" suffix.
    group_names = set()
    for g in groups:
        nm = g["name"]
        if nm.endswith("_nd"):
            nm = nm[: -len("_nd")]
        group_names.add(nm)

    expected_blocks = {f"block_{i}" for i in range(24)}
    missing = expected_blocks - group_names
    assert not missing, f"MISSING block groups (LLRD broken): {missing}"
    assert "decoder" in group_names, f"decoder missing: {group_names}"
    # backbone_body holds cls_token / pos_embed / encoder.norm
    assert "backbone_body" in group_names, f"backbone_body missing: {group_names}"


def test_llrd_lr_ordering() -> None:
    m = _FakeViTL()
    groups = build_param_groups_llrd(
        m, base_lr=3e-5, decay=0.75, n_blocks=24, weight_decay=1e-4,
    )

    def _first_lr(prefix: str) -> float:
        for g in groups:
            nm = g["name"]
            if nm.endswith("_nd"):
                nm = nm[: -len("_nd")]
            if nm == prefix:
                return float(g["lr"])
        raise KeyError(prefix)

    lr_b0 = _first_lr("block_0")
    lr_b23 = _first_lr("block_23")
    lr_dec = _first_lr("decoder")
    print(f"block_0={lr_b0:.3e}  block_23={lr_b23:.3e}  decoder={lr_dec:.3e}")
    assert lr_b0 < lr_b23 < lr_dec, "LLRD LR ordering broken"


if __name__ == "__main__":
    test_strip_compile_prefix_idempotent()
    print("PASS test_strip_compile_prefix_idempotent")
    test_llrd_groups_compiled_vit_l()
    print("PASS test_llrd_groups_compiled_vit_l")
    test_llrd_lr_ordering()
    print("PASS test_llrd_lr_ordering")
    print()
    print("ALL TESTS PASSED — LLRD compile-prefix fix verified")
