"""Unit tests for Phase D building blocks: EMA, LLRD, Lovász-Hinge.

These tests run on CPU in a few seconds and use only torch — no GEE,
no cloud runner, no live data. They verify the *contract* of each component
(shape correctness, gradient flow, parameter group LR mapping, EMA
update math). Numerical fidelity vs published implementations is
checked where possible.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
import pytest

from cryosentinel.training.callbacks import EMA
from cryosentinel.training.optim import (
    build_param_groups_llrd,
    summarise_param_groups,
)
from cryosentinel.training.losses import (
    MegaLoss,
    lovasz_hinge_loss,
)


# ─────────────────────────────────────────────────────────────────
# 1. Lovász-Hinge loss
# ─────────────────────────────────────────────────────────────────
class TestLovaszHinge:
    def test_zero_loss_at_perfect_pred(self):
        # Strong positive logits inside lake, strong negative outside →
        # margin errors (1 - logits*sign) are all <= 0 → loss = 0.
        target = torch.zeros(2, 1, 8, 8, dtype=torch.long)
        target[:, :, 2:6, 2:6] = 1
        logits = torch.where(target.bool(), 5.0, -5.0).float()
        loss = lovasz_hinge_loss(logits, target.squeeze(1))
        assert loss.item() < 1e-3

    def test_high_loss_at_inverted_pred(self):
        target = torch.zeros(2, 1, 8, 8, dtype=torch.long)
        target[:, :, 2:6, 2:6] = 1
        logits = torch.where(target.bool(), -5.0, 5.0).float()  # inverted
        loss = lovasz_hinge_loss(logits, target.squeeze(1))
        assert loss.item() > 1.0

    def test_gradient_flows(self):
        target = torch.zeros(1, 1, 16, 16, dtype=torch.long)
        target[:, :, 4:12, 4:12] = 1
        logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
        loss = lovasz_hinge_loss(logits, target.squeeze(1))
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0  # at least some pixel got gradient

    def test_per_image_vs_flat_disagree_on_imbalanced_batch(self):
        """per_image=True averages per-chip Lovász; per_image=False merges all
        valid pixels into one sequence. They should give different values when
        the batch has imbalanced foreground."""
        # Chip 0: empty, chip 1: half-filled
        target = torch.zeros(2, 1, 8, 8, dtype=torch.long)
        target[1, :, :, :4] = 1
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 8, 8)
        per_img = lovasz_hinge_loss(logits, target.squeeze(1), per_image=True)
        flat = lovasz_hinge_loss(logits, target.squeeze(1), per_image=False)
        # They aren't expected to be identical
        assert abs(per_img.item() - flat.item()) > 1e-4

    def test_valid_mask_excludes_pixels(self):
        target = torch.zeros(1, 1, 8, 8, dtype=torch.long)
        target[:, :, 2:6, 2:6] = 1
        logits = torch.full_like(target, -5.0, dtype=torch.float32)  # all wrong inside
        # First disable mask: loss should be high
        unmasked = lovasz_hinge_loss(logits, target.squeeze(1))
        # Now mark inside-lake pixels as invalid → Lovász sees only background
        # which is now correctly predicted → loss → 0
        valid = torch.ones(1, 8, 8, dtype=torch.bool)
        valid[:, 2:6, 2:6] = False
        masked = lovasz_hinge_loss(logits, target.squeeze(1), valid_mask=valid)
        assert masked < unmasked
        assert masked.item() < 1e-3

    def test_megaloss_lovasz_path(self):
        loss_fn = MegaLoss(
            w_bce=0.0, w_focal=0.0, w_dice=0.0, w_tversky=0.0, w_boundary=0.0,
            w_lovasz=1.0,
        )
        target = torch.zeros(2, 1, 8, 8, dtype=torch.long)
        target[:, :, 2:6, 2:6] = 1
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        total, comps = loss_fn(logits, target.squeeze(1))
        assert "loss/lovasz" in comps
        assert comps["loss/lovasz"].item() > 0
        total.backward()
        assert logits.grad is not None

    def test_invalid_logits_shape(self):
        target = torch.zeros(2, 1, 4, 4, dtype=torch.long)
        bad = torch.zeros(2, 3, 4, 4)  # 3 channels, expected 1
        with pytest.raises(ValueError):
            lovasz_hinge_loss(bad, target.squeeze(1))


# ─────────────────────────────────────────────────────────────────
# 2. Layer-wise LR Decay
# ─────────────────────────────────────────────────────────────────
class _MockTerraMind(nn.Module):
    """Minimal stand-in matching the encoder.encoder.{i}.<rest> naming."""
    def __init__(self, n_blocks: int = 4):
        super().__init__()
        self.encoder = nn.Module()
        # Backbone body (positional embedding & cls token)
        self.encoder.cls_token = nn.Parameter(torch.zeros(1, 1, 16))
        self.encoder.pos_embed = nn.Parameter(torch.zeros(1, 65, 16))
        self.encoder.norm = nn.LayerNorm(16)
        # Transformer blocks
        self.encoder.encoder = nn.ModuleList([
            nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16))
            for _ in range(n_blocks)
        ])
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.Linear(8, 1),
        )


class TestLLRD:
    def test_lr_geometric_decay(self):
        model = _MockTerraMind(n_blocks=4)
        groups = build_param_groups_llrd(
            model, base_lr=1e-3, decay=0.5, n_blocks=4,
        )
        # Helper: any group whose name starts with the prefix (decay or _nd)
        def lr_of(prefix: str) -> float:
            for g in groups:
                if g["name"].startswith(prefix):
                    return g["lr"]
            raise AssertionError(f"no group with name prefix {prefix!r}")
        # Decoder must be at base_lr (both decoder and decoder_nd have same LR)
        assert lr_of("decoder") == pytest.approx(1e-3)
        # block_3 (closest to decoder) = base * 0.5^1 = 5e-4
        assert lr_of("block_3") == pytest.approx(1e-3 * 0.5 ** 1)
        # block_0 (deepest) = base * 0.5^4 = 6.25e-5
        assert lr_of("block_0") == pytest.approx(1e-3 * 0.5 ** 4)
        # backbone_body uses decay^(n_blocks+1) = 0.5^5
        # In the mock, every backbone_body param matches a no-decay keyword
        # (cls_token, pos_embed, norm.weight, bias) so the group is
        # named "backbone_body_nd" rather than "backbone_body".
        assert lr_of("backbone_body") == pytest.approx(1e-3 * 0.5 ** 5)

    def test_no_decay_groups_split(self):
        model = _MockTerraMind(n_blocks=4)
        groups = build_param_groups_llrd(model, base_lr=1e-3, weight_decay=0.05)
        names = [g["name"] for g in groups]
        # We expect both decayed and no-decay sub-groups for at least decoder
        decoder_groups = [g for g in groups if g["name"].startswith("decoder")]
        assert len(decoder_groups) >= 2
        wd_values = {g["weight_decay"] for g in decoder_groups}
        assert 0.0 in wd_values
        assert 0.05 in wd_values

    def test_all_trainable_params_assigned(self):
        model = _MockTerraMind(n_blocks=4)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        groups = build_param_groups_llrd(model, base_lr=1e-3)
        n_assigned = sum(len(g["params"]) for g in groups)
        assert n_assigned == n_trainable

    def test_summary_runs(self):
        model = _MockTerraMind(n_blocks=4)
        groups = build_param_groups_llrd(model, base_lr=1e-3)
        s = summarise_param_groups(groups)
        assert "decoder" in s
        assert "block_0" in s

    def test_optim_construction(self):
        model = _MockTerraMind(n_blocks=4)
        groups = build_param_groups_llrd(model, base_lr=1e-3)
        opt = torch.optim.AdamW(groups, lr=1e-3)
        assert len(opt.param_groups) == len(groups)


# ─────────────────────────────────────────────────────────────────
# 3. EMA callback
# ─────────────────────────────────────────────────────────────────
class _ToyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.linear(x).squeeze(-1)
        loss = self.loss_fn(pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.linear(x).squeeze(-1)
        return self.loss_fn(pred, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.5)


def _make_loader(n: int = 32):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = (x.sum(dim=-1)).float()
    return DataLoader(TensorDataset(x, y), batch_size=4)


class TestEMA:
    def test_ema_decays_toward_live(self):
        # After many steps with the same gradient direction, EMA should be a
        # smoothed version of live weights — strictly between init and live.
        ema = EMA(decay=0.5, cpu_shadow=True, apply_at_validation=False)
        module = _ToyModule()
        init_w = module.linear.weight.detach().clone()
        loader = _make_loader(n=64)
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[ema],
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(module, loader)
        live_w = module.linear.weight.detach().clone()
        # Sanity: live moved
        assert not torch.allclose(init_w, live_w, atol=1e-4)
        # EMA shadow exists
        assert ema._shadow is not None
        shadow_w = ema._shadow["linear.weight"]
        # Shadow != live but != init
        assert not torch.allclose(shadow_w, live_w, atol=1e-3)
        assert not torch.allclose(shadow_w, init_w, atol=1e-3)

    def test_invalid_decay(self):
        with pytest.raises(ValueError):
            EMA(decay=1.0)
        with pytest.raises(ValueError):
            EMA(decay=0.0)
        with pytest.raises(ValueError):
            EMA(decay=-0.5)

    def test_state_dict_round_trip(self):
        ema = EMA(decay=0.99)
        module = _ToyModule()
        ema._maybe_init(module)
        state = ema.state_dict()
        ema2 = EMA(decay=0.99)
        ema2.load_state_dict(state)
        assert ema2._initialised == ema._initialised
        assert ema2._shadow is not None
        assert set(ema2._shadow.keys()) == set(ema._shadow.keys())

    def test_swap_in_out_round_trip(self):
        # Swap in changes weights to shadow, swap out restores live.
        ema = EMA(decay=0.5, cpu_shadow=True)
        module = _ToyModule()
        ema._maybe_init(module)
        # Mutate live weights
        with torch.no_grad():
            module.linear.weight.fill_(2.0)
        live_before = module.linear.weight.detach().clone()
        # Mutate shadow to a known value
        ema._shadow["linear.weight"].fill_(0.5)
        ema._swap_in(module)
        assert module.linear.weight.mean().item() == pytest.approx(0.5)
        ema._swap_out(module)
        assert torch.allclose(module.linear.weight, live_before)
