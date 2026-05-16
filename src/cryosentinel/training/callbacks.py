"""Custom Lightning callbacks for the SOTA training recipe.

Two callbacks live here:

* :class:`EMA` вЂ” exponential moving average of model weights, evaluated at
  validation/test time. Standard 0.9999 decay; tracked on a CPU shadow copy
  so we don't compete with the live model for VRAM.

* :class:`StochasticWeightAveraging` is provided by Lightning itself; we
  re-export a pre-configured factory :func:`make_swa_callback` to keep the
  config YAML one-liner.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


# в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# 1. EMA (Exponential Moving Average) callback
# в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
class EMA(Callback):
    """Maintain an EMA copy of the LightningModule weights.

    Standard recipe used by SegFormer / Mask2Former / TerraMind / ConvNeXt:
    after every optimizer step (i.e. after the live weights are updated by
    the optimizer), the EMA weights are pulled toward the live ones with
    decay :math:`\\alpha`:

    .. math::
        w_{\\text{ema}} \\leftarrow \\alpha \\, w_{\\text{ema}} +
                       (1 - \\alpha) \\, w_{\\text{live}}

    During ``validation`` / ``test`` the EMA weights are *swapped in*; on
    the way out the live weights are restored. Checkpoints contain BOTH
    the live and the EMA state-dict so the user can pick at evaluation.

    Parameters
    ----------
    decay : float
        EMA decay (e.g. 0.999, 0.9999). Higher = slower / smoother.
    apply_at_validation : bool
        Swap EMA weights during ``validation_*`` hooks.
    apply_at_test : bool
        Swap EMA weights during ``test_*`` hooks.
    cpu_shadow : bool
        Keep the EMA copy on CPU (default). Trades a small CPU<->GPU
        transfer per step for the ~280 MB VRAM that 100 M-param backbones
        otherwise duplicate.
    skip_buffers : bool
        If true (default), only ``parameters()`` are EMA'd. BatchNorm
        running stats and other buffers are left as-is, mirroring the
        timm / Hugging Face conventions for ViT-style backbones.

    Notes
    -----
    * Only updates **trainable** parameters вЂ” frozen params are skipped.
    * Safe with mixed precision: EMA copy is always float32.
    * Compatible with ``StochasticWeightAveraging``: SWA averages over the
      last K epochs of weights, EMA tracks a continuous moving average.
      Use one or the other (running both wastes compute and the SWA
      checkpoint is what gets evaluated at the end).
    """

    state_key: str = "EMA"

    def __init__(
        self,
        decay: float = 0.9999,
        *,
        apply_at_validation: bool = True,
        apply_at_test: bool = True,
        cpu_shadow: bool = True,
        skip_buffers: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = decay
        self.apply_at_validation = apply_at_validation
        self.apply_at_test = apply_at_test
        self.cpu_shadow = cpu_shadow
        self.skip_buffers = skip_buffers

        # name -> tensor (float32). Built lazily on the first optimizer step
        # so the model has been moved to its final device by then.
        self._shadow: dict[str, torch.Tensor] | None = None
        self._backup: dict[str, torch.Tensor] | None = None
        self._initialised: bool = False

    # в”Ђв”Ђ Lifecycle в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Lazy-init in on_train_batch_end (after the first step) so the live
        # weights have already received any setup fixes (e.g. zero_init).
        self._initialised = False

    def _maybe_init(self, pl_module: LightningModule) -> None:
        if self._initialised:
            return
        shadow: dict[str, torch.Tensor] = {}
        device = torch.device("cpu") if self.cpu_shadow else next(pl_module.parameters()).device
        for name, p in pl_module.named_parameters():
            if not p.requires_grad:
                continue
            shadow[name] = p.detach().to(device=device, dtype=torch.float32).clone()
        self._shadow = shadow
        self._initialised = True

    # в”Ђв”Ђ Update step вЂ” after optimizer.step() в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._maybe_init(pl_module)
        assert self._shadow is not None
        d = self.decay
        for name, p in pl_module.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self._shadow.get(name)
            if shadow is None:
                # New trainable param appeared mid-training (unfreeze schedule);
                # adopt current weights as the EMA seed.
                self._shadow[name] = p.detach().to(
                    device=shadow.device if shadow is not None else
                           (torch.device("cpu") if self.cpu_shadow else p.device),
                    dtype=torch.float32,
                ).clone()
                continue
            live = p.detach().to(device=shadow.device, dtype=torch.float32)
            # in-place: shadow.mul_(d).add_(live * (1 - d))
            shadow.mul_(d).add_(live, alpha=1.0 - d)

    # в”Ђв”Ђ Evaluation: swap EMA in / out в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    def _swap_in(self, pl_module: LightningModule) -> None:
        if self._shadow is None or self._backup is not None:
            return
        backup: dict[str, torch.Tensor] = {}
        for name, p in pl_module.named_parameters():
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            backup[name] = p.detach().clone()
            p.data.copy_(shadow.to(device=p.device, dtype=p.dtype))
        self._backup = backup

    def _swap_out(self, pl_module: LightningModule) -> None:
        if self._backup is None:
            return
        for name, p in pl_module.named_parameters():
            buf = self._backup.get(name)
            if buf is None:
                continue
            p.data.copy_(buf.to(device=p.device, dtype=p.dtype))
        self._backup = None

    def on_validation_start(self, trainer, pl_module):
        if self.apply_at_validation:
            self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if self.apply_at_validation:
            self._swap_out(pl_module)

    def on_test_start(self, trainer, pl_module):
        if self.apply_at_test:
            self._swap_in(pl_module)

    def on_test_end(self, trainer, pl_module):
        if self.apply_at_test:
            self._swap_out(pl_module)

    # в”Ђв”Ђ Checkpoint plumbing в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self._shadow,
            "initialised": self._initialised,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self._shadow = state_dict.get("shadow", None)
        self._initialised = state_dict.get("initialised", False)


# в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# 2. SWA convenience factory
# в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
def make_swa_callback(
    *,
    swa_epoch_start: float = 0.75,
    swa_lrs: float | list[float] | None = 1e-5,
    annealing_epochs: int = 5,
    annealing_strategy: str = "cos",
) -> Callback:
    """Create a Lightning ``StochasticWeightAveraging`` callback.

    Defaults are tuned for the v3 SOTA recipe:
      * Begin SWA averaging at 75 % of total epochs.
      * Use a constant ``swa_lrs`` (1e-5) вЂ” the tail LR keeps weights close
        to the optimum so the average is a real ensemble.
      * 5-epoch cosine anneal at the start of SWA before the new LR kicks in.

    The class itself ships with Lightning >= 1.7. We re-export a factory so
    config YAMLs can construct it via ``_target_: ... .make_swa_callback``.
    """
    from lightning.pytorch.callbacks import StochasticWeightAveraging

    return StochasticWeightAveraging(
        swa_epoch_start=swa_epoch_start,
        swa_lrs=swa_lrs,
        annealing_epochs=annealing_epochs,
        annealing_strategy=annealing_strategy,
    )



__all__ = [
    "EMA",
    "make_swa_callback",
]
