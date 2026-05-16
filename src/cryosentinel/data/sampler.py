"""Weighted samplers for class-imbalanced glacial-lake training.

The default ``MultiModalChipDataset`` is heavily imbalanced — roughly 95 %
of chips kept by the v3 ingest are positive (water_frac ≥ 0.2 %) and 5 %
are random hard-negatives. With uniform sampling, gradient steps see
almost no negatives → the model learns to predict "always lake" and IoU
plateaus around the prior.

:class:`HardNegativeWeightedSampler` rebalances by giving each chip a
weight equal to ``class_weight[is_positive]`` and uses
``torch.utils.data.WeightedRandomSampler`` semantics. The default
``positive_to_negative_ratio = 3.0`` means each gradient step sees on
average 3 positives per 1 negative — a good middle ground that keeps
the model's positive-class confidence calibrated while still learning
clear water/non-water boundaries.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class HardNegativeWeightedSampler(Sampler[int]):
    """Weighted sampler with explicit positive / hard-negative ratio.

    Parameters
    ----------
    is_positive : Sequence[bool]
        Per-chip flag, same length as the dataset. Typically derived from
        ``water_frac >= threshold`` at index-build time.
    positive_to_negative_ratio : float
        Desired (sampling) ratio of positives to negatives in expectation.
        ``3.0`` => 75 % positives, 25 % negatives per epoch.
    num_samples : int | None
        Number of samples drawn per epoch. Defaults to ``len(is_positive)``.
    replacement : bool
        Whether sampling is with replacement. Default ``True`` (standard
        for weighted samplers; required when many chips have weight 0).
    generator : torch.Generator | None
        Random source for reproducibility.
    """

    def __init__(
        self,
        is_positive: Sequence[bool],
        *,
        positive_to_negative_ratio: float = 3.0,
        num_samples: int | None = None,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        # NOTE: torch.utils.data.Sampler in PyTorch ≥ 2.x has no
        # ``__init__``; calling ``super().__init__(data_source=...)``
        # forwards to ``object.__init__`` and raises a TypeError.
        super().__init__()
        if positive_to_negative_ratio <= 0:
            raise ValueError("positive_to_negative_ratio must be > 0")

        is_positive_t = torch.as_tensor(list(is_positive), dtype=torch.bool)
        n_pos = int(is_positive_t.sum())
        n_neg = int((~is_positive_t).sum())
        n_total = is_positive_t.numel()
        if n_total == 0:
            raise ValueError("is_positive is empty")
        if n_pos == 0 or n_neg == 0:
            # Degenerate dataset → fall back to uniform weights so we don't
            # divide by zero. The sampler still works.
            self._weights = torch.ones(n_total, dtype=torch.double)
        else:
            ratio = float(positive_to_negative_ratio)
            # Per-class weights such that p(pos)/p(neg) == ratio
            #   p(pos) = ratio * w_pos / Z      with each pos having weight w_pos
            #   total positive prob = n_pos * w_pos = ratio * total negative prob
            w_pos = ratio / n_pos
            w_neg = 1.0 / n_neg
            weights = torch.where(
                is_positive_t,
                torch.full((n_total,), w_pos, dtype=torch.double),
                torch.full((n_total,), w_neg, dtype=torch.double),
            )
            self._weights = weights

        self._num_samples = num_samples if num_samples is not None else n_total
        self._replacement = replacement
        self._generator = generator
        self._n_pos = n_pos
        self._n_neg = n_neg

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self._weights,
            self._num_samples,
            replacement=self._replacement,
            generator=self._generator,
        )
        return iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self._num_samples

    @property
    def n_positive(self) -> int:
        return self._n_pos

    @property
    def n_negative(self) -> int:
        return self._n_neg


__all__ = ["HardNegativeWeightedSampler"]
