"""CryoSentinel training utilities (losses, TTA, Lightning modules)."""
from .losses import (
    BCEDiceBoundaryLoss,
    MegaLoss,
    boundary_loss,
    dice_loss,
    focal_loss,
    generalized_dice_loss,
    hard_iou,
    lovasz_hinge_loss,
    ohem_pool,
    soft_iou,
    tversky_loss,
)
from .tta import (
    flip_only_tta,
    flip_only_tta_logits,
)
from .lightning_module import (
    TerraMindSegmentationModule,
    DEFAULT_NECKS,
)
from .callbacks import (
    EMA,
    make_swa_callback,
)
from .optim import (
    build_param_groups_llrd,
    summarise_param_groups,
)

__all__ = [
    # Losses
    "BCEDiceBoundaryLoss",
    "MegaLoss",
    "boundary_loss",
    "dice_loss",
    "focal_loss",
    "generalized_dice_loss",
    "hard_iou",
    "lovasz_hinge_loss",
    "ohem_pool",
    "soft_iou",
    "tversky_loss",
    # TTA
    "flip_only_tta",
    "flip_only_tta_logits",
    # Module
    "TerraMindSegmentationModule",
    "DEFAULT_NECKS",
    # Callbacks
    "EMA",
    "make_swa_callback",
    # Optimisers
    "build_param_groups_llrd",
    "summarise_param_groups",
]
