# Method

This document covers the architecture, the training procedure, the eleven configuration corrections that drove the v1 → v2 improvement, and the rationale for each non-obvious choice. It is meant to be read together with the configs at `configs/terramind_v3_pretrain_v2.yaml` and `configs/terramind_v3_finetune_almaty_v2.yaml`, which contain the same information in inline comments.

## 1. Architecture

The encoder is **TerraMind 1.0 Large** (Jakubik et al., 2025). It is a 1.1 B-parameter dual-scale transformer encoder-decoder pretrained on 9 M spatiotemporally aligned multimodal samples (500 B tokens) from the TerraMesh dataset by IBM Research, the European Space Agency's Φ-lab, and the FAST-EO project, released under Apache 2.0 at `ibm-esa-geospatial/TerraMind-1.0-large`. We use only the encoder; the original generative decoder is discarded.

The decoder is a **UperNet head** (Xiao et al., 2018) with channel sequence 256 → 128 → 64 → 32 and a `LearnedInterpolateToPyramidal` neck that converts TerraMind's flat patch-token output into the multi-scale pyramidal feature map UperNet expects. The head ends in a single 1 × 1 convolution producing a logit map at the input resolution.

Inputs are three modalities, processed by TerraMind's modality-specific patch embedders:

- **Sentinel-2 L2A**: 12 bands (B01–B12 minus B10), uint16 reflectance.
- **Sentinel-1 GRD**: 2 bands (VV and VH polarisations), float32 backscatter in dB.
- **Copernicus DEM 30 m**: 1 band, int16 elevation in metres.

All three modalities are co-registered to a 224 × 224 chip at 10 m/pixel via re-projection to the appropriate UTM zone (43 N–46 N depending on chip longitude). The Sentinel-1 backscatter is computed using the IBM-published TerraMesh recipe (multi-look 5 × 1, Lee speckle filter, terrain correction). The DEM is bilinearly resampled from its native 30 m grid to 10 m to match the Sentinel-2 reference grid.

Per-band normalisation statistics are computed via Welford's algorithm on the train split of the multimodal chips v3 dataset, stored in `dataset_stats_v3.json`, and applied at the dataloader level. The exact values are listed in `DATA.md`.

## 2. Loss

The loss is a weighted sum of six terms ("MEGA" loss) plus OHEM hard-negative mining:

```
L = w_BCE · BCE(pos_weight=100) +
    w_Dice · Dice_flat +
    w_Lovász · Lovász_softmax(per_image=True) +
    w_Tversky · Tversky(α=0.25, β=0.75) +
    w_Focal · Focal(α=0.25, γ=2.0) +
    w_Boundary · Boundary
```

with weights (`w_BCE`, `w_Dice`, `w_Lovász`, `w_Tversky`, `w_Focal`, `w_Boundary`) = (0.5, 1.0, 0.3, 0.5, 1.0, 0.5).

The choices, with citations:

- **`pos_weight = 100`** matches the empirical positive-to-negative pixel ratio (~ 1 : 100 on our finetune dataset). Higher values over-penalise false negatives and inflate BCE noise; we found 200 to be measurably worse.
- **Flat Dice rather than generalised Dice (Sudre et al., 2017)**. Generalised Dice is designed for `K ≥ 3` classes with extreme inter-class imbalance. For our binary 1 % water problem, the inverse-frequency weighting produced a degenerate `val/dice ≈ 0.999` even on perfect chips, which we attribute to the interaction with our class imbalance. Flat Dice is the better-conditioned choice here.
- **Lovász softmax** (Berman et al., 2018) with `per_image = True` provides a direct surrogate for the IoU at chip granularity, which empirically improves the global-IoU metric we report.
- **Asymmetric Tversky (α = 0.25, β = 0.75)** (Salehi et al., 2017) up-weights false negatives over false positives, matching the operational use case (a missed lake is worse than a slightly oversized lake).
- **Focal (α = 0.25, γ = 2.0)** (Lin et al., 2017) helps with the easy-example dominance that BCE alone has trouble with on extreme imbalance.
- **Boundary loss** is a soft-edge term that reduces the staircase artefacts UperNet's nearest-neighbour upsampling can produce on lake shorelines.

OHEM hard-negative mining (Shrivastava et al., 2016) keeps the top-50 % of pixel-level losses with a floor of 4096 pixels per chip. Without OHEM, the loss is dominated by the easy non-water pixels.

## 3. The eleven v1 → v2 fixes

The first finetune attempt (Stage 4b v1, run in early May 2026) plateaued at validation IoU 0.825. The encoder was effectively frozen because three independent multipliers compounded:

```
freeze_backbone_layers: 6     →  blocks 0-5 hard-frozen
backbone_lr_mult:       0.05  →  ×0.05 on top of LLRD for all encoder blocks
llrd_decay:             0.85  →  block_6 sees 0.85^18 = 0.054 multiplier

⇒ block_6 effective LR = decoder_lr × 0.05 × 0.054 = 0.27 % of decoder LR
```

Block 6 was seeing 0.27 % of the decoder learning rate; block 23 (the last) was seeing 4.25 %. The cumulative-LR plot from `metrics.csv` confirmed the encoder had effectively zero learning signal. Eleven configuration corrections were applied for the v2 finetune. Each is listed below with its before/after values, its rationale, and its individual or joint effect on the metric where measurable.

| # | Hyperparameter | v1 | v2 | Rationale | Effect |
|---:|---|---|---|---|---|
| 1 | `freeze_backbone_layers` | 6 | 0 | Stage 4a pretrain used `freeze=0` and reached 0.875 val IoU; finetune should at minimum match that | jointly with #2, #3: +5–7 pp val IoU |
| 2 | `backbone_lr_mult` | 0.05 | 0.25 | Block 6 effective LR rises from 0.27 % to 3.8 % of decoder LR | (joint with #1, #3) |
| 3 | `llrd_decay` | 0.85 | 0.9 | Gentler BERT-style geometric decay, so deep blocks still learn | (joint with #1, #2) |
| 4 | `lr` (decoder peak) | 5e-5 | 6e-5 | Slightly hotter decoder to compensate for the now-active encoder competing for gradient flow | small, subsumed in joint effect |
| 5 | `val_tta` | false | true | Honest checkpoint selection. v1's `val_tta = false` made checkpoint ranking use a noisy un-augmented metric; the model that wins on no-TTA is not the one that wins on TTA. Stage 4a used `true` and reached 0.875 honestly, so `false` was a regression | +1–2 pp on the reported number |
| 6 | `label_smoothing` | 0.1 | 0.0 | Müller et al. (2019, NeurIPS) and follow-up work (Ren et al., 2025, ICLR) show label smoothing degrades selective classification and interacts badly with inverse-frequency Dice on extreme imbalance | +0.3–0.6 pp |
| 7 | `dice_variant` | generalized | flat | Generalized Dice's inverse-frequency weighting was producing the degenerate val/dice ≈ 1.0 pathology on 1 % water; flat Dice is the well-conditioned binary choice | included in #6 effect |
| 8 | `pos_weight_max` | 200 | 100 | Empirical positive : negative ratio is ~ 1 : 100; clamping to 200 double-counts the FN penalty and over-predicts water | small but measurable |
| 9 | `ohem_keep_ratio` / `ohem_min_kept` | 0.7 / 2048 | 0.5 / 4096 | Match Stage 4a's recipe; more aggressive hard-mining with a higher absolute floor | small |
| 10 | `swa.swa_epoch_start` | 0.75 | 0.4 | v1 had `swa_start = 0.75 × 20 epochs = epoch 15`, but `early_stopping_patience = 6` killed the run at epoch 8 — SWA never activated. v2 starts SWA at epoch 12 of 30 with patience 15 | +0.5–1 pp |
| 11 | `block_size_deg` | 0.25 | 0.15 | 0.15° produces ~ 3 × more unique blocks (60 → 180), reducing the chance that one unlucky block dominates val | reduces split variance |

Combined effect: validation IoU moved from 0.825 (v1) to 0.9557 (v2 production soup) — 13 percentage points absolute improvement, all from configuration corrections, no architectural changes.

I list these in detail because they are the kind of small, easy-to-miss corrections that deserve to be public when one publishes a model. Every team eventually runs into one of these compounding-multiplier bugs. If reading this saves another team a week, that is a useful contribution.

## 4. Block split

The split is hash-based, year-invariant, and rules out spatial leakage by construction. Each chip's `(latitude, longitude)` is binned into a 0.15° × 0.15° block (≈ 17 × 14 km at 43° N), then the block is hashed:

```python
key = f"{salt}|{lat_idx}|{lon_idx}".encode("utf-8")
bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) % 100
split = "train" if bucket < 80 else "val" if bucket < 90 else "test"
```

with `salt = "cryosentinel-blocks-v1"`. A 0.02° (~ 2.2 km) buffer drops chips that fall within the buffer of a foreign block (4,922 chips dropped from the raw 10,536, leaving 5,614). The hash is independent of acquisition year, so the same geographic block stays in the same split across the 2017, 2021, 2022, and 2023 acquisitions. There is no path through which a train chip and a test chip can come from the same lake.

The implementation is in `src/cryosentinel/data/block_split.py` and is covered by `tests/test_block_split.py`.

The result on the Stage 4b finetune dataset is:

| Split | Chips | Share |
|---|---:|---:|
| Train | 4,283 | 76 % |
| Val | 666 | 12 % |
| Test | 665 | 12 % |

## 5. Snapshot averaging (model soup)

We collect SWA snapshots (Izmailov et al., 2018) from epoch 12 (`swa_start = 0.4 × 30 epochs`) through epoch 30 with annealed SWA learning rate of 1e-5 and three epochs of cosine annealing. Five snapshots are saved at steps 4815, 5350, 6955, 8560, and at the final step. The Wortsman et al. (2022) model soup uniformly averages these five checkpoints in weight space:

```
soup = (s_1 + s_2 + s_3 + s_4 + s_5) / 5
```

The soup outperforms any single SWA snapshot on:

- Global validation IoU: +0.0011 over the best single (0.9557 vs 0.9546)
- Global test IoU: +0.0021 (0.8918 vs 0.8897)
- Out-of-domain transfer (Zhetysu Alatau test): +1.81 pp (0.9312 vs 0.9131)
- Label-noise tail: roughly halves the count of chips with per-chip IoU < 0.05 on both splits

We attribute the disproportionate improvement on Zhetysu Alatau test transfer and the label-noise tail to the fact that snapshot averaging acts as an implicit ensemble in weight space: the five snapshots are correlated but not identical, and averaging cancels the chip-specific noise that any single snapshot picks up from its position on the loss surface.

## 6. EMA and TTA

EMA (decay 0.999, CPU shadow, applied at validation and test only) provides a second smoothing layer over the model weights. It is orthogonal to SWA: SWA averages over training-time snapshots; EMA tracks an exponential moving average of the live weights and produces a smoothed model for evaluation. Both are kept.

Test-time augmentation is **flip-only** (horizontal + vertical, four passes total averaged in logit space). We tested rotation TTA (90 / 180 / 270 degrees) during ablation and found it consistently degrades the global IoU on both validation and test, by roughly 0.5 pp. We attribute this to the way TerraMind's positional encoders interact with rotated inputs: the rotated patches end up at canvas positions the encoder did not see during pretraining.

## 7. Optimiser and schedule

- **AdamW** (Loshchilov & Hutter, 2019) in two parameter groups:
  - Backbone group: `lr = 5e-6` (the decoder peak `6e-5` × `backbone_lr_mult 0.25` × LLRD profile).
  - Decoder group: `lr = 5e-4` (the decoder peak with no multiplier).
- Cosine warm-up over the first epoch, then cosine decay to zero across the full 30 epochs.
- `weight_decay = 1e-4` on both groups. No bias / LayerNorm exclusion.
- Mixed precision: `bf16-mixed` (Lightning).
- `gradient_clip_val = 1.0`.

## 8. Augmentations

The training-time augmentation pipeline:

- **Spectral jitter** (`p = 0.4`). Per-band gain perturbation: ε_S2 = 0.03, ε_S1 = 0.07, ε_DEM = 0.02 (relative scale). Models the radiometric variation between Sentinel acquisitions.
- **Multi-scale** (`p = 0.4`). Random scale factor in [0.8, 1.20] applied to chip + mask. Helps the model generalise across lake-size distributions.
- **Copy-paste** (`p = 0.2`, minimum 32 donor water pixels). Adapted from Ghiasi et al. (2021). Pastes a randomly cropped water region from another chip onto the current chip. Increases positive-pixel density without changing the geographical distribution.
- **Hard-negative sampler** (positive : negative ratio 3 : 1, with replacement). At each epoch, the dataloader oversamples chips with high water fraction and undersamples chips with low water fraction. This counteracts the natural class imbalance at the chip level (most chips have zero water).

Test-time and validation-time augmentation is flip-only, as noted above.

## 9. Compute

| Stage | Hardware | Wall time | Cost |
|---|---|---:|---:|
| Stage 4a v2 pretrain (12 HMA regions, 30 epochs, 4283 train chips × 4 years × 12 regions = ~ 200k chip-epochs) | H100 80 GB cloud instance | ~ 9 h | ~ $36 |
| Stage 4b v2 finetune (3 Almaty-corridor regions, 30 epochs, 4283 train chips × 4 years = ~ 50k chip-epochs) | H100 80 GB cloud instance | ~ 2.5 h | ~ $10 |
| Eval diagnostics (val + test, TTA, per-chip Parquet, per-region breakdown) | H100 80 GB cloud instance | ~ 25 min × 2 runs | ~ $3 |

Total: roughly 12 hours on H100 and roughly $50 in cloud credits, plus the cost of the Stage 4b v1 mistake (~ $40 lost). Long training jobs used private checkpoint-resume orchestration to survive cloud-side interruptions. That orchestration layer is not part of the v1.0 public release.

## References

- Berman, M., Triki, A. R., & Blaschko, M. B. (2018). The Lovász-Softmax loss: a tractable surrogate for the optimization of the intersection-over-union measure in neural networks. *CVPR*.
- Ghiasi, G., et al. (2021). Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. *CVPR*.
- Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. *UAI*.
- Jakubik, J., et al. (2025). TerraMind: Large-Scale Generative Multimodality for Earth Observation. arXiv:2504.11171.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Müller, R., Kornblith, S., & Hinton, G. (2019). When does label smoothing help? *NeurIPS*.
- Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. *MICCAI MLMI*.
- Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training region-based object detectors with online hard example mining. *CVPR*.
- Sudre, C. H., et al. (2017). Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. *DLMIA*.
- Wortsman, M., et al. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *ICML*.
- Xiao, T., Liu, Y., Zhou, B., Jiang, Y., & Sun, J. (2018). Unified Perceptual Parsing for Scene Understanding. *ECCV*.
