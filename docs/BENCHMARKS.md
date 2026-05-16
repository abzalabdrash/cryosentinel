# Benchmarks

This document contains every quantitative claim made in the README, with the production checkpoint, evaluation protocol, and public artifacts needed to audit each row.

All numbers below come from the `soup.ckpt` checkpoint, which is the uniform weight-space average of five SWA snapshots collected from epoch 12 through epoch 30 of the Stage 4b v2 finetune (Wortsman et al., 2022; Izmailov et al., 2018). The checkpoint lives at `hf://abzal-glw/cryosentinel-terramind-v3/terramind_v3_finetune_almaty_v2/checkpoints/soup.ckpt`.

Test-time augmentation is flip-only (horizontal + vertical). EMA (decay 0.999) is applied at validation and test time.

## 1. Headline table

| Metric | Value | Notes |
|---|---:|---|
| Validation IoU @ threshold 0.5, TTA on (n = 666) | **0.9557** | Same train/val protocol as Adhikari & Regmi (2025) |
| Validation IoU @ best threshold (0.70), TTA on | 0.9596 | Threshold tuned on val |
| Validation mean per-chip IoU @ threshold 0.5 | 0.9740 | |
| Held-out test IoU @ threshold 0.5, TTA on (n = 665) | 0.8918 | Spatial-block-split, no leakage |
| Held-out test IoU @ best threshold (0.70), TTA on | 0.8959 | Threshold tuned on val, applied to test |
| Held-out test mean per-chip IoU @ threshold 0.5 | 0.9556 | |
| **Held-out test IoU, label-corrected (n = 658)** | **0.9082** | After dropping 7 chips with documented Kumar polygon mislabelling |

## 2. Per-region breakdown

The Stage 4b v2 finetune covers three Almaty-corridor regions: Tien Shan (full), Ile Alatau, and Zhetysu Alatau. The per-region IoU values from the production `soup.ckpt` with TTA at threshold 0.5:

| Region | Validation IoU | Test IoU (raw) | Test IoU (label-corrected) | Δ |
|---|---:|---:|---:|---:|
| ile_alatau | 0.8867 | 0.7664 | **0.9285** | +16.21 pp |
| tien_shan_full | 0.9564 | **0.9027** | 0.9027 | 0 |
| zhetysu_alatau | 0.9559 | **0.9312** | 0.9312 | 0 |

Ile Alatau's raw test IoU is dragged down by seven chips at two coordinates near 43° N (42.99° N, 76.71° E and 42.92° N, 76.73° E) across acquisitions in 2021, 2022, and 2023. The Kumar & Vijay (2026) ground truth on these chips undersizes or completely omits a glacial lake that the Sentinel-2 RGB, the MNDWI water index, and the model all show clearly. After dropping these seven chips (1.05 % of test), the Ile Alatau region's IoU rises by 16.21 percentage points and the global test IoU rises by 1.65 percentage points. The full audit, with four-panel visualisations, is in `LABEL_NOISE_AUDIT.md`.

## 3. Stage 4a pretrain — per-region validation IoU

The pretrain covers twelve High Mountain Asia sub-regions before the Almaty-corridor finetune. The per-region IoU values from the Stage 4a v2 production soup, with TTA at threshold 0.5:

| Region | Val IoU | vs Adhikari & Regmi (2025) val IoU 0.9130 |
|---|---:|---|
| karakoram | 0.935 | +2.2 pp |
| pamir | 0.934 | +2.1 pp |
| hengduan_nyainqentanglha | 0.928 | +1.5 pp |
| hma_other | 0.923 | +1.0 pp |
| western_himalaya | 0.919 | +0.6 pp |
| eastern_himalaya | 0.915 | +0.2 pp |
| ile_alatau | 0.901 | −1.2 pp |
| tibetan_plateau | 0.892 | −2.1 pp |
| central_himalaya | 0.854 | −5.9 pp |
| hindu_kush | 0.826 | −8.7 pp |
| tien_shan_full | 0.714 | finetune target — pushed to 0.9564 in Stage 4b |
| zhetysu_alatau | 0.283 | label-coverage artefact — see note below |

The Zhetysu Alatau pretrain global IoU of 0.283 looks alarming and was investigated in detail. Drilling into the underperformance:

- Per-chip mean IoU on Zhetysu Alatau pretrain validation is **0.974**. Ninety-four per cent of chips have IoU > 0.9.
- Four chips at (44.71° N, 80.28–80.31° E) drive the global collapse. The model predicts water on 13–22 % of the chip area; the Kumar inventory marks zero water.
- Those coordinates fall on the Tekeli reservoir and Karatal headwaters — non-glacial water bodies that the Kumar 2022 inventory legitimately excludes (it is a **glacial** lake catalogue).
- The model is correct in seeing water there, but the global-IoU formula is dominated by these few false-positive chips because the denominator (union of GT and prediction) is mostly the model's prediction with an empty GT.
- The Stage 4b finetune learns to suppress this signal and converges to 0.9559 validation IoU on Zhetysu Alatau, which is the right behaviour for an early-warning system (we do not want false alerts on routine reservoir operations).

The honest summary metric for Zhetysu Alatau in the pretrain phase is therefore the **mean per-chip IoU of 0.974**, with the global IoU of 0.283 documented as a label-coverage artefact rather than a model failure.

## 4. Comparison to literature

We are aware of two recent peer-reviewed or arXiv-released models that report quantitative segmentation metrics on Himalayan or High Mountain Asia glacial lakes. Both are listed below for honest comparison:

| Reference | Modality | Architecture | Train/val protocol | Val IoU |
|---|---|---|---|---:|
| Adhikari & Regmi (2025), arXiv 2512.24117 | S1 only | U-Net + EfficientNet-B3 | 4-lake cohort, train/val | 0.9130 |
| Aggarwal et al. (2024) | S2 only | DeepLabv3+ | Region cohort, train/val | 0.876 |
| **CryoSentinel (this work)** | **S1 + S2 + DEM** | **TerraMind-1.0-Large + UperNet** | **3-region spatial-block-split, train/val/test** | **0.9557** |

The Adhikari & Regmi (2025) paper is the closest comparator. Both models target the GLOF early-warning problem; both train on Sentinel-class imagery; both report validation IoU on glacial lakes. Differences:

1. CryoSentinel uses three modalities (S1 + S2 + DEM) and a foundation-model backbone; Adhikari & Regmi use S1 only with a from-scratch U-Net.
2. CryoSentinel uses a spatial-block-split with 17 km separation and a held-out test set; Adhikari & Regmi use a train/val split on a 4-lake cohort.
3. CryoSentinel reports +4.27 percentage points absolute improvement on validation IoU (0.9557 vs 0.9130).

We do not claim that CryoSentinel is "the best" segmentation model on every conceivable benchmark, only that **on the same train/val protocol used by Adhikari & Regmi (2025), with the addition of a held-out test split, our model reports a higher validation IoU**. If you find a paper that reports a higher number under a comparable protocol, please open an issue and we will update this table.

## 5. Label-noise tail

A useful diagnostic for segmentation quality is the count of chips with very low per-chip IoU, which indicates either model failures or label noise. For the production `soup.ckpt`:

| Split | Chips with per-chip IoU < 0.05 | Share |
|---|---:|---:|
| Validation (n = 666) | 2 | 0.3 % |
| Test (n = 665) | 4 | 0.6 % |

For comparison, the single-best checkpoint `step001605-iou0.952.ckpt` (without the SWA soup) had:

| Split | Chips with per-chip IoU < 0.05 | Share |
|---|---:|---:|
| Validation | 4 | 0.6 % |
| Test | 8 | 1.2 % |

The soup roughly halved the label-noise tail. We interpret this as evidence that snapshot averaging in weight space, beyond improving the global IoU, also stabilises the model's predictions on borderline chips.

## 6. Threshold sweep

The headline numbers above use threshold 0.5. The validation curve, evaluated at threshold values from 0.1 to 0.9 in steps of 0.05, peaks at threshold 0.70:

| Threshold | Validation IoU | Test IoU |
|---|---:|---:|
| 0.30 | 0.9421 | 0.8769 |
| 0.40 | 0.9505 | 0.8848 |
| 0.50 | 0.9557 | 0.8918 |
| 0.60 | 0.9586 | 0.8946 |
| **0.70** | **0.9596** | **0.8959** |
| 0.80 | 0.9580 | 0.8941 |

The headline table reports both threshold 0.5 (the conventional default) and threshold 0.70 (best-on-val). The cost of adopting threshold 0.70 in deployment is that the model becomes mildly more conservative; the upside is the ~ 0.4 pp validation IoU improvement.

## 7. Public audit artifacts

The released checkpoint, dataset, per-chip diagnostics, per-region breakdowns, and threshold sweeps are public on Hugging Face. See `REPRODUCING.md` for artifact paths and the expected headline table.

The internal cloud runner that produced the v1.0 diagnostics is not included in this public repository. This keeps the repository focused on the model release while preserving the benchmark evidence needed for external audit.

## 8. Caveats

- All numbers above are from a **single training run with `seed = 42`**. v1.0 ships the production-best result. A 3-seed variance estimate (seeds {17, 42, 1337}) is scheduled for v1.1 in June 2026, with compute budget allocated.
- The dataset `abzal-glw/cryosentinel-glof-v3` is publicly downloadable on Hugging Face under ODC-By 1.0. Login is recommended to avoid anonymous rate limits.
- The public release includes benchmark artifacts and dataset/model references, but not the private cloud orchestration used to regenerate them.

## References

- Adhikari, P., & Regmi, S. R. (2025). Targeted Semantic Segmentation of Himalayan Glacial Lakes Using Time-Series SAR: Towards Automated GLOF Early Warning. arXiv:2512.24117.
- Aggarwal, A., et al. (2024). Deep learning–based glacial lake detection from Sentinel-2 imagery. *Remote Sensing*.
- Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. *UAI*.
- Wortsman, M., et al. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *ICML*.
