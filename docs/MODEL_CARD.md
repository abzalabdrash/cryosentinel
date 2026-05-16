---
license: apache-2.0
library_name: pytorch
tags:
  - earth-observation
  - remote-sensing
  - semantic-segmentation
  - glacial-lakes
  - GLOF
  - foundation-model
  - terramind
  - multimodal
  - sentinel-2
  - sentinel-1
  - copernicus-dem
  - high-mountain-asia
  - tien-shan
  - climate-risk
datasets:
  - abzal-glw/cryosentinel-glof-v3
language:
  - en
metrics:
  - iou
  - f1
pipeline_tag: image-segmentation
base_model: ibm-esa-geospatial/TerraMind-1.0-large
model-index:
  - name: cryosentinel-terramind-v3
    results:
      - task:
          type: image-segmentation
          name: Glacial-lake semantic segmentation
        dataset:
          type: abzal-glw/cryosentinel-glof-v3
          name: CryoSentinel multimodal chips v3 (Almaty corridor)
          split: validation
        metrics:
          - type: iou
            value: 0.9557
            name: IoU @ thr 0.5, TTA
          - type: iou
            value: 0.9596
            name: IoU @ best thr 0.70, TTA
      - task:
          type: image-segmentation
          name: Glacial-lake semantic segmentation
        dataset:
          type: abzal-glw/cryosentinel-glof-v3
          name: CryoSentinel multimodal chips v3 (Almaty corridor)
          split: test
        metrics:
          - type: iou
            value: 0.8918
            name: IoU @ thr 0.5, TTA (raw)
          - type: iou
            value: 0.9082
            name: IoU @ thr 0.5, TTA (label-corrected, n=658)
---

# CryoSentinel — TerraMind-v3 Almaty Corridor Soup

A foundation-model semantic-segmentation checkpoint for glacial lakes from joint Sentinel-1 SAR, Sentinel-2 optical, and Copernicus DEM imagery in High Mountain Asia. Released under Apache 2.0 as the supervision component of the [GLOFcast](https://github.com/abzalabdrash/glofcast) operational early-warning system.

This model card documents the **Stage 4b v2 production soup** — a uniform weight-space average of five SWA snapshots from the Almaty-corridor finetune. It is the recommended checkpoint for inference.

## Key results

| Metric | Value |
|---|---:|
| Validation IoU @ thr = 0.5, TTA on (n = 666) | **0.9557** |
| Validation IoU @ best thr (0.70), TTA on | 0.9596 |
| Held-out test IoU @ thr = 0.5, TTA on (n = 665) | 0.8918 |
| Held-out test IoU, label-corrected (n = 658) | **0.9082** |

Comparison to literature — same train/val protocol as Adhikari & Regmi (2025), arXiv:2512.24117:

| Reference | Modality | Val IoU |
|---|---|---:|
| Adhikari & Regmi (2025) | S1 only | 0.9130 |
| **CryoSentinel (this model)** | **S1 + S2 + DEM** | **0.9557** |

## Model details

- **Architecture**: TerraMind 1.0 Large encoder (1.1 B parameters; Jakubik et al., 2025) + UperNet decoder (channel sequence 256 → 128 → 64 → 32; Xiao et al., 2018) with a `LearnedInterpolateToPyramidal` neck.
- **Inputs**: 224 × 224 chips at 10 m/pixel:
  - 12 Sentinel-2 L2A bands (B01–B12 minus B10), uint16 reflectance
  - 2 Sentinel-1 GRD polarisations (VV, VH), float32 dB
  - 1 Copernicus DEM 30 m band, int16 m elevation
- **Output**: per-pixel water sigmoid logit at the input resolution.
- **Training**: 30 epochs Stage 4a HMA pretrain → 30 epochs Stage 4b Almaty-corridor finetune.
- **Soup**: uniform mean of five SWA snapshots (epochs 12, 14, 17, 22, 30) per Wortsman et al. (2022). EMA decay 0.999. Flip-only TTA at validation and test time.
- **Loss**: BCE (pos_weight = 100) + flat Dice + per-image Lovász softmax + asymmetric Tversky (α = 0.25, β = 0.75) + Focal (γ = 2) + Boundary, with OHEM hard-negative mining.
- **Train compute**: ~ 12 hours on an H100 80 GB cloud instance, ~ $50 in cloud credits. Private checkpoint-resume orchestration was used for long-running jobs.

## Files

```
terramind_v3_finetune_almaty_v2/
  checkpoints/
    soup.ckpt                       ← recommended for inference (this card)
    step001605-iou0.952.ckpt        ← single-best fallback
    last.ckpt                       ← resume-from-here
  diagnostics/
    per_chip_val.parquet
    per_chip_test.parquet
    per_region_breakdown.json
    threshold_sweep.json
  logs/
    metrics.csv
    config.yaml
```

## Intended uses

**Direct uses.** Operational glacial-lake monitoring in High Mountain Asia. Multi-year change detection by running per-year inference and differencing masks. Research on multi-modal foundation-model fine-tuning, label-noise robustness, and snapshot averaging in segmentation.

**Downstream uses.** Component of GLOF early-warning systems (CryoSentinel produces the lake-area inputs; the breach forecast lives in [GLOFcast](https://github.com/abzalabdrash/glofcast)). Per-lake hazard scoring pipelines that consume per-pixel water masks. Comparative benchmarking against new foundation models on the public test split.

**Out of scope.** General water mapping (rivers, irrigation reservoirs, coastal water). The model is intentionally trained to suppress non-glacial water signatures; see `docs/LIMITATIONS.md` § 4. Areas outside HMA — no out-of-domain validation has been performed.

## Limitations

The full list is in `docs/LIMITATIONS.md`; the headline items:

- **Not a forecaster.** Segmentation only; no breach probability or time-to-failure output.
- **HMA only.** Twelve sub-regions trained, no Andes / Alps / Caucasus / Patagonia validation.
- **Sub-hectare lakes are unreliable.** Minimum trustworthy area ~ 0.5 ha (≈ 50 px at 10 m/px).
- **Single-snapshot.** No temporal modelling.
- **No uncertainty quantification.** Single sigmoid logit per pixel.
- **Single-seed.** All numbers from one run with `seed = 42`. Multi-seed variance (3 seeds) shipping in v1.1, June 2026; budget allocated.
- **Not affiliated.** CryoSentinel is a research-stage prototype, not affiliated with «Казселезащита», UNESCO GLOFCA, or any other public agency. See `docs/LIMITATIONS.md` § 9.

## Bias, risks, and ethical considerations

The model is trained on a Kumar & Vijay (2026) inventory that integrates Landsat-8, Sentinel-1, and Sentinel-2 imagery with manual quality control. Documented bias modes:

- **Label undersizing** on a small fraction (~ 1 %) of chips. We have audited seven instances in the test split where the Kumar polygon undersizes a real lake; the model correctly identifies the larger water body in all seven (see `docs/LABEL_NOISE_AUDIT.md`). This bias is documented but not corrected at training time, because the inverse direction — chips where Kumar oversizes a small puddle — would require an audit we have not performed.
- **Cloud climatology bias.** The training imagery is selected from late-summer composites with cloud cover < 30 %. Performance on persistently cloudy regions (eastern Himalaya monsoon, Andes Pacific slope) will degrade; we have not quantified by how much.
- **Operational misuse.** The biggest risk we are aware of is treating CryoSentinel as a forecaster rather than a segmenter. We document this prominently in the README and `LIMITATIONS.md`. If you find the model deployed in a way that conflates it with a breach forecaster, please flag it.

## How to use

### Hugging Face Hub

```python
from huggingface_hub import hf_hub_download
import torch

ckpt_path = hf_hub_download(
    repo_id="abzal-glw/cryosentinel-terramind-v3",
    filename="terramind_v3_finetune_almaty_v2/checkpoints/soup.ckpt",
)
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
# Pass to a CryoSentinel Lightning module configured with
#   configs/terramind_v3_finetune_almaty_v2.yaml
```

### Benchmark artifacts

The released checkpoint, dataset, per-chip diagnostics, per-region breakdowns, and threshold sweeps are public. See `docs/REPRODUCING.md` in the GitHub repository for artifact paths and expected headline metrics.

### Download the released checkpoint

```bash
huggingface-cli download \
    abzal-glw/cryosentinel-terramind-v3 \
    terramind_v3_finetune_almaty_v2/checkpoints/soup.ckpt \
    --local-dir ./checkpoints
```

Input chip layout: 12 S2 L2A bands → 2 S1 GRD bands (VV, VH dB) → 1 Copernicus DEM 30 m band, all at 10 m/pixel in a 224 × 224 GeoTIFF.

## Training data

`abzal-glw/cryosentinel-glof-v3`: 42,237 chips across 12 HMA sub-regions × 4 years (2017, 2021, 2022, 2023) at 224 × 224 / 10 m / pixel. 30.4 GiB on disk. Built from Kumar & Vijay (2026) labels + ESA Copernicus Sentinel-1/2 + Copernicus DEM 30 m. Publicly downloadable on Hugging Face under ODC-By 1.0; see `docs/DATA.md` for download instructions and full preprocessing details.

Stage 4b finetune split (after spatial-block-split with 0.15° blocks and 0.02° / 2.2 km buffer):

| Split | Chips | Share |
|---|---:|---:|
| Train | 4,283 | 76 % |
| Val | 666 | 12 % |
| Test | 665 | 12 % |

The split is hash-based, year-invariant, and rules out spatial leakage by construction. See `docs/METHOD.md` § 4.

## Evaluation protocol

- Production checkpoint: `soup.ckpt` (uniform mean of five SWA snapshots).
- TTA: flip-only (horizontal + vertical, four passes averaged in logit space).
- EMA: decay 0.999, applied at validation and test time.
- Threshold: 0.5 (default headline) and 0.70 (best-on-val).
- Metric: global IoU computed as (sum of intersection) / (sum of union) across all chips in the split. Per-chip mean IoU also reported as a robustness check.

## Citation

```bibtex
@software{abdrash2026cryosentinel,
    title  = {CryoSentinel: A Foundation-Model Glacial Lake Segmenter for High Mountain Asia},
    author = {Abdrash, Abzal},
    year   = {2026},
    url    = {https://github.com/abzalabdrash/Cryosentinel},
    version = {v1.0.0},
    license = {Apache-2.0}
}
```

If you use CryoSentinel in operational early-warning, please also cite the underlying TerraMind backbone:

```bibtex
@article{jakubik2025terramind,
    title   = {TerraMind: Large-Scale Generative Multimodality for Earth Observation},
    author  = {Jakubik, Johannes and others},
    journal = {arXiv:2504.11171},
    year    = {2025}
}
```

And the Kumar & Vijay glacial-lake inventory:

```bibtex
@dataset{kumar2026hma,
    title     = {Inventory of Glacial Lakes in High Mountain Asia for the Years 2016 and 2022},
    author    = {Kumar, Ravindra and Vijay, Saurabh},
    year      = {2026},
    publisher = {PANGAEA},
    doi       = {10.1594/PANGAEA.983845}
}
```

## Acknowledgments

- **TerraMind** by IBM Research, the European Space Agency Φ-lab, and the FAST-EO project (Apache 2.0). Released April 2025.
- **TerraTorch** by IBM Research (Apache 2.0). The training framework that made the multi-modal patch-embedder dispatch and LLRD profile manageable.
- **Cloud GPU providers** for access to H100 80 GB compute.
- **Hugging Face** for hosting the model checkpoints and the training dataset.
- The **«Казселезащита»** institutional GLOF early-warning service in Almaty, Kazakhstan, and the **UNESCO GLOFCA** programme — the institutional context this work is positioned to complement, not replace. CryoSentinel is not affiliated with either organisation; this acknowledgment recognises their decades of operational work on the same problem.

## Contact

Open an issue at https://github.com/abzalabdrash/Cryosentinel/issues for bug reports, reproducibility checks, or out-of-domain validation results.
