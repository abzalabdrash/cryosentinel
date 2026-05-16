# Changelog

All notable changes to CryoSentinel will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-16 (initial public release)

### Added

- TerraMind 1.0 Large + UperNet decoder fine-tuned on 5,614 spatial-block-split chips across the Tien Shan, Zhetysu Alatau, and Ile Alatau ranges.
- Production checkpoint `soup.ckpt` (uniform weight-space average of five SWA snapshots from epochs 12–30, after Wortsman et al. 2022 / Izmailov et al. 2018) and a single-best fallback `step001605-iou0.952.ckpt`, both released under Apache 2.0 on HuggingFace at `abzal-glw/cryosentinel-terramind-v3`.
- Hash-based spatial block split (`src/cryosentinel/data/block_split.py`): 0.15° × 0.15° blocks, 0.02° (~ 2.2 km) buffer, year-invariant SHA-1 assignment with `salt = "cryosentinel-blocks-v1"`. Mathematically rules out spatial and temporal leakage.
- Multi-modal data loader for Sentinel-2 L2A (12 bands), Sentinel-1 GRD (VV + VH), and Copernicus DEM 30 m at 224 × 224 chips, 10 m/pixel. Per-band normalisation statistics computed via Welford's algorithm.
- MEGA loss (BCE with `pos_weight = 100` + flat Dice + per-image Lovász softmax + asymmetric Tversky + Focal + Boundary), OHEM hard-negative mining, EMA decay 0.999, flip-only TTA at validation and test time.
- Eleven engineering corrections from the v1 finetune (documented inline in `configs/terramind_v3_finetune_almaty_v2.yaml` and tabulated in `docs/METHOD.md` §3) that lifted validation IoU from 0.825 to 0.9557.
- Label-noise audit (`docs/LABEL_NOISE_AUDIT.md`): seven test chips with documented Kumar & Vijay (2026) polygon mislabelling identified via MNDWI cross-check; label-corrected test IoU is 0.9082 (vs 0.8918 raw) and Ile Alatau test IoU is 0.9285 (vs 0.7664 raw).
- Robust checkpoint-resume mechanism (`scripts/train_terramind.py::_sanitize_resume_ckpt`): wipes absolute-path references in the Lightning `ModelCheckpoint` callback state so a long-running training job can be resumed bit-identically after any interruption via `--resume-from hf://...last.ckpt`.
- Reproducibility scripts: `scripts/reproduce_benchmarks.sh` (one-command reproduction of the headline table) and a Modal entrypoint for full pretrain + finetune from scratch.
- Per-chip diagnostics for the validation and test splits as Parquet files at `abzal-glw/cryosentinel-terramind-v3/terramind_v3_finetune_almaty_v2__soup_tta1/diagnostics/`.

### Headline metrics

- Validation IoU @ thr = 0.5, TTA: **0.9557** (vs Adhikari & Regmi 2025: 0.9130; +4.27 pp absolute under the same train/val protocol)
- Validation IoU @ best threshold (0.70): 0.9596
- Held-out test IoU, raw (n = 665): 0.8918
- Held-out test IoU, label-corrected (n = 658): **0.9082**
- Per-region test IoU: 0.9027 (Tien Shan full), 0.9312 (Zhetysu Alatau), 0.9285 (Ile Alatau, label-corrected)

### Known limitations

- Single-seed runs only at v1.0; multi-seed variance estimate (3 seeds: 17, 42, 1337) shipping in v1.1 (June 2026).
- No out-of-domain validation outside High Mountain Asia.
- Single-snapshot model; no explicit temporal modelling.
- No uncertainty quantification.
- Not affiliated with «Казселезащита», UNESCO GLOFCA, or any public agency. Research-stage prototype only.

[1.0.0]: https://github.com/abzalabdrash/Cryosentinel/releases/tag/v1.0.0
