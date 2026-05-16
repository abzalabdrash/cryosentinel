# Limitations

This document is the place where we are explicit about what the model cannot do, what it has not been tested on, and what we suspect but have not verified. We list these because we believe a model release without an honest limitations document is incomplete.

## 1. CryoSentinel does not predict GLOF events

CryoSentinel is a **segmentation** model. It takes a satellite chip and returns a binary water mask. It does not output a probability of an outburst, an estimated time-to-failure, or any kind of forecast.

Breach prediction is a separate problem and lives in the [GLOFcast](https://github.com/abzalabdrash/glofcast) operational system. GLOFcast scores GLOF hazard from a composite of:

- **Static factors**: lake morphometry (area, depth, dam type, freeboard), dam geometry (slope, breach width), downstream slope and watercourse length, glacier proximity, permafrost zone proxies.
- **Dynamic factors**: lake-area trend over multi-year windows, glacier velocity (ITS_LIVE annual mosaics), surface temperature trend (MODIS LST winter-trend), lake-surface InSAR creep when available.
- **Triggers**: heatwaves and precipitation extremes (ERA5 + GPM IMERG), seismicity (USGS Earthquake Catalog), avalanche / landslide proxies.

Those scores are calibrated separately on a casebook of historical GLOFs and validated on a 240-lake-month live replay (zero false-fires). CryoSentinel feeds GLOFcast the lake-area inputs; it is not itself the alarm.

Confusing the two would be the most consequential misuse of this release.

## 2. Geographic scope is High Mountain Asia only

The training data covers twelve HMA sub-regions (pretrain: Karakoram, Pamir, Hengduan / Nyainqentanglha, HMA-other, Western Himalaya, Eastern Himalaya, Ile Alatau, Tibetan Plateau, Central Himalaya, Hindu Kush, Tien Shan full, Zhetysu Alatau) plus the Almaty-corridor finetune (Tien Shan full, Ile Alatau, Zhetysu Alatau). We have not run the model on:

- Andes (Peruvian / Bolivian / Patagonian glaciers)
- European Alps
- Caucasus
- Scandes
- Greenland or Arctic Canada
- Antarctic Peninsula

We do not know how the model transfers. Plausible failure modes outside HMA:

- Different cloud climatology (Andes have heavier persistent cloud than Tien Shan, especially on the eastern slopes).
- Different debris-cover regimes (Patagonian glaciers shed dust differently from Karakoram, and the resulting reflectance signature in S2 may not map to HMA training distribution).
- Different lake morphometry distributions (proglacial lakes in the Alps tend to be smaller and more bounded by bedrock than HMA moraine-dammed lakes).

If you run the model out-of-domain, please share the per-chip diagnostics. We will collate community results in `docs/EXTERNAL_VALIDATION.md` (currently empty).

## 3. Sub-hectare lakes are not reliable

The minimum lake area we trust is ~ 0.5 ha (≈ 50 pixels at 10 m/pixel). Below that, the segmentation becomes unreliable in ways that depend on context:

- Very small puddles (< 100 m²) on glacier surfaces are sometimes detected, sometimes missed, depending on the BG/ratio of the surrounding ice.
- Thermokarst features on glacier tongues are often false-positively segmented as water; the model is not trained to distinguish them.
- Irrigation features (canals, drainage ponds) at the foot of the mountains are sometimes false-positively segmented; the Kumar inventory excludes them, so the model has not seen them as positives during training, but the cross-region transfer from Zhetysu Alatau (where they exist) to Ile Alatau (where they are sparse) can produce edge cases.

If your application targets lakes < 0.5 ha, treat the model output as a candidate map rather than a definitive segmentation.

## 4. Non-glacial water bodies are intentionally suppressed

During Stage 4a pretrain (twelve regions), the Zhetysu Alatau global validation IoU was 0.283 — alarmingly low. Investigation showed that four chips at (44.71° N, 80.28–80.31° E) drove the global collapse: the model predicted 13–22 % water on chips where the Kumar inventory marks zero. Those coordinates fall on the **Tekeli reservoir** and the **Karatal headwaters** — non-glacial water bodies that the Kumar 2022 catalogue legitimately excludes (it is a glacial lake catalogue, not a general water inventory).

The Stage 4b finetune learned to suppress these signals and converged to 0.9559 validation IoU on Zhetysu Alatau. **This is the intended behaviour for an early-warning system that should not raise alerts on routine reservoir operations.** It also means **CryoSentinel is not a general water segmenter**: rivers, lakes downstream of glacier termini, irrigation reservoirs, and seasonal floodplains are not the target class and may be either missed or false-positively segmented depending on training-distribution proximity.

If your application is general water mapping, use a model trained for that task (e.g. the IBM Sen1Floods11 checkpoint, or a custom finetune).

## 5. Single-snapshot, no temporal modelling

The model takes one chip at one acquisition date and returns one mask. It does not look at the previous year's mask, the previous month's, or any moving-window composite.

Year-on-year change detection is supported by running the model independently on per-year chips and differencing the masks afterward, which is what we do in the GLOFcast operational pipeline. But the model itself has no temporal context.

Temporal modelling (e.g. TimeSformer over 12-month chip stacks, or LSTM over annual-mean MNDWI) is on the v2 roadmap. The Adhikari & Regmi (2025) "temporal-first" framing is a reasonable direction we are likely to follow.

## 6. No uncertainty quantification

The model emits a single sigmoid logit per pixel. It does not output an uncertainty map.

A sensible MC-dropout or deep-ensemble extension is straightforward to add and is on the v1.1 roadmap. Until then, downstream users who need uncertainty should:

1. Run inference at multiple thresholds (e.g. 0.4, 0.5, 0.7) and take the symmetric difference as a soft-uncertainty band.
2. Combine with an independent water indicator (MNDWI, NDWI) and treat disagreement between the two as a low-confidence flag.

The label-noise audit in `LABEL_NOISE_AUDIT.md` uses MNDWI cross-check this way and found seven Kumar-mislabelled chips.

## 7. Single-seed runs only (multi-seed variance shipping in v1.1, June 2026)

Every reported v1.0 metric comes from a **single training run with `seed = 42`**. v1.0 ships the production-best result. A 3-seed variance estimate (seeds {17, 42, 1337}) is scheduled for **v1.1 in June 2026**, after the UN Zayed Sustainability Prize 2026 submission deadline. Compute budget for it (~ $138 on H100) is allocated.

Seed-to-seed variance in transfer-learning regimes with 4,000-chip finetune sets is typically modest (1–2 percentage points on global IoU), and the Stage 4a pretrain already absorbs the bulk of the seed-sensitivity at the encoder level. We expect the variance band on the headline 0.9557 / 0.9082 numbers to be tight, but v1.1 will report it empirically.

The Stage 4a v2 pretrain itself ran once end-to-end. Long-running training jobs are resumed bit-identically through `_sanitize_resume_ckpt` (SWA buffer, EMA shadow, AdamW state, and cosine LR position all transfer), so the trajectory represents a single seed of the pretrain — and will be re-run alongside the finetune in v1.1.

## 8. Test-set audit is incomplete

We audited the seven `ile_alatau` chips with the lowest test IoU and confirmed they are Kumar-polygon mislabels (see `LABEL_NOISE_AUDIT.md`). We did **not** manually audit the four chips with per-chip IoU < 0.05 in `tien_shan_full`. They could be legitimate model failures, additional Kumar mislabels, or a mixture.

If they turn out to also be Kumar mislabels, the global label-corrected test IoU would rise from 0.9082 to ~ 0.91. We do not claim this number because we have not done the audit.

## 9. Affiliation disclosure

CryoSentinel is a **research-stage prototype** developed independently as the supervision component of the [GLOFcast](https://github.com/abzalabdrash/glofcast) project. It is conceived as a complement to existing public-agency early-warning systems — in particular «Казселезащита» (Kazakhstan) and the UNESCO GLOFCA programme — but **is not currently affiliated with, endorsed by, or operationally integrated with these or any other agency**. Partnership outreach is planned post-v1.0 publication. References to these agencies in the documentation describe institutional context, not partnership.

## 10. Inference speed

A single 224 × 224 chip takes roughly 80 ms on an H100 with batch 1 (TTA on, fp16) and roughly 30 ms with batch 8 amortised. On a 24 GiB consumer GPU (RTX 4090 / 4080 Super) the per-chip latency is roughly 200 ms.

For dense regional inference (e.g. tiling the Almaty corridor at 10 m/pixel and predicting all chips), expect roughly 1 hour per 10,000 km² on H100 with TTA and our default batch / num_workers settings. The bottleneck is currently the disk I/O on the multimodal chip read, not the GPU forward pass.

A faster student model (TerraMind 1.0 Small + UperNet) is on the v1.2 roadmap.
