# Data

This document describes the dataset used to train and evaluate CryoSentinel: where it comes from, how it was preprocessed, what its statistics are, and how to access it.

## 1. Source labels — Kumar & Vijay (2026)

The supervision signal comes from the **Inventory of Glacial Lakes in High Mountain Asia for the Years 2016 and 2022** by Kumar, R. and Vijay, S., published on PANGAEA in 2026:

- DOI: [10.1594/PANGAEA.983845](https://doi.org/10.1594/PANGAEA.983845)
- Total lakes in the inventory: 31,698 across HMA in 2022.
- Format: GeoJSON polygons in WGS-84 (EPSG:4326).
- Compilation method: integration of Landsat-8, Sentinel-1, and Sentinel-2 imagery with manual quality control. The PANGAEA dataset page documents the methodology.

We extract 162 polygons that fall in our finetune region (the Almaty corridor: Tien Shan full + Ile Alatau + Zhetysu Alatau) and ~ 22,000 polygons across the twelve HMA sub-regions used for pretrain.

## 2. Imagery sources

Three modalities are stacked at the chip level:

### 2.1. Sentinel-2 L2A (12 bands)

- Source: ESA Copernicus, accessed via Microsoft Planetary Computer or Google Earth Engine.
- Bands kept: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12. Band B10 (cirrus) is dropped because it is not delivered as surface reflectance in L2A.
- Resolution: native 10 m / 20 m / 60 m depending on band; we resample everything to 10 m via bilinear interpolation, matching the chip target grid.
- Acquisition windows: late summer composite (July–September) per year, selected for cloud cover < 30 % via the SCL band, with a fallback to October if no qualifying tile exists.
- Years: 2017, 2021, 2022, 2023 for the finetune; the same years plus broader HMA coverage for the pretrain.
- Storage: uint16 reflectance, scale factor 10,000 (i.e. raw value 5,000 = reflectance 0.5).

### 2.2. Sentinel-1 GRD (2 polarisations)

- Source: ESA Copernicus, accessed via Microsoft Planetary Computer.
- Polarisations: VV and VH.
- Pre-processing pipeline (matches IBM's TerraMesh recipe):
  1. Apply orbit file (precise where available, restituted otherwise).
  2. GRD border noise removal.
  3. Thermal noise removal.
  4. Calibration to sigma-naught.
  5. Multi-look 5 × 1 (range × azimuth).
  6. Lee speckle filter, 5 × 5 window.
  7. Range-Doppler terrain correction using SRTM 30 m DEM as the elevation reference.
  8. Conversion to dB.
- Resolution after terrain correction: 10 m, co-registered to the Sentinel-2 grid.
- Acquisition windows: same season as the matched S2 tile, with ≤ 7 days separation.
- Storage: float32 backscatter in dB.

### 2.3. Copernicus DEM 30 m (1 band)

- Source: ESA, GLO-30 product, accessed via Microsoft Planetary Computer.
- Resolution: native 30 m, bilinearly resampled to 10 m to match the Sentinel-2 grid.
- Storage: int16 elevation in metres above EGM2008 geoid.
- Pre-processing: no DEM-specific processing beyond the resampling. We do not derive slope or curvature at the dataloader level; the model sees raw elevation and learns the spatial gradient itself.

## 3. Chip extraction

For each `(region, year)` pair:

1. Load the relevant Kumar polygons clipped to the region's bounding box.
2. Buffer each polygon by 224 × 10 m / 2 = 1.12 km to ensure the lake is centred in a chip.
3. Tile the bounding box into 224 × 224 windows at 10 m/pixel in the appropriate UTM zone (43 N, 44 N, 45 N, or 46 N depending on longitude).
4. Drop windows that fall outside any buffered polygon (we want positive-rich training chips, not pure background).
5. For each surviving window, read the matched S2 + S1 + DEM stack, the Kumar mask rasterised with `all_touched = True`, and the chip metadata (latitude, longitude, year, region, UTM zone, source tile IDs).

The result is roughly 10,500 raw chips per `(region, year)` pair before the sanity filter.

## 4. Sanity filter

A high fraction of raw chips have low agreement between the Kumar mask and the imagery, for several reasons:

- Frozen lakes in summer 2022 at altitudes above 5,000 m. The Kumar polygon marks the open-water extent, but the S2 imagery shows ice. SAR backscatter is also ambiguous on frozen lakes.
- Cloud-shadowed lakes in the S2 composite. Even with the cloud-cover filter, residual cirrus or shadow can hide water.
- Misregistered Kumar polygons. The PANGAEA dataset has a documented horizontal registration error of up to 30 m at the polygon boundary, which shifts small lakes by several pixels at 10 m/pixel.
- Lakes that have changed (drained, frozen, drained-and-refilled) since the Kumar snapshot.

We compute MNDWI = (B03 − B11) / (B03 + B11) on the S2 stack, threshold it at 0 to get an MNDWI water mask, and compute the IoU between the MNDWI mask and the Kumar mask at the chip level. We then drop chips whose Kumar-MNDWI IoU is below 0.20 OR whose `pred_water_pixels / kumar_water_pixels` ratio exceeds 3.0 (suggesting Kumar undersizes a real lake).

The diagnostic from this filter on `central_himalaya / 2022` (10 tiles, 415 chips):

| Filter category | Count | Share |
|---|---:|---:|
| Regime A (chip has Kumar lake) — kept | 108 | 28 % |
| Regime A — dropped, Kumar–MNDWI IoU < 0.05 | 230 | 59 % |
| Regime A — dropped, IoU in [0.05, 0.10) | 25 | 6 % |
| Regime A — dropped, IoU in [0.10, 0.15) | 16 | 4 % |
| Regime A — dropped, IoU in [0.15, 0.20) | 11 | 3 % |
| Regime A — dropped, fp_ratio > 3.0 | 54 | 14 % |
| Regime B (chip is background) — kept | 21 | 5 % |
| Regime B — dropped, MNDWI water > 5 % | 4 | 1 % |

The key observation is that 82 % of dropped Regime A chips have IoU < 0.05, meaning the Kumar polygon and the MNDWI water do not overlap at all. These are not borderline cases; they are truly mislabelled or unusable chips. Lowering the IoU threshold from 0.20 to 0.10 would only recover ~ 10 % of the dropped chips, at the cost of substantial added noise.

We document this here because the filter is opinionated and a reproducer should know the trade-off: training on a smaller but cleaner dataset vs a larger but noisier one. Our choice was the cleaner option; the audit above is the evidence behind it.

After the sanity filter, the median IoU of the kept chips is 0.48 (Kumar–MNDWI agreement), which corresponds to high-quality labels by remote-sensing standards.

## 5. Final dataset statistics (multimodal_chips_v3)

Across the twelve HMA sub-regions and the four years, after the sanity filter:

| Property | Value |
|---|---|
| Total chips | 42,237 |
| Total `(region, year)` pairs | 48 (12 × 4, minus the missing kungey_alatau) |
| Total shards | 114 |
| Total size on disk | 30.4 GiB |
| Schema-consistent across shards | yes |

For the Stage 4b v2 finetune (3 regions × 4 years), after the spatial block split:

| Split | Chips | Share |
|---|---:|---:|
| Train | 4,283 | 76 % |
| Val | 666 | 12 % |
| Test | 665 | 12 % |
| Dropped (block buffer) | 4,922 | (excluded from total) |

Per-region chip totals across all years:

| Region | Chips | Year coverage |
|---|---:|---|
| ile_alatau | 2,323 × 4 | 2017, 2021, 2022, 2023 |
| zhetysu_alatau | 3,711 × 4 | 2017, 2021, 2022, 2023 |
| karakoram | up to 1,609 / year | 2022 = largest |
| pamir | varies | |
| hengduan_nyainqentanglha | 146 × 4 | 2023 = smallest pair |
| (others) | varies | |

## 6. Per-band normalisation

Per-band statistics computed via Welford's algorithm on the train split, stored in `dataset_stats_v3.json`:

### Sentinel-2 L2A (12 bands)

```
B01:  mean = 857.6,  std = 626.3
B02:  mean = 1044.0, std = 709.2
B03:  mean = 1356.7, std = 748.7
B04:  mean = 1574.4, std = 825.5
B05:  mean = 1786.2, std = 773.5
B06:  mean = 2076.5, std = 786.1
B07:  mean = 2215.2, std = 810.8
B08:  mean = 2277.8, std = 834.9
B8A:  mean = 2348.5, std = 833.6
B09:  mean = 2243.7, std = 729.6
B11:  mean = 2665.0, std = 875.7
B12:  mean = 2217.2, std = 851.3
```

### Sentinel-1 GRD (2 bands, dB)

```
VV:   mean = -9.25,  std = 5.90
VH:   mean = -18.00, std = 5.92
```

### Copernicus DEM 30 m

```
elevation: mean = 4299.8 m, std = 901.0 m
```

These statistics are loaded by the dataloader and applied as `(x - mean) / std` per band. Inference workflows load the same JSON file to ensure consistency between training and inference normalisation.

## 7. Access

The dataset lives at `abzal-glw/cryosentinel-glof-v3` on Hugging Face and is publicly downloadable. All upstream data sources are open (ESA Copernicus open-access terms for Sentinel-1, Sentinel-2, and Copernicus DEM 30 m; PANGAEA CC-BY for the Kumar & Vijay 2026 inventory). Reuse must respect the attribution requirements listed in §8.

To download:

```bash
huggingface-cli login   # if you have not already
huggingface-cli download \
    --repo-type dataset \
    abzal-glw/cryosentinel-glof-v3 \
    --local-dir ./data/multimodal_chips_v3
```

The full dataset is 30.4 GiB. If you only need the Almaty-corridor subset for the Stage 4b finetune (≈ 10 GiB), use `--allow-patterns "*ile_alatau*" "*tien_shan_full*" "*zhetysu_alatau*"`.

## 8. Licensing

The CryoSentinel-derived dataset (the chip stacks, the per-band statistics, the manifest) is released under the Open Data Commons Attribution License (ODC-By 1.0), which is compatible with the upstream Copernicus and PANGAEA terms.

Reuse must:

- Cite Kumar, R. & Vijay, S. (2026), Inventory of Glacial Lakes in High Mountain Asia for the Years 2016 and 2022, PANGAEA, doi:10.1594/PANGAEA.983845.
- Cite ESA Copernicus for the Sentinel-1, Sentinel-2, and Copernicus DEM imagery.
- Cite Abdrash, A. (2026), CryoSentinel: A Foundation-Model Glacial Lake Segmenter for High Mountain Asia, https://github.com/abzalabdrash/Cryosentinel.
