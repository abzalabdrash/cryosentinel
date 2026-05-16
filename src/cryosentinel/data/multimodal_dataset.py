"""Multi-modal Parquet dataset for CryoSentinel Stage-3.

Reads precomputed multimodal chip shards and emits dicts
ready to be fed into a TerraMind v1 backbone:

    {
        "S2L2A":      float32 [12, 224, 224]   # standardised by TerraMind v1 stats
        "S1GRD":      float32 [ 2, 224, 224]   # VV, VH (dB), standardised
        "DEM":        float32 [ 1, 224, 224]   # metres, standardised
        "mask":       int64   [224, 224]       # 0 = land, 1 = lake
        "valid_mask": bool    [224, 224]       # True = valid S2 pixel (not nodata)
        "chip_id":    str     (if return_meta)
        "region":     str     (if return_meta)
        "snapshot_year": int  (if return_meta)
        "split":      str     (if return_meta)
    }

Design highlights
-----------------
* ``split`` column propagated from the per-chip metadata → no lake leakage.
* Per-shard lazy LRU cache keeps RAM bounded even with hundreds of shards.
* **Nodata detection (v2 data)**: GEE ``unmask(0)`` fills masked pixels with 0
  for all 12 S2 bands. We detect these as ``(s2 == 0).all(axis=0)`` and
  expose them as ``valid_mask = False``. Training losses use this to skip
  nodata pixels rather than learning spurious "all-zero → no lake" shortcuts.
  For legacy v1 data (nodata = 16000 after clip from uint16 32768) we also
  catch ``(s2 == 16000).all(axis=0)`` as nodata.
* Augmentations are flip-only by default to remain safe for ViT absolute
  pos-emb encoders (rotations degrade TerraMind / Prithvi). ``valid_mask``
  is flipped consistently with image and mask.
* **Spatial block split (Phase A.1)**: chips can be partitioned into
  train/val/test by hashing their containing ~25 km block instead of by
  their closest lake. This eliminates the chip-overlap leakage that the
  old lake-level split allowed (50 % stride → adjacent chips share half
  their pixels but could land in different splits). Buffer chips near
  block boundaries are dropped to mathematically preclude any spatial
  overlap between splits. See :mod:`cryosentinel.data.block_split` and
  ``tests/test_block_split.py`` for the validation.
* **Nodata propagation (Phase A.2 / A.3)**: in addition to S2 nodata, the
  ``valid_mask`` now also excludes pixels where the DEM is the int16
  sentinel (±32000+) or where S1 is non-finite. Without these the model
  silently learned spurious "low-elev → water" / "dark-SAR → water"
  associations on hidden nodata pixels.
"""
from __future__ import annotations

import functools
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from .block_split import (
    BlockSplitGrid,
    DEFAULT_BLOCK_SIZE_DEG,
    DEFAULT_BUFFER_DEG,
    DEFAULT_SALT,
)

try:
    import lightning.pytorch as pl              # type: ignore
    _BASE_DM = pl.LightningDataModule
except ImportError:
    try:
        import pytorch_lightning as pl          # type: ignore
        _BASE_DM = pl.LightningDataModule
    except ImportError:
        _BASE_DM = object  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
#  TerraMind v1 official pre-training statistics (from terratorch source)
# ──────────────────────────────────────────────────────────────────────
TERRAMIND_S2L2A_MEAN: tuple[float, ...] = (
    1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975,
    2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648,
)
TERRAMIND_S2L2A_STD: tuple[float, ...] = (
    2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
    1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311,
)
TERRAMIND_S1GRD_MEAN: tuple[float, ...] = (-12.599, -20.293)
TERRAMIND_S1GRD_STD: tuple[float, ...]  = (5.195, 5.890)
TERRAMIND_DEM_MEAN: tuple[float, ...]   = (670.665,)
TERRAMIND_DEM_STD: tuple[float, ...]    = (951.272,)

# Nodata sanitation bounds (raw values BEFORE standardisation)
S2_CLIP  = (0, 16_000)      # uint16 reflectance × 10_000
DEM_CLIP = (-500, 9_000)    # metres


# ──────────────────────────────────────────────────────────────────────
#  Dataset-specific normalisation stats (Phase B of SOTA roadmap)
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NormStats:
    """Per-modality mean/std tuples for channel-wise standardisation.

    The TerraMind v1 pretraining stats (TERRAMIND_*) are a reasonable prior
    for most of the globe, but they have a critical mismatch in HMA: the
    pretrain DEM mean is ~670 m while HMA chips have a true mean of ~3.5–
    4.5 km. Wrong DEM standardisation leaves the model's DEM channel in an
    out-of-distribution regime for the entire pretraining feature space,
    which is estimated to cost +1.5…+3.0 IoU on glacial-lake segmentation.

    Use :func:`load_norm_stats` or :meth:`NormStats.from_json` to load dataset-
    specific stats stored with the released dataset.
    """
    s2_mean: tuple[float, ...]
    s2_std:  tuple[float, ...]
    s1_mean: tuple[float, ...]
    s1_std:  tuple[float, ...]
    dem_mean: tuple[float, ...]
    dem_std:  tuple[float, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "NormStats":
        """Load from a ``dataset_stats_*.json`` file.

        Expected schema::

            {"S2L2A": {"means": [...12...], "stds": [...12...]},
             "S1GRD": {"means": [...2...],  "stds": [...2...]},
             "DEM":   {"means": [...1...],  "stds": [...1...]}}
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            s2_mean=tuple(data["S2L2A"]["means"]),
            s2_std=tuple(data["S2L2A"]["stds"]),
            s1_mean=tuple(data["S1GRD"]["means"]),
            s1_std=tuple(data["S1GRD"]["stds"]),
            dem_mean=tuple(data["DEM"]["means"]),
            dem_std=tuple(data["DEM"]["stds"]),
        )

    @classmethod
    def terramind_default(cls) -> "NormStats":
        """TerraMind v1 pretrain stats — fallback when no JSON is provided."""
        return cls(
            s2_mean=TERRAMIND_S2L2A_MEAN,
            s2_std=TERRAMIND_S2L2A_STD,
            s1_mean=TERRAMIND_S1GRD_MEAN,
            s1_std=TERRAMIND_S1GRD_STD,
            dem_mean=TERRAMIND_DEM_MEAN,
            dem_std=TERRAMIND_DEM_STD,
        )


def load_norm_stats(path: str | Path | None) -> NormStats:
    """Load normalisation stats; fallback to TerraMind defaults if ``path`` is None/missing.

    If ``path`` is provided but does not exist, prints a warning and falls
    back silently so training pipelines don't explode when the stats JSON is
    computed lazily on a remote volume.
    """
    if path is None:
        return NormStats.terramind_default()
    p = Path(path)
    if not p.exists():
        print(f"[norm_stats] WARN: {p} not found — falling back to TerraMind defaults")
        return NormStats.terramind_default()
    print(f"[norm_stats] loading dataset-specific stats from {p}")
    stats = NormStats.from_json(p)
    # Quick sanity: HMA DEM mean should be ≳2000 m. If <1000 m the JSON is
    # probably from a non-HMA corpus and we'd rather fail loudly.
    if stats.dem_mean and stats.dem_mean[0] < 1000:
        print(f"[norm_stats] WARN: DEM mean={stats.dem_mean[0]:.0f} m looks low "
              f"for HMA. Verify {p} was computed over the right volume.")
    return stats


# ──────────────────────────────────────────────────────────────────────
#  Manifest construction
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ChipIndex:
    """Lightweight pointer into a Parquet shard.

    ``split`` is the *effective* split (block-based when block split is
    enabled, lake-based otherwise). ``lake_split`` keeps the original
    lake-level assignment for diagnostics / ablation studies.

    ``water_frac`` is pulled eagerly so downstream samplers (Phase D.5
    hard-negative weighting) can classify chips without touching the
    underlying Parquet at sampler-build time.
    """
    shard: Path
    row:   int
    split: str
    region: str
    year:  int
    lat:   float = 0.0
    lon:   float = 0.0
    lake_split: str = "UNKNOWN"
    water_frac: float = 0.0


def _scan_shards(
    data_dir: Path,
    regions: Sequence[str] | None = None,
    years:   Sequence[int] | None = None,
) -> list[Path]:
    """Return all ``shard_*.parquet`` matching region/year filters."""
    shards: list[Path] = []
    for shard in sorted(data_dir.rglob("shard_*.parquet")):
        try:
            year   = int(shard.parent.name)
            region = shard.parent.parent.name
        except (ValueError, IndexError):
            continue
        if regions is not None and region not in regions:
            continue
        if years is not None and year not in years:
            continue
        shards.append(shard)
    return shards


def _build_index(
    shards: list[Path],
    splits: Sequence[str] | None = None,
    block_split: BlockSplitGrid | None = None,
) -> list[ChipIndex]:
    """Read metadata columns from each shard to build a global chip index.

    If ``block_split`` is provided, the chip's ``split`` field is
    *overridden* by the spatial block-split lookup based on the chip's
    (lat, lon) centre. Chips landing in the block-buffer (within
    ``buffer_deg`` of any block boundary) are dropped entirely — they
    would leak between splits because their 224×224 spatial extent
    crosses into a neighbouring block.

    The original lake-level split is always preserved as ``lake_split``
    for diagnostics and ablation comparisons.

    Args:
        shards: list of Parquet shard paths.
        splits: optional whitelist (e.g. ``{"train", "val"}``) applied to
            the *effective* split, after block remapping.
        block_split: when not None, switches to spatial block split.
    """
    keep_splits = set(splits) if splits is not None else None
    # lat/lon are always read when present — downstream consumers
    # (lake-registry polygonization, georeferencing) need them even
    # when block_split is disabled. _read_best falls back gracefully
    # if a legacy shard lacks lat/lon.
    cols_to_read = ["split", "lat", "lon"]
    # water_frac is read opportunistically: legacy v1 shards may not have
    # it, in which case we fall back to 0.0 and the hard-negative sampler
    # simply treats all chips as "background" (weights collapse to uniform).
    cols_to_read += ["water_frac"]

    def _read_best(shard: Path):
        """Read the largest subset of ``cols_to_read`` this shard has."""
        for cols in (cols_to_read,
                      [c for c in cols_to_read if c != "water_frac"],
                      ["split", "lat", "lon"],
                      ["split"]):
            try:
                return pq.read_table(shard, columns=cols)
            except Exception:  # noqa: BLE001
                continue
        return None

    index: list[ChipIndex] = []
    n_buffer = 0
    for shard in shards:
        try:
            year   = int(shard.parent.name)
            region = shard.parent.parent.name
        except (ValueError, IndexError):
            continue

        tbl = _read_best(shard)
        if tbl is None:
            continue
        split_col = tbl.column("split").to_pylist()

        # lat/lon may be missing on legacy shards — fall back gracefully.
        # We always populate them when present (regardless of block_split)
        # because downstream consumers (e.g. lake-registry polygon
        # extraction) need ChipIndex.lat / .lon for georeferencing.
        if "lat" in tbl.schema.names and "lon" in tbl.schema.names:
            lat_col = tbl.column("lat").to_pylist()
            lon_col = tbl.column("lon").to_pylist()
        else:
            lat_col = lon_col = None

        wf_col = tbl.column("water_frac").to_pylist() if "water_frac" in tbl.schema.names else None

        for row, lake_split_raw in enumerate(split_col):
            lake_split = lake_split_raw or "UNKNOWN"

            # Read lat/lon when available — always needed for downstream
            # georeferencing (independent of split selection).
            if lat_col is not None:
                lat = float(lat_col[row])
                lon = float(lon_col[row])
            else:
                lat = lon = 0.0

            # Split selection — block-split path only when enabled AND
            # we have valid lat/lon.
            if block_split is not None and lat_col is not None:
                effective_split = block_split.split_for(lat, lon)
                if effective_split == "BUFFER":
                    n_buffer += 1
                    continue
            else:
                effective_split = lake_split

            if keep_splits is not None and effective_split not in keep_splits:
                continue

            water_frac = float(wf_col[row]) if wf_col is not None else 0.0

            index.append(ChipIndex(
                shard=shard, row=row, split=effective_split,
                region=region, year=year,
                lat=lat, lon=lon, lake_split=lake_split,
                water_frac=water_frac,
            ))

    if block_split is not None and n_buffer > 0:
        print(f"[block_split] dropped {n_buffer} boundary-buffer chips "
              f"(within {block_split.buffer_deg}° of a block edge)")
    return index


# ──────────────────────────────────────────────────────────────────────
#  Per-shard LRU cache (worker-local)
# ──────────────────────────────────────────────────────────────────────
class _ShardCache:
    """Tiny LRU cache holding Arrow tables (dict-of-lists) in RAM."""

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._cache: "OrderedDict[Path, dict]" = OrderedDict()

    def get(self, shard: Path) -> dict:
        if shard in self._cache:
            self._cache.move_to_end(shard)
            return self._cache[shard]
        tbl  = pq.read_table(shard)
        cols = {name: tbl.column(name).to_pylist() for name in tbl.schema.names}
        self._cache[shard] = cols
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return cols


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────
class MultiModalChipDataset(Dataset):
    """PyTorch Dataset of multi-modal glacial-lake chips."""

    def __init__(
        self,
        index: list[ChipIndex],
        *,
        augment: bool = False,
        rot90: bool = False,
        return_meta: bool = False,
        shard_cache_size: int = 2,
        norm_stats: NormStats | None = None,
        sample_augs: "Sequence[Callable[[dict], dict]] | None" = None,
    ):
        """Parameters
        ----------
        sample_augs : Sequence[Callable] | None
            Additional sample-level augmentations applied AFTER the built-in
            flip / rot90. Each callable takes a sample dict and returns a
            modified sample dict. Use ``None`` (default) to disable.
            Typical values: ``[SpectralJitter(p=0.5), MultiScale(p=0.5)]``.
        """
        self.index = index
        self.augment = augment
        self.rot90 = rot90
        self.return_meta = return_meta
        self._cache_size = shard_cache_size
        self.sample_augs = list(sample_augs) if sample_augs else []
        self._cache: _ShardCache | None = None

        # Precomputed normalisation tensors
        stats = norm_stats if norm_stats is not None else NormStats.terramind_default()
        self._norm_stats = stats
        self._s2_mean  = torch.tensor(stats.s2_mean).view(len(stats.s2_mean), 1, 1)
        self._s2_std   = torch.tensor(stats.s2_std).view(len(stats.s2_std), 1, 1)
        self._s1_mean  = torch.tensor(stats.s1_mean).view(len(stats.s1_mean), 1, 1)
        self._s1_std   = torch.tensor(stats.s1_std).view(len(stats.s1_std), 1, 1)
        self._dem_mean = torch.tensor(stats.dem_mean).view(len(stats.dem_mean), 1, 1)
        self._dem_std  = torch.tensor(stats.dem_std).view(len(stats.dem_std), 1, 1)

    def __len__(self) -> int:
        return len(self.index)

    def _ensure_cache(self) -> _ShardCache:
        if self._cache is None:
            self._cache = _ShardCache(capacity=self._cache_size)
        return self._cache

    def __getitem__(self, i: int) -> dict:
        ent   = self.index[i]
        cache = self._ensure_cache()
        cols  = cache.get(ent.shard)
        row   = ent.row

        # ── Decode raw bytes → ndarrays ─────────────────────────────────
        s2 = np.frombuffer(cols["s2_bytes"][row], dtype=np.uint16).reshape(
            cols["s2_shape"][row]
        ).astype(np.float32)
        s1 = np.frombuffer(cols["s1_bytes"][row], dtype=np.float32).reshape(
            cols["s1_shape"][row]
        ).astype(np.float32)
        dem = np.frombuffer(cols["dem_bytes"][row], dtype=np.int16).reshape(
            cols["dem_shape"][row]
        ).astype(np.float32)
        mask = np.frombuffer(cols["mask_bytes"][row], dtype=np.uint8).reshape(
            cols["mask_shape"][row]
        ).astype(np.int64)

        # ── Nodata detection (BEFORE clip / nan_to_num) ────────────────
        # The order matters: every sanitisation step that follows would
        # silently turn nodata into a plausible "valid" value, so we must
        # capture all sentinels first and propagate them through
        # ``valid_mask`` so the loss skips them.

        # S2: v2 GEE unmask(0) → all 12 bands are exactly 0; v1 → 16000.
        nodata_v2 = (s2 == 0.0).all(axis=0)        # shape [H, W]
        s2 = np.clip(s2, *S2_CLIP)                  # clip before v1 check
        nodata_v1 = (s2 == float(S2_CLIP[1])).all(axis=0)

        # S1: NaN/Inf would otherwise be replaced by -25 dB → mimics water
        # backscatter (-15..-25 dB) and the model learns spurious
        # "dark SAR → lake". Detect non-finite pixels in *any* band before
        # nan_to_num replaces them.
        s1_nodata = ~np.isfinite(s1).all(axis=0)   # [H, W]

        # DEM: Copernicus DEM-30 nodata is int16 ±32768. After
        # ``np.clip(-32768, -500, 9000)`` it would become -500 m and
        # standardise to a "valid" coastal elevation → spurious
        # "low-elev → water" cue. Detect *before* clipping. ``dem`` here is
        # already cast from int16 to float32 (no precision loss), so the
        # ±32000 thresholds catch both sentinel values with safety margin.
        dem_nodata = ((dem <= -32000.0) | (dem >= 32000.0)).any(axis=0)

        valid_np = ~(nodata_v2 | nodata_v1 | s1_nodata | dem_nodata)

        # ── Remaining sanitation ────────────────────────────────────────
        dem = np.clip(dem, *DEM_CLIP)
        s1  = np.nan_to_num(s1, nan=-25.0, posinf=5.0, neginf=-30.0)

        # ── To tensors + standardise ────────────────────────────────────
        s2_t   = (torch.from_numpy(s2)   - self._s2_mean)  / self._s2_std
        s1_t   = (torch.from_numpy(s1)   - self._s1_mean)  / self._s1_std
        dem_t  = (torch.from_numpy(dem)  - self._dem_mean) / self._dem_std
        mask_t = torch.from_numpy(mask)
        vm_t   = torch.from_numpy(valid_np)          # bool [H, W]

        # ── Augmentation (flip-safe for ViT pos-emb) ───────────────────
        if self.augment:
            if torch.rand(()) < 0.5:                 # horizontal flip
                s2_t   = torch.flip(s2_t,   dims=[-1])
                s1_t   = torch.flip(s1_t,   dims=[-1])
                dem_t  = torch.flip(dem_t,  dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
                vm_t   = torch.flip(vm_t,   dims=[-1])
            if torch.rand(()) < 0.5:                 # vertical flip
                s2_t   = torch.flip(s2_t,   dims=[-2])
                s1_t   = torch.flip(s1_t,   dims=[-2])
                dem_t  = torch.flip(dem_t,  dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])
                vm_t   = torch.flip(vm_t,   dims=[-2])
            if self.rot90 and torch.rand(()) < 0.5:
                k = int(torch.randint(1, 4, ()).item())
                s2_t   = torch.rot90(s2_t,   k, dims=(-2, -1))
                s1_t   = torch.rot90(s1_t,   k, dims=(-2, -1))
                dem_t  = torch.rot90(dem_t,  k, dims=(-2, -1))
                mask_t = torch.rot90(mask_t, k, dims=(-2, -1))
                vm_t   = torch.rot90(vm_t,   k, dims=(-2, -1))

        sample: dict = {
            "S2L2A":      s2_t.contiguous(),
            "S1GRD":      s1_t.contiguous(),
            "DEM":        dem_t.contiguous(),
            "mask":       mask_t.contiguous(),
            "valid_mask": vm_t.contiguous(),  # bool [H, W]
        }

        # ── Sample-level augmentations (Phase D.4) ─────────────────────
        # Applied only at training; ``self.augment`` already gates flip/rot90,
        # so we follow the same convention here for consistency.
        if self.augment and self.sample_augs:
            for aug in self.sample_augs:
                sample = aug(sample)

        if self.return_meta:
            sample["chip_id"]       = cols["chip_id"][row]
            sample["region"]        = ent.region
            sample["snapshot_year"] = ent.year
            sample["split"]         = ent.split
            sample["water_frac"]    = float(cols["water_frac"][row])
        return sample


# ──────────────────────────────────────────────────────────────────────
#  Lightning DataModule
# ──────────────────────────────────────────────────────────────────────
class MultiModalDataModule(_BASE_DM):  # type: ignore[misc, valid-type]
    """Wraps :class:`MultiModalChipDataset` for PyTorch Lightning.

    UNKNOWN-split chips are folded into train to keep the pre-training corpus
    large (they come from regions without a lake-level split CSV).
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        batch_size: int = 8,
        num_workers: int = 4,
        augment: bool = True,
        rot90: bool = False,
        regions: Iterable[str] | None = None,
        years: Iterable[int] | None = None,
        include_unknown_in_train: bool = True,
        shard_cache_size: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # ── Spatial block split (Phase A.1) ────────────────────────────
        use_block_split: bool = True,
        block_size_deg: float = DEFAULT_BLOCK_SIZE_DEG,
        block_buffer_deg: float = DEFAULT_BUFFER_DEG,
        block_salt: str = DEFAULT_SALT,
        # ── Dataset-specific normalisation (Phase B) ───────────────────
        norm_stats_path: str | Path | None = None,
        # ── Sample-level augmentations (Phase D.4) ─────────────────────
        spectral_jitter: dict | None = None,
        multi_scale: dict | None = None,
        # ── Batch-level augmentations (Phase D.4) ──────────────────────
        copy_paste: dict | None = None,
        mosaic: dict | None = None,
        # ── Hard-negative weighted sampler (Phase D.5) ─────────────────
        hard_neg_sampler: dict | None = None,
        hard_neg_water_frac_threshold: float = 0.005,
        # ── Sequential-read IO escape hatch ─────────────────────────────
        # On cold remote volumes the per-shard read is expensive, and the
        # default RandomSampler scatters batch indices across many shards.
        # Setting ``shuffle_train: false`` switches train_dataloader to
        # SequentialSampler so chips are read shard-by-shard, hitting the
        # _ShardCache LRU on >99 % of accesses. This makes training
        # tractable on a cold volume in exchange for a small loss of
        # gradient noise (we can re-enable shuffle once the volume is
        # warmed up or after a one-time prefetch pass).
        shuffle_train: bool = True,
    ) -> None:
        """Parameters (Phase D additions)
        -------------------------------
        spectral_jitter, multi_scale : dict | None
            kwargs forwarded to :class:`SpectralJitter` /
            :class:`MultiScale`. ``None`` disables. Typical:
            ``{"p": 0.5, "eps_s2": 0.05}``.
        copy_paste, mosaic : dict | None
            kwargs forwarded to :class:`CopyPaste` / :class:`Mosaic`.
            ``None`` disables. When either is set the ``train_dataloader``
            uses a :class:`MosaicCopyPasteCollator` custom collate fn.
        hard_neg_sampler : dict | None
            kwargs forwarded to :class:`HardNegativeWeightedSampler`
            (``positive_to_negative_ratio``, etc.). ``None`` keeps plain
            shuffle-based sampling.
        hard_neg_water_frac_threshold : float
            ``water_frac`` threshold above which a chip counts as "positive"
            for the hard-neg sampler. Default 0.005 (matches
            ``--water-frac-min 0.002`` ingest with a ≥0.3 % safety margin).
        """
        if _BASE_DM is not object:
            super().__init__()
        self.data_dir   = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment    = augment
        self.rot90      = rot90
        self.regions    = list(regions) if regions is not None else None
        self.years      = list(years)   if years   is not None else None
        self.include_unknown_in_train = include_unknown_in_train
        self.shard_cache_size = shard_cache_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.shuffle_train = shuffle_train
        # Block-split parameters
        self.use_block_split = use_block_split
        self.block_size_deg = block_size_deg
        self.block_buffer_deg = block_buffer_deg
        self.block_salt = block_salt
        # Normalisation stats — dataset-specific JSON (Phase B) or TM defaults
        self.norm_stats_path = norm_stats_path
        self._norm_stats = load_norm_stats(norm_stats_path)
        # Phase D.4 / D.5 config (stored as dict → instantiated lazily in setup())
        self._spectral_jitter_cfg = spectral_jitter
        self._multi_scale_cfg = multi_scale
        self._copy_paste_cfg = copy_paste
        self._mosaic_cfg = mosaic
        self._hard_neg_sampler_cfg = hard_neg_sampler
        self._hard_neg_water_frac_threshold = hard_neg_water_frac_threshold
        # Computed lazily in setup()
        self._train_collate_fn: Callable | None = None
        self._train_sampler = None

        self.train_dataset: MultiModalChipDataset | None = None
        self.val_dataset:   MultiModalChipDataset | None = None
        self.test_dataset:  MultiModalChipDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        shards = _scan_shards(self.data_dir, self.regions, self.years)
        if not shards:
            raise FileNotFoundError(
                f"No shard_*.parquet found under {self.data_dir!s} "
                f"(regions={self.regions}, years={self.years}). "
                f"Verify the dataset download or ingestion step first."
            )

        block_split: BlockSplitGrid | None = None
        if self.use_block_split:
            block_split = BlockSplitGrid(
                block_size_deg=self.block_size_deg,
                buffer_deg=self.block_buffer_deg,
                salt=self.block_salt,
            )
            lat_km, lon_km = block_split.block_size_km(lat=35.0)
            chip_ok_km = block_split.chip_half_extent_km_required(max_lat_deg=45.0)
            print(
                f"[MultiModalDataModule] block split ON  "
                f"size={self.block_size_deg}° (≈{lat_km:.0f}×{lon_km:.0f} km)  "
                f"buffer={self.block_buffer_deg}° (covers chips up to {chip_ok_km:.2f} km half-extent)"
            )

        idx_full = _build_index(shards, block_split=block_split)

        train, val, test = [], [], []
        for ent in idx_full:
            if ent.split == "train":
                train.append(ent)
            elif ent.split == "val":
                val.append(ent)
            elif ent.split == "test":
                test.append(ent)
            elif ent.split == "UNKNOWN" and self.include_unknown_in_train:
                # Only happens in legacy lake-split mode — block_split
                # always returns one of {train, val, test, BUFFER} and
                # BUFFER chips are filtered upstream in _build_index.
                train.append(ent)

        # ── Phase D.4 sample-level augmentations ───────────────────────
        sample_augs: list = []
        if self._spectral_jitter_cfg is not None:
            from .augmentations import SpectralJitter
            sample_augs.append(SpectralJitter(**self._spectral_jitter_cfg))
        if self._multi_scale_cfg is not None:
            from .augmentations import MultiScale
            sample_augs.append(MultiScale(**self._multi_scale_cfg))

        self.train_dataset = MultiModalChipDataset(
            train, augment=self.augment, rot90=self.rot90,
            shard_cache_size=self.shard_cache_size,
            norm_stats=self._norm_stats,
            sample_augs=sample_augs or None,
        )
        self.val_dataset = MultiModalChipDataset(
            val, augment=False, rot90=False,
            shard_cache_size=self.shard_cache_size,
            norm_stats=self._norm_stats,
        )
        self.test_dataset = MultiModalChipDataset(
            test, augment=False, rot90=False, return_meta=True,
            shard_cache_size=self.shard_cache_size,
            norm_stats=self._norm_stats,
        )

        # ── Phase D.4 batch-level collator (Mosaic / CopyPaste) ────────
        if self._mosaic_cfg is not None or self._copy_paste_cfg is not None:
            from .augmentations import (
                CopyPaste,
                Mosaic,
                MosaicCopyPasteCollator,
            )
            self._train_collate_fn = MosaicCopyPasteCollator(
                mosaic=Mosaic(**self._mosaic_cfg) if self._mosaic_cfg else None,
                copy_paste=CopyPaste(**self._copy_paste_cfg) if self._copy_paste_cfg else None,
            )

        # ── Phase D.5 hard-negative weighted sampler ───────────────────
        if self._hard_neg_sampler_cfg is not None:
            from .sampler import HardNegativeWeightedSampler
            thr = self._hard_neg_water_frac_threshold
            is_pos = [e.water_frac >= thr for e in train]
            self._train_sampler = HardNegativeWeightedSampler(
                is_pos, **self._hard_neg_sampler_cfg,
            )
            n_pos = self._train_sampler.n_positive
            n_neg = self._train_sampler.n_negative
            print(
                f"[MultiModalDataModule] hard-neg sampler ON  "
                f"threshold={thr}  n_pos={n_pos}  n_neg={n_neg}  "
                f"ratio={self._hard_neg_sampler_cfg.get('positive_to_negative_ratio', 3.0)}"
            )

        n_shards = len(shards)
        n_total  = len(idx_full)
        split_mode = "BLOCK" if self.use_block_split else "LAKE"
        print(
            f"[MultiModalDataModule] {split_mode}-split: {n_shards} shards, {n_total} chips → "
            f"train {len(train)} | val {len(val)} | test {len(test)}"
        )
        if sample_augs:
            print(
                f"[MultiModalDataModule] sample-level augs: "
                f"{[type(a).__name__ for a in sample_augs]}"
            )
        if self._train_collate_fn is not None:
            print(
                f"[MultiModalDataModule] batch-level augs: "
                f"mosaic={self._mosaic_cfg is not None} "
                f"copy_paste={self._copy_paste_cfg is not None}"
            )

    def _loader(
        self,
        ds: Dataset,
        *,
        shuffle: bool,
        sampler=None,
        collate_fn: Callable | None = None,
    ) -> DataLoader:
        # If a sampler is given Lightning requires shuffle=False on the loader.
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=shuffle,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call .setup() first"
        return self._loader(
            self.train_dataset,
            shuffle=self.shuffle_train,
            sampler=self._train_sampler,
            collate_fn=self._train_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call .setup() first"
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Call .setup() first"
        return self._loader(self.test_dataset, shuffle=False)


# ──────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed/multimodal_chips")
    ap.add_argument("--regions",  nargs="*", default=None)
    ap.add_argument("--years",    nargs="*", type=int, default=None)
    args = ap.parse_args()

    dm = MultiModalDataModule(
        data_dir=args.data_dir,
        batch_size=4,
        num_workers=0,
        augment=True,
        regions=args.regions,
        years=args.years,
    )
    dm.setup()
    if dm.train_dataset is None or len(dm.train_dataset) == 0:
        sys.exit("no train chips found")

    sample = dm.train_dataset[0]
    print("sample keys:", list(sample.keys()))
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            stats = f"shape={tuple(v.shape)}  dtype={v.dtype}"
            if v.is_floating_point():
                stats += f"  min={v.min().item():.3f}  max={v.max().item():.3f}"
            elif v.dtype == torch.bool:
                valid_pct = 100.0 * v.float().mean().item()
                stats += f"  valid_pct={valid_pct:.1f}%"
            else:
                stats += f"  unique={v.unique().tolist()}"
            print(f"  {k:12s}  {stats}")
    print("OK.")
