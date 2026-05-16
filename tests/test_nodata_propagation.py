"""Tests for the ``valid_mask`` nodata-propagation fixes (Phase A.2 / A.3).

Each test synthesises a tiny in-memory Parquet shard with a controlled
nodata pattern in one of the modalities (S2, S1, or DEM), then exercises
``MultiModalChipDataset.__getitem__`` and asserts that the returned
``valid_mask`` correctly excludes those pixels.

Why these tests matter
----------------------
* **S2 nodata** (already handled before this change) \u2014 covered for
  regression.
* **S1 nodata** \u2014 ``np.nan_to_num`` previously replaced NaN/Inf with
  -25 dB which is indistinguishable from real water backscatter. The
  model would learn a spurious "dark SAR \u2192 lake" cue.
* **DEM nodata** \u2014 Copernicus DEM-30 stores nodata as int16 \u00b132768.
  ``np.clip(-32768, -500, 9000)`` previously turned this into a "valid"
  -500 m elevation.

All three sentinel pixels must end up with ``valid_mask = False`` so the
loss skips them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cryosentinel.data.multimodal_dataset import (
    ChipIndex,
    MultiModalChipDataset,
)


CHIP_H, CHIP_W = 16, 16  # tiny chips for fast unit tests; valid_mask logic is shape-agnostic


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def _build_synthetic_shard(
    tmp_path: Path,
    *,
    s2_nodata_mask: np.ndarray | None = None,   # bool [H, W]
    s1_nan_mask: np.ndarray | None = None,      # bool [H, W]
    dem_nodata_mask: np.ndarray | None = None,  # bool [H, W]
) -> Path:
    """Write a 1-row Parquet shard whose chip has the requested nodata pattern.

    Returns the shard path. Data outside the mask is filled with realistic
    typical values so that the chip would otherwise be perfectly valid.
    """
    H, W = CHIP_H, CHIP_W

    # S2: 12 bands of typical reflectance ~ 2000 (uint16). Nodata pixels
    # set to 0 to trigger the v2 GEE unmask(0) branch.
    s2 = np.full((12, H, W), 2000, dtype=np.uint16)
    if s2_nodata_mask is not None:
        s2[:, s2_nodata_mask] = 0

    # S1: 2 bands of typical -15 dB. Nodata set to NaN (Inf also exercises
    # the same branch but NaN is the most common GEE-emitted sentinel).
    s1 = np.full((2, H, W), -15.0, dtype=np.float32)
    if s1_nan_mask is not None:
        s1[:, s1_nan_mask] = np.nan

    # DEM: 1 band of typical 4000 m. Nodata = int16 -32768.
    dem = np.full((1, H, W), 4000, dtype=np.int16)
    if dem_nodata_mask is not None:
        dem[:, dem_nodata_mask] = -32768

    mask = np.zeros((H, W), dtype=np.uint8)

    row = {
        "chip_id": "test_chip",
        "region": "test",
        "snapshot_year": 2022,
        "split": "train",
        "lake_id": "",
        "lat": 35.0,
        "lon": 90.0,
        "water_frac": 0.0,
        "s2_bytes": s2.tobytes(),
        "s1_bytes": s1.tobytes(),
        "dem_bytes": dem.tobytes(),
        "mask_bytes": mask.tobytes(),
        "s2_shape": list(s2.shape),
        "s1_shape": list(s1.shape),
        "dem_shape": list(dem.shape),
        "mask_shape": list(mask.shape),
    }
    table = pa.Table.from_pylist([row])
    region_dir = tmp_path / "test" / "2022"
    region_dir.mkdir(parents=True)
    shard_path = region_dir / "shard_000.parquet"
    pq.write_table(table, shard_path)
    return shard_path


def _load_one_chip(shard_path: Path) -> dict:
    """Build a 1-element dataset over the synthetic shard and return chip 0."""
    idx = [ChipIndex(
        shard=shard_path, row=0, split="train",
        region="test", year=2022,
    )]
    ds = MultiModalChipDataset(idx, augment=False, return_meta=False)
    return ds[0]


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────
def test_valid_mask_clean_chip_is_all_true(tmp_path: Path):
    """A chip without any nodata yields ``valid_mask`` all True."""
    import torch
    shard = _build_synthetic_shard(tmp_path)
    sample = _load_one_chip(shard)
    vm = sample["valid_mask"]
    assert isinstance(vm, torch.Tensor) and vm.dtype == torch.bool
    assert vm.all().item(), "clean chip should have all valid pixels"


def test_s2_nodata_propagates_to_valid_mask(tmp_path: Path):
    """Pixels where all 12 S2 bands are 0 (GEE unmask sentinel) \u2192 valid_mask False."""
    nodata = np.zeros((CHIP_H, CHIP_W), dtype=bool)
    nodata[0:4, 0:4] = True  # 4\u00d74 nodata patch
    shard = _build_synthetic_shard(tmp_path, s2_nodata_mask=nodata)
    sample = _load_one_chip(shard)
    vm = sample["valid_mask"].numpy()
    assert (vm[0:4, 0:4] == False).all(), "S2 nodata patch must be masked out"
    assert (vm[5:, 5:] == True).all(), "non-nodata region must remain valid"


def test_s1_nan_propagates_to_valid_mask(tmp_path: Path):
    """Pixels where any S1 band is NaN \u2192 valid_mask False (regardless of nan_to_num)."""
    nodata = np.zeros((CHIP_H, CHIP_W), dtype=bool)
    nodata[8:12, 8:12] = True
    shard = _build_synthetic_shard(tmp_path, s1_nan_mask=nodata)
    sample = _load_one_chip(shard)
    vm = sample["valid_mask"].numpy()
    assert (vm[8:12, 8:12] == False).all(), "S1 NaN patch must be masked out"
    # Crucial: even though nan_to_num filled S1 with -25 dB, valid_mask
    # still reflects the *original* NaN. Verify the post-nan_to_num S1 is
    # not NaN (so the network can ingest it without crashing) but masked.
    s1 = sample["S1GRD"].numpy()
    assert np.isfinite(s1).all(), "S1 tensor must be finite after nan_to_num"


def test_dem_nodata_propagates_to_valid_mask(tmp_path: Path):
    """Pixels where DEM is the int16 sentinel (-32768) \u2192 valid_mask False."""
    nodata = np.zeros((CHIP_H, CHIP_W), dtype=bool)
    nodata[2:6, 10:14] = True
    shard = _build_synthetic_shard(tmp_path, dem_nodata_mask=nodata)
    sample = _load_one_chip(shard)
    vm = sample["valid_mask"].numpy()
    assert (vm[2:6, 10:14] == False).all(), "DEM nodata patch must be masked out"
    # Verify the post-clip DEM is finite (no -32768 leaking into the network)
    dem = sample["DEM"].numpy()
    assert np.isfinite(dem).all()


def test_combined_nodata_propagates(tmp_path: Path):
    """All three nodata patterns combine correctly via OR."""
    s2_no = np.zeros((CHIP_H, CHIP_W), dtype=bool); s2_no[0:2, :] = True
    s1_no = np.zeros((CHIP_H, CHIP_W), dtype=bool); s1_no[:, 0:2] = True
    dem_no = np.zeros((CHIP_H, CHIP_W), dtype=bool); dem_no[14:, 14:] = True

    shard = _build_synthetic_shard(
        tmp_path,
        s2_nodata_mask=s2_no,
        s1_nan_mask=s1_no,
        dem_nodata_mask=dem_no,
    )
    sample = _load_one_chip(shard)
    vm = sample["valid_mask"].numpy()
    expected_invalid = s2_no | s1_no | dem_no
    assert (vm == ~expected_invalid).all(), (
        "combined valid_mask must equal NOT(s2_nodata | s1_nodata | dem_nodata)"
    )


def test_dem_positive_sentinel_also_caught(tmp_path: Path):
    """Some int16 DEM tiles use +32767 as nodata \u2014 also captured."""
    H, W = CHIP_H, CHIP_W
    s2 = np.full((12, H, W), 2000, dtype=np.uint16)
    s1 = np.full((2, H, W), -15.0, dtype=np.float32)
    dem = np.full((1, H, W), 4000, dtype=np.int16)
    dem[0, 4:8, 4:8] = 32767                         # positive sentinel
    mask = np.zeros((H, W), dtype=np.uint8)

    row = {
        "chip_id": "test_chip", "region": "test", "snapshot_year": 2022,
        "split": "train", "lake_id": "", "lat": 35.0, "lon": 90.0,
        "water_frac": 0.0,
        "s2_bytes": s2.tobytes(), "s1_bytes": s1.tobytes(),
        "dem_bytes": dem.tobytes(), "mask_bytes": mask.tobytes(),
        "s2_shape": list(s2.shape), "s1_shape": list(s1.shape),
        "dem_shape": list(dem.shape), "mask_shape": list(mask.shape),
    }
    region_dir = tmp_path / "test" / "2022"
    region_dir.mkdir(parents=True)
    shard_path = region_dir / "shard_000.parquet"
    pq.write_table(pa.Table.from_pylist([row]), shard_path)

    sample = _load_one_chip(shard_path)
    vm = sample["valid_mask"].numpy()
    assert (vm[4:8, 4:8] == False).all()
    assert vm[0, 0].item() is True  # untouched corner stays valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
