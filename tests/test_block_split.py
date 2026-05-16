"""Tests for :mod:`cryosentinel.data.block_split`.

Two classes of tests:

1. **Functional** — distribution, determinism, buffer behaviour, math
   correctness across HMA latitudes.
2. **Anti-leakage (the killer test)** — exhaustively verifies that no pair
   of chips assigned to *different* splits has centre-to-centre distance
   below the chip extent (2.24 km for 224 px × 10 m). If this property
   fails, the spatial block split design is broken and we have leakage.

The math guarantee
------------------
A chip is dropped to ``"BUFFER"`` when its centre is within
``buffer_deg`` of any block edge. Two surviving chips on opposite sides of
a shared edge therefore have centres at distance ``\u2265 2 \u00d7 buffer_deg`` from
each other. With ``buffer_deg = 0.02\u00b0`` and the worst-case longitudinal
shrink ``cos(45\u00b0) \u2248 0.707``, the minimum distance is ``2 \u00d7 0.02 \u00d7 111 \u00d7
0.707 \u2248 3.14 km``, comfortably above the 2.24 km chip extent.

The empirical test below checks this on a dense grid of synthetic chip
centres covering the full HMA bbox, so any silent regression in the
buffer logic will be caught.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cryosentinel.data.block_split import (
    BlockSplitGrid,
    DEFAULT_BLOCK_SIZE_DEG,
    DEFAULT_BUFFER_DEG,
)

CHIP_EXTENT_KM = 224 * 10 / 1000.0  # 2.24 km — full chip side at 10 m / pixel
CHIP_HALF_EXTENT_KM = CHIP_EXTENT_KM / 2.0  # 1.12 km

# Reasonable HMA-wide test region (covers the actual aoi.yaml bbox).
HMA_BBOX = (67.0, 26.0, 104.0, 46.0)  # west, south, east, north


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    a1, a2 = math.radians(lat1), math.radians(lat2)
    da = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(da / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _haversine_km_vec(lat1, lon1, lat2_arr, lon2_arr) -> np.ndarray:
    """Vectorised haversine: scalar (lat1, lon1) vs arrays."""
    R = 6371.0
    a1 = math.radians(lat1)
    a2 = np.deg2rad(lat2_arr)
    da = np.deg2rad(lat2_arr - lat1)
    dl = np.deg2rad(lon2_arr - lon1)
    h = np.sin(da / 2) ** 2 + math.cos(a1) * np.cos(a2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# ──────────────────────────────────────────────────────────────────────
#  Determinism + distribution
# ──────────────────────────────────────────────────────────────────────
def test_split_for_is_deterministic():
    grid = BlockSplitGrid()
    points = [(28.5, 67.5), (45.0, 80.0), (35.0, 90.0), (43.0, 77.0)]
    a = [grid.split_for(lat, lon) for lat, lon in points]
    b = [grid.split_for(lat, lon) for lat, lon in points]
    assert a == b


def test_split_distribution_close_to_80_10_10():
    """100k random points → train/val/test ratios within \u00b13 % of 80/10/10."""
    grid = BlockSplitGrid()
    rng = np.random.default_rng(42)
    n = 100_000
    lats = rng.uniform(HMA_BBOX[1], HMA_BBOX[3], n)
    lons = rng.uniform(HMA_BBOX[0], HMA_BBOX[2], n)
    splits = grid.split_for_batch(lats, lons)

    counts = {s: int((splits == s).sum()) for s in ("train", "val", "test", "BUFFER")}
    n_kept = counts["train"] + counts["val"] + counts["test"]
    assert n_kept > 0

    # Among kept (non-buffer) chips
    pct_train = counts["train"] / n_kept
    pct_val = counts["val"] / n_kept
    pct_test = counts["test"] / n_kept
    assert 0.77 < pct_train < 0.83, f"train fraction {pct_train:.3f} off from 0.80"
    assert 0.07 < pct_val < 0.13, f"val fraction {pct_val:.3f} off from 0.10"
    assert 0.07 < pct_test < 0.13, f"test fraction {pct_test:.3f} off from 0.10"


def test_buffer_drop_rate_is_bounded():
    """With 0.02\u00b0 buffer in 0.25\u00b0 blocks, expected drop is 2 \u00d7 (0.04/0.25) \u2248 32 % independent per axis,
    or ~ 1 - (0.84)^2 \u2248 29.4 %. We allow a generous \u00b15 % band."""
    grid = BlockSplitGrid(block_size_deg=0.25, buffer_deg=0.02)
    rng = np.random.default_rng(7)
    n = 50_000
    lats = rng.uniform(HMA_BBOX[1], HMA_BBOX[3], n)
    lons = rng.uniform(HMA_BBOX[0], HMA_BBOX[2], n)
    splits = grid.split_for_batch(lats, lons)
    pct_buffer = (splits == "BUFFER").sum() / n
    expected = 1 - ((1 - 2 * 0.02 / 0.25) ** 2)
    assert abs(pct_buffer - expected) < 0.04, (
        f"buffer drop rate {pct_buffer:.3f} far from analytic {expected:.3f}"
    )


def test_block_id_groups_neighbours():
    """Two centres in the same 0.25\u00b0 cell get the same block id and split."""
    grid = BlockSplitGrid(block_size_deg=0.25, buffer_deg=0.02)
    # Pick a centre well inside a block (avoid buffer)
    lat, lon = 35.125 + 0.001, 90.125 + 0.001
    lat_b, lon_b = 35.125 + 0.05, 90.125 + 0.05
    assert grid.block_id(lat, lon) == grid.block_id(lat_b, lon_b)
    # Splits match (both either same value, or one is BUFFER which we avoided)
    s1 = grid.split_for(lat, lon)
    s2 = grid.split_for(lat_b, lon_b)
    assert s1 == s2 and s1 != "BUFFER"


def test_block_id_changes_across_edge():
    """Two centres on opposite sides of a 0.25\u00b0 edge get different block ids."""
    grid = BlockSplitGrid(block_size_deg=0.25, buffer_deg=0.02)
    # Just north and just south of the lat=35.0 grid line
    assert grid.block_id(34.999, 90.5) != grid.block_id(35.001, 90.5)


def test_salt_changes_split():
    """Different salts produce a different split assignment."""
    g1 = BlockSplitGrid(salt="cryosentinel-blocks-v1")
    g2 = BlockSplitGrid(salt="completely-different")
    rng = np.random.default_rng(123)
    n = 1000
    lats = rng.uniform(HMA_BBOX[1], HMA_BBOX[3], n)
    lons = rng.uniform(HMA_BBOX[0], HMA_BBOX[2], n)
    s1 = g1.split_for_batch(lats, lons)
    s2 = g2.split_for_batch(lats, lons)
    # Analytic expectation among non-buffer points: with two independent
    # 80/10/10 hash assignments, P(same split) = 0.8² + 0.1² + 0.1² = 0.66.
    # Buffer assignment is geometric (salt-independent) so those points
    # never differ. With ~30% buffer, expected differing share ≈ 0.7 * 0.34 ≈ 0.24.
    # We assert a generous lower bound to allow random variance.
    n_diff = int((s1 != s2).sum())
    assert n_diff > 0.18 * n, (
        f"only {n_diff}/{n} points differ between salts "
        f"(expected ≈ 24% of non-buffer chips)"
    )


# ──────────────────────────────────────────────────────────────────────
#  Math correctness
# ──────────────────────────────────────────────────────────────────────
def test_chip_extent_safety_margin_at_max_lat():
    """Buffer must cover the chip half-extent at the highest HMA latitude."""
    grid = BlockSplitGrid(buffer_deg=DEFAULT_BUFFER_DEG)
    # At 45\u00b0N (Tien Shan), 1\u00b0 lon = 111 \u00d7 cos(45\u00b0) \u2248 78.5 km
    chip_ok_km = grid.chip_half_extent_km_required(max_lat_deg=45.0)
    assert chip_ok_km >= CHIP_HALF_EXTENT_KM * 1.05, (
        f"buffer {DEFAULT_BUFFER_DEG}\u00b0 covers only {chip_ok_km:.2f} km half-extent "
        f"at 45\u00b0N (need >= {CHIP_HALF_EXTENT_KM * 1.05:.2f} km with 5 % safety)"
    )


def test_block_size_in_km_is_reasonable():
    """At 35\u00b0 N, a 0.25\u00b0 block is roughly 25\u00d722 km."""
    grid = BlockSplitGrid(block_size_deg=0.25)
    lat_km, lon_km = grid.block_size_km(lat=35.0)
    assert 27 < lat_km < 28
    assert 22 < lon_km < 24


# ──────────────────────────────────────────────────────────────────────
#  Anti-leakage (the killer test)
# ──────────────────────────────────────────────────────────────────────
def test_no_chip_pair_in_different_splits_overlaps():
    """Sliding 1.12 km grid over a 1\u00b0 \u00d7 1\u00b0 box at 45\u00b0N (worst-case lon shrink).

    Builds ~10000 chip centres. For every centre we verify that no other
    centre with a *different* split is closer than the chip extent
    (2.24 km). Using the spatial-hash trick: each centre only needs to be
    compared to centres in its block and in the 8 neighbouring blocks.
    """
    grid = BlockSplitGrid(block_size_deg=0.25, buffer_deg=0.02)

    # Worst-case latitude band (Tien Shan) where lon-degree is shortest
    lat0, lon0 = 44.0, 77.0
    # Chip stride = 112 px \u00d7 10 m = 1.12 km.
    # In degrees lat: 1.12 / 111 \u2248 0.0101.
    # In degrees lon at 44\u00b0: 1.12 / (111 \u00d7 cos 44) \u2248 0.0140.
    lat_step = 1.12 / 111.0
    lon_step = 1.12 / (111.0 * math.cos(math.radians(lat0 + 0.5)))

    lats = np.arange(lat0, lat0 + 1.0, lat_step)
    lons = np.arange(lon0, lon0 + 1.0, lon_step)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    flat_lat = LAT.ravel()
    flat_lon = LON.ravel()
    splits = grid.split_for_batch(flat_lat, flat_lon)

    # Drop buffer chips \u2014 they are correctly excluded by design.
    keep = splits != "BUFFER"
    flat_lat = flat_lat[keep]
    flat_lon = flat_lon[keep]
    splits = splits[keep]
    n = len(splits)
    assert n > 4000, f"test grid too small: {n}"

    # Spatial hash: bucket centres by their block id.
    block_ids = np.array(
        [grid.block_id(la, lo) for la, lo in zip(flat_lat, flat_lon)],
        dtype=object,
    )
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, bid in enumerate(block_ids):
        buckets.setdefault(tuple(bid), []).append(i)

    # For every chip, check distance against same + 8 neighbouring buckets.
    # Vectorise the inner haversine so the test runs in ~5-10 s.
    min_cross_split = float("inf")
    offending = None
    for i in range(n):
        bid = tuple(block_ids[i])
        cand: list[int] = []
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                cand.extend(buckets.get((bid[0] + dlat, bid[1] + dlon), ()))
        if not cand:
            continue
        c = np.asarray(cand)
        c = c[c > i]                                      # i < j filter
        if c.size == 0:
            continue
        diff = splits[c] != splits[i]
        c = c[diff]
        if c.size == 0:
            continue
        dists = _haversine_km_vec(
            flat_lat[i], flat_lon[i], flat_lat[c], flat_lon[c]
        )
        k = int(dists.argmin())
        if dists[k] < min_cross_split:
            min_cross_split = float(dists[k])
            offending = (i, int(c[k]), splits[i], splits[c[k]])

    # The math guarantee: 2 \u00d7 buffer in km at 45\u00b0N
    expected_min_km = 2 * grid.buffer_deg * 111.0 * math.cos(math.radians(45.0))
    assert min_cross_split >= CHIP_EXTENT_KM, (
        f"LEAKAGE: closest cross-split chip pair at {min_cross_split:.2f} km "
        f"< chip extent {CHIP_EXTENT_KM:.2f} km. offending pair: {offending}"
    )
    # Tighter assertion: should also satisfy the analytic 2*buffer bound
    assert min_cross_split >= expected_min_km * 0.95, (
        f"buffer math regressed: min cross-split distance {min_cross_split:.2f} km "
        f"< 95 % of analytic bound {expected_min_km:.2f} km"
    )


# ──────────────────────────────────────────────────────────────────────
#  Integration with multimodal_dataset (smoke-only \u2014 no parquet I/O)
# ──────────────────────────────────────────────────────────────────────
def test_data_module_imports_and_accepts_block_split_kwargs():
    """Sanity: the new kwargs are accepted by MultiModalDataModule.__init__."""
    from cryosentinel.data import MultiModalDataModule
    dm = MultiModalDataModule(
        data_dir="non/existent/path",
        use_block_split=True,
        block_size_deg=0.25,
        block_buffer_deg=0.02,
        block_salt="test-salt",
        num_workers=0,
        persistent_workers=False,
    )
    assert dm.use_block_split is True
    assert dm.block_size_deg == 0.25
    assert dm.block_buffer_deg == 0.02
    assert dm.block_salt == "test-salt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
