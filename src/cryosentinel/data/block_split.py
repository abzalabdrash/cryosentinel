"""Spatial block split for leakage-free train/val/test partitioning.

Why we need this
----------------
The original CryoSentinel pipeline used a *lake-level* split — every chip was
mapped to its closest Kumar lake centroid and inherited that lake's split. That
guarantees that a single lake's pixels do not appear in two splits, **but it
fails to prevent chip-overlap leakage**:

* Chips are extracted at stride 112 over a 224×224 window → adjacent chips
  share **50 % of pixels**.
* Two neighbouring chip centres can fall closer to two *different* lakes, so
  the lake-based split assigns them to different splits even though their
  spatial extents overlap.
* The model sees the same 112×112 patch as "train" via chip A and as "val"
  via chip B → inflated val-IoU.

A reviewer of an arXiv-grade paper will spot this immediately. The
canonical fix used by every state-of-the-art remote-sensing benchmark
(GeoBench, EarthNets, AI4Boundaries, …) is a **spatial block split**: tile
the AOI into ~25 × 25 km blocks, assign each block whole-cloth to a split
via a deterministic hash, and drop chips whose extent crosses a block
boundary.

Design
------
* **Grid** — equirectangular lat/lon grid with cell size ``block_size_deg``
  (default 0.25° ≈ 25 km lat, 19-25 km lon at HMA latitudes 28-45°N).
* **Hashing** — ``sha1(salt | block_id) → bucket ∈ [0, 100)`` →
  ``[0, 80) train | [80, 90) val | [90, 100) test``.
* **Cross-year consistency** — the hash is independent of ``snapshot_year``
  so the same block is in the same split for 2016 *and* 2022.
* **Buffer drop** — a chip whose centre lies within ``buffer_deg`` of any
  block boundary is dropped (split = ``"BUFFER"``). With our chip extent of
  224 px × 10 m = 2.24 km (half-extent 1.12 km), a buffer of 0.02° ≈ 1.4 km
  lat / 1.6 km lon at 45°N guarantees that any *kept* chip's spatial extent
  is fully inside one block — i.e. cannot leak into a neighbouring split.

Validation
----------
The companion test ``tests/test_block_split.py`` checks for any pair of
chips in *different* splits whose centre distance is below the chip extent
(2.24 km). The buffer rule above mathematically guarantees this property,
but the empirical check guards against silent regressions.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

# ──────────────────────────────────────────────────────────────────────
#  Defaults — calibrated for HMA (lat 26-45°N) and 224 × 10m chips
# ──────────────────────────────────────────────────────────────────────
DEFAULT_BLOCK_SIZE_DEG: float = 0.25      # ~25 km lat, ~19-25 km lon at HMA
DEFAULT_BUFFER_DEG: float = 0.02          # ~2.2 km lat, ~1.4-2.0 km lon
DEFAULT_SALT: str = "cryosentinel-blocks-v1"

# Canonical bucket boundaries (80 / 10 / 10) — keep aligned with
# the original lake-level diagnostic split for comparability.
TRAIN_HI: int = 80
VAL_HI: int = 90


@dataclass(frozen=True)
class BlockSplitGrid:
    """Deterministic spatial block split based on lat/lon hashing.

    Args:
        block_size_deg: side length of each block in degrees (lat=lon).
        buffer_deg: chips whose centre is within this many degrees of any
            block boundary are dropped (assigned ``"BUFFER"``). The buffer
            must be ≥ ``chip_half_extent_km / 111 km/deg`` at the highest
            latitude in the dataset. For 224 × 10 m chips up to 45°N,
            ``0.02`` provides a comfortable safety margin.
        salt: tweak this to obtain a *different* deterministic split.
    """

    block_size_deg: float = DEFAULT_BLOCK_SIZE_DEG
    buffer_deg: float = DEFAULT_BUFFER_DEG
    salt: str = DEFAULT_SALT

    # ── Block id ───────────────────────────────────────────────────────
    def block_id(self, lat: float, lon: float) -> tuple[int, int]:
        """Map a chip centre to its (lat_idx, lon_idx) block id."""
        lat_idx = int(math.floor(lat / self.block_size_deg))
        lon_idx = int(math.floor(lon / self.block_size_deg))
        return (lat_idx, lon_idx)

    # ── Buffer detection ───────────────────────────────────────────────
    def is_in_buffer(self, lat: float, lon: float) -> bool:
        """True iff the chip centre is within ``buffer_deg`` of a block edge.

        Such chips are dropped to guarantee that no kept chip's spatial
        extent (±1.12 km around its centre) crosses into a neighbouring
        block — and therefore cannot leak between splits.
        """
        # Distance (in degrees) from the centre to the nearest block edge
        # along each axis. A point at the exact edge has 0 distance.
        lat_in_block = (lat / self.block_size_deg) % 1.0
        lon_in_block = (lon / self.block_size_deg) % 1.0
        lat_edge = min(lat_in_block, 1.0 - lat_in_block) * self.block_size_deg
        lon_edge = min(lon_in_block, 1.0 - lon_in_block) * self.block_size_deg
        return lat_edge < self.buffer_deg or lon_edge < self.buffer_deg

    # ── Bucket assignment ──────────────────────────────────────────────
    def _bucket(self, block_id: tuple[int, int]) -> int:
        key = f"{self.salt}|{block_id[0]}|{block_id[1]}".encode("utf-8")
        h = hashlib.sha1(key).hexdigest()
        return int(h[:8], 16) % 100

    def split_for(self, lat: float, lon: float) -> str:
        """Return ``"train" | "val" | "test" | "BUFFER"`` for this chip centre.

        ``"BUFFER"`` chips are within the boundary buffer and should be
        dropped from training (they introduce leakage risk).
        """
        if self.is_in_buffer(lat, lon):
            return "BUFFER"
        bucket = self._bucket(self.block_id(lat, lon))
        if bucket < TRAIN_HI:
            return "train"
        if bucket < VAL_HI:
            return "val"
        return "test"

    # ── Vectorised helpers (for batch index construction) ──────────────
    def split_for_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        """Vectorised ``split_for`` returning a string ndarray.

        Faster than per-row Python iteration when building a global chip
        index. Uses sha1 in a Python loop because hashlib is C-fast and
        the per-call overhead is small relative to Parquet IO.
        """
        if lats.shape != lons.shape:
            raise ValueError(f"lats/lons shape mismatch: {lats.shape} vs {lons.shape}")
        out = np.empty(lats.shape, dtype=object)
        size = self.block_size_deg
        buf = self.buffer_deg
        salt = self.salt
        for i in range(lats.size):
            lat = float(lats.flat[i])
            lon = float(lons.flat[i])
            lat_in = (lat / size) % 1.0
            lon_in = (lon / size) % 1.0
            lat_edge = min(lat_in, 1.0 - lat_in) * size
            lon_edge = min(lon_in, 1.0 - lon_in) * size
            if lat_edge < buf or lon_edge < buf:
                out.flat[i] = "BUFFER"
                continue
            lat_idx = int(math.floor(lat / size))
            lon_idx = int(math.floor(lon / size))
            key = f"{salt}|{lat_idx}|{lon_idx}".encode("utf-8")
            bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) % 100
            if bucket < TRAIN_HI:
                out.flat[i] = "train"
            elif bucket < VAL_HI:
                out.flat[i] = "val"
            else:
                out.flat[i] = "test"
        return out

    # ── Diagnostics ────────────────────────────────────────────────────
    def block_size_km(self, lat: float = 35.0) -> tuple[float, float]:
        """Return ``(lat_km, lon_km)`` for the block size at a given latitude."""
        lat_km = self.block_size_deg * 111.0
        lon_km = self.block_size_deg * 111.0 * max(0.05, math.cos(math.radians(lat)))
        return lat_km, lon_km

    def chip_half_extent_km_required(
        self, *, max_lat_deg: float = 45.0
    ) -> float:
        """Maximum chip half-extent (in km) safely covered by ``buffer_deg``.

        At ``max_lat_deg`` the lon-degree is shortest (cos(lat) factor), so
        this is the binding constraint. Returns the chip half-extent in km
        that fits inside ``buffer_deg`` longitudinally — the chip will not
        cross block boundaries as long as its actual half-extent ≤ this.
        """
        return self.buffer_deg * 111.0 * math.cos(math.radians(max_lat_deg))


__all__ = [
    "BlockSplitGrid",
    "DEFAULT_BLOCK_SIZE_DEG",
    "DEFAULT_BUFFER_DEG",
    "DEFAULT_SALT",
]
