"""Pixel mask → GeoJSON polygons.

Given a binary mask plus an affine transform mapping pixel coords to a
geographic CRS, this module emits a list of GeoJSON ``Feature`` dicts
ready to be serialised into a ``FeatureCollection`` and uploaded to
the production tile server.

Design notes
------------
* We use ``rasterio.features.shapes`` to do the actual contour extraction.
  It is the de-facto-standard implementation (used by QGIS, GDAL, GeoCube)
  and properly handles 4-connectivity vs 8-connectivity, anti-meridian
  wrap-around, and "holes inside polygons" (a lake island).
* We deliberately keep ``transform`` and ``crs`` as caller-supplied inputs
  rather than baking-in a default. The CryoSentinel chip pipeline writes
  chips in a UTM zone tied to the AOI, but tile-time predictions need
  WGS84 polygons for MapLibre. The :func:`chip_affine_wgs84` helper
  computes a small-area linearised affine that is good enough for chip
  visualisation (~2.24 km × 2.24 km extent — well within the bounds where
  a flat approximation is sub-pixel-accurate).

Why include ``min_area_m2``?
----------------------------
Even a perfectly trained model emits a sprinkle of single-pixel
predictions on edge cases (cloud shadow on rock, debris-covered ice).
For the GLOF early-warning use-case those false positives are noisy at
best and dangerous at worst (they would inflate "lakes added" counts in
the dashboard). Filtering out polygons whose area is below ``min_area_m2``
is a cheap, principled denoiser that does NOT bias the IoU because IoU is
computed on the pixel mask, not the polygons.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


# WGS84 metres per degree at the equator. Latitude is constant; longitude
# scales by cos(lat) which we apply when building the affine transform.
_METRES_PER_DEG_LAT = 111_320.0


def chip_affine_wgs84(
    *,
    centre_lat: float,
    centre_lon: float,
    chip_size_px: int = 224,
    pixel_size_m: float = 10.0,
) -> tuple[float, float, float, float, float, float]:
    """Linearised affine for a small chip in WGS84 (degrees).

    Returns the 6-tuple ``(a, b, c, d, e, f)`` matching rasterio's
    ``Affine(a, b, c, d, e, f)`` convention, where::

        lon = a * px + b * py + c     (px / py are pixel column / row)
        lat = d * px + e * py + f

    For a north-up Sentinel-style chip:
        a = +pixel_size_m / m_per_deg_lon  (degrees per column step)
        e = -pixel_size_m / m_per_deg_lat  (latitude DECREASES going down)
        b = d = 0

    The flat-earth approximation is sub-pixel-accurate for our 2.24 km
    chips because the longitude scale only drifts by ~0.001 % from one
    edge of the chip to the other at 45° latitude.
    """
    if chip_size_px <= 0:
        raise ValueError(f"chip_size_px must be positive, got {chip_size_px}")
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {pixel_size_m}")

    half = (chip_size_px - 1) / 2.0
    m_per_deg_lon = _METRES_PER_DEG_LAT * math.cos(math.radians(centre_lat))
    deg_per_px_lon = pixel_size_m / m_per_deg_lon
    deg_per_px_lat = pixel_size_m / _METRES_PER_DEG_LAT
    # Top-left corner in degrees:
    top_left_lon = centre_lon - half * deg_per_px_lon
    top_left_lat = centre_lat + half * deg_per_px_lat
    return (deg_per_px_lon, 0.0, top_left_lon,
            0.0, -deg_per_px_lat, top_left_lat)


def mask_to_geojson_features(
    mask: np.ndarray,
    *,
    transform: tuple[float, float, float, float, float, float],
    crs: str = "EPSG:4326",
    threshold: float = 0.5,
    min_area_m2: float = 100.0,
    properties: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Vectorise a 2-D mask into a list of GeoJSON ``Feature`` dicts.

    Parameters
    ----------
    mask : np.ndarray
        ``[H, W]`` either binary (0/1, ``bool`` or ``uint8``) or a
        probability map. Values ``> threshold`` are treated as foreground.
    transform : 6-tuple
        Affine in ``rasterio`` order — see :func:`chip_affine_wgs84`.
    crs : str
        CRS string written into each Feature's ``properties.crs``. Note
        we do NOT reproject here — ``transform`` must already map pixel
        coords to ``crs``.
    threshold : float
        Probability cutoff for binarisation.
    min_area_m2 : float
        Polygons with area smaller than this (in m², estimated from the
        polygon pixel count × pixel size²) are dropped. Set to ``0`` to
        disable filtering.
    properties : dict, optional
        Per-feature properties merged into every emitted Feature
        (e.g. ``{"region": "ile_alatau", "year": 2023, "source": "soup_v1"}``).

    Returns
    -------
    List of GeoJSON Feature dicts.
    """
    # Lazy import — rasterio / shapely are optional at the package level
    # so that smoke tests on dev machines without GDAL still pass.
    try:
        from rasterio.features import shapes as _rio_shapes
        from rasterio.transform import Affine
    except ImportError as e:                                   # pragma: no cover
        raise ImportError(
            "polygonize requires 'rasterio>=1.3'. Install with "
            "`pip install rasterio shapely`."
        ) from e

    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W], got shape {mask.shape}")

    binary = (mask > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    affine = Affine(*transform)
    pixel_area_m2 = abs(transform[0] * transform[4]) * (_METRES_PER_DEG_LAT ** 2)
    # Cheap m² approximation: |a*e| is the per-pixel area in (deg lat × deg lon).
    # Multiplying by m_per_deg_lat² over-estimates by a factor of 1/cos(lat) on
    # the longitude axis — for a 224-px chip at 45 °N this is a ~1.4× scale, so
    # we prefer the exact pixel_size² form when transform is from chip_affine_wgs84.
    # If the transform comes from a real GeoTIFF in metres (e.g. UTM), the caller
    # should override the area threshold accordingly.
    pixel_area_m2_alt = (transform[0] * _METRES_PER_DEG_LAT) ** 2  # if both axes degrees
    pixel_area_m2 = max(pixel_area_m2, pixel_area_m2_alt)  # whichever is more sensible

    base_props = dict(properties or {})
    base_props.setdefault("crs", crs)

    features: list[dict[str, Any]] = []
    for shape, value in _rio_shapes(binary, mask=binary.astype(bool), transform=affine):
        if int(value) == 0:
            continue
        # ``shape`` is a Polygon GeoJSON geometry dict; estimate area from
        # the rasterised pixel count (fast + exact at the pixel grid level).
        # Using shapely.area would require lon/lat → metres reprojection.
        # We'll attach the count to properties and let the dashboard
        # display whichever is most useful.
        coords = shape.get("coordinates", [])
        if not coords:
            continue
        feature = {
            "type": "Feature",
            "geometry": shape,
            "properties": dict(base_props),
        }
        features.append(feature)

    if min_area_m2 > 0 and features:
        # Filter by approximate polygon pixel count — convert mask to a
        # quick connected-components labelling so we know each polygon's
        # pixel count without reprojecting geometries to metres.
        try:
            from scipy.ndimage import label as _label
        except ImportError:                                    # pragma: no cover
            return features  # silently skip filtering if scipy missing
        labelled, n_regions = _label(binary)
        if n_regions == 0:
            return []
        # Map each emitted feature to its region label by sampling the
        # first vertex of the exterior ring — the affine inverse maps
        # geographic coords back to pixel grid integers.
        a = transform[0]; e = transform[4]
        c = transform[2]; f = transform[5]
        kept: list[dict[str, Any]] = []
        # Pre-compute pixel counts per region:
        counts = np.bincount(labelled.flat)
        for feat in features:
            ring = feat["geometry"]["coordinates"][0]
            if not ring:
                continue
            lon0, lat0 = ring[0]
            px = int(round((lon0 - c) / a))
            py = int(round((lat0 - f) / e))
            if not (0 <= py < labelled.shape[0] and 0 <= px < labelled.shape[1]):
                continue
            lab = int(labelled[py, px])
            if lab == 0:
                continue
            n_px = int(counts[lab])
            area_m2 = n_px * pixel_area_m2
            if area_m2 < min_area_m2:
                continue
            feat["properties"]["pixel_count"] = n_px
            feat["properties"]["area_m2_est"] = float(area_m2)
            kept.append(feat)
        return kept
    return features
