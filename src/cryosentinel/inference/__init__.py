"""CryoSentinel inference utilities.

This package houses everything that turns a trained model + raster input
into deployable artefacts (binary masks, GeoJSON polygons, side-by-side
visualisations). It is deliberately decoupled from training so it can be
imported in a smaller runtime without pulling Lightning.

Modules
-------
* :mod:`.sliding_window` — Gaussian-blended sliding window inference for
  full-scene rasters that exceed the model's native window size (224×224
  for TerraMind v1). Pure PyTorch — no GDAL/rasterio dependency.

* :mod:`.polygonize` — convert a binary pixel mask plus an affine geo
  transform into a list of GeoJSON Feature dicts. Uses :mod:`rasterio`
  for the actual mask→polygon contouring; the rest is pure Python so it
  is trivial to test offline.
"""
from __future__ import annotations

from .polygonize import (
    chip_affine_wgs84,
    mask_to_geojson_features,
)
from .sliding_window import (
    gaussian_window_2d,
    sliding_window_inference,
)

__all__ = [
    # sliding_window
    "gaussian_window_2d",
    "sliding_window_inference",
    # polygonize
    "chip_affine_wgs84",
    "mask_to_geojson_features",
]
