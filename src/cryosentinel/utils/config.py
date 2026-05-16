"""YAML config loading helpers.

We use Pydantic models so configs are typed and validated at import time
instead of failing in the middle of a 4-hour training run.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from cryosentinel.utils.paths import CONFIGS_DIR


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file. Raises FileNotFoundError with a helpful message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. "
            f"Did you run from the repository root, or did you delete it?"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_aoi_config(path: Path | None = None) -> dict[str, Any]:
    """Load configs/aoi.yaml — Areas of Interest definition."""
    return _load_yaml(path or (CONFIGS_DIR / "aoi.yaml"))


@lru_cache(maxsize=1)
def load_models_config(path: Path | None = None) -> dict[str, Any]:
    """Load configs/models.yaml — model architecture + training hyperparams."""
    return _load_yaml(path or (CONFIGS_DIR / "models.yaml"))


def get_primary_bbox() -> list[float]:
    """Return the primary AOI bounding box [w, s, e, n]."""
    return load_aoi_config()["primary"]["bbox"]


def get_test_site(site_id: str) -> dict[str, Any]:
    """Return one test site config by id, e.g. 'tuyuksu'."""
    sites = load_aoi_config()["test_sites"]
    for s in sites:
        if s["id"] == site_id:
            return s
    raise KeyError(f"Test site '{site_id}' not found in aoi.yaml")
