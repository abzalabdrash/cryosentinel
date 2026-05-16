"""Shared utilities (paths, config loading, logging)."""

from cryosentinel.utils.paths import (
    PROJECT_ROOT,
    DATA_ROOT,
    RAW_DIR,
    PROCESSED_DIR,
    EXTERNAL_DIR,
    CONFIGS_DIR,
)
from cryosentinel.utils.config import load_aoi_config, load_models_config

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RAW_DIR",
    "PROCESSED_DIR",
    "EXTERNAL_DIR",
    "CONFIGS_DIR",
    "load_aoi_config",
    "load_models_config",
]
