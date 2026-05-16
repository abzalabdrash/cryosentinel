"""Filesystem paths used across the project.

Single source of truth so notebooks/scripts never hardcode strings.
Override `DATA_ROOT` via the `CRYOSENTINEL_DATA_ROOT` env var if your
data lives outside the repo (e.g. on an external SSD).
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root = parent of `src/`
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# Data root — overridable via env var for big-disk setups
DATA_ROOT: Path = Path(
    os.environ.get("CRYOSENTINEL_DATA_ROOT", PROJECT_ROOT / "data")
).resolve()

RAW_DIR: Path = DATA_ROOT / "raw"
PROCESSED_DIR: Path = DATA_ROOT / "processed"
EXTERNAL_DIR: Path = DATA_ROOT / "external"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"


def ensure_data_dirs() -> None:
    """Create data sub-folders if they don't exist (idempotent)."""
    for p in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR):
        p.mkdir(parents=True, exist_ok=True)
