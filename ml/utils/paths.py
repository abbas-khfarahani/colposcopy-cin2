"""Resolve project roots without baking machine-specific absolute paths."""

from __future__ import annotations

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent


def default_my_dataset_root() -> Path:
    """Prefer curated layout; fall back to legacy root symlink/folder."""
    curated = PROJECT_ROOT / "data" / "curated" / "My_Dataset"
    legacy = PROJECT_ROOT / "My_Dataset"
    if curated.is_dir():
        return curated
    return legacy


def default_splits_dir() -> Path:
    return default_my_dataset_root() / "splits"
