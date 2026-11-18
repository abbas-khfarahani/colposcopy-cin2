"""Shared paths for Orginal Data cleanup / review tools."""

from __future__ import annotations

from pathlib import Path

# tools/original_data_cleanup/ → project root
KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[1]
DATA = ROOT / "Orginal Data"
LOGS = KIT / "logs"
CACHE = KIT / "cache"
REPORTS = LOGS / "reports"


def ensure_dirs() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
