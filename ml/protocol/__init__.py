"""Leakage-safe patient-level CIN2+ evaluation protocol.

This package is the scientific evaluation layer for the bachelor project.
It does **not** replace the frozen exploratory holdout under `ml/runs/`;
those results stay as a legacy benchmark. All protocol artifacts are written
to `ml/runs/protocol/`.
"""

from .constants import PROTOCOL_VERSION, RUNS_DIR

__all__ = ["PROTOCOL_VERSION", "RUNS_DIR"]
