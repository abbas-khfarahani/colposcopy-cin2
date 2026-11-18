"""JSON/CSV helpers, environment lock, and atomic writes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        f.write("\n")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def capture_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "captured_utc": utc_now(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_runtime"] = torch.version.cuda
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = str(exc)
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = None
    try:
        import sklearn

        info["sklearn"] = sklearn.__version__
    except Exception:
        info["sklearn"] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_commit"] = commit
        info["git_dirty"] = bool(dirty)
    except Exception:
        info["git_commit"] = None
        info["git_dirty"] = None
    return info


def write_environment_lock(path: Path) -> dict[str, Any]:
    env = capture_environment()
    save_json(path.with_suffix(".json"), env)
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        freeze = "# pip freeze failed\n"
    write_text(path, freeze)
    return env
