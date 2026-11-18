"""Attention / error galleries for spatial-attention checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from ml.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from ml.protocol.datasets import open_rgb
from ml.protocol.encoders import ProtocolImageModel, build_protocol_model
from ml.protocol.io_utils import write_text
from ml.protocol.train_common import load_checkpoint, resolve_device
from ml.protocol.transforms import build_protocol_transforms


def _denorm(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


def _upsample_attn(attn: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    a = F.interpolate(attn, size=size, mode="bilinear", align_corners=False)
    a = a[0, 0].detach().cpu().numpy()
    a = a - a.min()
    if a.max() > 1e-8:
        a = a / a.max()
    return a


def generate_attention_gallery(
    *,
    fold_ckpts: list[dict[str, Any]],
    oof: pd.DataFrame,
    dataset_root: Path,
    out_dir: Path,
    n_per_cell: int = 6,
    device_name: str = "cuda",
) -> None:
    """fold_ckpts: {fold, path, backbone, model_type, center_prior}."""
    import matplotlib.pyplot as plt

    device = resolve_device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = build_protocol_transforms("eval", 224)

    # Prefer image-level OOF rows; else skip.
    if "file" not in oof.columns:
        write_text(out_dir / "README.txt", "Gallery skipped: no image-level OOF rows.\n")
        return

    oof = oof.copy()
    if "pred70" not in oof.columns:
        oof["pred70"] = (oof["calibrated_probability"] >= oof.get("threshold_op70", 0.5)).astype(int)
    cells = {
        "TP": (oof["person_label"] == 1) & (oof["pred70"] == 1),
        "FN": (oof["person_label"] == 1) & (oof["pred70"] == 0),
        "FP": (oof["person_label"] == 0) & (oof["pred70"] == 1),
        "TN": (oof["person_label"] == 0) & (oof["pred70"] == 0),
    }
    ckpt_by_fold = {int(c["fold"]): c for c in fold_ckpts}
    html = [
        "<!doctype html><meta charset='utf-8'><title>Attention gallery</title>",
        "<style>body{font-family:sans-serif;margin:24px} img{max-width:100%;border:1px solid #ddd}",
        "section{margin-bottom:32px} h2{margin-bottom:8px}</style>",
        "<h1>Attention error gallery (OOF, OP70)</h1>",
        "<p>Attention maps are for auditing, not causal proof of CIN morphology.</p>",
    ]

    models: dict[int, ProtocolImageModel] = {}
    for cell, mask in cells.items():
        subset = oof.loc[mask]
        if subset.empty:
            html.append(f"<h2>{cell}</h2><p>None.</p>")
            continue
        # High-confidence first.
        if cell.endswith("P"):
            subset = subset.sort_values("calibrated_probability", ascending=False)
        else:
            subset = subset.sort_values("calibrated_probability", ascending=True)
        # One image per person.
        subset = subset.drop_duplicates("person_id").head(n_per_cell)
        html.append(f"<h2>{cell} (n shown={len(subset)})</h2><section>")
        cell_dir = out_dir / cell
        cell_dir.mkdir(exist_ok=True)
        for i, row in subset.iterrows():
            fold = int(row["outer_fold"])
            spec = ckpt_by_fold.get(fold)
            if spec is None:
                continue
            if fold not in models:
                model = build_protocol_model(
                    backbone=spec["backbone"],
                    model_type=spec["model_type"],
                    pretrained=False,
                    center_prior=bool(spec.get("center_prior", True)),
                )
                ckpt = load_checkpoint(Path(spec["path"]), device)
                model.load_state_dict(ckpt["model_state"])
                model.to(device).eval()
                models[fold] = model
            model = models[fold]
            path = dataset_root / str(row["image_path_rel"])
            if not path.is_file():
                continue
            pil = open_rgb(path)
            tensor = tf(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                aux = model(tensor, return_aux=True)
            rgb = _denorm(tensor[0])
            if aux.attn is None:
                continue
            attn = _upsample_attn(aux.attn, rgb.shape[:2])
            overlay = _overlay(rgb, attn)
            fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
            axes[0].imshow(rgb)
            axes[0].set_title("Original")
            axes[1].imshow(attn, cmap="magma")
            axes[1].set_title("Attention")
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            for ax in axes:
                ax.axis("off")
            title = (
                f"{row['person_id']}  p={float(row['calibrated_probability']):.3f}  "
                f"y={int(row['person_label'])}  fold={fold}"
            )
            fig.suptitle(title, fontsize=10)
            fig.tight_layout()
            out_png = cell_dir / f"{row['person_id']}.png"
            fig.savefig(out_png)
            plt.close(fig)
            html.append(
                f"<div><p><b>{cell}</b> {row['person_id']} "
                f"p={float(row['calibrated_probability']):.3f}</p>"
                f"<img src='{cell}/{out_png.name}'></div>"
            )
        html.append("</section>")
    write_text(out_dir / "index.html", "\n".join(html))
